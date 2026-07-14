from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import PlanetParameters
from .filters import normalize_filter, passband_label
from .models import JobStatus, LEAPSError, StageEvent, StageID
from .project import ProjectWorkspace

Emitter = Callable[[StageEvent], None]


def _read_fits_image(path: Path) -> tuple[np.ndarray, Any]:
    """Read a scaled FITS image without modifying or memory-mapping scaled pixels.

    Astropy cannot expose FITS images containing BZERO, BSCALE, or BLANK through
    its usual scaled memmap path. Reading the stored values and applying the
    standard FITS scaling ourselves keeps raw files read-only and avoids loading
    an additional, implicitly scaled copy.
    """
    from astropy.io import fits
    from astropy.utils.exceptions import AstropyUserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Header block contains null bytes instead of spaces for padding.*",
            category=AstropyUserWarning,
        )
        with fits.open(
            path,
            memmap=True,
            do_not_scale_image_data=True,
            ignore_missing_end=True,
        ) as hdus:
            hdu = next(candidate for candidate in hdus if getattr(candidate, "data", None) is not None)
            stored = np.asarray(hdu.data)
            data = stored.astype(np.float32, copy=True)
            header = hdu.header.copy()

    blank = header.get("BLANK")
    if blank is not None:
        data[data == float(blank)] = np.nan
    scale = float(header.get("BSCALE", 1.0))
    zero = float(header.get("BZERO", 0.0))
    if scale != 1.0:
        data *= scale
    if zero != 0.0:
        data += zero
    return data, header


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise LEAPSError(
                "JOB_CANCELLED",
                "Processing was safely cancelled",
                "Verified checkpoints were kept. Resume or restart this stage when ready.",
                ["Resume", "Restart stage"],
            )


@dataclass(slots=True)
class ReductionConfig:
    exposure_key: str = "EXPTIME"
    date_key: str = "DATE-OBS"
    time_key: str = "TIME-OBS"
    filter_name: str = "R"
    combine_method: str = "median"
    binning: int = 1
    crop: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class PhotometryConfig:
    aperture_radius: float = 8.0
    sky_inner_aperture: float = 1.7
    sky_outer_aperture: float = 2.4
    saturation_fraction: float = 0.95
    camera_gain: float = 1.0
    variable_aperture: bool = True
    geometric_center: bool = False
    centroids_snr: float = 4.0
    stars_snr: float = 4.0


@dataclass(slots=True)
class InspectionResult:
    frames: list[dict[str, Any]]
    median_sky: float
    median_psf: float


@dataclass(slots=True)
class PlateSolveAttempt:
    index: int
    pixel_scale: float
    status: str
    detail: str


@dataclass(slots=True)
class PlateSolveResult:
    solved: bool
    attempts: list[PlateSolveAttempt]
    target_xy: tuple[float, float] | None = None
    identified_stars: int = 0
    wcs_header: dict[str, Any] = field(default_factory=dict)
    unverified: bool = False


def _emit(
    emit: Emitter | None,
    stage: StageID,
    status: JobStatus,
    message: str,
    current: int = 0,
    total: int = 0,
    checkpoint: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if emit:
        emit(
            StageEvent(
                stage,
                status,
                message,
                current,
                total,
                checkpoint,
                details=details or {},
            )
        )


class ReductionService:
    def run(
        self,
        project: ProjectWorkspace,
        config: ReductionConfig,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Path:
        from astropy.io import fits

        token = token or CancellationToken()
        files = project.manifest.raw_files
        science = [project.resolve(path) for path in files.get("science", [])]
        if not science:
            raise LEAPSError(
                "NO_SCIENCE_FRAMES",
                "No science frames are assigned",
                "Return to Data & Target and confirm the FITS frame assignments.",
                ["Review frame assignments"],
                stage=StageID.REDUCTION,
            )
        pending, target = project.begin_transaction(StageID.REDUCTION)
        _emit(emit, StageID.REDUCTION, JobStatus.RUNNING, "Building calibration frames", 0, len(science))
        master_bias, bias_exposure = self._master_bias(project, files.get("bias", []), config)
        master_dark = self._master_dark(project, files.get("dark", []), config, master_bias, bias_exposure)
        master_dark_flat = self._master_dark(
            project, files.get("dark_flat", []), config, master_bias, bias_exposure, fallback=master_dark
        )
        master_flat = self._master_flat(
            project, files.get("flat", []), config, master_bias, master_dark_flat, bias_exposure
        )
        metadata: list[dict[str, Any]] = []
        for index, path in enumerate(science, start=1):
            token.raise_if_cancelled()
            try:
                data, header = _read_fits_image(path)
                exposure = float(header.get(config.exposure_key, 0.0))
                reduced = (
                    data - master_bias - max(0.0, exposure - bias_exposure) * master_dark
                ) / master_flat
                reduced[~np.isfinite(reduced)] = 0
                if config.crop:
                    x1, x2, y1, y2 = config.crop
                    reduced = reduced[y1 : y2 or None, x1 : x2 or None]
                if config.binning > 1:
                    from hops.hops_tools.image_analysis import bin_frame

                    reduced = bin_frame(reduced, config.binning)
                mean, std, psf = self._statistics(reduced, header)
                output_name = f"r_{index:05d}_{path.name}"
                output = pending / output_name
                output_header = header.copy()
                output_header["LEAPSVER"] = "0.1.0"
                output_header["HOPSJD"] = _julian_date(output_header, config)
                output_header["HOPSMEAN"] = mean
                output_header["HOPSSTD"] = std
                output_header["HOPSPSF"] = psf
                output_header["HOPSSKIP"] = bool(not np.isfinite(psf))
                output_header["HOPSFLT"] = config.filter_name
                fits.PrimaryHDU(reduced.astype(np.float32), header=output_header).writeto(
                    output, overwrite=True
                )
                metadata.append(
                    {
                        "file": output_name,
                        "source": project.relative(path),
                        "mean": mean,
                        "std": std,
                        "psf": psf,
                        "exposure": exposure,
                        "skip": bool(not np.isfinite(psf)),
                    }
                )
                checkpoint = project.checkpoints_dir / "reduction.json"
                checkpoint.write_text(
                    json.dumps({"completed": index, "files": metadata}, indent=2), encoding="utf-8"
                )
                _emit(
                    emit,
                    StageID.REDUCTION,
                    JobStatus.RUNNING,
                    f"Reduced {path.name}",
                    index,
                    len(science),
                    project.relative(checkpoint),
                )
            except LEAPSError:
                raise
            except Exception as exc:
                raise LEAPSError(
                    "REDUCTION_FRAME_FAILED",
                    f"{path.name} could not be reduced",
                    "The last successful reduction remains available and the source FITS file was not modified.",
                    ["Inspect the FITS header", "Exclude this frame", "Export diagnostics"],
                    stage=StageID.REDUCTION,
                    technical_details=str(exc),
                ) from exc
        (pending / "frames.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.REDUCTION, JobStatus.SUCCEEDED, "Reduction complete", len(science), len(science))
        return target

    @staticmethod
    def _load(project: ProjectWorkspace, paths: Iterable[str]) -> list[tuple[np.ndarray, Any]]:
        result: list[tuple[np.ndarray, Any]] = []
        for relative in paths:
            path = project.resolve(relative)
            try:
                result.append(_read_fits_image(path))
            except Exception as exc:
                raise LEAPSError(
                    "CALIBRATION_FRAME_UNREADABLE",
                    f"{path.name} could not be read",
                    "The calibration frames could not be combined. The raw FITS file was not modified.",
                    ["Review the frame assignment", "Open the FITS file", "Export diagnostics"],
                    stage=StageID.REDUCTION,
                    technical_details=f"{type(exc).__name__}: {exc}",
                ) from exc
        return result

    def _master_bias(
        self, project: ProjectWorkspace, paths: list[str], config: ReductionConfig
    ) -> tuple[np.ndarray | float, float]:
        frames = self._load(project, paths)
        if not frames:
            return 0.0, 0.0
        exposures = np.array([float(header.get(config.exposure_key, 0.0)) for _, header in frames])
        median_exposure = float(np.median(exposures))
        arrays = [array for (array, _), use in zip(frames, np.isclose(exposures, median_exposure)) if use]
        return _combine(arrays, config.combine_method), median_exposure

    def _master_dark(
        self,
        project: ProjectWorkspace,
        paths: list[str],
        config: ReductionConfig,
        master_bias: np.ndarray | float,
        bias_exposure: float,
        fallback: np.ndarray | float = 0.0,
    ) -> np.ndarray | float:
        frames = self._load(project, paths)
        if not frames:
            return fallback
        corrected = [
            (array - master_bias) / max(float(header.get(config.exposure_key, 0.0)) - bias_exposure, 1e-9)
            for array, header in frames
        ]
        return _combine(corrected, config.combine_method)

    def _master_flat(
        self,
        project: ProjectWorkspace,
        paths: list[str],
        config: ReductionConfig,
        master_bias: np.ndarray | float,
        master_dark_flat: np.ndarray | float,
        bias_exposure: float,
    ) -> np.ndarray | float:
        frames = self._load(project, paths)
        if not frames:
            return 1.0
        corrected = []
        for array, header in frames:
            exposure = max(float(header.get(config.exposure_key, 0.0)) - bias_exposure, 0.0)
            flat = array - master_bias - exposure * master_dark_flat
            median = float(np.nanmedian(flat))
            if not math.isfinite(median) or median == 0:
                continue
            corrected.append(flat / median)
        if not corrected:
            return 1.0
        master = _combine(corrected, config.combine_method)
        master = np.where(np.isfinite(master) & (master > 0), master, 1.0)
        return master / np.nanmedian(master)

    @staticmethod
    def _statistics(data: np.ndarray, header: Any) -> tuple[float, float, float]:
        from hops.hops_tools.image_analysis import image_mean_std, image_psf

        mean, std = image_mean_std(data)
        saturation = float(header.get("SATURATE", np.nanmax(data)))
        psf = image_psf(data, header, mean, std, saturation)
        return float(mean), float(std), float(psf)


class InspectionService:
    def run(
        self,
        project: ProjectWorkspace,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> InspectionResult:
        token = token or CancellationToken()
        reduction = project.outputs_dir / StageID.REDUCTION.value
        frames = sorted(reduction.glob("*.fit*"))
        if not frames:
            raise LEAPSError(
                "NO_REDUCED_FRAMES",
                "No reduced frames are available",
                "Run Reduction before Inspection.",
                ["Open Reduction"],
                stage=StageID.INSPECTION,
            )
        from astropy.io import fits

        values: list[dict[str, Any]] = []
        for index, path in enumerate(frames, start=1):
            token.raise_if_cancelled()
            header = fits.getheader(path)
            values.append(
                {
                    "file": path.name,
                    "sky": float(header.get("HOPSMEAN", 0.0)),
                    "sky_std": float(header.get("HOPSSTD", 0.0)),
                    "psf": float(header.get("HOPSPSF", float("nan"))),
                    "excluded": bool(header.get("HOPSSKIP", False)),
                }
            )
            _emit(emit, StageID.INSPECTION, JobStatus.RUNNING, f"Checked {path.name}", index, len(frames))
        skies = np.array([record["sky"] for record in values], dtype=float)
        psfs = np.array([record["psf"] for record in values], dtype=float)
        sky_median, psf_median = float(np.nanmedian(skies)), float(np.nanmedian(psfs))
        sky_mad = max(float(np.nanmedian(np.abs(skies - sky_median))), 1e-9)
        psf_mad = max(float(np.nanmedian(np.abs(psfs - psf_median))), 1e-9)
        for record in values:
            if abs(record["sky"] - sky_median) > 5 * sky_mad or abs(record["psf"] - psf_median) > 5 * psf_mad:
                record["suggest_exclude"] = True
            else:
                record["suggest_exclude"] = False
        pending, target = project.begin_transaction(StageID.INSPECTION)
        result = InspectionResult(values, sky_median, psf_median)
        (pending / "inspection.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.INSPECTION, JobStatus.SUCCEEDED, "Inspection complete", len(frames), len(frames))
        return result


class AlignmentService:
    def run(
        self,
        project: ProjectWorkspace,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Path:
        token = token or CancellationToken()
        from astropy.io import fits

        from hops.hops_tools.image_analysis import image_find_stars
        from hops.thirdparty import twirl

        frames = sorted((project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if len(frames) < 2:
            raise LEAPSError(
                "ALIGNMENT_INPUT_MISSING",
                "Alignment needs at least two reduced frames",
                "Run Reduction first.",
                ["Open Reduction"],
                stage=StageID.ALIGNMENT,
            )
        reference_data, reference_header = fits.getdata(frames[0], header=True)
        detected_reference = image_find_stars(reference_data, reference_header, star_limit=60) or []
        reference_stars = np.asarray(detected_reference, dtype=float)
        if reference_stars.size:
            reference_stars = reference_stars[:, :2]
        if len(reference_stars) < 5:
            raise LEAPSError(
                "TOO_FEW_ALIGNMENT_STARS",
                "Too few stars were found for alignment",
                "Try a lower star-detection threshold or inspect the first frame.",
                ["Review the first frame", "Adjust advanced alignment settings"],
                stage=StageID.ALIGNMENT,
            )
        records = []
        for index, path in enumerate(frames, start=1):
            token.raise_if_cancelled()
            try:
                data, header = fits.getdata(path, header=True)
                detected = image_find_stars(data, header, star_limit=60) or []
                stars = np.asarray(detected, dtype=float)
                if not stars.size:
                    raise ValueError("No alignment stars were detected")
                stars = stars[:, :2]
                count = min(20, len(reference_stars), len(stars))
                transform = twirl.utils.find_transform(
                    reference_stars[:count], stars[:count], n=count, tolerance=12
                )
                matrix = np.asarray(transform)
                rotation = float(math.atan2(matrix[1, 0], matrix[0, 0]))
                x0, y0 = float(matrix[0, 2]), float(matrix[1, 2])
                header["HOPSX0"] = x0
                header["HOPSY0"] = y0
                header["HOPSU0"] = rotation
                fits.writeto(path, data, header, overwrite=True)
                records.append(
                    {
                        "file": path.name,
                        "x0": x0,
                        "y0": y0,
                        "rotation": rotation,
                        "matched": count,
                        "matrix": matrix.tolist(),
                    }
                )
            except Exception as exc:
                records.append({"file": path.name, "failed": True, "reason": str(exc)})
            _emit(emit, StageID.ALIGNMENT, JobStatus.RUNNING, f"Aligned {path.name}", index, len(frames))
        pending, target = project.begin_transaction(StageID.ALIGNMENT)
        (pending / "alignment.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.ALIGNMENT, JobStatus.SUCCEEDED, "Alignment complete", len(frames), len(frames))
        return target


class PlateSolveService:
    def solve(
        self,
        frame: str | Path,
        ra: str,
        dec: str,
        pixel_scale: float,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> PlateSolveResult:
        token = token or CancellationToken()
        import astropy.units as units
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        from hops.hops_tools.image_analysis import image_find_stars, image_plate_solve

        coordinate = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
        _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, "Coordinates validated", 0, 3)
        data, header = _read_fits_image(Path(frame))
        mean = float(header.get("HOPSMEAN", np.nanmedian(data)))
        std = float(header.get("HOPSSTD", 1.4826 * np.nanmedian(np.abs(data - mean))))
        psf = max(float(header.get("HOPSPSF", 2.0)), 1.0)
        burn_limit = float(header.get("HOPSSAT", header.get("SATURATE", np.nanmax(data))))
        stars = image_find_stars(
            data,
            header,
            mean=mean,
            std=std,
            psf=psf,
            burn_limit=burn_limit,
            star_limit=100,
        ) or []
        if len(stars) < 5:
            raise LEAPSError(
                "TOO_FEW_PLATE_STARS",
                "Too few stars were detected",
                "Plate solving needs at least five usable stars.",
                ["Adjust contrast and detection threshold", "Choose another frame"],
                stage=StageID.PHOTOMETRY,
            )
        existing_wcs = None
        try:
            from astropy.wcs import WCS
            from astropy.wcs.utils import proj_plane_pixel_scales
            from astropy.wcs.wcs import FITSFixedWarning

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                existing_wcs = WCS(header)
            if existing_wcs.has_celestial:
                x, y = coordinate.to_pixel(existing_wcs)
                x, y = float(np.asarray(x)), float(np.asarray(y))
                nearest_star = min(
                    (
                        math.hypot(float(star[0]) - x, float(star[1]) - y)
                        for star in stars
                    ),
                    default=float("inf"),
                )
                if (
                    0 <= x < data.shape[1]
                    and 0 <= y < data.shape[0]
                    and nearest_star <= max(5.0 * psf, 8.0)
                ):
                    scales = proj_plane_pixel_scales(existing_wcs.celestial) * 3600.0
                    detected_scale = float(np.nanmedian(scales))
                    attempt = PlateSolveAttempt(
                        0,
                        detected_scale,
                        "complete",
                        "Existing FITS WCS validated and contains the target",
                    )
                    _emit(
                        emit,
                        StageID.PHOTOMETRY,
                        JobStatus.SUCCEEDED,
                        "Existing FITS WCS validated",
                        1,
                        1,
                    )
                    return PlateSolveResult(
                        True,
                        [attempt],
                        (x, y),
                        len(stars),
                        dict(existing_wcs.to_header()),
                    )
        except Exception:
            pass
        _emit(
            emit,
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            f"{len(stars)} stars detected",
            0,
            3,
        )
        cache_key = hashlib.sha256(f"{coordinate.ra.deg:.6f},{coordinate.dec.deg:.6f}".encode()).hexdigest()[:16]
        gaia_cache = Path(frame).resolve().parents[2] / "cache" / f"gaia-{cache_key}.ecsv"
        gaia_query = None
        catalog_limit = max(100, 10 * len(stars))
        if gaia_cache.exists():
            try:
                from astropy.table import Table

                gaia_query = Table.read(gaia_cache, format="ascii.ecsv")
                if len(gaia_query) < catalog_limit:
                    gaia_query = None
            except Exception:
                gaia_query = None
        if gaia_query is None:
            try:
                from hops.hops_tools.centroids_and_stars import _get_gaia_stars

                gaia_query = _get_gaia_stars(
                    coordinate.ra.deg,
                    coordinate.dec.deg,
                    0.5,
                    limit=catalog_limit,
                )
                gaia_cache.parent.mkdir(parents=True, exist_ok=True)
                gaia_query.write(gaia_cache, format="ascii.ecsv", overwrite=True)
            except Exception as exc:
                raise LEAPSError(
                    "GAIA_CATALOG_UNAVAILABLE",
                    "Gaia catalogue could not be reached",
                    "Manual target selection is still available. Retry online or install offline data for this target region.",
                    ["Select target manually", "Retry Gaia", "Open Offline Data settings"],
                    stage=StageID.PHOTOMETRY,
                    technical_details=f"{type(exc).__name__}: {exc}",
                ) from exc
        if existing_wcs is not None and existing_wcs.has_celestial:
            corrected = self._correct_existing_wcs(
                existing_wcs,
                data.shape,
                stars,
                gaia_query,
                coordinate,
                psf,
            )
            if corrected is not None:
                _emit(
                    emit,
                    StageID.PHOTOMETRY,
                    JobStatus.SUCCEEDED,
                    "Existing FITS WCS corrected with Gaia",
                    1,
                    1,
                )
                return corrected
        attempts: list[PlateSolveAttempt] = []
        base_scale = pixel_scale if pixel_scale > 0 else 2.0 / psf
        for index, scale in enumerate((base_scale, base_scale * 0.5, base_scale * 2.0), start=1):
            token.raise_if_cancelled()
            try:
                timestamp = (
                    Time(float(header["HOPSJD"]), format="jd")
                    if header.get("HOPSJD") is not None
                    else Time(header.get("DATE-OBS", Time.now().isot))
                )
                solution = image_plate_solve(
                    data,
                    header,
                    coordinate.ra.deg,
                    coordinate.dec.deg,
                    timestamp,
                    stars=stars,
                    pixel=scale,
                    mean=mean,
                    std=std,
                    psf=psf,
                    burn_limit=burn_limit,
                    gaia_query_ext=gaia_query,
                    verbose=False,
                )
                identified = len(solution["identified_stars"])
                if identified < 5:
                    raise ValueError(f"Only {identified} of {len(stars)} detected stars matched")
                x, y = coordinate.to_pixel(solution["plate_solution"])
                nearest_star = min(
                    math.hypot(float(star[0]) - float(x), float(star[1]) - float(y))
                    for star in stars
                )
                if nearest_star > max(5.0 * psf, 8.0):
                    raise ValueError(
                        f"Solved target is {nearest_star:.1f} pixels from the nearest detected star"
                    )
                attempts.append(PlateSolveAttempt(index, scale, "complete", f"{identified} stars matched"))
                _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Plate solution found", index, 3)
                return PlateSolveResult(
                    True,
                    attempts,
                    (float(x), float(y)),
                    identified,
                    dict(solution["plate_solution"].to_header(relax=True)),
                )
            except Exception as exc:
                attempts.append(PlateSolveAttempt(index, scale, "failed", str(exc)))
                _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, f"Solve attempt {index} failed", index, 3)
        details = "\n".join(f"Attempt {item.index}: {item.detail}" for item in attempts)
        raise LEAPSError(
            "PLATE_SOLVE_FAILED",
            "Plate solve needs attention",
            "The image and detected stars are safe. LEAPS stopped after three bounded attempts.",
            ["Retry plate solve", "Place the target manually and continue with an unverified WCS"],
            stage=StageID.PHOTOMETRY,
            technical_details=details,
        )

    @staticmethod
    def manual(target_xy: tuple[float, float]) -> PlateSolveResult:
        return PlateSolveResult(False, [], target_xy=target_xy, unverified=True)

    @staticmethod
    def _correct_existing_wcs(
        existing_wcs: Any,
        image_shape: tuple[int, int],
        stars: list[Any],
        gaia_query: Any,
        coordinate: Any,
        psf: float,
    ) -> PlateSolveResult | None:
        """Correct a plausible header WCS for telescope pointing offset.

        Many acquisition programs write the requested target coordinates as
        CRVAL even when the actual pointing is tens of pixels away. This keeps
        HOPS's Gaia catalogue and WCS fit, but gives it a robust translation
        seed before the bounded blind attempts.
        """
        from astropy.coordinates import SkyCoord
        from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
        from scipy.spatial import cKDTree

        detected = np.asarray([[star[0], star[1]] for star in stars], dtype=float)
        if len(detected) < 5:
            return None
        try:
            world = np.column_stack(
                (
                    np.asarray(gaia_query["ra"], dtype=float),
                    np.asarray(gaia_query["dec"], dtype=float),
                )
            )
            projected = np.asarray(existing_wcs.all_world2pix(world, 0), dtype=float)
        except Exception:
            return None
        height, width = image_shape
        margin = 0.2 * min(width, height)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] > -margin)
            & (projected[:, 0] < width + margin)
            & (projected[:, 1] > -margin)
            & (projected[:, 1] < height + margin)
        )
        projected = projected[valid][:150]
        world = world[valid][:150]
        if len(projected) < 5:
            return None

        tree = cKDTree(detected)
        tolerance = max(3.0 * psf, 8.0)
        max_shift = 0.25 * min(width, height)
        best_score = 0
        best_shift = None
        for catalogue_point in projected[:50]:
            for detected_point in detected[:50]:
                shift = detected_point - catalogue_point
                if np.linalg.norm(shift) > max_shift:
                    continue
                distances, indices = tree.query(projected + shift, k=1)
                score = len(set(indices[distances < tolerance].tolist()))
                if score > best_score:
                    best_score = score
                    best_shift = shift
        if best_shift is None or best_score < 5:
            return None

        distances, indices = tree.query(projected + best_shift, k=1)
        candidate_rows = np.where(distances < tolerance)[0]
        pairs: list[tuple[int, int]] = []
        used_detected: set[int] = set()
        for catalogue_index in sorted(candidate_rows, key=lambda index: distances[index]):
            detected_index = int(indices[catalogue_index])
            if detected_index not in used_detected:
                pairs.append((int(catalogue_index), detected_index))
                used_detected.add(detected_index)
        if len(pairs) < 5:
            return None

        catalogue_indices = np.asarray([pair[0] for pair in pairs], dtype=int)
        detected_indices = np.asarray([pair[1] for pair in pairs], dtype=int)
        try:
            solution = fit_wcs_from_points(
                detected[detected_indices].T,
                SkyCoord(world[catalogue_indices], unit="deg"),
                sip_degree=None,
            )
            refined = np.asarray(solution.all_world2pix(world, 0), dtype=float)
            refined_distances, refined_indices = tree.query(refined, k=1)
            candidate_rows = np.where(refined_distances < tolerance)[0]
            pairs = []
            used_detected = set()
            for catalogue_index in sorted(
                candidate_rows, key=lambda index: refined_distances[index]
            ):
                detected_index = int(refined_indices[catalogue_index])
                if detected_index not in used_detected:
                    pairs.append((int(catalogue_index), detected_index))
                    used_detected.add(detected_index)
            if len(pairs) < 5:
                return None
            catalogue_indices = np.asarray([pair[0] for pair in pairs], dtype=int)
            detected_indices = np.asarray([pair[1] for pair in pairs], dtype=int)
            solution = fit_wcs_from_points(
                detected[detected_indices].T,
                SkyCoord(world[catalogue_indices], unit="deg"),
                sip_degree=2 if len(pairs) >= 10 else None,
            )
            target_x, target_y = map(float, coordinate.to_pixel(solution))
        except Exception:
            return None

        nearest_index = int(
            np.argmin(np.hypot(detected[:, 0] - target_x, detected[:, 1] - target_y))
        )
        nearest_distance = float(
            math.hypot(
                detected[nearest_index, 0] - target_x,
                detected[nearest_index, 1] - target_y,
            )
        )
        if nearest_distance > max(5.0 * psf, 8.0):
            return None
        scales = proj_plane_pixel_scales(solution.celestial) * 3600.0
        pixel_scale = float(np.nanmedian(scales))
        return PlateSolveResult(
            True,
            [
                PlateSolveAttempt(
                    0,
                    pixel_scale,
                    "complete",
                    f"Existing FITS WCS corrected with {len(pairs)} Gaia matches",
                )
            ],
            (
                float(detected[nearest_index, 0]),
                float(detected[nearest_index, 1]),
            ),
            len(pairs),
            dict(solution.to_header(relax=True)),
        )


class PhotometryService:
    def locate_star(
        self,
        frame: str | Path,
        x: float,
        y: float,
        config: PhotometryConfig | None = None,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> dict[str, float]:
        if token:
            token.raise_if_cancelled()
        config = config or PhotometryConfig()
        data, header = _read_fits_image(Path(frame))
        return self._locate_star(data, header, x, y, config.aperture_radius, config)

    @staticmethod
    def _locate_star(
        data: np.ndarray,
        header: Any,
        x: float,
        y: float,
        aperture: float,
        config: PhotometryConfig,
    ) -> dict[str, float]:
        from hops.hops_tools.image_analysis import image_find_stars

        mean = float(header.get("HOPSMEAN", np.nanmedian(data)))
        std = float(header.get("HOPSSTD", 1.4826 * np.nanmedian(np.abs(data - mean))))
        psf = max(float(header.get("HOPSPSF", 2.0)), 1.0)
        saturation = float(
            header.get("HOPSSAT", header.get("SATURATE", np.nanmax(data)))
        ) * config.saturation_fraction
        search = max(5.0 * psf, aperture * 2.0)
        stars = image_find_stars(
            data,
            header,
            x_low=x - search,
            x_upper=x + search,
            y_low=y - search,
            y_upper=y + search,
            x_centre=x,
            y_centre=y,
            mean=mean,
            std=std,
            burn_limit=saturation,
            psf=psf,
            centroids_snr=config.centroids_snr,
            stars_snr=config.stars_snr,
            order_by_flux=False,
            absolute_aperture=aperture,
            sky_inner_aperture=config.sky_inner_aperture,
            sky_outer_aperture=config.sky_outer_aperture,
            star_limit=5,
        ) or []
        if not stars:
            raise LEAPSError(
                "PHOTOMETRY_STAR_NOT_FOUND",
                "No acceptable star was found at that position",
                "Click closer to the center of an unsaturated star inside the usable field of view.",
                ["Choose another star", "Adjust advanced detection settings"],
                stage=StageID.PHOTOMETRY,
            )
        star = min(stars, key=lambda value: math.hypot(float(value[0]) - x, float(value[1]) - y))
        gaussian_x, gaussian_y = float(star[0]), float(star[1])
        aperture_x, aperture_y = gaussian_x, gaussian_y
        total_flux = float(star[6])
        sky_flux = float(star[8])
        if config.geometric_center:
            from photutils.aperture import CircularAperture, aperture_photometry

            half_width = max(int(3.0 * psf), 1)
            x1 = max(int(gaussian_x) - half_width, 0)
            x2 = min(int(gaussian_x) + half_width + 1, data.shape[1])
            y1 = max(int(gaussian_y) - half_width, 0)
            y2 = min(int(gaussian_y) + half_width + 1, data.shape[0])
            area = np.asarray(data[y1:y2, x1:x2], dtype=float)
            area_x, area_y = np.meshgrid(
                np.arange(x1, x2) + 0.5,
                np.arange(y1, y2) + 0.5,
            )
            finite = np.isfinite(area)
            weight = float(np.sum(area[finite]))
            if weight != 0 and math.isfinite(weight):
                aperture_x = float(np.sum(area[finite] * area_x[finite]) / weight)
                aperture_y = float(np.sum(area[finite] * area_y[finite]) / weight)
                total_flux = float(
                    aperture_photometry(
                        data,
                        CircularAperture(
                            np.array([aperture_x - 0.5, aperture_y - 0.5]), aperture
                        ),
                    )["aperture_sum"][0]
                )
        gaussian_flux = float(2.0 * math.pi * star[2] * star[4] * star[5])
        aperture_flux = total_flux - sky_flux
        return {
            "x": aperture_x,
            "y": aperture_y,
            "gaussian_x": gaussian_x,
            "gaussian_y": gaussian_y,
            "aperture": float(aperture),
            "peak": float(star[2] + star[3]),
            "total_flux": total_flux,
            "background_flux": sky_flux,
            "background_error": float(star[9]),
            "aperture_flux": aperture_flux,
            "aperture_error": float(
                math.sqrt(abs(aperture_flux) / max(config.camera_gain, 1e-9) + float(star[9]) ** 2)
            ),
            "gaussian_flux": gaussian_flux,
            "gaussian_error": float(math.sqrt(abs(gaussian_flux) / max(config.camera_gain, 1e-9))),
            "hwhm": float(0.5 * 2.355 * max(star[4], star[5])),
        }

    def rank_comparisons(
        self, frame: str | Path, target_xy: tuple[float, float], limit: int = 10
    ) -> list[dict[str, float]]:
        from hops.hops_tools.image_analysis import image_find_stars

        data, header = _read_fits_image(Path(frame))
        stars = np.asarray(image_find_stars(data, header, star_limit=150) or [])
        if stars.size == 0:
            return []
        tx, ty = target_xy
        config = PhotometryConfig()
        try:
            target_flux = self._locate_star(
                data, header, target_xy[0], target_xy[1], config.aperture_radius, config
            )["aperture_flux"]
        except LEAPSError:
            target_flux = float(np.nanmedian(stars[:, 7]))
        ranked = []
        for star in stars:
            x, y = map(float, star[:2])
            peak = float(star[2] + star[3])
            flux = float(star[7])
            distance = math.hypot(x - tx, y - ty)
            if distance < max(10.0, float(header.get("HOPSPSF", 3)) * 5):
                continue
            saturation = float(header.get("HOPSSAT", np.nanmax(data)))
            flux_similarity = abs(math.log10(max(flux, 1) / max(target_flux, 1)))
            score = 1.0 - flux_similarity - 0.02 * math.log10(max(distance, 1))
            if peak >= 0.95 * saturation:
                score -= 2.0
            ranked.append(
                {
                    "x": x,
                    "y": y,
                    "peak": peak,
                    "flux": flux,
                    "distance": distance,
                    "score": score,
                }
            )
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

    def run(
        self,
        project: ProjectWorkspace,
        target_xy: tuple[float, float],
        comparisons: list[tuple[float, float]],
        aperture_radius: float,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
        config: PhotometryConfig | None = None,
    ) -> Path:
        token = token or CancellationToken()
        frames = sorted((project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if not frames:
            raise LEAPSError(
                "PHOTOMETRY_INPUT_MISSING",
                "No reduced frames are available",
                "Run Reduction before starting photometry.",
                ["Open Reduction"],
                stage=StageID.PHOTOMETRY,
            )
        positions = [target_xy, *comparisons]
        if len(positions) < 2:
            raise LEAPSError(
                "COMPARISON_STARS_REQUIRED",
                "Choose at least one comparison star",
                "Differential photometry needs the target and one or more comparison stars.",
                ["Review suggested comparison stars"],
                stage=StageID.PHOTOMETRY,
            )
        config = config or PhotometryConfig(aperture_radius=aperture_radius)
        config.aperture_radius = aperture_radius
        alignment_path = project.outputs_dir / StageID.ALIGNMENT.value / "alignment.json"
        alignment_records = []
        if alignment_path.exists():
            alignment_records = json.loads(alignment_path.read_text(encoding="utf-8"))
            failed_frames = {
                str(record.get("file"))
                for record in alignment_records
                if record.get("failed")
            }
            frames = [path for path in frames if path.name not in failed_frames]
            if not frames:
                raise LEAPSError(
                    "PHOTOMETRY_ALIGNMENT_MISSING",
                    "No successfully aligned frames are available",
                    "Review the Alignment diagnostics and rerun that stage.",
                    ["Open Alignment", "Review diagnostics"],
                    stage=StageID.PHOTOMETRY,
                )
        transforms = {record.get("file"): self._alignment_matrix(record) for record in alignment_records}
        reference_transform = transforms.get(frames[0].name, np.eye(3))
        try:
            inverse_reference = np.linalg.inv(reference_transform)
        except np.linalg.LinAlgError:
            inverse_reference = np.eye(3)

        fingerprint_payload = {
            "frames": [path.name for path in frames],
            "positions": positions,
            "config": asdict(config),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint = project.checkpoints_dir / "photometry.json"
        rows: list[dict[str, Any]] = []
        if checkpoint.exists():
            try:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("fingerprint") == fingerprint:
                    rows = list(saved.get("rows", []))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                rows = []
        start = len(rows)
        reference_psf = max(float(_read_fits_image(frames[0])[1].get("HOPSPSF", 1.0)), 1e-9)
        for index, path in enumerate(frames[start:], start=start + 1):
            token.raise_if_cancelled()
            data, header = _read_fits_image(path)
            transform = transforms.get(path.name, np.eye(3)) @ inverse_reference
            psf = max(float(header.get("HOPSPSF", reference_psf)), 1e-9)
            scale = psf / reference_psf if config.variable_aperture else 1.0
            aperture = aperture_radius * scale
            measurements = []
            for x, y in positions:
                predicted = transform @ np.array([x, y, 1.0])
                try:
                    measurement = self._locate_star(
                        data,
                        header,
                        float(predicted[0]),
                        float(predicted[1]),
                        aperture,
                        config,
                    )
                    measurement["failed"] = False
                except LEAPSError as exc:
                    measurement = {
                        "x": float(predicted[0]),
                        "y": float(predicted[1]),
                        "gaussian_x": float(predicted[0]),
                        "gaussian_y": float(predicted[1]),
                        "aperture": aperture,
                        "aperture_flux": float("nan"),
                        "aperture_error": float("nan"),
                        "gaussian_flux": float("nan"),
                        "gaussian_error": float("nan"),
                        "failed": True,
                        "reason": exc.message,
                    }
                measurements.append(measurement)
            rows.append(
                {
                    "file": path.name,
                    "jd": float(header.get("HOPSJD", index)),
                    "measurements": measurements,
                }
            )
            checkpoint.write_text(
                json.dumps({"fingerprint": fingerprint, "rows": rows}, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, f"Measured {path.name}", index, len(frames))

        aperture_array = self._light_curve(rows, "aperture_flux", "aperture_error")
        gaussian_array = self._light_curve(rows, "gaussian_flux", "gaussian_error")
        pending, target = project.begin_transaction(StageID.PHOTOMETRY)
        output = pending / "light_curve_aperture.txt"
        np.savetxt(output, aperture_array, header="JD_UTC relative_flux relative_flux_uncertainty")
        np.savetxt(
            pending / "light_curve_gauss.txt",
            gaussian_array,
            header="JD_UTC relative_flux relative_flux_uncertainty",
        )
        np.savetxt(pending / "PHOTOMETRY_APERTURE.txt", aperture_array)
        np.savetxt(pending / "PHOTOMETRY_GAUSS.txt", gaussian_array)
        np.savetxt(
            pending / "PHOTOMETRY_a.txt",
            self._measurement_table(rows, "aperture_flux", "aperture_error"),
            fmt="%s",
        )
        np.savetxt(
            pending / "PHOTOMETRY_g.txt",
            self._measurement_table(rows, "gaussian_flux", "gaussian_error"),
            fmt="%s",
        )
        (pending / "measurements.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
        )
        (pending / "photometry.json").write_text(
            json.dumps(
                {
                    "engine": "HOPS photometry",
                    "target": target_xy,
                    "comparisons": comparisons,
                    "config": asdict(config),
                    "checkpoint_fingerprint": fingerprint,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (pending / "ExoClock_info.txt").write_text(
            "\n".join(
                (
                    "LEAPS / HOPS-compatible photometry",
                    f"Target: {project.manifest.target_name or 'Unnamed target'}",
                    f"Coordinates: {project.manifest.target_ra} {project.manifest.target_dec}",
                    "Time format: JD_UTC",
                    "Time stamp: exposure start",
                    "Flux format: target flux / summed comparison flux",
                    "Suggested upload: PHOTOMETRY_APERTURE.txt",
                )
            ),
            encoding="utf-8",
        )
        self._write_figures(
            pending,
            frames[0],
            positions,
            aperture_radius,
            aperture_array,
            gaussian_array,
        )
        project.commit_transaction(pending, target)
        checkpoint.unlink(missing_ok=True)
        _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Photometry complete", len(frames), len(frames))
        return target / output.name

    @staticmethod
    def _alignment_matrix(record: dict[str, Any]) -> np.ndarray:
        if record.get("matrix"):
            matrix = np.asarray(record["matrix"], dtype=float)
            if matrix.shape == (3, 3):
                return matrix
        rotation = float(record.get("rotation", 0.0) or 0.0)
        cosine, sine = math.cos(rotation), math.sin(rotation)
        return np.array(
            [
                [cosine, -sine, float(record.get("x0", 0.0) or 0.0)],
                [sine, cosine, float(record.get("y0", 0.0) or 0.0)],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _light_curve(
        rows: list[dict[str, Any]],
        flux_key: str,
        error_key: str,
        active_comparisons: list[int] | None = None,
    ) -> np.ndarray:
        times = np.asarray([row["jd"] for row in rows], dtype=float)
        fluxes = np.asarray(
            [[star.get(flux_key, float("nan")) for star in row["measurements"]] for row in rows],
            dtype=float,
        )
        errors = np.asarray(
            [[star.get(error_key, float("nan")) for star in row["measurements"]] for row in rows],
            dtype=float,
        )
        comparison_indices = (
            list(range(1, fluxes.shape[1]))
            if active_comparisons is None
            else list(active_comparisons)
        )
        if not comparison_indices:
            raise ValueError("At least one active comparison star is required")
        comparison_flux = np.nansum(fluxes[:, comparison_indices], axis=1)
        relative = fluxes[:, 0] / comparison_flux
        comparison_error = np.sqrt(
            np.nansum(errors[:, comparison_indices] ** 2, axis=1)
        )
        relative_error = np.abs(relative) * np.sqrt(
            (errors[:, 0] / fluxes[:, 0]) ** 2
            + (comparison_error / comparison_flux) ** 2
        )
        normalization = float(np.nanmedian(relative))
        if not math.isfinite(normalization) or normalization == 0:
            normalization = 1.0
        return np.column_stack((times, relative / normalization, relative_error / normalization))

    @staticmethod
    def _measurement_table(
        rows: list[dict[str, Any]], flux_key: str, error_key: str
    ) -> np.ndarray:
        table: list[list[Any]] = []
        for row in rows:
            values: list[Any] = [row["file"], row["jd"]]
            for star in row["measurements"]:
                gaussian = flux_key == "gaussian_flux"
                values.extend(
                    (
                        star.get("gaussian_x" if gaussian else "x", float("nan")),
                        star.get("gaussian_y" if gaussian else "y", float("nan")),
                        star.get(flux_key, float("nan")),
                        star.get(error_key, float("nan")),
                    )
                )
            table.append(values)
        return np.asarray(table, dtype=object)

    @staticmethod
    def _write_figures(
        destination: Path,
        reference_frame: Path,
        positions: list[tuple[float, float]],
        aperture: float,
        aperture_curve: np.ndarray,
        gaussian_curve: np.ndarray,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle

        data = np.asarray(_read_fits_image(reference_frame)[0], dtype=float)
        median = float(np.nanmedian(data))
        std = float(1.4826 * np.nanmedian(np.abs(data - median))) or 1.0
        field = Figure(figsize=(8, 8), facecolor="white")
        FigureCanvasAgg(field)
        axis = field.add_subplot(111)
        axis.imshow(
            data,
            origin="lower",
            cmap="gray_r",
            vmin=median - 3 * std,
            vmax=median + 20 * std,
        )
        for index, (x, y) in enumerate(positions):
            color = "#d99000" if index == 0 else "#00a6d6"
            label = "T" if index == 0 else f"C{index}"
            axis.add_patch(Circle((x, y), aperture, fill=False, color=color, linewidth=1.2))
            axis.text(x + aperture + 3, y + aperture + 3, label, color=color, fontsize=9)
        axis.set_title("Selected photometry field")
        field.savefig(destination / "FOV.png", dpi=160, bbox_inches="tight")
        field.savefig(destination / "FOV.pdf", bbox_inches="tight")

        results = Figure(figsize=(10, 5), facecolor="white")
        FigureCanvasAgg(results)
        axis = results.add_subplot(111)
        start = aperture_curve[0, 0]
        axis.errorbar(
            (aperture_curve[:, 0] - start) * 24,
            aperture_curve[:, 1],
            yerr=aperture_curve[:, 2],
            fmt="ko",
            markersize=3,
            linewidth=0.7,
            label="Aperture",
        )
        axis.plot(
            (gaussian_curve[:, 0] - start) * 24,
            gaussian_curve[:, 1],
            "o",
            color="#d85845",
            markersize=3,
            label="Gaussian",
        )
        axis.set_xlabel("Time from first exposure (hours)")
        axis.set_ylabel("Normalized relative flux")
        axis.legend()
        axis.grid(alpha=0.2)
        results.savefig(destination / "RESULTS.png", dpi=160, bbox_inches="tight")
        results.savefig(destination / "RESULTS.pdf", bbox_inches="tight")


class LightCurveReviewService:

    @dataclass(slots=True)
    class Curve:
        label: str
        active: bool
        aperture: np.ndarray = field(repr=False)
        gaussian: np.ndarray = field(repr=False)
        missing_frames: int = 0

    @dataclass(slots=True)
    class Result:
        curves: list[LightCurveReviewService.Curve]
        active_comparisons: list[bool]
        preview_path: Path
        frame_count: int
        rows: list[dict[str, Any]] = field(repr=False)

    def load(
        self,
        project: ProjectWorkspace,
        active_comparisons: list[bool] | None = None,
        *,
        destination: Path | None = None,
    ) -> Result:
        rows = self._load_rows(project)
        star_count = len(rows[0]["measurements"])
        comparison_count = star_count - 1
        if active_comparisons is None:
            active_comparisons = self._saved_selection(project, comparison_count)
        active_comparisons = [bool(value) for value in active_comparisons]
        if len(active_comparisons) != comparison_count:
            raise LEAPSError(
                "LIGHT_CURVE_SELECTION_INVALID",
                "The comparison selection needs attention",
                "The saved selection no longer matches the photometry measurements.",
                ["Review all comparison stars", "Run Photometry again if stars changed"],
                stage=StageID.LIGHT_CURVE,
            )
        if not any(active_comparisons):
            raise LEAPSError(
                "LIGHT_CURVE_COMPARISON_REQUIRED",
                "Keep at least one comparison star",
                "Differential photometry needs one or more active comparison stars.",
                ["Enable a comparison star"],
                stage=StageID.LIGHT_CURVE,
            )

        times = np.asarray([row["jd"] for row in rows], dtype=float)
        curves: list[LightCurveReviewService.Curve] = []
        active_indices = [
            index + 1 for index, active in enumerate(active_comparisons) if active
        ]
        for star_index in range(star_count):
            is_active = star_index == 0 or active_comparisons[star_index - 1]
            comparison_indices = (
                active_indices
                if star_index == 0
                else [index for index in active_indices if index != star_index]
                if is_active
                else []
            )
            aperture = self._individual_curve(
                rows,
                times,
                star_index,
                comparison_indices,
                "aperture_flux",
                "aperture_error",
            )
            gaussian = self._individual_curve(
                rows,
                times,
                star_index,
                comparison_indices,
                "gaussian_flux",
                "gaussian_error",
            )
            missing = sum(
                not math.isfinite(
                    float(row["measurements"][star_index].get("aperture_flux", float("nan")))
                )
                for row in rows
            )
            curves.append(
                self.Curve(
                    label="Target" if star_index == 0 else f"C{star_index}",
                    active=is_active,
                    aperture=aperture,
                    gaussian=gaussian,
                    missing_frames=missing,
                )
            )
        preview_path = destination or project.temporary_dir / "light-curve-review.png"
        result = self.Result(
            curves=curves,
            active_comparisons=active_comparisons,
            preview_path=preview_path,
            frame_count=len(rows),
            rows=rows,
        )
        self._write_preview(result, preview_path)
        return result

    def commit(
        self, project: ProjectWorkspace, active_comparisons: list[bool]
    ) -> Path:
        result = self.load(project, active_comparisons)
        active_indices = [
            index + 1
            for index, active in enumerate(result.active_comparisons)
            if active
        ]
        aperture = PhotometryService._light_curve(
            result.rows, "aperture_flux", "aperture_error", active_indices
        )
        gaussian = PhotometryService._light_curve(
            result.rows, "gaussian_flux", "gaussian_error", active_indices
        )
        pending, target = project.begin_transaction(StageID.LIGHT_CURVE)
        try:
            for filename, curve in (
                ("light_curve_aperture.txt", aperture),
                ("PHOTOMETRY_APERTURE.txt", aperture),
                ("light_curve_gauss.txt", gaussian),
                ("PHOTOMETRY_GAUSS.txt", gaussian),
            ):
                np.savetxt(
                    pending / filename,
                    curve,
                    header="JD_UTC relative_flux relative_flux_uncertainty",
                )
            (pending / "review.json").write_text(
                json.dumps(
                    {
                        "active_comparisons": result.active_comparisons,
                        "active_labels": [
                            f"C{index + 1}"
                            for index, active in enumerate(result.active_comparisons)
                            if active
                        ],
                        "frame_count": result.frame_count,
                        "source": project.relative(
                            project.outputs_dir
                            / StageID.PHOTOMETRY.value
                            / "measurements.json"
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._write_preview(result, pending / "light-curves.png")
            project.commit_transaction(pending, target)
        except BaseException:
            project.discard_pending_transaction(StageID.LIGHT_CURVE)
            raise
        project.manifest.settings["light_curve_review"] = {
            "active_comparisons": result.active_comparisons
        }
        project.save()
        return target / "light_curve_aperture.txt"

    @staticmethod
    def _load_rows(project: ProjectWorkspace) -> list[dict[str, Any]]:
        path = project.outputs_dir / StageID.PHOTOMETRY.value / "measurements.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) < 2:
                raise ValueError("No per-frame measurements were found")
            star_count = len(rows[0].get("measurements", []))
            if star_count < 2 or any(
                len(row.get("measurements", [])) != star_count for row in rows
            ):
                raise ValueError("Measurement rows have inconsistent star counts")
            return rows
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LEAPSError(
                "LIGHT_CURVE_MEASUREMENTS_INVALID",
                "The photometry measurements cannot be reviewed",
                "The target and comparison-star measurements are missing or incomplete.",
                ["Run Photometry again", "Export diagnostics if this repeats"],
                stage=StageID.LIGHT_CURVE,
                technical_details=f"{path}\n{exc}",
            ) from exc

    @staticmethod
    def _saved_selection(project: ProjectWorkspace, count: int) -> list[bool]:
        saved = project.manifest.settings.get("light_curve_review", {}).get(
            "active_comparisons", []
        )
        return [bool(value) for value in saved] if len(saved) == count else [True] * count

    @staticmethod
    def _individual_curve(
        rows: list[dict[str, Any]],
        times: np.ndarray,
        star_index: int,
        comparison_indices: list[int],
        flux_key: str,
        error_key: str,
    ) -> np.ndarray:
        if not comparison_indices:
            return np.column_stack(
                (times, np.full(times.size, np.nan), np.full(times.size, np.nan))
            )
        fluxes = np.asarray(
            [
                [star.get(flux_key, float("nan")) for star in row["measurements"]]
                for row in rows
            ],
            dtype=float,
        )
        errors = np.asarray(
            [
                [star.get(error_key, float("nan")) for star in row["measurements"]]
                for row in rows
            ],
            dtype=float,
        )
        denominator = np.nansum(fluxes[:, comparison_indices], axis=1)
        denominator_error = np.sqrt(
            np.nansum(errors[:, comparison_indices] ** 2, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = fluxes[:, star_index] / denominator
            relative_error = np.abs(relative) * np.sqrt(
                (errors[:, star_index] / fluxes[:, star_index]) ** 2
                + (denominator_error / denominator) ** 2
            )
        relative[denominator == 0] = np.nan
        relative_error[denominator == 0] = np.nan
        normalization = float(np.nanmedian(relative))
        if not math.isfinite(normalization) or normalization == 0:
            normalization = 1.0
        return np.column_stack(
            (times, relative / normalization, relative_error / normalization)
        )

    @staticmethod
    def _write_preview(result: Result, destination: Path) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        destination.parent.mkdir(parents=True, exist_ok=True)
        height = max(4.5, 1.5 * len(result.curves))
        figure = Figure(figsize=(10, height), facecolor="#0b2638", constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(result.curves), 1, sharex=True, squeeze=False)[:, 0]
        start = result.curves[0].aperture[0, 0]
        for index, (axis, curve) in enumerate(zip(axes, result.curves, strict=True)):
            axis.set_facecolor("#071827")
            axis.tick_params(colors="#a9bdd0", labelsize=8)
            axis.grid(color="#28516b", alpha=0.35)
            for spine in axis.spines.values():
                spine.set_color("#28516b")
            axis.set_ylabel(curve.label, color="#dce9f3", rotation=0, labelpad=24)
            if curve.active and np.any(np.isfinite(curve.aperture[:, 1])):
                hours = (curve.aperture[:, 0] - start) * 24
                axis.plot(
                    hours,
                    curve.aperture[:, 1],
                    "o",
                    color="#20c5f4",
                    markersize=2.4,
                    label="Aperture",
                )
                axis.plot(
                    hours,
                    curve.gaussian[:, 1],
                    "o",
                    color="#ffc443",
                    markersize=2.0,
                    alpha=0.72,
                    label="PSF",
                )
                if index == 0:
                    axis.legend(
                        loc="best",
                        facecolor="#0b2638",
                        edgecolor="#28516b",
                        labelcolor="#dce9f3",
                    )
            else:
                message = (
                    "Excluded from comparison ensemble"
                    if not curve.active
                    else "A second active comparison is needed to plot this star"
                )
                axis.text(
                    0.5,
                    0.5,
                    message,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#71859a",
                )
        axes[-1].set_xlabel("Time from first exposure (hours)", color="#dce9f3")
        figure.savefig(destination, dpi=150, facecolor=figure.get_facecolor())


class FittingService:

    @dataclass(slots=True)
    class Result:
        full: bool
        planet: str
        passband: str
        preview_path: Path
        output_path: Path | None
        residual_std: float | None
        raw: dict[str, Any] = field(repr=False)

    def run(
        self,
        project: ProjectWorkspace,
        parameters: PlanetParameters,
        *,
        full: bool,
        exposure_time: float,
        filter_name: str,
        latitude: float | None,
        longitude: float | None,
        light_curve: str = "aperture",
        detrending: str = "automatic",
        iterations: int = 5000,
        burn_in: int = 1000,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Result:
        token = token or CancellationToken()
        started_at = time.monotonic()
        sampling_started_at: float | None = None

        def report_progress(
            phase: str,
            current: int = 0,
            total: int = 0,
            details: dict[str, Any] | None = None,
        ) -> None:
            nonlocal sampling_started_at
            now = time.monotonic()
            payload = dict(details or {})
            payload["phase"] = phase
            payload["elapsed_seconds"] = max(0.0, now - started_at)
            if phase == "sampling":
                if sampling_started_at is None:
                    sampling_started_at = now
                sampling_elapsed = max(0.0, now - sampling_started_at)
                if current > 0 and total > current:
                    payload["eta_seconds"] = sampling_elapsed * (total - current) / current
            message = {
                "preparing_observations": "Preparing observations",
                "optimizing_initial_parameters": "Optimizing initial parameters",
                "sampling": "Sampling posterior",
                "writing_results": "Writing fit results",
            }.get(phase, "Running full fit")
            _emit(
                emit,
                StageID.FITTING,
                JobStatus.RUNNING,
                message,
                current,
                total,
                checkpoint=phase,
                details=payload,
            )

        def check_cancelled() -> None:
            if token.cancelled:
                raise LEAPSError(
                    "JOB_CANCELLED",
                    "Fit cancelled",
                    "The incomplete fitting attempt was discarded. Previous results were preserved.",
                    ["Run the fit again when ready"],
                    stage=StageID.FITTING,
                )

        report_progress("preparing_observations")
        check_cancelled()
        filter_name = normalize_filter(filter_name) or filter_name
        try:
            import exoclock

            import hops.pylightcurve41 as plc
        except BaseException as exc:
            raise LEAPSError(
                "FITTING_ASSETS_UNAVAILABLE",
                "The fitting assets are not ready",
                "PyLightcurve or ExoClock could not be opened for this fit.",
                ["Open Settings → Offline Data", "Validate or update the fitting data", "Retry"],
                stage=StageID.FITTING,
                technical_details=str(exc),
            ) from exc
        cancelled_error = getattr(plc, "PyLCCancelled", ())

        light_curve_key = light_curve.strip().casefold()
        light_curve_files = {
            "aperture": "light_curve_aperture.txt",
            "gaussian": "light_curve_gauss.txt",
        }
        if light_curve_key not in light_curve_files:
            raise LEAPSError(
                "FITTING_LIGHT_CURVE_UNKNOWN",
                "Choose a valid light curve",
                "The selected Photometry light curve is not available for fitting.",
                ["Choose Aperture photometry", "Choose Gaussian photometry"],
                stage=StageID.FITTING,
                technical_details=f"Selected light curve: {light_curve}",
            )
        light_curve_path = (
            project.outputs_dir / StageID.LIGHT_CURVE.value / light_curve_files[light_curve_key]
        )
        try:
            light_curve = np.loadtxt(light_curve_path, unpack=True)
            if light_curve.ndim != 2 or light_curve.shape[0] < 3 or light_curve.shape[1] < 10:
                raise ValueError("The light curve must contain at least 10 rows and three columns")
            if not np.all(np.isfinite(light_curve[:3])):
                raise ValueError("The light curve contains non-finite time, flux, or uncertainty values")
        except (OSError, ValueError) as exc:
            raise LEAPSError(
                "FITTING_LIGHT_CURVE_INVALID",
                "The photometry light curve cannot be fitted",
                f"The selected {light_curve_key} light curve is missing or incomplete.",
                ["Review the Light Curve", "Run Photometry again", "Retry Preview Fit"],
                stage=StageID.FITTING,
                technical_details=f"{light_curve_path}\n{exc}",
            ) from exc
        tess_import = project.manifest.settings.get("tess_import")
        tess_metadata: dict[str, Any] | None = None
        time_format = "JD_UTC"
        time_stamp = "start"
        if isinstance(tess_import, dict):
            check_cancelled()
            light_curve, parameters, tess_metadata = self._prepare_tess_phase_folded_curve(
                light_curve,
                parameters,
            )
            time_format = "BJD_TDB"
            time_stamp = "mid"
        if filter_name not in plc.all_filters():
            raise LEAPSError(
                "FITTING_FILTER_UNAVAILABLE",
                "The selected filter is not installed",
                f"{filter_name} is not available to the HOPS fitting engine.",
                ["Choose a HOPS-compatible filter", "Validate Offline Data", "Retry Preview Fit"],
                stage=StageID.FITTING,
                technical_details=f"Available filters: {', '.join(sorted(plc.all_filters()))}",
            )
        if exposure_time <= 0:
            raise LEAPSError(
                "FITTING_EXPOSURE_INVALID",
                "The exposure time needs attention",
                "A positive science-frame exposure time is required for fitting.",
                ["Return to Data & Target", "Confirm the science FITS headers", "Retry Preview Fit"],
                stage=StageID.FITTING,
            )
        planet = plc.Planet(
            parameters.name,
            exoclock.Hours(parameters.ra).deg(),
            exoclock.Degrees(parameters.dec).deg_coord(),
            parameters.logg,
            parameters.temperature,
            parameters.metallicity,
            parameters.rp_over_rs,
            parameters.period,
            parameters.sma_over_rs,
            parameters.eccentricity,
            parameters.inclination,
            parameters.periastron,
            parameters.mid_time,
        )
        has_observer_location = latitude is not None and longitude is not None
        detrending_key = detrending.strip().casefold()
        if detrending_key == "automatic":
            detrending_key = "linear" if tess_metadata is not None else (
                "airmass" if has_observer_location else "linear"
            )
        detrending_options = {
            "airmass": ("airmass", 1),
            "quadratic": ("time", 2),
            "linear": ("time", 1),
        }
        if detrending_key not in detrending_options:
            raise LEAPSError(
                "FITTING_DETRENDING_UNKNOWN",
                "Choose a valid de-trending method",
                "The selected de-trending method is not available to the HOPS fitting engine.",
                ["Choose Airmass, Quadratic, or Linear"],
                stage=StageID.FITTING,
                technical_details=f"Selected de-trending method: {detrending}",
            )
        if tess_metadata is not None and detrending_key == "airmass":
            raise LEAPSError(
                "TESS_AIRMASS_UNAVAILABLE",
                "Airmass is not available for TESS data",
                "TESS photometry is space-based. Use Linear or Quadratic de-trending for the phase-folded primary transit.",
                ["Choose Linear de-trending", "Choose Quadratic de-trending"],
                stage=StageID.FITTING,
            )
        if detrending_key == "airmass" and not has_observer_location:
            raise LEAPSError(
                "FITTING_AIRMASS_LOCATION_REQUIRED",
                "Observer location is required for Airmass",
                "Add the observatory latitude and longitude before using Airmass de-trending.",
                ["Open Settings", "Choose Linear or Quadratic de-trending"],
                stage=StageID.FITTING,
            )
        detrending_series, detrending_order = detrending_options[detrending_key]
        pending: Path | None = None
        target: Path | None = None
        try:
            planet.add_observation(
                time=light_curve[0],
                time_format=time_format,
                exp_time=exposure_time,
                time_stamp=time_stamp,
                flux=light_curve[1],
                flux_unc=light_curve[2],
                flux_format="flux",
                filter_name=filter_name,
                observatory_latitude=latitude,
                observatory_longitude=longitude,
                detrending_series=detrending_series,
                detrending_order=detrending_order,
            )
            if full:
                pending, target = project.begin_transaction(StageID.FITTING)
                preview_path = pending / "fit-preview.png"
            else:
                preview_path = project.temporary_dir / "fitting-preview.png"
            result = planet.transit_fitting(
                output_folder=str(pending) if full else None,
                scale_uncertainties=True,
                filter_outliers=True,
                fit_sma_over_rs=False,
                fit_inclination=False,
                counter=None,
                window_counter=False,
                iterations=iterations,
                burn_in=burn_in,
                optimiser="emcee" if full else "curve_fit",
                return_traces=full,
                progress_callback=report_progress,
                cancelled=lambda: token.cancelled,
            )
            check_cancelled()
            observation = result.get("observations", {}).get("obs0")
            if not observation:
                raise ValueError("The fitting engine returned no observation result")
            predicted_model = planet.transit_integrated(
                observation["detrended_series"]["time"],
                float(exposure_time),
                filter_name,
            )
            _write_fit_preview(
                observation,
                predicted_model,
                preview_path,
                catalog_parameters=parameters,
                observation_times_jd=light_curve[0],
                exposure_time=exposure_time,
                filter_name=filter_name,
            )
            check_cancelled()
            residual_std = observation.get("detrended_statistics", {}).get("res_std")
            actual_walkers = result.get("settings", {}).get("walkers")
            fitted_mid_time = _optional_finite_float(
                observation.get("parameters", {}).get("mid_time", {}).get("value")
            )
            summary = {
                "planet": parameters.name,
                "source": parameters.source,
                "source_date": parameters.source_date,
                "passband": filter_name,
                "exposure_time": exposure_time,
                "light_curve": light_curve_key,
                "detrending": detrending_key,
                "walkers": actual_walkers,
                "walker_policy": "hops_auto",
                "iterations": iterations,
                "burn_in": burn_in,
                "residual_std": residual_std,
                "complete": bool(result),
                "parameters": asdict(parameters),
                "fitted_ephemeris": {
                    "period": parameters.period,
                    "mid_time": fitted_mid_time or parameters.mid_time,
                    "time_standard": "BJD_TDB",
                },
            }
            if tess_metadata is not None:
                summary["tess_phase_folded_fit"] = tess_metadata
            if full and pending is not None and target is not None:
                (pending / "fit-summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                project.commit_transaction(pending, target)
                preview_path = target / preview_path.name
            else:
                (project.temporary_dir / "fitting-preview.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
            return self.Result(
                full=full,
                planet=parameters.name,
                passband=filter_name,
                preview_path=preview_path,
                output_path=target if full else None,
                residual_std=float(residual_std) if residual_std is not None else None,
                raw=result,
            )
        except cancelled_error as exc:
            project.discard_pending_transaction(StageID.FITTING)
            raise LEAPSError(
                "JOB_CANCELLED",
                "Full fit cancelled",
                "The incomplete fitting attempt was discarded. Previous results were preserved.",
                ["Run Full Fit again when ready"],
                stage=StageID.FITTING,
                technical_details=str(exc),
            ) from exc
        except plc.PyLCInputError as exc:
            project.discard_pending_transaction(StageID.FITTING)
            raise LEAPSError(
                "FITTING_INPUT_INVALID",
                "The fitting setup needs attention",
                "The HOPS fitting engine rejected one or more observation settings.",
                ["Review the planet and filter", "Run Preview Fit again", "Export diagnostics if it repeats"],
                stage=StageID.FITTING,
                technical_details=str(exc),
            ) from exc
        except LEAPSError:
            project.discard_pending_transaction(StageID.FITTING)
            raise
        except BaseException as exc:
            project.discard_pending_transaction(StageID.FITTING)
            raise LEAPSError(
                "FITTING_FAILED",
                "The fit could not be completed",
                "LEAPS kept the last successful fitting result unchanged.",
                ["Run Preview Fit", "Review the fitting setup", "Export diagnostics if it repeats"],
                stage=StageID.FITTING,
                technical_details=str(exc),
            ) from exc

    @staticmethod
    def _prepare_tess_phase_folded_curve(
        light_curve: np.ndarray,
        parameters: PlanetParameters,
    ) -> tuple[np.ndarray, PlanetParameters, dict[str, Any]]:
        """Refine a known TESS transit ephemeris and fold it into one HOPS observation.

        HOPS is designed for a single ground-based observing run.  A TESS
        download can span many sectors and contain both primary and secondary
        events, so passing the full time series to HOPS makes it classify the
        observation incorrectly.  This catalog-guided BLS step finds the
        primary transit near the known period, then phase-folds and bins only a
        local primary-transit window for the normal HOPS fit.
        """
        from astropy.time import Time
        from astropy.timeseries import BoxLeastSquares

        time_utc, flux, uncertainty = (np.asarray(column, dtype=float) for column in light_curve[:3])
        times_bjd = np.asarray(Time(time_utc, format="jd", scale="utc").tdb.jd, dtype=float)
        period_catalog = float(parameters.period)
        if not math.isfinite(period_catalog) or period_catalog <= 0:
            raise LEAPSError(
                "TESS_PERIOD_INVALID",
                "The catalog period needs attention",
                "A positive catalog period is needed to phase-fold imported TESS data.",
                ["Choose a catalogued planet", "Check the period in Fitting"],
                stage=StageID.FITTING,
            )
        duration_hours = SecondaryEclipseService.estimate_duration_hours(parameters)
        duration_days = max(duration_hours / 24.0, 1.0 / 24.0)
        search_half_width = max(3.0e-5, min(0.02, period_catalog * 0.002))
        lower_period = max(duration_days * 1.5, period_catalog - search_half_width)
        periods = np.linspace(lower_period, period_catalog + search_half_width, 1001)
        durations = np.unique(
            np.clip(
                duration_days * np.asarray((0.65, 0.8, 1.0, 1.2, 1.45)),
                1.0 / 24.0,
                lower_period * 0.3,
            )
        )
        refined_period = period_catalog
        refined_mid_time = float(parameters.mid_time)
        bls_power: float | None = None
        try:
            bls = BoxLeastSquares(times_bjd, flux, dy=uncertainty)
            result = bls.power(periods, durations, objective="snr")
            index = int(np.nanargmax(result.power))
            candidate_period = float(result.period[index])
            candidate_mid_time = float(result.transit_time[index])
            candidate_power = float(result.power[index])
            if (
                math.isfinite(candidate_period)
                and candidate_period > 0
                and math.isfinite(candidate_mid_time)
                and math.isfinite(candidate_power)
            ):
                refined_period = candidate_period
                refined_mid_time = candidate_mid_time
                bls_power = candidate_power
        except (ValueError, FloatingPointError, IndexError):
            # The catalog ephemeris remains a safe fallback when BLS cannot be
            # evaluated, for example for a short or unusually noisy sector.
            pass

        refined_parameters = replace(
            parameters,
            period=refined_period,
            mid_time=refined_mid_time,
        )
        phase = np.mod((times_bjd - refined_mid_time) / refined_period + 0.5, 1.0) - 0.5
        duration_phase = duration_days / refined_period
        window_phase = min(0.18, max(0.025, duration_phase * 2.5))
        local = np.abs(phase) <= window_phase
        if int(local.sum()) < 30:
            raise LEAPSError(
                "TESS_PRIMARY_TRANSIT_UNCOVERED",
                "The imported TESS data do not cover enough primary transit",
                "LEAPS could not build a phase-folded primary-transit fit near the catalog ephemeris.",
                ["Choose more TESS sectors", "Check the selected planet and ephemeris"],
                stage=StageID.FITTING,
            )
        folded = FittingService._bin_tess_phase_curve(
            phase[local],
            flux[local],
            uncertainty[local],
            mid_time=refined_mid_time,
            period=refined_period,
            window_phase=window_phase,
        )
        if folded.shape[1] < 20:
            raise LEAPSError(
                "TESS_PRIMARY_TRANSIT_SPARSE",
                "The phase-folded TESS transit is too sparse",
                "LEAPS could not form enough reliable phase bins for a primary-transit fit.",
                ["Choose more TESS sectors", "Check the selected planet and ephemeris"],
                stage=StageID.FITTING,
            )
        return (
            folded,
            refined_parameters,
            {
                "method": "catalog-guided Box Least Squares + phase-folded HOPS transit fit",
                "source_points": int(time_utc.size),
                "local_primary_points": int(local.sum()),
                "phase_bins": int(folded.shape[1]),
                "period_catalog_days": period_catalog,
                "period_bls_days": refined_period,
                "mid_time_bls_bjd_tdb": refined_mid_time,
                "bls_snr": bls_power,
                "duration_hours": duration_hours,
                "window_phase": window_phase,
            },
        )

    @staticmethod
    def _bin_tess_phase_curve(
        phase: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        *,
        mid_time: float,
        period: float,
        window_phase: float,
    ) -> np.ndarray:
        """Compress a many-sector TESS phase curve without hiding its scatter."""
        order = np.argsort(phase)
        phase, flux, uncertainty = phase[order], flux[order], uncertainty[order]
        cadence_phase = float(np.nanmedian(np.diff(phase))) if phase.size > 1 else 0.0
        target_width = max(cadence_phase, 2.0 * window_phase / 320.0)
        bin_count = int(np.clip(round(2.0 * window_phase / target_width), 60, 360))
        edges = np.linspace(-window_phase, window_phase, bin_count + 1)
        assignment = np.clip(np.digitize(phase, edges) - 1, 0, bin_count - 1)
        times: list[float] = []
        fluxes: list[float] = []
        uncertainties: list[float] = []
        for index in range(bin_count):
            points = assignment == index
            count = int(points.sum())
            if count < 2:
                continue
            local_uncertainty = uncertainty[points]
            weights = 1.0 / np.square(local_uncertainty)
            weight_sum = float(weights.sum())
            if not math.isfinite(weight_sum) or weight_sum <= 0:
                continue
            phase_value = float(np.average(phase[points], weights=weights))
            flux_value = float(np.average(flux[points], weights=weights))
            formal = math.sqrt(1.0 / weight_sum)
            scatter = float(np.std(flux[points], ddof=1)) / math.sqrt(count)
            times.append(mid_time + phase_value * period)
            fluxes.append(flux_value)
            uncertainties.append(max(formal, scatter))
        return np.asarray((times, fluxes, uncertainties), dtype=float)


class SecondaryEclipseService:
    """Fit a conservative, fixed-phase secondary-eclipse model to an approved light curve.

    This deliberately does not scan arbitrary phases for the deepest dip.  It evaluates the
    user-visible ephemeris and nearby control phases instead, so a result is easier to interpret
    and less likely to turn ordinary systematics into a discovery claim.
    """

    LIGHT_CURVE_FILES = {
        "aperture": "light_curve_aperture.txt",
        "gaussian": "light_curve_gauss.txt",
    }

    @dataclass(slots=True)
    class Result:
        planet: str
        outcome: str
        outcome_label: str
        message: str
        preview_path: Path
        output_path: Path
        depth_ppm: float | None
        depth_uncertainty_ppm: float | None
        significance: float | None
        red_noise_beta: float | None
        residual_rms_ppm: float | None
        expected_phase: float
        duration_hours: float
        event_count: int
        local_points: int
        in_eclipse_points: int
        control_significance: float | None
        time_standard: str
        raw: dict[str, Any] = field(repr=False)

    @staticmethod
    def estimate_duration_hours(parameters: PlanetParameters) -> float:
        """Return a defensible first duration estimate from the catalogued transit geometry."""
        try:
            period = float(parameters.period)
            scaled_axis = float(parameters.sma_over_rs)
            radius_ratio = float(parameters.rp_over_rs)
            inclination = math.radians(float(parameters.inclination))
            if period <= 0 or scaled_axis <= 0:
                raise ValueError("period and scaled semi-major axis must be positive")
            impact_parameter = abs(scaled_axis * math.cos(inclination))
            chord_squared = (1.0 + radius_ratio) ** 2 - impact_parameter**2
            denominator = scaled_axis * max(math.sin(inclination), 1e-6)
            if chord_squared <= 0 or denominator <= 0:
                raise ValueError("catalogued geometry is not transiting")
            argument = min(1.0, max(0.0, math.sqrt(chord_squared) / denominator))
            duration_hours = period * 24.0 * math.asin(argument) / math.pi
            if not math.isfinite(duration_hours):
                raise ValueError("duration is not finite")
            return min(max(duration_hours, 0.25), min(12.0, period * 12.0))
        except (TypeError, ValueError, OverflowError):
            return 2.0

    def run(
        self,
        project: ProjectWorkspace,
        parameters: PlanetParameters,
        *,
        expected_phase: float = 0.5,
        duration_hours: float | None = None,
        light_curve: str = "aperture",
        baseline: str = "linear",
        latitude: float | None = None,
        longitude: float | None = None,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Result:
        token = token or CancellationToken()
        if duration_hours is None:
            duration_hours = self.estimate_duration_hours(parameters)
        self._validate_inputs(parameters, expected_phase, duration_hours, light_curve, baseline)

        def check_cancelled() -> None:
            if token.cancelled:
                raise LEAPSError(
                    "JOB_CANCELLED",
                    "Secondary-eclipse analysis cancelled",
                    "The incomplete analysis was discarded. Previous eclipse results were preserved.",
                    ["Run the analysis again when ready"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )

        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Loading approved light curve",
            0,
            4,
            checkpoint="loading_curve",
        )
        check_cancelled()
        time_utc, flux, uncertainty = self._load_curve(project, light_curve)
        if isinstance(project.manifest.settings.get("tess_import"), dict):
            # The importer converted mission BJD_TDB values through UTC only
            # to match LEAPS' on-disk light-curve convention.  Convert them
            # back without adding a ground-observatory light-travel term.
            from astropy.time import Time

            times_bjd = np.asarray(Time(time_utc, format="jd", scale="utc").tdb.jd, dtype=float)
            time_standard = "TESS BJD_TDB (mission-corrected)"
        else:
            times_bjd, time_standard = self._to_bjd_tdb(
                time_utc,
                parameters,
                latitude=latitude,
                longitude=longitude,
            )
        check_cancelled()
        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Converting observation times",
            1,
            4,
            checkpoint="converting_times",
        )

        period = float(parameters.period)
        duration_phase = float(duration_hours) / (period * 24.0)
        window_phase = min(0.24, max(0.035, duration_phase * 3.0))
        phase = self._relative_phase(times_bjd, float(parameters.mid_time), period, expected_phase)
        fit = self._fit_window(
            phase,
            times_bjd,
            flux,
            uncertainty,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        check_cancelled()
        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Fitting fixed-phase eclipse model",
            2,
            4,
            checkpoint="fitting_eclipse",
        )

        controls = self._control_fits(
            times_bjd,
            flux,
            uncertainty,
            parameters,
            expected_phase,
            duration_phase,
            window_phase,
            baseline,
        )
        outcome, label, message = self._classify(fit, controls)
        event_count = self._observed_event_count(
            times_bjd,
            phase,
            float(parameters.mid_time),
            period,
            expected_phase,
            fit["local_mask"],
        )
        if outcome == "candidate" and event_count < 2:
            message += " Only one eclipse window is represented, so independent data are essential."
        if time_standard not in {"BJD_TDB", "TESS BJD_TDB (mission-corrected)"}:
            message += " Set observatory coordinates for a barycentric timing correction."
        check_cancelled()

        depth = fit.get("depth")
        depth_uncertainty = fit.get("depth_uncertainty")
        significance = fit.get("significance")
        beta = fit.get("red_noise_beta")
        residual_rms = fit.get("residual_rms")
        control_significance = self._strongest_control_significance(controls)
        summary = {
            "analysis": "fixed-phase secondary eclipse / occultation",
            "planet": parameters.name,
            "parameters": asdict(parameters),
            "light_curve": light_curve,
            "baseline": baseline,
            "expected_phase": expected_phase,
            "duration_hours": duration_hours,
            "duration_phase": duration_phase,
            "window_phase": window_phase,
            "time_standard": time_standard,
            "outcome": outcome,
            "outcome_label": label,
            "message": message,
            "depth_ppm": self._as_ppm(depth),
            "depth_uncertainty_ppm": self._as_ppm(depth_uncertainty),
            "significance": significance,
            "red_noise_beta": beta,
            "residual_rms_ppm": self._as_ppm(residual_rms),
            "event_count": event_count,
            "local_points": int(fit["local_mask"].sum()),
            "in_eclipse_points": int(fit["in_eclipse_mask"].sum()),
            "control_significance": control_significance,
            "coverage": fit["coverage"],
            "controls": controls,
            "model": {
                "depth": depth,
                "depth_uncertainty": depth_uncertainty,
                "delta_chi_squared": fit.get("delta_chi_squared"),
                "points_used": fit.get("points_used"),
            },
            "interpretation": (
                "A secondary eclipse measures the planet-to-star flux ratio in this passband. "
                "It is not, by itself, a measurement of geometric albedo or a confirmation of a planet."
            ),
        }
        pending: Path | None = None
        try:
            _emit(
                emit,
                StageID.SECONDARY_ECLIPSE,
                JobStatus.RUNNING,
                "Writing eclipse diagnostics",
                3,
                4,
                checkpoint="writing_results",
            )
            pending, target = project.begin_transaction(StageID.SECONDARY_ECLIPSE)
            self._write_outputs(
                pending,
                phase,
                times_bjd,
                flux,
                uncertainty,
                fit,
                summary,
                duration_phase=duration_phase,
                window_phase=window_phase,
            )
            check_cancelled()
            project.commit_transaction(pending, target)
            _emit(
                emit,
                StageID.SECONDARY_ECLIPSE,
                JobStatus.SUCCEEDED,
                "Secondary-eclipse analysis complete",
                4,
                4,
                checkpoint="complete",
            )
        except BaseException:
            project.discard_pending_transaction(StageID.SECONDARY_ECLIPSE)
            raise

        return self.Result(
            planet=parameters.name,
            outcome=outcome,
            outcome_label=label,
            message=message,
            preview_path=target / "secondary-eclipse.png",
            output_path=target,
            depth_ppm=self._as_ppm(depth),
            depth_uncertainty_ppm=self._as_ppm(depth_uncertainty),
            significance=significance,
            red_noise_beta=beta,
            residual_rms_ppm=self._as_ppm(residual_rms),
            expected_phase=expected_phase,
            duration_hours=duration_hours,
            event_count=event_count,
            local_points=int(fit["local_mask"].sum()),
            in_eclipse_points=int(fit["in_eclipse_mask"].sum()),
            control_significance=control_significance,
            time_standard=time_standard,
            raw=summary,
        )

    @classmethod
    def _validate_inputs(
        cls,
        parameters: PlanetParameters,
        expected_phase: float,
        duration_hours: float,
        light_curve: str,
        baseline: str,
    ) -> None:
        if light_curve not in cls.LIGHT_CURVE_FILES:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_LIGHT_CURVE_UNKNOWN",
                "Choose a valid light curve",
                "The selected approved light curve is not available for secondary-eclipse analysis.",
                ["Choose Aperture photometry", "Choose Gaussian photometry"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        if baseline not in {"constant", "linear", "quadratic"}:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_BASELINE_UNKNOWN",
                "Choose a valid baseline",
                "Use a constant, linear, or quadratic baseline for the local eclipse fit.",
                ["Choose a listed baseline"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        if not math.isfinite(float(parameters.period)) or float(parameters.period) <= 0:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_PERIOD_INVALID",
                "The fitted period needs attention",
                "A positive, finite orbital period is required to phase the secondary eclipse.",
                ["Run the full transit fit again", "Check the fitted ephemeris"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        if not 0.05 <= float(expected_phase) <= 0.95:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_PHASE_INVALID",
                "The expected eclipse phase needs attention",
                "Choose a phase between 0.05 and 0.95 so this analysis cannot overlap the primary transit.",
                ["Use phase 0.50 for a circular orbit", "Check an eccentric-orbit ephemeris"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        if not math.isfinite(float(duration_hours)) or not 0.05 <= float(duration_hours) <= 24.0:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_DURATION_INVALID",
                "The eclipse duration needs attention",
                "Choose an expected duration between 3 minutes and 24 hours.",
                ["Use the suggested duration", "Check the transit geometry"],
                stage=StageID.SECONDARY_ECLIPSE,
            )

    def _load_curve(
        self, project: ProjectWorkspace, light_curve: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = project.outputs_dir / StageID.LIGHT_CURVE.value / self.LIGHT_CURVE_FILES[light_curve]
        try:
            data = np.loadtxt(path, unpack=True)
            if data.ndim != 2 or data.shape[0] < 3:
                raise ValueError("the file does not contain three columns")
            time_utc, flux, uncertainty = (np.asarray(column, dtype=float) for column in data[:3])
            valid = np.isfinite(time_utc) & np.isfinite(flux) & np.isfinite(uncertainty) & (uncertainty > 0)
            time_utc, flux, uncertainty = time_utc[valid], flux[valid], uncertainty[valid]
            if time_utc.size < 10:
                raise ValueError("fewer than 10 finite measurements remain")
            return time_utc, flux, uncertainty
        except (OSError, ValueError) as exc:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_LIGHT_CURVE_INVALID",
                "The approved light curve cannot be analysed",
                "LEAPS needs at least 10 finite time, flux, and uncertainty measurements.",
                ["Review the Light Curve", "Run Photometry again", "Choose the other photometry method"],
                stage=StageID.SECONDARY_ECLIPSE,
                technical_details=f"{path}\n{exc}",
            ) from exc

    @staticmethod
    def _to_bjd_tdb(
        time_utc: np.ndarray,
        parameters: PlanetParameters,
        *,
        latitude: float | None,
        longitude: float | None,
    ) -> tuple[np.ndarray, str]:
        from astropy.time import Time

        times = Time(time_utc, format="jd", scale="utc")
        if latitude is None or longitude is None:
            return np.asarray(times.tdb.jd, dtype=float), "TDB (observatory correction unavailable)"
        try:
            import astropy.units as units
            from astropy.coordinates import EarthLocation, SkyCoord

            location = EarthLocation.from_geodetic(
                float(longitude) * units.deg,
                float(latitude) * units.deg,
            )
            target = SkyCoord(parameters.ra, parameters.dec, unit=(units.hourangle, units.deg))
            observed = Time(time_utc, format="jd", scale="utc", location=location)
            barycentric = observed.light_travel_time(target, kind="barycentric")
            return np.asarray((observed.tdb + barycentric).jd, dtype=float), "BJD_TDB"
        except Exception:
            return np.asarray(times.tdb.jd, dtype=float), "TDB (observatory correction unavailable)"

    @staticmethod
    def _relative_phase(
        times: np.ndarray, mid_time: float, period: float, expected_phase: float
    ) -> np.ndarray:
        cycles = (np.asarray(times, dtype=float) - mid_time) / period - expected_phase
        return np.mod(cycles + 0.5, 1.0) - 0.5

    @staticmethod
    def _eclipse_template(phase: np.ndarray, duration_phase: float) -> np.ndarray:
        half_duration = max(duration_phase / 2.0, 1e-8)
        ingress = max(half_duration * 0.18, 1e-8)
        flat_half = max(0.0, half_duration - ingress)
        absolute_phase = np.abs(np.asarray(phase, dtype=float))
        template = np.zeros_like(absolute_phase)
        template[absolute_phase <= flat_half] = 1.0
        slope = (absolute_phase > flat_half) & (absolute_phase < half_duration)
        template[slope] = (half_duration - absolute_phase[slope]) / ingress
        return template

    @classmethod
    def _fit_window(
        cls,
        phase: np.ndarray,
        times: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        *,
        duration_phase: float,
        window_phase: float,
        baseline: str,
    ) -> dict[str, Any]:
        local_mask = np.abs(phase) <= window_phase
        in_eclipse_mask = local_mask & (np.abs(phase) <= duration_phase / 2.0)
        before_mask = local_mask & (phase < -duration_phase / 2.0)
        after_mask = local_mask & (phase > duration_phase / 2.0)
        coverage = {
            "available": bool(
                local_mask.sum() >= 12
                and in_eclipse_mask.sum() >= 3
                and before_mask.sum() >= 3
                and after_mask.sum() >= 3
            ),
            "local_points": int(local_mask.sum()),
            "in_eclipse_points": int(in_eclipse_mask.sum()),
            "before_points": int(before_mask.sum()),
            "after_points": int(after_mask.sum()),
        }
        result: dict[str, Any] = {
            "local_mask": local_mask,
            "in_eclipse_mask": in_eclipse_mask,
            "coverage": coverage,
            "model": np.full(int(local_mask.sum()), np.nan),
            "baseline_model": np.full(int(local_mask.sum()), np.nan),
            "residuals": np.full(int(local_mask.sum()), np.nan),
            "template": cls._eclipse_template(phase[local_mask], duration_phase),
            "depth": None,
            "depth_uncertainty": None,
            "significance": None,
            "red_noise_beta": None,
            "residual_rms": None,
            "delta_chi_squared": None,
            "points_used": 0,
        }
        if not coverage["available"]:
            return result

        local_phase = phase[local_mask]
        local_flux = flux[local_mask]
        local_uncertainty = uncertainty[local_mask]
        template = result["template"]
        x = local_phase / max(window_phase, 1e-8)
        design_baseline = [np.ones_like(x)]
        if baseline in {"linear", "quadratic"}:
            design_baseline.append(x)
        if baseline == "quadratic":
            design_baseline.append(x**2)
        baseline_matrix = np.column_stack(design_baseline)
        design = np.column_stack((baseline_matrix, -template))
        if np.linalg.matrix_rank(design) < design.shape[1]:
            coverage["available"] = False
            coverage["reason"] = "The observed phase range cannot separate an eclipse from the baseline."
            return result

        keep = np.ones(local_flux.size, dtype=bool)
        for _ in range(2):
            _, _, _, residuals, _ = cls._weighted_fit(
                design[keep], local_flux[keep], local_uncertainty[keep], design, local_flux
            )
            scatter = max(
                cls._robust_scatter(residuals[keep]),
                float(np.nanmedian(local_uncertainty[keep])),
            )
            updated_keep = np.abs(residuals) <= 5.0 * max(scatter, 1e-12)
            if updated_keep.sum() < design.shape[1] + 3 or np.array_equal(updated_keep, keep):
                break
            keep = updated_keep

        coefficients, covariance, model, residuals, chi_squared = cls._weighted_fit(
            design[keep], local_flux[keep], local_uncertainty[keep], design, local_flux
        )
        _, _, _, _, no_eclipse_chi_squared = cls._weighted_fit(
            baseline_matrix[keep],
            local_flux[keep],
            local_uncertainty[keep],
            baseline_matrix,
            local_flux,
        )
        formal_uncertainty = float(math.sqrt(max(float(covariance[-1, -1]), 0.0)))
        beta = cls._red_noise_beta(residuals[keep], times[local_mask][keep])
        depth_uncertainty = formal_uncertainty * beta
        depth = float(coefficients[-1])
        significance = depth / depth_uncertainty if depth_uncertainty > 0 else None
        result.update(
            {
                "model": model,
                "baseline_model": model + depth * template,
                "residuals": residuals,
                "depth": depth,
                "depth_uncertainty": depth_uncertainty,
                "significance": significance,
                "red_noise_beta": beta,
                "residual_rms": float(np.std(residuals[keep], ddof=1)) if keep.sum() > 1 else None,
                "delta_chi_squared": max(0.0, no_eclipse_chi_squared - chi_squared),
                "points_used": int(keep.sum()),
                "kept_mask": keep,
            }
        )
        return result

    @staticmethod
    def _weighted_fit(
        design_used: np.ndarray,
        flux_used: np.ndarray,
        uncertainty_used: np.ndarray,
        design_all: np.ndarray,
        flux_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        weights = 1.0 / np.square(uncertainty_used)
        weighted_design = design_used * np.sqrt(weights)[:, None]
        weighted_flux = flux_used * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_flux, rcond=None)
        model_used = design_used @ coefficients
        residuals_used = flux_used - model_used
        degrees_of_freedom = max(1, flux_used.size - design_used.shape[1])
        chi_squared = float(np.sum(np.square(residuals_used / uncertainty_used)))
        scale = max(1.0, chi_squared / degrees_of_freedom)
        covariance = np.linalg.pinv(design_used.T @ (weights[:, None] * design_used)) * scale
        model_all = design_all @ coefficients
        return coefficients, covariance, model_all, np.asarray(flux_all) - model_all, chi_squared

    @staticmethod
    def _robust_scatter(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)[np.isfinite(values)]
        if finite.size < 2:
            return 0.0
        median = float(np.median(finite))
        return float(1.4826 * np.median(np.abs(finite - median)))

    @staticmethod
    def _red_noise_beta(residuals: np.ndarray, times: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=float)
        times = np.asarray(times, dtype=float)
        if residuals.size < 8:
            return 1.0
        unbinned = float(np.std(residuals, ddof=1))
        if not math.isfinite(unbinned) or unbinned <= 0:
            return 1.0
        beta = 1.0
        for minutes in (10.0, 20.0, 30.0):
            bins = np.floor((times - times.min()) * 24.0 * 60.0 / minutes).astype(int)
            unique = np.unique(bins)
            if unique.size < 2:
                continue
            binned = np.asarray([np.mean(residuals[bins == value]) for value in unique])
            counts = np.asarray([np.count_nonzero(bins == value) for value in unique])
            if binned.size < 2 or np.median(counts) <= 1:
                continue
            expected = unbinned / math.sqrt(float(np.median(counts)))
            expected *= math.sqrt(binned.size / (binned.size - 1))
            if expected > 0:
                beta = max(beta, float(np.std(binned, ddof=1)) / expected)
        return min(beta, 5.0)

    def _control_fits(
        self,
        times: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        parameters: PlanetParameters,
        expected_phase: float,
        duration_phase: float,
        window_phase: float,
        baseline: str,
    ) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        for offset in (-0.15, 0.15):
            phase_center = (expected_phase + offset) % 1.0
            phase = self._relative_phase(
                times,
                float(parameters.mid_time),
                float(parameters.period),
                phase_center,
            )
            fit = self._fit_window(
                phase,
                times,
                flux,
                uncertainty,
                duration_phase=duration_phase,
                window_phase=window_phase,
                baseline=baseline,
            )
            controls.append(
                {
                    "phase": phase_center,
                    "available": fit["coverage"]["available"],
                    "significance": fit.get("significance"),
                    "depth_ppm": self._as_ppm(fit.get("depth")),
                    "coverage": fit["coverage"],
                }
            )
        return controls

    @staticmethod
    def _strongest_control_significance(controls: list[dict[str, Any]]) -> float | None:
        values = [abs(float(control["significance"])) for control in controls if control["significance"] is not None]
        return max(values) if values else None

    def _classify(
        self, fit: dict[str, Any], controls: list[dict[str, Any]]
    ) -> tuple[str, str, str]:
        if not fit["coverage"]["available"]:
            coverage = fit["coverage"]
            detail = (
                f"{coverage['in_eclipse_points']} in-eclipse points, "
                f"{coverage['before_points']} before, and {coverage['after_points']} after."
            )
            return (
                "inconclusive",
                "Inconclusive · no usable coverage",
                "No supported local eclipse fit was possible at the expected phase ("
                + detail
                + "). This light curve cannot constrain a secondary-eclipse depth.",
            )
        depth = float(fit["depth"])
        significance = float(fit["significance"] or 0.0)
        strongest_control = self._strongest_control_significance(controls)
        controls_are_quieter = strongest_control is None or strongest_control < max(3.0, significance - 1.5)
        if depth > 0 and significance >= 5.0 and controls_are_quieter:
            return (
                "candidate",
                "Candidate signal · independent check required",
                (
                    f"A positive fixed-phase eclipse depth is recovered at {significance:.1f}σ. "
                    "Treat this as a candidate signal, not a confirmation."
                ),
            )
        if depth > 0 and significance >= 3.0:
            return (
                "marginal",
                "Marginal signal · not a detection",
                (
                    f"A positive depth is present at {significance:.1f}σ, below the conservative candidate threshold. "
                    "More eclipse coverage is needed."
                ),
            )
        return (
            "inconclusive",
            "Inconclusive",
            "No reliable positive secondary-eclipse depth was recovered from this approved light curve.",
        )

    @staticmethod
    def _observed_event_count(
        times: np.ndarray,
        phase: np.ndarray,
        mid_time: float,
        period: float,
        expected_phase: float,
        local_mask: np.ndarray,
    ) -> int:
        if not np.any(local_mask):
            return 0
        cycles = (times[local_mask] - mid_time) / period - expected_phase
        return int(np.unique(np.rint(cycles).astype(int)).size)

    @staticmethod
    def _as_ppm(value: float | None) -> float | None:
        return None if value is None or not math.isfinite(float(value)) else float(value) * 1_000_000.0

    @staticmethod
    def _write_outputs(
        destination: Path,
        phase: np.ndarray,
        times_bjd: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        fit: dict[str, Any],
        summary: dict[str, Any],
        *,
        duration_phase: float,
        window_phase: float,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        local = fit["local_mask"]
        local_phase = phase[local]
        local_times = times_bjd[local]
        local_flux = flux[local]
        local_uncertainty = uncertainty[local]
        local_model = np.asarray(fit["model"], dtype=float)
        local_residuals = np.asarray(fit["residuals"], dtype=float)
        local_template = np.asarray(fit["template"], dtype=float)
        table = np.column_stack(
            (
                local_times,
                local_phase,
                local_flux,
                local_uncertainty,
                local_model,
                local_residuals,
                local_template,
            )
        )
        time_label = "BJD_TDB" if summary.get("time_standard") == "BJD_TDB" else "JD_TDB"
        np.savetxt(
            destination / "secondary-eclipse.csv",
            table,
            delimiter=",",
            header=(
                f"{time_label},phase_from_expected,relative_flux,relative_flux_uncertainty,"
                "model,residual,eclipse_template"
            ),
            comments="",
        )
        (destination / "secondary-eclipse.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_secondary_eclipse_preview(
            phase,
            flux,
            uncertainty,
            fit,
            destination / "secondary-eclipse.png",
            duration_phase=duration_phase,
            window_phase=window_phase,
            summary=summary,
        )


def _write_secondary_eclipse_preview(
    phase: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    fit: dict[str, Any],
    destination: Path,
    *,
    duration_phase: float,
    window_phase: float,
    summary: dict[str, Any],
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination.parent.mkdir(parents=True, exist_ok=True)
    available = bool(fit["coverage"]["available"])
    figure = Figure(
        figsize=(11.4, 8.1 if available else 5.2),
        facecolor="#0b2638",
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    if available:
        grid = figure.add_gridspec(4, 1, height_ratios=(0.72, 3.25, 1.15, 0.08))
        header = figure.add_subplot(grid[0])
        axis = figure.add_subplot(grid[1])
        residual_axis = figure.add_subplot(grid[2], sharex=axis)
        header.set_axis_off()
        local = fit["local_mask"]
        local_phase = phase[local]
        local_flux = flux[local]
        local_uncertainty = uncertainty[local]
        model = np.asarray(fit["model"], dtype=float)
        baseline_model = np.asarray(fit["baseline_model"], dtype=float)
        residuals = np.asarray(fit["residuals"], dtype=float)
        order = np.argsort(local_phase)
        period_days = _secondary_preview_period_days(summary)
        hours_per_phase = period_days * 24.0
        local_hours = local_phase * hours_per_phase
        window_hours = window_phase * hours_per_phase
        duration_hours = duration_phase * hours_per_phase
        phase_bins = _secondary_phase_bins(
            local_phase,
            local_flux,
            local_uncertainty,
            residuals,
            window_phase=window_phase,
        )

        grid_phase = np.linspace(-window_phase, window_phase, 800)
        grid_baseline = np.interp(grid_phase, local_phase[order], baseline_model[order])
        grid_model = np.interp(grid_phase, local_phase[order], model[order])
        grid_hours = grid_phase * hours_per_phase

        logo_path = Path(__file__).resolve().parent / "assets" / "leaps-mark.png"
        if logo_path.exists():
            from matplotlib.image import imread

            logo_axis = header.inset_axes([0.0, 0.08, 0.06, 0.84])
            logo_axis.imshow(imread(logo_path))
            logo_axis.set_axis_off()
        header.text(
            0.069,
            0.62,
            "LEAPS",
            transform=header.transAxes,
            ha="left",
            va="center",
            color="#f4f8fb",
            fontsize=16,
            fontweight="bold",
        )
        header.text(
            0.069,
            0.34,
            "Secondary Eclipse Analysis",
            transform=header.transAxes,
            ha="left",
            va="center",
            color="#20c5f4",
            fontsize=7.5,
        )
        header.text(
            0.5,
            0.53,
            str(summary.get("planet", "Secondary eclipse")),
            transform=header.transAxes,
            ha="center",
            va="center",
            color="#f4f8fb",
            fontsize=24,
            fontweight="bold",
        )
        header.text(
            1.0,
            0.53,
            (
                f"Expected phase: {float(summary.get('expected_phase', 0.5)):.3f}\n"
                f"Duration: {duration_hours:.2f} h  ·  {int(summary.get('event_count', 0))} windows"
            ),
            transform=header.transAxes,
            ha="right",
            va="center",
            color="#dce9f3",
            fontsize=9.5,
            linespacing=1.3,
        )

        axis.scatter(
            local_hours,
            (local_flux - 1.0) * 1_000_000.0,
            s=3.0,
            color="#6c8fa9",
            alpha=0.14,
            linewidths=0,
            rasterized=True,
            label="Approved photometry",
            zorder=1,
        )
        axis.errorbar(
            phase_bins["phase"] * hours_per_phase,
            (phase_bins["flux"] - 1.0) * 1_000_000.0,
            yerr=phase_bins["uncertainty"] * 1_000_000.0,
            fmt="o",
            color="#f4f8fb",
            ecolor="#94aec2",
            markeredgecolor="#081925",
            markersize=4.0,
            elinewidth=0.8,
            capsize=1.6,
            label=f"Phase bins ({phase_bins['bin_minutes']:.1f} min)",
            zorder=4,
        )
        axis.plot(
            grid_hours,
            (grid_baseline - 1.0) * 1_000_000.0,
            color="#20c5f4",
            lw=2.1,
            label="No-eclipse baseline",
            zorder=5,
        )
        axis.plot(
            grid_hours,
            (grid_model - 1.0) * 1_000_000.0,
            color="#ff5b62",
            lw=2.8,
            label="Fixed-phase eclipse fit",
            zorder=6,
        )
        axis.axhline(0.0, color="#5d7b90", lw=0.8, alpha=0.55, zorder=0)
        axis.axvline(0.0, color="#5d7b90", lw=0.8, ls="--", alpha=0.55, zorder=0)
        axis.axvspan(-duration_hours / 2.0, duration_hours / 2.0, color="#20c5f4", alpha=0.10)

        depth = summary.get("depth_ppm")
        depth_uncertainty = summary.get("depth_uncertainty_ppm")
        significance = summary.get("significance")
        beta = summary.get("red_noise_beta")
        if depth is not None and depth_uncertainty is not None and significance is not None:
            beta_detail = f"  ·  red-noise β={float(beta):.2f}" if beta is not None else ""
            detail = (
                f"{summary.get('outcome_label', 'Secondary eclipse')}\n"
                f"Depth {float(depth):.0f} ± {float(depth_uncertainty):.0f} ppm  ·  S/N {float(significance):.1f}"
                f"{beta_detail}"
            )
            axis.text(
                0.015,
                0.975,
                detail,
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#dce9f3",
                fontsize=9.4,
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.38",
                    "facecolor": "#0b2638",
                    "edgecolor": "#28516b",
                    "alpha": 0.92,
                },
                zorder=7,
            )

        residual_axis.scatter(
            local_hours,
            residuals * 1_000_000.0,
            s=3.0,
            color="#6c8fa9",
            alpha=0.14,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        residual_axis.errorbar(
            phase_bins["phase"] * hours_per_phase,
            phase_bins["residual"] * 1_000_000.0,
            yerr=phase_bins["residual_uncertainty"] * 1_000_000.0,
            fmt="o",
            color="#f4f8fb",
            ecolor="#94aec2",
            markeredgecolor="#081925",
            markersize=4.0,
            elinewidth=0.8,
            capsize=1.6,
            zorder=4,
        )
        residual_axis.axhline(0.0, color="#ff5b62", lw=1.6)
        residual_axis.axvline(0.0, color="#5d7b90", lw=0.8, ls="--", alpha=0.55)
        residual_axis.axvspan(-duration_hours / 2.0, duration_hours / 2.0, color="#20c5f4", alpha=0.10)
        axis.set_ylabel("Relative flux (ppm)")
        residual_axis.set_ylabel("Residual\n(ppm)")
        residual_axis.set_xlabel("Hours from expected secondary-eclipse center")

        _secondary_preview_limits(
            axis,
            residual_axis,
            phase_bins,
            grid_model,
            grid_baseline,
        )
        title = str(summary.get("outcome_label", "Secondary eclipse"))
    else:
        axis = figure.subplots(1, 1)
        axis.plot(phase, flux, "o", color="#52718a", markersize=2.8, alpha=0.6)
        axis.text(
            0.5,
            0.5,
            "No approved data cover the expected secondary-eclipse window.\n"
            "This observation cannot constrain an occultation depth.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#dce9f3",
            fontsize=12,
        )
        axis.set_xlabel("Phase from expected secondary eclipse")
        axis.set_ylabel("Relative flux")
        axis.set_xlim(-0.5, 0.5)
        title = "Secondary eclipse · insufficient coverage"
    plot_axes = [axis] if not available else [axis, residual_axis]
    for current_axis in plot_axes:
        current_axis.set_facecolor("#071827")
        current_axis.tick_params(colors="#a9bdd0", labelsize=8)
        current_axis.grid(color="#28516b", alpha=0.35)
        for spine in current_axis.spines.values():
            spine.set_color("#28516b")
        if not available:
            current_axis.axvspan(-duration_phase / 2.0, duration_phase / 2.0, color="#20c5f4", alpha=0.10)
            current_axis.set_xlim(-0.5, 0.5)
        else:
            current_axis.set_xlim(-window_hours, window_hours)
        current_axis.xaxis.label.set_color("#dce9f3")
        current_axis.yaxis.label.set_color("#dce9f3")
    if not available:
        axis.set_title(title, color="#ffffff", loc="left", fontsize=14, fontweight="bold")
    if available:
        axis.legend(
            loc="lower left",
            facecolor="#0b2638",
            edgecolor="#28516b",
            labelcolor="#dce9f3",
            fontsize=8.1,
            ncol=2,
        )
    figure.savefig(destination, dpi=160, facecolor=figure.get_facecolor())
    figure.savefig(destination.with_suffix(".pdf"), facecolor=figure.get_facecolor())


def _secondary_preview_period_days(summary: dict[str, Any]) -> float:
    """Read the fitted period defensively so diagnostic plotting never blocks outputs."""
    try:
        period = float(summary.get("parameters", {}).get("period"))
        if math.isfinite(period) and period > 0:
            return period
    except (TypeError, ValueError):
        pass
    return 1.0


def _secondary_phase_bins(
    phase: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    residual: np.ndarray,
    *,
    window_phase: float,
) -> dict[str, np.ndarray | float]:
    """Create uncertainty-aware phase bins for a readable eclipse diagnostic.

    The fit itself always uses every approved point.  Binning is display-only,
    with a scatter-based error floor so correlated photometry is not made to
    look more precise than it is.
    """
    valid = (
        np.isfinite(phase)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & np.isfinite(residual)
        & (uncertainty > 0)
    )
    phase, flux, uncertainty, residual = (
        np.asarray(values, dtype=float)[valid]
        for values in (phase, flux, uncertainty, residual)
    )
    point_count = phase.size
    if point_count == 0:
        empty = np.asarray([], dtype=float)
        return {
            "phase": empty,
            "flux": empty,
            "uncertainty": empty,
            "residual": empty,
            "residual_uncertainty": empty,
            "bin_minutes": 0.0,
        }

    bin_count = int(np.clip(round(math.sqrt(point_count) * 2.0), 8, 72))
    bin_count = min(bin_count, max(2, point_count // 3))
    edges = np.linspace(-window_phase, window_phase, bin_count + 1)
    indices = np.clip(np.digitize(phase, edges) - 1, 0, bin_count - 1)
    phase_values: list[float] = []
    flux_values: list[float] = []
    uncertainty_values: list[float] = []
    residual_values: list[float] = []
    residual_uncertainty_values: list[float] = []
    for index in range(bin_count):
        points = indices == index
        count = int(points.sum())
        if count < 2:
            continue
        weights = 1.0 / np.square(uncertainty[points])
        weight_sum = float(weights.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            continue
        formal = math.sqrt(1.0 / weight_sum)
        flux_scatter = float(np.std(flux[points], ddof=1)) / math.sqrt(count)
        residual_scatter = float(np.std(residual[points], ddof=1)) / math.sqrt(count)
        phase_values.append(float(np.average(phase[points], weights=weights)))
        flux_values.append(float(np.average(flux[points], weights=weights)))
        uncertainty_values.append(max(formal, flux_scatter))
        residual_values.append(float(np.average(residual[points], weights=weights)))
        residual_uncertainty_values.append(max(formal, residual_scatter))
    return {
        "phase": np.asarray(phase_values),
        "flux": np.asarray(flux_values),
        "uncertainty": np.asarray(uncertainty_values),
        "residual": np.asarray(residual_values),
        "residual_uncertainty": np.asarray(residual_uncertainty_values),
        "bin_minutes": 2.0 * window_phase / max(bin_count, 1) * 24.0 * 60.0,
    }


def _secondary_preview_limits(
    axis: Any,
    residual_axis: Any,
    bins: dict[str, np.ndarray | float],
    model: np.ndarray,
    baseline: np.ndarray,
) -> None:
    """Focus the display on the binned eclipse signal instead of raw-point scatter."""
    binned_flux = np.asarray(bins["flux"], dtype=float)
    binned_error = np.asarray(bins["uncertainty"], dtype=float)
    binned_residual = np.asarray(bins["residual"], dtype=float)
    binned_residual_error = np.asarray(bins["residual_uncertainty"], dtype=float)
    model_ppm = (np.asarray(model, dtype=float) - 1.0) * 1_000_000.0
    baseline_ppm = (np.asarray(baseline, dtype=float) - 1.0) * 1_000_000.0
    signal_values = np.concatenate(
        (
            (binned_flux - 1.0) * 1_000_000.0 - binned_error * 1_000_000.0,
            (binned_flux - 1.0) * 1_000_000.0 + binned_error * 1_000_000.0,
            model_ppm,
            baseline_ppm,
        )
    )
    finite_signal = signal_values[np.isfinite(signal_values)]
    if finite_signal.size:
        center = float(np.median(finite_signal))
        half_span = max(20.0, float(np.max(np.abs(finite_signal - center))) * 1.15)
        axis.set_ylim(center - half_span, center + half_span)

    residual_values = np.concatenate(
        (
            binned_residual * 1_000_000.0 - binned_residual_error * 1_000_000.0,
            binned_residual * 1_000_000.0 + binned_residual_error * 1_000_000.0,
        )
    )
    finite_residual = residual_values[np.isfinite(residual_values)]
    if finite_residual.size:
        half_span = max(20.0, float(np.max(np.abs(finite_residual))) * 1.15)
        residual_axis.set_ylim(-half_span, half_span)


def _write_fit_preview(
    observation: dict[str, Any],
    predicted_model: Any,
    destination: Path,
    *,
    catalog_parameters: PlanetParameters,
    observation_times_jd: Any,
    exposure_time: float,
    filter_name: str,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    series = observation["detrended_series"]
    time = np.asarray(series["time"])
    flux = np.asarray(series["flux"])
    uncertainty = np.asarray(series["flux_unc"])
    model = np.asarray(series["model"])
    residuals = np.asarray(series["residuals"])
    predicted = np.asarray(predicted_model, dtype=float)
    if predicted.ndim != 1 or predicted.shape != time.shape:
        raise ValueError(
            "The predicted transit must contain one value for every detrended timestamp"
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("The predicted transit contains non-finite values")

    best_mid_time = _fit_parameter_result(observation, "mid_time")
    best_rp_over_rs = _fit_parameter_result(observation, "rp_over_rs")
    expected_mid_time = _expected_transit_mid_time(
        observation,
        time,
        catalog_parameters.mid_time,
        catalog_parameters.period,
    )
    best_fit_label = (
        "Best-fit transit\n"
        rf"$T_{{\mathrm{{mid}}}}={_parameter_math(best_mid_time)}$, "
        rf"$R_{{\mathrm{{p}}}}/R_\star={_parameter_math(best_rp_over_rs)}$"
    )
    timing_offset_minutes = (
        float(best_mid_time["value"]) - expected_mid_time
    ) * 24.0 * 60.0
    timing_minus = _optional_finite_float(best_mid_time.get("m_error"))
    timing_plus = _optional_finite_float(best_mid_time.get("p_error"))
    if timing_minus is not None:
        timing_minus *= 24.0 * 60.0
    if timing_plus is not None:
        timing_plus *= 24.0 * 60.0
    timing_math = _measurement_math(
        timing_offset_minutes,
        timing_minus,
        timing_plus,
    )
    predicted_label = (
        "Predicted transit\n"
        rf"$T_{{\mathrm{{mid}}}}={expected_mid_time:.8f}$, "
        rf"$R_{{\mathrm{{p}}}}/R_\star={catalog_parameters.rp_over_rs:.5f}$, "
        rf"$O\! -\! C={timing_math}\ \mathrm{{min}}$"
    )
    observation_header = _fit_preview_header(
        observation_times_jd,
        exposure_time,
        filter_name,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = float(np.floor(np.min(time)))

    figure = Figure(figsize=(10, 7), facecolor="#0b2638", constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(0.72, 3, 1, 0.12))
    header = figure.add_subplot(grid[0])
    curve = figure.add_subplot(grid[1])
    residual = figure.add_subplot(grid[2], sharex=curve)
    header.set_axis_off()
    logo_path = Path(__file__).resolve().parent / "assets" / "leaps-mark.png"
    if logo_path.exists():
        from matplotlib.image import imread

        logo_axis = header.inset_axes([0.0, 0.08, 0.075, 0.84])
        logo_axis.imshow(imread(logo_path))
        logo_axis.set_axis_off()
    header.text(
        0.083,
        0.62,
        "LEAPS",
        transform=header.transAxes,
        ha="left",
        va="center",
        color="#f4f8fb",
        fontsize=16,
        fontweight="bold",
    )
    header.text(
        0.083,
        0.34,
        "Exoplanet Transit Analysis",
        transform=header.transAxes,
        ha="left",
        va="center",
        color="#20c5f4",
        fontsize=7.5,
    )
    header.text(
        0.5,
        0.53,
        catalog_parameters.name,
        transform=header.transAxes,
        ha="center",
        va="center",
        color="#f4f8fb",
        fontsize=24,
        fontweight="bold",
    )
    header.text(
        1.0,
        0.53,
        observation_header,
        transform=header.transAxes,
        ha="right",
        va="center",
        color="#dce9f3",
        fontsize=9.5,
        linespacing=1.25,
    )
    for axis in (curve, residual):
        axis.set_facecolor("#071827")
        axis.tick_params(colors="#a9bdd0")
        for spine in axis.spines.values():
            spine.set_color("#28516b")
        axis.grid(color="#28516b", alpha=0.35)
    curve.errorbar(
        time - offset,
        flux,
        yerr=uncertainty,
        fmt="o",
        color="#c4d5e4",
        ecolor="#52758e",
        markersize=2.5,
        linewidth=0.5,
        label="Observed flux",
    )
    curve.plot(
        time - offset,
        model,
        color="#20c5f4",
        linewidth=1.7,
        label=best_fit_label,
        zorder=3,
    )
    curve.plot(
        time - offset,
        predicted,
        color="#ff624c",
        linewidth=1.7,
        label=predicted_label,
        zorder=2,
    )
    curve.set_ylabel("Relative flux", color="#dce9f3")
    curve.legend(
        loc="lower left",
        facecolor="#0b2638",
        edgecolor="#28516b",
        labelcolor="#dce9f3",
        fontsize=8.2,
        handlelength=2.8,
        labelspacing=0.9,
    )
    residual.axhline(0, color="#20c5f4", linewidth=1)
    residual.plot(time - offset, residuals, "o", color="#c4d5e4", markersize=2.5)
    residual.set_ylabel("Residual", color="#dce9f3")
    residual.set_xlabel(f"BJD − {offset:.0f}", color="#dce9f3")
    figure.savefig(destination, dpi=240, facecolor=figure.get_facecolor())


def _fit_parameter_result(observation: dict[str, Any], name: str) -> dict[str, Any]:
    parameter = observation.get("parameters", {}).get(name)
    if not isinstance(parameter, dict):
        raise ValueError(f"The fitting result does not contain {name}")
    value = _optional_finite_float(parameter.get("value"))
    if value is None:
        raise ValueError(f"The fitted {name} value is not finite")
    return parameter


def _fit_preview_header(
    observation_times_jd: Any,
    exposure_time: float,
    filter_name: str,
) -> str:
    from astropy.time import Time

    times = np.asarray(observation_times_jd, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("The observation times for the fit header are missing or invalid")
    if not math.isfinite(exposure_time) or exposure_time <= 0:
        raise ValueError("The exposure time for the fit header must be positive and finite")
    start_jd = float(np.min(times))
    end_jd = float(np.max(times)) + exposure_time / 86400.0
    start = Time(start_jd, format="jd", scale="utc").to_datetime()
    duration_hours = (end_jd - start_jd) * 24.0
    return (
        f"{start:%Y-%m-%d %H:%M} (UT)\n"
        f"Dur: {duration_hours:.1f}h / Exp: {exposure_time:.1f}s\n"
        f"Filter: {passband_label(filter_name)}"
    )


def _expected_transit_mid_time(
    observation: dict[str, Any],
    time: np.ndarray,
    catalog_mid_time: float,
    catalog_period: float,
) -> float:
    if not math.isfinite(catalog_mid_time):
        raise ValueError("The catalog mid-transit time is not finite")
    if not math.isfinite(catalog_period) or catalog_period <= 0:
        raise ValueError("The catalog orbital period must be positive and finite")
    epoch_value = observation.get("model_info", {}).get("epoch")
    try:
        epoch = int(epoch_value)
    except (TypeError, ValueError, OverflowError):
        epoch = int(round((float(np.mean(time)) - catalog_mid_time) / catalog_period))
    expected = catalog_mid_time + epoch * catalog_period
    if not math.isfinite(expected):
        raise ValueError("The predicted mid-transit time is not finite")
    return expected


def _parameter_math(parameter: dict[str, Any]) -> str:
    value = float(parameter["value"])
    minus = _optional_finite_float(parameter.get("m_error"))
    plus = _optional_finite_float(parameter.get("p_error"))
    value_text = str(parameter.get("print_value", value))
    if minus is None or plus is None:
        return value_text
    minus_text = str(parameter.get("print_m_error", minus))
    plus_text = str(parameter.get("print_p_error", plus))
    return rf"{value_text}^{{+{plus_text}}}_{{-{minus_text}}}"


def _measurement_math(
    value: float,
    minus: float | None,
    plus: float | None,
) -> str:
    if minus is None or plus is None or minus <= 0 or plus <= 0:
        return f"{value:.2f}"
    smallest_error = min(abs(minus), abs(plus))
    exponent = math.floor(math.log10(smallest_error))
    leading_digit = int(smallest_error / (10.0**exponent))
    significant_digits = 2 if leading_digit in (1, 2) else 1
    decimals = max(0, min(8, -exponent + significant_digits - 1))
    return (
        f"{value:.{decimals}f}"
        rf"^{{+{plus:.{decimals}f}}}_{{-{minus:.{decimals}f}}}"
    )


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _combine(arrays: list[np.ndarray], method: str) -> np.ndarray:
    if not arrays:
        raise ValueError("At least one array is required")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise LEAPSError(
            "CALIBRATION_SHAPE_MISMATCH",
            "Calibration frames have different sizes",
            "All calibration frames must match the science frame dimensions.",
            ["Review frame assignments", "Exclude the mismatched frame"],
            stage=StageID.REDUCTION,
        )
    stack = np.stack([np.asarray(array, dtype=np.float32) for array in arrays])
    return np.nanmean(stack, axis=0) if method == "mean" else np.nanmedian(stack, axis=0)


def _julian_date(header: Any, config: ReductionConfig) -> float:
    from astropy.time import Time

    date = str(header.get(config.date_key, ""))
    if "T" not in date and header.get(config.time_key):
        date = f"{date}T{header[config.time_key]}"
    try:
        return float(Time(date, format="isot", scale="utc").jd)
    except Exception:
        return float(header.get("JD", header.get("MJD-OBS", 0.0)))
