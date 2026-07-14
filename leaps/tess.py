"""Import calibrated TESS SPOC light curves into a portable LEAPS project.

This intentionally accepts the mission light-curve products (``*_lc.fits``)
rather than target-pixel files or full-frame images.  Those files already
contain the calibrated PDCSAP photometry required for transit fitting; LEAPS
keeps the originals read-only and creates its normal approved-light-curve
outputs beside them.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import LEAPSError, StageID, StageState, StageStatus
from .project import ProjectWorkspace


@dataclass(slots=True)
class TessImportResult:
    """The project and provenance produced by a successful TESS import."""

    project: ProjectWorkspace
    source_files: list[Path]
    sectors: list[int]
    tic_id: str
    imported_points: int
    rejected_points: int
    cadence_seconds: float
    output_path: Path


@dataclass(slots=True)
class _TessFile:
    path: Path
    tic_id: str
    object_name: str
    ra_deg: float
    dec_deg: float
    sector: int | None
    time_bjd_tdb: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    rejected_points: int
    cadence_seconds: float


class TessImportService:
    """Convert downloaded SPOC PDCSAP light curves to LEAPS input files."""

    PRODUCT_DESCRIPTION = "TESS SPOC PDCSAP light-curve import"

    def run(self, paths: Iterable[str | Path], *, emit=None, token=None) -> TessImportResult:
        """Create a new project beside the selected TESS light-curve files.

        The source FITS files are only read.  The generated project root is a
        sibling named ``TESS-TIC-<id>`` under the common selected-data folder.
        """
        if token is not None:
            token.raise_if_cancelled()
        source_files = self._normalise_paths(paths)
        records = []
        for path in source_files:
            if token is not None:
                token.raise_if_cancelled()
            records.append(self._read_file(path))
        first = records[0]
        self._validate_same_target(records)

        time_bjd_tdb, flux, uncertainty, rejected_points, cadence_seconds = self._combine(records)
        root = self._project_root(source_files, first)
        if root.exists():
            raise LEAPSError(
                "TESS_PROJECT_EXISTS",
                "A TESS LEAPS project already exists",
                f"LEAPS did not overwrite the existing project at {root.name}.",
                ["Use Open project to resume it", "Rename or move the existing project to import again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(root),
            )

        try:
            project = ProjectWorkspace.create(root, f"TESS TIC {first.tic_id}")
            ra, dec = self._coordinates(first.ra_deg, first.dec_deg)
            project.manifest.target_name = self._target_name(first)
            project.manifest.target_ra = ra
            project.manifest.target_dec = dec
            project.manifest.settings.update(
                {
                    "filter": "TESS",
                    "exposure_time": cadence_seconds,
                    "observation_metadata": {
                        "science_frames_inspected": 0,
                        "filter": "TESS",
                        "filter_status": "tess",
                        "exposure_time": cadence_seconds,
                        "source": "Imported TESS PDCSAP light curve",
                    },
                    "tess_import": {
                        "product": self.PRODUCT_DESCRIPTION,
                        "source_files": [str(path) for path in source_files],
                        "tic_id": first.tic_id,
                        "sectors": sorted({record.sector for record in records if record.sector is not None}),
                        "source_time_standard": "BTJD / BJD_TDB",
                        "stored_time_standard": "JD_UTC (converted from TESS BJD_TDB)",
                        "quality_filter": "QUALITY == 0; finite positive PDCSAP flux and uncertainty",
                        "imported_points": int(time_bjd_tdb.size),
                        "rejected_points": int(rejected_points),
                        "cadence_seconds": cadence_seconds,
                    },
                }
            )

            if token is not None:
                token.raise_if_cancelled()
            output_path = self._write_light_curve(project, time_bjd_tdb, flux, uncertainty)
            self._mark_imported_workflow(project, output_path)
            return TessImportResult(
                project=project,
                source_files=source_files,
                sectors=sorted({record.sector for record in records if record.sector is not None}),
                tic_id=first.tic_id,
                imported_points=int(time_bjd_tdb.size),
                rejected_points=int(rejected_points),
                cadence_seconds=cadence_seconds,
                output_path=output_path,
            )
        except LEAPSError:
            raise
        except Exception as exc:
            raise LEAPSError(
                "TESS_IMPORT_FAILED",
                "TESS light-curve import could not finish",
                "LEAPS left the selected TESS FITS files unchanged.",
                ["Check that the files are SPOC *_lc.fits products", "Try importing a smaller selection"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc

    @staticmethod
    def _normalise_paths(paths: Iterable[str | Path]) -> list[Path]:
        unique: dict[Path, None] = {}
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise LEAPSError(
                    "TESS_FILE_MISSING",
                    "A selected TESS file is unavailable",
                    "LEAPS could not read one of the selected light-curve files.",
                    ["Choose the downloaded *_lc.fits files again"],
                    stage=StageID.DATA_TARGET,
                    technical_details=str(path),
                )
            unique[path] = None
        result = sorted(unique)
        if not result:
            raise LEAPSError(
                "TESS_FILES_REQUIRED",
                "Choose TESS light-curve files first",
                "Select one or more calibrated TESS SPOC *_lc.fits files to import.",
                ["Choose Import TESS light curves", "Select downloaded *_lc.fits files"],
                stage=StageID.DATA_TARGET,
            )
        return result

    @staticmethod
    def _read_file(path: Path) -> _TessFile:
        try:
            from astropy.io import fits

            with fits.open(path, memmap=False) as hdul:
                primary = hdul[0].header
                light_curve_hdu = next(
                    (
                        hdu
                        for hdu in hdul[1:]
                        if getattr(hdu, "data", None) is not None
                        and getattr(getattr(hdu, "columns", None), "names", None)
                        and "TIME" in hdu.columns.names
                    ),
                    None,
                )
                if light_curve_hdu is None:
                    raise ValueError("No binary light-curve table containing TIME was found")
                names = {name.upper() for name in light_curve_hdu.columns.names}
                required = {"TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR"}
                if not required <= names:
                    raise ValueError("PDCSAP_FLUX and PDCSAP_FLUX_ERR columns are required")
                data = light_curve_hdu.data
                header = light_curve_hdu.header
                time = np.asarray(data["TIME"], dtype=float)
                flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
                uncertainty = np.asarray(data["PDCSAP_FLUX_ERR"], dtype=float)
                quality = (
                    np.asarray(data["QUALITY"], dtype=np.int64)
                    if "QUALITY" in names
                    else np.zeros(time.size, dtype=np.int64)
                )
                reference = float(header.get("BJDREFI", primary.get("BJDREFI", 2457000.0)))
                reference += float(header.get("BJDREFF", primary.get("BJDREFF", 0.0)))
                tic_value = primary.get("TICID", header.get("TICID", ""))
                tic_id = str(tic_value).strip()
                object_name = str(primary.get("OBJECT", header.get("OBJECT", ""))).strip()
                ra_deg = float(primary.get("RA_OBJ", header.get("RA_OBJ")))
                dec_deg = float(primary.get("DEC_OBJ", header.get("DEC_OBJ")))
                sector_value = primary.get("SECTOR", header.get("SECTOR"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise LEAPSError(
                "TESS_LIGHT_CURVE_INVALID",
                "That file is not a supported TESS light curve",
                "Choose calibrated SPOC *_lc.fits files with TIME, PDCSAP_FLUX, and PDCSAP_FLUX_ERR columns. "
                "Target-pixel files and full-frame images need a different extraction workflow.",
                ["Choose downloaded *_lc.fits files", "Check the TESS product type"],
                stage=StageID.DATA_TARGET,
                technical_details=f"{path}\n{exc}",
            ) from exc

        if not tic_id:
            tic_id = object_name.removeprefix("TIC ").strip() or "unknown"
        valid = (
            np.isfinite(time)
            & np.isfinite(flux)
            & np.isfinite(uncertainty)
            & (flux > 0)
            & (uncertainty > 0)
            & (quality == 0)
        )
        if valid.sum() < 10:
            raise LEAPSError(
                "TESS_LIGHT_CURVE_EMPTY",
                "The selected TESS light curve has too little usable data",
                "After mission quality flags and invalid values were removed, fewer than ten photometric points remained.",
                ["Choose another sector", "Check that you selected a calibrated *_lc.fits file"],
                stage=StageID.DATA_TARGET,
                technical_details=str(path),
            )
        time, flux, uncertainty = time[valid] + reference, flux[valid], uncertainty[valid]
        scale = float(np.nanmedian(flux))
        if not np.isfinite(scale) or scale <= 0:
            raise LEAPSError(
                "TESS_LIGHT_CURVE_SCALE_INVALID",
                "The TESS light curve could not be normalized",
                "The calibrated flux values do not have a positive finite median.",
                ["Choose another TESS light-curve product"],
                stage=StageID.DATA_TARGET,
                technical_details=str(path),
            )
        order = np.argsort(time)
        time, flux, uncertainty = time[order], flux[order] / scale, uncertainty[order] / scale
        cadence = float(np.nanmedian(np.diff(time)) * 86400.0) if time.size > 1 else 0.0
        if not np.isfinite(cadence) or cadence <= 0:
            cadence = 120.0
        return _TessFile(
            path=path,
            tic_id=tic_id,
            object_name=object_name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            sector=int(sector_value) if sector_value is not None else None,
            time_bjd_tdb=time,
            flux=flux,
            uncertainty=uncertainty,
            rejected_points=int((~valid).sum()),
            cadence_seconds=cadence,
        )

    @staticmethod
    def _validate_same_target(records: list[_TessFile]) -> None:
        first = records[0]
        mismatched_tics = [record for record in records[1:] if record.tic_id != first.tic_id]
        if mismatched_tics:
            identifiers = ", ".join(sorted({record.tic_id for record in records}))
            raise LEAPSError(
                "TESS_TARGET_MISMATCH",
                "The selected TESS files are from different targets",
                f"Import one target at a time. The selected files contain TIC IDs: {identifiers}.",
                ["Choose sectors for one target only"],
                stage=StageID.DATA_TARGET,
            )
        for record in records[1:]:
            if abs(record.ra_deg - first.ra_deg) > 1e-4 or abs(record.dec_deg - first.dec_deg) > 1e-4:
                raise LEAPSError(
                    "TESS_TARGET_COORDINATES_MISMATCH",
                    "The selected TESS files disagree on target coordinates",
                    "Choose sectors for the same target only.",
                    ["Choose the matching TESS light-curve files again"],
                    stage=StageID.DATA_TARGET,
                )

    @staticmethod
    def _combine(records: list[_TessFile]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        time = np.concatenate([record.time_bjd_tdb for record in records])
        flux = np.concatenate([record.flux for record in records])
        uncertainty = np.concatenate([record.uncertainty for record in records])
        order = np.argsort(time)
        time, flux, uncertainty = time[order], flux[order], uncertainty[order]
        _, indices = np.unique(time, return_index=True)
        time, flux, uncertainty = time[indices], flux[indices], uncertainty[indices]
        if time.size < 10:
            raise LEAPSError(
                "TESS_IMPORT_TOO_SHORT",
                "Too little unique TESS data remains",
                "LEAPS needs at least ten non-duplicate, quality-filtered points to create a light curve.",
                ["Select more TESS sectors"],
                stage=StageID.DATA_TARGET,
            )
        rejected = sum(record.rejected_points for record in records)
        cadence = float(np.nanmedian([record.cadence_seconds for record in records]))
        return time, flux, uncertainty, rejected, cadence

    @staticmethod
    def _project_root(paths: list[Path], record: _TessFile) -> Path:
        common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
        identifier = record.tic_id.replace(" ", "-") or "target"
        return common_parent / f"TESS-TIC-{identifier}"

    @staticmethod
    def _target_name(record: _TessFile) -> str:
        return f"TIC {record.tic_id}" if record.tic_id and record.tic_id != "unknown" else record.object_name

    @staticmethod
    def _coordinates(ra_deg: float, dec_deg: float) -> tuple[str, str]:
        import astropy.units as units
        from astropy.coordinates import SkyCoord

        coordinate = SkyCoord(ra=ra_deg * units.deg, dec=dec_deg * units.deg, frame="icrs")
        return (
            coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2, pad=True),
            coordinate.dec.to_string(unit=units.deg, sep=":", precision=2, alwayssign=True, pad=True),
        )

    @staticmethod
    def _write_light_curve(
        project: ProjectWorkspace,
        time_bjd_tdb: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
    ) -> Path:
        from astropy.time import Time

        # LEAPS stores its fitting inputs as UTC JD.  Converting the mission's
        # BTJD/BJD_TDB values to UTC here preserves the physical instant; the
        # fitting and eclipse services convert back to TDB for phase work.
        time_utc_jd = Time(time_bjd_tdb, format="jd", scale="tdb").utc.jd
        curve = np.column_stack((time_utc_jd, flux, uncertainty))
        pending, target = project.begin_transaction(StageID.LIGHT_CURVE)
        try:
            header = "JD_UTC relative_flux relative_flux_uncertainty\nImported TESS SPOC PDCSAP light curve"
            for filename in (
                "light_curve_aperture.txt",
                "light_curve_gauss.txt",
                "PHOTOMETRY_APERTURE.txt",
                "PHOTOMETRY_GAUSS.txt",
            ):
                np.savetxt(pending / filename, curve, header=header)
            (pending / "tess-import.json").write_text(
                '{\n  "product": "TESS SPOC PDCSAP light curve"\n}\n', encoding="utf-8"
            )
            project.commit_transaction(pending, target)
        except Exception:
            project.discard_pending_transaction(StageID.LIGHT_CURVE)
            raise
        return target / "light_curve_aperture.txt"

    @staticmethod
    def _mark_imported_workflow(project: ProjectWorkspace, output_path: Path) -> None:
        labels = {
            StageID.DATA_TARGET: "TESS target and metadata imported",
            StageID.REDUCTION: "Not applicable · calibrated SPOC PDCSAP product",
            StageID.INSPECTION: "Mission quality flags applied during import",
            StageID.ALIGNMENT: "Not applicable · extracted TESS light curve",
            StageID.PHOTOMETRY: "Not applicable · mission PDCSAP photometry",
            StageID.LIGHT_CURVE: "Imported TESS PDCSAP light curve approved",
        }
        for stage, summary in labels.items():
            project.manifest.stages[stage.value] = StageState(
                status=StageStatus.COMPLETE,
                summary=summary,
                progress=1.0,
                output_path=project.relative(output_path) if stage == StageID.LIGHT_CURVE else None,
            )
        project.manifest.stages[StageID.FITTING.value] = StageState(
            status=StageStatus.READY,
            summary="Ready · choose a planet and run a primary-transit fit",
        )
        project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState()
        project.save()
