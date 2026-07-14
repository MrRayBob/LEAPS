from __future__ import annotations

import errno
import hashlib
import math
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .filters import normalize_filter
from .models import LEAPSError, StageID

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fz"}
PROJECT_WORKSPACE_NAMES = {"LEAPS", ".leaps"}


def normalized_fits_header(header: Any) -> Any:
    """Return a standards-compliant copy of an Astropy FITS header.

    Some camera software writes recoverable cards such as unquoted sexagesimal
    RA/DEC strings.  Astropy can read the image but refuses to write a copied
    header until those cards are normalized.  Repairing the copied cards keeps
    the source FITS file byte-for-byte unchanged.
    """
    normalized = header.copy()
    for card in normalized.cards:
        card.verify("silentfix")
    return normalized


def is_fits_path(path: Path) -> bool:
    """Return whether *path* uses a supported FITS filename extension."""
    # External drives formatted as exFAT/FAT commonly store macOS resource
    # forks as AppleDouble sidecars such as ``._bias_001.fits``.  They inherit
    # the data file's suffix but are metadata containers, not FITS images.
    return not path.name.startswith("._") and path.suffix.casefold() in FITS_EXTENSIONS


def is_generated_project_path(path: Path) -> bool:
    return any(
        part in PROJECT_WORKSPACE_NAMES or part.startswith(".LEAPS-reset-")
        for part in path.parts
    )


def preflight_observing_run_access(root: str | Path) -> None:
    """Trigger folder access and verify that at least one FITS file can be opened.

    On macOS, touching the user-selected folder is what allows the operating
    system to present its Files and Folders consent prompt. Non-permission I/O
    errors are left to the full inventory scan, which can describe damaged or
    otherwise unreadable FITS files more accurately.
    """
    pending = [Path(root).expanduser()]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if (
                            entry.name not in PROJECT_WORKSPACE_NAMES
                            and not entry.name.startswith(".LEAPS-reset-")
                        ):
                            pending.append(path)
                    elif entry.is_file(follow_symlinks=False) and is_fits_path(path):
                        with path.open("rb") as handle:
                            handle.read(1)
                        return
        except OSError as exc:
            if _is_access_error(exc):
                raise _folder_access_failure(current, exc) from exc
            return


@dataclass(slots=True)
class FrameRecord:
    path: str
    category: str
    confidence: float
    reason: str
    shape: tuple[int, ...] | None
    bitpix: int | None
    exposure: float | None
    checksum: str
    target_name: str = ""
    target_ra: str = ""
    target_dec: str = ""
    filter_name: str = ""
    observatory: str = ""
    observatory_latitude: float | None = None
    observatory_longitude: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_coordinates(ra: str, dec: str) -> tuple[str, str]:
    try:
        import astropy.units as units
        from astropy.coordinates import SkyCoord

        coordinate = SkyCoord(ra, dec, unit=(units.hourangle, units.deg), frame="icrs")
        return coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2), coordinate.dec.to_string(
            unit=units.deg, sep=":", precision=2, alwayssign=True
        )
    except Exception as exc:
        raise LEAPSError(
            "INVALID_COORDINATES",
            "Target coordinates are not valid",
            "Enter right ascension as hh:mm:ss and declination as +dd:mm:ss.",
            ["Correct the coordinates", "Use coordinates detected in the FITS header"],
            stage=StageID.DATA_TARGET,
            technical_details=str(exc),
        ) from exc


def target_from_header(header: dict[str, object]) -> tuple[str, str, str]:
    """Extract a normalized target name and ICRS coordinates from common FITS keywords."""
    name = next(
        (
            str(header.get(key, "")).strip()
            for key in ("OBJECT", "OBJNAME", "TARGET", "TARGNAME")
            if str(header.get(key, "")).strip()
        ),
        "",
    )
    ra = next(
        (header.get(key) for key in ("OBJCTRA", "RA", "TELRA") if header.get(key) not in (None, "")),
        None,
    )
    dec = next(
        (header.get(key) for key in ("OBJCTDEC", "DEC", "TELDEC") if header.get(key) not in (None, "")),
        None,
    )
    try:
        if ra is not None and dec is not None:
            ra_text = str(ra).strip()
            dec_text = str(dec).strip()
            try:
                ra_degrees = float(ra_text)
                dec_degrees = float(dec_text)
            except ValueError:
                normalized_ra, normalized_dec = validate_coordinates(
                    ":".join(ra_text.split()),
                    ":".join(dec_text.split()),
                )
                return name, normalized_ra, normalized_dec
            else:
                import astropy.units as units
                from astropy.coordinates import SkyCoord

                coordinate = SkyCoord(
                    ra_degrees, dec_degrees, unit=(units.deg, units.deg)
                )
                return (
                    name,
                    coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2),
                    coordinate.dec.to_string(unit=units.deg, sep=":", precision=2, alwayssign=True),
                )
        if header.get("CRVAL1") is not None and header.get("CRVAL2") is not None:
            import astropy.units as units
            from astropy.coordinates import SkyCoord

            coordinate = SkyCoord(
                float(header["CRVAL1"]), float(header["CRVAL2"]), unit=(units.deg, units.deg)
            )
            normalized_ra = coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2)
            normalized_dec = coordinate.dec.to_string(unit=units.deg, sep=":", precision=2, alwayssign=True)
            return name, normalized_ra, normalized_dec
    except (TypeError, ValueError, LEAPSError):
        pass
    return name, "", ""


def observatory_from_header(
    header: dict[str, object],
) -> tuple[str, float | None, float | None]:
    """Extract an observing-site label and east-positive coordinates."""
    name = next(
        (
            str(header.get(key, "")).strip()
            for key in (
                "OBSERVAT",
                "OBSERVATORY",
                "SITENAME",
                "SITE",
                "TELESCOP",
            )
            if str(header.get(key, "")).strip()
        ),
        "",
    )
    latitude = _header_angle(
        header,
        ("SITELAT", "LATITUDE", "OBS-LAT", "OBSLAT", "LAT-OBS", "GEOLAT", "OBSGEO-B"),
    )
    longitude = _header_angle(
        header,
        (
            "SITELONG",
            "SITELON",
            "LONGITUD",
            "LONGITUDE",
            "OBS-LONG",
            "OBSLONG",
            "LON-OBS",
            "GEOLON",
            "OBSGEO-L",
        ),
    )
    if latitude is not None and not -90 <= latitude <= 90:
        latitude = None
    if longitude is not None:
        longitude = (longitude + 180.0) % 360.0 - 180.0
    return name, latitude, longitude


def _header_angle(
    header: dict[str, object], keys: tuple[str, ...]
) -> float | None:
    for key in keys:
        value = header.get(key)
        if value in (None, ""):
            continue
        try:
            angle = float(value)
        except (TypeError, ValueError):
            try:
                import astropy.units as units
                from astropy.coordinates import Angle

                angle = float(Angle(str(value).strip(), unit=units.deg).deg)
            except (TypeError, ValueError):
                continue
        if math.isfinite(angle):
            return angle
    return None


class FITSInventory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def discover(self) -> list[FrameRecord]:
        try:
            paths = sorted(
                path
                for path in self.root.rglob("*")
                if path.is_file()
                and is_fits_path(path)
                and not is_generated_project_path(path.relative_to(self.root))
                and not any(
                    part.startswith("reduction") or part.startswith("photometry")
                    for part in path.relative_to(self.root).parts
                )
            )
        except OSError as exc:
            raise _folder_access_failure(self.root, exc) from exc
        if not paths:
            raise LEAPSError(
                "NO_FITS_FILES_FOUND",
                "No FITS images were found",
                f"LEAPS could not find .fits, .fit, or .fts images inside {self.root}.",
                [
                    "Choose the folder containing the observing run",
                    "On macOS, allow LEAPS under System Settings > Privacy & Security > Files and Folders",
                    "Confirm the files are stored locally and readable",
                ],
                stage=StageID.DATA_TARGET,
                technical_details=str(self.root),
            )
        records = [self.inspect(path) for path in paths]
        if not any(record.shape is not None for record in records):
            raise LEAPSError(
                "FITS_HEADERS_UNREADABLE",
                "The FITS images could not be read",
                "Files were found, but LEAPS could not read an image or header from any of them.",
                [
                    "Choose the folder again to grant access",
                    "On macOS, allow LEAPS under System Settings > Privacy & Security > Files and Folders",
                    "Verify the files open in another FITS viewer",
                ],
                stage=StageID.DATA_TARGET,
                technical_details="\n".join(str(path) for path in paths[:20]),
            )
        return records

    def inspect(self, path: Path) -> FrameRecord:
        header: dict[str, object] = {}
        shape: tuple[int, ...] | None = None
        bitpix: int | None = None
        exposure: float | None = None
        filter_name = ""
        try:
            from astropy.io import fits

            with fits.open(path, memmap=True, do_not_scale_image_data=True, ignore_missing_end=True) as hdus:
                hdu = next(
                    (candidate for candidate in hdus if getattr(candidate, "data", None) is not None), hdus[0]
                )
                safe_header = normalized_fits_header(hdu.header)
                header = dict(safe_header)
                shape = tuple(hdu.shape) if hdu.shape else None
                bitpix = int(safe_header.get("BITPIX", 0)) or None
                for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
                    if key in safe_header:
                        exposure = float(safe_header[key])
                        break
                raw_filter = next(
                    (
                        safe_header.get(key)
                        for key in ("FILTER", "FILT", "FILTER1", "FILTER2")
                        if safe_header.get(key) not in (None, "")
                    ),
                    "",
                )
                filter_name = normalize_filter(raw_filter) or str(raw_filter).strip()
        except OSError as exc:
            if _is_access_error(exc):
                raise _folder_access_failure(path, exc) from exc
        except Exception:
            pass
        category, confidence, reason = classify_frame(path, header)
        target_name, target_ra, target_dec = target_from_header(header)
        observatory, observatory_latitude, observatory_longitude = observatory_from_header(
            header
        )
        return FrameRecord(
            path=path.relative_to(self.root).as_posix(),
            category=category,
            confidence=confidence,
            reason=reason,
            shape=shape,
            bitpix=bitpix,
            exposure=exposure,
            checksum=self._fingerprint_or_failure(path),
            target_name=target_name,
            target_ra=target_ra,
            target_dec=target_dec,
            filter_name=filter_name,
            observatory=observatory,
            observatory_latitude=observatory_latitude,
            observatory_longitude=observatory_longitude,
        )

    @staticmethod
    def _fingerprint_or_failure(path: Path) -> str:
        try:
            return _fingerprint(path)
        except OSError as exc:
            raise _folder_access_failure(path, exc) from exc

    @staticmethod
    def group(records: Iterable[FrameRecord]) -> dict[str, list[str]]:
        grouped = {key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")}
        for record in records:
            grouped.setdefault(record.category, []).append(record.path)
        return grouped


def summarize_observation_records(
    records: Iterable[FrameRecord], science_paths: Iterable[str] | None = None
) -> dict[str, Any]:
    """Summarize the assigned science passband and exposure without reading pixels."""
    selected = set(science_paths or ())
    science = [
        record
        for record in records
        if (record.path in selected if selected else record.category == "science")
    ]
    filters = {
        canonical
        for record in science
        if (canonical := normalize_filter(record.filter_name)) is not None
    }
    exposures = [record.exposure for record in science if record.exposure and record.exposure > 0]
    observatory_names = [record.observatory for record in science if record.observatory]
    locations = [
        (record.observatory_latitude, record.observatory_longitude)
        for record in science
        if record.observatory_latitude is not None
        and record.observatory_longitude is not None
    ]
    if len(filters) == 1:
        filter_name = next(iter(filters))
        filter_status = "detected"
    elif len(filters) > 1:
        filter_name = None
        filter_status = "mixed"
    else:
        filter_name = None
        filter_status = "unknown"
    observatory = Counter(observatory_names).most_common(1)[0][0] if observatory_names else ""
    latitude: float | None = None
    longitude: float | None = None
    if locations:
        latitudes = [float(location[0]) for location in locations]
        longitudes = [float(location[1]) for location in locations]
        if max(latitudes) - min(latitudes) <= 0.01 and max(longitudes) - min(longitudes) <= 0.01:
            latitude = float(median(latitudes))
            longitude = float(median(longitudes))
            location_status = "detected"
        else:
            location_status = "mixed"
    elif any(
        record.observatory_latitude is not None
        or record.observatory_longitude is not None
        for record in science
    ):
        location_status = "partial"
    else:
        location_status = "unknown"
    return {
        "filter": filter_name,
        "filter_status": filter_status,
        "filters_detected": sorted(filters),
        "exposure_time": float(median(exposures)) if exposures else None,
        "science_frames_inspected": len(science),
        "source": "science_fits",
        "observatory": observatory,
        "latitude": latitude,
        "longitude": longitude,
        "location_status": location_status,
        "location_source": "science_fits" if observatory or locations else "unknown",
    }


def classify_frame(path: Path, header: dict[str, object]) -> tuple[str, float, str]:
    image_type = " ".join(
        str(header.get(key, "")) for key in ("IMAGETYP", "IMAGETYPE", "IMTYPE", "FRAME", "OBSTYPE", "OBJECT")
    ).lower()
    filename = path.stem.lower().replace("-", "_")
    combined = f"{image_type} {filename}"
    if any(token in combined for token in ("dark_flat", "darkflat", "flat_dark")):
        return (
            "dark_flat",
            0.98 if "dark" in image_type else 0.86,
            "Header or filename identifies a dark-flat",
        )
    if "bias" in combined or "zero" in image_type:
        return "bias", 0.98 if image_type else 0.82, "Header or filename identifies a bias"
    if "dark" in combined:
        return "dark", 0.98 if image_type else 0.82, "Header or filename identifies a dark"
    if "flat" in combined:
        return "flat", 0.98 if image_type else 0.82, "Header or filename identifies a flat"
    if any(token in image_type for token in ("light", "science", "object")):
        return "science", 0.98, "FITS header identifies a science exposure"
    if any(token in filename for token in ("light", "science", "object", "autosave")):
        return "science", 0.76, "Filename resembles a science exposure"
    return "unknown", 0.0, "No reliable frame type was found"


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(64 * 1024))
    digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def _is_access_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or exc.errno in {errno.EACCES, errno.EPERM}


def _folder_access_failure(path: Path, exc: OSError) -> LEAPSError:
    return LEAPSError(
        "OBSERVING_RUN_ACCESS_DENIED" if _is_access_error(exc) else "OBSERVING_RUN_UNREADABLE",
        "LEAPS cannot access the observing run",
        f"The selected folder or FITS file is not readable: {path}",
        [
            "Choose the folder again to grant access",
            "On macOS, allow LEAPS under System Settings > Privacy & Security > Files and Folders",
            "Confirm your account has read and write permission for the observing-run folder",
        ],
        stage=StageID.DATA_TARGET,
        technical_details=f"{type(exc).__name__}: {exc}",
    )


def observing_run_access_failure(path: str | Path, exc: OSError) -> LEAPSError:
    """Build the typed observing-run error used by later processing stages."""
    return _folder_access_failure(Path(path), exc)
