from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .filters import normalize_filter
from .models import LEAPSError, StageID

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fz"}
PROJECT_WORKSPACE_NAMES = {"LEAPS", ".leaps"}


def is_generated_project_path(path: Path) -> bool:
    return any(
        part in PROJECT_WORKSPACE_NAMES or part.startswith(".LEAPS-reset-")
        for part in path.parts
    )


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
            if isinstance(ra, (int, float)) or ":" not in str(ra):
                import astropy.units as units
                from astropy.coordinates import SkyCoord

                coordinate = SkyCoord(float(ra), float(dec), unit=(units.deg, units.deg))
                return (
                    name,
                    coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2),
                    coordinate.dec.to_string(unit=units.deg, sep=":", precision=2, alwayssign=True),
                )
            normalized_ra, normalized_dec = validate_coordinates(str(ra).strip(), str(dec).strip())
            return name, normalized_ra, normalized_dec
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


class FITSInventory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def discover(self) -> list[FrameRecord]:
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in FITS_EXTENSIONS
            and not is_generated_project_path(path)
            and not any(part.startswith("reduction") or part.startswith("photometry") for part in path.parts)
        )
        return [self.inspect(path) for path in paths]

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
                header = dict(hdu.header)
                shape = tuple(hdu.shape) if hdu.shape else None
                bitpix = int(hdu.header.get("BITPIX", 0)) or None
                for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
                    if key in hdu.header:
                        exposure = float(hdu.header[key])
                        break
                raw_filter = next(
                    (
                        hdu.header.get(key)
                        for key in ("FILTER", "FILT", "FILTER1", "FILTER2")
                        if hdu.header.get(key) not in (None, "")
                    ),
                    "",
                )
                filter_name = normalize_filter(raw_filter) or str(raw_filter).strip()
        except Exception:
            pass
        category, confidence, reason = classify_frame(path, header)
        target_name, target_ra, target_dec = target_from_header(header)
        return FrameRecord(
            path=path.relative_to(self.root).as_posix(),
            category=category,
            confidence=confidence,
            reason=reason,
            shape=shape,
            bitpix=bitpix,
            exposure=exposure,
            checksum=_fingerprint(path),
            target_name=target_name,
            target_ra=target_ra,
            target_dec=target_dec,
            filter_name=filter_name,
        )

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
    if len(filters) == 1:
        filter_name = next(iter(filters))
        filter_status = "detected"
    elif len(filters) > 1:
        filter_name = None
        filter_status = "mixed"
    else:
        filter_name = None
        filter_status = "unknown"
    return {
        "filter": filter_name,
        "filter_status": filter_status,
        "filters_detected": sorted(filters),
        "exposure_time": float(median(exposures)) if exposures else None,
        "science_frames_inspected": len(science),
        "source": "science_fits",
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
