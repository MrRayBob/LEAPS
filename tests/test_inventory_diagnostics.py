from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from leaps.diagnostics import DiagnosticLogger
from leaps.fits_inventory import (
    FITSInventory,
    is_fits_path,
    observatory_from_header,
    preflight_observing_run_access,
    summarize_observation_records,
    target_from_header,
    validate_coordinates,
)
from leaps.models import LEAPSError, StageID
from leaps.project import ProjectWorkspace


def _write_fits(
    path: Path, image_type: str, exposure: float = 30.0, filter_name: str = ""
) -> None:
    header = fits.Header()
    header["IMAGETYP"] = image_type
    header["EXPTIME"] = exposure
    header["OBSERVER"] = "Private Name"
    if filter_name:
        header["FILTER"] = filter_name
    fits.writeto(path, np.arange(64, dtype=np.uint16).reshape(8, 8), header=header)


def _write_fits_with_unquoted_coordinates(path: Path) -> None:
    header = fits.Header(
        {
            "IMAGETYP": "Light Frame",
            "EXPTIME": 30.0,
            "RA": "23 54 40.53",
            "DEC": "-37 37 41.61",
        }
    )
    fits.writeto(path, np.arange(64, dtype=np.uint16).reshape(8, 8), header=header)
    contents = bytearray(path.read_bytes())
    for keyword, value, comment in (
        ("RA", "23 54 40.53", "Right Ascension"),
        ("DEC", "-37 37 41.61", "Declination"),
    ):
        marker = f"{keyword:<8}=".encode("ascii")
        offset = contents.index(marker)
        replacement = f"{keyword:<8}= {value:<20} / {comment}".ljust(80)
        contents[offset : offset + 80] = replacement.encode("ascii")
    path.write_bytes(contents)


def test_inventory_reads_headers_only_and_groups_frames(tmp_path: Path) -> None:
    _write_fits(tmp_path / "light_001.fits", "Light Frame")
    _write_fits(tmp_path / "master_dark.fit", "Dark Frame")
    _write_fits(tmp_path / "flat_001.fts", "Flat Field")
    records = FITSInventory(tmp_path).discover()
    grouped = FITSInventory.group(records)
    assert len(grouped["science"]) == 1
    assert len(grouped["dark"]) == 1
    assert len(grouped["flat"]) == 1
    assert all(record.shape == (8, 8) for record in records)


def test_inventory_repairs_unquoted_coordinate_cards_in_memory(tmp_path: Path) -> None:
    frame = tmp_path / "LTT-9779_001.fits"
    _write_fits_with_unquoted_coordinates(frame)
    before = hashlib.sha256(frame.read_bytes()).hexdigest()

    record = FITSInventory(tmp_path).discover()[0]

    assert record.shape == (8, 8)
    assert record.target_ra.startswith("23:54:40.53")
    assert record.target_dec.startswith("-37:37:41.61")
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("filename", ["image.fits", "image.fit", "image.fts", "IMAGE.FIT"])
def test_supported_fits_filename_extensions(filename: str) -> None:
    assert is_fits_path(Path(filename))


@pytest.mark.parametrize("filename", ["._bias_001.fits", "._LIGHT.FIT", "._flat.fts"])
def test_macos_appledouble_sidecars_are_not_fits_images(filename: str) -> None:
    assert not is_fits_path(Path(filename))


def test_inventory_ignores_macos_appledouble_sidecars_on_external_drives(
    tmp_path: Path,
) -> None:
    _write_fits(tmp_path / "bias_001.fits", "Bias Frame", 0.0)
    (tmp_path / "._bias_001.fits").write_bytes(b"\x00\x05\x16\x07macOS metadata")

    records = FITSInventory(tmp_path).discover()

    assert [record.path for record in records] == ["bias_001.fits"]


def test_inventory_reports_empty_folder_with_recovery(tmp_path: Path) -> None:
    with pytest.raises(LEAPSError) as error:
        FITSInventory(tmp_path).discover()
    assert error.value.code == "NO_FITS_FILES_FOUND"
    assert "Privacy & Security" in " ".join(error.value.recovery)


def test_inventory_reports_unreadable_headers_instead_of_silently_continuing(tmp_path: Path) -> None:
    (tmp_path / "broken.fits").write_bytes(b"not a FITS image")
    with pytest.raises(LEAPSError) as error:
        FITSInventory(tmp_path).discover()
    assert error.value.code == "FITS_HEADERS_UNREADABLE"


def test_selected_folder_preflight_opens_a_nested_fits_file_without_modifying_it(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "external-ssd" / "night-1"
    nested.mkdir(parents=True)
    frame = nested / "light_001.fits"
    _write_fits(frame, "Light Frame")
    before = frame.read_bytes()

    preflight_observing_run_access(tmp_path / "external-ssd")

    assert frame.read_bytes() == before


def test_selected_folder_preflight_reports_permission_denial(
    tmp_path: Path, monkeypatch
) -> None:
    def denied(_path):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("leaps.fits_inventory.os.scandir", denied)

    with pytest.raises(LEAPSError) as error:
        preflight_observing_run_access(tmp_path)

    assert error.value.code == "OBSERVING_RUN_ACCESS_DENIED"
    assert "Choose the folder again" in error.value.recovery[0]


def test_inventory_normalizes_hops_filter_and_summarizes_science_metadata(tmp_path: Path) -> None:
    _write_fits(tmp_path / "light_001.fits", "Light Frame", 30.0, "Cousins_R")
    _write_fits(tmp_path / "light_002.fits", "Light Frame", 32.0, "Rc")
    records = FITSInventory(tmp_path).discover()
    metadata = summarize_observation_records(records)

    assert {record.filter_name for record in records} == {"COUSINS_R"}
    assert metadata["filter"] == "COUSINS_R"
    assert metadata["filter_status"] == "detected"
    assert metadata["exposure_time"] == 31.0


def test_inventory_detects_observatory_and_coordinates_from_science_fits(
    tmp_path: Path,
) -> None:
    for name in ("light_001.fits", "light_002.fits"):
        path = tmp_path / name
        _write_fits(path, "Light Frame", 60.0, "Cousins_R")
        fits.setval(path, "TELESCOP", value="SARA-ORM")
        fits.setval(path, "SITELAT", value=28.76117)
        fits.setval(path, "SITELONG", value=-17.87808)

    records = FITSInventory(tmp_path).discover()
    metadata = summarize_observation_records(records)

    name, latitude, longitude = observatory_from_header(
        {"OBSERVAT": "Example", "OBS-LAT": "+28:45:40", "OBS-LONG": "-17:52:41"}
    )
    assert name == "Example"
    assert latitude == pytest.approx(28.761111, abs=1e-6)
    assert longitude == pytest.approx(-17.878056, abs=1e-6)
    assert {record.observatory for record in records} == {"SARA-ORM"}
    assert metadata["observatory"] == "SARA-ORM"
    assert metadata["latitude"] == pytest.approx(28.76117)
    assert metadata["longitude"] == pytest.approx(-17.87808)
    assert metadata["location_status"] == "detected"


def test_observation_metadata_requires_user_choice_for_mixed_filters(tmp_path: Path) -> None:
    _write_fits(tmp_path / "light_r.fits", "Light Frame", 30.0, "Cousins_R")
    _write_fits(tmp_path / "light_v.fits", "Light Frame", 30.0, "V")

    metadata = summarize_observation_records(FITSInventory(tmp_path).discover())

    assert metadata["filter"] is None
    assert metadata["filter_status"] == "mixed"
    assert metadata["filters_detected"] == ["COUSINS_R", "JOHNSON_V"]


def test_inventory_excludes_project_workspaces_and_interrupted_reset_data(tmp_path: Path) -> None:
    _write_fits(tmp_path / "light_001.fits", "Light Frame")
    for folder in ("LEAPS", ".leaps", ".LEAPS-reset-partial"):
        generated = tmp_path / folder / "outputs" / "reduction"
        generated.mkdir(parents=True)
        _write_fits(generated / "generated.fits", "Light Frame")

    records = FITSInventory(tmp_path).discover()

    assert [record.path for record in records] == ["light_001.fits"]


def test_parent_folder_named_leaps_does_not_hide_observing_run(tmp_path: Path) -> None:
    run = tmp_path / "LEAPS" / "NGTS-10"
    run.mkdir(parents=True)
    _write_fits(run / "science_001.fit", "Light Frame")

    records = FITSInventory(run).discover()

    assert [record.path for record in records] == ["science_001.fit"]


def test_coordinates_are_validated_without_requiring_a_name() -> None:
    ra, dec = validate_coordinates("19:34:55.87", "+36:48:55.79")
    assert ra.startswith("19:34:55")
    assert dec.startswith("+36:48:55")
    with pytest.raises(LEAPSError) as error:
        validate_coordinates("not-ra", "not-dec")
    assert error.value.code == "INVALID_COORDINATES"


def test_target_coordinates_are_normalized_from_common_fits_headers() -> None:
    name, ra, dec = target_from_header({"OBJECT": "TrES-3 b", "RA": "17:52:06.998", "DEC": "+37:32:46.195"})
    assert name == "TrES-3 b"
    assert ra.startswith("17:52:07")
    assert dec.startswith("+37:32:46")

    _, spaced_ra, spaced_dec = target_from_header(
        {"RA": "23 54 40.53", "DEC": "-37 37 41.61"}
    )
    assert spaced_ra.startswith("23:54:40.53")
    assert spaced_dec.startswith("-37:37:41.61")

    _, wcs_ra, wcs_dec = target_from_header({"CRVAL1": 268.029161, "CRVAL2": 37.546166})
    assert wcs_ra.startswith("17:52:07")
    assert wcs_dec.startswith("+37:32:46")


def test_redacted_diagnostics_contains_headers_but_never_pixels(tmp_path: Path) -> None:
    raw = tmp_path / "light.fits"
    _write_fits(raw, "Light Frame")
    project = ProjectWorkspace.create(tmp_path)
    logger = DiagnosticLogger(project)
    logger.record("test_event")
    output = logger.export_bundle(tmp_path / "diagnostics.zip", [raw])
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "project.json" in names
        assert "logs/leaps.jsonl" in names
        assert "headers/header-1.json" in names
        assert not any(name.endswith((".fits", ".fit", ".fts")) for name in names)
        header = json.loads(archive.read("headers/header-1.json"))
        assert "OBSERVER" not in header
        assert header["IMAGETYP"] == "Light Frame"


def test_typed_failure_is_logged_once_with_its_stage(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    logger = DiagnosticLogger(project)
    failure = LEAPSError(
        "CALIBRATION_FRAME_UNREADABLE",
        "A bias frame could not be read",
        "Review the calibration assignment.",
        ["Review frame assignments"],
        stage=StageID.REDUCTION,
    )

    assert logger.failure(failure, StageID.REDUCTION) is failure
    event = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["event"] == "failure"
    assert event["stage"] == "reduction"
    assert event["code"] == "CALIBRATION_FRAME_UNREADABLE"
    assert event["diagnostic_id"] == failure.diagnostic_id
