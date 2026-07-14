from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from leaps.catalog import PlanetParameters
from leaps.filters import normalize_filter
from leaps.models import LEAPSError, StageID, StageStatus
from leaps.project import ProjectWorkspace
from leaps.science import FittingService, SecondaryEclipseService
from leaps.tess import TessImportService


def _write_tess_light_curve(
    path: Path,
    *,
    sector: int,
    tic_id: int = 100100827,
    ra_deg: float = 24.35430629,
    dec_deg: float = -45.67788237,
    time_offset: float = 0.0,
) -> None:
    time = np.arange(16, dtype=float) / 720.0 + 2000.0 + time_offset
    flux = np.linspace(1000.0, 1015.0, time.size)
    uncertainty = np.full(time.size, 4.0)
    quality = np.zeros(time.size, dtype=np.int32)
    quality[3] = 1
    columns = fits.ColDefs(
        [
            fits.Column(name="TIME", format="D", array=time),
            fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
            fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=uncertainty),
            fits.Column(name="QUALITY", format="J", array=quality),
        ]
    )
    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = f"TIC {tic_id}"
    primary.header["TICID"] = tic_id
    primary.header["RA_OBJ"] = ra_deg
    primary.header["DEC_OBJ"] = dec_deg
    primary.header["SECTOR"] = sector
    light_curve = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")
    light_curve.header["BJDREFI"] = 2457000
    light_curve.header["BJDREFF"] = 0.0
    fits.HDUList([primary, light_curve]).writeto(path)


def test_tess_import_creates_approved_light_curve_and_preserves_sources(tmp_path: Path) -> None:
    first = tmp_path / "sector-01_lc.fits"
    second = tmp_path / "sector-02_lc.fits"
    _write_tess_light_curve(first, sector=1)
    _write_tess_light_curve(second, sector=2, time_offset=30.0)
    checksums = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)}

    result = TessImportService().run([second, first])

    assert result.project.root == tmp_path / "TESS-TIC-100100827"
    assert result.tic_id == "100100827"
    assert result.sectors == [1, 2]
    assert result.imported_points == 30
    assert result.rejected_points == 2
    assert result.cadence_seconds == pytest.approx(120.0)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == checksum for path, checksum in checksums.items())

    curve = np.loadtxt(result.output_path)
    assert curve.shape == (30, 3)
    assert np.all(np.diff(curve[:, 0]) > 0)
    assert np.median(curve[:15, 1]) == pytest.approx(1.0, rel=0.01)
    assert Time(curve[0, 0], format="jd", scale="utc").tdb.jd == pytest.approx(2459000.0)

    project = ProjectWorkspace.open(result.project.root)
    assert project.manifest.target_name == "TIC 100100827"
    assert project.manifest.target_ra == "01:37:25.03"
    assert project.manifest.target_dec == "-45:40:40.38"
    assert project.manifest.settings["filter"] == "TESS"
    assert project.manifest.settings["tess_import"]["sectors"] == [1, 2]
    assert project.manifest.stages[StageID.LIGHT_CURVE.value].status == StageStatus.COMPLETE
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert project.manifest.stages[StageID.SECONDARY_ECLIPSE.value].status == StageStatus.LOCKED


def test_tess_import_rejects_mixed_targets(tmp_path: Path) -> None:
    first = tmp_path / "sector-01_lc.fits"
    second = tmp_path / "sector-02_lc.fits"
    _write_tess_light_curve(first, sector=1, tic_id=100)
    _write_tess_light_curve(second, sector=2, tic_id=200)

    with pytest.raises(LEAPSError, match="one target at a time") as failure:
        TessImportService().run([first, second])

    assert failure.value.code == "TESS_TARGET_MISMATCH"
    assert not (tmp_path / "TESS-TIC-100").exists()


def test_tess_import_rejects_non_light_curve_fits(tmp_path: Path) -> None:
    source = tmp_path / "tess_target_pixel.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="TIME", format="D", array=np.arange(12, dtype=float))]
            ),
        ]
    ).writeto(source)

    with pytest.raises(LEAPSError, match="PDCSAP_FLUX") as failure:
        TessImportService().run([source])

    assert failure.value.code == "TESS_LIGHT_CURVE_INVALID"


def test_tess_passband_is_available_to_fitting() -> None:
    assert normalize_filter("TESS") == "TESS"


def test_tess_primary_fit_phase_folds_many_epochs_before_hops() -> None:
    time_utc = 2459000.0 + np.arange(0, 24.0, 2.0 / (24.0 * 60.0))
    time_tdb = Time(time_utc, format="jd", scale="utc").tdb.jd
    period = 1.8
    mid_time = float(time_tdb[180])
    phase = (time_tdb - mid_time + period / 2.0) % period - period / 2.0
    flux = np.ones(time_utc.size)
    flux[np.abs(phase) < 0.05] -= 0.012
    uncertainty = np.full(time_utc.size, 0.0015)
    parameters = PlanetParameters(
        name="Synthetic b",
        ra="01:37:25.03",
        dec="-45:40:40.38",
        period=period,
        mid_time=mid_time,
        rp_over_rs=0.11,
        sma_over_rs=6.0,
        inclination=88.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.0,
        temperature=6000.0,
        logg=4.2,
        source="Test",
    )

    folded, refined, metadata = FittingService._prepare_tess_phase_folded_curve(
        np.asarray((time_utc, flux, uncertainty)), parameters
    )

    assert folded.shape[0] == 3
    assert 20 <= folded.shape[1] <= 360
    assert np.all(np.diff(folded[0]) > 0)
    assert refined.period == pytest.approx(period, rel=0.002)
    assert metadata["source_points"] == time_utc.size
    assert metadata["phase_bins"] == folded.shape[1]


def test_tess_secondary_eclipse_uses_mission_barycentric_timing(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path / "tess-project", "Synthetic TESS")
    project.manifest.settings["tess_import"] = {"product": "test"}
    project.save()
    output = project.outputs_dir / StageID.LIGHT_CURVE.value
    output.mkdir()
    time = 2460000.0 + np.arange(1600) * 2.0 / (24.0 * 60.0)
    flux = 1.0 + 0.00015 * np.sin(np.arange(time.size) / 17.0)
    uncertainty = np.full(time.size, 0.0005)
    np.savetxt(output / "light_curve_aperture.txt", np.column_stack((time, flux, uncertainty)))
    parameters = PlanetParameters(
        name="Synthetic b",
        ra="01:37:25.03",
        dec="-45:40:40.38",
        period=0.5,
        mid_time=2460000.1,
        rp_over_rs=0.1,
        sma_over_rs=5.0,
        inclination=88.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.0,
        temperature=6000.0,
        logg=4.2,
        source="Test",
    )

    result = SecondaryEclipseService().run(
        project,
        parameters,
        duration_hours=2.0,
        baseline="linear",
    )

    assert result.time_standard == "TESS BJD_TDB (mission-corrected)"
    assert "Set observatory coordinates" not in result.message
