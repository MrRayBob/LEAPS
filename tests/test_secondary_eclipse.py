from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.time import Time

from leaps.catalog import PlanetParameters
from leaps.models import ProjectManifest, StageID, StageStatus
from leaps.project import ProjectWorkspace
from leaps.science import SecondaryEclipseService, _secondary_phase_bins


def _parameters(mid_time: float) -> PlanetParameters:
    return PlanetParameters(
        name="Synthetic b",
        ra="12:00:00",
        dec="+20:00:00",
        period=3.0,
        mid_time=mid_time,
        rp_over_rs=0.12,
        sma_over_rs=8.5,
        inclination=87.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.0,
        temperature=5700.0,
        logg=4.4,
        source="Synthetic catalog",
    )


def _write_curve(project: ProjectWorkspace, time_utc: np.ndarray, flux: np.ndarray, error: float) -> None:
    output = project.outputs_dir / StageID.LIGHT_CURVE.value
    output.mkdir(parents=True)
    curve = np.column_stack((time_utc, flux, np.full(time_utc.size, error)))
    np.savetxt(output / "light_curve_aperture.txt", curve)
    np.savetxt(output / "light_curve_gauss.txt", curve)


def test_secondary_eclipse_recovers_fixed_phase_signal_and_exports_diagnostics(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Synthetic eclipse")
    primary_mid_time = Time(2461000.0, format="jd", scale="utc").tdb.jd
    parameters = _parameters(primary_mid_time)
    eclipse_center_utc = Time(
        primary_mid_time + parameters.period / 2.0, format="jd", scale="tdb"
    ).utc.jd
    time_utc = eclipse_center_utc + np.linspace(-4.0, 4.0, 260) / 24.0
    time_tdb = Time(time_utc, format="jd", scale="utc").tdb.jd
    phase = SecondaryEclipseService._relative_phase(
        time_tdb, parameters.mid_time, parameters.period, 0.5
    )
    template = SecondaryEclipseService._eclipse_template(phase, 2.0 / (parameters.period * 24.0))
    rng = np.random.default_rng(15)
    flux = 1.0 - 0.0010 * template + rng.normal(0.0, 0.00015, time_utc.size)
    _write_curve(project, time_utc, flux, 0.00015)

    result = SecondaryEclipseService().run(
        project,
        parameters,
        duration_hours=2.0,
        baseline="linear",
    )

    assert result.outcome == "candidate"
    assert result.depth_ppm is not None and 700.0 < result.depth_ppm < 1300.0
    assert result.significance is not None and result.significance > 5.0
    assert result.preview_path.exists()
    assert (result.output_path / "secondary-eclipse.pdf").exists()
    assert (result.output_path / "secondary-eclipse.csv").exists()
    assert (result.output_path / "secondary-eclipse.json").exists()


def test_secondary_eclipse_reports_no_coverage_without_inventing_depth(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "No eclipse coverage")
    primary_mid_time = Time(2461000.0, format="jd", scale="utc").tdb.jd
    parameters = _parameters(primary_mid_time)
    primary_center_utc = Time(primary_mid_time, format="jd", scale="tdb").utc.jd
    time_utc = primary_center_utc + np.linspace(-3.0, 3.0, 80) / 24.0
    _write_curve(project, time_utc, np.ones(time_utc.size), 0.0003)

    result = SecondaryEclipseService().run(project, parameters, duration_hours=2.0)

    assert result.outcome == "inconclusive"
    assert result.depth_ppm is None
    assert result.local_points == 0
    assert result.preview_path.exists()
    assert "cannot constrain" in result.message


def test_secondary_preview_uses_display_bins_without_changing_the_fit_inputs() -> None:
    phase = np.linspace(-0.10, 0.10, 12_000)
    flux = 1.0 + 4e-5 * phase
    uncertainty = np.full(phase.size, 3e-4)
    residual = flux - 1.0

    binned = _secondary_phase_bins(
        phase,
        flux,
        uncertainty,
        residual,
        window_phase=0.10,
    )

    assert 2 <= len(binned["phase"]) <= 72
    assert np.all(np.asarray(binned["uncertainty"]) > 0)
    assert np.all(np.asarray(binned["residual_uncertainty"]) > 0)
    assert float(binned["bin_minutes"]) > 0


def test_full_fit_unlocks_secondary_eclipse_stage(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Workflow")

    project.set_stage(StageID.FITTING, StageStatus.COMPLETE, "Complete")

    state = project.manifest.stages[StageID.SECONDARY_ECLIPSE.value]
    assert state.status == StageStatus.READY
    assert state.summary == "Ready"


def test_existing_completed_fit_gains_ready_secondary_eclipse_stage() -> None:
    payload = ProjectManifest().to_dict()
    payload["stages"].pop(StageID.SECONDARY_ECLIPSE.value)
    payload["stages"][StageID.FITTING.value]["status"] = "complete"
    payload["stages"][StageID.FITTING.value]["summary"] = "Complete"

    migrated = ProjectManifest.from_dict(payload)

    assert migrated.stages[StageID.SECONDARY_ECLIPSE.value].status == StageStatus.READY
