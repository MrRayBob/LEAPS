from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from leaps.catalog import PlanetCatalogResolver, PlanetParameters
from leaps.filters import normalize_filter, passband_label
from leaps.models import JobStatus, LEAPSError, StageEvent, StageID, StageStatus, target_fingerprint
from leaps.project import ProjectWorkspace
from leaps.science import (
    CancellationToken,
    FittingService,
    _normalize_mid_transit_time,
    _write_fit_preview,
)
from leaps.ui.main_window import MainWindow, _manual_planet_parameters
from leaps.ui.pages import FittingPage


def _parameters(name: str = "TrES-3b") -> PlanetParameters:
    return PlanetParameters(
        name=name,
        ra="17:52:07.0185",
        dec="+37:32:46.237",
        period=1.306186314,
        mid_time=2457657.754796,
        rp_over_rs=0.16309,
        sma_over_rs=6.0,
        inclination=82.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=-0.19,
        temperature=5650.0,
        logg=4.58,
        source="ExoClock",
    )


def _write_approved_curve(project: ProjectWorkspace) -> tuple[np.ndarray, np.ndarray]:
    output = project.outputs_dir / StageID.LIGHT_CURVE.value
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(2461000.40, 2461000.60, 21)
    flux = 1.0 - 0.025 * np.exp(-((times - 2461000.50) / 0.025) ** 4)
    np.savetxt(
        output / "light_curve_aperture.txt",
        np.column_stack((times, flux, np.full(times.size, 0.001))),
    )
    project.manifest.stages[StageID.LIGHT_CURVE.value].status = StageStatus.COMPLETE
    return times, flux


def test_hops_filter_aliases_normalize_fits_and_ui_names() -> None:
    assert normalize_filter("Cousins_R") == "COUSINS_R"
    assert normalize_filter("R") == "COUSINS_R"
    assert normalize_filter("Rc") == "COUSINS_R"
    assert normalize_filter("SDSS r'") == "sdss_r"
    assert normalize_filter("not-a-real-filter") is None
    assert passband_label("COUSINS_R") == "Cousins R"


@pytest.mark.parametrize(
    ("entered", "expected"),
    [
        (2459065.5097, 2459065.5097),
        (9065.5097, 2459065.5097),
        (59065.5097, 2459065.5097),
        (0.5097, 2459065.5097),
    ],
)
def test_mid_transit_shorthand_is_expanded_around_observation(
    entered: float, expected: float
) -> None:
    times = np.linspace(2459065.38, 2459065.53, 50)

    assert _normalize_mid_transit_time(entered, times) == pytest.approx(expected)


def test_catalog_candidates_prefer_the_requested_planet_at_project_coordinates(monkeypatch) -> None:
    planets = {
        "TrES-3b": {
            "name": "TrES-3b",
            "star": {
                "ra": "17:52:07.0185",
                "dec": "+37:32:46.237",
                "ra_deg": 268.02924375,
                "dec_deg": 37.54617694,
            },
            "planet": {
                "ephem_period": 1.3,
                "ephem_mid_time": 2457000.0,
                "rp_over_rs": 0.16,
                "sma_over_rs": 6.0,
                "inclination": 82.0,
                "eccentricity": 0.0,
                "periastron": 0.0,
                "meta": 0.0,
                "teff": 5600,
                "logg": 4.5,
            },
        },
        "TrES-3c": {
            "name": "TrES-3c",
            "star": {
                "ra": "17:52:07.0185",
                "dec": "+37:32:46.237",
                "ra_deg": 268.02924375,
                "dec_deg": 37.54617694,
            },
            "planet": {
                "ephem_period": 2.6,
                "ephem_mid_time": 2457001.0,
                "rp_over_rs": 0.1,
                "sma_over_rs": 8.0,
                "inclination": 85.0,
                "eccentricity": 0.0,
                "periastron": 0.0,
                "meta": 0.0,
                "teff": 5600,
                "logg": 4.5,
            },
        },
    }
    fake = SimpleNamespace(
        get_all_planets=lambda: list(planets),
        get_planet=lambda name: planets[name],
    )
    monkeypatch.setitem(sys.modules, "exoclock", fake)

    candidates = PlanetCatalogResolver().resolve_candidates(
        "17:52:06.99", "+37:32:46.15", "TrES-3c"
    )

    assert [candidate.name for candidate in candidates] == ["TrES-3c", "TrES-3b"]


def test_fitting_page_has_no_demo_target_and_requires_preview_before_full_fit(
    qapp, tmp_path, monkeypatch
) -> None:
    page = FittingPage()
    assert page.planet.currentText() == ""
    assert "WTS-2" not in page.planet.currentText()
    assert page.light_curve.currentData() == "gaussian"
    assert page.detrending.currentData() == "quadratic"
    assert [page.light_curve.itemData(index) for index in range(page.light_curve.count())] == [
        "aperture",
        "gaussian",
    ]
    assert [page.detrending.itemData(index) for index in range(page.detrending.count())] == [
        "airmass",
        "quadratic",
        "linear",
    ]

    page.set_planet_candidates([_parameters()])
    page.set_observation_metadata("Cousins_R", 30.0)
    page.set_observatory_metadata(
        "SARA-ORM", 28.76117, -17.87808, source="science FITS"
    )
    assert page.planet.currentText() == "TrES-3b"
    assert page.period.value() == pytest.approx(1.306186314)
    assert page.values()["filter"] == "COUSINS_R"
    assert page.values()["exposure_time"] == pytest.approx(30.0)
    assert page.exposure_time.value() == pytest.approx(30.0)
    assert page.values()["light_curve"] == "gaussian"
    assert page.values()["detrending"] == "quadratic"
    assert "SARA-ORM" in page.observatory.text()
    assert "science FITS" in page.observatory.text()
    assert page.values()["observatory_latitude"] == pytest.approx(28.76117)
    assert page.values()["observatory_longitude"] == pytest.approx(-17.87808)
    assert "walkers" not in page.values()
    assert not hasattr(page, "walkers")
    assert page.preview.isEnabled()
    assert page.preview.property("primary") is True
    assert not page.full.isEnabled()
    assert page.full.property("primary") is False
    assert not page.view_in_files.isEnabled()

    preview = tmp_path / "preview.png"
    pixmap = page.grab()
    assert pixmap.save(str(preview))
    monkeypatch.setattr(page, "_preview_device_pixel_ratio", lambda: 2.0)
    revealed = []
    page.viewInFilesRequested.connect(revealed.append)
    page.show_preview(
        preview,
        planet="TrES-3b",
        passband="COUSINS_R",
        residual_std=0.0028,
    )
    assert page.full.isEnabled()
    assert page.preview.property("primary") is False
    assert page.full.property("primary") is True
    assert page.view_in_files.isEnabled()
    assert page._rendered_preview_pixmap.devicePixelRatio() == 2.0
    page.view_in_files.click()
    assert revealed == [preview]
    page.period.setValue(1.4)
    assert not page.full.isEnabled()
    assert page.preview.property("primary") is True
    assert page.full.property("primary") is False
    page.close()


def test_fitting_page_requires_or_accepts_manual_exposure_time(qapp) -> None:
    page = FittingPage()
    page.set_planet_candidates([_parameters()])
    page.set_observation_metadata("COUSINS_R", None)

    assert page.exposure_time.value() == 0
    assert not page.preview.isEnabled()
    assert "enter it above" in page.observation_source.text()

    page.exposure_time.setValue(30.0)

    assert page.preview.isEnabled()
    assert page.values()["exposure_time"] == pytest.approx(30.0)
    assert page.values()["exposure_time_source"] == "manual override"
    page.close()


def test_fitting_page_shows_sampling_progress_and_stopping_state(qapp) -> None:
    page = FittingPage()
    page.set_busy(True, full=True)
    page.update_event(
        StageEvent(
            StageID.FITTING,
            JobStatus.RUNNING,
            "Sampling posterior",
            current=250,
            total=5000,
            checkpoint="sampling",
            details={
                "phase": "sampling",
                "walkers": 12,
                "elapsed_seconds": 65,
                "eta_seconds": 1200,
            },
        )
    )

    assert page.fit_progress.value() == 250
    assert "250 of 5,000" in page.fit_progress.format()
    assert "12 automatic HOPS walkers" in page.progress_details.text()
    assert "about 20m" in page.progress_details.text()

    page.set_stopping()
    assert page.cancel.text() == "Stopping…"
    assert not page.cancel.isEnabled()
    page.set_busy(False)
    assert page.cancel.text() == "Cancel"
    page.close()


def test_main_window_subscribes_fitting_worker_progress(qapp, tmp_path, monkeypatch) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.settings["exposure_time"] = 30.0
    approved = project.outputs_dir / StageID.LIGHT_CURVE.value
    approved.mkdir()
    np.savetxt(
        approved / "light_curve_aperture.txt",
        np.column_stack(
            (
                np.linspace(2460000.0, 2460000.1, 12),
                np.ones(12),
                np.full(12, 0.001),
            )
        ),
    )
    project.manifest.stages[StageID.LIGHT_CURVE.value].status = StageStatus.COMPLETE
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    captured = {}

    def capture_start(_function, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(window.runner, "start", capture_start)
    parameters = _parameters()
    window.run_fitting(
        {
            "catalog_parameters": parameters,
            "period": parameters.period,
            "mid_time": parameters.mid_time,
            "depth": parameters.rp_over_rs**2,
            "filter": "COUSINS_R",
            "iterations": 5000,
            "burn": 1000,
        },
        full=True,
    )

    assert captured["event"] == window._stage_event
    window.close()


def test_cached_fitting_setup_defaults_to_project_target(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:06.99"
    project.manifest.target_dec = "+37:32:46.15"
    project.manifest.settings.update(
        {
            "filter": "COUSINS_R",
            "exposure_time": 30.0,
            "fitting_setup": {
                "target_fingerprint": target_fingerprint(
                    project.manifest.target_ra, project.manifest.target_dec
                ),
                "selected_planet": "TrES-3b",
                "light_curve": "gaussian",
                "detrending": "quadratic",
                "exposure_time_override": 45.0,
                "candidates": [asdict(_parameters())],
                "observation": {
                    "filter": "COUSINS_R",
                    "filter_status": "detected",
                    "exposure_time": 30.0,
                    "science_frames_inspected": 354,
                },
            },
        }
    )
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    window.open_stage(StageID.FITTING)
    qapp.processEvents()

    assert window.fitting_page.planet.currentText() == "TrES-3b"
    assert window.fitting_page.values()["catalog_parameters"].name == "TrES-3b"
    assert window.fitting_page.values()["light_curve"] == "gaussian"
    assert window.fitting_page.values()["detrending"] == "quadratic"
    assert window.fitting_page.values()["exposure_time"] == pytest.approx(45.0)
    assert "manual fitting override" in window.fitting_page.observation_source.text()
    assert "COUSINS_R" in window.fitting_page.observation_source.text()
    window.close()


def test_fitting_setup_refreshes_missing_cached_exposure_from_science_fits(
    qapp, tmp_path, monkeypatch
) -> None:
    frame = tmp_path / "science_001.fits"
    fits.writeto(
        frame,
        np.ones((16, 16), dtype=np.float32),
        header=fits.Header(
            {
                "IMAGETYP": "Light Frame",
                "EXPTIME": 30.0,
                "FILTER": "Clear",
            }
        ),
    )
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:06.99"
    project.manifest.target_dec = "+37:32:46.15"
    project.manifest.raw_files["science"] = [frame.name]
    project.manifest.settings["observation_metadata"] = {
        "filter": "clear",
        "filter_status": "detected",
        "exposure_time": None,
        "science_frames_inspected": 1,
        "location_status": "unavailable",
    }
    project.manifest.settings["fitting_setup"] = {
        "target_fingerprint": target_fingerprint(
            project.manifest.target_ra, project.manifest.target_dec
        ),
        "selected_planet": "TrES-3b",
        "candidates": [asdict(_parameters())],
        "observation": dict(project.manifest.settings["observation_metadata"]),
        "light_curve": "gaussian",
        "detrending": "quadratic",
    }
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)

    monkeypatch.setattr(
        PlanetCatalogResolver,
        "resolve_candidates",
        lambda *_args, **_kwargs: [_parameters()],
    )

    def run_immediately(function, **kwargs):
        payload = function()
        kwargs["result"](payload)
        kwargs["finished"]()
        return SimpleNamespace()

    monkeypatch.setattr(window.fitting_lookup_runner, "start", run_immediately)

    window.prepare_fitting_setup()

    assert window.fitting_page.exposure_time.value() == pytest.approx(30.0)
    assert project.manifest.settings["observation_metadata"]["exposure_time"] == pytest.approx(30.0)
    window.close()


def test_uncatalogued_setup_estimates_light_curve_and_requires_period(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "WTS-2")
    project.manifest.target_name = "WTS-2"
    project.manifest.target_ra = "19:34:55.87"
    project.manifest.target_dec = "+36:48:56.00"
    times, flux = _write_approved_curve(project)
    project.save()

    parameters = _manual_planet_parameters(project)
    expected_depth = (np.median(flux) - np.percentile(flux, 10)) / abs(np.median(flux))

    assert parameters.is_manual is True
    assert parameters.period == 0
    assert parameters.mid_time == pytest.approx(np.median(times))
    assert parameters.rp_over_rs**2 == pytest.approx(expected_depth)
    assert parameters.sma_over_rs == 10
    assert parameters.inclination == 90
    assert parameters.temperature == 5500

    page = FittingPage()
    page.set_planet_candidates([parameters])
    page.set_observation_metadata("COUSINS_R", 120.0)

    assert page.manual_notice.isHidden() is False
    assert page.manual_toggle.isHidden() is False
    assert page.manual_assumptions.isHidden() is True
    assert "Manual / uncatalogued" in page.catalog_source.text()
    assert page.period.value() == 0
    assert page.mid_time.value() == pytest.approx(np.median(times))
    assert page.depth.value() == pytest.approx(expected_depth, abs=0.00001)
    assert not page.preview.isEnabled()

    page.manual_toggle.click()
    assert page.manual_assumptions.isHidden() is False
    page.period.setValue(1.0187)
    assert page.preview.isEnabled()
    values = page.values()
    assert values["catalog_parameters"].is_manual is True
    assert values["sma_over_rs"] == 10
    assert values["inclination"] == 90
    page.close()


def test_failed_manual_search_preserves_edits_and_catalog_retry_replaces_it(
    qapp, tmp_path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "WTS-2")
    project.manifest.target_name = "WTS-2"
    project.manifest.target_ra = "19:34:55.87"
    project.manifest.target_dec = "+36:48:56.00"
    _write_approved_curve(project)
    project.manifest.settings.update(
        {
            "filter": "COUSINS_R",
            "exposure_time": 120.0,
            "observation_metadata": {
                "filter": "COUSINS_R",
                "filter_status": "detected",
                "exposure_time": 120.0,
                "science_frames_inspected": 0,
            },
        }
    )
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    candidates: list[PlanetParameters] = []
    failures: list[LEAPSError] = []
    window._show_failure = failures.append

    monkeypatch.setattr(
        PlanetCatalogResolver,
        "resolve_candidates",
        lambda *_args, **_kwargs: list(candidates),
    )

    def run_immediately(function, **kwargs):
        try:
            payload = function()
        except BaseException as exc:
            kwargs["error"](exc)
        else:
            kwargs["result"](payload)
        finally:
            kwargs["finished"]()
        return SimpleNamespace()

    monkeypatch.setattr(window.fitting_lookup_runner, "start", run_immediately)

    window.prepare_fitting_setup(force=True)

    assert failures == []
    assert window.fitting_page.values()["catalog_parameters"].is_manual is True
    assert "COUSINS_R" in window.fitting_page.observation_source.text()
    cached = project.manifest.settings["fitting_setup"]["candidates"][0]
    assert cached["is_manual"] is True

    window.fitting_page.period.setValue(2.345)
    window.fitting_page.sma_over_rs.setValue(12.75)
    window.fitting_page.planet.setEditText("Still missing b")
    window.prepare_fitting_setup(force=True, requested_name="Still missing b")

    assert failures == []
    assert window.fitting_page.planet.currentText() == "Still missing b"
    assert window.fitting_page.period.value() == pytest.approx(2.345)
    assert window.fitting_page.sma_over_rs.value() == pytest.approx(12.75)
    cached = project.manifest.settings["fitting_setup"]["candidates"][0]
    assert cached["period"] == pytest.approx(2.345)
    assert cached["sma_over_rs"] == pytest.approx(12.75)

    project.manifest.target_ra = "19:34:56.00"
    project.manifest.settings.pop("fitting_setup", None)
    project.save()
    window.prepare_fitting_setup(force=True)

    assert window.fitting_page.period.value() == 0
    assert window.fitting_page.sma_over_rs.value() == pytest.approx(10.0)
    assert project.manifest.settings["fitting_setup"]["target_fingerprint"] == target_fingerprint(
        "19:34:56.00", "+36:48:56.00"
    )

    candidates.append(_parameters())
    window.prepare_fitting_setup(force=True, requested_name="TrES-3b")

    assert window.fitting_page.values()["catalog_parameters"].is_manual is False
    assert window.fitting_page.planet.currentText() == "TrES-3b"
    assert window.fitting_page.manual_notice.isHidden() is True
    assert window.fitting_page.manual_toggle.isHidden() is True
    window.close()


def test_manual_values_are_persisted_and_passed_to_fitting_service(
    qapp, tmp_path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "WTS-2")
    project.manifest.target_name = "WTS-2"
    project.manifest.target_ra = "19:34:55.87"
    project.manifest.target_dec = "+36:48:56.00"
    times, _ = _write_approved_curve(project)
    project.manifest.settings["exposure_time"] = 120.0
    project.manifest.settings["observation_metadata"] = {
        "observatory": "SARA-ORM",
        "latitude": 28.76117,
        "longitude": -17.87808,
        "location_status": "detected",
    }
    project.save()
    parameters = _manual_planet_parameters(project)
    window = MainWindow(demo=True)
    window.set_project(project)
    captured: dict[str, object] = {}

    def capture_start(function, **kwargs):
        captured["function"] = function
        captured.update(kwargs)
        return SimpleNamespace()

    def capture_run(_self, _project, fitted_parameters, **kwargs):
        captured["parameters"] = fitted_parameters
        captured["fit_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(window.runner, "start", capture_start)
    monkeypatch.setattr(FittingService, "run", capture_run)
    values = {
        "planet": "WTS-2 manual",
        "catalog_parameters": parameters,
        "period": 1.0187068,
        "mid_time": float(np.median(times) % 1),
        "depth": parameters.rp_over_rs**2,
        "sma_over_rs": 9.75,
        "inclination": 87.2,
        "eccentricity": 0.04,
        "periastron": 15.0,
        "temperature": 6250.0,
        "logg": 4.2,
        "metallicity": -0.15,
        "filter": "COUSINS_R",
        "exposure_time": 45.0,
        "light_curve": "aperture",
        "detrending": "linear",
        "iterations": 5000,
        "burn": 1000,
    }

    window.run_fitting(values, full=False)
    captured["function"]()

    fitted = captured["parameters"]
    assert isinstance(fitted, PlanetParameters)
    assert fitted.name == "WTS-2 manual"
    assert fitted.is_manual is True
    assert fitted.period == pytest.approx(1.0187068)
    assert fitted.mid_time == pytest.approx(float(np.median(times)))
    assert window.fitting_page.mid_time.value() == pytest.approx(float(np.median(times)))
    assert fitted.sma_over_rs == pytest.approx(9.75)
    assert fitted.inclination == pytest.approx(87.2)
    assert fitted.temperature == pytest.approx(6250)
    fit_kwargs = captured["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert fit_kwargs["latitude"] == pytest.approx(28.76117)
    assert fit_kwargs["longitude"] == pytest.approx(-17.87808)
    assert fit_kwargs["exposure_time"] == pytest.approx(45.0)
    assert project.manifest.settings["fitting_setup"]["exposure_time_override"] == pytest.approx(45.0)
    cached = project.manifest.settings["fitting_setup"]["candidates"][0]
    assert cached == asdict(fitted)
    window.close()


def test_busy_runner_rejects_second_photometry_action_without_runtime_error(qapp) -> None:
    window = MainWindow(demo=True)
    failures: list[LEAPSError] = []
    window._show_failure = failures.append
    window.runner.current = object()
    window.runner.current_operation = "photometry"

    window.select_photometry_star("target", 10.0, 10.0)

    assert failures[0].code == "OPERATION_IN_PROGRESS"
    assert "photometry" in failures[0].message
    window.runner.current = None
    window.close()


class _Angle:
    def __init__(self, value):
        self.value = value

    def deg(self):
        return 1.0

    def deg_coord(self):
        return 1.0


class _PyLCInputError(BaseException):
    pass


class _PyLCCancelled(BaseException):
    pass


_AUTO_PREDICTION = object()


class _FakePlanet:
    last_args = None
    last_observation = None
    last_fit = None
    last_prediction = None
    prediction_result = _AUTO_PREDICTION

    def __init__(self, *args):
        self.args = args
        type(self).last_args = args

    def add_observation(self, **kwargs):
        type(self).last_observation = kwargs

    def transit_integrated(self, time, exposure_time, filter_name):
        type(self).last_prediction = {
            "time": time,
            "exposure_time": exposure_time,
            "filter_name": filter_name,
        }
        if type(self).prediction_result is not _AUTO_PREDICTION:
            return type(self).prediction_result
        values = np.asarray(time)
        return np.ones(values.size) - 0.008 * np.exp(
            -((values - values.mean()) / 0.012) ** 2
        )

    def transit_fitting(self, **kwargs):
        type(self).last_fit = kwargs
        callback = kwargs.get("progress_callback")
        if callback:
            callback("optimizing_initial_parameters", 0, 0, {"walkers": 12, "dimensions": 4})
            if kwargs.get("optimiser") == "emcee":
                callback("sampling", 10, kwargs["iterations"], {"walkers": 12, "dimensions": 4})
                callback(
                    "sampling",
                    kwargs["iterations"],
                    kwargs["iterations"],
                    {"walkers": 12, "dimensions": 4},
                )
                callback("writing_results", 1, 1, {"walkers": 12, "dimensions": 4})
        time = np.linspace(2460000.0, 2460000.1, 12)
        flux = np.ones(12)
        model = np.ones(12) - 0.01 * np.exp(-((time - time.mean()) / 0.01) ** 2)
        residuals = flux - model
        return {
            "settings": {"walkers": 12},
            "observations": {
                "obs0": {
                    "model_info": {"epoch": 1793},
                    "parameters": {
                        "rp_over_rs": {
                            "value": 0.1612,
                            "m_error": 0.0011,
                            "p_error": 0.0013,
                            "print_value": "0.1612",
                            "print_m_error": "0.0011",
                            "print_p_error": "0.0013",
                        },
                        "mid_time": {
                            "value": 2460000.0502,
                            "m_error": 0.0002,
                            "p_error": 0.0003,
                            "print_value": "2460000.0502",
                            "print_m_error": "0.0002",
                            "print_p_error": "0.0003",
                        },
                    },
                    "detrended_series": {
                        "time": time,
                        "flux": flux,
                        "flux_unc": np.full(12, 0.001),
                        "model": model,
                        "residuals": residuals,
                    },
                    "detrended_statistics": {"res_std": float(np.std(residuals))},
                }
            }
        }


def _install_fake_fitting_modules(monkeypatch) -> None:
    _FakePlanet.last_args = None
    _FakePlanet.last_observation = None
    _FakePlanet.last_fit = None
    _FakePlanet.last_prediction = None
    _FakePlanet.prediction_result = _AUTO_PREDICTION
    monkeypatch.setitem(
        sys.modules,
        "exoclock",
        SimpleNamespace(Hours=_Angle, Degrees=_Angle),
    )
    monkeypatch.setitem(
        sys.modules,
        "hops.pylightcurve41",
        SimpleNamespace(
            Planet=_FakePlanet,
            PyLCInputError=_PyLCInputError,
            PyLCCancelled=_PyLCCancelled,
            all_filters=lambda: ["COUSINS_R"],
        ),
    )


def test_preview_fit_uses_hops_passband_and_does_not_override_walkers_or_commit_output(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    result = FittingService().run(
        project,
        _parameters(),
        full=False,
        exposure_time=30.0,
        filter_name="COUSINS_R",
        latitude=None,
        longitude=None,
        iterations=500,
        burn_in=100,
    )

    assert result.preview_path.exists()
    assert not (project.outputs_dir / StageID.FITTING.value).exists()
    assert _FakePlanet.last_observation["filter_name"] == "COUSINS_R"
    assert _FakePlanet.last_observation["detrending_series"] == "time"
    assert np.array_equal(
        _FakePlanet.last_prediction["time"],
        result.raw["observations"]["obs0"]["detrended_series"]["time"],
    )
    assert _FakePlanet.last_prediction["exposure_time"] == 30.0
    assert _FakePlanet.last_prediction["filter_name"] == "COUSINS_R"
    assert "walkers" not in _FakePlanet.last_fit
    assert _FakePlanet.last_fit["optimiser"] == "curve_fit"
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.LOCKED


def test_preview_fit_normalizes_fractional_mid_transit_before_building_hops_planet(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    FittingService().run(
        project,
        replace(_parameters(), mid_time=0.05),
        full=False,
        exposure_time=30.0,
        filter_name="COUSINS_R",
        latitude=None,
        longitude=None,
    )

    assert _FakePlanet.last_args[-1] == pytest.approx(2460000.05)
    summary = json.loads(
        (project.temporary_dir / "fitting-preview.json").read_text(encoding="utf-8")
    )
    assert summary["timing_input"] == {
        "entered_mid_time": pytest.approx(0.05),
        "normalized_mid_time": pytest.approx(2460000.05),
    }
    assert summary["parameters"]["mid_time"] == pytest.approx(2460000.05)


def test_hops_eclipse_classification_is_reported_as_timing_error(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    def reject_as_eclipse(_self, **_kwargs):
        raise _PyLCInputError("You need to add only transit observation to proceed.")

    monkeypatch.setattr(_FakePlanet, "transit_fitting", reject_as_eclipse)

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            replace(_parameters(), mid_time=9065.5097),
            full=False,
            exposure_time=30.0,
            filter_name="COUSINS_R",
            latitude=None,
            longitude=None,
        )

    assert error.value.code == "FITTING_TIMING_CLASSIFIED_AS_ECLIPSE"
    assert "Normalized mid-transit: 2459065.50970000" in error.value.technical_details
    assert "decimal-day fraction" in error.value.recovery[0]


def test_preview_fit_omits_isolated_nonfinite_light_curve_rows(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    flux = np.ones(12)
    uncertainty = np.full(12, 0.001)
    flux[5] = np.nan
    uncertainty[5] = np.nan
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, flux, uncertainty)),
    )

    FittingService().run(
        project,
        _parameters(),
        full=False,
        exposure_time=30.0,
        filter_name="COUSINS_R",
        latitude=None,
        longitude=None,
    )

    assert len(_FakePlanet.last_observation["time"]) == 11
    assert np.all(np.isfinite(_FakePlanet.last_observation["flux"]))
    summary = json.loads(
        (project.temporary_dir / "fitting-preview.json").read_text(encoding="utf-8")
    )
    assert summary["data_quality"] == {
        "source_points": 12,
        "excluded_invalid_points": 1,
        "points_passed_to_hops": 11,
    }


def test_preview_fit_rejects_curve_when_too_few_finite_rows_remain(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    flux = np.full(12, np.nan)
    uncertainty = np.full(12, np.nan)
    flux[:9] = 1.0
    uncertainty[:9] = 0.001
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, flux, uncertainty)),
    )

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            _parameters(),
            full=False,
            exposure_time=30.0,
            filter_name="COUSINS_R",
            latitude=None,
            longitude=None,
        )

    assert error.value.code == "FITTING_LIGHT_CURVE_INVALID"
    assert "Fewer than 10 finite measurements" in error.value.technical_details


def test_preview_fit_can_use_gaussian_light_curve_and_quadratic_detrending(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    gaussian_flux = np.linspace(0.98, 1.02, 12)
    np.savetxt(
        light_curve_output / "light_curve_gauss.txt",
        np.column_stack((time, gaussian_flux, np.full(12, 0.002))),
    )

    FittingService().run(
        project,
        _parameters(),
        full=False,
        exposure_time=30.0,
        filter_name="COUSINS_R",
        latitude=None,
        longitude=None,
        light_curve="gaussian",
        detrending="quadratic",
        iterations=500,
        burn_in=100,
    )

    assert np.array_equal(_FakePlanet.last_observation["flux"], gaussian_flux)
    assert _FakePlanet.last_observation["detrending_series"] == "time"
    assert _FakePlanet.last_observation["detrending_order"] == 2


def test_airmass_detrending_requires_observer_location(tmp_path: Path, monkeypatch) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    light_curve_output = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve_output.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        light_curve_output / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            _parameters(),
            full=False,
            exposure_time=30.0,
            filter_name="COUSINS_R",
            latitude=None,
            longitude=None,
            detrending="airmass",
        )

    assert error.value.code == "FITTING_AIRMASS_LOCATION_REQUIRED"


def test_full_fit_replaces_output_only_after_success(tmp_path: Path, monkeypatch) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )
    previous = project.outputs_dir / StageID.FITTING.value
    previous.mkdir()
    (previous / "old-result.txt").write_text("preserved until commit", encoding="utf-8")

    result = FittingService().run(
        project,
        _parameters(),
        full=True,
        exposure_time=30.0,
        filter_name="Cousins_R",
        latitude=42.0,
        longitude=-83.0,
        iterations=500,
        burn_in=100,
    )

    assert result.output_path == previous
    assert (previous / "fit-summary.json").exists()
    assert (previous / "fit-preview.png").exists()
    assert not (previous / "old-result.txt").exists()
    assert _FakePlanet.last_observation["detrending_series"] == "airmass"
    assert _FakePlanet.last_fit["optimiser"] == "emcee"
    summary = json.loads((previous / "fit-summary.json").read_text(encoding="utf-8"))
    assert summary["walkers"] == 12
    assert summary["walker_policy"] == "hops_auto"


def test_fitting_service_passes_prediction_to_preview_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )
    predicted = np.linspace(0.97, 1.0, 12)
    _FakePlanet.prediction_result = predicted
    captured = {}
    parameters = replace(
        _parameters(),
        source="Manual / uncatalogued",
        is_manual=True,
    )

    def capture_preview(
        observation,
        predicted_model,
        destination,
        *,
        catalog_parameters,
        observation_times_jd,
        exposure_time,
        filter_name,
    ):
        captured["observation"] = observation
        captured["predicted_model"] = predicted_model
        captured["catalog_parameters"] = catalog_parameters
        captured["observation_times_jd"] = observation_times_jd
        captured["exposure_time"] = exposure_time
        captured["filter_name"] = filter_name
        destination.write_bytes(b"preview")

    monkeypatch.setattr("leaps.science._write_fit_preview", capture_preview)

    result = FittingService().run(
        project,
        parameters,
        full=False,
        exposure_time=30.0,
        filter_name="COUSINS_R",
        latitude=None,
        longitude=None,
    )

    assert result.preview_path.exists()
    assert captured["predicted_model"] is predicted
    assert captured["observation"] is result.raw["observations"]["obs0"]
    assert captured["catalog_parameters"] == parameters
    assert np.array_equal(captured["observation_times_jd"], time)
    assert captured["exposure_time"] == 30.0
    assert captured["filter_name"] == "COUSINS_R"
    summary = json.loads(
        (project.temporary_dir / "fitting-preview.json").read_text(encoding="utf-8")
    )
    assert summary["source"] == "Manual / uncatalogued"
    assert summary["parameters"]["is_manual"] is True


@pytest.mark.parametrize(
    "prediction",
    [
        None,
        np.ones(11),
        np.array([1.0] * 11 + [np.nan]),
    ],
    ids=["missing", "wrong-size", "non-finite"],
)
def test_invalid_prediction_preserves_previous_full_fit(
    tmp_path: Path, monkeypatch, prediction
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )
    previous = project.outputs_dir / StageID.FITTING.value
    previous.mkdir()
    previous_preview = previous / "fit-preview.png"
    previous_preview.write_bytes(b"last successful preview")
    _FakePlanet.prediction_result = prediction

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            _parameters(),
            full=True,
            exposure_time=30.0,
            filter_name="COUSINS_R",
            latitude=None,
            longitude=None,
        )

    assert error.value.code == "FITTING_FAILED"
    assert previous_preview.read_bytes() == b"last successful preview"
    assert not (project.temporary_dir / "fitting-pending").exists()


def test_fit_preview_renders_best_fit_and_predicted_transits(
    tmp_path: Path, monkeypatch
) -> None:
    from matplotlib.figure import Figure

    time = np.linspace(2461231.41, 2461231.56, 120)
    best_fit = 1.0 - 0.028 * np.exp(-((time - 2461231.480) / 0.018) ** 4)
    predicted = 1.0 - 0.021 * np.exp(-((time - 2461231.476) / 0.015) ** 4)
    flux = best_fit + 0.0006 * np.sin(np.linspace(0, 10, time.size))
    observation = {
        "model_info": {"epoch": 2736},
        "parameters": {
            "rp_over_rs": {
                "value": 0.1557,
                "m_error": 0.0015,
                "p_error": 0.0015,
                "print_value": "0.1557",
                "print_m_error": "0.0015",
                "print_p_error": "0.0015",
            },
            "mid_time": {
                "value": 2461231.47999,
                "m_error": 0.00022,
                "p_error": 0.00022,
                "print_value": "2461231.47999",
                "print_m_error": "0.00022",
                "print_p_error": "0.00022",
            },
        },
        "detrended_series": {
            "time": time,
            "flux": flux,
            "flux_unc": np.full(time.size, 0.0008),
            "model": best_fit,
            "residuals": flux - best_fit,
        }
    }
    labels = []
    rendered_text = []
    original_savefig = Figure.savefig

    def capture_legend(figure, *args, **kwargs):
        for axis in figure.axes:
            labels.extend(axis.get_legend_handles_labels()[1])
        rendered_text.extend(
            text.get_text() for axis in figure.axes for text in axis.texts
        )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_legend)
    preview = tmp_path / "fit-preview.png"

    parameters = _parameters()
    observation_times_jd = np.linspace(
        2461231.4066090495,
        2461231.5546502187,
        time.size,
    )
    _write_fit_preview(
        observation,
        predicted,
        preview,
        catalog_parameters=parameters,
        observation_times_jd=observation_times_jd,
        exposure_time=30.0,
        filter_name="COUSINS_R",
    )

    pixels = np.asarray(Image.open(preview).convert("RGB"), dtype=int)
    assert pixels.shape[:2] == (1680, 2400)
    best_fit_pixels = np.all(np.abs(pixels - np.array([32, 197, 244])) <= 2, axis=2)
    predicted_pixels = np.all(np.abs(pixels - np.array([255, 98, 76])) <= 2, axis=2)
    assert best_fit_pixels.sum() > 20
    assert predicted_pixels.sum() > 20
    best_fit_label = next(label for label in labels if label.startswith("Best-fit transit"))
    predicted_label = next(label for label in labels if label.startswith("Predicted transit"))
    expected_mid_time = parameters.mid_time + 2736 * parameters.period
    assert "2461231.47999" in best_fit_label
    assert "0.1557" in best_fit_label
    assert f"{expected_mid_time:.8f}" in predicted_label
    assert f"{parameters.rp_over_rs:.5f}" in predicted_label
    assert r"O\! -\! C=" in predicted_label
    assert "-0.8^{+0.3}_{-0.3}" in predicted_label
    assert r"\mathrm{min}" in predicted_label
    assert "LEAPS" in rendered_text
    assert "Exoplanet Transit Analysis" in rendered_text
    assert "TrES-3b" in rendered_text
    metadata = next(text for text in rendered_text if "(UT)" in text)
    assert "2026-07-09 21:45 (UT)" in metadata
    assert "Dur: 3.6h / Exp: 30.0s" in metadata
    assert "Filter: Cousins R" in metadata
    assert "Observatory" not in metadata

    manual_label_start = len(labels)
    _write_fit_preview(
        observation,
        predicted,
        tmp_path / "manual-fit-preview.png",
        catalog_parameters=replace(
            parameters,
            source="Manual / uncatalogued",
            is_manual=True,
        ),
        observation_times_jd=observation_times_jd,
        exposure_time=30.0,
        filter_name="COUSINS_R",
    )
    assert any(
        label.startswith("Predicted transit (manual inputs)")
        for label in labels[manual_label_start:]
    )


@pytest.mark.parametrize(
    ("phase", "current", "total"),
    [
        ("optimizing_initial_parameters", 0, 0),
        ("sampling", 10, 500),
        ("writing_results", 0, 1),
    ],
)
def test_fitting_service_reports_progress_and_discards_cancelled_attempt(
    tmp_path: Path, monkeypatch, phase: str, current: int, total: int
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )
    previous = project.outputs_dir / StageID.FITTING.value
    previous.mkdir()
    (previous / "last-success.txt").write_text("keep", encoding="utf-8")
    token = CancellationToken()
    events: list[StageEvent] = []

    def cancel_during_phase(self, **kwargs):
        kwargs["progress_callback"](
            phase, current, total, {"walkers": 12, "dimensions": 4}
        )
        token.cancel()
        if kwargs["cancelled"]():
            raise _PyLCCancelled("cancelled")

    monkeypatch.setattr(_FakePlanet, "transit_fitting", cancel_during_phase)

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            _parameters(),
            full=True,
            exposure_time=30.0,
            filter_name="COUSINS_R",
            latitude=None,
            longitude=None,
            iterations=500,
            burn_in=100,
            emit=events.append,
            token=token,
        )

    assert error.value.code == "JOB_CANCELLED"
    assert (previous / "last-success.txt").read_text(encoding="utf-8") == "keep"
    assert not (project.temporary_dir / "fitting-pending").exists()
    assert any(event.checkpoint == phase for event in events)
    progress_event = next(event for event in events if event.checkpoint == phase)
    assert progress_event.details["walkers"] == 12


def test_hops_auto_walkers_and_batched_progress() -> None:
    script = """
import json
import sys
from types import SimpleNamespace
import numpy as np
sys.modules['exoclock'] = SimpleNamespace(FixedTarget=object, Degrees=lambda value: value)
from hops.pylightcurve41.analysis.optimisation import Fitting
from hops.pylightcurve41.errors import PyLCCancelled
progress = []
fitting = Fitting(
    np.arange(4, dtype=float), np.ones(4), np.ones(4),
    lambda x, first, second: np.full_like(x, first + second),
    [0.5, 0.5], [0.0, 0.0], [1.0, 1.0],
    iterations=25, burn_in=5, optimise_initial_parameters=False,
    progress_callback=lambda phase, current, total, details: progress.append(
        (phase, current, total, details)
    ),
)
class Sampler:
    def run_mcmc(self, *_args, **_kwargs):
        return None
fitting.sampler = Sampler()
fitting.counter = SimpleNamespace(update=lambda: None)
fitting._emcee_run_headless()
cancelled_fitting = Fitting(
    np.arange(4, dtype=float), np.ones(4), np.ones(4),
    lambda x, first, second: np.full_like(x, first + second),
    [0.5, 0.5], [0.0, 0.0], [1.0, 1.0],
    optimise_initial_parameters=False, cancelled=lambda: True,
)
cancelled = False
try:
    cancelled_fitting._probability(np.array([0.5, 0.5]))
except PyLCCancelled:
    cancelled = True
print(json.dumps({
    'walkers': fitting.walkers,
    'steps': [item[1] for item in progress if item[0] == 'sampling'],
    'reported_walkers': [item[3]['walkers'] for item in progress],
    'cancelled_in_probability': cancelled,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["walkers"] == 6
    assert payload["steps"] == [0, 10, 20, 25]
    assert all(value == 6 for value in payload["reported_walkers"])
    assert payload["cancelled_in_probability"] is True


def test_preview_completion_keeps_fitting_revisitable_instead_of_complete(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.stages[StageID.FITTING.value].status = StageStatus.READY
    project.save()
    preview = tmp_path / "preview.png"
    page = FittingPage()
    assert page.grab().save(str(preview))
    page.close()
    result = FittingService.Result(
        full=False,
        planet="TrES-3b",
        passband="COUSINS_R",
        preview_path=preview,
        output_path=None,
        residual_std=0.0028,
        raw={},
    )
    window = MainWindow(demo=True)
    window.set_project(project)

    window._fitting_complete(result)

    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert project.manifest.stages[StageID.FITTING.value].summary == "Preview ready"
    window.close()


def test_interrupted_full_fit_recovers_without_removing_last_success(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.stages[StageID.FITTING.value].status = StageStatus.RUNNING
    project.manifest.stages[StageID.FITTING.value].summary = "Sampling posterior"
    previous = project.outputs_dir / StageID.FITTING.value
    previous.mkdir()
    (previous / "last-success.txt").write_text("keep", encoding="utf-8")
    pending = project.temporary_dir / "fitting-pending"
    pending.mkdir()
    (pending / "partial.txt").write_text("discard", encoding="utf-8")
    project.save()

    window = MainWindow(demo=True)
    window.set_project(project)

    state = project.manifest.stages[StageID.FITTING.value]
    assert state.status == StageStatus.READY
    assert state.summary == "Interrupted · ready to run again"
    assert state.checkpoint == "interrupted"
    assert "FITTING_INTERRUPTED" in state.warning_codes
    assert not pending.exists()
    assert (previous / "last-success.txt").read_text(encoding="utf-8") == "keep"
    window.close()


def test_discard_pending_fit_does_not_follow_symlink(tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path / "run", "TrES-3")
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    pending = project.temporary_dir / "fitting-pending"
    pending.symlink_to(outside, target_is_directory=True)

    assert project.discard_pending_transaction(StageID.FITTING)
    assert not pending.exists()
    assert protected.read_text(encoding="utf-8") == "keep"


def test_cancelled_fit_is_not_presented_as_failure(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.stages[StageID.FITTING.value].status = StageStatus.RUNNING
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    shown: list[LEAPSError] = []
    window._show_failure = shown.append

    window._fitting_failed(
        LEAPSError(
            "JOB_CANCELLED",
            "Full fit cancelled",
            "The incomplete fitting attempt was discarded.",
            ["Run Full Fit again"],
            stage=StageID.FITTING,
        ),
        full=True,
    )

    assert shown == []
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert project.manifest.stages[StageID.FITTING.value].summary == "Full fit cancelled"
    assert "previous preview" in window.fitting_page.message.text()
    window.close()


def test_fitting_service_reports_unavailable_passband_as_typed_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    with pytest.raises(LEAPSError) as error:
        FittingService().run(
            project,
            _parameters(),
            full=False,
            exposure_time=30.0,
            filter_name="not-a-passband",
            latitude=None,
            longitude=None,
        )

    assert error.value.code == "FITTING_FILTER_UNAVAILABLE"


def test_fitting_service_translates_legacy_r_before_calling_hops(tmp_path: Path, monkeypatch) -> None:
    _install_fake_fitting_modules(monkeypatch)
    project = ProjectWorkspace.create(tmp_path)
    photometry = project.outputs_dir / StageID.LIGHT_CURVE.value
    photometry.mkdir()
    time = np.linspace(2460000.0, 2460000.1, 12)
    np.savetxt(
        photometry / "light_curve_aperture.txt",
        np.column_stack((time, np.ones(12), np.full(12, 0.001))),
    )

    FittingService().run(
        project,
        _parameters(),
        full=False,
        exposure_time=30.0,
        filter_name="R",
        latitude=None,
        longitude=None,
    )

    assert _FakePlanet.last_observation["filter_name"] == "COUSINS_R"
