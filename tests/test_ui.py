from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton

import leaps.ui.main_window as main_window_module
from leaps.fits_inventory import FrameRecord
from leaps.models import JobStatus, StageEvent, StageID, StageState, StageStatus
from leaps.project import ProjectWorkspace
from leaps.ui.main_window import MainWindow, ProjectResetDialog
from leaps.ui.pages import (
    ComparisonStarsPage,
    DataTargetPage,
    FittingPage,
    LightCurvePage,
    PlateSolvePage,
    ProcessingPage,
    SecondaryEclipsePage,
)
from leaps.ui.widgets import FITSWorkspace, InfoButton, StageNavButton


def test_macos_app_icon_has_native_size_and_transparent_corners() -> None:
    path = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-app-icon.png"
    icon = Image.open(path)
    assert icon.size == (1024, 1024)
    assert icon.mode == "RGBA"
    alpha = icon.getchannel("A")
    assert all(alpha.getpixel(point) == 0 for point in ((0, 0), (1023, 0), (0, 1023), (1023, 1023)))
    assert alpha.getbbox() is not None


def test_shared_leaps_mark_is_centered_in_its_tile() -> None:
    path = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-mark.png"
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    luminosity = pixels.mean(axis=2)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    white = (luminosity > 120) & (saturation < 55)
    cyan = (
        (pixels[:, :, 2] > 100)
        & (pixels[:, :, 1] > 75)
        & (pixels[:, :, 0] < 80)
        & ((pixels[:, :, 1] - pixels[:, :, 0]) > 25)
    )
    y, x = np.where(white | cyan)
    mark_center = ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)
    tile_center = ((pixels.shape[1] - 1) / 2, (pixels.shape[0] - 1) / 2)
    assert abs(mark_center[0] - tile_center[0]) <= 1
    assert abs(mark_center[1] - tile_center[1]) <= 1


def test_demo_window_opens_selected_plate_solve_state(qapp) -> None:
    window = MainWindow(demo=True)
    window.show()
    qapp.processEvents()
    assert window.stack.currentWidget() is window.plate_page
    assert window.stage_buttons[StageID.PHOTOMETRY].active
    assert window.plate_page.inspector.retry.isEnabled()
    window.close()


def test_scientific_controls_have_accessible_information_buttons(qapp) -> None:
    window = MainWindow(demo=True)
    info_buttons = window.findChildren(InfoButton)
    assert len(info_buttons) >= 15
    assert all(button.toolTip().strip() for button in info_buttons)
    assert all(button.accessibleName() == "Information" for button in info_buttons)
    window.close()


def test_sidebar_does_not_show_inert_collapse_control(qapp) -> None:
    window = MainWindow(demo=True)
    assert not any(
        button.toolTip() == "Collapse the workflow sidebar."
        for button in window.sidebar.findChildren(QPushButton)
    )
    window.close()


def test_light_curve_is_required_workflow_stage_not_a_tool(qapp) -> None:
    window = MainWindow(demo=True)
    stages = list(window.stage_buttons)
    assert stages.index(StageID.LIGHT_CURVE) == stages.index(StageID.PHOTOMETRY) + 1
    assert stages.index(StageID.FITTING) == stages.index(StageID.LIGHT_CURVE) + 1
    assert stages.index(StageID.SECONDARY_ECLIPSE) == stages.index(StageID.FITTING) + 1
    assert "light_curve" not in window.tool_buttons
    assert window.pages[StageID.LIGHT_CURVE] is window.light_curve_page
    assert window.pages[StageID.SECONDARY_ECLIPSE] is window.secondary_eclipse_page
    window.close()


def test_secondary_eclipse_page_needs_full_fit_context_before_analysis(qapp) -> None:
    page = SecondaryEclipsePage()
    assert not page.analyze.isEnabled()

    from leaps.catalog import PlanetParameters

    parameters = PlanetParameters(
        name="WASP-12 b",
        ra="06:30:32.79",
        dec="+29:40:20.3",
        period=1.09142,
        mid_time=2454508.97682,
        rp_over_rs=0.117,
        sma_over_rs=3.0,
        inclination=83.4,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.3,
        temperature=6300.0,
        logg=4.2,
        source="ExoClock",
    )
    page.set_fit_context(parameters, passband="COUSINS_R", duration_hours=2.9)

    assert page.analyze.isEnabled()
    assert page.expected_phase.value() == 0.5
    assert page.duration_hours.value() == 2.9
    assert "WASP-12 b" in page.fit_context.text()
    page.close()


def test_secondary_eclipse_page_reloads_saved_setup_and_flags_strong_control(qapp, tmp_path) -> None:
    page = SecondaryEclipsePage()
    from leaps.catalog import PlanetParameters

    page.set_fit_context(
        PlanetParameters(
            name="WASP-18 b",
            ra="01:37:25.07",
            dec="-45:40:40.1",
            period=0.94145,
            mid_time=2458354.45,
            rp_over_rs=0.1018,
            sma_over_rs=3.48,
            inclination=83.5,
            eccentricity=0.0,
            periastron=0.0,
            metallicity=0.0,
            temperature=6400.0,
            logg=4.3,
            source="Test",
        )
    )
    page.show_saved_result(
        {
            "light_curve": "gaussian",
            "baseline": "quadratic",
            "expected_phase": 0.5003,
            "duration_hours": 2.21,
            "message": "A positive fixed-phase eclipse is recovered.",
            "outcome": "candidate",
            "outcome_label": "Candidate signal · independent check required",
            "depth_ppm": 377.0,
            "depth_uncertainty_ppm": 14.0,
            "significance": 26.3,
            "red_noise_beta": 1.76,
            "local_points": 51_507,
            "in_eclipse_points": 10_501,
            "event_count": 170,
            "control_significance": 5.6,
        },
        tmp_path / "missing-preview.png",
    )

    assert page.light_curve.currentData() == "gaussian"
    assert page.baseline.currentData() == "quadratic"
    assert page.expected_phase.value() == 0.5003
    assert page.duration_hours.value() == 2.21
    assert "Nearby control phase: 5.6σ" in page.message.text()
    assert "review" in page.metric_values["control"].text()
    page.close()


def test_secondary_eclipse_ml_check_is_explicitly_gated_by_a_saved_result(qapp, tmp_path) -> None:
    page = SecondaryEclipsePage()
    from leaps.catalog import PlanetParameters

    page.set_fit_context(
        PlanetParameters(
            name="WASP-18 b",
            ra="01:37:25.07",
            dec="-45:40:40.1",
            period=0.94145,
            mid_time=2458354.45,
            rp_over_rs=0.1018,
            sma_over_rs=3.48,
            inclination=83.5,
            eccentricity=0.0,
            periastron=0.0,
            metallicity=0.0,
            temperature=6400.0,
            logg=4.3,
            source="Test",
        )
    )
    page.set_ml_context(True, "Uses held-out TESS sectors; it never changes the LEAPS eclipse decision.")
    page.set_cross_target_context(True, "Use this target plus two other saved ML trial tables.")
    assert not page.run_ml.isEnabled()
    assert not page.run_cross_target.isEnabled()

    preview = tmp_path / "secondary-eclipse.png"
    Image.new("RGB", (800, 500), "#0b2638").save(preview)
    page.show_saved_result(
        {
            "light_curve": "aperture",
            "baseline": "linear",
            "expected_phase": 0.5,
            "duration_hours": 2.21,
            "outcome": "candidate",
            "outcome_label": "Candidate signal · independent check required",
            "depth_ppm": 377.0,
            "depth_uncertainty_ppm": 14.0,
            "significance": 26.3,
            "red_noise_beta": 1.76,
            "local_points": 100,
            "in_eclipse_points": 30,
            "event_count": 4,
        },
        preview,
    )

    assert page.run_ml.isEnabled()
    ml_preview = tmp_path / "ml-validation.png"
    Image.new("RGB", (800, 500), "#0b2638").save(ml_preview)
    page.show_saved_ml_result(
        {
            "metrics": {
                "test_roc_auc": 0.94,
                "test_ml_false_alarm_rate": 0.0,
                "calibration_false_alarm_target": 0.0,
            },
            "recommendation": "Validation only.",
        },
        ml_preview,
    )
    assert page.run_cross_target.isEnabled()
    assert page.values()["ml_trials_per_split"] == 240
    page.close()


def test_light_curve_page_defaults_comparisons_on_and_keeps_one_active(qapp, tmp_path) -> None:
    preview = tmp_path / "light-curves.png"
    image = Image.new("RGB", (800, 600), "#0b2638")
    image.save(preview)
    result = type(
        "ReviewResult",
        (),
        {
            "active_comparisons": [True, True, True],
            "preview_path": preview,
            "frame_count": 354,
            "curves": [
                type("Curve", (), {"label": label, "missing_frames": 0})
                for label in ("Target", "C1", "C2", "C3")
            ],
        },
    )()
    page = LightCurvePage()
    page.set_review(result)
    assert page.active_comparisons() == [True, True, True]
    assert page.continue_button.isEnabled()

    selections = []
    page.selectionChanged.connect(selections.append)
    page.comparison_checks[1].setChecked(False)
    assert selections[-1] == [True, False, True]
    page.comparison_checks[0].setChecked(False)
    page.comparison_checks[2].setChecked(False)
    assert page.active_comparisons() == [False, False, True]
    assert "At least one" in page.message.text()
    page.close()


def test_information_button_opens_immediately_on_hover_and_click(qapp) -> None:
    button = InfoButton("Explains this scientific setting.")
    button.show()
    qapp.processEvents()

    QTest.mouseMove(button, QPoint(button.width() // 2, button.height() // 2))
    qapp.processEvents()
    assert button._popover.isVisible()
    assert button._popover.label.text() == "Explains this scientific setting."
    assert button._popover.width() >= 360

    button._popover.hide()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert button._popover.isVisible()
    assert button._pinned is True
    button._popover.hide()
    button.close()


def test_fit_preview_reveal_selects_file_in_platform_file_manager(
    tmp_path, monkeypatch
) -> None:
    preview = tmp_path / "fit-preview.png"
    preview.write_bytes(b"preview")
    calls = []
    monkeypatch.setattr(
        main_window_module,
        "_start_detached",
        lambda program, arguments: calls.append((program, arguments)) or True,
    )

    for platform, program, arguments in (
        ("darwin", "/usr/bin/open", ["-R", str(preview)]),
        ("win32", "explorer.exe", ["/select,", str(preview)]),
    ):
        monkeypatch.setattr(main_window_module.sys, "platform", platform)
        main_window_module._reveal_in_file_manager(preview)
        assert calls[-1] == (program, arguments)


def test_locked_stage_uses_prominent_solid_lock_without_enlarging_other_states(qapp) -> None:
    button = StageNavButton(StageID.FITTING, "Fitting")
    button.update_state(StageState(status=StageStatus.LOCKED, summary="Locked"))
    qapp.processEvents()
    assert button.status_icon.pixmap().size().width() == 30
    assert button.status_icon.width() == 36
    assert button.title.objectName() == "stageTitle"
    assert not button.isEnabled()

    button.update_state(StageState(status=StageStatus.READY, summary="Ready"))
    qapp.processEvents()
    assert button.status_icon.pixmap().size().width() == 23
    assert button.isEnabled()


def test_running_stage_spins_only_in_sidebar(qapp) -> None:
    button = StageNavButton(StageID.REDUCTION, "Reduction")
    button.update_state(StageState(status=StageStatus.RUNNING, summary="Processing"))
    assert button._spinner_timer.isActive()
    before = button._spinner_phase
    button._advance_spinner()
    assert button._spinner_phase != before

    button.update_state(StageState(status=StageStatus.COMPLETE, summary="Complete"))
    assert not button._spinner_timer.isActive()
    assert button._spinner_phase == 0


def test_frame_assignment_cards_use_live_filename_classifiers(qapp) -> None:
    page = DataTargetPage()
    records = [
        FrameRecord("bias_001.fits", "bias", 1.0, "", (20, 20), 16, 0.0, "a"),
        FrameRecord("d_001.fits", "dark", 1.0, "", (20, 20), 16, 30.0, "b"),
        FrameRecord("flat_001.fits", "flat", 1.0, "", (20, 20), 16, 5.0, "c"),
        FrameRecord("image_001.fits", "science", 1.0, "", (20, 20), 16, 30.0, "d"),
    ]
    page.set_records(records)
    assert page.assignment_cards["bias"].count.text() == "1 selected"
    assert page.assignment_cards["dark"].count.text() == "0 selected"
    assert page.assignment_cards["flat"].count.text() == "1 selected"
    assert page.assignment_cards["science"].count.text() == "1 selected"
    assert page.counts.text() == "3 assigned · 1 unmatched"

    page.assignment_cards["dark"].classifier.setText("Dark, D")
    qapp.processEvents()
    assert page.assignment_cards["dark"].count.text() == "1 selected"
    assert page.counts.text() == "4 assigned · 0 unmatched"


def test_data_target_page_exposes_tess_light_curve_import(qapp) -> None:
    page = DataTargetPage()
    assert page.import_tess.text() == "Import TESS light curves"
    assert "PDCSAP" in page.import_tess.toolTip()
    page.show_tess_import_result("Imported 1,234 TESS points.")
    assert not page.tess_import_status.isHidden()
    assert "1,234" in page.tess_import_status.text()
    page.close()


def test_frame_assignment_counts_filename_matches_before_header_scan(qapp, tmp_path) -> None:
    for index in range(21):
        (tmp_path / f"bias_{index + 1:03d}.fits").touch()
    for index in range(7):
        (tmp_path / f"dark_60_{index + 1:03d}.fits").touch()
    for index in range(5):
        (tmp_path / f"flat_{index + 1:03d}.fits").touch()
    for index in range(12):
        (tmp_path / f"TrES-3_Cousins_R_{index + 1:03d}.fits").touch()

    page = DataTargetPage()
    page.preview_folder(tmp_path)
    assert page.assignment_cards["bias"].count.text() == "21 selected"
    assert page.assignment_cards["dark"].count.text() == "7 selected"
    assert page.assignment_cards["flat"].count.text() == "5 selected"
    assert page.assignment_cards["science"].count.text() == "0 selected"
    assert page.counts.text() == "33 assigned · 12 unmatched"

    page.assignment_cards["science"].classifier.setText("TrES-3")
    qapp.processEvents()
    assert page.assignment_cards["science"].count.text() == "12 selected"
    assert page.counts.text() == "45 assigned · 0 unmatched"


def test_saved_assignments_restore_without_rescanning(qapp) -> None:
    page = DataTargetPage()
    page.restore_project_assignments(
        {
            "bias": [f"bias_{index:03d}.fits" for index in range(21)],
            "dark": [f"dark_{index:03d}.fits" for index in range(7)],
            "flat": [f"flat_{index:03d}.fits" for index in range(5)],
            "science": [f"TrES-3_{index:03d}.fits" for index in range(354)],
        },
        {"bias": "Bias", "dark": "Dark", "flat": "Flat", "science": "TrES-3"},
        {"bias": False, "dark": False, "flat": False},
    )
    assert page.assignment_cards["bias"].count.text() == "21 selected"
    assert page.assignment_cards["science"].count.text() == "354 selected"
    assert page.counts.text() == "387 assigned · 0 unmatched"


def test_frame_assignments_do_not_show_calibration_waiver_checkboxes(qapp) -> None:
    page = DataTargetPage()
    labels = [checkbox.text() for checkbox in page.frames_card.findChildren(QCheckBox)]
    assert not any(label.startswith("Continue without") for label in labels)


def test_missing_calibration_confirmation_has_cancel_and_acknowledge(qapp, monkeypatch) -> None:
    buttons: list[str] = []
    messages: list[str] = []

    class FakeMessageBox:
        acknowledge_next = True

        class Icon:
            Warning = object()

        class ButtonRole:
            RejectRole = object()
            AcceptRole = object()

        def __init__(self, parent) -> None:
            self.acknowledge = None
            self.cancel = None

        def setIcon(self, icon) -> None:
            pass

        def setWindowTitle(self, title: str) -> None:
            assert title == "Missing calibration frames"

        def setText(self, text: str) -> None:
            messages.append(text)

        def setInformativeText(self, text: str) -> None:
            assert "reduce the quality" in text

        def addButton(self, text: str, role):
            button = object()
            buttons.append(text)
            if text == "Acknowledge":
                self.acknowledge = button
            else:
                self.cancel = button
            return button

        def setDefaultButton(self, button) -> None:
            pass

        def setEscapeButton(self, button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self.acknowledge if self.acknowledge_next else self.cancel

    monkeypatch.setattr("leaps.ui.main_window.QMessageBox", FakeMessageBox)
    window = MainWindow(demo=True)
    assert window._confirm_missing_calibration_frames(["bias", "dark", "flat"])
    assert buttons == ["Cancel", "Acknowledge"]
    assert "Bias, Darks, Flats" in messages[0]
    FakeMessageBox.acknowledge_next = False
    assert not window._confirm_missing_calibration_frames(["bias"])
    window.close()


def test_real_fits_workspace_pan_zoom_invert_and_reset(qapp, tmp_path) -> None:
    frame = tmp_path / "TrES-3_reference.fits"
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    fits.writeto(frame, data, overwrite=True)
    workspace = FITSWorkspace(tmp_path / "missing-demo.png")
    workspace.resize(800, 650)
    workspace.show()
    workspace.load_fits(frame, 1.2)
    qapp.processEvents()

    initial_key = workspace.image.image_item.pixmap().cacheKey()
    workspace.mode_buttons["invert"].click()
    qapp.processEvents()
    assert workspace.image.image_item.pixmap().cacheKey() != initial_key
    workspace.zoom.setCurrentText("200%")
    assert workspace.image.transform().m11() == 2.0
    workspace.reset_view()
    assert workspace.zoom.currentText() == "Fit"
    assert workspace.filename.text() == "FITS: TrES-3_reference.fits"
    assert workspace.dimensions.text() == "64 × 64 px"
    workspace.close()


def test_middle_mouse_drag_pans_fits_workspace_without_leaving_selection_mode(qapp, tmp_path) -> None:
    frame = tmp_path / "TrES-3_reference.fits"
    fits.writeto(frame, np.arange(256 * 256, dtype=np.float32).reshape(256, 256), overwrite=True)
    workspace = FITSWorkspace(tmp_path / "missing-demo.png")
    workspace.resize(800, 650)
    workspace.show()
    workspace.load_fits(frame, 1.2)
    qapp.processEvents()
    workspace.zoom.setCurrentText("400%")
    workspace.begin_selection("comparison")
    viewport = workspace.image.viewport()
    start = QPoint(viewport.width() // 2, viewport.height() // 2)
    before = (
        workspace.image.horizontalScrollBar().value(),
        workspace.image.verticalScrollBar().value(),
    )

    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseMove(viewport, QPoint(start.x() - 30, start.y() - 24))
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, pos=QPoint(start.x() - 30, start.y() - 24))
    qapp.processEvents()

    after = (
        workspace.image.horizontalScrollBar().value(),
        workspace.image.verticalScrollBar().value(),
    )
    assert after != before
    assert workspace.image.mode == "select"
    assert workspace.image.selection_role == "comparison"
    workspace.close()


def test_run_and_cancel_buttons_show_active_processing_state(qapp, tmp_path) -> None:
    processing = ProcessingPage(StageID.REDUCTION, "Reduction", "Calibrate frames", [])
    assert processing.run.text() == "Run Reduction"
    assert processing.cancel.text() == "Cancel"
    assert not processing.cancel.isEnabled()

    processing.set_busy(True)
    qapp.processEvents()
    assert processing.run.text() == "Running Reduction…"
    assert processing.run.property("running") is True
    assert processing.run.icon().isNull()
    assert not processing.run.isEnabled()
    assert processing.cancel.property("cancelActive") is True
    assert processing.cancel.isEnabled()

    processing.set_busy(False)
    assert processing.run.text() == "Run Reduction"
    assert processing.run.property("running") is False
    assert processing.run.isEnabled()
    assert processing.cancel.property("cancelActive") is False
    assert not processing.cancel.isEnabled()

    photometry = PlateSolvePage(tmp_path / "missing-preview.png")
    photometry.inspector.set_busy(True)
    assert photometry.inspector.run.text() == "Running Photometry…"
    assert photometry.inspector.run.icon().isNull()
    assert photometry.inspector.cancel.isEnabled()
    assert photometry.inspector.cancel.property("cancelActive") is True
    photometry.inspector.set_busy(False)
    assert photometry.inspector.run.text() == "Run HOPS photometry"

    comparison = ComparisonStarsPage()
    comparison.set_busy(True)
    assert comparison.run.text() == "Running Photometry…"
    assert comparison.run.icon().isNull()
    assert comparison.cancel.isEnabled()
    comparison.set_busy(False)
    assert comparison.run.text() == "Run photometry"

    fitting = FittingPage()
    fitting.set_busy(True, full=True)
    assert fitting.full.text() == "Running Full Fit…"
    assert fitting.full.property("running") is True
    assert fitting.full.icon().isNull()
    assert fitting.cancel.isEnabled()
    fitting.set_busy(False)
    assert fitting.preview.text() == "Preview Fit"
    assert fitting.full.text() == "Run Full Fit"
    assert not fitting.cancel.isEnabled()


def test_fitting_catalog_and_observation_have_separate_spacious_cards(qapp) -> None:
    fitting = FittingPage()
    fitting.resize(1000, 900)
    fitting.catalog_source.setText("ExoClock · matched to the project coordinates")
    fitting.observation_source.setText(
        "Cousins R (COUSINS_R) · 30 s exposures · detected from science FITS · "
        "observer location not set, so Preview Fit will detrend against time"
    )
    fitting.show()
    qapp.processEvents()

    catalog_card = fitting.catalog_source.parentWidget()
    observation_card = fitting.observation_source.parentWidget()
    assert catalog_card.objectName() == "fittingMetadataCard"
    assert observation_card.objectName() == "fittingMetadataCard"
    assert catalog_card is not observation_card
    assert catalog_card.geometry().bottom() < observation_card.geometry().top()
    assert fitting.catalog_source.height() >= 36
    assert fitting.observation_source.height() >= 54
    fitting.close()


def test_real_project_replaces_demo_target_and_restores_counts(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:06.99"
    project.manifest.target_dec = "+37:32:46.15"
    project.manifest.raw_files["bias"] = [f"bias_{index:03d}.fits" for index in range(21)]
    project.manifest.raw_files["science"] = [f"TrES-3_{index:03d}.fits" for index in range(4)]
    project.save()
    reduction = project.outputs_dir / "reduction"
    reduction.mkdir()
    frame = reduction / "r_00001_TrES-3.fits"
    fits.writeto(frame, np.arange(64 * 64, dtype=np.float32).reshape(64, 64))

    window = MainWindow(demo=True)
    window.set_project(project)
    qapp.processEvents()

    assert window.plate_page.inspector.target_name.text() == "TrES-3"
    assert window.plate_page.workspace.filename.text() == "FITS: r_00001_TrES-3.fits"
    assert window.plate_page.workspace.image.data is not None
    assert window.data_page.assignment_cards["bias"].count.text() == "21 selected"
    assert window.data_page.assignment_cards["science"].count.text() == "4 selected"
    assert not window.data_page.project_actions.isHidden()
    assert window.data_page.reveal_project.isEnabled()
    assert window.data_page.reset_project.isEnabled()
    window.close()


def test_open_existing_project_accepts_run_or_leaps_folder_and_opens_eclipse(qapp, tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    root = tmp_path / "WASP-18_TESS"
    project = ProjectWorkspace.create(root, "WASP-18 b — TESS")
    project.manifest.target_name = "WASP-18"
    project.manifest.stages[StageID.FITTING.value] = StageState(
        status=StageStatus.COMPLETE, summary="Imported primary-transit fit"
    )
    project.save()
    fitting = project.outputs_dir / StageID.FITTING.value
    fitting.mkdir()
    (fitting / "fit-summary.json").write_text(
        json.dumps(
            {
                "passband": "TESS",
                "light_curve": "aperture",
                "parameters": {
                    "name": "WASP-18 b",
                    "ra": "01:37:25.07",
                    "dec": "-45:40:40.10",
                    "period": 0.941452,
                    "mid_time": 2458354.458,
                    "rp_over_rs": 0.1018,
                    "sma_over_rs": 3.48,
                    "inclination": 83.5,
                    "eccentricity": 0.0,
                    "periastron": 0.0,
                    "metallicity": 0.0,
                    "temperature": 6432.0,
                    "logg": 4.31,
                    "source": "Test",
                    "source_date": "",
                },
                "fitted_ephemeris": {"period": 0.941452, "mid_time": 2458354.458},
            }
        ),
        encoding="utf-8",
    )

    window = MainWindow(settings=settings)
    window.open_existing_project(root / "LEAPS")
    qapp.processEvents()

    assert window.project is not None
    assert window.project.root == root
    assert window.stack.currentWidget() is window.secondary_eclipse_page
    assert window.data_page.open_existing_project.text() == "Open project"
    window.close()


def test_project_reset_dialog_shows_scope_and_requires_exact_project_name(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.raw_files["science"] = ["light_001.fits", "light_002.fits"]
    project.save()
    dialog = ProjectResetDialog(project)
    dialog.show()
    qapp.processEvents()

    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert str(project.workspace) in labels
    assert "Raw files preserved: 2" in labels
    assert not dialog.reset_button.isEnabled()
    dialog.confirmation.setText("wrong project")
    assert not dialog.reset_button.isEnabled()
    dialog.confirmation.setText("TrES-3")
    assert dialog.reset_button.isEnabled()
    dialog.close()


def test_project_reset_state_clears_only_recent_project_and_current_ui(qapp, tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("setup/complete", True)
    settings.setValue("window/geometry", b"kept")
    project = ProjectWorkspace.create(tmp_path / "run", "TrES-3")
    window = MainWindow(settings=settings)
    window.set_project(project)
    assert settings.value("projects/recent") == str(project.root)

    window._clear_current_project(project.root)

    assert not settings.contains("projects/recent")
    assert settings.value("setup/complete", type=bool) is True
    assert settings.value("window/geometry") == b"kept"
    assert window.project is None
    assert window.data_page.folder.text() == ""
    assert window.data_page.name.text() == ""
    assert window.data_page.project_actions.isHidden()
    assert window.stack.currentWidget() is window.data_page
    window.close()


def test_project_reset_is_disabled_while_processing(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    window = MainWindow(demo=True)
    window.set_project(project)

    window.runner.busyChanged.emit(True)
    assert not window.data_page.reset_project.isEnabled()
    assert window.data_page.reveal_project.isEnabled()
    window.runner.busyChanged.emit(False)
    assert window.data_page.reset_project.isEnabled()
    window.close()


def test_photometry_comparisons_can_be_reviewed_before_run(qapp, tmp_path) -> None:
    page = PlateSolvePage(tmp_path / "missing-preview.png")
    page.clear_selection()
    page.set_target(12.0, 14.0, radius=8.0, label="TrES-3", verified=False)
    page.add_comparison(24.0, 26.0, radius=8.0)
    page.add_comparison(30.0, 32.0, radius=8.0)
    page.set_comparison_active(0, False)

    selected: list[list[tuple[float, float]]] = []
    page.runRequested.connect(lambda comparisons, _radius: selected.append(comparisons))
    page.inspector.run.click()
    qapp.processEvents()

    assert page.inspector.comparison_selection.text() == "Comparisons: 1 active · 2 selected"
    assert page.workspace.image.marker_items["comparison-1"][0].opacity() < 0.5
    assert selected == [[(30.0, 32.0)]]
    page.close()


def test_photometry_progress_filename_does_not_resize_inspector(qapp) -> None:
    window = MainWindow(demo=True)
    window.show()
    qapp.processEvents()
    inspector = window.plate_page.inspector

    window._stage_event(
        StageEvent(
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            "Measured r_00008_short.fits",
            8,
            354,
        )
    )
    qapp.processEvents()
    first_width = inspector.width()
    window._stage_event(
        StageEvent(
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            "Measured r_00128_TrES-3_Cousins_R_extremely_long_filename.fits",
            128,
            354,
        )
    )
    qapp.processEvents()

    assert inspector.banner_title.text() == "Measuring frame 128 of 354"
    assert inspector.banner_title.toolTip().endswith("extremely_long_filename.fits")
    assert inspector.width() == first_width
    window.close()


def test_data_setup_orders_run_first_populates_target_and_highlights_errors(qapp) -> None:
    page = DataTargetPage()
    assert page.content_layout.itemAt(0).widget() is page.folder_card
    assert page.content_layout.itemAt(1).widget() is page.target_card
    assert page.content_layout.itemAt(2).widget() is page.frames_card

    record = FrameRecord(
        "TrES-3_Cousins_R_001.fits",
        "science",
        0.98,
        "",
        (2048, 2048),
        16,
        30.0,
        "checksum",
        "TrES-3 b",
        "17:52:07.00",
        "+37:32:46.20",
    )
    page.populate_target_from_records([record])
    assert page.name.text() == "TrES-3 b"
    assert page.ra.text() == "17:52:07.00"
    assert page.dec.text() == "+37:32:46.20"
    assert "Detected from FITS header" in page.target_source.text()

    page.show_error("Coordinates need attention", "target")
    qapp.processEvents()
    assert page.target_card.property("validationError") is True
    assert page.folder_card.property("validationError") is False
    page.clear_section_errors()
    assert page.target_card.property("validationError") is False
