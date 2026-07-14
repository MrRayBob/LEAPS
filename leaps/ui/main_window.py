from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from leaps.catalog import PlanetCatalogResolver, PlanetParameters
from leaps.diagnostics import DiagnosticLogger
from leaps.exports import TransitExporter
from leaps.filters import normalize_filter
from leaps.fits_inventory import (
    FITSInventory,
    FrameRecord,
    summarize_observation_records,
    validate_coordinates,
)
from leaps.models import (
    LEAPSError,
    ProjectManifest,
    StageEvent,
    StageID,
    StageState,
    StageStatus,
    target_fingerprint,
)
from leaps.offline import OfflineDataManager, format_bytes
from leaps.project import ProjectWorkspace
from leaps.science import (
    AlignmentService,
    FittingService,
    InspectionService,
    LightCurveReviewService,
    PhotometryConfig,
    PhotometryService,
    PlateSolveService,
    ReductionConfig,
    ReductionService,
    SecondaryEclipseService,
    _normalize_mid_transit_time,
)
from leaps.targets import ResolvedTarget, TargetNameResolver
from leaps.tess import TessImportResult, TessImportService

from .pages import (
    ComparisonStarsPage,
    DataTargetPage,
    FittingPage,
    InspectionPage,
    LightCurvePage,
    ObservingPlannerPage,
    PlateSolvePage,
    ProcessingPage,
    ReportsPage,
    SecondaryEclipsePage,
    SimpleToolPage,
)
from .settings_dialog import SettingsDialog, default_offline_root
from .theme import COLORS
from .widgets import StageNavButton, ToolNavButton, icon
from .workers import TaskRunner

STAGE_LABELS = {
    StageID.DATA_TARGET: "Data & Target",
    StageID.REDUCTION: "Reduction",
    StageID.INSPECTION: "Inspection",
    StageID.ALIGNMENT: "Alignment",
    StageID.PHOTOMETRY: "Photometry",
    StageID.LIGHT_CURVE: "Light Curve",
    StageID.FITTING: "Fitting",
    StageID.SECONDARY_ECLIPSE: "Secondary Eclipse",
}


def _optional_float(value: object) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _project_observatory(
    project: ProjectWorkspace,
    observation: dict[str, Any] | None = None,
) -> tuple[str, float | None, float | None, str]:
    metadata = observation or project.manifest.settings.get("observation_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    profile = project.manifest.global_profile
    fits_name = str(metadata.get("observatory", "")).strip()
    fits_latitude = _optional_float(metadata.get("latitude"))
    fits_longitude = _optional_float(metadata.get("longitude"))
    profile_name = str(profile.get("observatory", "")).strip()
    profile_latitude = _optional_float(profile.get("latitude"))
    profile_longitude = _optional_float(profile.get("longitude"))
    if fits_latitude is not None and fits_longitude is not None:
        return (
            fits_name or profile_name,
            fits_latitude,
            fits_longitude,
            "science FITS",
        )
    if profile_latitude is not None and profile_longitude is not None:
        return (
            fits_name or profile_name,
            profile_latitude,
            profile_longitude,
            "Settings profile",
        )
    if fits_name:
        return fits_name, None, None, "science FITS"
    if profile_name:
        return profile_name, None, None, "Settings profile"
    return "", None, None, ""


def _manual_planet_parameters(project: ProjectWorkspace) -> PlanetParameters:
    """Build an editable fitting seed when no catalogue contains the target."""
    mid_time = 0.0
    depth = 0.0
    light_curve = project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt"
    try:
        data = np.loadtxt(light_curve, unpack=True)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("The approved light curve must contain time and flux columns")
        finite = np.isfinite(data[0]) & np.isfinite(data[1])
        times = np.asarray(data[0][finite], dtype=float)
        flux = np.asarray(data[1][finite], dtype=float)
        if times.size < 3:
            raise ValueError("The approved light curve contains too few finite rows")
        baseline = float(np.median(flux))
        lower_flux = float(np.percentile(flux, 10))
        if not np.isfinite(baseline) or baseline == 0:
            raise ValueError("The approved light curve has no finite baseline")
        mid_time = float(np.median(times))
        depth = float(np.clip((baseline - lower_flux) / abs(baseline), 0.00001, 0.99))
    except (OSError, ValueError, TypeError):
        # The fitting action retains its existing typed light-curve validation. Keeping
        # zeroes here lets the page open and ask for manual values instead of failing
        # during catalogue setup.
        pass

    return PlanetParameters(
        name=project.manifest.target_name.strip()
        or project.manifest.name.strip()
        or "Uncatalogued target",
        ra=project.manifest.target_ra,
        dec=project.manifest.target_dec,
        period=0.0,
        mid_time=mid_time,
        rp_over_rs=depth**0.5,
        sma_over_rs=10.0,
        inclination=90.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.0,
        temperature=5500.0,
        logg=4.5,
        source="Manual / uncatalogued",
        is_manual=True,
    )


def _parameters_with_values(
    parameters: PlanetParameters, values: dict[str, Any]
) -> PlanetParameters:
    name = str(values.get("planet", "")).strip() if parameters.is_manual else parameters.name
    updates: dict[str, Any] = {
        "name": name or parameters.name,
        "period": float(values.get("period", parameters.period)),
        "mid_time": float(values.get("mid_time", parameters.mid_time)),
        "rp_over_rs": max(float(values.get("depth", parameters.rp_over_rs**2)), 0.0) ** 0.5,
    }
    if parameters.is_manual:
        updates.update(
            {
                "sma_over_rs": float(values.get("sma_over_rs", parameters.sma_over_rs)),
                "inclination": float(values.get("inclination", parameters.inclination)),
                "eccentricity": float(values.get("eccentricity", parameters.eccentricity)),
                "periastron": float(values.get("periastron", parameters.periastron)),
                "metallicity": float(values.get("metallicity", parameters.metallicity)),
                "temperature": float(values.get("temperature", parameters.temperature)),
                "logg": float(values.get("logg", parameters.logg)),
            }
        )
    return replace(parameters, **updates)


def _start_detached(program: str, arguments: list[str]) -> bool:
    result = QProcess.startDetached(program, arguments)
    return bool(result[0] if isinstance(result, tuple) else result)


def _reveal_in_file_manager(path: Path) -> None:
    if sys.platform == "darwin":
        if not _start_detached("/usr/bin/open", ["-R", str(path)]):
            raise OSError("Finder could not reveal the preview image")
        return
    if sys.platform == "win32":
        if not _start_detached("explorer.exe", ["/select,", str(path)]):
            raise OSError("File Explorer could not reveal the preview image")
        return
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent))):
        raise OSError("The file manager could not open the preview folder")


class ProjectResetDialog(QDialog):
    def __init__(self, project: ProjectWorkspace, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reset LEAPS Project Data")
        self.setModal(True)
        self.resize(610, 340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(14)
        title = QLabel("Remove generated project data?")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)
        summary = QLabel(
            "This removes the LEAPS project manifest, logs, caches, checkpoints, and generated "
            "outputs. Raw FITS and calibration frames remain untouched."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        details = QLabel(
            f"Project folder: {project.workspace}\n"
            f"Generated storage: {format_bytes(project.workspace_size())}\n"
            f"Raw files preserved: {sum(len(paths) for paths in project.manifest.raw_files.values())}"
        )
        details.setObjectName("muted")
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.setWordWrap(True)
        layout.addWidget(details)
        prompt = QLabel(f'Type “{project.manifest.name}” to confirm:')
        layout.addWidget(prompt)
        self.confirmation = QLineEdit()
        self.confirmation.setAccessibleName("Project name confirmation")
        layout.addWidget(self.confirmation)
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.reset_button = buttons.addButton("Reset Project Data", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.reset_button.setProperty("danger", True)
        self.reset_button.setEnabled(False)
        buttons.rejected.connect(self.reject)
        self.reset_button.clicked.connect(self.accept)
        self.confirmation.textChanged.connect(
            lambda value: self.reset_button.setEnabled(value == project.manifest.name)
        )
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    projectChanged = Signal(object)
    offlineProgress = Signal(str, object, object)

    def __init__(self, *, demo: bool = False, settings: QSettings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LEAPS — Exoplanet Transit Analysis")
        self.setMinimumSize(1120, 720)
        self.resize(1440, 960)
        self.demo = demo
        self.settings = settings if settings is not None else QSettings()
        self.project: ProjectWorkspace | None = None
        self.logger: DiagnosticLogger | None = None
        self.records: list[FrameRecord] = []
        self.last_failure: LEAPSError | None = None
        self._resetting_project = False
        self._inspection_previous_state: StageState | None = None
        self.runner = TaskRunner(self)
        self.offline_manager = OfflineDataManager(default_offline_root())
        self.target_lookup_runner = TaskRunner(self)
        self.fitting_lookup_runner = TaskRunner(self)
        self.target_resolver = TargetNameResolver(
            cache_path=self.offline_manager.root / "target-name-cache.json",
            nasa_snapshot=self._nasa_snapshot_path(),
        )
        self.target_lookup_timeout = QTimer(self)
        self.target_lookup_timeout.setSingleShot(True)
        self.target_lookup_timeout.setInterval(10_000)
        self.target_lookup_timeout.timeout.connect(self._target_name_lookup_timed_out)
        self._active_target_lookup_name = ""
        self._timed_out_target_lookups: set[str] = set()
        self._build_ui()
        self._connect()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(30_000)
        self.autosave_timer.timeout.connect(self.autosave_project)
        self.autosave_timer.start()
        geometry = self.settings.value("window/geometry")
        if geometry and not demo:
            self.restoreGeometry(geometry)
        if demo:
            self.resize(1487, 1018)
            self._load_demo_state()
        else:
            recent = self.settings.value("projects/recent", "")
            if recent and ProjectWorkspace.has_workspace(recent):
                try:
                    self.set_project(ProjectWorkspace.open(recent))
                except BaseException as exc:
                    QTimer.singleShot(0, lambda error=exc: self._handle_error(error))

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self.sidebar = self._build_sidebar()
        main.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        shell_layout.addLayout(main, 1)
        shell_layout.addWidget(self._build_status_bar())
        self.setCentralWidget(shell)

        asset = Path(__file__).resolve().parents[1] / "assets" / "demo-starfield.png"
        self.pages: dict[str | StageID, QWidget] = {}
        self.data_page = DataTargetPage()
        self.reduction_page = ProcessingPage(
            StageID.REDUCTION,
            "Reduction",
            "Apply bias, dark, and flat calibration while keeping every raw FITS frame read-only.",
            [
                (
                    "Median-combine calibration frames",
                    "Reject isolated cosmic rays when building master bias, dark, and flat frames.",
                ),
                (
                    "Preserve last successful reduction",
                    "New outputs replace the previous reduction only after the complete run succeeds.",
                ),
            ],
        )
        self.inspection_page = InspectionPage(asset)
        self.alignment_page = ProcessingPage(
            StageID.ALIGNMENT,
            "Alignment",
            "Register the reduced sequence against a stable reference frame.",
            [
                (
                    "Use the first accepted frame as reference",
                    "Match subsequent star fields to the first non-excluded reduced frame.",
                ),
                (
                    "Checkpoint each frame",
                    "Allow a safely cancelled run to resume from the latest verified result.",
                ),
            ],
        )
        self.plate_page = PlateSolvePage(asset)
        self.light_curve_page = LightCurvePage()
        self.fitting_page = FittingPage()
        self.secondary_eclipse_page = SecondaryEclipsePage()
        for stage, page in (
            (StageID.DATA_TARGET, self.data_page),
            (StageID.REDUCTION, self.reduction_page),
            (StageID.INSPECTION, self.inspection_page),
            (StageID.ALIGNMENT, self.alignment_page),
            (StageID.PHOTOMETRY, self.plate_page),
            (StageID.LIGHT_CURVE, self.light_curve_page),
            (StageID.FITTING, self.fitting_page),
            (StageID.SECONDARY_ECLIPSE, self.secondary_eclipse_page),
        ):
            self.pages[stage] = page
            self.stack.addWidget(page)
        self.comparison_page = ComparisonStarsPage()
        self.pages["apertures"] = self.comparison_page
        self.stack.addWidget(self.comparison_page)
        for key, title, subtitle, icon_name in (
            (
                "diagnostics",
                "Diagnostics",
                "Export a redacted support ZIP with logs, versions, settings, stage summaries, and sanitized FITS headers.",
                "fa6s.stethoscope",
            ),
        ):
            page = SimpleToolPage(title, subtitle, icon_name)
            self.pages[key] = page
            self.stack.addWidget(page)
        self.reports_page = ReportsPage()
        self.pages["reports"] = self.reports_page
        self.stack.addWidget(self.reports_page)
        planner = ObservingPlannerPage()
        self.pages["planner"] = planner
        self.stack.addWidget(planner)
        self.stack.setCurrentWidget(self.data_page)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(265)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 18, 0, 8)
        layout.setSpacing(0)
        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(20, 2, 16, 16)
        mark = QLabel()
        mark.setPixmap(
            QPixmap(str(Path(__file__).resolve().parents[1] / "assets" / "leaps-mark.png")).scaled(
                48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        brand_layout.addWidget(mark)
        word = QVBoxLayout()
        word.setSpacing(0)
        name = QLabel("LEAPS")
        name.setStyleSheet("font-size: 25px; font-weight: 750; letter-spacing: 1px;")
        descriptor = QLabel("Exoplanet Transit Analysis")
        descriptor.setStyleSheet(f"color: {COLORS['cyan']}; font-size: 10px;")
        word.addWidget(name)
        word.addWidget(descriptor)
        brand_layout.addLayout(word, 1)
        layout.addWidget(brand)
        workflow = QLabel("WORKFLOW")
        workflow.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 11px; font-weight: 650; padding: 10px 19px 6px;"
        )
        layout.addWidget(workflow)
        self.stage_buttons: dict[StageID, StageNavButton] = {}
        for stage in StageID:
            button = StageNavButton(stage, STAGE_LABELS[stage])
            self.stage_buttons[stage] = button
            layout.addWidget(button)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {COLORS['border_soft']}; margin: 8px 20px;")
        layout.addWidget(divider)
        tools_label = QLabel("TOOLS")
        tools_label.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 11px; font-weight: 650; padding: 4px 19px 6px;"
        )
        layout.addWidget(tools_label)
        self.tool_buttons: dict[str, ToolNavButton] = {}
        for key, label, icon_name in (
            ("diagnostics", "Diagnostics", "fa6s.stethoscope"),
            ("reports", "Reports", "fa6s.file-lines"),
            ("planner", "Observing Planner", "fa6s.moon"),
            ("settings", "Settings", "fa6s.gear"),
        ):
            button = ToolNavButton(label, icon_name)
            self.tool_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch()
        return sidebar

    def _build_status_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("statusBar")
        frame.setFixedHeight(68)
        frame.setStyleSheet(
            "QFrame#statusBar { background: #081725; border: 0; border-top: 1px solid #000000; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("font-weight: 600;")
        self.autosave = QLabel("No project open")
        self.autosave.setObjectName("muted")
        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_text)
        layout.addWidget(QLabel("·"))
        layout.addWidget(self.autosave)
        layout.addStretch()
        self.project_label = QLabel("")
        self.project_label.setObjectName("muted")
        layout.addWidget(self.project_label)
        return frame

    def _connect(self) -> None:
        for stage, button in self.stage_buttons.items():
            button.clicked.connect(self.open_stage)
        for key, button in self.tool_buttons.items():
            if key == "settings":
                button.clicked.connect(self.open_settings)
            elif key == "diagnostics":
                button.clicked.connect(self.export_diagnostics)
            else:
                button.clicked.connect(lambda checked=False, page_key=key: self.open_tool(page_key))
        self.data_page.scanRequested.connect(self.scan_folder)
        self.data_page.saveRequested.connect(self.save_data_target)
        self.data_page.targetLookupRequested.connect(self.resolve_target_name)
        self.data_page.openProjectRequested.connect(self.open_existing_project)
        self.data_page.tessImportRequested.connect(self.import_tess_light_curves)
        self.data_page.revealProjectRequested.connect(self.open_project_folder)
        self.data_page.resetProjectRequested.connect(self.request_project_reset)
        self.runner.busyChanged.connect(self._runner_busy_changed)
        self.fitting_lookup_runner.busyChanged.connect(self._runner_busy_changed)
        for page in (self.reduction_page, self.alignment_page):
            page.runRequested.connect(self.run_stage)
            page.cancelRequested.connect(self.runner.cancel)
        self.inspection_page.runRequested.connect(self.run_inspection)
        self.inspection_page.cancelRequested.connect(self.runner.cancel)
        self.inspection_page.draftChanged.connect(self.save_inspection_draft)
        self.inspection_page.confirmRequested.connect(self.confirm_inspection)
        self.plate_page.retryRequested.connect(self.retry_plate_solve)
        self.plate_page.copyDiagnosticsRequested.connect(self.copy_diagnostics)
        self.plate_page.starSelectionRequested.connect(self.select_photometry_star)
        self.plate_page.rankRequested.connect(self.rank_comparison_stars)
        self.plate_page.runRequested.connect(self.run_photometry)
        self.plate_page.inspector.cancelRequested.connect(self.runner.cancel)
        self.plate_page.selectionChanged.connect(self._save_photometry_selection)
        self.plate_page.pixelScaleChanged.connect(self._pixel_scale_changed)
        self.comparison_page.rankRequested.connect(self.rank_comparison_stars)
        self.comparison_page.runRequested.connect(self.run_photometry)
        self.comparison_page.cancelRequested.connect(self.runner.cancel)
        self.light_curve_page.selectionChanged.connect(self.review_light_curves)
        self.light_curve_page.continueRequested.connect(self.confirm_light_curve_review)
        self.fitting_page.previewRequested.connect(lambda values: self.run_fitting(values, full=False))
        self.fitting_page.fullFitRequested.connect(lambda values: self.run_fitting(values, full=True))
        self.fitting_page.cancelRequested.connect(self.cancel_fitting)
        self.fitting_page.viewInFilesRequested.connect(self.view_fit_preview_in_files)
        self.fitting_page.planetSearchRequested.connect(
            lambda name: self.prepare_fitting_setup(force=True, requested_name=name)
        )
        self.secondary_eclipse_page.analyzeRequested.connect(self.run_secondary_eclipse)
        self.secondary_eclipse_page.cancelRequested.connect(self.cancel_secondary_eclipse)
        self.secondary_eclipse_page.viewInFilesRequested.connect(self.view_secondary_eclipse_in_files)
        self.reports_page.openFolderRequested.connect(self.open_outputs_folder)
        self.reports_page.exportExoClockRequested.connect(lambda: self.export_transit("exoclock"))
        self.reports_page.exportETDRequested.connect(lambda: self.export_transit("etd"))
        self.offlineProgress.connect(self._offline_progress)

    def _load_demo_state(self) -> None:
        manifest = ProjectManifest(
            name="WTS-2 b · 2026-06-28",
            target_name="WTS-2 b",
            target_ra="19:34:55.87",
            target_dec="+36:48:55.79",
        )
        summaries = {
            StageID.DATA_TARGET: "Target selected",
            StageID.REDUCTION: "Calibrated",
            StageID.INSPECTION: "Looks good",
            StageID.ALIGNMENT: "Solution found",
        }
        for stage, summary in summaries.items():
            manifest.stages[stage.value].status = StageStatus.COMPLETE
            manifest.stages[stage.value].summary = summary
        manifest.stages[StageID.PHOTOMETRY.value].status = StageStatus.READY
        manifest.stages[StageID.PHOTOMETRY.value].summary = "Plate solve"
        manifest.stages[StageID.FITTING.value].status = StageStatus.LOCKED
        manifest.stages[StageID.FITTING.value].summary = "Locked"
        self._apply_manifest(manifest)
        self.open_stage(StageID.PHOTOMETRY)
        self.status_text.setText("Session saved")
        self.autosave.setText("autosave 1m ago")
        self.project_label.setText("WTS-2 b · 200 science frames")

    def set_project(self, project: ProjectWorkspace) -> None:
        self._recover_interrupted_fitting(project)
        self._recover_interrupted_secondary_eclipse(project)
        self.project = project
        self.logger = DiagnosticLogger(project)
        self.fitting_page.reset_setup("Open Fitting to load the selected target and FITS metadata.")
        self.secondary_eclipse_page.reset_setup(
            "Run a full primary-transit fit to load the eclipse ephemeris."
        )
        photometry_stage = project.manifest.stages[StageID.PHOTOMETRY.value]
        if (
            photometry_stage.status == StageStatus.NEEDS_ATTENTION
            and not (project.outputs_dir / StageID.PHOTOMETRY.value).exists()
        ):
            photometry_stage.status = StageStatus.READY
            photometry_stage.summary = "Select target and comparisons"
        light_curve_stage = project.manifest.stages[StageID.LIGHT_CURVE.value]
        if (
            photometry_stage.status == StageStatus.COMPLETE
            and light_curve_stage.status == StageStatus.LOCKED
        ):
            light_curve_stage.status = StageStatus.READY
            light_curve_stage.summary = "Review comparison stars"
            project.save()
        if not self.demo:
            self.settings.setValue("projects/recent", str(project.root))
        self.data_page.folder.setText(str(project.root))
        self.data_page.name.setText(project.manifest.target_name)
        self.data_page.ra.setText(project.manifest.target_ra)
        self.data_page.dec.setText(project.manifest.target_dec)
        self.data_page.mark_current_coordinates_as_saved()
        self.data_page.restore_project_assignments(
            project.manifest.raw_files,
            project.manifest.settings.get("frame_classifiers", {}),
            project.manifest.settings.get("calibration_waivers", {}),
        )
        pixel_scale = float(project.manifest.settings.get("pixel_scale", 0.0))
        self.plate_page.clear_selection()
        reference_frame = self._photometry_reference_frame(project, require_alignment=False)
        estimated_pixel_scale = 0.0
        if reference_frame:
            estimated_pixel_scale = self.plate_page.workspace.load_fits(
                reference_frame, pixel_scale
            )
        self.plate_page.inspector.set_project_target(
            project.manifest.target_name or "Unnamed target",
            f"{project.manifest.target_ra}  {project.manifest.target_dec}",
            pixel_scale,
            estimated_pixel_scale,
        )
        fingerprint = target_fingerprint(project.manifest.target_ra, project.manifest.target_dec)
        saved_photometry = project.manifest.settings.get("photometry", {})
        if saved_photometry.get("target_fingerprint") == fingerprint:
            self.plate_page.inspector.apply_photometry_config(
                saved_photometry.get("config", saved_photometry)
            )
            target = saved_photometry.get("target")
            radius = float(saved_photometry.get("aperture_radius", 8.0))
            if target:
                self.plate_page.set_target(
                    float(target[0]),
                    float(target[1]),
                    radius=radius,
                    label=project.manifest.target_name or "Target",
                    verified=bool(saved_photometry.get("verified", False)),
                )
            active_comparisons = saved_photometry.get("comparison_active", [])
            for index, comparison in enumerate(saved_photometry.get("comparisons", [])):
                self.plate_page.add_comparison(
                    float(comparison[0]),
                    float(comparison[1]),
                    radius=radius,
                    active=(
                        bool(active_comparisons[index])
                        if index < len(active_comparisons)
                        else True
                    ),
                )
        solution = project.manifest.settings.get("plate_solution", {})
        if solution and solution.get("target_fingerprint") != fingerprint:
            project.manifest.settings.pop("plate_solution", None)
            solution = {}
        if (
            self.plate_page.target is None
            and solution.get("target_fingerprint") == fingerprint
            and solution.get("target_xy")
        ):
            target = solution["target_xy"]
            self.plate_page.set_target(
                float(target[0]),
                float(target[1]),
                radius=8.0,
                label=project.manifest.target_name or "Target",
                verified=not bool(solution.get("unverified", False)),
            )
        self._load_inspection_page(project)
        self._apply_manifest(project.manifest)
        self.project_label.setText(project.manifest.name)
        self.status_text.setText("Session saved")
        self.autosave.setText("autosaved just now")
        self.data_page.set_project_actions_available(
            True, busy=self.runner.current is not None or self._resetting_project
        )
        try:
            import astropy.units as units
            from astropy.coordinates import SkyCoord

            coordinate = SkyCoord(
                project.manifest.target_ra,
                project.manifest.target_dec,
                unit=(units.hourangle, units.deg),
            )
            self.offline_manager.add_gaia_region(coordinate.ra.deg, coordinate.dec.deg, 0.5)
        except Exception:
            pass
        self.projectChanged.emit(project)

    @staticmethod
    def _recover_interrupted_fitting(project: ProjectWorkspace) -> None:
        state = project.manifest.stages[StageID.FITTING.value]
        if state.status != StageStatus.RUNNING:
            return
        try:
            project.discard_pending_transaction(StageID.FITTING)
        except OSError as exc:
            raise LEAPSError(
                "FITTING_RECOVERY_FAILED",
                "The interrupted fit could not be cleaned up",
                "LEAPS left the previous successful fitting output unchanged.",
                ["Check access to the LEAPS temporary folder", "Retry opening the project"],
                stage=StageID.FITTING,
                technical_details=str(exc),
            ) from exc
        project.set_stage(
            StageID.FITTING,
            StageStatus.READY,
            "Interrupted · ready to run again",
            progress=0.0,
            checkpoint="interrupted",
        )
        if "FITTING_INTERRUPTED" not in state.warning_codes:
            state.warning_codes.append("FITTING_INTERRUPTED")
            project.save()

    @staticmethod
    def _recover_interrupted_secondary_eclipse(project: ProjectWorkspace) -> None:
        state = project.manifest.stages[StageID.SECONDARY_ECLIPSE.value]
        if state.status != StageStatus.RUNNING:
            return
        try:
            project.discard_pending_transaction(StageID.SECONDARY_ECLIPSE)
        except OSError as exc:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_RECOVERY_FAILED",
                "The interrupted eclipse analysis could not be cleaned up",
                "LEAPS left the previous successful secondary-eclipse result unchanged.",
                ["Check access to the LEAPS temporary folder", "Retry opening the project"],
                stage=StageID.SECONDARY_ECLIPSE,
                technical_details=str(exc),
            ) from exc
        project.set_stage(
            StageID.SECONDARY_ECLIPSE,
            StageStatus.READY,
            "Interrupted · ready to run again",
            progress=0.0,
            checkpoint="interrupted",
        )
        if "SECONDARY_ECLIPSE_INTERRUPTED" not in state.warning_codes:
            state.warning_codes.append("SECONDARY_ECLIPSE_INTERRUPTED")
            project.save()

    def _apply_manifest(self, manifest: ProjectManifest) -> None:
        for stage, button in self.stage_buttons.items():
            button.update_state(manifest.stages[stage.value])

    def open_stage(self, stage: StageID) -> None:
        self.stack.setCurrentWidget(self.pages[stage])
        for key, button in self.stage_buttons.items():
            button.set_active(key == stage)
        if stage == StageID.INSPECTION and self.project:
            self._load_inspection_page(self.project)
        elif stage == StageID.LIGHT_CURVE and self.project:
            self.review_light_curves()
        elif stage == StageID.FITTING and self.project:
            self.prepare_fitting_setup()
        elif stage == StageID.SECONDARY_ECLIPSE and self.project:
            self.prepare_secondary_eclipse_setup()

    def _load_inspection_page(self, project: ProjectWorkspace) -> None:
        if isinstance(project.manifest.settings.get("tess_import"), dict):
            self.inspection_page.set_mission_state(
                "TESS mission quality flags were applied during import; individual reduced-frame inspection is not applicable."
            )
            return
        try:
            result = InspectionService.load(project, required=False)
        except LEAPSError as failure:
            self.inspection_page.set_failure(failure)
            return
        if result is None:
            self.inspection_page.set_empty()
            return
        self.inspection_page.set_result(
            result,
            project.outputs_dir / StageID.REDUCTION.value,
        )

    @staticmethod
    def _photometry_reference_frame(
        project: ProjectWorkspace,
        *,
        require_alignment: bool = True,
    ) -> Path | None:
        alignment_complete = (
            project.manifest.stages[StageID.ALIGNMENT.value].status
            == StageStatus.COMPLETE
        )
        if alignment_complete:
            return AlignmentService.successful_frames(project)[0]
        if require_alignment:
            raise LEAPSError(
                "PHOTOMETRY_ALIGNMENT_MISSING",
                "Complete Alignment before Photometry",
                "The current confirmed frame list has not been aligned yet.",
                ["Open Alignment"],
                stage=StageID.PHOTOMETRY,
            )
        try:
            return InspectionService.confirmed_frames(project)[0]
        except LEAPSError:
            reduced = sorted(
                (project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*")
            )
            return reduced[0] if reduced else None

    def open_tool(self, key: str) -> None:
        self.stack.setCurrentWidget(self.pages[key])
        for button in self.stage_buttons.values():
            button.set_active(False)

    def scan_folder(self, root: Path) -> None:
        if not self._ensure_runner_idle("scan the observing run", StageID.DATA_TARGET):
            return

        def scan(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            return FITSInventory(root).discover()

        self.status_text.setText("Scanning FITS headers…")
        self.runner.start(
            scan,
            result=self._scan_complete,
            error=self._handle_error,
            finished=self._scan_finished,
            operation="FITS header scan",
        )

    def open_existing_project(self, selected_folder: Path) -> None:
        """Open an already-created portable LEAPS project without rescanning raw FITS files."""
        if not self._ensure_runner_idle("open a LEAPS project", StageID.DATA_TARGET):
            return
        root = selected_folder.expanduser().resolve()
        if root.name in {ProjectWorkspace.WORKSPACE_NAME, ProjectWorkspace.LEGACY_WORKSPACE_NAME}:
            root = root.parent
        try:
            if not ProjectWorkspace.has_project(root):
                raise LEAPSError(
                    "PROJECT_NOT_FOUND",
                    "No LEAPS project was found there",
                    "Choose the observing-run folder that contains LEAPS/project.json, not a data subfolder.",
                    ["Choose Open project again", "Select the folder directly above LEAPS"],
                    stage=StageID.DATA_TARGET,
                )
            project = ProjectWorkspace.open(root)
            self.set_project(project)
            fitting = project.manifest.stages[StageID.FITTING.value]
            self.open_stage(
                StageID.SECONDARY_ECLIPSE
                if fitting.status == StageStatus.COMPLETE
                else StageID.DATA_TARGET
            )
        except BaseException as exc:
            self._handle_error(exc)

    def import_tess_light_curves(self, selected_files: list[Path]) -> None:
        """Import local TESS SPOC light-curve files and continue at primary fitting."""
        if not self._ensure_runner_idle("import TESS light curves", StageID.DATA_TARGET):
            return
        self.data_page.set_tess_import_busy(True)
        self.status_dot.setStyleSheet(f"color: {COLORS['cyan']};")
        self.status_text.setText("Importing TESS PDCSAP light curves…")

        def import_tess(*, emit=None, token=None):
            return TessImportService().run(selected_files, emit=emit, token=token)

        self.runner.start(
            import_tess,
            event=self._stage_event,
            result=self._tess_import_complete,
            error=self._tess_import_failed,
            finished=lambda: self.data_page.set_tess_import_busy(False),
            operation="TESS light-curve import",
        )

    def _tess_import_complete(self, result: TessImportResult) -> None:
        self.records = []
        self.set_project(result.project)
        sectors = ", ".join(str(sector) for sector in result.sectors) or "unknown sector"
        self.data_page.show_tess_import_result(
            f"Imported {result.imported_points:,} quality-filtered PDCSAP points from TESS sector(s) {sectors}. "
            "Raw TESS FITS files were not changed."
        )
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText("TESS light curve imported · choose a planet and preview the primary transit")
        self.autosave.setText("autosaved just now")
        self.open_stage(StageID.FITTING)

    def _tess_import_failed(self, exc: BaseException) -> None:
        self._handle_error(exc)

    def _nasa_snapshot_path(self) -> Path | None:
        folder = self.offline_manager.root / "nasa"
        marker = folder / "installed.json"
        try:
            if marker.exists():
                filename = json.loads(marker.read_text(encoding="utf-8")).get("filename")
                if filename and Path(filename).suffix.casefold() == ".json":
                    return folder / Path(filename).name
            return next(path for path in folder.glob("*.json") if path.name != "installed.json")
        except (OSError, StopIteration, ValueError, TypeError, json.JSONDecodeError):
            return None

    def resolve_target_name(self, name: str) -> None:
        if self.target_lookup_runner.current is not None:
            self.data_page.show_target_lookup_error(
                name, "Another target lookup is finishing. Press Enter to retry."
            )
            return

        self.target_resolver.nasa_snapshot = self._nasa_snapshot_path()

        def lookup(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            return self.target_resolver.resolve(name)

        lookup_key = name.strip().casefold()
        self._active_target_lookup_name = name
        self._timed_out_target_lookups.discard(lookup_key)
        self.target_lookup_timeout.start()
        self.status_text.setText(f"Looking up {name}…")
        self.target_lookup_runner.start(
            lookup,
            result=lambda resolved: self._target_name_resolved(name, resolved),
            error=lambda exc: self._target_name_lookup_failed(name, exc),
            finished=lambda: self._target_name_lookup_finished(name),
            inhibit_sleep=False,
        )

    def _target_name_resolved(self, requested_name: str, resolved: ResolvedTarget) -> None:
        if requested_name.strip().casefold() in self._timed_out_target_lookups:
            return
        self.target_lookup_timeout.stop()
        self.data_page.apply_target_resolution(requested_name, resolved)

    def _target_name_lookup_failed(self, requested_name: str, exc: BaseException) -> None:
        if requested_name.strip().casefold() in self._timed_out_target_lookups:
            return
        self.target_lookup_timeout.stop()
        failure = self._as_failure(exc, StageID.DATA_TARGET)
        self.data_page.show_target_lookup_error(requested_name, failure.message)

    def _target_name_lookup_timed_out(self) -> None:
        requested_name = self._active_target_lookup_name
        if not requested_name:
            return
        self._timed_out_target_lookups.add(requested_name.strip().casefold())
        self.target_lookup_runner.cancel()
        self.data_page.show_target_lookup_error(
            requested_name,
            f"No coordinates found for “{requested_name}”. Check the name or enter RA/DEC manually.",
        )
        self.status_text.setText("Ready")

    def _target_name_lookup_finished(self, requested_name: str) -> None:
        if self._active_target_lookup_name.strip().casefold() == requested_name.strip().casefold():
            self.target_lookup_timeout.stop()
            self._active_target_lookup_name = ""
        self.status_text.setText("Ready")

    def _scan_complete(self, records: list[FrameRecord]) -> None:
        self.records = records
        self.data_page.set_records(records)
        self.data_page.populate_target_from_records(records)

    def _scan_finished(self) -> None:
        self.data_page.scan_progress.setVisible(False)
        self.status_text.setText("Ready")

    def save_data_target(self, values: dict[str, Any]) -> None:
        self.data_page.clear_section_errors()
        try:
            if not values["root"]:
                raise LEAPSError(
                    "PROJECT_FOLDER_REQUIRED",
                    "Choose the observing run",
                    "Select the folder that contains the FITS images.",
                    ["Choose folder"],
                    stage=StageID.DATA_TARGET,
                )
            ra, dec = validate_coordinates(values["ra"], values["dec"])
            grouped = values["assignments"]
            if not grouped["science"]:
                raise LEAPSError(
                    "SCIENCE_FRAMES_REQUIRED",
                    "No science frames were assigned",
                    "Confirm at least one light/science exposure before continuing.",
                    ["Review frame assignments"],
                    stage=StageID.DATA_TARGET,
                )
            missing = [
                kind for kind in ("bias", "dark", "flat") if not grouped[kind] and not values["waivers"][kind]
            ]
            if missing and not self._confirm_missing_calibration_frames(missing):
                return
            for kind in ("bias", "dark", "flat"):
                values["waivers"][kind] = not grouped[kind]
            root = Path(values["root"])
            project = (
                ProjectWorkspace.open(root)
                if ProjectWorkspace.has_workspace(root)
                else ProjectWorkspace.create(root, values["target_name"] or root.name)
            )
            previous_fingerprint = target_fingerprint(
                project.manifest.target_ra, project.manifest.target_dec
            )
            previous_science = list(project.manifest.raw_files.get("science", []))
            next_fingerprint = target_fingerprint(ra, dec)
            project.manifest.target_name = values["target_name"]
            project.manifest.target_ra = ra
            project.manifest.target_dec = dec
            project.manifest.raw_files = grouped
            project.manifest.settings["calibration_waivers"] = values["waivers"]
            project.manifest.settings["frame_classifiers"] = values["frame_classifiers"]
            observation = summarize_observation_records(self.records, grouped["science"])
            if observation["science_frames_inspected"]:
                project.manifest.settings["observation_metadata"] = observation
                if observation["filter"]:
                    project.manifest.settings["filter"] = observation["filter"]
                else:
                    project.manifest.settings.pop("filter", None)
                if observation["exposure_time"]:
                    project.manifest.settings["exposure_time"] = observation["exposure_time"]
            if previous_science != list(grouped["science"]):
                project.manifest.settings.pop("fitting_setup", None)
            if previous_fingerprint != next_fingerprint:
                project.manifest.settings.pop("plate_solution", None)
                project.manifest.settings.pop("photometry", None)
                project.manifest.settings.pop("fitting_setup", None)
                alignment = project.manifest.stages[StageID.ALIGNMENT.value]
                project.manifest.stages[StageID.PHOTOMETRY.value] = StageState(
                    status=(
                        StageStatus.READY
                        if alignment.status == StageStatus.COMPLETE
                        else StageStatus.LOCKED
                    ),
                    summary=(
                        "Select target and comparisons"
                        if alignment.status == StageStatus.COMPLETE
                        else "Locked"
                    ),
                )
                project.manifest.stages[StageID.LIGHT_CURVE.value] = StageState()
                project.manifest.stages[StageID.FITTING.value] = StageState()
                project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState()
            project.set_stage(StageID.DATA_TARGET, StageStatus.COMPLETE, "Target selected", progress=1.0)
            self.set_project(project)
            self.open_stage(StageID.REDUCTION)
        except BaseException as exc:
            failure = self._as_failure(exc, StageID.DATA_TARGET)
            section = {
                "PROJECT_FOLDER_REQUIRED": "folder",
                "INVALID_COORDINATES": "target",
                "SCIENCE_FRAMES_REQUIRED": "frames",
                "CALIBRATION_CONFIRMATION_REQUIRED": "frames",
            }.get(failure.code)
            self.data_page.show_error(f"{failure.title}: {failure.message}", section)

    def _confirm_missing_calibration_frames(self, missing: list[str]) -> bool:
        labels = {"bias": "Bias", "dark": "Darks", "flat": "Flats"}
        names = ", ".join(labels[kind] for kind in missing)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Missing calibration frames")
        dialog.setText(f"No {names} frames are assigned.")
        dialog.setInformativeText(
            "Continuing without these calibration frames can reduce the quality of the reduction."
        )
        cancel = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        acknowledge = dialog.addButton("Acknowledge", QMessageBox.ButtonRole.AcceptRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        return dialog.clickedButton() is acknowledge

    def run_inspection(self) -> None:
        if not self.project:
            return
        if not self._ensure_runner_idle("run Inspection", StageID.INSPECTION):
            return
        self._inspection_previous_state = replace(
            self.project.manifest.stages[StageID.INSPECTION.value],
            warning_codes=list(
                self.project.manifest.stages[StageID.INSPECTION.value].warning_codes
            ),
        )
        self.inspection_page.set_busy(True)
        self.project.set_stage(
            StageID.INSPECTION,
            StageStatus.RUNNING,
            "Reading frame metrics",
            progress=0.0,
        )
        self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['cyan']};")
        self.status_text.setText("Running Inspection…")
        self.runner.start(
            InspectionService().run,
            self.project,
            event=self._stage_event,
            result=self._inspection_scan_complete,
            error=self._inspection_scan_failed,
            finished=lambda: self.inspection_page.set_busy(False),
            operation="Inspection",
        )

    def _inspection_scan_complete(self, result: Any) -> None:
        if not self.project:
            return
        previous_complete = bool(
            self._inspection_previous_state
            and self._inspection_previous_state.status == StageStatus.COMPLETE
        )
        if previous_complete and not result.confirmed:
            self._invalidate_after_inspection(None, None)
        status = StageStatus.COMPLETE if result.confirmed else StageStatus.READY
        summary = (
            f"{result.included_count} included · {result.excluded_count} excluded"
            if result.confirmed
            else f"Review {len(result.frames)} frames"
        )
        self.project.set_stage(
            StageID.INSPECTION,
            status,
            summary,
            progress=1.0,
            output_path=(
                self.project.outputs_dir
                / StageID.INSPECTION.value
                / "inspection.json"
            ),
        )
        self.inspection_page.set_result(
            result,
            self.project.outputs_dir / StageID.REDUCTION.value,
        )
        self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText("Inspection ready for review")
        self.autosave.setText("autosaved just now")

    def _inspection_scan_failed(self, exc: BaseException) -> None:
        if not self.project:
            return
        cancelled = isinstance(exc, LEAPSError) and exc.code == "JOB_CANCELLED"
        if self._inspection_previous_state is not None:
            self.project.manifest.stages[StageID.INSPECTION.value] = replace(
                self._inspection_previous_state,
                warning_codes=list(self._inspection_previous_state.warning_codes),
            )
            self.project.save()
        elif not cancelled:
            self.project.set_stage(
                StageID.INSPECTION,
                StageStatus.NEEDS_ATTENTION,
                "Needs attention",
            )
        self._apply_manifest(self.project.manifest)
        if cancelled:
            self.inspection_page.set_cancelled()
            self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
            self.status_text.setText("Inspection cancelled safely")
            return
        failure = self._as_failure(exc, StageID.INSPECTION)
        self.inspection_page.set_failure(failure)
        self._show_failure(failure)

    def save_inspection_draft(self, exclusions: dict[str, bool]) -> None:
        if not self.project:
            return
        try:
            InspectionService.save_draft(self.project, exclusions)
            self.autosave.setText("autosaved just now")
        except BaseException as exc:
            self._handle_error(exc)

    def confirm_inspection(self, exclusions: dict[str, bool]) -> None:
        if not self.project:
            return
        if not self._ensure_runner_idle(
            "confirm Inspection", StageID.INSPECTION
        ):
            return
        try:
            previous = InspectionService.load(
                self.project, include_draft=False, required=False
            )
            previous_exclusions = (
                {
                    str(record["file"]): bool(record.get("excluded"))
                    for record in previous.frames
                }
                if previous and previous.confirmed
                else {}
            )
            previous_reference = self._first_included_name(previous)
            result = InspectionService.confirm(self.project, exclusions)
            current_exclusions = {
                str(record["file"]): bool(record.get("excluded"))
                for record in result.frames
            }
            changed = bool(previous_exclusions) and previous_exclusions != current_exclusions
            if changed:
                self._invalidate_after_inspection(
                    previous_reference,
                    self._first_included_name(result),
                )
            self.project.set_stage(
                StageID.INSPECTION,
                StageStatus.COMPLETE,
                f"{result.included_count} included · {result.excluded_count} excluded",
                progress=1.0,
                output_path=(
                    self.project.outputs_dir
                    / StageID.INSPECTION.value
                    / "inspection.json"
                ),
            )
            self.inspection_page.set_result(
                result,
                self.project.outputs_dir / StageID.REDUCTION.value,
            )
            self._apply_manifest(self.project.manifest)
            self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
            self.status_text.setText("Frame inspection confirmed")
            self.autosave.setText("autosaved just now")
            self.open_stage(StageID.ALIGNMENT)
        except BaseException as exc:
            failure = self._as_failure(exc, StageID.INSPECTION)
            self.inspection_page.set_failure(failure)
            self._show_failure(failure)

    @staticmethod
    def _first_included_name(result: Any | None) -> str | None:
        if result is None:
            return None
        return next(
            (
                str(record.get("file"))
                for record in result.frames
                if not record.get("excluded")
            ),
            None,
        )

    def _invalidate_after_inspection(
        self,
        previous_reference: str | None,
        current_reference: str | None,
    ) -> None:
        if not self.project:
            return
        self.project.manifest.stages[StageID.ALIGNMENT.value] = StageState(
            status=StageStatus.READY,
            summary="Inspection changed · rerun alignment",
        )
        for stage in (
            StageID.PHOTOMETRY,
            StageID.LIGHT_CURVE,
            StageID.FITTING,
            StageID.SECONDARY_ECLIPSE,
        ):
            self.project.manifest.stages[stage.value] = StageState()
        self.project.manifest.settings.pop("light_curve_review", None)
        if previous_reference is None or previous_reference != current_reference:
            self.project.manifest.settings.pop("plate_solution", None)
            self.project.manifest.settings.pop("photometry", None)
            self.plate_page.clear_selection()
        self.project.save()

    def run_stage(self, stage: StageID) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "Choose and confirm an observing run before processing.",
                    ["Open Data & Target"],
                    stage=stage,
                )
            )
            return
        if not self._ensure_runner_idle(f"run {STAGE_LABELS[stage]}", stage):
            return
        page = self.pages[stage]
        assert isinstance(page, ProcessingPage)
        functions = {
            StageID.REDUCTION: (ReductionService().run, {"config": ReductionConfig()}),
            StageID.ALIGNMENT: (AlignmentService().run, {}),
        }
        function, kwargs = functions[stage]
        page.set_busy(True)
        self.project.set_stage(stage, StageStatus.RUNNING, "Processing", progress=0.0)
        self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['cyan']};")
        self.status_text.setText(f"Running {STAGE_LABELS[stage]}…")
        self.runner.start(
            function,
            self.project,
            event=self._stage_event,
            result=lambda result, current_stage=stage: self._stage_complete(current_stage, result),
            error=lambda exc, current_stage=stage: self._stage_failed(current_stage, exc),
            finished=lambda current_page=page: current_page.set_busy(False),
            operation=STAGE_LABELS[stage],
            **kwargs,
        )

    def _ensure_runner_idle(self, requested: str, stage: StageID | None = None) -> bool:
        if self.runner.current is not None:
            active = self.runner.current_operation or "another operation"
            self._show_failure(
                LEAPSError(
                    "OPERATION_IN_PROGRESS",
                    "Another operation is still running",
                    f"LEAPS is finishing {active}. It cannot {requested} at the same time.",
                    ["Wait for the current operation", "Cancel safely, then retry"],
                    stage=stage,
                )
            )
            return False
        return self._ensure_project_access(stage)

    def _ensure_project_access(self, stage: StageID | None) -> bool:
        if self.project is None or stage in {None, StageID.DATA_TARGET}:
            return True
        try:
            self.project.verify_process_access(stage)
        except LEAPSError as failure:
            self._request_project_access(failure)
            return False
        return True

    def _request_project_access(self, failure: LEAPSError) -> None:
        self.last_failure = failure
        self.status_dot.setStyleSheet(f"color: {COLORS['amber']};")
        self.status_text.setText("Project access required")
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Project access required")
        dialog.setText("LEAPS cannot read or write the current observing run.")
        if sys.platform == "darwin":
            dialog.setInformativeText(
                f"{failure.message}\n\n"
                "Choose the current observing-run folder again in the native macOS picker. "
                "That is the system-supported way for LEAPS to request access to an external "
                "SSD. macOS decides whether to display a permission popup; if it has already "
                "recorded a decision, open Privacy Settings and review Files and Folders."
            )
        else:
            dialog.setInformativeText(
                f"{failure.message}\n\n"
                "Choose the current observing-run folder again after confirming that your "
                "account can read the raw files and write to the LEAPS folder."
            )
        if failure.technical_details:
            dialog.setDetailedText(failure.technical_details)
        cancel = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        privacy = None
        if sys.platform == "darwin":
            privacy = dialog.addButton(
                "Open Privacy Settings", QMessageBox.ButtonRole.ActionRole
            )
        grant = dialog.addButton(
            "Choose Folder & Grant Access…", QMessageBox.ButtonRole.AcceptRole
        )
        dialog.setDefaultButton(grant)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if privacy is not None and clicked is privacy:
            QDesktopServices.openUrl(
                QUrl(
                    "x-apple.systempreferences:com.apple.preference.security?"
                    "Privacy_FilesAndFolders"
                )
            )
            return
        if clicked is not grant or self.project is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            "Grant access to the current observing run",
            str(self.project.root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        selected = Path(folder).expanduser().resolve()
        if selected.name in {
            ProjectWorkspace.WORKSPACE_NAME,
            ProjectWorkspace.LEGACY_WORKSPACE_NAME,
        }:
            selected = selected.parent
        if selected != self.project.root:
            self._show_failure(
                LEAPSError(
                    "PROJECT_FOLDER_MISMATCH",
                    "Choose the current observing run",
                    f"This project uses {self.project.root}. LEAPS did not switch to {selected}.",
                    ["Retry the process", "Choose the current observing-run folder"],
                    stage=failure.stage,
                )
            )
            return
        try:
            self.project.verify_process_access(failure.stage)
        except LEAPSError as retry_failure:
            self._show_failure(retry_failure)
            return
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText("Project access restored · retry the process")

    def _stage_event(self, event: StageEvent) -> None:
        page = self.pages[event.stage]
        if isinstance(page, ProcessingPage):
            page.update_event(event)
        elif isinstance(page, InspectionPage):
            page.update_event(event)
            self.status_text.setText(event.message)
        elif isinstance(page, FittingPage):
            page.update_event(event)
            self.status_text.setText(event.message)
        elif isinstance(page, SecondaryEclipsePage):
            page.update_event(event)
            self.status_text.setText(event.message)
        elif event.stage == StageID.PHOTOMETRY:
            title = event.message
            if event.total > 0:
                digits = len(str(event.total))
                title = f"Measuring frame {event.current:0{digits}d} of {event.total}"
            self.plate_page.inspector.banner_title.setText(title)
            self.plate_page.inspector.banner_title.setToolTip(event.message)
        if self.project:
            state = self.project.manifest.stages[event.stage.value]
            state.progress = event.fraction
            state.checkpoint = event.checkpoint
            state.summary = event.message
            if event.stage == StageID.FITTING and (
                event.current == 0
                or event.current == event.total
                or (event.current > 0 and event.current % 100 == 0)
            ):
                self.project.save()

    def _stage_complete(self, stage: StageID, result: Any) -> None:
        if self.project:
            self.project.set_stage(
                stage,
                StageStatus.COMPLETE,
                "Complete",
                progress=1.0,
                output_path=getattr(result, "__fspath__", lambda: None)(),
            )
            self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText(f"{STAGE_LABELS[stage]} complete")
        self.autosave.setText("autosaved just now")
        if stage == StageID.REDUCTION and self.project:
            self.project.manifest.stages[StageID.INSPECTION.value] = StageState(
                status=StageStatus.READY,
                summary="Run Inspection",
            )
            for downstream in (
                StageID.ALIGNMENT,
                StageID.PHOTOMETRY,
                StageID.LIGHT_CURVE,
                StageID.FITTING,
                StageID.SECONDARY_ECLIPSE,
            ):
                self.project.manifest.stages[downstream.value] = StageState()
            self.project.manifest.settings.pop("light_curve_review", None)
            self.project.save()
            self._apply_manifest(self.project.manifest)
            frames = sorted((self.project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
            if frames:
                pixel_scale = float(self.project.manifest.settings.get("pixel_scale", 0.0))
                estimated_pixel_scale = self.plate_page.workspace.load_fits(
                    frames[0], pixel_scale
                )
                self.plate_page.inspector.set_pixel_scale(pixel_scale, estimated_pixel_scale)
            self._load_inspection_page(self.project)
        elif stage == StageID.ALIGNMENT and self.project:
            reference = self._photometry_reference_frame(self.project)
            if reference:
                pixel_scale = float(self.project.manifest.settings.get("pixel_scale", 0.0))
                estimated = self.plate_page.workspace.load_fits(reference, pixel_scale)
                self.plate_page.inspector.set_pixel_scale(pixel_scale, estimated)
        elif stage == StageID.PHOTOMETRY:
            self.plate_page.inspector.banner_title.setText("Photometry complete")
            self.plate_page.inspector.banner_title.setStyleSheet(
                f"color: {COLORS['green']}; font-size: 16px; font-weight: 650;"
            )
            if self.project:
                self.open_stage(StageID.LIGHT_CURVE)

    def _stage_failed(self, stage: StageID, exc: BaseException) -> None:
        if isinstance(exc, LEAPSError) and exc.code == "JOB_CANCELLED":
            if self.logger:
                self.logger.record("cancelled", stage=stage, message=exc.message)
            if self.project:
                self.project.set_stage(
                    stage,
                    StageStatus.READY,
                    "Cancelled · ready to resume",
                    progress=0.0,
                    checkpoint="cancelled",
                )
                self._apply_manifest(self.project.manifest)
            page = self.pages[stage]
            if isinstance(page, ProcessingPage):
                page.set_cancelled()
            self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
            self.status_text.setText(f"{STAGE_LABELS[stage]} cancelled safely")
            return
        failure = self._as_failure(exc, stage)
        if self.project:
            self.project.set_stage(stage, StageStatus.NEEDS_ATTENTION, "Needs attention")
            self._apply_manifest(self.project.manifest)
        page = self.pages[stage]
        if isinstance(page, ProcessingPage):
            page.set_failure(failure)
        elif stage == StageID.PHOTOMETRY:
            self.plate_page.inspector.set_failure(failure)
        self._show_failure(failure)

    def retry_plate_solve(self) -> None:
        if not self.project:
            return
        if not self._ensure_runner_idle("retry plate solving", StageID.PHOTOMETRY):
            return
        try:
            frame = self._photometry_reference_frame(self.project)
        except LEAPSError as failure:
            self._handle_error(failure)
            return
        assert frame is not None
        self.runner.start(
            PlateSolveService().solve,
            frame,
            self.project.manifest.target_ra,
            self.project.manifest.target_dec,
            float(self.project.manifest.settings.get("pixel_scale", 0.0)),
            event=self._stage_event,
            result=self._plate_complete,
            error=self._plate_failed,
            operation="plate solving",
        )

    def _pixel_scale_changed(self, value: object) -> None:
        if not self.project:
            return
        if value is None:
            self.project.manifest.settings.pop("pixel_scale", None)
            self.plate_page.workspace.set_pixel_scale(0.0)
        else:
            try:
                pixel_scale = float(value)
            except (TypeError, ValueError):
                return
            if pixel_scale <= 0:
                return
            self.project.manifest.settings["pixel_scale"] = pixel_scale
            self.plate_page.workspace.set_pixel_scale(pixel_scale)
        self.project.save()
        self.autosave.setText("autosaved just now")

    def _plate_complete(self, result: Any) -> None:
        if self.project:
            solved_scale = (
                float(result.attempts[-1].pixel_scale)
                if getattr(result, "attempts", None)
                else float(self.project.manifest.settings.get("pixel_scale", 0.0))
            )
            if solved_scale > 0:
                self.project.manifest.settings["pixel_scale"] = solved_scale
                self.plate_page.inspector.set_pixel_scale(
                    solved_scale,
                    self.plate_page.workspace.estimated_pixel_scale,
                )
                self.plate_page.workspace.set_pixel_scale(solved_scale)
            self.project.manifest.settings["plate_solution"] = {
                "target_xy": result.target_xy,
                "identified_stars": result.identified_stars,
                "unverified": result.unverified,
                "wcs_header": result.wcs_header,
                "target_fingerprint": target_fingerprint(
                    self.project.manifest.target_ra, self.project.manifest.target_dec
                ),
            }
            self.project.set_stage(StageID.PHOTOMETRY, StageStatus.READY, "Plate solved", progress=0.2)
            self._apply_manifest(self.project.manifest)
            if result.target_xy:
                self.plate_page.set_target(
                    float(result.target_xy[0]),
                    float(result.target_xy[1]),
                    radius=self.plate_page.inspector.aperture.value(),
                    label=self.project.manifest.target_name or "Target",
                    verified=True,
                )
                self._save_photometry_selection(verified=True)
        self.plate_page.inspector.banner.setStyleSheet(
            f"background: {COLORS['surface_3']}; border-bottom: 1px solid {COLORS['border']};"
        )
        self.plate_page.inspector.banner_icon.setPixmap(
            icon("fa6s.circle-check", COLORS["green"]).pixmap(24, 24)
        )
        self.plate_page.inspector.banner_title.setText("Target located")
        self.plate_page.inspector.banner_title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 16px; font-weight: 650;"
        )
        self.plate_page.inspector.explanation.setText(
            "The project coordinates were placed in the real FITS image. Add comparison stars, then run HOPS photometry."
        )
        self.plate_page.inspector._restore_retry()

    def _plate_failed(self, exc: BaseException) -> None:
        failure = self._as_failure(exc, StageID.PHOTOMETRY)
        self.last_failure = failure
        self.plate_page.inspector.set_failure(failure)
        if self.project:
            self.project.set_stage(
                StageID.PHOTOMETRY,
                StageStatus.READY,
                "Manual selection available",
                progress=0.0,
            )
            self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['amber']};")
        self.status_text.setText("Plate solve unavailable · manual selection ready")

    def manual_target_placed(self, x: float, y: float) -> None:
        if self.project:
            self.project.manifest.settings["plate_solution"] = {
                "target_xy": [x, y],
                "unverified": True,
                "target_fingerprint": target_fingerprint(
                    self.project.manifest.target_ra, self.project.manifest.target_dec
                ),
            }
            if not any(
                warning.get("code") == "UNVERIFIED_WCS"
                for warning in self.project.manifest.warnings
            ):
                self.project.manifest.warnings.append(
                    {
                        "code": "UNVERIFIED_WCS",
                        "message": "Target was placed manually after plate solve failure.",
                    }
                )
            self.project.set_stage(
                StageID.PHOTOMETRY, StageStatus.READY, "Manual target · unverified WCS", progress=0.2
            )
            self._apply_manifest(self.project.manifest)

    def select_photometry_star(self, role: str, x: float, y: float) -> None:
        requested = (
            "select the target star"
            if role == "target"
            else "refine another comparison star"
        )
        if not self._ensure_runner_idle(requested, StageID.PHOTOMETRY):
            return
        try:
            frame, _ = self._photometry_inputs(require_target=False)
        except BaseException as exc:
            self._handle_error(exc)
            return
        config = PhotometryConfig(aperture_radius=self.plate_page.inspector.aperture.value())
        self.status_text.setText("Refining star position…")
        self.runner.start(
            PhotometryService().locate_star,
            frame,
            x,
            y,
            config,
            result=lambda star: self._photometry_star_selected(role, star),
            error=lambda exc: self._photometry_star_selection_failed(role, exc),
            finished=lambda: self.status_text.setText("Photometry setup ready"),
            inhibit_sleep=False,
            operation="star-position refinement",
        )

    def _photometry_star_selected(self, role: str, star: dict[str, float]) -> None:
        radius = float(star.get("aperture", self.plate_page.inspector.aperture.value()))
        if role == "target":
            self.plate_page.set_target(
                star["x"],
                star["y"],
                radius=radius,
                label=self.project.manifest.target_name if self.project else "Target",
                verified=False,
            )
            self.manual_target_placed(star["x"], star["y"])
            self.plate_page.inspector.banner_title.setText("Manual target selected")
            self.plate_page.inspector.explanation.setText(
                "The click was centered on the selected target star. Add comparison stars to continue."
            )
        else:
            self.plate_page.add_comparison(star["x"], star["y"], radius=radius)
        self._save_photometry_selection(verified=self.plate_page.target_verified)

    def _photometry_star_selection_failed(self, role: str, exc: BaseException) -> None:
        if (
            role == "target"
            and isinstance(exc, LEAPSError)
            and exc.code == "PHOTOMETRY_STAR_NOT_FOUND"
        ):
            exc = LEAPSError(
                "PHOTOMETRY_TARGET_NOT_CENTERED",
                "The target star could not be centered automatically",
                (
                    "LEAPS could not refine this click using the current detection and "
                    "saturation limits. Your target choice was not replaced."
                ),
                [
                    "Click the center of the same target again",
                    "Open Advanced settings and review Saturation fraction",
                    "Use unsaturated exposures if the target reaches the detector limit",
                ],
                stage=StageID.PHOTOMETRY,
                technical_details=exc.technical_details,
            )
        self._handle_error(exc)

    def _save_photometry_selection(self, *, verified: bool | None = None) -> None:
        if not self.project or not self.plate_page.target:
            return
        self.project.manifest.settings["photometry"] = {
            "target_fingerprint": target_fingerprint(
                self.project.manifest.target_ra, self.project.manifest.target_dec
            ),
            "reference_frame": self.plate_page.workspace.filename.text().removeprefix("FITS: "),
            "target": list(self.plate_page.target),
            "comparisons": [list(value) for value in self.plate_page.comparisons],
            "comparison_active": list(self.plate_page.comparison_active),
            "aperture_radius": self.plate_page.inspector.aperture.value(),
            "verified": self.plate_page.target_verified if verified is None else verified,
            "config": self.plate_page.inspector.photometry_config(),
        }
        self.project.save()

    def _photometry_inputs(
        self, *, require_target: bool = True
    ) -> tuple[Path, tuple[float, float] | None]:
        if not self.project:
            raise LEAPSError(
                "PROJECT_REQUIRED",
                "Open a project first",
                "Choose an observing run before photometry.",
                ["Open Data & Target"],
            )
        frame = self._photometry_reference_frame(self.project)
        assert frame is not None
        target = self.plate_page.target
        if target is None and require_target:
            raise LEAPSError(
                "TARGET_POSITION_REQUIRED",
                "The target position is not confirmed",
                "Complete plate solving or place the target manually first.",
                ["Open Photometry"],
                stage=StageID.PHOTOMETRY,
            )
        return frame, target

    def rank_comparison_stars(self) -> None:
        if not self._ensure_runner_idle("rank comparison stars", StageID.PHOTOMETRY):
            return
        try:
            frame, target = self._photometry_inputs()
        except BaseException as exc:
            self._handle_error(exc)
            return

        def rank(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            assert target is not None
            return PhotometryService().rank_comparisons(frame, target)

        self.status_text.setText("Ranking comparison stars…")
        self.runner.start(
            rank,
            result=self.plate_page.set_candidates,
            error=self._handle_error,
            finished=lambda: self.status_text.setText("Comparison ranking ready"),
            operation="comparison-star ranking",
        )

    def run_photometry(self, comparisons: list[tuple[float, float]], radius: float) -> None:
        if not self._ensure_runner_idle("run photometry", StageID.PHOTOMETRY):
            return
        try:
            _, target = self._photometry_inputs()
            if not self.project:
                return
            assert target is not None
            self._save_photometry_selection(verified=False)
            config = PhotometryConfig(**self.plate_page.inspector.photometry_config())
            self._set_photometry_busy(True)
            self.project.manifest.stages[StageID.LIGHT_CURVE.value] = StageState()
            self.project.manifest.stages[StageID.FITTING.value] = StageState()
            self.project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState()
            self.project.manifest.settings.pop("light_curve_review", None)
            self.project.set_stage(StageID.PHOTOMETRY, StageStatus.RUNNING, "Measuring light curve")
            self.runner.start(
                PhotometryService().run,
                self.project,
                target,
                comparisons,
                radius,
                config=config,
                event=self._stage_event,
                result=lambda result: self._stage_complete(StageID.PHOTOMETRY, result),
                error=lambda exc: self._stage_failed(StageID.PHOTOMETRY, exc),
                finished=lambda: self._set_photometry_busy(False),
                operation="photometry",
            )
            self.comparison_page.status.setText("Photometry is running in the background…")
        except BaseException as exc:
            self._set_photometry_busy(False)
            self._handle_error(exc)

    def _set_photometry_busy(self, busy: bool) -> None:
        self.plate_page.inspector.set_busy(busy)
        self.comparison_page.set_busy(busy)

    def review_light_curves(self, active_comparisons: list[bool] | None = None) -> None:
        if not self.project:
            return
        try:
            result = LightCurveReviewService().load(
                self.project, active_comparisons
            )
            self.light_curve_page.set_review(result)
            self.status_text.setText("Light curves ready for review")
        except BaseException as exc:
            failure = self._as_failure(exc, StageID.LIGHT_CURVE)
            self.light_curve_page.show_failure(failure)
            if active_comparisons is None:
                self._show_failure(failure)

    def confirm_light_curve_review(self, active_comparisons: list[bool]) -> None:
        if not self.project:
            return
        if not self._ensure_runner_idle(
            "save the light-curve review", StageID.LIGHT_CURVE
        ):
            return
        try:
            output = LightCurveReviewService().commit(
                self.project, active_comparisons
            )
            self.project.set_stage(
                StageID.LIGHT_CURVE,
                StageStatus.COMPLETE,
                f"{sum(active_comparisons)} comparisons approved",
                progress=1.0,
                output_path=output,
            )
            fitting = self.project.manifest.stages[StageID.FITTING.value]
            fitting.status = StageStatus.READY
            fitting.summary = (
                "Ready · previous result preserved"
                if (self.project.outputs_dir / StageID.FITTING.value).exists()
                else "Ready"
            )
            self.project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState()
            self.project.save()
            self._apply_manifest(self.project.manifest)
            self.status_text.setText("Light curve approved")
            self.autosave.setText("autosaved just now")
            self.open_stage(StageID.FITTING)
        except BaseException as exc:
            failure = self._as_failure(exc, StageID.LIGHT_CURVE)
            self.light_curve_page.show_failure(failure)
            self._show_failure(failure)

    def prepare_fitting_setup(self, *, force: bool = False, requested_name: str = "") -> None:
        if not self.project or self.fitting_lookup_runner.current is not None:
            return
        project = self.project
        fingerprint = target_fingerprint(project.manifest.target_ra, project.manifest.target_dec)
        cached = project.manifest.settings.get("fitting_setup", {})
        cached_observation = cached.get("observation", {})
        has_science = bool(project.manifest.raw_files.get("science"))
        location_refresh_needed = has_science and (
            not isinstance(cached_observation, dict)
            or "location_status" not in cached_observation
        )
        cached_exposure = (
            _optional_float(cached_observation.get("exposure_time"))
            if isinstance(cached_observation, dict)
            else None
        )
        cached_override = _optional_float(cached.get("exposure_time_override"))
        exposure_refresh_needed = has_science and not (
            cached_override is not None and cached_override > 0
        ) and (cached_exposure is None or cached_exposure <= 0)
        if (
            not force
            and not location_refresh_needed
            and not exposure_refresh_needed
            and cached.get("target_fingerprint") == fingerprint
        ):
            try:
                candidates = [PlanetParameters(**values) for values in cached["candidates"]]
                if candidates:
                    self._apply_fitting_setup(
                        project,
                        candidates,
                        cached.get("observation", {}),
                        expected_fingerprint=fingerprint,
                    )
                    return
            except (KeyError, TypeError, ValueError):
                project.manifest.settings.pop("fitting_setup", None)

        manual_snapshot: PlanetParameters | None = None
        if cached.get("target_fingerprint") == fingerprint:
            try:
                current_values = self.fitting_page.values()
                current_parameters = current_values.get("catalog_parameters")
                if isinstance(current_parameters, PlanetParameters) and current_parameters.is_manual:
                    manual_snapshot = _parameters_with_values(current_parameters, current_values)
            except (TypeError, ValueError):
                pass
        cached_manual: PlanetParameters | None = None
        if cached.get("target_fingerprint") == fingerprint:
            try:
                cached_manual = next(
                    parameters
                    for parameters in (
                        PlanetParameters(**values) for values in cached.get("candidates", [])
                    )
                    if parameters.is_manual
                )
            except (StopIteration, TypeError, ValueError):
                pass

        if not self._ensure_project_access(StageID.FITTING):
            return

        self.fitting_page.set_loading("Matching the selected target and reading science FITS headers…")
        self.status_text.setText("Preparing fitting setup…")
        saved_observation = project.manifest.settings.get("observation_metadata", {})
        if not isinstance(saved_observation, dict):
            saved_observation = {}
        assigned_science = list(project.manifest.raw_files.get("science", []))
        current_records = list(self.records)

        def load(*, emit=None, token=None):
            resolver = PlanetCatalogResolver(self._nasa_snapshot_path())
            candidates = resolver.resolve_candidates(
                project.manifest.target_ra,
                project.manifest.target_dec,
                requested_name or project.manifest.target_name,
            )
            observation = saved_observation
            refresh_location = bool(assigned_science) and (
                "location_status" not in observation
            )
            saved_exposure = _optional_float(observation.get("exposure_time"))
            refresh_exposure = bool(assigned_science) and (
                saved_exposure is None or saved_exposure <= 0
            )
            if (
                int(observation.get("science_frames_inspected", 0))
                != len(assigned_science)
                or refresh_location
                or refresh_exposure
            ):
                by_path = {record.path: record for record in current_records}
                records: list[FrameRecord] = []
                inventory = FITSInventory(project.root)
                for relative_path in assigned_science:
                    if token:
                        token.raise_if_cancelled()
                    record = by_path.get(relative_path)
                    record_exposure = (
                        _optional_float(record.exposure) if record is not None else None
                    )
                    if (
                        record is None
                        or not record.filter_name
                        or record_exposure is None
                        or record_exposure <= 0
                        or (
                            refresh_location
                            and not record.observatory
                            and record.observatory_latitude is None
                            and record.observatory_longitude is None
                        )
                    ):
                        record = inventory.inspect(project.resolve(relative_path))
                    records.append(record)
                observation = summarize_observation_records(records, assigned_science)
            if not candidates:
                candidates = [
                    manual_snapshot
                    or cached_manual
                    or _manual_planet_parameters(project)
                ]
            return candidates, observation

        self.fitting_lookup_runner.start(
            load,
            result=lambda payload: self._apply_fitting_setup(
                project,
                *payload,
                expected_fingerprint=fingerprint,
                preferred_name=requested_name,
            ),
            error=self._fitting_setup_failed,
            finished=lambda: self.status_text.setText("Ready"),
            inhibit_sleep=False,
            operation="fitting setup",
        )

    def _apply_fitting_setup(
        self,
        project: ProjectWorkspace,
        candidates: list[PlanetParameters],
        observation: dict[str, Any],
        *,
        expected_fingerprint: str,
        preferred_name: str = "",
    ) -> None:
        if not self.project or self.project.manifest.project_id != project.manifest.project_id:
            return
        fingerprint = target_fingerprint(project.manifest.target_ra, project.manifest.target_dec)
        if fingerprint != expected_fingerprint:
            return
        previous = project.manifest.settings.get("fitting_setup", {})
        observatory, latitude, longitude, observatory_source = _project_observatory(
            project, observation
        )
        tess_import = isinstance(project.manifest.settings.get("tess_import"), dict)
        default_light_curve = "aperture" if tess_import else "gaussian"
        default_detrending = "linear" if tess_import else "quadratic"
        light_curve = str(previous.get("light_curve", default_light_curve))
        detrending = str(previous.get("detrending", default_detrending))
        exposure_override = _optional_float(previous.get("exposure_time_override"))
        if exposure_override is not None and exposure_override <= 0:
            exposure_override = None
        detected_exposure = _optional_float(observation.get("exposure_time"))
        if detected_exposure is not None and detected_exposure <= 0:
            detected_exposure = None
        saved_exposure = _optional_float(project.manifest.settings.get("exposure_time"))
        if saved_exposure is not None and saved_exposure <= 0:
            saved_exposure = None
        effective_exposure = exposure_override or detected_exposure or saved_exposure
        exposure_source = (
            "manual override"
            if exposure_override is not None
            else "science FITS"
            if detected_exposure is not None
            else "saved project"
            if saved_exposure is not None
            else ""
        )
        candidate_names = {parameters.name.casefold(): parameters.name for parameters in candidates}
        selected_name = candidate_names.get(preferred_name.casefold(), "") if preferred_name else ""
        if not selected_name and not preferred_name:
            previous_name = str(previous.get("selected_planet", ""))
            selected_name = candidate_names.get(previous_name.casefold(), "")
        selected_name = selected_name or candidates[0].name
        fitting_setup = {
            "target_fingerprint": fingerprint,
            "selected_planet": selected_name or candidates[0].name,
            "candidates": [asdict(parameters) for parameters in candidates],
            "observation": observation,
            "light_curve": light_curve,
            "detrending": detrending,
        }
        if exposure_override is not None:
            fitting_setup["exposure_time_override"] = exposure_override
        project.manifest.settings["fitting_setup"] = fitting_setup
        project.manifest.settings["observation_metadata"] = observation
        detected_filter = normalize_filter(observation.get("filter"))
        if detected_filter:
            project.manifest.settings["filter"] = detected_filter
        if observation.get("exposure_time"):
            project.manifest.settings["exposure_time"] = float(observation["exposure_time"])
        project.save()
        self.fitting_page.set_planet_candidates(candidates, selected_name)
        self.fitting_page.set_fitting_options(light_curve, detrending)
        self.fitting_page.set_observation_metadata(
            str(project.manifest.settings.get("filter", "")) or None,
            effective_exposure,
            filter_status=str(observation.get("filter_status", "unknown")),
            exposure_source=exposure_source,
        )
        if tess_import:
            self.fitting_page.set_observatory_metadata(
                "TESS", None, None, source="mission metadata"
            )
            self.fitting_page.observation_source.setText(
                self.fitting_page.observation_source.text()
                + " · primary transit will be BLS-refined and phase-folded before the HOPS fit"
            )
        else:
            self.fitting_page.set_observatory_metadata(
                observatory,
                latitude,
                longitude,
                source=observatory_source,
            )
        if not tess_import and (latitude is None or longitude is None):
            self.fitting_page.observation_source.setText(
                self.fitting_page.observation_source.text()
                + " · observer location not set; Airmass de-trending requires a location"
            )
        existing_preview = project.outputs_dir / StageID.FITTING.value / "fit-preview.png"
        if (
            existing_preview.exists()
            and project.manifest.stages[StageID.FITTING.value].status == StageStatus.COMPLETE
        ):
            residual_std = None
            try:
                summary = json.loads(
                    (existing_preview.parent / "fit-summary.json").read_text(encoding="utf-8")
                )
                residual_std = _optional_float(summary.get("residual_std"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            self.fitting_page.show_preview(
                existing_preview,
                planet=selected_name,
                passband=str(project.manifest.settings.get("filter", "")),
                residual_std=residual_std,
            )

    def _fitting_setup_failed(self, exc: BaseException) -> None:
        failure = self._as_failure(exc, StageID.FITTING)
        self.fitting_page.show_failure(f"{failure.title}: {failure.message}")
        self._show_failure(failure)

    def run_fitting(self, values: dict[str, Any], *, full: bool) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "A project light curve is required for fitting.",
                    ["Open Data & Target"],
                    stage=StageID.FITTING,
                )
            )
            return
        approved_curve = (
            self.project.outputs_dir
            / StageID.LIGHT_CURVE.value
            / "light_curve_aperture.txt"
        )
        if (
            self.project.manifest.stages[StageID.LIGHT_CURVE.value].status
            != StageStatus.COMPLETE
            or not approved_curve.exists()
        ):
            self._handle_error(
                LEAPSError(
                    "LIGHT_CURVE_REVIEW_REQUIRED",
                    "Review the comparison stars first",
                    "Fitting uses the approved comparison ensemble from the Light Curve stage.",
                    ["Open Light Curve", "Confirm at least one comparison star"],
                    stage=StageID.LIGHT_CURVE,
                )
            )
            return
        if not self._ensure_runner_idle(
            "run the full fit" if full else "build a fit preview", StageID.FITTING
        ):
            return
        values = dict(values)
        try:
            observation_times = np.atleast_1d(
                np.loadtxt(approved_curve, usecols=(0,), dtype=float)
            )
            entered_mid_time = float(values.get("mid_time", 0.0))
            normalized_mid_time = _normalize_mid_transit_time(
                entered_mid_time, observation_times
            )
            values["mid_time"] = normalized_mid_time
            if normalized_mid_time != entered_mid_time:
                signals_were_blocked = self.fitting_page.mid_time.blockSignals(True)
                self.fitting_page.mid_time.setValue(normalized_mid_time)
                self.fitting_page.mid_time.blockSignals(signals_were_blocked)
        except (OSError, TypeError, ValueError):
            # FittingService retains the authoritative typed light-curve error.
            # Do not replace it with a UI-only failure while preparing the setup.
            pass
        parameters = values.get("catalog_parameters")
        if not isinstance(parameters, PlanetParameters):
            self._handle_error(
                LEAPSError(
                    "FITTING_PLANET_REQUIRED",
                    "Enter planet parameters",
                    "Use a catalog match or the manual uncatalogued-target setup before fitting.",
                    ["Choose a suggested planet", "Complete the manual parameter fields"],
                    stage=StageID.FITTING,
                )
            )
            return
        try:
            parameters = _parameters_with_values(parameters, values)
        except (TypeError, ValueError):
            parameters = None
        if parameters is None or not (
            np.isfinite(parameters.period)
            and parameters.period > 0
            and np.isfinite(parameters.mid_time)
            and parameters.mid_time > 0
            and np.isfinite(parameters.rp_over_rs)
            and 0 < parameters.rp_over_rs <= 1
            and np.isfinite(parameters.sma_over_rs)
            and parameters.sma_over_rs > 0
            and np.isfinite(parameters.inclination)
            and 0 < parameters.inclination <= 90
            and np.isfinite(parameters.eccentricity)
            and 0 <= parameters.eccentricity < 1
            and np.isfinite(parameters.periastron)
            and np.isfinite(parameters.metallicity)
            and np.isfinite(parameters.temperature)
            and parameters.temperature > 0
            and np.isfinite(parameters.logg)
        ):
            self._handle_error(
                LEAPSError(
                    "FITTING_PARAMETERS_REQUIRED",
                    "Complete the planet parameters",
                    "Period, timing, depth, and the orbital and stellar assumptions must be valid before fitting.",
                    ["Enter a positive orbital period", "Review the manual assumptions"],
                    stage=StageID.FITTING,
                )
            )
            return
        filter_name = normalize_filter(values.get("filter"))
        if not filter_name:
            self._handle_error(
                LEAPSError(
                    "FITTING_FILTER_REQUIRED",
                    "Choose the observation filter",
                    "LEAPS could not translate the selected filter to a HOPS passband.",
                    ["Choose a filter from the list", "Check the science FITS header"],
                    stage=StageID.FITTING,
                )
            )
            return
        entered_exposure = _optional_float(values.get("exposure_time"))
        saved_exposure = _optional_float(
            self.project.manifest.settings.get("exposure_time")
        )
        exposure_time = (
            entered_exposure
            if entered_exposure is not None and entered_exposure > 0
            else saved_exposure
        )
        if exposure_time is None or exposure_time <= 0:
            self._handle_error(
                LEAPSError(
                    "FITTING_EXPOSURE_REQUIRED",
                    "Enter the science exposure time",
                    "The assigned science FITS do not provide a usable exposure time, so enter the duration of one frame on the Fitting page.",
                    ["Enter a positive exposure time", "Check the science FITS header", "Retry Preview Fit"],
                    stage=StageID.FITTING,
                )
            )
            return

        tess_import = isinstance(self.project.manifest.settings.get("tess_import"), dict)
        _, detected_latitude, detected_longitude, _ = _project_observatory(self.project)
        latitude = None if tess_import else detected_latitude
        longitude = None if tess_import else detected_longitude
        project = self.project
        setup = project.manifest.settings.get("fitting_setup", {})
        setup["selected_planet"] = parameters.name
        default_light_curve = "aperture" if tess_import else "gaussian"
        default_detrending = "linear" if tess_import else "quadratic"
        setup["light_curve"] = str(values.get("light_curve", default_light_curve))
        setup["detrending"] = str(values.get("detrending", default_detrending))
        observation = setup.get("observation", {})
        detected_exposure = (
            _optional_float(observation.get("exposure_time"))
            if isinstance(observation, dict)
            else None
        )
        if entered_exposure is not None and entered_exposure > 0:
            if detected_exposure is not None and np.isclose(
                entered_exposure, detected_exposure
            ):
                setup.pop("exposure_time_override", None)
            else:
                setup["exposure_time_override"] = float(entered_exposure)
        if parameters.is_manual:
            setup["candidates"] = [asdict(parameters)]
        project.manifest.settings["filter"] = filter_name
        project.manifest.settings["exposure_time"] = float(exposure_time)
        project.manifest.settings["fitting_setup"] = setup
        project.save()

        def fit(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            result = FittingService().run(
                project,
                parameters,
                full=full,
                exposure_time=float(exposure_time),
                filter_name=filter_name,
                latitude=latitude,
                longitude=longitude,
                light_curve=setup["light_curve"],
                detrending=setup["detrending"],
                iterations=int(values["iterations"]),
                burn_in=int(values["burn"]),
                emit=emit,
                token=token,
            )
            if token:
                token.raise_if_cancelled()
            return result

        self.fitting_page.set_busy(True, full=full)
        project.set_stage(
            StageID.FITTING,
            StageStatus.RUNNING if full else StageStatus.READY,
            "Running full fit" if full else "Building preview",
            progress=0.0,
        )
        self._apply_manifest(project.manifest)
        self.status_text.setText("Running full fit…" if full else "Building fit preview…")
        self.runner.start(
            fit,
            event=self._stage_event,
            result=self._fitting_complete,
            error=lambda exc: self._fitting_failed(exc, full=full),
            finished=lambda: self.fitting_page.set_busy(False),
            operation="full transit fit" if full else "fit preview",
        )

    def cancel_fitting(self) -> None:
        if self.runner.current is None:
            return
        self.fitting_page.set_stopping()
        self.status_text.setText("Stopping fit safely…")
        self.runner.cancel()

    def _fitting_complete(self, result: FittingService.Result) -> None:
        self.fitting_page.show_preview(
            result.preview_path,
            planet=result.planet,
            passband=result.passband,
            residual_std=result.residual_std,
        )
        if self.project:
            self.project.set_stage(
                StageID.FITTING,
                StageStatus.COMPLETE if result.full else StageStatus.READY,
                "Complete" if result.full else "Preview ready",
                progress=1.0 if result.full else 0.5,
                checkpoint="complete" if result.full else "preview_ready",
                output_path=result.output_path,
            )
            state = self.project.manifest.stages[StageID.FITTING.value]
            if "FITTING_INTERRUPTED" in state.warning_codes:
                state.warning_codes.remove("FITTING_INTERRUPTED")
                self.project.save()
            self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText("Fitting complete" if result.full else "Fit preview ready")
        self.autosave.setText("autosaved just now")

    def _fitting_failed(self, exc: BaseException, *, full: bool) -> None:
        failure = self._as_failure(exc, StageID.FITTING)
        if failure.code == "JOB_CANCELLED":
            if self.project:
                self.project.set_stage(
                    StageID.FITTING,
                    StageStatus.READY,
                    "Full fit cancelled" if full else "Preview cancelled",
                    progress=0.0,
                    checkpoint="cancelled",
                )
                self._apply_manifest(self.project.manifest)
            message = (
                "Full fit cancelled. The incomplete attempt was discarded and the previous "
                "preview and successful results were preserved."
                if full
                else "Fit preview cancelled. Existing fitting results were preserved."
            )
            self.fitting_page.show_cancelled(message)
            self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
            self.status_text.setText("Full fit cancelled" if full else "Fit preview cancelled")
            self.autosave.setText("autosaved just now")
            return
        if self.project:
            self.project.set_stage(
                StageID.FITTING,
                StageStatus.NEEDS_ATTENTION,
                "Full fit needs attention" if full else "Preview needs attention",
            )
            self._apply_manifest(self.project.manifest)
        self.fitting_page.show_failure(f"{failure.title}: {failure.message}")
        self._show_failure(failure)

    def prepare_secondary_eclipse_setup(self) -> None:
        if not self.project:
            return
        project = self.project
        fitting_state = project.manifest.stages[StageID.FITTING.value]
        if fitting_state.status != StageStatus.COMPLETE:
            self.secondary_eclipse_page.reset_setup(
                "Run a completed primary-transit fit first. Eclipse analysis reuses that saved ephemeris."
            )
            return
        summary_path = project.outputs_dir / StageID.FITTING.value / "fit-summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            parameters = PlanetParameters(**summary["parameters"])
            fitted_ephemeris = summary.get("fitted_ephemeris", {})
            parameters = replace(
                parameters,
                period=_optional_float(fitted_ephemeris.get("period")) or parameters.period,
                mid_time=_optional_float(fitted_ephemeris.get("mid_time")) or parameters.mid_time,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.secondary_eclipse_page.show_failure(
                "The completed transit fit does not contain reusable ephemeris data. Run Full Fit again."
            )
            self._handle_error(
                LEAPSError(
                    "SECONDARY_ECLIPSE_EPHEMERIS_MISSING",
                    "The fitted ephemeris is unavailable",
                    "LEAPS could not read the completed transit fit needed for secondary-eclipse analysis.",
                    ["Open Fitting", "Run Full Fit again", "Export diagnostics if it repeats"],
                    stage=StageID.SECONDARY_ECLIPSE,
                    technical_details=f"{summary_path}\n{exc}",
                )
            )
            return
        saved = project.manifest.settings.get("secondary_eclipse_setup", {})
        fingerprint = target_fingerprint(project.manifest.target_ra, project.manifest.target_dec)
        saved_is_current = saved.get("target_fingerprint") == fingerprint
        saved_duration = _optional_float(saved.get("duration_hours")) if saved_is_current else None
        if saved_is_current:
            for control, value in (
                (self.secondary_eclipse_page.expected_phase, _optional_float(saved.get("expected_phase"))),
                (self.secondary_eclipse_page.duration_hours, _optional_float(saved.get("duration_hours"))),
            ):
                if value is not None:
                    blocked = control.blockSignals(True)
                    control.setValue(value)
                    control.blockSignals(blocked)
            for control, value in (
                (self.secondary_eclipse_page.light_curve, saved.get("light_curve")),
                (self.secondary_eclipse_page.baseline, saved.get("baseline")),
            ):
                index = control.findData(value)
                if index >= 0:
                    blocked = control.blockSignals(True)
                    control.setCurrentIndex(index)
                    control.blockSignals(blocked)
        self.secondary_eclipse_page.set_fit_context(
            parameters,
            passband=str(summary.get("passband", "")),
            light_curve=str(
                saved.get("light_curve", summary.get("light_curve", "aperture"))
                if saved_is_current
                else summary.get("light_curve", "aperture")
            ),
            duration_hours=saved_duration or SecondaryEclipseService.estimate_duration_hours(parameters),
        )
        result_path = project.outputs_dir / StageID.SECONDARY_ECLIPSE.value / "secondary-eclipse.json"
        preview_path = result_path.with_name("secondary-eclipse.png")
        if result_path.exists():
            try:
                self.secondary_eclipse_page.show_saved_result(
                    json.loads(result_path.read_text(encoding="utf-8")), preview_path
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self.secondary_eclipse_page.invalidate_result()
        else:
            self.secondary_eclipse_page.invalidate_result()

    def run_secondary_eclipse(self, values: dict[str, Any]) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "An approved light curve and completed primary-transit fit are required.",
                    ["Open Data & Target"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )
            )
            return
        project = self.project
        if project.manifest.stages[StageID.FITTING.value].status != StageStatus.COMPLETE:
            self._handle_error(
                LEAPSError(
                    "SECONDARY_ECLIPSE_FIT_REQUIRED",
                    "Complete the primary-transit fit first",
                    "Secondary-eclipse analysis uses the saved full-fit ephemeris and approved light curve.",
                    ["Open Fitting", "Run Full Fit"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )
            )
            return
        parameters = values.get("catalog_parameters")
        if not isinstance(parameters, PlanetParameters):
            self._handle_error(
                LEAPSError(
                    "SECONDARY_ECLIPSE_EPHEMERIS_MISSING",
                    "The fitted ephemeris is unavailable",
                    "Open Secondary Eclipse again to reload the completed primary-transit fit.",
                    ["Open Secondary Eclipse", "Run Full Fit again if needed"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )
            )
            return
        if not self._ensure_runner_idle("analyse a secondary eclipse", StageID.SECONDARY_ECLIPSE):
            return
        _, latitude, longitude, _ = _project_observatory(project)
        fingerprint = target_fingerprint(project.manifest.target_ra, project.manifest.target_dec)
        setup = {
            "target_fingerprint": fingerprint,
            "light_curve": str(values.get("light_curve", "aperture")),
            "expected_phase": float(values.get("expected_phase", 0.5)),
            "duration_hours": float(values.get("duration_hours", 2.0)),
            "baseline": str(values.get("baseline", "linear")),
        }
        project.manifest.settings["secondary_eclipse_setup"] = setup
        project.save()

        def analyse(*, emit=None, token=None):
            return SecondaryEclipseService().run(
                project,
                parameters,
                expected_phase=setup["expected_phase"],
                duration_hours=setup["duration_hours"],
                light_curve=setup["light_curve"],
                baseline=setup["baseline"],
                latitude=latitude,
                longitude=longitude,
                emit=emit,
                token=token,
            )

        self.secondary_eclipse_page.set_busy(True)
        project.set_stage(
            StageID.SECONDARY_ECLIPSE,
            StageStatus.RUNNING,
            "Analysing expected eclipse",
            progress=0.0,
        )
        self._apply_manifest(project.manifest)
        self.status_text.setText("Analysing secondary eclipse…")
        self.runner.start(
            analyse,
            event=self._stage_event,
            result=self._secondary_eclipse_complete,
            error=self._secondary_eclipse_failed,
            finished=lambda: self.secondary_eclipse_page.set_busy(False),
            operation="secondary-eclipse analysis",
        )

    def cancel_secondary_eclipse(self) -> None:
        if self.runner.current is None:
            return
        self.secondary_eclipse_page.set_stopping()
        self.status_text.setText("Stopping secondary-eclipse analysis safely…")
        self.runner.cancel()

    def _secondary_eclipse_complete(self, result: SecondaryEclipseService.Result) -> None:
        self.secondary_eclipse_page.show_result(result)
        if self.project:
            self.project.set_stage(
                StageID.SECONDARY_ECLIPSE,
                StageStatus.COMPLETE,
                result.outcome_label,
                progress=1.0,
                checkpoint="complete",
                output_path=result.output_path,
            )
            state = self.project.manifest.stages[StageID.SECONDARY_ECLIPSE.value]
            if "SECONDARY_ECLIPSE_INTERRUPTED" in state.warning_codes:
                state.warning_codes.remove("SECONDARY_ECLIPSE_INTERRUPTED")
                self.project.save()
            self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText("Secondary-eclipse analysis complete")
        self.autosave.setText("autosaved just now")

    def _secondary_eclipse_failed(self, exc: BaseException) -> None:
        failure = self._as_failure(exc, StageID.SECONDARY_ECLIPSE)
        if failure.code == "JOB_CANCELLED":
            if self.project:
                self.project.set_stage(
                    StageID.SECONDARY_ECLIPSE,
                    StageStatus.READY,
                    "Analysis cancelled",
                    progress=0.0,
                    checkpoint="cancelled",
                )
                self._apply_manifest(self.project.manifest)
            self.secondary_eclipse_page.show_cancelled(
                "Analysis cancelled. The incomplete result was discarded and the previous eclipse analysis was preserved."
            )
            self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
            self.status_text.setText("Secondary-eclipse analysis cancelled")
            self.autosave.setText("autosaved just now")
            return
        if self.project:
            self.project.set_stage(
                StageID.SECONDARY_ECLIPSE,
                StageStatus.NEEDS_ATTENTION,
                "Analysis needs attention",
            )
            self._apply_manifest(self.project.manifest)
        self.secondary_eclipse_page.show_failure(f"{failure.title}: {failure.message}")
        self._show_failure(failure)

    def _as_failure(self, exc: BaseException, stage: StageID | None = None) -> LEAPSError:
        if self.logger:
            return self.logger.failure(exc, stage)
        if isinstance(exc, LEAPSError):
            return exc
        return LEAPSError(
            "UNEXPECTED_FAILURE",
            "This step could not be completed",
            "LEAPS kept the last successful result. Retry or export diagnostics.",
            ["Retry", "Export diagnostics"],
            stage=stage,
            technical_details=str(exc),
        )

    def _handle_error(self, exc: BaseException) -> None:
        self._show_failure(self._as_failure(exc))

    def _show_failure(self, failure: LEAPSError) -> None:
        self.last_failure = failure
        self.status_dot.setStyleSheet(f"color: {COLORS['amber']};")
        self.status_text.setText(failure.title)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle(failure.title)
        dialog.setText(failure.message)
        dialog.setInformativeText(
            "Suggested recovery: "
            + " · ".join(failure.recovery)
            + f"\n\nDiagnostic reference: {failure.diagnostic_id}"
        )
        if failure.technical_details:
            dialog.setDetailedText(failure.technical_details)
        dialog.exec()

    def copy_diagnostics(self) -> None:
        failure = self.last_failure
        payload = (
            failure.as_dict()
            if failure
            else {
                "failure": "PLATE_SOLVE_FAILED",
                "diagnostic_reference": "LEAPS-DEMO-503",
                "attempts": [
                    "coordinates validated",
                    "42 stars detected",
                    "Gaia HTTP 503",
                    "stopped after 3 bounded attempts",
                ],
            }
        )
        if self.project:
            payload["target"] = self.project.manifest.target_name
            payload["coordinates"] = f"{self.project.manifest.target_ra} {self.project.manifest.target_dec}"
            payload["pixel_scale"] = self.project.manifest.settings.get("pixel_scale")
            payload["estimated_pixel_scale"] = self.plate_page.inspector.estimated_pixel_scale
        QApplication.clipboard().setText(json.dumps(payload, indent=2))
        self.status_text.setText("Plate-solve diagnostics copied")

    def export_diagnostics(self) -> None:
        if not self.logger:
            self.open_tool("diagnostics")
            return
        default = str(self.project.root / f"LEAPS-diagnostics-{self.project.manifest.project_id[:8]}.zip")
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export redacted diagnostics", default, "ZIP archive (*.zip)"
        )
        if destination:
            samples = [
                self.project.resolve(path) for path in self.project.manifest.raw_files.get("science", [])[:3]
            ]
            output = self.logger.export_bundle(destination, samples)
            self.status_text.setText(f"Diagnostics exported: {output.name}")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.offline_manager, self)
        dialog.offline.downloadAllRequested.connect(lambda: self._download_offline(dialog))
        dialog.offline.refreshRequested.connect(lambda: self._refresh_offline(dialog))
        dialog.offline.removeRequested.connect(lambda asset_id: self._remove_offline(dialog, asset_id))
        dialog.exec()

    def _download_offline(self, dialog: SettingsDialog) -> None:
        if not self._ensure_runner_idle("download offline data"):
            return
        self._settings_dialog = dialog

        def download(*, emit=None, token=None):
            self.offline_manager.download_all(
                progress=self.offlineProgress.emit,
                cancelled=lambda: bool(token and token.cancelled),
            )
            return True

        dialog.offline.download.setEnabled(False)
        self.runner.start(
            download,
            result=lambda _: dialog.offline.finish_progress(),
            error=self._handle_error,
            finished=lambda: dialog.offline.download.setEnabled(True),
            operation="offline-data download",
        )

    def _offline_progress(self, label: str, current: int, total: int) -> None:
        dialog = getattr(self, "_settings_dialog", None)
        if dialog is not None:
            dialog.offline.set_progress(label, current, total)

    def _refresh_offline(self, dialog: SettingsDialog) -> None:
        if not self._ensure_runner_idle("refresh offline data"):
            return

        def refresh(*, emit=None, token=None):
            return self.offline_manager.load_remote_manifest()

        self.runner.start(
            refresh,
            result=lambda _: dialog.offline.refresh(),
            error=self._handle_error,
            operation="offline-data refresh",
        )

    def _remove_offline(self, dialog: SettingsDialog, asset_id: str) -> None:
        self.offline_manager.remove(asset_id)
        dialog.offline.refresh()

    def open_outputs_folder(self) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "Project outputs are created beside an observing run.",
                    ["Open Data & Target"],
                )
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.outputs_dir)))

    def open_project_folder(self) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "Choose and confirm an observing run before opening its project files.",
                    ["Open Data & Target"],
                    stage=StageID.DATA_TARGET,
                )
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.workspace)))

    def view_fit_preview_in_files(self, path: Path) -> None:
        preview = Path(path)
        if not preview.is_file():
            self._handle_error(
                LEAPSError(
                    "FIT_PREVIEW_MISSING",
                    "The fit preview is no longer available",
                    "The preview image may have been moved or replaced since it was displayed.",
                    ["Run Preview Fit again"],
                    stage=StageID.FITTING,
                    technical_details=str(preview),
                )
            )
            return
        try:
            _reveal_in_file_manager(preview)
        except OSError as exc:
            self._handle_error(
                LEAPSError(
                    "FIT_PREVIEW_REVEAL_FAILED",
                    "The fit preview could not be shown in files",
                    "LEAPS could not open the system file manager.",
                    ["Open the LEAPS fitting output folder manually"],
                    stage=StageID.FITTING,
                    technical_details=f"{preview}\n{exc}",
                )
            )

    def view_secondary_eclipse_in_files(self, path: Path) -> None:
        preview = Path(path)
        if not preview.is_file():
            self._handle_error(
                LEAPSError(
                    "SECONDARY_ECLIPSE_PREVIEW_MISSING",
                    "The eclipse plot is no longer available",
                    "The plot may have been moved or replaced since it was displayed.",
                    ["Run secondary-eclipse analysis again"],
                    stage=StageID.SECONDARY_ECLIPSE,
                    technical_details=str(preview),
                )
            )
            return
        try:
            _reveal_in_file_manager(preview)
        except OSError as exc:
            self._handle_error(
                LEAPSError(
                    "SECONDARY_ECLIPSE_REVEAL_FAILED",
                    "The eclipse plot could not be shown in files",
                    "LEAPS could not open the system file manager.",
                    ["Open the LEAPS secondary-eclipse output folder manually"],
                    stage=StageID.SECONDARY_ECLIPSE,
                    technical_details=f"{preview}\n{exc}",
                )
            )

    def request_project_reset(self) -> None:
        if not self.project:
            return
        if self.runner.current is not None or self.fitting_lookup_runner.current is not None:
            self._handle_error(
                LEAPSError(
                    "PROJECT_RESET_BUSY",
                    "Finish the current operation first",
                    "Project data cannot be reset while LEAPS is processing this run.",
                    ["Cancel safely and wait for it to finish", "Try reset again"],
                    stage=StageID.DATA_TARGET,
                )
            )
            return
        project = self.project
        dialog = ProjectResetDialog(project, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._resetting_project = True
        self.data_page.set_project_actions_available(True, busy=True)
        self.status_dot.setStyleSheet(f"color: {COLORS['cyan']};")
        self.status_text.setText("Resetting generated project data…")

        def reset(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            return project.delete_generated_data()

        self.runner.start(
            reset,
            result=lambda removed: self._project_reset_complete(project.root, int(removed)),
            error=lambda exc: self._project_reset_failed(project, exc),
            inhibit_sleep=False,
            operation="project reset",
        )

    def _project_reset_complete(self, root: Path, removed_bytes: int) -> None:
        self._resetting_project = False
        self._clear_current_project(root)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText(f"Project reset complete · {format_bytes(removed_bytes)} removed")

    def _project_reset_failed(self, project: ProjectWorkspace, exc: BaseException) -> None:
        self._resetting_project = False
        failure = self._as_failure(exc, StageID.DATA_TARGET)
        if not project.workspace.exists():
            self._clear_current_project(project.root)
        else:
            self.data_page.set_project_actions_available(True, busy=False)
        self._show_failure(failure)

    def _clear_current_project(self, root: Path) -> None:
        recent = self.settings.value("projects/recent", "")
        if recent:
            try:
                matches = Path(str(recent)).expanduser().resolve() == root.resolve()
            except OSError:
                matches = str(recent) == str(root)
            if matches:
                self.settings.remove("projects/recent")
        self.project = None
        self.logger = None
        self.records = []
        self.data_page.clear_session()
        self.plate_page.clear_selection()
        self.fitting_page.reset_setup("Open a project to load fitting parameters.")
        self.secondary_eclipse_page.reset_setup("Open a project and run a full fit first.")
        empty = ProjectManifest()
        self._apply_manifest(empty)
        self.project_label.clear()
        self.autosave.setText("No project open")
        self.open_stage(StageID.DATA_TARGET)
        self.projectChanged.emit(None)

    def _runner_busy_changed(self, busy: bool) -> None:
        busy = busy or self.runner.current is not None or self.fitting_lookup_runner.current is not None
        self.data_page.set_project_actions_available(
            self.project is not None,
            busy=busy or self._resetting_project,
        )

    def export_transit(self, format_name: str) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "A successful light curve is required for export.",
                    ["Open Data & Target"],
                )
            )
            return
        label = "ExoClock" if format_name == "exoclock" else "ETD"
        default = str(
            self.project.outputs_dir / f"{self.project.manifest.target_name or 'transit'}-{label}.txt"
        )
        destination, _ = QFileDialog.getSaveFileName(self, f"Export {label}", default, "Text table (*.txt)")
        if not destination:
            return
        try:
            exporter = TransitExporter(self.project)
            output = (
                exporter.export_exoclock(destination)
                if format_name == "exoclock"
                else exporter.export_etd(destination)
            )
            self.status_text.setText(f"{label} export created: {output.name}")
        except BaseException as exc:
            self._handle_error(exc)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.runner.cancel()
        self.target_lookup_runner.cancel()
        self.fitting_lookup_runner.cancel()
        if self.project and not self._resetting_project:
            self.project.save()
        event.accept()

    def autosave_project(self) -> None:
        if self.project and not self._resetting_project:
            self.project.save()
            self.autosave.setText("autosaved just now")
