from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, QTime, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from leaps.catalog import PlanetParameters
from leaps.filters import normalize_filter, passband_choices, passband_label
from leaps.fits_inventory import FITS_EXTENSIONS, FrameRecord, is_generated_project_path
from leaps.models import LEAPSError, StageEvent, StageID
from leaps.targets import ResolvedTarget

from .theme import COLORS
from .widgets import ActionButton, FITSWorkspace, InfoButton, LabelWithInfo, PageHeader, icon


def _scroll_page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    total = max(0, round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _optional_float(value: object) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


class FrameAssignmentCard(QFrame):
    classifierChanged = Signal()

    def __init__(
        self,
        title: str,
        default_classifier: str,
        icon_name: str,
        tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("assignmentCard")
        self.setStyleSheet(
            f"QFrame#assignmentCard {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 14, 14)
        layout.setSpacing(9)

        heading = QHBoxLayout()
        category_icon = QLabel()
        category_icon.setPixmap(icon(icon_name, COLORS["cyan"]).pixmap(19, 19))
        heading.addWidget(category_icon)
        label = QLabel(title)
        label.setStyleSheet("font-size: 14px; font-weight: 650;")
        heading.addWidget(label)
        heading.addWidget(InfoButton(tooltip))
        heading.addStretch()
        self.count = QLabel("0 selected")
        self.count.setStyleSheet(
            f"color: {COLORS['green']}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 3px 8px; font-weight: 600;"
        )
        heading.addWidget(self.count)
        layout.addLayout(heading)

        classifier_label = QLabel("Filename classifier")
        classifier_label.setObjectName("muted")
        layout.addWidget(classifier_label)
        self.classifier = QLineEdit(default_classifier)
        self.classifier.setPlaceholderText(default_classifier)
        self.classifier.setToolTip(
            "Case-insensitive filename text. Separate multiple classifiers with commas, for example: dark, d."
        )
        self.classifier.textChanged.connect(self.classifierChanged)
        layout.addWidget(self.classifier)

    def set_count(self, count: int) -> None:
        self.count.setText(f"{count} selected")
        color = COLORS["green"] if count else COLORS["muted"]
        self.count.setStyleSheet(
            f"color: {color}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 3px 8px; font-weight: 600;"
        )


class DataTargetPage(QWidget):
    scanRequested = Signal(object)
    saveRequested = Signal(dict)
    targetLookupRequested = Signal(str)
    revealProjectRequested = Signal()
    resetProjectRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = PageHeader(
            "Data & Target", "Choose a FITS folder, verify the target, and confirm frame assignments."
        )
        outer.addWidget(header)

        body = QWidget()
        layout = QVBoxLayout(body)
        self.content_layout = layout
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        target_card = QFrame()
        target_card.setObjectName("card")
        self.target_card = target_card
        target_layout = QVBoxLayout(target_card)
        target_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Target coordinates")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "Coordinates are the canonical identity. The target name is a convenient label and is not required for plate solving."
            )
        )
        heading.addStretch()
        target_layout.addLayout(heading)
        form = QGridLayout()
        form.setHorizontalSpacing(15)
        form.setVerticalSpacing(10)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Optional, e.g. WTS-2 b")
        self.name.setToolTip(
            "Enter a SIMBAD star or exoplanet name. LEAPS looks up RA/DEC after you pause typing, press Enter, or leave the field."
        )
        self.target_lookup_timer = QTimer(self)
        self.target_lookup_timer.setSingleShot(True)
        self.target_lookup_timer.setInterval(800)
        self.target_lookup_timer.timeout.connect(self._request_target_lookup)
        self.name.returnPressed.connect(self._request_target_lookup)
        self.name.editingFinished.connect(self._request_target_lookup)
        self.name.textEdited.connect(self._target_name_edited)
        self.ra = QLineEdit()
        self.ra.setPlaceholderText("19:34:55.87")
        self.dec = QLineEdit()
        self.dec.setPlaceholderText("+36:48:55.79")
        form.addWidget(
            LabelWithInfo(
                "Target name", "A familiar label for reports. Plate solving uses the coordinates below."
            ),
            0,
            0,
        )
        form.addWidget(self.name, 0, 1)
        self.target_lookup_status = QLabel("Enter a name to look up coordinates automatically.")
        self.target_lookup_status.setObjectName("muted")
        form.addWidget(self.target_lookup_status, 1, 1)
        form.addWidget(
            LabelWithInfo(
                "Right ascension", "ICRS right ascension in hours, minutes, and seconds: hh:mm:ss."
            ),
            2,
            0,
        )
        form.addWidget(self.ra, 2, 1)
        form.addWidget(
            LabelWithInfo(
                "Declination", "ICRS declination in signed degrees, minutes, and seconds: +dd:mm:ss."
            ),
            3,
            0,
        )
        form.addWidget(self.dec, 3, 1)
        form.setColumnStretch(1, 1)
        target_layout.addLayout(form)
        self.target_source = QLabel("")
        self.target_source.setVisible(False)
        self.target_source.setWordWrap(True)
        target_layout.addWidget(self.target_source)

        folder_card = QFrame()
        folder_card.setObjectName("card")
        self.folder_card = folder_card
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(18, 16, 18, 18)
        row = QHBoxLayout()
        title = QLabel("Observing run")
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addWidget(
            InfoButton(
                "LEAPS reads raw FITS files in place. Project information, logs, and generated outputs are stored in a visible, portable LEAPS folder beside them."
            )
        )
        row.addStretch()
        folder_layout.addLayout(row)
        pick = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText("Select the folder containing the FITS run")
        browse = ActionButton(
            "Choose folder",
            "fa6s.folder-open",
            tooltip="Choose a folder containing science and calibration FITS frames.",
        )
        browse.clicked.connect(self._choose_folder)
        pick.addWidget(self.folder, 1)
        pick.addWidget(browse)
        folder_layout.addLayout(pick)
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        folder_layout.addWidget(self.scan_progress)
        self.project_actions = QWidget()
        project_actions_layout = QHBoxLayout(self.project_actions)
        project_actions_layout.setContentsMargins(0, 6, 0, 0)
        project_actions_layout.setSpacing(10)
        self.project_storage = QLabel("Project files: LEAPS/")
        self.project_storage.setObjectName("muted")
        project_actions_layout.addWidget(self.project_storage)
        project_actions_layout.addStretch()
        self.reveal_project = ActionButton(
            "Open LEAPS Folder",
            "fa6s.folder-open",
            tooltip="Reveal the project manifest, structured logs, checkpoints, caches, and generated outputs.",
        )
        self.reveal_project.clicked.connect(self.revealProjectRequested)
        project_actions_layout.addWidget(self.reveal_project)
        self.reset_project = ActionButton(
            "Reset Project Data",
            "fa6s.trash",
            tooltip="Remove only LEAPS-generated project information and outputs. Raw FITS frames are never deleted.",
        )
        self.reset_project.setProperty("danger", True)
        self.reset_project.clicked.connect(self.resetProjectRequested)
        project_actions_layout.addWidget(self.reset_project)
        self.project_actions.setVisible(False)
        folder_layout.addWidget(self.project_actions)
        layout.addWidget(folder_card)
        layout.addWidget(target_card)

        frames_card = QFrame()
        frames_card.setObjectName("card")
        self.frames_card = frames_card
        frames_layout = QVBoxLayout(frames_card)
        frames_layout.setContentsMargins(18, 16, 18, 18)
        top = QHBoxLayout()
        title = QLabel("Frame assignments")
        title.setObjectName("sectionTitle")
        top.addWidget(title)
        top.addWidget(
            InfoButton(
                "LEAPS classifies frames from FITS headers and filenames. Review low-confidence or unknown assignments before continuing."
            )
        )
        top.addStretch()
        self.counts = QLabel("No FITS files scanned")
        self.counts.setObjectName("muted")
        top.addWidget(self.counts)
        frames_layout.addLayout(top)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        card_definitions = (
            (
                "bias",
                "Bias",
                "Bias",
                "fa6s.sliders",
                "Zero-exposure calibration frames used to remove the detector's electronic offset.",
            ),
            (
                "dark",
                "Darks",
                "Dark",
                "fa6s.moon",
                "Closed-shutter calibration frames used to remove thermal signal at the science exposure time.",
            ),
            (
                "flat",
                "Flats",
                "Flat",
                "fa6s.sun",
                "Evenly illuminated calibration frames used to correct dust and sensitivity variations.",
            ),
            (
                "science",
                "Science Images",
                "Image",
                "fa6s.star",
                "The target-field images recorded across the transit observing run.",
            ),
        )
        self.assignment_cards: dict[str, FrameAssignmentCard] = {}
        self.default_classifiers: dict[str, str] = {}
        for index, (key, label, default, icon_name, tip) in enumerate(card_definitions):
            card = FrameAssignmentCard(label, default, icon_name, tip)
            card.classifierChanged.connect(self._refresh_assignments)
            self.assignment_cards[key] = card
            self.default_classifiers[key] = default
            cards.addWidget(card, index // 2, index % 2)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        frames_layout.addLayout(cards)

        hint = QLabel(
            "Classifiers match filenames without regard to capitalization. Use commas for aliases, such as “dark, d”. Each file is assigned to the first matching box."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        frames_layout.addWidget(hint)
        layout.addWidget(frames_card)

        footer = QHBoxLayout()
        self.validation = QLabel("")
        self.validation.setObjectName("muted")
        footer.addWidget(self.validation, 1)
        save = ActionButton(
            "Confirm data & target",
            "fa6s.arrow-right",
            primary=True,
            tooltip="Validate coordinates and frame assignments, save this project, and unlock Reduction.",
        )
        save.clicked.connect(self._save)
        footer.addWidget(save)
        layout.addLayout(footer)
        layout.addStretch()
        self.scroll = _scroll_page(body)
        outer.addWidget(self.scroll, 1)
        self.records: list[FrameRecord] = []
        self.file_paths: list[str] = []
        self._lookup_requested_name = ""
        self._lookup_coordinate_snapshot = ("", "")
        self._last_resolved_coordinates: tuple[str, str] | None = None
        self.assignments: dict[str, list[str]] = {
            key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        }
        self.calibration_waivers = {key: False for key in ("bias", "dark", "flat")}

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose observing run")
        if folder:
            self.name.clear()
            self.ra.clear()
            self.dec.clear()
            self.target_source.setVisible(False)
            self._target_name_edited()
            self.folder.setText(folder)
            self.preview_folder(Path(folder))
            self.scan_progress.setRange(0, 0)
            self.scan_progress.setVisible(True)
            self.scanRequested.emit(Path(folder))

    def _target_name_edited(self) -> None:
        self._lookup_requested_name = ""
        self.target_lookup_status.setText("Enter a name to look up coordinates automatically.")
        self.target_lookup_status.setStyleSheet("")
        self.target_lookup_timer.stop()
        if len(self.name.text().strip()) >= 2:
            self.target_lookup_timer.start()

    def _request_target_lookup(self) -> None:
        self.target_lookup_timer.stop()
        name = self.name.text().strip()
        if not name or name.casefold() == self._lookup_requested_name.casefold():
            return
        self._lookup_requested_name = name
        self._lookup_coordinate_snapshot = (self.ra.text(), self.dec.text())
        self.target_lookup_status.setText(f"Looking up {name}…")
        self.target_lookup_status.setStyleSheet(f"color: {COLORS['cyan']};")
        self.targetLookupRequested.emit(name)

    def apply_target_resolution(self, requested_name: str, resolved: ResolvedTarget) -> None:
        if self.name.text().strip().casefold() != requested_name.strip().casefold():
            return
        current = (self.ra.text(), self.dec.text())
        safe_to_fill = current == self._lookup_coordinate_snapshot and (
            not any(current) or current == self._last_resolved_coordinates
        )
        if safe_to_fill:
            self.ra.setText(resolved.ra)
            self.dec.setText(resolved.dec)
            self._last_resolved_coordinates = (resolved.ra, resolved.dec)
            self.target_lookup_status.setText(f"Coordinates found via {resolved.source}.")
            self.target_lookup_status.setStyleSheet(f"color: {COLORS['green']};")
        else:
            self.target_lookup_status.setText(
                f"Coordinates found via {resolved.source}; existing RA/DEC were kept."
            )
            self.target_lookup_status.setStyleSheet(f"color: {COLORS['amber']};")

    def mark_current_coordinates_as_saved(self) -> None:
        """Allow a later successful name lookup to replace unchanged saved coordinates."""
        self._last_resolved_coordinates = (self.ra.text(), self.dec.text())

    def show_target_lookup_error(self, requested_name: str, message: str) -> None:
        if self.name.text().strip().casefold() != requested_name.strip().casefold():
            return
        self._lookup_requested_name = ""
        self.target_lookup_status.setText(message)
        self.target_lookup_status.setStyleSheet(f"color: {COLORS['amber']};")

    def set_records(self, records: list[FrameRecord]) -> None:
        self.scan_progress.setVisible(False)
        self.records = list(records)
        self.file_paths = [record.path for record in records]
        self._refresh_assignments()

    def populate_target_from_records(self, records: list[FrameRecord]) -> None:
        assigned_science = set(self.assignments["science"])
        candidates = sorted(
            records,
            key=lambda record: (
                record.path not in assigned_science,
                record.category != "science",
            ),
        )
        detected = next((record for record in candidates if record.target_ra and record.target_dec), None)
        if detected is None:
            self.target_source.setText(
                "No target coordinates were found in the FITS headers. Enter them manually."
            )
            self.target_source.setStyleSheet(f"color: {COLORS['amber']};")
            self.target_source.setVisible(True)
            return
        if detected.target_name:
            self.name.setText(detected.target_name)
        self.ra.setText(detected.target_ra)
        self.dec.setText(detected.target_dec)
        self.target_source.setText(f"Detected from FITS header · {detected.path}")
        self.target_source.setStyleSheet(f"color: {COLORS['green']};")
        self.target_source.setVisible(True)

    def preview_folder(self, root: Path) -> None:
        """Populate live counts from filenames before FITS header inspection finishes."""
        self.file_paths = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.casefold() in FITS_EXTENSIONS
            and not is_generated_project_path(path)
            and not any(part.startswith("reduction") or part.startswith("photometry") for part in path.parts)
        ]
        self._refresh_assignments()

    def set_project_actions_available(self, available: bool, *, busy: bool = False) -> None:
        self.project_actions.setVisible(available)
        self.reveal_project.setEnabled(available)
        self.reset_project.setEnabled(available and not busy)

    def clear_session(self) -> None:
        self.target_lookup_timer.stop()
        self.folder.clear()
        self.name.clear()
        self.ra.clear()
        self.dec.clear()
        self.target_source.clear()
        self.target_source.setVisible(False)
        self.target_lookup_status.setText("Enter a name to look up coordinates automatically.")
        self.target_lookup_status.setStyleSheet("")
        self._lookup_requested_name = ""
        self._lookup_coordinate_snapshot = ("", "")
        self._last_resolved_coordinates = None
        self.records = []
        self.file_paths = []
        for key, default in self.default_classifiers.items():
            editor = self.assignment_cards[key].classifier
            blocked = editor.blockSignals(True)
            editor.setText(default)
            editor.blockSignals(blocked)
        self.calibration_waivers = {key: False for key in ("bias", "dark", "flat")}
        self._set_assignments(
            {key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")}
        )
        self.scan_progress.setVisible(False)
        self.clear_section_errors()
        self.set_project_actions_available(False)

    def set_assignment_patterns(self, patterns: dict[str, str]) -> None:
        for key, value in patterns.items():
            if key in self.assignment_cards and value:
                editor = self.assignment_cards[key].classifier
                blocked = editor.blockSignals(True)
                editor.setText(str(value))
                editor.blockSignals(blocked)
        if self.file_paths:
            self._refresh_assignments()

    def restore_project_assignments(
        self,
        assignments: dict[str, list[str]],
        patterns: dict[str, str] | None = None,
        waivers: dict[str, bool] | None = None,
    ) -> None:
        """Restore saved assignments without requiring another FITS scan."""
        if patterns:
            self.set_assignment_patterns(patterns)
        keys = ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        restored = {key: list(assignments.get(key, [])) for key in keys}
        self.file_paths = list(dict.fromkeys(path for key in keys for path in restored[key]))
        self.records = []
        self._set_assignments(restored)
        decisions = waivers or {}
        self.calibration_waivers = {
            key: bool(decisions.get(key, False)) for key in ("bias", "dark", "flat")
        }

    def assignment_patterns(self) -> dict[str, str]:
        return {key: card.classifier.text().strip() for key, card in self.assignment_cards.items()}

    def _refresh_assignments(self) -> None:
        assignments = {key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")}
        order = ("bias", "dark", "flat", "science")
        patterns = {
            key: [token.strip().casefold() for token in card.classifier.text().split(",") if token.strip()]
            for key, card in self.assignment_cards.items()
        }
        for path in self.file_paths:
            stem = Path(path).stem.casefold()
            segments = [segment for segment in re.split(r"[^a-z0-9]+", stem) if segment]
            category = next(
                (
                    key
                    for key in order
                    if any(self._matches_classifier(stem, segments, token) for token in patterns[key])
                ),
                "unknown",
            )
            assignments[category].append(path)
        self._set_assignments(assignments)

    def _set_assignments(self, assignments: dict[str, list[str]]) -> None:
        self.assignments = assignments
        for key, card in self.assignment_cards.items():
            card.set_count(len(assignments[key]))
        assigned = sum(len(assignments[key]) for key in ("bias", "dark", "flat", "science"))
        unmatched = len(assignments["unknown"])
        if not self.file_paths:
            self.counts.setText("No FITS files scanned")
        else:
            self.counts.setText(f"{assigned} assigned · {unmatched} unmatched")

    @staticmethod
    def _matches_classifier(stem: str, segments: list[str], token: str) -> bool:
        if len(token) == 1:
            return any(segment == token or segment.startswith(token) for segment in segments)
        return token in stem

    def _save(self) -> None:
        self.saveRequested.emit(
            {
                "root": self.folder.text().strip(),
                "target_name": self.name.text().strip(),
                "ra": self.ra.text().strip(),
                "dec": self.dec.text().strip(),
                "waivers": dict(self.calibration_waivers),
                "assignments": {key: list(paths) for key, paths in self.assignments.items()},
                "frame_classifiers": self.assignment_patterns(),
            }
        )

    def clear_section_errors(self) -> None:
        for card in (self.folder_card, self.target_card, self.frames_card):
            card.setProperty("validationError", False)
            card.style().unpolish(card)
            card.style().polish(card)
        self.validation.clear()

    def show_error(self, message: str, section: str | None = None) -> None:
        self.clear_section_errors()
        self.validation.setText(message)
        self.validation.setStyleSheet(f"color: {COLORS['amber']};")
        cards = {
            "folder": self.folder_card,
            "target": self.target_card,
            "frames": self.frames_card,
        }
        card = cards.get(section or "")
        if card is not None:
            card.setProperty("validationError", True)
            card.style().unpolish(card)
            card.style().polish(card)
            QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(card, 20, 20))


class ProcessingPage(QWidget):
    runRequested = Signal(object)
    cancelRequested = Signal()

    def __init__(
        self, stage: StageID, title: str, subtitle: str, options: list[tuple[str, str]], parent=None
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PageHeader(title, subtitle))
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        option_card = QFrame()
        option_card.setObjectName("card")
        option_layout = QVBoxLayout(option_card)
        option_layout.setContentsMargins(18, 16, 18, 18)
        heading = QLabel("Processing options")
        heading.setObjectName("sectionTitle")
        option_layout.addWidget(heading)
        self.option_widgets: dict[str, QCheckBox] = {}
        for label, tooltip in options:
            row = QHBoxLayout()
            check = QCheckBox(label)
            check.setChecked(True)
            check.setToolTip(tooltip)
            row.addWidget(check)
            row.addWidget(InfoButton(tooltip))
            row.addStretch()
            option_layout.addLayout(row)
            self.option_widgets[label] = check
        layout.addWidget(option_card)

        progress_card = QFrame()
        progress_card.setObjectName("card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 16, 18, 18)
        top = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("sectionTitle")
        top.addWidget(self.status)
        top.addStretch()
        self.counter = QLabel("")
        self.counter.setObjectName("muted")
        top.addWidget(self.counter)
        progress_layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress details and checkpoints will appear here.")
        self.log.setMinimumHeight(220)
        progress_layout.addWidget(self.log)
        layout.addWidget(progress_card)

        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel = ActionButton(
            "Cancel",
            "fa6s.stop",
            tooltip="Stop after the current safe checkpoint. Completed outputs remain intact.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        self.run = ActionButton(
            f"Run {title}",
            "fa6s.play",
            primary=True,
            tooltip=f"Run {title.lower()} in the background without freezing the interface.",
        )
        self.run.clicked.connect(lambda: self.runRequested.emit(self.stage))
        actions.addWidget(self.cancel)
        actions.addWidget(self.run)
        layout.addLayout(actions)
        layout.addStretch()
        outer.addWidget(_scroll_page(body), 1)

    def set_busy(self, busy: bool) -> None:
        self.run.set_running(busy, f"Running {self.stage.value.replace('_', ' ').title()}…")
        self.cancel.set_cancel_active(busy)
        self.run.setEnabled(not busy)
        self.cancel.setEnabled(busy)

    def update_event(self, event: StageEvent) -> None:
        self.status.setText(event.message)
        self.progress.setValue(round(event.fraction * 100))
        self.counter.setText(f"{event.current} of {event.total}" if event.total else "")
        self.log.appendPlainText(event.message)

    def set_failure(self, failure: LEAPSError) -> None:
        self.status.setText(failure.title)
        self.status.setStyleSheet(f"color: {COLORS['amber']};")
        self.log.appendPlainText(f"{failure.code}: {failure.message}")
        self.log.appendPlainText("Next: " + " · ".join(failure.recovery))


class TimelineRow(QWidget):
    def __init__(self, time: str, text: str, status: str, detail: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("QWidget, QLabel { background: transparent; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(11)
        icon_names = {
            "ok": ("fa6s.circle-check", COLORS["green"]),
            "error": ("fa6s.circle-xmark", COLORS["red"]),
            "paused": ("fa6s.circle-pause", COLORS["amber"]),
            "waiting": ("fa6s.circle", COLORS["muted_2"]),
        }
        name, color = icon_names[status]
        picture = QLabel()
        picture.setPixmap(icon(name, color).pixmap(21, 21))
        picture.setFixedWidth(24)
        row.addWidget(picture)
        timestamp = QLabel(time)
        timestamp.setObjectName("muted")
        timestamp.setFixedWidth(62 if time else 0)
        timestamp.setVisible(bool(time))
        row.addWidget(timestamp)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        primary = QLabel(text)
        primary.setWordWrap(True)
        primary.setMinimumWidth(0)
        labels.addWidget(primary)
        if detail:
            secondary = QLabel(detail)
            secondary.setObjectName("muted")
            secondary.setWordWrap(True)
            labels.addWidget(secondary)
        row.addLayout(labels, 1)


class RecoveryInspector(QFrame):
    retryRequested = Signal()
    manualRequested = Signal()
    comparisonRequested = Signal()
    rankRequested = Signal()
    runRequested = Signal(list, float)
    cancelRequested = Signal()
    copyRequested = Signal()
    comparisonActiveChanged = Signal(int, bool)
    comparisonRemoved = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("recoveryInspector")
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)
        self.setStyleSheet(
            f"QFrame#recoveryInspector {{background: {COLORS['surface']}; border-left: 1px solid {COLORS['border_soft']};}}"
            "QLabel, QPushButton { font-size: 14px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.banner = QFrame()
        self.banner.setStyleSheet(
            f"background: {COLORS['surface_3']}; border-bottom: 1px solid {COLORS['border']};"
        )
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(19, 13, 15, 13)
        self.banner_icon = QLabel()
        self.banner_icon.setPixmap(icon("fa6s.crosshairs", COLORS["cyan"]).pixmap(24, 24))
        self.banner_title = QLabel("Photometry setup")
        self.banner_title.setStyleSheet(f"color: {COLORS['cyan']}; font-size: 16px; font-weight: 650;")
        self.banner_title.setMinimumWidth(0)
        self.banner_title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        banner_layout.addWidget(self.banner_icon)
        banner_layout.addWidget(self.banner_title, 1)
        layout.addWidget(self.banner)

        scroll_content = QWidget()
        scroll_content.setObjectName("recoveryContent")
        scroll_content.setStyleSheet(
            f"QWidget#recoveryContent {{ background: {COLORS['surface']}; }} QLabel {{ background: transparent; }}"
        )
        content = QVBoxLayout(scroll_content)
        content.setContentsMargins(20, 17, 20, 22)
        content.setSpacing(15)
        self.explanation = QLabel(
            "Select the target and comparison stars in the real reduced FITS image. Plate solving can locate the target automatically, but it is optional."
        )
        self.explanation.setWordWrap(True)
        self.explanation.setStyleSheet("font-size: 14px; line-height: 1.4;")
        content.addWidget(self.explanation)
        content.addWidget(self._divider())

        row = QHBoxLayout()
        heading = QLabel("Solve attempt timeline")
        heading.setObjectName("sectionTitle")
        row.addWidget(heading)
        row.addWidget(
            InfoButton(
                "Every bounded plate-solve attempt is recorded so you can see exactly where and why it stopped."
            )
        )
        row.addStretch()
        content.addLayout(row)
        self.timeline = QVBoxLayout()
        self._demo_timeline()
        content.addLayout(self.timeline)
        content.addWidget(self._divider())

        target_header = QHBoxLayout()
        target_title = QLabel("Target information")
        target_title.setObjectName("sectionTitle")
        target_header.addWidget(target_title)
        target_header.addWidget(InfoButton("These validated values were used for all solve attempts."))
        target_header.addStretch()
        content.addLayout(target_header)
        info = QGridLayout()
        info.setVerticalSpacing(10)
        info.setColumnStretch(1, 1)
        self.target_name = QLabel("Unnamed target")
        self.target_name.setWordWrap(True)
        self.coordinates = QLabel("Coordinates not set")
        self.coordinates.setWordWrap(True)
        self.pixel_scale = QLabel("Not set")
        self.pixel_scale.setWordWrap(True)
        for value in (self.target_name, self.coordinates, self.pixel_scale):
            value.setMinimumWidth(0)
        info.addWidget(QLabel("Target"), 0, 0)
        info.addWidget(self.target_name, 0, 1)
        info.addWidget(
            LabelWithInfo(
                "Coordinates (ICRS)", "Validated ICRS sky coordinates used as the canonical target identity."
            ),
            1,
            0,
        )
        info.addWidget(self.coordinates, 1, 1)
        info.addWidget(
            LabelWithInfo(
                "Pixel scale",
                "Estimated sky angle covered by each image pixel. A close estimate helps match stars to Gaia.",
            ),
            2,
            0,
        )
        info.addWidget(self.pixel_scale, 2, 1)
        content.addLayout(info)

        tooltip_card = QFrame()
        tooltip_card.setObjectName("card")
        tip_layout = QHBoxLayout(tooltip_card)
        tip_layout.setContentsMargins(12, 10, 12, 10)
        tip_layout.addWidget(
            InfoButton(
                "Pixel scale can usually be calculated from camera pixel size and telescope focal length."
            )
        )
        tip = QLabel(
            "Estimated sky angle covered by each image pixel. A close estimate helps match stars to Gaia."
        )
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        tip_layout.addWidget(tip, 1)
        content.addWidget(tooltip_card)

        self.retry = ActionButton(
            "Locate target from coordinates",
            "fa6s.rotate",
            primary=True,
            tooltip="Retry with validated coordinates and pixel scale, up to the remaining bounded attempts.",
        )
        self.retry.setMinimumHeight(44)
        self.retry.clicked.connect(self.retryRequested)
        self.manual = ActionButton(
            "Set target",
            "fa6s.crosshairs",
            tooltip="Click the target in the FITS image. Results remain clearly marked as unverified WCS.",
        )
        self.manual.setMinimumHeight(44)
        self.manual.setStyleSheet(
            f"color: {COLORS['cyan']}; border: 1px solid {COLORS['cyan']}; background: transparent;"
        )
        self.manual.clicked.connect(self.manualRequested)
        self.copy = ActionButton(
            "Copy diagnostics",
            "fa6s.copy",
            tooltip="Copy the failure code, diagnostic reference, attempt timeline, and validated inputs.",
        )
        self.copy.setStyleSheet(f"color: {COLORS['cyan']}; border: 0; background: transparent;")
        self.copy.clicked.connect(self.copyRequested)
        content.addWidget(self.retry)
        content.addWidget(self.manual)
        content.addWidget(self._divider())

        stars_heading = QHBoxLayout()
        stars_title = QLabel("Selected stars")
        stars_title.setObjectName("sectionTitle")
        stars_heading.addWidget(stars_title)
        stars_heading.addWidget(
            InfoButton(
                "HOPS refines each click to the nearest acceptable star and tracks it through the aligned sequence."
            )
        )
        stars_heading.addStretch()
        content.addLayout(stars_heading)
        self.target_selection = QLabel("Target: not selected")
        self.target_selection.setObjectName("muted")
        content.addWidget(self.target_selection)
        self.comparison_selection = QLabel("Comparisons: 0 selected")
        self.comparison_selection.setObjectName("muted")
        content.addWidget(self.comparison_selection)
        self.comparison_rows_widget = QWidget()
        self.comparison_rows = QVBoxLayout(self.comparison_rows_widget)
        self.comparison_rows.setContentsMargins(0, 0, 0, 0)
        self.comparison_rows.setSpacing(5)
        content.addWidget(self.comparison_rows_widget)
        self.add_comparison = ActionButton(
            "Add comparison star",
            "fa6s.plus",
            tooltip="Click another stable, unsaturated star in the FITS image.",
        )
        self.add_comparison.clicked.connect(self.comparisonRequested)
        content.addWidget(self.add_comparison)
        self.rank = ActionButton(
            "Suggest comparison stars",
            "fa6s.wand-magic-sparkles",
            tooltip="Rank nearby stars with usable brightness and automatically propose an ensemble.",
        )
        self.rank.clicked.connect(self.rankRequested)
        content.addWidget(self.rank)

        aperture_row = QHBoxLayout()
        aperture_row.addWidget(
            LabelWithInfo(
                "Aperture radius",
                "Radius in pixels. HOPS scales it with each frame's PSF when variable aperture is enabled.",
            )
        )
        self.aperture = QDoubleSpinBox()
        self.aperture.setRange(1.6, 100.0)
        self.aperture.setDecimals(1)
        self.aperture.setValue(8.0)
        self.aperture.setSuffix(" px")
        aperture_row.addWidget(self.aperture)
        content.addLayout(aperture_row)

        self.advanced_button = ActionButton(
            "Advanced settings",
            "fa6s.sliders",
            tooltip="Show the original HOPS aperture, sky, saturation, and camera controls.",
        )
        self.advanced_button.setCheckable(True)
        content.addWidget(self.advanced_button)
        self.advanced = QFrame()
        self.advanced.setObjectName("card")
        advanced_form = QFormLayout(self.advanced)
        advanced_form.setContentsMargins(12, 12, 12, 12)
        self.variable_aperture = QCheckBox("Scale aperture with PSF")
        self.variable_aperture.setChecked(True)
        self.geometric_center = QCheckBox("Use geometric center")
        self.sky_inner = QDoubleSpinBox()
        self.sky_inner.setRange(1.01, 20.0)
        self.sky_inner.setValue(1.7)
        self.sky_outer = QDoubleSpinBox()
        self.sky_outer.setRange(1.02, 30.0)
        self.sky_outer.setValue(2.4)
        self.saturation = QDoubleSpinBox()
        self.saturation.setRange(0.01, 1.0)
        self.saturation.setSingleStep(0.05)
        self.saturation.setValue(0.95)
        self.camera_gain = QDoubleSpinBox()
        self.camera_gain.setRange(0.01, 1000.0)
        self.camera_gain.setValue(1.0)
        self.camera_gain.setSuffix(" e⁻/ADU")
        advanced_form.addRow(self.variable_aperture)
        advanced_form.addRow(self.geometric_center)
        advanced_form.addRow("Inner sky ring", self.sky_inner)
        advanced_form.addRow("Outer sky ring", self.sky_outer)
        advanced_form.addRow("Saturation fraction", self.saturation)
        advanced_form.addRow("Camera gain", self.camera_gain)
        self.advanced.setVisible(False)
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        content.addWidget(self.advanced)

        self.run = ActionButton(
            "Run HOPS photometry",
            "fa6s.play",
            primary=True,
            tooltip="Track the selected stars and calculate aperture and Gaussian light curves in the background.",
        )
        self._busy = False
        self.run.setEnabled(False)
        self.run.clicked.connect(self._run)
        self.cancel = ActionButton(
            "Cancel",
            "fa6s.stop",
            tooltip="Stop after the current safe checkpoint. Completed outputs remain intact.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        content.addWidget(self.run)
        content.addWidget(self.cancel)
        content.addWidget(self.copy)
        content.addStretch()
        scroll = QScrollArea()
        scroll.setStyleSheet(f"QScrollArea {{ background: {COLORS['surface']}; border: 0; }}")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

    def _demo_timeline(self) -> None:
        self.timeline.addWidget(
            TimelineRow("", "Plate solve is optional · manual selection is ready", "waiting")
        )

    def set_retrying(self) -> None:
        self.retry.setEnabled(False)
        self.retry.setText("Locating target…")

    def set_failure(self, failure: LEAPSError) -> None:
        """Replace the demo context with the exact typed runtime failure."""
        self.banner_title.setText(failure.title)
        self.banner_icon.setPixmap(
            icon("fa6s.triangle-exclamation", COLORS["amber"]).pixmap(24, 24)
        )
        self.banner.setStyleSheet(
            f"background: #6d4703; border-bottom: 1px solid {COLORS['amber_dark']};"
        )
        self.banner_title.setStyleSheet(
            f"color: {COLORS['amber']}; font-size: 16px; font-weight: 650;"
        )
        self.explanation.setText(failure.message)
        while self.timeline.count():
            item = self.timeline.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        current = QTime.currentTime().toString("HH:mm:ss")
        self.timeline.addWidget(TimelineRow(current, "Coordinates and pixel scale validated", "ok"))
        details = [line.strip() for line in failure.technical_details.splitlines() if line.strip()]
        for detail in details[-3:]:
            title, _, secondary = detail.partition(":")
            self.timeline.addWidget(TimelineRow(current, title or failure.code, "error", secondary.strip()))
        self.timeline.addWidget(TimelineRow(current, f"Stopped safely · {failure.diagnostic_id}", "paused"))
        self.retry.setEnabled(True)
        self.retry.setText("Retry plate solve")

    def set_project_target(self, name: str, coordinates: str, pixel_scale: float) -> None:
        self.target_name.setText(name or "Unnamed target")
        self.coordinates.setText(coordinates or "Coordinates not set")
        self.pixel_scale.setText(
            f"{pixel_scale:.2f} arcsec/pixel" if pixel_scale > 0 else "Estimated from stellar PSF"
        )

    def set_target_selected(self, x: float, y: float, verified: bool) -> None:
        suffix = "plate solved" if verified else "manual"
        self.target_selection.setText(f"Target: x {x:.1f}, y {y:.1f} · {suffix}")
        self._update_run_state()

    def set_comparisons(
        self, comparisons: list[tuple[float, float]], active: list[bool]
    ) -> None:
        while self.comparison_rows.count():
            item = self.comparison_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        active_count = sum(active)
        self.comparison_selection.setText(
            f"Comparisons: {active_count} active · {len(comparisons)} selected"
        )
        for index, ((x, y), enabled) in enumerate(zip(comparisons, active, strict=True)):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            use = QCheckBox(f"C{index + 1} · x {x:.1f}, y {y:.1f}")
            use.setChecked(enabled)
            use.setToolTip("Include this comparison star in the differential-flux ensemble.")
            use.toggled.connect(
                lambda checked, row_index=index: self.comparisonActiveChanged.emit(
                    row_index, checked
                )
            )
            remove = QPushButton()
            remove.setIcon(icon("fa6s.xmark", COLORS["muted"]))
            remove.setToolTip("Remove this comparison star.")
            remove.setAccessibleName(f"Remove comparison star {index + 1}")
            remove.setFixedSize(28, 28)
            remove.clicked.connect(
                lambda _checked=False, row_index=index: self.comparisonRemoved.emit(row_index)
            )
            row_layout.addWidget(use, 1)
            row_layout.addWidget(remove)
            self.comparison_rows.addWidget(row)
        self._update_run_state()

    def _update_run_state(self) -> None:
        self.run.setEnabled(
            not self._busy
            and "not selected" not in self.target_selection.text()
            and not self.comparison_selection.text().startswith("Comparisons: 0 active")
        )

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.run.set_running(busy, "Running Photometry…")
        self.cancel.set_cancel_active(busy)
        self.cancel.setEnabled(busy)
        self._update_run_state()

    def _run(self) -> None:
        self.runRequested.emit([], self.aperture.value())

    def photometry_config(self) -> dict[str, float | bool]:
        return {
            "aperture_radius": self.aperture.value(),
            "sky_inner_aperture": self.sky_inner.value(),
            "sky_outer_aperture": self.sky_outer.value(),
            "saturation_fraction": self.saturation.value(),
            "camera_gain": self.camera_gain.value(),
            "variable_aperture": self.variable_aperture.isChecked(),
            "geometric_center": self.geometric_center.isChecked(),
        }

    def apply_photometry_config(self, values: dict[str, Any]) -> None:
        if not values:
            return
        self.aperture.setValue(float(values.get("aperture_radius", self.aperture.value())))
        self.sky_inner.setValue(float(values.get("sky_inner_aperture", self.sky_inner.value())))
        self.sky_outer.setValue(float(values.get("sky_outer_aperture", self.sky_outer.value())))
        self.saturation.setValue(float(values.get("saturation_fraction", self.saturation.value())))
        self.camera_gain.setValue(float(values.get("camera_gain", self.camera_gain.value())))
        self.variable_aperture.setChecked(
            bool(values.get("variable_aperture", self.variable_aperture.isChecked()))
        )
        self.geometric_center.setChecked(
            bool(values.get("geometric_center", self.geometric_center.isChecked()))
        )

    def _restore_retry(self) -> None:
        self.retry.setEnabled(True)
        self.retry.setText("Locate target from coordinates")

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {COLORS['border_soft']};")
        return line


class PlateSolvePage(QWidget):
    retryRequested = Signal()
    copyDiagnosticsRequested = Signal()
    manualTargetPlaced = Signal(float, float)
    starSelectionRequested = Signal(str, float, float)
    rankRequested = Signal()
    runRequested = Signal(list, float)
    selectionChanged = Signal()

    def __init__(self, asset: Path, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Photometry",
                "Select the target and comparison stars, then run the original HOPS measurement workflow.",
            )
        )
        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(0)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(16, 14, 16, 14)
        self.workspace = FITSWorkspace(asset)
        center_layout.addWidget(self.workspace)
        split.addWidget(center, 1)
        self.inspector = RecoveryInspector()
        self.target: tuple[float, float] | None = None
        self.target_label = "Target"
        self.target_verified = False
        self.comparisons: list[tuple[float, float]] = []
        self.comparison_active: list[bool] = []
        split.addWidget(self.inspector)
        outer.addLayout(split, 1)
        self.inspector.retryRequested.connect(self._retry)
        self.inspector.manualRequested.connect(self.workspace.begin_manual_target)
        self.inspector.comparisonRequested.connect(
            lambda: self.workspace.begin_selection("comparison")
        )
        self.inspector.rankRequested.connect(self.rankRequested)
        self.inspector.runRequested.connect(
            lambda _ignored, radius: self.runRequested.emit(
                [
                    comparison
                    for comparison, active in zip(
                        self.comparisons, self.comparison_active, strict=True
                    )
                    if active
                ],
                radius,
            )
        )
        self.inspector.comparisonActiveChanged.connect(self.set_comparison_active)
        self.inspector.comparisonRemoved.connect(self.remove_comparison)
        self.inspector.aperture.valueChanged.connect(self.set_aperture_radius)
        self.inspector.sky_inner.valueChanged.connect(self.sky_ring_changed)
        self.inspector.sky_outer.valueChanged.connect(self.sky_ring_changed)
        for control in (
            self.inspector.variable_aperture,
            self.inspector.geometric_center,
            self.inspector.saturation,
            self.inspector.camera_gain,
        ):
            if isinstance(control, QCheckBox):
                control.toggled.connect(lambda _checked: self.selectionChanged.emit())
            else:
                control.valueChanged.connect(lambda _value: self.selectionChanged.emit())
        self.inspector.copyRequested.connect(self.copyDiagnosticsRequested)
        self.workspace.pointSelected.connect(self.starSelectionRequested)

    def _retry(self) -> None:
        self.inspector.set_retrying()
        self.retryRequested.emit()

    def set_target(self, x: float, y: float, *, radius: float, label: str, verified: bool) -> None:
        self.target = (x, y)
        self.target_label = label
        self.target_verified = verified
        self.workspace.place_target_marker(
            x,
            y,
            radius,
            label,
            sky_inner=self.inspector.sky_inner.value(),
            sky_outer=self.inspector.sky_outer.value(),
        )
        self.inspector.aperture.setValue(radius)
        self.inspector.set_target_selected(x, y, verified)
        self.selectionChanged.emit()

    def add_comparison(
        self, x: float, y: float, *, radius: float, active: bool = True
    ) -> None:
        if any((cx - x) ** 2 + (cy - y) ** 2 < 4 for cx, cy in self.comparisons):
            return
        self.comparisons.append((x, y))
        self.comparison_active.append(active)
        self._refresh_comparisons(radius)
        self.selectionChanged.emit()

    def set_comparison_active(self, index: int, active: bool) -> None:
        if 0 <= index < len(self.comparison_active):
            self.comparison_active[index] = active
            self.workspace.image.set_marker_active(f"comparison-{index + 1}", active)
            self.inspector.set_comparisons(self.comparisons, self.comparison_active)
            self.selectionChanged.emit()

    def remove_comparison(self, index: int) -> None:
        if 0 <= index < len(self.comparisons):
            self.comparisons.pop(index)
            self.comparison_active.pop(index)
            self._refresh_comparisons(self.inspector.aperture.value())
            self.selectionChanged.emit()

    def _refresh_comparisons(self, radius: float) -> None:
        for key in list(self.workspace.image.marker_items):
            if key.startswith("comparison-"):
                self.workspace.image.remove_marker(key)
        for index, ((x, y), active) in enumerate(
            zip(self.comparisons, self.comparison_active, strict=True), start=1
        ):
            self.workspace.place_comparison_marker(
                index,
                x,
                y,
                radius,
                active=active,
                sky_inner=self.inspector.sky_inner.value(),
                sky_outer=self.inspector.sky_outer.value(),
            )
        self.inspector.set_comparisons(self.comparisons, self.comparison_active)

    def set_aperture_radius(self, radius: float) -> None:
        self.refresh_aperture_overlays()
        self.selectionChanged.emit()

    def sky_ring_changed(self, _value: float) -> None:
        self.refresh_aperture_overlays()
        self.selectionChanged.emit()

    def refresh_aperture_overlays(self) -> None:
        radius = self.inspector.aperture.value()
        if self.target:
            self.workspace.place_target_marker(
                self.target[0],
                self.target[1],
                radius,
                self.target_label,
                sky_inner=self.inspector.sky_inner.value(),
                sky_outer=self.inspector.sky_outer.value(),
            )
        self._refresh_comparisons(radius)

    def set_candidates(self, candidates: list[dict[str, float]]) -> None:
        for candidate in candidates[:5]:
            self.add_comparison(
                float(candidate["x"]),
                float(candidate["y"]),
                radius=self.inspector.aperture.value(),
            )

    def clear_selection(self) -> None:
        self.target = None
        self.target_verified = False
        self.comparisons = []
        self.comparison_active = []
        self.workspace.image.clear_markers()
        self.inspector.target_selection.setText("Target: not selected")
        self.inspector.set_comparisons([], [])


class LightCurvePage(QWidget):
    selectionChanged = Signal(list)
    continueRequested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preview_pixmap = QPixmap()
        self._updating = False
        self.comparison_checks: list[QCheckBox] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Light Curve",
                "Review the target and comparison-star curves, exclude anomalous comparisons, then continue to fitting.",
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("card")
        controls.setMinimumWidth(275)
        controls.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Active stars")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "The target is always retained. Uncheck comparison stars whose light curves show trends, jumps, or unusual scatter."
            )
        )
        heading.addStretch()
        controls_layout.addLayout(heading)
        self.summary = QLabel("Run Photometry to generate the individual light curves.")
        self.summary.setObjectName("muted")
        self.summary.setWordWrap(True)
        controls_layout.addWidget(self.summary)
        self.selection_widget = QWidget()
        self.selection_layout = QVBoxLayout(self.selection_widget)
        self.selection_layout.setContentsMargins(0, 8, 0, 8)
        self.selection_layout.setSpacing(10)
        controls_layout.addWidget(self.selection_widget)
        controls_layout.addStretch()
        self.message = QLabel()
        self.message.setObjectName("muted")
        self.message.setWordWrap(True)
        controls_layout.addWidget(self.message)
        self.continue_button = ActionButton(
            "Continue to Fitting",
            "fa6s.arrow-right",
            primary=True,
            tooltip="Save the active comparison ensemble and build the approved light curve used by Fitting and exports.",
        )
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(
            lambda: self.continueRequested.emit(self.active_comparisons())
        )
        controls_layout.addWidget(self.continue_button)
        layout.addWidget(controls)

        plot_card = QFrame()
        plot_card.setObjectName("card")
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(18, 16, 18, 18)
        plot_title = QHBoxLayout()
        label = QLabel("Differential light curves")
        label.setObjectName("sectionTitle")
        plot_title.addWidget(label)
        plot_title.addWidget(
            InfoButton(
                "Target is divided by the active comparison ensemble. Each active comparison is divided by the other active comparisons, matching HOPS."
            )
        )
        plot_title.addStretch()
        plot_layout.addLayout(plot_title)
        self.preview_image = QLabel("Individual light curves will appear here after Photometry.")
        self.preview_image.setObjectName("muted")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(520, 420)
        self.preview_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout.addWidget(self.preview_image, 1)
        layout.addWidget(plot_card, 1)
        outer.addWidget(body, 1)

    def set_review(self, result: Any) -> None:
        self._updating = True
        while self.selection_layout.count():
            item = self.selection_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.comparison_checks = []
        target = QCheckBox("Target")
        target.setChecked(True)
        target.setEnabled(False)
        target.setToolTip("The target star is always included.")
        self.selection_layout.addWidget(target)
        for index, active in enumerate(result.active_comparisons, start=1):
            checkbox = QCheckBox(f"C{index}")
            checkbox.setChecked(bool(active))
            checkbox.setAccessibleName(f"Use comparison star C{index}")
            checkbox.setToolTip(
                f"Include C{index} in the comparison ensemble used for the target light curve."
            )
            checkbox.toggled.connect(self._selection_toggled)
            self.comparison_checks.append(checkbox)
            self.selection_layout.addWidget(checkbox)
        self._updating = False
        self._preview_pixmap = QPixmap(str(result.preview_path))
        self._render_preview()
        self.summary.setText(
            f"{result.frame_count} frames · {sum(result.active_comparisons)} of "
            f"{len(result.active_comparisons)} comparisons active"
        )
        failed = [
            f"{curve.label}: {curve.missing_frames} missing"
            for curve in result.curves
            if curve.missing_frames
        ]
        self.message.setText(" · ".join(failed) if failed else "All selected stars were measured in every frame.")
        self.continue_button.setEnabled(bool(result.active_comparisons))

    def active_comparisons(self) -> list[bool]:
        return [checkbox.isChecked() for checkbox in self.comparison_checks]

    def _selection_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        if not checked and not any(self.active_comparisons()):
            checkbox = self.sender()
            if isinstance(checkbox, QCheckBox):
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
            self.message.setText("At least one comparison star must remain active.")
            return
        self.selectionChanged.emit(self.active_comparisons())

    def show_failure(self, failure: LEAPSError) -> None:
        self.message.setText(f"{failure.title}: {failure.message}")
        self.continue_button.setEnabled(False)

    def _render_preview(self) -> None:
        if self._preview_pixmap.isNull():
            return
        self.preview_image.setPixmap(
            self._preview_pixmap.scaled(
                self.preview_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview()


class FittingPage(QWidget):
    previewRequested = Signal(dict)
    fullFitRequested = Signal(dict)
    planetSearchRequested = Signal(str)
    cancelRequested = Signal()
    viewInFilesRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Fitting",
                "Preview the transit model first, then run a full uncertainty analysis when the setup looks right.",
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        form_card = QFrame()
        form_card.setObjectName("card")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(18, 16, 18, 18)
        selection_form = QFormLayout()
        selection_form.setSpacing(10)
        self.light_curve = QComboBox()
        self.light_curve.addItem("Aperture photometry", "aperture")
        self.light_curve.addItem("Gaussian photometry", "gaussian")
        selection_form.addRow(
            LabelWithInfo(
                "Light curve",
                "Choose which approved Light Curve output to fit. Aperture photometry preserves LEAPS' existing default.",
            ),
            self.light_curve,
        )
        form_layout.addLayout(selection_form)
        title = QLabel("Planet parameters")
        title.setObjectName("sectionTitle")
        form_layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(10)
        self._parameters: dict[str, PlanetParameters] = {}
        self._preview_valid = False
        self._busy = False
        self._preview_pixmap = QPixmap()
        self._rendered_preview_pixmap = QPixmap()
        self._preview_path: Path | None = None
        self.planet = QComboBox()
        self.planet.setEditable(True)
        self.planet.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.planet.setPlaceholderText("Loading from the selected target…")
        self.planet.lineEdit().returnPressed.connect(
            lambda: self.planetSearchRequested.emit(self.planet.currentText().strip())
        )
        self.period = QDoubleSpinBox()
        self.period.setDecimals(8)
        self.period.setRange(0.000001, 100000)
        self.mid_time = QDoubleSpinBox()
        self.mid_time.setDecimals(8)
        self.mid_time.setRange(0, 4_000_000)
        self.depth = QDoubleSpinBox()
        self.depth.setDecimals(5)
        self.depth.setRange(0, 1)
        self.filter = QComboBox()
        self.filter.setEditable(True)
        self.filter.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for label, identifier in passband_choices():
            self.filter.addItem(label, identifier)
        self.filter.setCurrentIndex(-1)
        self.filter.setPlaceholderText("Detecting from science FITS…")
        self.detrending = QComboBox()
        self.detrending.addItem("Airmass", "airmass")
        self.detrending.addItem("Quadratic", "quadratic")
        self.detrending.addItem("Linear", "linear")
        self.detrending.setCurrentIndex(self.detrending.findData("linear"))
        form.addRow(
            LabelWithInfo(
                "Planet",
                "Resolve parameters from ExoClock, then the bundled NASA snapshot, or reuse validated manual values.",
            ),
            self.planet,
        )
        form.addRow(LabelWithInfo("Period (days)", "Time between consecutive transits."), self.period)
        form.addRow(
            LabelWithInfo("Mid-transit (BJD)", "Expected center of the transit in barycentric Julian date."),
            self.mid_time,
        )
        form.addRow(
            LabelWithInfo("Expected depth", "Approximate fractional loss of light during transit."),
            self.depth,
        )
        form.addRow(
            LabelWithInfo(
                "Observation filter",
                "Detected from the assigned science FITS headers and translated to the original HOPS passband name.",
            ),
            self.filter,
        )
        form.addRow(
            LabelWithInfo(
                "De-trending",
                "Remove a trend using airmass, a quadratic time curve, or a linear time curve. LEAPS keeps its existing automatic default.",
            ),
            self.detrending,
        )
        form_layout.addLayout(form)

        metadata_layout = QVBoxLayout()
        metadata_layout.setContentsMargins(0, 4, 0, 4)
        metadata_layout.setSpacing(10)

        catalog_card = QFrame()
        catalog_card.setObjectName("fittingMetadataCard")
        catalog_card.setStyleSheet(
            f"QFrame#fittingMetadataCard {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 7px; }}"
        )
        catalog_layout = QVBoxLayout(catalog_card)
        catalog_layout.setContentsMargins(12, 10, 12, 11)
        catalog_layout.setSpacing(5)
        catalog_title = QLabel("Catalog")
        catalog_title.setObjectName("eyebrow")
        catalog_layout.addWidget(catalog_title)
        self.catalog_source = QLabel("Planet parameters have not been loaded.")
        self.catalog_source.setObjectName("muted")
        self.catalog_source.setWordWrap(True)
        self.catalog_source.setMinimumHeight(36)
        self.catalog_source.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        catalog_layout.addWidget(self.catalog_source)
        metadata_layout.addWidget(catalog_card)

        observation_card = QFrame()
        observation_card.setObjectName("fittingMetadataCard")
        observation_card.setStyleSheet(
            f"QFrame#fittingMetadataCard {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 7px; }}"
        )
        observation_layout = QVBoxLayout(observation_card)
        observation_layout.setContentsMargins(12, 10, 12, 11)
        observation_layout.setSpacing(5)
        observation_title = QLabel("Observation")
        observation_title.setObjectName("eyebrow")
        observation_layout.addWidget(observation_title)
        self.observation_source = QLabel("Science-frame metadata has not been loaded.")
        self.observation_source.setObjectName("muted")
        self.observation_source.setWordWrap(True)
        self.observation_source.setMinimumHeight(54)
        self.observation_source.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        observation_layout.addWidget(self.observation_source)
        metadata_layout.addWidget(observation_card)
        form_layout.addLayout(metadata_layout)
        self.advanced_toggle = QPushButton("Advanced MCMC controls")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip(
            "Show sampling controls. Defaults are recommended for most observing runs."
        )
        form_layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        self.iterations = QSpinBox()
        self.iterations.setRange(100, 1_000_000)
        self.iterations.setValue(5000)
        self.burn = QSpinBox()
        self.burn.setRange(0, 900_000)
        self.burn.setValue(1000)
        advanced_form.addRow(
            LabelWithInfo("Iterations", "Samples generated per walker for the full fit."), self.iterations
        )
        advanced_form.addRow(
            LabelWithInfo("Burn-in", "Initial samples discarded before summarizing the posterior."), self.burn
        )
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        form_layout.addWidget(self.advanced)
        form_layout.addStretch()
        buttons = QHBoxLayout()
        self.preview = ActionButton(
            "Preview Fit",
            "fa6s.chart-line",
            primary=True,
            tooltip="Run a quick deterministic fit to validate data, timing, and priors.",
        )
        self.preview.clicked.connect(lambda: self.previewRequested.emit(self.values()))
        self.full = ActionButton(
            "Run Full Fit",
            "fa6s.play",
            tooltip="Run the full MCMC uncertainty analysis in the background.",
        )
        self.full.clicked.connect(lambda: self.fullFitRequested.emit(self.values()))
        self.full.setEnabled(False)
        self.cancel = ActionButton(
            "Cancel",
            "fa6s.stop",
            tooltip="Stop after the current safe checkpoint. Completed outputs remain intact.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.preview)
        buttons.addWidget(self.full)
        form_layout.addLayout(buttons)
        layout.addWidget(form_card, 2)
        result = QFrame()
        result.setObjectName("card")
        self.result_card = result
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(18, 16, 18, 18)
        result_title = QLabel("Fit preview")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)
        self.message = QLabel(
            "Run Preview Fit to inspect the model and residuals before committing time to the full fit."
        )
        self.message.setWordWrap(True)
        self.message.setObjectName("muted")
        result_layout.addWidget(self.message)
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 100)
        self.fit_progress.setValue(0)
        self.fit_progress.setTextVisible(True)
        self.fit_progress.setVisible(False)
        result_layout.addWidget(self.fit_progress)
        self.progress_details = QLabel()
        self.progress_details.setObjectName("muted")
        self.progress_details.setWordWrap(True)
        self.progress_details.setVisible(False)
        result_layout.addWidget(self.progress_details)
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(420, 320)
        self.preview_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_image.setVisible(False)
        result_layout.addWidget(self.preview_image, 1)
        preview_actions = QHBoxLayout()
        preview_actions.addStretch()
        self.view_in_files = ActionButton(
            "View in Files",
            "fa6s.folder-open",
            tooltip="Reveal this fit-preview image in Finder or File Explorer.",
        )
        self.view_in_files.setMinimumWidth(160)
        self.view_in_files.setEnabled(False)
        self.view_in_files.clicked.connect(self._request_view_in_files)
        preview_actions.addWidget(self.view_in_files)
        result_layout.addLayout(preview_actions)
        result_layout.addStretch()
        layout.addWidget(result, 3)
        outer.addWidget(_scroll_page(body), 1)

        for control in (
            self.planet,
            self.period,
            self.mid_time,
            self.depth,
            self.filter,
            self.light_curve,
            self.detrending,
            self.iterations,
            self.burn,
        ):
            if isinstance(control, QComboBox):
                control.currentTextChanged.connect(self.invalidate_preview)
            else:
                control.valueChanged.connect(self.invalidate_preview)
        self.planet.activated.connect(self._planet_activated)
        self.reset_setup("Loading the selected target and science-frame metadata…")

    def values(self) -> dict[str, Any]:
        planet_name = self.planet.currentText().strip()
        parameters = next(
            (
                value
                for name, value in self._parameters.items()
                if name.casefold() == planet_name.casefold()
            ),
            None,
        )
        return {
            "planet": planet_name,
            "catalog_parameters": parameters,
            "period": self.period.value(),
            "mid_time": self.mid_time.value(),
            "depth": self.depth.value(),
            "filter": self.filter.currentData()
            or normalize_filter(self.filter.currentText()),
            "light_curve": self.light_curve.currentData(),
            "detrending": self.detrending.currentData(),
            "iterations": self.iterations.value(),
            "burn": self.burn.value(),
        }

    def set_fitting_options(self, light_curve: str, detrending: str) -> None:
        for control, value in (
            (self.light_curve, light_curve),
            (self.detrending, detrending),
        ):
            index = control.findData(value)
            if index >= 0:
                blocked = control.blockSignals(True)
                control.setCurrentIndex(index)
                control.blockSignals(blocked)

    def set_loading(self, message: str) -> None:
        self.catalog_source.setText(message)
        self.preview.setEnabled(False)
        self.full.setEnabled(False)

    def reset_setup(self, message: str) -> None:
        self._parameters = {}
        self._preview_valid = False
        self._preview_pixmap = QPixmap()
        self._rendered_preview_pixmap = QPixmap()
        self._preview_path = None
        self.planet.blockSignals(True)
        self.planet.clear()
        self.planet.blockSignals(False)
        self.filter.blockSignals(True)
        self.filter.setCurrentIndex(-1)
        self.filter.blockSignals(False)
        self.observation_source.setText("Science-frame metadata has not been loaded.")
        self.preview_image.clear()
        self.preview_image.setVisible(False)
        self.view_in_files.setEnabled(False)
        self.message.setText(
            "Run Preview Fit to inspect the model and residuals before committing time to the full fit."
        )
        self.set_loading(message)

    def set_planet_candidates(
        self, candidates: list[PlanetParameters], selected_name: str = ""
    ) -> None:
        self._parameters = {parameters.name: parameters for parameters in candidates}
        self.planet.blockSignals(True)
        self.planet.clear()
        for parameters in candidates:
            self.planet.addItem(parameters.name)
        selected = next(
            (
                index
                for index, parameters in enumerate(candidates)
                if selected_name
                and parameters.name.casefold() == selected_name.casefold()
            ),
            0,
        )
        self.planet.setCurrentIndex(selected if candidates else -1)
        self.planet.blockSignals(False)
        if candidates:
            self._apply_parameters(candidates[selected])
        else:
            self.catalog_source.setText(
                "No catalog match was found for the saved target coordinates. Check Data & Target, then retry."
            )
        self._refresh_actions()

    def _planet_activated(self, index: int) -> None:
        if index >= 0:
            parameters = self._parameters.get(self.planet.itemText(index))
            if parameters:
                self._apply_parameters(parameters)

    def _apply_parameters(self, parameters: PlanetParameters) -> None:
        for control in (self.period, self.mid_time, self.depth):
            control.blockSignals(True)
        self.period.setValue(parameters.period)
        self.mid_time.setValue(parameters.mid_time)
        self.depth.setValue(parameters.rp_over_rs**2)
        for control in (self.period, self.mid_time, self.depth):
            control.blockSignals(False)
        dated = f" · snapshot {parameters.source_date}" if parameters.source_date else ""
        self.catalog_source.setText(f"{parameters.source}{dated} · matched to project coordinates")
        self.invalidate_preview()

    def set_observation_metadata(
        self,
        filter_name: str | None,
        exposure_time: float | None,
        *,
        filter_status: str = "detected",
    ) -> None:
        canonical = normalize_filter(filter_name) if filter_name else None
        self.filter.blockSignals(True)
        index = self.filter.findData(canonical) if canonical else -1
        self.filter.setCurrentIndex(index)
        if canonical and index < 0:
            self.filter.setEditText(canonical)
        self.filter.blockSignals(False)
        exposure = f"{exposure_time:g} s exposures" if exposure_time else "exposure time unavailable"
        if canonical:
            self.observation_source.setText(
                f"{passband_label(canonical)} ({canonical}) · {exposure} · detected from science FITS"
            )
        elif filter_status == "mixed":
            self.observation_source.setText(
                f"Science FITS contain multiple filters · {exposure}. Choose the passband for this light curve."
            )
        else:
            self.observation_source.setText(
                f"No recognized FITS filter was found · {exposure}. Choose the passband used for the observation."
            )
        self.invalidate_preview()

    def set_busy(self, busy: bool, *, full: bool = False) -> None:
        self._busy = busy
        self.preview.set_running(busy and not full, "Running Preview…")
        self.full.set_running(busy and full, "Running Full Fit…")
        self.cancel.set_cancel_active(busy)
        self.cancel.setEnabled(busy)
        self.cancel.setText("Cancel")
        if busy:
            self.message.setText("Running full uncertainty fit…" if full else "Building fit preview…")
            self.fit_progress.setRange(0, 0)
            self.fit_progress.setVisible(True)
            self.progress_details.clear()
            self.progress_details.setVisible(True)
        else:
            self.fit_progress.setVisible(False)
            self.progress_details.setVisible(False)
        self._refresh_actions()

    def set_stopping(self) -> None:
        self.cancel.setText("Stopping…")
        self.cancel.setEnabled(False)
        self.message.setText("Stopping safely and discarding the incomplete fitting attempt…")

    def update_event(self, event: StageEvent) -> None:
        self.fit_progress.setVisible(True)
        self.progress_details.setVisible(True)
        details = event.details
        phase = details.get("phase", event.checkpoint or "")
        walkers = details.get("walkers")
        elapsed = _format_duration(details.get("elapsed_seconds"))
        eta = _format_duration(details.get("eta_seconds"))
        if phase == "sampling" and event.total > 0:
            self.fit_progress.setRange(0, event.total)
            self.fit_progress.setValue(min(event.current, event.total))
            self.fit_progress.setFormat(f"{event.current:,} of {event.total:,} MCMC steps")
            self.message.setText("Sampling the posterior distribution…")
        elif phase == "writing_results":
            self.fit_progress.setRange(0, 1)
            self.fit_progress.setValue(min(event.current, 1))
            self.fit_progress.setFormat("Writing results…")
            self.message.setText("Writing the completed fit without replacing the last result early…")
        else:
            self.fit_progress.setRange(0, 0)
            self.message.setText(f"{event.message}…")
        parts = []
        if walkers:
            parts.append(f"{int(walkers)} automatic HOPS walkers")
        if elapsed:
            parts.append(f"elapsed {elapsed}")
        if eta:
            parts.append(f"about {eta} remaining")
        self.progress_details.setText(" · ".join(parts))

    def show_cancelled(self, message: str) -> None:
        self.message.setText(message)
        self._refresh_actions()

    def invalidate_preview(self, *_args: Any) -> None:
        self._preview_valid = False
        self._refresh_actions()

    def show_preview(
        self, path: Path, *, planet: str, passband: str, residual_std: float | None
    ) -> None:
        self._preview_path = Path(path)
        self._preview_pixmap = QPixmap(str(self._preview_path))
        available = self._preview_path.is_file() and not self._preview_pixmap.isNull()
        self.preview_image.setVisible(available)
        if available:
            self._render_preview()
        else:
            self.preview_image.clear()
        self.view_in_files.setEnabled(available)
        residual = f" · residual scatter {residual_std:.5f}" if residual_std is not None else ""
        self.message.setText(
            f"Preview ready for {planet} using {passband}{residual}. Review the curve and residuals, then run the full fit."
        )
        self._preview_valid = available
        self._refresh_actions()

    def show_failure(self, message: str) -> None:
        self.message.setText(message)
        self._preview_valid = False
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        ready = bool(self._parameters) and bool(
            self.filter.currentData() or normalize_filter(self.filter.currentText())
        )
        self.preview.set_primary(not self._preview_valid)
        self.full.set_primary(self._preview_valid)
        self.preview.setEnabled(ready and not self._busy)
        self.full.setEnabled(ready and self._preview_valid and not self._busy)

    def _render_preview(self) -> None:
        if self._preview_pixmap.isNull():
            return
        logical_size = self.preview_image.size()
        pixel_ratio = self._preview_device_pixel_ratio()
        physical_size = QSize(
            max(1, round(logical_size.width() * pixel_ratio)),
            max(1, round(logical_size.height() * pixel_ratio)),
        )
        rendered = self._preview_pixmap.scaled(
            physical_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        rendered.setDevicePixelRatio(pixel_ratio)
        self._rendered_preview_pixmap = rendered
        self.preview_image.setPixmap(rendered)

    def _preview_device_pixel_ratio(self) -> float:
        return max(1.0, float(self.devicePixelRatioF()))

    def _request_view_in_files(self) -> None:
        if self._preview_path and self._preview_path.is_file():
            self.viewInFilesRequested.emit(self._preview_path)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview()


class SecondaryEclipsePage(QWidget):
    analyzeRequested = Signal(dict)
    cancelRequested = Signal()
    viewInFilesRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._parameters: PlanetParameters | None = None
        self._busy = False
        self._result_valid = False
        self._preview_path: Path | None = None
        self._preview_pixmap = QPixmap()
        self._rendered_preview_pixmap = QPixmap()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Secondary Eclipse",
                "Search at the predicted occultation phase for the planet's dayside light disappearing behind its star.",
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        setup_card = QFrame()
        setup_card.setObjectName("card")
        setup_layout = QVBoxLayout(setup_card)
        setup_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Occultation setup")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "A secondary eclipse, also called an occultation, occurs when the planet passes behind the star. "
                "The fitted depth is the planet-to-star flux ratio in this passband; it is not automatically an albedo measurement."
            )
        )
        heading.addStretch()
        setup_layout.addLayout(heading)

        self.fit_context = QLabel("Run a full primary-transit fit to load its ephemeris here.")
        self.fit_context.setObjectName("muted")
        self.fit_context.setWordWrap(True)
        self.fit_context.setMinimumHeight(42)
        setup_layout.addWidget(self.fit_context)

        context_card = QFrame()
        context_card.setObjectName("eclipseContextCard")
        context_card.setStyleSheet(
            f"QFrame#eclipseContextCard {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 7px; }}"
        )
        context_layout = QVBoxLayout(context_card)
        context_layout.setContentsMargins(12, 10, 12, 11)
        context_layout.setSpacing(4)
        context_title = QLabel("How to read this")
        context_title.setObjectName("eyebrow")
        context_layout.addWidget(context_title)
        context_text = QLabel(
            "LEAPS fits only the expected phase and checks two nearby control phases. A candidate still needs independent observations; an inconclusive result is useful evidence about this dataset's sensitivity."
        )
        context_text.setObjectName("muted")
        context_text.setWordWrap(True)
        context_layout.addWidget(context_text)
        setup_layout.addWidget(context_card)

        form = QFormLayout()
        form.setSpacing(10)
        self.light_curve = QComboBox()
        self.light_curve.addItem("Aperture photometry", "aperture")
        self.light_curve.addItem("Gaussian photometry", "gaussian")
        self.expected_phase = QDoubleSpinBox()
        self.expected_phase.setRange(0.05, 0.95)
        self.expected_phase.setDecimals(4)
        self.expected_phase.setSingleStep(0.01)
        self.expected_phase.setValue(0.50)
        self.duration_hours = QDoubleSpinBox()
        self.duration_hours.setRange(0.05, 24.0)
        self.duration_hours.setDecimals(2)
        self.duration_hours.setSingleStep(0.10)
        self.duration_hours.setSuffix(" h")
        self.duration_hours.setValue(2.0)
        self.baseline = QComboBox()
        self.baseline.addItem("Constant", "constant")
        self.baseline.addItem("Linear", "linear")
        self.baseline.addItem("Quadratic", "quadratic")
        self.baseline.setCurrentIndex(self.baseline.findData("linear"))
        form.addRow(
            LabelWithInfo(
                "Approved light curve",
                "Both curves use the comparison-star selection approved in Light Curve. Start with the method used for the full transit fit.",
            ),
            self.light_curve,
        )
        form.addRow(
            LabelWithInfo(
                "Expected phase",
                "For a circular orbit, secondary eclipse occurs at phase 0.50. Change this only for a justified eccentric-orbit prediction.",
            ),
            self.expected_phase,
        )
        form.addRow(
            LabelWithInfo(
                "Event duration",
                "LEAPS suggests a duration from the fitted transit geometry. A secondary eclipse is usually similar in duration, but you may adjust it if you have a better prediction.",
            ),
            self.duration_hours,
        )
        form.addRow(
            LabelWithInfo(
                "Local baseline",
                "Fits a local trend outside the expected eclipse. Linear is the safest default; use quadratic only when the nearby baseline clearly needs it.",
            ),
            self.baseline,
        )
        setup_layout.addLayout(form)
        setup_layout.addStretch()
        buttons = QHBoxLayout()
        self.cancel = ActionButton(
            "Cancel",
            "fa6s.stop",
            tooltip="Stop safely before results replace the previous secondary-eclipse analysis.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        self.analyze = ActionButton(
            "Analyse Eclipse",
            "fa6s.chart-line",
            primary=True,
            tooltip="Fit the expected secondary eclipse, inspect nearby control phases, and export the diagnostic plots and data.",
        )
        self.analyze.clicked.connect(lambda: self.analyzeRequested.emit(self.values()))
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.analyze)
        setup_layout.addLayout(buttons)
        layout.addWidget(setup_card, 2)

        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 16, 18, 18)
        result_heading = QHBoxLayout()
        result_title = QLabel("Secondary-eclipse result")
        result_title.setObjectName("sectionTitle")
        result_heading.addWidget(result_title)
        result_heading.addStretch()
        self.outcome = QLabel("Waiting for full fit")
        self.outcome.setStyleSheet(
            f"color: {COLORS['muted']}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 4px 8px; font-weight: 650;"
        )
        result_heading.addWidget(self.outcome)
        result_layout.addLayout(result_heading)
        self.message = QLabel(
            "This stage uses the completed primary-transit ephemeris and never searches arbitrary phases for a dip."
        )
        self.message.setObjectName("muted")
        self.message.setWordWrap(True)
        result_layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        result_layout.addWidget(self.progress)
        self.progress_details = QLabel()
        self.progress_details.setObjectName("muted")
        self.progress_details.setVisible(False)
        result_layout.addWidget(self.progress_details)

        metrics = QFrame()
        metrics.setObjectName("eclipseMetrics")
        metrics.setStyleSheet(
            f"QFrame#eclipseMetrics {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 7px; }}"
        )
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(12, 10, 12, 10)
        metrics_layout.setHorizontalSpacing(14)
        metrics_layout.setVerticalSpacing(7)
        self.metric_values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate(
            (
                ("depth", "Eclipse depth"),
                ("significance", "Red-noise S/N"),
                ("noise", "Red-noise correction"),
                ("coverage", "Coverage"),
                ("events", "Eclipse windows"),
                ("control", "Strongest control"),
            )
        ):
            name = QLabel(label)
            name.setObjectName("muted")
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics_layout.addWidget(name, row, 0)
            metrics_layout.addWidget(value, row, 1)
            self.metric_values[key] = value
        result_layout.addWidget(metrics)
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(420, 300)
        self.preview_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_image.setVisible(False)
        result_layout.addWidget(self.preview_image, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        self.view_in_files = ActionButton(
            "View in Files",
            "fa6s.folder-open",
            tooltip="Reveal the eclipse plot, CSV data, JSON summary, and PDF in Finder or File Explorer.",
        )
        self.view_in_files.setEnabled(False)
        self.view_in_files.clicked.connect(self._request_view_in_files)
        actions.addWidget(self.view_in_files)
        result_layout.addLayout(actions)
        layout.addWidget(result_card, 3)
        outer.addWidget(_scroll_page(body), 1)

        for control in (self.light_curve, self.expected_phase, self.duration_hours, self.baseline):
            if isinstance(control, QComboBox):
                control.currentTextChanged.connect(self.invalidate_result)
            else:
                control.valueChanged.connect(self.invalidate_result)
        self.reset_setup("Run a full primary-transit fit to load the eclipse ephemeris.")

    def values(self) -> dict[str, Any]:
        return {
            "catalog_parameters": self._parameters,
            "light_curve": str(self.light_curve.currentData()),
            "expected_phase": self.expected_phase.value(),
            "duration_hours": self.duration_hours.value(),
            "baseline": str(self.baseline.currentData()),
        }

    def reset_setup(self, message: str) -> None:
        self._parameters = None
        self._result_valid = False
        self._preview_path = None
        self._preview_pixmap = QPixmap()
        self._rendered_preview_pixmap = QPixmap()
        for control, value in (
            (self.light_curve, "aperture"),
            (self.baseline, "linear"),
        ):
            blocked = control.blockSignals(True)
            control.setCurrentIndex(control.findData(value))
            control.blockSignals(blocked)
        for control, value in ((self.expected_phase, 0.50), (self.duration_hours, 2.0)):
            blocked = control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(blocked)
        self.fit_context.setText(message)
        self.message.setText(
            "This stage uses the completed primary-transit ephemeris and never searches arbitrary phases for a dip."
        )
        self.preview_image.clear()
        self.preview_image.setVisible(False)
        self.view_in_files.setEnabled(False)
        self._set_outcome("waiting", "Waiting for full fit")
        self._set_metrics()
        self._refresh_actions()

    def set_fit_context(
        self,
        parameters: PlanetParameters,
        *,
        passband: str = "",
        light_curve: str = "aperture",
        duration_hours: float = 2.0,
    ) -> None:
        self._parameters = parameters
        light_curve_index = self.light_curve.findData(light_curve)
        if light_curve_index >= 0:
            self.light_curve.blockSignals(True)
            self.light_curve.setCurrentIndex(light_curve_index)
            self.light_curve.blockSignals(False)
        self.duration_hours.blockSignals(True)
        self.duration_hours.setValue(duration_hours)
        self.duration_hours.blockSignals(False)
        passband_detail = f" · {passband}" if passband else ""
        self.fit_context.setText(
            f"{parameters.name}{passband_detail} · P = {parameters.period:.8f} d · "
            "ephemeris from the completed primary-transit fit"
        )
        self._refresh_actions()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.analyze.set_running(busy, "Analysing Eclipse…")
        self.cancel.set_cancel_active(busy)
        self.cancel.setEnabled(busy)
        self.cancel.setText("Cancel")
        self.progress.setVisible(busy)
        self.progress_details.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
            self.message.setText("Preparing the fixed-phase eclipse model and control checks…")
        self._refresh_actions()

    def set_stopping(self) -> None:
        self.cancel.setText("Stopping…")
        self.cancel.setEnabled(False)
        self.message.setText("Stopping safely and preserving the last completed eclipse analysis…")

    def update_event(self, event: StageEvent) -> None:
        self.progress.setVisible(True)
        self.progress_details.setVisible(True)
        if event.total > 0:
            self.progress.setRange(0, event.total)
            self.progress.setValue(min(event.current, event.total))
            self.progress.setFormat(f"{event.current} of {event.total}")
        else:
            self.progress.setRange(0, 0)
        self.message.setText(f"{event.message}…")
        phase = str(event.checkpoint or event.details.get("phase", ""))
        self.progress_details.setText(phase.replace("_", " ").capitalize() if phase else "")

    def show_result(self, result: Any) -> None:
        self._preview_path = Path(result.preview_path)
        self._preview_pixmap = QPixmap(str(self._preview_path))
        available = self._preview_path.is_file() and not self._preview_pixmap.isNull()
        self.preview_image.setVisible(available)
        if available:
            self._render_preview()
        else:
            self.preview_image.clear()
        self.view_in_files.setEnabled(available)
        self.message.setText(result.message)
        self._set_outcome(result.outcome, result.outcome_label)
        self._set_metrics(
            depth=result.depth_ppm,
            depth_uncertainty=result.depth_uncertainty_ppm,
            significance=result.significance,
            beta=result.red_noise_beta,
            local_points=result.local_points,
            in_eclipse_points=result.in_eclipse_points,
            events=result.event_count,
            control=result.control_significance,
        )
        self._result_valid = available
        self._refresh_actions()

    def show_saved_result(self, summary: dict[str, Any], preview_path: Path) -> None:
        self._preview_path = Path(preview_path)
        self._preview_pixmap = QPixmap(str(self._preview_path))
        available = self._preview_path.is_file() and not self._preview_pixmap.isNull()
        self.preview_image.setVisible(available)
        if available:
            self._render_preview()
        else:
            self.preview_image.clear()
        self.view_in_files.setEnabled(available)
        self.message.setText(str(summary.get("message", "A saved secondary-eclipse analysis is available.")))
        self._set_outcome(
            str(summary.get("outcome", "inconclusive")),
            str(summary.get("outcome_label", "Saved analysis")),
        )
        self._set_metrics(
            depth=_optional_float(summary.get("depth_ppm")),
            depth_uncertainty=_optional_float(summary.get("depth_uncertainty_ppm")),
            significance=_optional_float(summary.get("significance")),
            beta=_optional_float(summary.get("red_noise_beta")),
            local_points=int(summary.get("local_points", 0)),
            in_eclipse_points=int(summary.get("in_eclipse_points", 0)),
            events=int(summary.get("event_count", 0)),
            control=_optional_float(summary.get("control_significance")),
        )
        self._result_valid = available
        self._refresh_actions()

    def show_failure(self, message: str) -> None:
        self.message.setText(message)
        self._set_outcome("failure", "Needs attention")
        self._result_valid = False
        self._refresh_actions()

    def show_cancelled(self, message: str) -> None:
        self.message.setText(message)
        self._set_outcome("waiting", "Analysis cancelled")
        self._refresh_actions()

    def invalidate_result(self, *_args: Any) -> None:
        if self._result_valid:
            self.message.setText("Settings changed. Analyse Eclipse again before interpreting a result.")
            self._set_outcome("waiting", "Settings changed")
            self.preview_image.setVisible(False)
            self.view_in_files.setEnabled(False)
        self._result_valid = False
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        self.analyze.set_primary(True)
        self.analyze.setEnabled(self._parameters is not None and not self._busy)

    def _set_outcome(self, outcome: str, text: str) -> None:
        colors = {
            "candidate": COLORS["green"],
            "marginal": COLORS["amber"],
            "inconclusive": COLORS["muted"],
            "failure": COLORS["amber"],
            "waiting": COLORS["muted"],
        }
        color = colors.get(outcome, COLORS["muted"])
        self.outcome.setText(text)
        self.outcome.setStyleSheet(
            f"color: {color}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 4px 8px; font-weight: 650;"
        )

    def _set_metrics(
        self,
        *,
        depth: float | None = None,
        depth_uncertainty: float | None = None,
        significance: float | None = None,
        beta: float | None = None,
        local_points: int = 0,
        in_eclipse_points: int = 0,
        events: int = 0,
        control: float | None = None,
    ) -> None:
        self.metric_values["depth"].setText(
            f"{depth:.0f} ± {depth_uncertainty:.0f} ppm"
            if depth is not None and depth_uncertainty is not None
            else "—"
        )
        self.metric_values["significance"].setText(f"{significance:.1f} σ" if significance is not None else "—")
        self.metric_values["noise"].setText(f"β = {beta:.2f}" if beta is not None else "—")
        self.metric_values["coverage"].setText(
            f"{local_points} local · {in_eclipse_points} in eclipse" if local_points else "No usable local window"
        )
        self.metric_values["events"].setText(f"{events}" if events else "—")
        self.metric_values["control"].setText(f"{control:.1f} σ" if control is not None else "Not covered")

    def _render_preview(self) -> None:
        if self._preview_pixmap.isNull():
            return
        pixel_ratio = self._preview_device_pixel_ratio()
        logical_size = self.preview_image.size()
        physical_size = QSize(
            max(1, round(logical_size.width() * pixel_ratio)),
            max(1, round(logical_size.height() * pixel_ratio)),
        )
        rendered = self._preview_pixmap.scaled(
            physical_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        rendered.setDevicePixelRatio(pixel_ratio)
        self._rendered_preview_pixmap = rendered
        self.preview_image.setPixmap(rendered)

    def _preview_device_pixel_ratio(self) -> float:
        return max(1.0, float(self.devicePixelRatioF()))

    def _request_view_in_files(self) -> None:
        if self._preview_path and self._preview_path.is_file():
            self.viewInFilesRequested.emit(self._preview_path)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview()


class ComparisonStarsPage(QWidget):
    rankRequested = Signal()
    runRequested = Signal(list, float)
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.candidates: list[dict[str, float]] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Apertures & Comparison Stars",
                "Review ranked stars and approve the ensemble used for differential photometry.",
            )
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Ranked comparison stars")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "Candidates are ranked by usable brightness, distance from the target, saturation margin, and measurement stability. You always approve the final ensemble."
            )
        )
        heading.addStretch()
        rank = ActionButton(
            "Rank candidates",
            "fa6s.wand-magic-sparkles",
            tooltip="Detect and rank comparison stars in the reference image without beginning photometry.",
        )
        rank.clicked.connect(self.rankRequested)
        heading.addWidget(rank)
        card_layout.addLayout(heading)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "Rank", "X", "Y", "Peak", "Score"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        card_layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        controls.addWidget(
            LabelWithInfo(
                "Aperture radius",
                "Circular aperture radius in pixels. Start near 1.5–2 times the measured stellar FWHM.",
            )
        )
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1, 100)
        self.radius.setDecimals(1)
        self.radius.setValue(8.0)
        self.radius.setSuffix(" px")
        controls.addWidget(self.radius)
        controls.addStretch()
        self.cancel = ActionButton(
            "Cancel",
            "fa6s.stop",
            tooltip="Stop after the current safe checkpoint. Completed outputs remain intact.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        controls.addWidget(self.cancel)
        self.run = ActionButton(
            "Run photometry",
            "fa6s.play",
            primary=True,
            tooltip="Measure the target and approved comparisons across every accepted aligned frame in the background.",
        )
        self.run.clicked.connect(self._run)
        controls.addWidget(self.run)
        card_layout.addLayout(controls)
        layout.addWidget(card, 1)
        self.status = QLabel("Plate solve or manually place the target, then rank candidates.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        outer.addWidget(body, 1)

    def set_candidates(self, candidates: list[dict[str, float]]) -> None:
        self.candidates = candidates
        self.table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            use = QTableWidgetItem()
            use.setFlags(use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(Qt.CheckState.Checked if row < 5 else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, use)
            values = (
                row + 1,
                f"{candidate['x']:.1f}",
                f"{candidate['y']:.1f}",
                f"{candidate['peak']:.0f}",
                f"{candidate['score']:.3f}",
            )
            for column, value in enumerate(values, start=1):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self.status.setText(
            f"{len(candidates)} candidates ranked. Review the checked ensemble before running photometry."
        )

    def _run(self) -> None:
        selected = []
        for row, candidate in enumerate(self.candidates):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                selected.append((candidate["x"], candidate["y"]))
        self.runRequested.emit(selected, self.radius.value())

    def set_busy(self, busy: bool) -> None:
        self.run.set_running(busy, "Running Photometry…")
        self.cancel.set_cancel_active(busy)
        self.run.setEnabled(not busy)
        self.cancel.setEnabled(busy)


class ReportsPage(QWidget):
    openFolderRequested = Signal()
    exportExoClockRequested = Signal()
    exportETDRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Reports & Exports", "Keep familiar HOPS outputs and prepare submission-ready transit data."
            )
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(14)
        for title, detail, button_text, icon_name, signal, tooltip in (
            (
                "Project outputs",
                "Open the portable outputs folder containing reduction metadata, light curves, fit summaries, and plots.",
                "Open outputs folder",
                "fa6s.folder-open",
                self.openFolderRequested,
                "Open the generated project output folder in the system file manager.",
            ),
            (
                "ExoClock export",
                "Create a normalized time, flux, and uncertainty table alongside a metadata summary for ExoClock review.",
                "Export ExoClock",
                "fa6s.file-export",
                self.exportExoClockRequested,
                "Write a fresh ExoClock-compatible export without modifying the successful photometry result.",
            ),
            (
                "ETD export",
                "Create a Julian-date, differential-magnitude, and uncertainty table for the Exoplanet Transit Database workflow.",
                "Export ETD",
                "fa6s.file-export",
                self.exportETDRequested,
                "Write an ETD-compatible magnitude export from the latest successful light curve.",
            ),
        ):
            card = QFrame()
            card.setObjectName("card")
            row = QHBoxLayout(card)
            row.setContentsMargins(18, 16, 18, 16)
            labels = QVBoxLayout()
            heading = QLabel(title)
            heading.setObjectName("sectionTitle")
            description = QLabel(detail)
            description.setWordWrap(True)
            description.setObjectName("muted")
            labels.addWidget(heading)
            labels.addWidget(description)
            row.addLayout(labels, 1)
            action = ActionButton(button_text, icon_name, tooltip=tooltip)
            action.clicked.connect(signal)
            row.addWidget(action)
            layout.addWidget(card)
        layout.addStretch()
        outer.addWidget(body, 1)


class SimpleToolPage(QWidget):
    def __init__(self, title: str, subtitle: str, icon_name: str, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PageHeader(title, subtitle))
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        self.graphic = QLabel()
        self.graphic.setPixmap(icon(icon_name, COLORS["cyan"]).pixmap(42, 42))
        self.graphic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        detail = QLabel(subtitle)
        detail.setWordWrap(True)
        detail.setObjectName("muted")
        layout.addWidget(self.graphic)
        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addStretch()
        shell = QVBoxLayout()
        shell.setContentsMargins(26, 24, 26, 28)
        shell.addWidget(card)
        outer.addLayout(shell, 1)

    def set_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self.graphic.setPixmap(
            pixmap.scaled(
                980,
                620,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ObservingPlannerPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Observing Planner", "Check whether a target transit is practical from your observing site."
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        form_card = QFrame()
        form_card.setObjectName("card")
        form = QFormLayout(form_card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(11)
        self.planner_ra = QLineEdit()
        self.planner_ra.setPlaceholderText("19:34:55.87")
        self.planner_dec = QLineEdit()
        self.planner_dec.setPlaceholderText("+36:48:55.79")
        self.planner_date = QLineEdit()
        self.planner_date.setPlaceholderText("2026-07-12 22:00")
        self.planner_lat = QDoubleSpinBox()
        self.planner_lat.setRange(-90, 90)
        self.planner_lat.setDecimals(5)
        self.planner_lat.setValue(42.33)
        self.planner_lon = QDoubleSpinBox()
        self.planner_lon.setRange(-180, 180)
        self.planner_lon.setDecimals(5)
        self.planner_lon.setValue(-83.05)
        form.addRow(
            LabelWithInfo("Right ascension", "ICRS target right ascension in hh:mm:ss."), self.planner_ra
        )
        form.addRow(
            LabelWithInfo("Declination", "ICRS target declination in signed dd:mm:ss."), self.planner_dec
        )
        form.addRow(
            LabelWithInfo("Local date & time", "Beginning of the observing window in local civil time."),
            self.planner_date,
        )
        form.addRow(LabelWithInfo("Latitude", "Observatory latitude; north is positive."), self.planner_lat)
        form.addRow(LabelWithInfo("Longitude", "Observatory longitude; east is positive."), self.planner_lon)
        calculate = ActionButton(
            "Evaluate observing window",
            "fa6s.moon",
            primary=True,
            tooltip="Calculate target altitude and airmass across a six-hour observing window.",
        )
        calculate.clicked.connect(self._calculate)
        form.addRow(calculate)
        layout.addWidget(form_card, 2)
        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Visibility summary")
        title.setObjectName("sectionTitle")
        self.planner_result = QLabel("Enter a target and site to evaluate its observing window.")
        self.planner_result.setObjectName("muted")
        self.planner_result.setWordWrap(True)
        result_layout.addWidget(title)
        result_layout.addWidget(self.planner_result)
        result_layout.addStretch()
        layout.addWidget(result_card, 3)
        outer.addWidget(body, 1)

    def _calculate(self) -> None:
        try:
            import astropy.units as units
            from astropy.coordinates import AltAz, EarthLocation, SkyCoord
            from astropy.time import Time

            target = SkyCoord(
                self.planner_ra.text(), self.planner_dec.text(), unit=(units.hourangle, units.deg)
            )
            location = EarthLocation(
                lat=self.planner_lat.value() * units.deg, lon=self.planner_lon.value() * units.deg
            )
            start = Time(self.planner_date.text().strip().replace(" ", "T"), format="isot")
            times = start + [0, 1, 2, 3, 4, 5, 6] * units.hour
            altitudes = target.transform_to(AltAz(obstime=times, location=location)).alt.deg
            peak = float(max(altitudes))
            hours = sum(value >= 30 for value in altitudes)
            self.planner_result.setText(
                f"Peak altitude: {peak:.1f}°\nHours above 30° in this window: about {hours}\n"
                f"Recommendation: {'Good observing geometry.' if peak >= 45 and hours >= 3 else 'Review timing or choose another night.'}"
            )
            self.planner_result.setStyleSheet(
                f"color: {COLORS['green'] if peak >= 45 and hours >= 3 else COLORS['amber']};"
            )
        except Exception as exc:
            self.planner_result.setText(f"The observing window could not be evaluated: {exc}")
            self.planner_result.setStyleSheet(f"color: {COLORS['amber']};")
