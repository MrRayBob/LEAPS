from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTime, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from leaps.fits_inventory import FITS_EXTENSIONS, FrameRecord
from leaps.models import LEAPSError, StageEvent, StageID

from .theme import COLORS
from .widgets import ActionButton, FITSWorkspace, InfoButton, LabelWithInfo, PageHeader, icon


def _scroll_page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


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
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        target_card = QFrame()
        target_card.setObjectName("card")
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
        form.addWidget(
            LabelWithInfo(
                "Right ascension", "ICRS right ascension in hours, minutes, and seconds: hh:mm:ss."
            ),
            1,
            0,
        )
        form.addWidget(self.ra, 1, 1)
        form.addWidget(
            LabelWithInfo(
                "Declination", "ICRS declination in signed degrees, minutes, and seconds: +dd:mm:ss."
            ),
            2,
            0,
        )
        form.addWidget(self.dec, 2, 1)
        form.setColumnStretch(1, 1)
        target_layout.addLayout(form)
        layout.addWidget(target_card)

        folder_card = QFrame()
        folder_card.setObjectName("card")
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(18, 16, 18, 18)
        row = QHBoxLayout()
        title = QLabel("Observing run")
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addWidget(
            InfoButton(
                "LEAPS reads raw FITS files in place. It stores only project state and generated outputs in a portable .leaps folder beside them."
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
        layout.addWidget(folder_card)

        frames_card = QFrame()
        frames_card.setObjectName("card")
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
        for index, (key, label, default, icon_name, tip) in enumerate(card_definitions):
            card = FrameAssignmentCard(label, default, icon_name, tip)
            card.classifierChanged.connect(self._refresh_assignments)
            self.assignment_cards[key] = card
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
        self.bias_waiver = QCheckBox(
            "Continue without bias frames — my acquisition workflow does not require a separate bias set"
        )
        self.dark_waiver = QCheckBox(
            "Continue without dark frames — I accept the additional calibration risk"
        )
        self.flat_waiver = QCheckBox(
            "Continue without flat frames — I accept uncorrected illumination and dust effects"
        )
        waivers_layout = QVBoxLayout()
        waivers_layout.setContentsMargins(0, 6, 0, 2)
        waivers_layout.setSpacing(10)
        for waiver, tip in (
            (
                self.bias_waiver,
                "Required only when no bias frames are assigned. This decision is saved in the project manifest.",
            ),
            (
                self.dark_waiver,
                "Required only when no dark frames are assigned. This decision is saved in the project manifest.",
            ),
            (
                self.flat_waiver,
                "Required only when no flat frames are assigned. This decision is saved in the project manifest.",
            ),
        ):
            waiver.setToolTip(tip)
            waivers_layout.addWidget(waiver)
        frames_layout.addLayout(waivers_layout)
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
        outer.addWidget(_scroll_page(body), 1)
        self.records: list[FrameRecord] = []
        self.file_paths: list[str] = []
        self.assignments: dict[str, list[str]] = {
            key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        }

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose observing run")
        if folder:
            self.folder.setText(folder)
            self.preview_folder(Path(folder))
            self.scan_progress.setRange(0, 0)
            self.scan_progress.setVisible(True)
            self.scanRequested.emit(Path(folder))

    def set_records(self, records: list[FrameRecord]) -> None:
        self.scan_progress.setVisible(False)
        self.records = list(records)
        self.file_paths = [record.path for record in records]
        self._refresh_assignments()

    def preview_folder(self, root: Path) -> None:
        """Populate live counts from filenames before FITS header inspection finishes."""
        self.file_paths = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.casefold() in FITS_EXTENSIONS
            and ".leaps" not in path.parts
            and not any(part.startswith("reduction") or part.startswith("photometry") for part in path.parts)
        ]
        self._refresh_assignments()

    def set_assignment_patterns(self, patterns: dict[str, str]) -> None:
        for key, value in patterns.items():
            if key in self.assignment_cards and value:
                self.assignment_cards[key].classifier.setText(str(value))
        self._refresh_assignments()

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
        self.assignments = assignments
        for key, card in self.assignment_cards.items():
            card.set_count(len(assignments[key]))
        assigned = sum(len(assignments[key]) for key in order)
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
                "waivers": {
                    "bias": self.bias_waiver.isChecked(),
                    "dark": self.dark_waiver.isChecked(),
                    "flat": self.flat_waiver.isChecked(),
                },
                "assignments": {key: list(paths) for key, paths in self.assignments.items()},
                "frame_classifiers": self.assignment_patterns(),
            }
        )

    def show_error(self, message: str) -> None:
        self.validation.setText(message)
        self.validation.setStyleSheet(f"color: {COLORS['amber']};")


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
            "Cancel safely",
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
        timestamp.setFixedWidth(62)
        row.addWidget(timestamp)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        labels.addWidget(QLabel(text))
        if detail:
            secondary = QLabel(detail)
            secondary.setObjectName("muted")
            secondary.setWordWrap(True)
            labels.addWidget(secondary)
        row.addLayout(labels, 1)


class RecoveryInspector(QFrame):
    retryRequested = Signal()
    manualRequested = Signal()
    copyRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("recoveryInspector")
        self.setMinimumWidth(320)
        self.setMaximumWidth(390)
        self.setStyleSheet(
            f"QFrame#recoveryInspector {{background: {COLORS['surface']}; border-left: 1px solid {COLORS['border_soft']};}}"
            "QLabel, QPushButton { font-size: 14px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        banner = QFrame()
        banner.setStyleSheet(f"background: #6d4703; border-bottom: 1px solid {COLORS['amber_dark']};")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(19, 13, 15, 13)
        warning = QLabel()
        warning.setPixmap(icon("fa6s.triangle-exclamation", COLORS["amber"]).pixmap(24, 24))
        self.banner_title = QLabel("Plate solve needs attention")
        self.banner_title.setStyleSheet(f"color: {COLORS['amber']}; font-size: 16px; font-weight: 650;")
        banner_layout.addWidget(warning)
        banner_layout.addWidget(self.banner_title, 1)
        layout.addWidget(banner)

        scroll_content = QWidget()
        scroll_content.setObjectName("recoveryContent")
        scroll_content.setStyleSheet(
            f"QWidget#recoveryContent {{ background: {COLORS['surface']}; }} QLabel {{ background: transparent; }}"
        )
        content = QVBoxLayout(scroll_content)
        content.setContentsMargins(20, 17, 20, 22)
        content.setSpacing(15)
        self.explanation = QLabel(
            "The plate solve could not be completed. You can retry, or place the target manually to continue with an unverified WCS."
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
        self.target_name = QLabel("WTS-2 b")
        self.coordinates = QLabel("19:34:55.87  +36:48:55.79")
        self.pixel_scale = QLabel("1.20 arcsec/pixel")
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
            "Retry plate solve",
            "fa6s.rotate",
            primary=True,
            tooltip="Retry with validated coordinates and pixel scale, up to the remaining bounded attempts.",
        )
        self.retry.setMinimumHeight(44)
        self.retry.clicked.connect(self.retryRequested)
        self.manual = ActionButton(
            "Place target manually",
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
        content.addWidget(self.copy)
        content.addStretch()
        scroll = QScrollArea()
        scroll.setStyleSheet(f"QScrollArea {{ background: {COLORS['surface']}; border: 0; }}")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

    def _demo_timeline(self) -> None:
        for row in (
            TimelineRow("10:21:14", "Coordinates validated", "ok"),
            TimelineRow("10:21:15", "42 stars detected", "ok"),
            TimelineRow("10:21:16", "Gaia catalog request failed", "error", "HTTP 503 Service Unavailable"),
            TimelineRow("10:21:16", "Solve stopped after 3 bounded attempts", "paused"),
        ):
            self.timeline.addWidget(row)

    def set_retrying(self) -> None:
        self.retry.setEnabled(False)
        self.retry.setText("Retrying…")
        QTimer.singleShot(1300, self._restore_retry)

    def set_failure(self, failure: LEAPSError) -> None:
        """Replace the demo context with the exact typed runtime failure."""
        self.banner_title.setText(failure.title)
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

    def _restore_retry(self) -> None:
        self.retry.setEnabled(True)
        self.retry.setText("Retry plate solve")

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

    def __init__(self, asset: Path, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PageHeader("Plate Solve", "Match detected stars to Gaia DR3."))
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
        split.addWidget(self.inspector)
        outer.addLayout(split, 1)
        self.inspector.retryRequested.connect(self._retry)
        self.inspector.manualRequested.connect(self.workspace.begin_manual_target)
        self.inspector.copyRequested.connect(self.copyDiagnosticsRequested)
        self.workspace.targetPlaced.connect(self._manual_target_placed)

    def _retry(self) -> None:
        self.inspector.set_retrying()
        self.retryRequested.emit()

    def _manual_target_placed(self, x: float, y: float) -> None:
        self.workspace.place_target_marker(x, y)
        self.inspector.manual.setText("Target placed — unverified WCS")
        self.inspector.manual.setEnabled(False)
        self.inspector.manual.setStyleSheet(
            f"color: {COLORS['amber']}; border-color: {COLORS['amber_dark']};"
        )
        self.manualTargetPlaced.emit(x, y)


class FittingPage(QWidget):
    previewRequested = Signal(dict)
    fullFitRequested = Signal(dict)

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
        title = QLabel("Planet parameters")
        title.setObjectName("sectionTitle")
        form_layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(10)
        self.planet = QLineEdit()
        self.planet.setPlaceholderText("WTS-2 b")
        self.period = QDoubleSpinBox()
        self.period.setDecimals(8)
        self.period.setRange(0.000001, 100000)
        self.period.setValue(1.0187068)
        self.mid_time = QDoubleSpinBox()
        self.mid_time.setDecimals(8)
        self.mid_time.setRange(0, 4_000_000)
        self.mid_time.setValue(2461220.42)
        self.depth = QDoubleSpinBox()
        self.depth.setDecimals(5)
        self.depth.setRange(0, 1)
        self.depth.setValue(0.03)
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
        form_layout.addLayout(form)
        self.advanced_toggle = QPushButton("Advanced MCMC controls")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip(
            "Show sampling controls. Defaults are recommended for most observing runs."
        )
        form_layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        self.walkers = QSpinBox()
        self.walkers.setRange(20, 1000)
        self.walkers.setValue(100)
        self.iterations = QSpinBox()
        self.iterations.setRange(100, 1_000_000)
        self.iterations.setValue(5000)
        self.burn = QSpinBox()
        self.burn.setRange(0, 900_000)
        self.burn.setValue(1000)
        advanced_form.addRow(
            LabelWithInfo("Walkers", "Independent MCMC chains used to explore the posterior."), self.walkers
        )
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
        preview = ActionButton(
            "Preview Fit",
            "fa6s.chart-line",
            tooltip="Run a quick deterministic fit to validate data, timing, and priors.",
        )
        preview.clicked.connect(lambda: self.previewRequested.emit(self.values()))
        full = ActionButton(
            "Run Full Fit",
            "fa6s.play",
            primary=True,
            tooltip="Run the full MCMC uncertainty analysis in the background.",
        )
        full.clicked.connect(lambda: self.fullFitRequested.emit(self.values()))
        buttons.addWidget(preview)
        buttons.addWidget(full)
        form_layout.addLayout(buttons)
        layout.addWidget(form_card, 2)
        result = QFrame()
        result.setObjectName("card")
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(18, 16, 18, 18)
        result_title = QLabel("Fit preview")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)
        message = QLabel(
            "Run Preview Fit to inspect the model and residuals before committing time to the full fit."
        )
        message.setWordWrap(True)
        message.setObjectName("muted")
        result_layout.addWidget(message)
        result_layout.addStretch()
        layout.addWidget(result, 3)
        outer.addWidget(_scroll_page(body), 1)

    def values(self) -> dict[str, Any]:
        return {
            "planet": self.planet.text().strip(),
            "period": self.period.value(),
            "mid_time": self.mid_time.value(),
            "depth": self.depth.value(),
            "walkers": self.walkers.value(),
            "iterations": self.iterations.value(),
            "burn": self.burn.value(),
        }


class ComparisonStarsPage(QWidget):
    rankRequested = Signal()
    runRequested = Signal(list, float)

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
        run = ActionButton(
            "Run photometry",
            "fa6s.play",
            primary=True,
            tooltip="Measure the target and approved comparisons across every accepted aligned frame in the background.",
        )
        run.clicked.connect(self._run)
        controls.addWidget(run)
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
        graphic = QLabel()
        graphic.setPixmap(icon(icon_name, COLORS["cyan"]).pixmap(42, 42))
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        detail = QLabel(subtitle)
        detail.setWordWrap(True)
        detail.setObjectName("muted")
        layout.addWidget(graphic)
        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addStretch()
        shell = QVBoxLayout()
        shell.setContentsMargins(26, 24, 26, 28)
        shell.addWidget(card)
        outer.addLayout(shell, 1)


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
