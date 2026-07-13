from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from leaps.offline import OfflineDataManager, format_bytes

from .widgets import ActionButton, InfoButton, LabelWithInfo


def default_offline_root() -> Path:
    override = os.getenv("LEAPS_OFFLINE_DATA_PATH")
    if override:
        return Path(override).expanduser()
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(location or Path.home() / ".leaps") / "offline-data"


class OfflineDataTab(QWidget):
    downloadAllRequested = Signal()
    refreshRequested = Signal()
    removeRequested = Signal(str)

    def __init__(self, manager: OfflineDataManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        title = QLabel("Offline Data")
        title.setStyleSheet("font-size: 17px; font-weight: 650;")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "Download reusable catalogues and model packages so reductions, fitting, and exports can continue without internet access."
            )
        )
        heading.addStretch()
        layout.addLayout(heading)
        description = QLabel(
            "Portable scientific packages can be downloaded together. Gaia is stored only for target regions, avoiding an impractical full DR3 mirror."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)

        location = QHBoxLayout()
        location.addWidget(
            LabelWithInfo(
                "Storage location",
                "Downloaded data is shared by LEAPS projects on this computer and can be moved in a future update.",
            )
        )
        self.path = QLineEdit(str(manager.root))
        self.path.setReadOnly(True)
        location.addWidget(self.path, 1)
        layout.addLayout(location)

        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Package", "Size", "Version", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("muted")
        layout.addWidget(self.progress_label)
        actions = QHBoxLayout()
        refresh = ActionButton(
            "Check for updates",
            "fa6s.rotate",
            tooltip="Refresh package versions, sizes, and checksums before downloading.",
        )
        refresh.clicked.connect(self.refreshRequested)
        remove = ActionButton(
            "Remove selected",
            "fa6s.trash",
            tooltip="Remove the selected offline package. Project outputs are not affected.",
        )
        remove.clicked.connect(self._remove_selected)
        self.download = ActionButton(
            "Download all for offline use",
            "fa6s.download",
            primary=True,
            tooltip="Download or update all listed packages with checksums and automatic resume support.",
        )
        self.download.clicked.connect(self.downloadAllRequested)
        actions.addWidget(refresh)
        actions.addWidget(remove)
        actions.addStretch()
        actions.addWidget(self.download)
        layout.addLayout(actions)
        self.refresh()

    def refresh(self) -> None:
        self.manager.refresh_installed()
        self.table.setRowCount(len(self.manager.assets))
        for row, asset in enumerate(self.manager.assets):
            values = (
                asset.label,
                asset.display_size,
                asset.version,
                "Ready offline" if asset.installed else "Not downloaded",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 3:
                    item.setForeground(Qt.GlobalColor.green if asset.installed else Qt.GlobalColor.lightGray)
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, asset.asset_id)
        total = format_bytes(self.manager.total_estimated_bytes)
        free = format_bytes(self.manager.free_bytes)
        ready = sum(asset.installed for asset in self.manager.assets)
        self.summary.setText(
            f"{ready} of {len(self.manager.assets)} packages ready offline  ·  {total} remaining  ·  {free} available"
        )
        self.table.resizeColumnsToContents()

    def set_progress(self, label: str, current: int, total: int) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.progress.setValue(round(1000 * current / max(total, 1)))
        self.progress_label.setText(f"Downloading {label} — {format_bytes(current)} of {format_bytes(total)}")

    def finish_progress(self) -> None:
        self.progress.setVisible(False)
        self.progress_label.setText("")
        self.refresh()

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            asset_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.removeRequested.emit(str(asset_id))


class SettingsDialog(QDialog):
    def __init__(self, manager: OfflineDataManager, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LEAPS Settings")
        self.resize(820, 620)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        general = QWidget()
        general_layout = QVBoxLayout(general)
        general_layout.setContentsMargins(20, 20, 20, 20)
        storage_title = QLabel("Storage & updates")
        storage_title.setStyleSheet("font-size: 17px; font-weight: 650;")
        general_layout.addWidget(storage_title)
        self.notify_updates = QCheckBox("Notify before installing application updates")
        self.notify_updates.setChecked(True)
        self.notify_data_updates = QCheckBox("Notify before installing offline data updates")
        self.notify_data_updates.setChecked(True)
        self.prevent_sleep = QCheckBox("Prevent sleep temporarily while a processing stage is running")
        self.prevent_sleep.setChecked(True)
        for control in (self.notify_updates, self.notify_data_updates, self.prevent_sleep):
            general_layout.addWidget(control)
        general_layout.addStretch()
        profile = QWidget()
        profile_layout = QVBoxLayout(profile)
        profile_layout.setContentsMargins(20, 20, 20, 20)
        profile_title = QLabel("Observer & equipment profile")
        profile_title.setStyleSheet("font-size: 17px; font-weight: 650;")
        profile_layout.addWidget(profile_title)
        profile_layout.addWidget(
            QLabel("One global profile is snapshotted into every project for reproducible reports.")
        )
        self.observer = QLineEdit()
        self.observer.setPlaceholderText("Observer or team")
        self.observatory = QLineEdit()
        self.observatory.setPlaceholderText("Observatory / site")
        self.telescope = QLineEdit()
        self.telescope.setPlaceholderText("Telescope and focal length")
        self.camera = QLineEdit()
        self.camera.setPlaceholderText("Camera and pixel size")
        for label, control, tip in (
            ("Observer", self.observer, "Name included in project snapshots and reports."),
            ("Observatory", self.observatory, "Site name used in reports and observing context."),
            ("Telescope", self.telescope, "Telescope, reducer, and effective focal length."),
            ("Camera", self.camera, "Camera model and physical pixel size used for scale estimates."),
        ):
            row = QHBoxLayout()
            row.addWidget(LabelWithInfo(label, tip))
            row.addWidget(control, 1)
            profile_layout.addLayout(row)
        profile_layout.addStretch()
        self.offline = OfflineDataTab(manager)
        tabs.addTab(general, "General")
        tabs.addTab(profile, "Observer & Equipment")
        tabs.addTab(self.offline, "Offline Data")
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class FirstRunDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to LEAPS")
        self.setModal(True)
        self.resize(560, 410)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Set up LEAPS")
        title.setStyleSheet("font-size: 23px; font-weight: 700;")
        layout.addWidget(title)
        subtitle = QLabel("Only three preferences are needed. You can change them later in Settings.")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(
            LabelWithInfo(
                "Project storage",
                "Raw FITS remain read-only. Each project stores logs, checkpoints, caches, and outputs in a visible LEAPS folder beside the data.",
            )
        )
        storage = QComboBox()
        storage.addItems(["Store project files beside FITS data (recommended)"])
        layout.addWidget(storage)
        layout.addWidget(
            LabelWithInfo(
                "Offline data",
                "You can download all reusable catalogues and model data now or later from Settings.",
            )
        )
        self.offline_choice = QComboBox()
        self.offline_choice.addItems(["Ask me later", "Open Offline Data settings after setup"])
        layout.addWidget(self.offline_choice)
        self.notify = QCheckBox("Notify me before installing application or scientific-data updates")
        self.notify.setChecked(True)
        layout.addWidget(self.notify)
        layout.addStretch()
        buttons = QDialogButtonBox()
        start = buttons.addButton("Start using LEAPS", QDialogButtonBox.ButtonRole.AcceptRole)
        start.setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
