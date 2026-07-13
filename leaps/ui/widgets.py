from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import qtawesome as qta
from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from leaps.models import StageID, StageState, StageStatus

from .theme import COLORS


def icon(name: str, color: str = COLORS["muted"], active: str | None = None):
    options = {"color": color}
    if active:
        options["color_active"] = active
    return qta.icon(name, **options)


class InfoPopover(QFrame):
    """A readable LEAPS information card used instead of the native tooltip bubble."""

    def __init__(self, text: str) -> None:
        flags = Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        super().__init__(None, flags)
        self.setObjectName("infoPopover")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            f"QFrame#infoPopover {{ background: #171c23; border: 1px solid {COLORS['border']};"
            " border-radius: 11px; }"
            f"QLabel {{ color: {COLORS['text']}; background: transparent; border: 0;"
            " font-size: 14px; font-weight: 600; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 16)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedWidth(360)
        layout.addWidget(self.label)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 7)
        shadow.setColor(QColor(0, 0, 0, 155))
        self.setGraphicsEffect(shadow)

    def show_beside(self, anchor: QWidget) -> None:
        self.adjustSize()
        anchor_center = anchor.mapToGlobal(QPoint(anchor.width() // 2, anchor.height() + 8))
        screen = anchor.screen()
        available = screen.availableGeometry()
        x = anchor_center.x() - self.width() // 2
        x = max(available.left() + 8, min(x, available.right() - self.width() - 8))
        y = anchor_center.y()
        if y + self.height() > available.bottom() - 8:
            y = anchor.mapToGlobal(QPoint(0, -self.height() - 8)).y()
        self.move(x, y)
        self.show()
        self.raise_()


class InfoButton(QToolButton):
    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(icon("fa6s.circle-info", COLORS["muted"]))
        self.setIconSize(QSize(15, 15))
        self.setFixedSize(26, 26)
        self.setToolTip(tooltip)
        self.setAccessibleName("Information")
        self.setAccessibleDescription(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._popover = InfoPopover(tooltip)
        self._pinned = False
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(350)
        self._hide_timer.timeout.connect(self._hide_unpinned)
        self.clicked.connect(self._show_pinned)

    def show_info(self, *, pinned: bool = False) -> None:
        """Show the information card immediately for hover, click, or keyboard focus."""
        self._hide_timer.stop()
        self._pinned = pinned
        self._popover.show_beside(self)

    def _show_pinned(self) -> None:
        self.show_info(pinned=True)

    def _hide_unpinned(self) -> None:
        if not self._pinned:
            self._popover.hide()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.show_info(pinned=self._pinned)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hide_timer.start()
        super().leaveEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802
        self.show_info(pinned=True)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._pinned = False
        self._popover.hide()
        super().focusOutEvent(event)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            self.show_info()
            event.accept()
            return True
        return super().event(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._popover.hide()
        super().hideEvent(event)


class LabelWithInfo(QWidget):
    def __init__(self, text: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(QLabel(text))
        layout.addWidget(InfoButton(tooltip))
        layout.addStretch()


class ActionButton(QPushButton):
    def __init__(
        self,
        text: str,
        icon_name: str | None = None,
        *,
        primary: bool = False,
        tooltip: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._idle_text = text
        self._idle_icon_name = icon_name
        self._primary = primary
        if icon_name:
            self.setIcon(icon(icon_name, "white" if primary else COLORS["muted"]))
        if primary:
            self.setProperty("primary", True)
        if tooltip:
            self.setToolTip(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_running(self, running: bool, text: str = "Running…") -> None:
        self.setProperty("running", running)
        self.setText(text if running else self._idle_text)
        if running:
            self.setIcon(QIcon())
        elif self._idle_icon_name:
            self.setIcon(
                icon(
                    self._idle_icon_name,
                    "white" if self._primary else COLORS["muted"],
                )
            )
        else:
            self.setIcon(QIcon())
        self._refresh_style()

    def set_primary(self, primary: bool) -> None:
        self._primary = primary
        self.setProperty("primary", primary)
        if self._idle_icon_name and not self.property("running"):
            self.setIcon(icon(self._idle_icon_name, "white" if primary else COLORS["muted"]))
        self._refresh_style()

    def set_cancel_active(self, active: bool) -> None:
        self.setProperty("cancelActive", active)
        if self._idle_icon_name:
            self.setIcon(icon(self._idle_icon_name, "white" if active else COLORS["muted"]))
        self._refresh_style()

    def _refresh_style(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class StageNavButton(QFrame):
    clicked = Signal(object)

    def __init__(self, stage: StageID, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stageNav")
        self.stage = stage
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedHeight(66)
        self.setAccessibleName(title)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 12, 6)
        layout.setSpacing(12)
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(36, 36)
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spinner_phase = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._advance_spinner)
        layout.addWidget(self.status_icon)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("stageTitle")
        self.title.setStyleSheet("font-size: 15px; font-weight: 650;")
        self.summary = QLabel("Locked")
        self.summary.setObjectName("stageSummary")
        self.summary.setStyleSheet("font-size: 13px;")
        labels.addWidget(self.title)
        labels.addWidget(self.summary)
        layout.addLayout(labels, 1)
        self.state = StageState()
        self.active = False
        self.update_state(self.state)

    def update_state(self, state: StageState) -> None:
        self.state = state
        self.summary.setText(state.summary)
        icons = {
            StageStatus.COMPLETE: ("fa6s.circle-check", COLORS["green"]),
            StageStatus.READY: ("fa6.circle", COLORS["cyan"]),
            StageStatus.RUNNING: ("fa6s.spinner", COLORS["cyan"]),
            StageStatus.NEEDS_ATTENTION: ("fa6s.triangle-exclamation", COLORS["amber"]),
            StageStatus.LOCKED: ("fa6s.lock", "#60778d"),
        }
        name, color = icons[state.status]
        icon_size = 30 if state.status == StageStatus.LOCKED else 23
        if state.status == StageStatus.RUNNING:
            self._spinner_phase = 0
            self._render_spinner()
            self._spinner_timer.start()
        else:
            self._spinner_timer.stop()
            self._spinner_phase = 0
            self.status_icon.setPixmap(icon(name, color).pixmap(icon_size, icon_size))
        self.setEnabled(state.status != StageStatus.LOCKED)
        self._restyle()

    def _advance_spinner(self) -> None:
        self._spinner_phase = (self._spinner_phase + 1) % 8
        self._render_spinner()

    def _render_spinner(self) -> None:
        canvas = QPixmap(30, 30)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        center = 15.0
        orbit_radius = 9.0
        opacities = (255, 220, 185, 150, 115, 85, 60, 38)
        radii = (2.6, 2.45, 2.3, 2.15, 2.0, 1.85, 1.7, 1.55)
        for index in range(8):
            trail = (self._spinner_phase - index) % 8
            angle = -math.pi / 2 + index * math.tau / 8
            x = center + math.cos(angle) * orbit_radius
            y = center + math.sin(angle) * orbit_radius
            color = QColor(COLORS["cyan"])
            color.setAlpha(opacities[trail])
            painter.setBrush(color)
            dot_radius = radii[trail]
            painter.drawEllipse(
                QRectF(
                    x - dot_radius,
                    y - dot_radius,
                    dot_radius * 2,
                    dot_radius * 2,
                )
            )
        painter.end()
        self.status_icon.setPixmap(canvas)

    def set_active(self, active: bool) -> None:
        self.active = active
        self._restyle()

    def _restyle(self) -> None:
        if self.active:
            self.setStyleSheet(
                f"QFrame#stageNav {{background: #173049; border-left: 3px solid {COLORS['cyan']};}}"
                f"QLabel {{color: {COLORS['text']};}}"
            )
        elif not self.isEnabled():
            self.setStyleSheet(
                "QFrame#stageNav {background: transparent; border-left: 3px solid transparent;}"
                "QLabel#stageTitle {color: #7f94a8;}"
                "QLabel#stageSummary {color: #61788e;}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#stageNav {{background: transparent; border-left: 3px solid transparent;}}"
                f"QFrame#stageNav:hover {{background: {COLORS['surface']};}}"
            )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.stage)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Space):
            self.clicked.emit(self.stage)
            event.accept()
            return
        super().keyPressEvent(event)


class ToolNavButton(QPushButton):
    def __init__(self, text: str, icon_name: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setIcon(icon(icon_name, COLORS["muted"]))
        self.setStyleSheet(
            f"QPushButton {{text-align: left; background: transparent; border: 0; color: {COLORS['muted']};"
            "padding-left: 15px; font-weight: 500;}"
            f"QPushButton:hover {{background: {COLORS['surface']}; color: {COLORS['text']};}}"
        )


class PageHeader(QFrame):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentHeader")
        self.setFixedHeight(114)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(38, 24, 26, 20)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        texts.addWidget(title_label)
        texts.addWidget(subtitle_label)
        layout.addLayout(texts)
        layout.addStretch()
        self.actions = QHBoxLayout()
        layout.addLayout(self.actions)


@dataclass(slots=True)
class StarOverlay:
    x: float
    y: float
    radius: float = 7.0


class FITSImageView(QGraphicsView):
    pointSelected = Signal(str, float, float)

    def __init__(self, asset: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#02070c"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scene_model = QGraphicsScene(self)
        self.setScene(self.scene_model)
        self.image_item = QGraphicsPixmapItem()
        self.scene_model.addItem(self.image_item)
        self.selection_role = ""
        self.mode = "pan"
        self.inverted = False
        self.data = None
        self.display_min = 0.0
        self.display_max = 1.0
        self.image_width = 0
        self.image_height = 0
        self.marker_items: dict[str, list[QGraphicsItem]] = {}
        self._middle_pan_active = False
        self._middle_pan_position = QPoint()
        self.set_mode("pan")
        if asset and asset.exists():
            self.load_pixmap(QPixmap(str(asset)))

    def load_pixmap(self, pixmap: QPixmap) -> None:
        self.data = None
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()
        self.image_item.setPixmap(pixmap)
        self.scene_model.setSceneRect(QRectF(pixmap.rect()))
        QTimer.singleShot(0, self.fit_image)

    def load_fits(self, path: Path) -> None:
        import numpy as np
        from astropy.io import fits

        data = np.asarray(fits.getdata(path, memmap=False), dtype=np.float32)
        if data.ndim > 2:
            data = data[0]
        finite = data[np.isfinite(data)]
        median = float(np.median(finite)) if finite.size else 0.0
        mad = float(np.median(np.abs(finite - median))) if finite.size else 1.0
        std = 1.4826 * mad or float(np.std(finite)) or 1.0
        self.data = data
        self.display_min = median - 3.0 * std
        self.display_max = median + 20.0 * std
        self.image_height, self.image_width = data.shape
        self.clear_markers()
        self._render_data()
        QTimer.singleShot(0, self.fit_image)

    def _render_data(self) -> None:
        if self.data is None:
            return
        import numpy as np

        span = max(self.display_max - self.display_min, 1e-9)
        display = np.clip((self.data - self.display_min) / span, 0.0, 1.0)
        if self.inverted:
            display = 1.0 - display
        pixels = np.ascontiguousarray(np.flipud(display) * 255, dtype=np.uint8)
        image = QImage(
            pixels.data,
            self.image_width,
            self.image_height,
            pixels.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self.image_item.setPixmap(QPixmap.fromImage(image))
        self.scene_model.setSceneRect(0, 0, self.image_width, self.image_height)

    def set_inverted(self, inverted: bool) -> None:
        self.inverted = inverted
        self._render_data()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.selection_role = "" if mode in {"pan", "zoom"} else self.selection_role
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(self._cursor_for_mode())

    def _cursor_for_mode(self) -> Qt.CursorShape:
        cursors = {
            "pan": Qt.CursorShape.OpenHandCursor,
            "zoom": Qt.CursorShape.SizeAllCursor,
            "select": Qt.CursorShape.CrossCursor,
        }
        return cursors.get(self.mode, Qt.CursorShape.ArrowCursor)

    def begin_selection(self, role: str) -> None:
        self.selection_role = role
        self.mode = "select"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_active = True
            self._middle_pan_position = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self.mode == "select" and event.button() == Qt.MouseButton.LeftButton:
            scene = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(scene):
                fits_y = self.image_height - 1 - scene.y()
                role = self.selection_role
                self.selection_role = ""
                self.set_mode("pan")
                self.pointSelected.emit(role, float(scene.x()), float(fits_y))
                event.accept()
                return
        if self.mode == "zoom":
            factor = 1.5 if event.button() == Qt.MouseButton.LeftButton else 1 / 1.5
            self.scale(factor, factor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._middle_pan_active:
            position = event.position().toPoint()
            delta = position - self._middle_pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._middle_pan_position = position
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._middle_pan_active = False
            self.viewport().setCursor(self._cursor_for_mode())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()

    def fit_image(self) -> None:
        if self.image_width and self.image_height:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_zoom_percent(self, percent: int) -> None:
        self.resetTransform()
        self.scale(percent / 100.0, percent / 100.0)

    def clear_markers(self) -> None:
        for items in self.marker_items.values():
            for item in items:
                self.scene_model.removeItem(item)
        self.marker_items.clear()

    def remove_marker(self, key: str) -> None:
        for item in self.marker_items.pop(key, []):
            self.scene_model.removeItem(item)

    def set_marker_active(self, key: str, active: bool) -> None:
        for item in self.marker_items.get(key, []):
            item.setOpacity(1.0 if active else 0.28)

    def set_marker(
        self,
        key: str,
        x: float,
        y: float,
        *,
        radius: float,
        label: str,
        target: bool = False,
        sky_inner: float = 1.7,
        sky_outer: float = 2.4,
    ) -> None:
        for item in self.marker_items.pop(key, []):
            self.scene_model.removeItem(item)
        scene_y = self.image_height - 1 - y
        color = QColor(COLORS["amber"] if target else COLORS["cyan"])
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        aperture = self.scene_model.addEllipse(
            x - radius, scene_y - radius, 2 * radius, 2 * radius, pen
        )
        inner = self.scene_model.addEllipse(
            x - sky_inner * radius,
            scene_y - sky_inner * radius,
            2 * sky_inner * radius,
            2 * sky_inner * radius,
            QPen(color, 1, Qt.PenStyle.DotLine),
        )
        outer = self.scene_model.addEllipse(
            x - sky_outer * radius,
            scene_y - sky_outer * radius,
            2 * sky_outer * radius,
            2 * sky_outer * radius,
            QPen(color, 1, Qt.PenStyle.DotLine),
        )
        text = self.scene_model.addText(label)
        text.setDefaultTextColor(color)
        text.setPos(x + radius + 5, scene_y - radius - 4)
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.marker_items[key] = [aperture, inner, outer, text]


class FITSWorkspace(QFrame):
    pointSelected = Signal(str, float, float)

    def __init__(self, asset: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setFixedHeight(54)
        toolbar.setStyleSheet(
            f"background: {COLORS['surface']}; border-bottom: 1px solid {COLORS['border_soft']};"
        )
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(13, 8, 13, 8)
        tools.setSpacing(0)
        self.mode_buttons: dict[str, QPushButton] = {}
        for text, icon_name, tip in (
            (
                "Pan",
                "fa6s.hand",
                "Drag the image to inspect a different area. Hold the middle mouse button to pan from any tool.",
            ),
            ("Zoom", "fa6s.magnifying-glass", "Zoom into a region of the FITS image."),
            (
                "Invert",
                "fa6s.circle-half-stroke",
                "Switch between white stars on black and black stars on white.",
            ),
            ("Reset", "fa6s.rotate-left", "Restore the original view and contrast."),
        ):
            button = ActionButton(text, icon_name, tooltip=tip)
            button.setFlat(True)
            if text == "Pan":
                button.setStyleSheet(f"background: {COLORS['surface_3']}; color: {COLORS['cyan']};")
            tools.addWidget(button)
            self.mode_buttons[text.casefold()] = button
        tools.addStretch()
        self.fullscreen = QToolButton()
        self.fullscreen.setIcon(icon("fa6s.expand", COLORS["muted"]))
        self.fullscreen.setToolTip("Expand the application to use the full screen.")
        tools.addWidget(self.fullscreen)
        self.zoom = QComboBox()
        self.zoom.addItems(["Fit", "25%", "50%", "100%", "200%", "400%", "800%"])
        self.zoom.setFixedWidth(90)
        self.zoom.setToolTip("Select a zoom level.")
        tools.addWidget(self.zoom)
        layout.addWidget(toolbar)

        self.image = FITSImageView(asset)
        self.image.pointSelected.connect(self.pointSelected)
        layout.addWidget(self.image, 1)
        self.mode_buttons["pan"].clicked.connect(lambda: self.image.set_mode("pan"))
        self.mode_buttons["zoom"].clicked.connect(lambda: self.image.set_mode("zoom"))
        self.mode_buttons["invert"].setCheckable(True)
        self.mode_buttons["invert"].toggled.connect(self.image.set_inverted)
        self.mode_buttons["reset"].clicked.connect(self.reset_view)
        self.zoom.currentTextChanged.connect(self._zoom_changed)
        self.fullscreen.clicked.connect(self._toggle_fullscreen)

        metadata = QFrame()
        metadata.setFixedHeight(47)
        metadata.setStyleSheet(
            f"background: {COLORS['surface']}; border-top: 1px solid {COLORS['border_soft']};"
        )
        meta_layout = QHBoxLayout(metadata)
        meta_layout.setContentsMargins(18, 0, 18, 0)
        self.filename = QLabel("FITS: light_2026-06-28T22-51-01.fits")
        self.dimensions = QLabel("2048 × 2048 px")
        self.bitdepth = QLabel("16-bit")
        self.scale = QLabel('Pixel scale: 1.20 "/pixel')
        for label in (self.filename, self.dimensions, self.bitdepth, self.scale):
            label.setObjectName("muted")
            meta_layout.addWidget(label)
            meta_layout.addStretch()
        layout.addWidget(metadata)

    def load_fits(self, path: Path, pixel_scale: float = 0.0) -> None:
        from astropy.io import fits

        self.image.load_fits(path)
        header = fits.getheader(path)
        self.filename.setText(f"FITS: {path.name}")
        self.dimensions.setText(f"{self.image.image_width} × {self.image.image_height} px")
        self.bitdepth.setText(f"{abs(int(header.get('BITPIX', 0)))}-bit")
        self.scale.setText(
            f'Pixel scale: {pixel_scale:.2f} "/pixel' if pixel_scale > 0 else "Pixel scale: not set"
        )

    def begin_selection(self, role: str) -> None:
        self.image.begin_selection(role)

    def begin_manual_target(self) -> None:
        self.begin_selection("target")

    def place_target_marker(
        self,
        x: float,
        y: float,
        radius: float = 8.0,
        label: str = "Target",
        *,
        sky_inner: float = 1.7,
        sky_outer: float = 2.4,
    ) -> None:
        self.image.set_marker(
            "target",
            x,
            y,
            radius=radius,
            label=label,
            target=True,
            sky_inner=sky_inner,
            sky_outer=sky_outer,
        )

    def place_comparison_marker(
        self,
        index: int,
        x: float,
        y: float,
        radius: float = 8.0,
        *,
        active: bool = True,
        sky_inner: float = 1.7,
        sky_outer: float = 2.4,
    ) -> None:
        key = f"comparison-{index}"
        self.image.set_marker(
            key,
            x,
            y,
            radius=radius,
            label=f"C{index}",
            target=False,
            sky_inner=sky_inner,
            sky_outer=sky_outer,
        )
        self.image.set_marker_active(key, active)

    def reset_view(self) -> None:
        self.mode_buttons["invert"].setChecked(False)
        self.image.set_mode("pan")
        self.image.fit_image()
        blocked = self.zoom.blockSignals(True)
        self.zoom.setCurrentText("Fit")
        self.zoom.blockSignals(blocked)

    def _zoom_changed(self, text: str) -> None:
        if text == "Fit":
            self.image.fit_image()
        else:
            self.image.set_zoom_percent(int(text.rstrip("%")))

    def _toggle_fullscreen(self) -> None:
        window = self.window()
        if window.isFullScreen():
            window.showNormal()
        else:
            window.showFullScreen()


class EmptyState(QWidget):
    def __init__(self, title: str, message: str, icon_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addStretch()
        picture = QLabel()
        picture.setPixmap(icon(icon_name, COLORS["muted_2"]).pixmap(42, 42))
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 17px; font-weight: 600;")
        message_label = QLabel(message)
        message_label.setObjectName("muted")
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(picture)
        layout.addWidget(title_label)
        layout.addWidget(message_label)
        layout.addStretch()
