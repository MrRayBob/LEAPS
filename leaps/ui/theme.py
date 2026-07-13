from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

COLORS = {
    "canvas": "#07131f",
    "sidebar": "#0b1b2b",
    "surface": "#0d2234",
    "surface_2": "#102a40",
    "surface_3": "#17344d",
    "border": "#24445c",
    "border_soft": "#19364c",
    "text": "#f3f7fb",
    "muted": "#9cadbf",
    "muted_2": "#71859a",
    "cyan": "#22c6f4",
    "cyan_soft": "#0f779a",
    "green": "#55d4bd",
    "amber": "#ffc443",
    "amber_dark": "#8b5b00",
    "red": "#ff624c",
}

def palette() -> QPalette:
    result = QPalette()
    result.setColor(QPalette.ColorRole.Window, QColor(COLORS["canvas"]))
    result.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    result.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface"]))
    result.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["surface_2"]))
    result.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    result.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface_2"]))
    result.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    result.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["cyan_soft"]))
    result.setColor(QPalette.ColorRole.HighlightedText, QColor(COLORS["text"]))
    result.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS["muted_2"]))
    return result


APP_STYLESHEET = f"""
* {{
    font-size: 13px;
    color: {COLORS["text"]};
}}
QMainWindow, QDialog {{ background: {COLORS["canvas"]}; }}
QWidget#appShell {{ background: {COLORS["canvas"]}; }}
QFrame#sidebar {{
    background: {COLORS["sidebar"]};
    border-right: 1px solid {COLORS["border_soft"]};
}}
QFrame#contentHeader {{
    background: {COLORS["surface"]};
    border-bottom: 1px solid {COLORS["border_soft"]};
}}
QFrame#card, QGroupBox {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border_soft"]};
    border-radius: 8px;
}}
QFrame#card[validationError="true"] {{
    border: 2px solid {COLORS["amber"]};
}}
QGroupBox {{
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; }}
QLabel#pageTitle {{ font-size: 26px; font-weight: 650; }}
QLabel#pageSubtitle, QLabel#muted {{ color: {COLORS["muted"]}; }}
QLabel#sectionTitle {{ font-size: 15px; font-weight: 650; }}
QLabel#eyebrow {{ color: {COLORS["muted"]}; font-size: 11px; font-weight: 650; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background: {COLORS["canvas"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 7px 9px;
    selection-background-color: {COLORS["cyan_soft"]};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
    border: 1px solid {COLORS["cyan"]};
}}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QPushButton {{
    background: {COLORS["surface_2"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    min-height: 34px;
    padding: 0 14px;
    font-weight: 550;
}}
QPushButton:hover {{ background: {COLORS["surface_3"]}; border-color: #35617e; }}
QPushButton:pressed {{ background: #0a1b29; }}
QPushButton:disabled {{ color: {COLORS["muted_2"]}; background: #0a1824; border-color: #183044; }}
QPushButton[primary="true"] {{ background: #138ec6; border-color: #1eb8ef; color: white; }}
QPushButton[primary="true"]:hover {{ background: #169fdc; }}
QPushButton[primary="true"]:disabled {{
    color: {COLORS["muted_2"]};
    background: #0a1824;
    border-color: #183044;
}}
QPushButton[danger="true"] {{ color: #ff8c7c; border-color: #7e3e39; }}
QPushButton[running="true"], QPushButton[running="true"]:disabled {{
    background: #197c5f;
    border-color: {COLORS["green"]};
    color: white;
}}
QPushButton[cancelActive="true"], QPushButton[cancelActive="true"]:hover {{
    background: #b93632;
    border-color: {COLORS["red"]};
    color: white;
}}
QToolButton {{
    background: transparent;
    border: 0;
    border-radius: 5px;
    padding: 5px;
}}
QToolButton:hover, QToolButton:focus {{ background: {COLORS["surface_3"]}; }}
QToolTip {{
    background: #0b1c2b;
    color: {COLORS["text"]};
    border: 1px solid #7890a5;
    padding: 8px;
    opacity: 250;
}}
QProgressBar {{
    background: #06121c;
    border: 1px solid {COLORS["border_soft"]};
    border-radius: 5px;
    text-align: center;
    min-height: 10px;
}}
QProgressBar::chunk {{ background: {COLORS["cyan"]}; border-radius: 4px; }}
QTableWidget, QTreeWidget {{
    background: {COLORS["surface"]};
    alternate-background-color: #0b1e2e;
    border: 1px solid {COLORS["border_soft"]};
    border-radius: 7px;
    gridline-color: {COLORS["border_soft"]};
}}
QHeaderView::section {{
    background: {COLORS["surface_2"]};
    color: {COLORS["muted"]};
    border: 0;
    border-bottom: 1px solid {COLORS["border"]};
    padding: 8px;
    font-weight: 600;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2b4960; min-height: 28px; border-radius: 5px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QTabWidget::pane {{ border: 1px solid {COLORS["border_soft"]}; border-radius: 7px; }}
QTabBar::tab {{ background: {COLORS["sidebar"]}; color: {COLORS["muted"]}; padding: 9px 14px; }}
QTabBar::tab:selected {{ color: {COLORS["cyan"]}; border-bottom: 2px solid {COLORS["cyan"]}; }}
QSplitter::handle {{ background: {COLORS["border_soft"]}; }}
"""
