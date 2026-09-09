"""Clinical dark theme - QSS, palette and font-metric helpers.

All visual sizing is derived from the application font and the screen DPI; no
pixel literals are hardcoded anywhere in the UI. This keeps the app crisp on a
4K desktop monitor and readable on a 7" Raspberry Pi touchscreen without any
per-platform styling branches.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QApplication

from ..config import (
    ACCENT,
    ACCENT_WARN,
    BORDER,
    CANVAS_BG,
    CANVAS_GRID_MAJOR,
    CANVAS_GRID_MINOR,
    LIVE_RED,
    PANEL_BG,
    PANEL_BG_ALT,
    TEXT_DIM,
    TEXT_PRIMARY,
)


class Palette:
    """Hex colors mirrored from :mod:`eeg_viewer.config` for QSS interpolation."""
    canvas_bg = CANVAS_BG
    canvas_grid_major = CANVAS_GRID_MAJOR
    canvas_grid_minor = CANVAS_GRID_MINOR
    panel_bg = PANEL_BG
    panel_bg_alt = PANEL_BG_ALT
    text_primary = TEXT_PRIMARY
    text_dim = TEXT_DIM
    accent = ACCENT
    accent_warn = ACCENT_WARN
    live_red = LIVE_RED
    border = BORDER


class Fonts:
    """Lazy font helpers built from the active application font.

    ``em()`` returns a unit proportional to the default font size so layouts
    scale with the user's DPI / font settings.
    """

    _app_font: QFont | None = None

    @classmethod
    def app_font(cls) -> QFont:
        app = QApplication.instance()
        if app is not None:
            return app.font()
        return QFont()

    @classmethod
    def em(cls, factor: float = 1.0) -> int:
        """Return ``factor`` * default font pixel size, rounded to int.

        Use everywhere a pixel size would otherwise be hardcoded.
        """
        fm = QFontMetrics(cls.app_font())
        return max(1, int(round(fm.height() * factor)))

    @classmethod
    def mono(cls, point_size: float | None = None) -> QFont:
        f = QFont("DejaVu Sans Mono")
        f.setStyleHint(QFont.StyleHint.Monospace)
        if point_size is not None:
            f.setPointSizeF(float(point_size))
        return f


_QSS = f"""
* {{
    color: {Palette.text_primary};
    font-family: 'Segoe UI', 'DejaVu Sans', 'Arial', sans-serif;
    outline: none;
}}

QWidget#Root, QMainWindow {{
    background-color: {Palette.panel_bg};
}}

QMenuBar {{
    background-color: {Palette.panel_bg_alt};
    padding: 2px 4px;
    border-bottom: 1px solid {Palette.border};
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: {Palette.accent};
}}

QMenu {{
    background-color: {Palette.panel_bg_alt};
    border: 1px solid {Palette.border};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 22px 6px 18px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {Palette.accent};
}}
QMenu::separator {{
    height: 1px;
    background: {Palette.border};
    margin: 4px 8px;
}}

QToolBar {{
    background-color: {Palette.panel_bg_alt};
    border: none;
    border-top: 1px solid {Palette.border};
    border-bottom: 1px solid {Palette.border};
    spacing: 4px;
    padding: 4px;
}}

QSplitter::handle {{
    background-color: {Palette.border};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

QPushButton {{
    background-color: {Palette.panel_bg_alt};
    border: 1px solid {Palette.border};
    border-radius: 4px;
    padding: 6px 14px;
    min-height: {20}px;
}}
QPushButton:hover {{
    border-color: {Palette.accent};
    background-color: #1f2937;
}}
QPushButton:pressed, QPushButton:checked {{
    background-color: {Palette.accent};
    border-color: {Palette.accent};
}}
QPushButton:disabled {{
    color: {Palette.text_dim};
    background-color: {Palette.panel_bg};
}}
QPushButton#Primary {{
    background-color: {Palette.accent};
    border-color: {Palette.accent};
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: #4a96ff; }}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px 8px;
}}
QToolButton:hover {{
    background-color: {Palette.panel_bg_alt};
    border-color: {Palette.border};
}}
QToolButton:checked {{
    background-color: {Palette.accent};
    color: #ffffff;
}}

/* Flat transport buttons (back / play-pause / forward / stop): no rectangular
   box, just a glyph that highlights on hover and turns accent while active. */
QPushButton#Transport {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px 8px;
    min-height: {20}px;
    color: {Palette.text_primary};
    font-size: {Fonts.em(1.1)}px;
}}
QPushButton#Transport:hover {{
    color: {Palette.accent};
    background: {Palette.panel_bg_alt};
}}
QPushButton#Transport:checked {{
    color: {Palette.accent};
    font-weight: 600;
}}
QPushButton#Transport:disabled {{
    color: {Palette.text_dim};
}}

QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: {Palette.panel_bg_alt};
    border: 1px solid {Palette.border};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {Palette.accent};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {Palette.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {Palette.panel_bg_alt};
    border: 1px solid {Palette.border};
    selection-background-color: {Palette.accent};
    outline: none;
}}

QCheckBox {{ spacing: 8px; padding: 2px 0; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {Palette.border};
    border-radius: 3px;
    background-color: {Palette.panel_bg_alt};
}}
QCheckBox::indicator:hover {{ border-color: {Palette.accent}; }}
QCheckBox::indicator:checked {{
    background-color: {Palette.accent};
    border-color: {Palette.accent};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {Palette.border};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {Palette.accent};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {Palette.text_primary};
    border: 2px solid {Palette.accent};
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{ background: {Palette.accent}; }}

QLabel {{ background: transparent; }}
QLabel#Dim {{ color: {Palette.text_dim}; }}
QLabel#Heading {{ font-weight: 600; font-size: 13px; padding: 4px 0; }}
QLabel#Mono {{ font-family: 'DejaVu Sans Mono', 'Consolas', monospace; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {Palette.border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {Palette.text_dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {Palette.border};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {Palette.text_dim}; }}

QStatusBar {{
    background-color: {Palette.panel_bg_alt};
    border-top: 1px solid {Palette.border};
    color: {Palette.text_dim};
}}
QStatusBar QLabel {{ padding: 0 10px; }}

QFrame#Pane {{
    background-color: {Palette.panel_bg_alt};
    border: 1px solid {Palette.border};
    border-radius: 6px;
}}

QFrame#RegionHeader {{
    background-color: {Palette.panel_bg};
    border: none;
    border-bottom: 1px solid {Palette.border};
}}
"""


def qss() -> str:
    """Return the full application stylesheet."""
    return _QSS


def apply_dark_clinical_theme(app: QApplication) -> None:
    """Apply the dark clinical theme + OpenGL policy to ``app``."""
    import pyqtgraph as pg
    from ..config import detect_platform

    profile = detect_platform()
    pg.setConfigOption("background", CANVAS_BG)
    pg.setConfigOption("foreground", TEXT_DIM)
    pg.setConfigOption("antialias", not profile.is_embedded)
    # OpenGL accelerates x86 desktops; on Pi the software path is more robust.
    pg.setConfigOption("useOpenGL", bool(profile.use_opengl))
    pg.setConfigOption("leftButtonPan", False)
    app.setStyleSheet(_QSS)


def hex_to_qcolor(hex_str: str) -> QColor:
    return QColor(hex_str)
