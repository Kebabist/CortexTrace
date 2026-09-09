"""UI package: PySide6 widgets + pyqtgraph canvas.

The view layer is intentionally thin. Widgets only render what the model tells
them to and translate user actions back into model/controller calls. Nothing
here reads files or paces the playhead.
"""

from .theme import apply_dark_clinical_theme, qss, Fonts, Palette
from .trace_canvas import TraceCanvas
from .channel_panel import ChannelPanel
from .playback_controls import PlaybackControls
from .status_bar import StatusBar
from .main_window import MainWindow

__all__ = [
    "apply_dark_clinical_theme",
    "qss",
    "Fonts",
    "Palette",
    "TraceCanvas",
    "ChannelPanel",
    "PlaybackControls",
    "StatusBar",
    "MainWindow",
]
