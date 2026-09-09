"""Application entry point.

Handles High-DPI setup, configures pyqtgraph according to the detected
platform, and constructs the :class:`~eeg_viewer.ui.main_window.MainWindow`.

Run with::

    python -m eeg_viewer.main        # or
    python eeg_viewer/main.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _bootstrap_highdpi() -> None:
    """Enable High-DPI scaling before the QApplication exists.

    Must run *before* ``QApplication`` is constructed. Done via env vars so the
    policy applies regardless of Qt version quirks.
    """
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    # Qt6 uses this rounding policy; safe to set even on older builds.
    try:
        from PySide6.QtCore import Qt
        # Touch the enum so it is initialised before QApplication.
        _ = Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    except Exception:  # noqa: BLE001
        pass


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    _bootstrap_highdpi()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # Suppress Qt's auto-creation of native sibling widgets around the
    # QOpenGLWidget-backed TraceCanvas (QTBUG-49657). Without this, the menubar,
    # statusbar, transport bar, and combo/menu popups fail to paint in fullscreen
    # (and partially in windowed mode) because the forced-native siblings break
    # Qt's normal widget compositing with the GL surface. Must be set before
    # QApplication is constructed.
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True
    )

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("EEG Viewer")
    app.setOrganizationName("EEGViewer")
    app.setApplicationVersion("1.0")

    # Apply theme + OpenGL policy based on the detected platform.
    from .config import detect_platform
    from .ui.theme import apply_dark_clinical_theme

    platform_profile = detect_platform()
    apply_dark_clinical_theme(app)

    from .ui.main_window import MainWindow

    window = MainWindow(platform_profile)
    window.show()

    # Soft cap on render FPS for low-power devices is enforced by the
    # controller's QTimer interval (see PlatformProfile).
    return app.exec()


if __name__ == "__main__":
    # When executed as ``python eeg_viewer/main.py`` the package parent must be
    # on sys.path so ``from eeg_viewer...`` imports resolve.
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    # Re-launch as the package so relative imports work.
    from eeg_viewer.main import main as _main  # noqa: E402

    raise SystemExit(_main())
