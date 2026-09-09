"""EEG Viewer - real-time, cross-platform 32-channel EEG monitor (PySide6).

Package marker. The application is launched via :mod:`eeg_viewer.main`; this
file exists so that ``import eeg_viewer`` (and the subpackages) works cleanly
from tools such as pytest without the ``sys.path`` workaround that
:mod:`eeg_viewer.main` performs for its ``__main__`` block.

Subpackages:

* :mod:`eeg_viewer.backend`  - data producers (worker thread -> ring buffer)
* :mod:`eeg_viewer.core`     - model, controller, filters, montage
* :mod:`eeg_viewer.ui`       - PySide6 widgets + pyqtgraph canvas
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:  # pragma: no cover - thin re-export for convenience
    """Convenience launcher (delegates to :func:`eeg_viewer.main.main`)."""
    from .main import main as _main

    return _main()
