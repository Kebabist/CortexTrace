"""UI integration tests using PySide6.QtTest and offscreen QApplication."""

from __future__ import annotations

import sys
import os
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from eeg_viewer.config import detect_platform
from eeg_viewer.ui.main_window import MainWindow
from eeg_viewer.ui.theme import apply_dark_clinical_theme


@pytest.fixture(scope="module")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    apply_dark_clinical_theme(app)
    yield app
    app.processEvents()


def test_main_window_creation(qapp):
    profile = detect_platform()
    window = MainWindow(profile)
    qapp.processEvents()
    assert window.windowTitle() == "EEG Viewer"
    assert window.model is not None
    assert window.controller is not None
    window.close()
    qapp.processEvents()


def test_main_window_load_file_and_ui_controls(qapp, tmp_path):
    file_path = tmp_path / "sample_eeg.txt"
    # Create a 32-channel text file with 50 samples
    header = " ".join([f"CH{i+1}" for i in range(32)])
    rows = [header]
    for _ in range(50):
        rows.append(" ".join(["1.0"] * 32))
    file_path.write_text("\n".join(rows), encoding="utf-8")

    profile = detect_platform()
    window = MainWindow(profile)
    qapp.processEvents()

    window.load_file(str(file_path))
    qapp.processEvents()

    assert window.model.n_channels == 32
    assert window.model.total_samples == 50

    # Test channel panel interaction
    channel_panel = window._channels
    channel_panel._on_solo(0)
    qapp.processEvents()
    assert window.model.visible_channels() == [0]

    channel_panel._on_all()
    qapp.processEvents()
    assert len(window.model.visible_channels()) == 32

    # Test playback controls interaction
    controls = window._controls
    controls._sens.setCurrentText("200 µV")
    qapp.processEvents()
    assert window.model.viewport.sensitivity_uv == 200.0

    controls._tb.setCurrentText("15s")
    qapp.processEvents()
    assert window.model.viewport.timebase_s == 15.0

    # Test DSA toggle
    controls._dsa.setChecked(True)
    qapp.processEvents()
    assert window.model.viewport.dsa_enabled is True

    window.close()
    qapp.processEvents()
