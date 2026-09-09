"""Main application window.

Composes the canvas, channel panel, transport bar and status bar into the
clinical layout. Owns the file-loading flow (File->Open + drag-and-drop) and
persists user preferences via QSettings:

* window geometry + state + splitter sizes
* last channel selection (restored on the next load)
* last working directory
* timebase / sensitivity / filter / speed settings
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import QByteArray, Qt, QSettings, QMimeData
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_NAME,
    APP_ORG,
    DEFAULT_HF_HZ,
    DEFAULT_LF_HZ,
    DEFAULT_NOTCH_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SENSITIVITY_UV,
    DEFAULT_SPEED,
    DEFAULT_TIMEBASE_S,
    DSA_DEFAULT,
    DSA_HEIGHT_RATIO,
    DSA_MODE_DEFAULT,
    KEY_CHANNELS_VISIBLE,
    KEY_DSA,
    KEY_DSA_MODE,
    KEY_HF,
    KEY_LAST_DIR,
    KEY_LF,
    KEY_NOTCH,
    KEY_SENSITIVITY,
    KEY_SPEED,
    KEY_SPLITTER,
    KEY_TIMEBASE,
    KEY_WINDOW_GEO,
    KEY_WINDOW_STATE,
)
from ..backend.data_source import DataSource, SourceState
from ..backend.file_replay import FileReplaySource, FileFormatError
from ..backend.ring_buffer import RingBuffer  # noqa: F401 - re-export convenience
from .dsa_panel import DSAPanel
from ..core.filters import FilterSettings
from ..core.model import EEGModel
from ..core.playback import PlaybackController
from .channel_panel import ChannelPanel
from .playback_controls import PlaybackControls
from .status_bar import StatusBar
from .theme import Fonts, Palette
from .trace_canvas import TraceCanvas


class _EmptyOverlay(QWidget):
    """Shown before any file is loaded."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Pane")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("No data loaded")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: {Fonts.em(1.6)}px; font-weight: 600;")
        hint = QLabel("Choose File ▸ Open  (or drag a .txt file here)")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setObjectName("Dim")
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)


class MainWindow(QMainWindow):
    """Top-level clinical-EEG window."""

    def __init__(self, platform_profile) -> None:
        super().__init__()
        self._platform = platform_profile
        self.setObjectName("Root")
        self.setWindowTitle("EEG Viewer")
        self.resize(1280, 800)

        self._model = EEGModel(self)
        self._controller = PlaybackController(self._model, platform_profile, self)

        self._settings = QSettings(APP_ORG, APP_NAME, self)

        self._build_ui()
        self._build_menu()
        self._restore_state()
        self._wire_signals()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setHandleWidth(2)
        self._splitter.setChildrenCollapsible(False)

        # Left: trace canvas + DSA panel stacked vertically (with an empty
        # overlay shown until a file loads). The DSA lives in its own
        # GL-disabled widget beneath the canvas; its X axis is linked to the
        # canvas's waveform plot so the two stay in time sync.
        self._canvas_host = QFrame()
        self._canvas_host.setObjectName("Pane")
        ch_layout = QVBoxLayout(self._canvas_host)
        ch_layout.setContentsMargins(2, 2, 2, 2)
        ch_layout.setSpacing(2)
        self._canvas = TraceCanvas(self._model, self._canvas_host)
        ch_layout.addWidget(self._canvas)
        self._dsa = DSAPanel(self._model, self._canvas.waveform_plot(), parent=self._canvas_host)
        ch_layout.addWidget(self._dsa)
        # Waveform takes the bulk of the height; DSA gets its configured share.
        dsa_share = max(DSA_HEIGHT_RATIO, 0.05)
        ch_layout.setStretchFactor(self._canvas, int(round((1.0 - dsa_share) * 100)))
        ch_layout.setStretchFactor(self._dsa, int(round(dsa_share * 100)))
        self._dsa.setVisible(self._model.viewport.dsa_enabled)
        self._overlay = _EmptyOverlay(self._canvas_host)
        ch_layout.addWidget(self._overlay)
        self._overlay.raise_()

        # Right: channel panel.
        self._channels = ChannelPanel(self._model)

        self._splitter.addWidget(self._canvas_host)
        self._splitter.addWidget(self._channels)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([900, 300])

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addWidget(self._splitter, stretch=1)

        self._controls = PlaybackControls(self._model, self._controller)
        outer.addWidget(self._controls)

        self.setCentralWidget(central)

        self._status = StatusBar(self._model)
        self.setStatusBar(self._status)

        self._canvas.seek_requested.connect(self._controller.seek_seconds)
        self._controller.frame_ready.connect(self._canvas.paint_frame)

        # Zoom keyboard shortcuts — active in review mode (paused). The canvas
        # methods are no-ops while in live/auto-follow mode, so these are safe
        # to leave wired at all times. Step ~sqrt(2): area halves every two
        # presses, the convention used in clinical imaging viewers.
        QShortcut(QKeySequence("Ctrl++"), self, activated=self._canvas.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self._canvas.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self,
                  activated=lambda: self._canvas.zoom_out(factor=1.4))
        # Home: reset zoom to the default time window (review mode only).
        QShortcut(QKeySequence("Home"), self, activated=self._canvas.reset_zoom)

        # Transport keyboard shortcuts. Left/Right step by one timebase and land
        # the cursor at the left edge of the new page (matches the ◀ ▶ buttons).
        # Space toggles play/pause, S stops+resets. These are standard
        # media-player conventions; Left/Right were previously unused (only
        # Ctrl+/- zoom used them with modifiers), so no conflicts.
        QShortcut(QKeySequence(Qt.Key_Left), self,
                  activated=lambda: self._controller.step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self,
                  activated=lambda: self._controller.step(+1))
        QShortcut(QKeySequence(Qt.Key_Space), self,
                  activated=self._controller.toggle)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self._controller.stop)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self._action_open = QAction("&Open…", self)
        self._action_open.setShortcut(QKeySequence.Open)
        self._action_open.triggered.connect(self._on_open)
        file_menu.addAction(self._action_open)

        self._action_close = QAction("&Close source", self)
        self._action_close.triggered.connect(self._on_close_source)
        file_menu.addAction(self._action_close)

        file_menu.addSeparator()
        self._action_quit = QAction("&Quit", self)
        self._action_quit.setShortcut(QKeySequence.Quit)
        self._action_quit.triggered.connect(self.close)
        file_menu.addAction(self._action_quit)

        view_menu = menubar.addMenu("&View")
        self._action_fullscreen = QAction("&Fullscreen", self)
        self._action_fullscreen.setShortcut(QKeySequence.FullScreen)
        self._action_fullscreen.setCheckable(True)
        self._action_fullscreen.toggled.connect(self._on_fullscreen)
        view_menu.addAction(self._action_fullscreen)

        self._action_hide_panel = QAction("Hide &channel panel", self)
        self._action_hide_panel.setShortcut(QKeySequence("Ctrl+P"))
        self._action_hide_panel.setCheckable(True)
        self._action_hide_panel.toggled.connect(self._channels.setHidden)
        view_menu.addAction(self._action_hide_panel)

        help_menu = menubar.addMenu("&Help")
        self._action_about = QAction("&About", self)
        self._action_about.triggered.connect(self._on_about)
        help_menu.addAction(self._action_about)

    # ------------------------------------------------------------------ #
    # Signal wiring
    # ------------------------------------------------------------------ #
    def _wire_signals(self) -> None:
        self._model.metadata_changed.connect(self._on_metadata)
        self._model.source_eof.connect(self._on_eof)
        self._channels.visibility_changed.connect(self._persist_channel_selection)

        # DSA panel lifecycle: show/hide, switch track layout, follow new
        # acquisition metadata, and recompute immediately on discrete view
        # changes (timebase/filter/visibility/seek). Continuous playback refresh
        # is driven by the panel's own slower timer — NOT by data_changed,
        # which fires per ingest chunk and would re-run the FFT far too often.
        self._model.dsa_enabled_changed.connect(self._dsa.set_enabled)
        self._model.dsa_mode_changed.connect(self._dsa.set_mode)
        self._model.metadata_changed.connect(self._dsa.on_metadata)
        # NOTE: connect to refresh(), not the QWidget.update() builtin the
        # DSA widget must not shadow (see DSAPanel.refresh docstring).
        self._model.viewport_changed.connect(self._dsa.refresh)
        self._model.visibility_changed.connect(self._dsa.refresh)
        self._model.filters_changed.connect(self._dsa.refresh)
        # After a seek the waveform X range moves; recompute the DSA so it
        # matches the newly displayed window without waiting for the timer.
        self._model.position_changed.connect(self._dsa.refresh)

    # ------------------------------------------------------------------ #
    # File loading
    # ------------------------------------------------------------------ #
    def _on_open(self) -> None:
        last_dir = self._settings.value(KEY_LAST_DIR, "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EEG recording",
            last_dir or os.path.expanduser("~"),
            "EEG text (*.txt);;All files (*)",
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        """Load and start streaming a text-file recording."""
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open file", f"File not found:\n{path}")
            return
        self._settings.setValue(KEY_LAST_DIR, os.path.dirname(path))

        # Tear down any previous source.
        self._controller.detach_source()

        source = FileReplaySource(path, DEFAULT_SAMPLE_RATE_HZ, parent=self)
        # Receive source errors on the GUI thread.
        source.error.connect(self._on_source_error)
        # Attach BEFORE load() so the metadata_ready signal reaches the model
        # and configures the ring buffer before the first chunk is emitted.
        self._controller.attach_source(source)

        try:
            source.load()   # emits metadata_ready synchronously
        except Exception as exc:  # noqa: BLE001 - surface to the user
            self._on_source_error(str(exc))
            return

        # Now that metadata is known, apply persisted view + channel selection.
        self._apply_persisted_settings(source.channel_names())

        self._overlay.hide()
        self._status.set_source_label(f"FILE  {os.path.basename(path)}")
        self._controller.play()

    def _on_close_source(self) -> None:
        self._controller.detach_source()
        self._model.configure_acquisition([], DEFAULT_SAMPLE_RATE_HZ)
        self._overlay.show()
        self._status.set_source_label("No source")

    def _on_source_error(self, message: str) -> None:
        QMessageBox.critical(self, "Source error", message)

    def _on_metadata(self) -> None:
        # Buffer sized - canvas will rebuild from its own metadata signal.
        self._status.set_live(self._model.is_live)

    def _on_eof(self) -> None:
        if self._model.viewport.loop and not self._model.is_live:
            # Restart from zero for looping playback.
            self._controller.seek_seconds(0.0)
            self._controller.play()

    # ------------------------------------------------------------------ #
    # Settings persistence
    # ------------------------------------------------------------------ #
    def _apply_persisted_settings(self, channel_names: List[str]) -> None:
        timebase = float(self._settings.value(KEY_TIMEBASE, DEFAULT_TIMEBASE_S, type=float))
        sensitivity = float(self._settings.value(KEY_SENSITIVITY, DEFAULT_SENSITIVITY_UV, type=float))
        speed = float(self._settings.value(KEY_SPEED, DEFAULT_SPEED, type=float))
        hf = float(self._settings.value(KEY_HF, DEFAULT_HF_HZ, type=float))
        lf = float(self._settings.value(KEY_LF, DEFAULT_LF_HZ, type=float))
        notch = float(self._settings.value(KEY_NOTCH, DEFAULT_NOTCH_HZ, type=float))

        # Route timebase through the controller: when a source is already
        # attached it refills the buffer from the file if the timebase grew.
        # At this point in load() no source is attached yet, so the controller
        # delegates straight to model.set_timebase (no refill, no crash).
        self._controller.set_timebase(timebase)
        self._model.set_sensitivity(sensitivity)
        self._controller.set_speed(speed)
        self._model.set_filters(FilterSettings(hf_hz=hf, lf_hz=lf, notch_hz=notch))

        # DSA trend panel: persisted enable flag + track layout.
        dsa_enabled = bool(self._settings.value(KEY_DSA, DSA_DEFAULT, type=bool))
        dsa_mode = str(self._settings.value(KEY_DSA_MODE, DSA_MODE_DEFAULT, type=str))
        self._model.set_dsa_enabled(dsa_enabled)
        self._model.set_dsa_mode(dsa_mode)

        # Channel selection (default: all visible on first run).
        raw = self._settings.value(KEY_CHANNELS_VISIBLE, "", type=str)
        if raw:
            try:
                wanted = {int(x) for x in raw.split(",") if x.strip().isdigit()}
                visible = [i for i in range(len(channel_names)) if i in wanted]
            except Exception:  # noqa: BLE001 - corrupt setting -> fall back to all
                visible = list(range(len(channel_names)))
        else:
            visible = list(range(len(channel_names)))
        self._model.set_visible_channels(visible)

        # Sync transport controls to restored values.
        self._controls._speed.setCurrentText(f"{speed:g}×")
        self._controls._sens.setCurrentText(f"{int(sensitivity)} µV")
        self._controls._tb.setCurrentText(f"{int(timebase)}s")
        # HF / LF are now preset comboboxes (like Notch): match by data value
        # rather than free-form setValue. If the persisted value isn't in the
        # preset list (e.g. an older setting), fall back to Off (index 0).
        for combo, val in ((self._controls._hf, hf),
                           (self._controls._lf, lf),
                           (self._controls._notch, notch)):
            idx = combo.findData(float(val))
            if idx < 0:
                idx = combo.findData(0.0)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        # DSA checkbox + layout dropdown reflect the persisted state. Block
        # signals so this programmatic sync does not round-trip back into the
        # model setters (which would be harmless but redundant).
        self._controls._dsa.blockSignals(True)
        self._controls._dsa.setChecked(dsa_enabled)
        self._controls._dsa.blockSignals(False)
        self._controls._dsa_mode.setEnabled(dsa_enabled)
        mode_idx = self._controls._dsa_mode.findData(dsa_mode)
        if mode_idx >= 0:
            self._controls._dsa_mode.blockSignals(True)
            self._controls._dsa_mode.setCurrentIndex(mode_idx)
            self._controls._dsa_mode.blockSignals(False)

    def _persist_channel_selection(self) -> None:
        visible = ",".join(str(i) for i in self._model.visible_channels())
        self._settings.setValue(KEY_CHANNELS_VISIBLE, visible)

    def _save_view_settings(self) -> None:
        v = self._model.viewport
        self._settings.setValue(KEY_TIMEBASE, float(v.timebase_s))
        self._settings.setValue(KEY_SENSITIVITY, float(v.sensitivity_uv))
        self._settings.setValue(KEY_SPEED, float(v.speed))
        self._settings.setValue(KEY_HF, float(self._model.filter_chain.settings.hf_hz))
        self._settings.setValue(KEY_LF, float(self._model.filter_chain.settings.lf_hz))
        self._settings.setValue(KEY_NOTCH, float(self._model.filter_chain.settings.notch_hz))
        self._settings.setValue(KEY_DSA, bool(v.dsa_enabled))
        self._settings.setValue(KEY_DSA_MODE, str(v.dsa_mode))

    def _restore_state(self) -> None:
        geo = self._settings.value(KEY_WINDOW_GEO)
        if geo is not None:
            self.restoreGeometry(geo)
        state = self._settings.value(KEY_WINDOW_STATE)
        if state is not None:
            self.restoreState(state)
        sp = self._settings.value(KEY_SPLITTER)
        if sp is not None:
            self._splitter.restoreState(sp)

    def _save_state(self) -> None:
        self._settings.setValue(KEY_WINDOW_GEO, self.saveGeometry())
        self._settings.setValue(KEY_WINDOW_STATE, self.saveState())
        self._settings.setValue(KEY_SPLITTER, self._splitter.saveState())
        self._save_view_settings()
        self._persist_channel_selection()

    # ------------------------------------------------------------------ #
    # View menu actions
    # ------------------------------------------------------------------ #
    def _on_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About EEG Viewer",
            "<h3>EEG Viewer</h3>"
            "<p>A real-time, cross-platform EEG monitor built with PySide6 "
            "and pyqtgraph.</p>"
            f"<p>Platform: <b>{self._platform.label}</b> · "
            f"Render: <b>{self._platform.render_fps} FPS</b> · "
            f"OpenGL: <b>{'on' if self._platform.use_opengl else 'off'}</b></p>"
            "<p style='color: #8b949e'>File→Open to load a recording, or drag a "
            ".txt file into the window.</p>",
        )

    # ------------------------------------------------------------------ #
    # Drag & drop
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if p.lower().endswith(".txt"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and os.path.isfile(p):
                self.load_file(p)
                event.acceptProposedAction()
                return
        event.ignore()

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        self._controller.detach_source()
        self._save_state()
        super().closeEvent(event)

    # Convenience accessors for tests / scripting ------------------------- #
    @property
    def model(self) -> EEGModel:
        return self._model

    @property
    def controller(self) -> PlaybackController:
        return self._controller
