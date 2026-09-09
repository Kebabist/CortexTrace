"""Density Spectral Array (DSA) trend panel.

A self-contained pyqtgraph widget shown directly beneath the waveform canvas.
It is a ``GraphicsLayoutWidget`` in its own right — crucially with OpenGL
*disabled* on its view — because ``ImageItem`` rendering under pyqtgraph's
``useOpenGL=True`` is the known-fragile combination that segfaults during the
GL texture-upload path. By owning its own ``GraphicsView`` (separate from the
GL-accelerated waveform canvas) this widget always renders on the reliable
software (QPainter) path, while the waveform keeps its GL acceleration.

The DSA's X axis is linked to the waveform's plot via ``setXLink``, which works
across separate ``GraphicsView`` instances (pyqtgraph links ViewBoxes through
``sigRangeChanged``), so the spectrogram tracks the same absolute-time window
in both sweep and scroll modes.

State lives in the model (``viewport.dsa_enabled`` / ``viewport.dsa_mode``);
this class only renders. Spectral computation is delegated to
:class:`~eeg_viewer.core.spectral.SpectralEngine`; data is read from the model's
ring buffer and filtered with the model's filter chain (notch/band-pass run
*before* the FFT so mains hum never stripes the display).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtWidgets import QSizePolicy

from ..config import (
    DSA_COLORMAP_STOPS,
    DSA_FREQ_MAX_HZ,
    DSA_FREQ_MIN_HZ,
    DSA_UPDATE_INTERVAL_MS,
)
from ..core.spectral import DSAFrame, SpectralEngine
from .theme import Fonts, Palette


def _build_lut() -> np.ndarray:
    """Build the blue->red lookup table from the config colour stops.

    The stops are stored as 0..1 floats, but pyqtgraph's ``mkColor`` casts each
    channel with ``int()`` (no ``*255``), so 0..1 floats are truncated to 0/1
    and the whole colormap renders near-black. Convert to 0..255 uint8 first.
    """
    positions = [p for p, _ in DSA_COLORMAP_STOPS]
    colors = np.array([c for _, c in DSA_COLORMAP_STOPS], dtype=np.float64)
    colors = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    cmap = pg.ColorMap(positions, colors)
    return cmap.getLookupTable(start=positions[0], stop=positions[-1], alpha=True)


class DSAPanel(pg.GraphicsLayoutWidget):
    """A DSA spectrogram panel rendered on the software path.

    Constructed once by the main window and shown/hidden through
    :meth:`set_enabled`. X-linked to the waveform plot so its time axis always
    matches the canvas above it.
    """

    def __init__(self, model, waveform_plot: pg.PlotItem, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._waveform = waveform_plot
        self._enabled = False
        self._track_labels: List[pg.TextItem] = []

        # Force the SOFTWARE renderer. The global config enables OpenGL on
        # desktop, but ImageItem + OpenGL segfaults during texture upload. This
        # widget owns its own GraphicsView, so disabling GL here is isolated
        # from the GL-accelerated waveform canvas.
        self.useOpenGL(False)
        self.setBackground(Palette.canvas_bg)
        self.setAntialiasing(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ---- plot + image -------------------------------------------------- #
        self._plot: pg.PlotItem = self.addPlot(row=0, col=0)
        self._plot.setXLink(waveform_plot)          # cross-widget sync guarantee
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=False, y=False, alpha=0)
        vb = self._plot.vb
        vb.setMouseEnabled(x=False, y=False)
        vb.disableAutoRange(pg.ViewBox.XYAxes)
        # Invert so the first image row (highest frequency of the top track) is
        # drawn at the top — matches the engine's row ordering.
        vb.invertY(True)

        # Y axis: frequency is implied per band, so hide numeric ticks; a label
        # clarifies the axis. Track names are drawn as TextItems.
        self._plot.hideAxis("left")
        left = self._plot.getAxis("left")
        left.setStyle(tickFont=Fonts.mono())
        left.setLabel("DSA", color=Palette.text_dim)
        self._plot.setLabel("bottom", "Time", units="s", color=Palette.text_dim)

        self._image = pg.ImageItem(axisOrder="col-major")
        # Cache the LUT and apply it ourselves in _map_to_rgba (ImageItem is fed
        # a pre-coloured uint8 RGBA array), so the float->colour mapping happens
        # in numpy rather than pyqtgraph's fragile auto-leveling path.
        self._lut = _build_lut()
        self._image.setZValue(0)
        self._plot.addItem(self._image)

        # ---- engine + update timer ---------------------------------------- #
        self._engine = SpectralEngine(model.sample_rate_hz, model.viewport.dsa_mode)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.setInterval(DSA_UPDATE_INTERVAL_MS)
        # Connect to refresh(), NOT the QWidget.update() builtin we must not
        # shadow (see refresh() docstring).
        self._timer.timeout.connect(self.refresh)

    # ------------------------------------------------------------------ #
    # Public API (driven by model signals, connected from the main window)
    # ------------------------------------------------------------------ #
    def set_enabled(self, enabled: bool) -> None:
        """Show/hide the panel and start/stop its update timer."""
        enabled = bool(enabled)
        self._enabled = enabled
        if enabled:
            self.show()
            self.refresh()
            self._timer.start()
        else:
            self._timer.stop()
            self._image.clear()
            self._clear_track_labels()
            self.hide()

    def set_mode(self, mode: str) -> None:
        """Switch the DSA track layout (hemisphere/regional)."""
        self._engine.reconfigure(mode=mode)
        self.refresh()

    def on_metadata(self) -> None:
        """Reconfigure the engine for a new acquisition (sample rate/channels)."""
        self._engine.reconfigure(
            sample_rate_hz=self._model.sample_rate_hz,
            names=self._model.channel_names,
        )
        if self._enabled:
            self.refresh()

    def refresh(self) -> None:
        """Recompute and redraw the DSA for the waveform's current X window.

        Named ``refresh`` (not ``update``) deliberately: this widget subclasses
        QWidget, whose ``update()`` is a builtin slot scheduling a repaint.
        Shadowing it would route Qt's internal ``update()`` calls into our
        spectral render path, causing re-entrancy / state corruption that can
        crash in C. Connections and the timer call this method by name.
        """
        if not self._enabled:
            return
        try:
            self._do_refresh()
        except Exception:  # noqa: BLE001 - never let a render fault crash the app
            # ImageItem rendering is the most fragile part of pyqtgraph; if a
            # render fails, blank the panel and let playback continue rather
            # than taking the whole window down. The next tick retries.
            import traceback
            traceback.print_exc()
            try:
                self._image.clear()
            except Exception:  # noqa: BLE001
                pass

    def _do_refresh(self) -> None:
        model = self._model
        if model.n_channels == 0 or model.buffer is None:
            self._image.clear()
            return

        # Use the waveform's live X range as the analysed window so the DSA
        # matches exactly what is displayed (sweep page or scrolling window).
        (start_t, end_t), _ = self._waveform.vb.viewRange()
        duration = end_t - start_t
        if duration <= 0:
            return

        n_samples = max(1, int(round(model.sample_rate_hz * duration)))
        view, available = model.buffer.latest_window(n_samples)
        if available < 2:
            self._image.clear()
            return
        if available < n_samples:
            view = view[:, -available:]

        # Filter (notch + band-pass) BEFORE the FFT so mains hum does not stripe
        # the DSA. Reuse the model's chain: it is effectively stateless across
        # process() calls (zero initial conditions each call), so sharing it
        # with the waveform path is safe.
        filtered = model.filter_chain.process(view)

        frame: DSAFrame = self._engine.compute(
            window=filtered,
            channel_names=model.channel_names,
            visible_indices=model.visible_channels(),
            start_t=start_t,
            end_t=end_t,
        )
        if not frame.valid or frame.image.size == 0:
            self._image.clear()
            return

        # Pre-map the normalised [0,1] float image through the LUT HERE, in
        # numpy, and hand ImageItem a uint8 RGBA array. This sidesteps three
        # fragile pyqtgraph code paths that have crashed (segfault) in practice:
        #   1. float-image + LUT auto-leveling (the levels/LUT-index contract);
        #   2. NaN/inf values reaching the C QImage converter (we sanitize here);
        #   3. the makeARGB float-LUT branch.
        rgba = self._map_to_rgba(frame.image)
        n_tracks = frame.n_tracks
        self._image.setImage(rgba, autoLevels=False)
        self._image.setRect(start_t, 0.0, duration, float(n_tracks))
        self._plot.setYRange(0.0, float(n_tracks), padding=0.0)
        self._update_track_labels(frame.track_names, n_tracks)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _clear_track_labels(self) -> None:
        for lbl in self._track_labels:
            try:
                self._plot.removeItem(lbl)
            except Exception:  # noqa: BLE001 - defensive
                pass
        self._track_labels.clear()

    def _map_to_rgba(self, image: np.ndarray) -> np.ndarray:
        """Map a normalised [0,1] float DSA image to a uint8 RGBA array.

        Doing the colour mapping in numpy (instead of letting ImageItem apply a
        LUT to a float image) keeps the fragile leveling/LUT-index code path
        out of the C QImage converter. NaN/inf are clamped to the lowest band
        here so they never reach native code.
        """
        if image.size == 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        # Sanitize: finite values clamped to [0,1]; non-finite -> 0 (lowest).
        finite = np.where(np.isfinite(image), image, 0.0)
        np.clip(finite, 0.0, 1.0, out=finite)
        lut = self._lut
        n = lut.shape[0]
        idx = (finite * (n - 1)).astype(np.intp)
        return np.ascontiguousarray(lut[idx])

    def _update_track_labels(self, track_names: List[str], n_tracks: int) -> None:
        """Place a name label at the vertical centre of each track band."""
        # Rebuild only when the set/order of tracks changes.
        current = [lbl.toPlainText() for lbl in self._track_labels]
        if current == track_names and len(self._track_labels) == n_tracks:
            return
        self._clear_track_labels()

        (start_t, end_t), _ = self._waveform.vb.viewRange()
        x_pos = start_t + 0.01 * (end_t - start_t)
        color = pg.mkColor(Palette.text_primary)
        color.setAlpha(200)
        for i, name in enumerate(track_names):
            # Track 0 is the top band. With invertY + setRect y=[0,n_tracks],
            # track i centre is at data-Y = n_tracks - 0.5 - i.
            y_pos = float(n_tracks - 0.5 - i)
            label = pg.TextItem(name, color=color, anchor=(0.0, 0.5))
            label.setPos(QPointF(x_pos, y_pos))
            label.setZValue(30)
            self._plot.addItem(label, ignoreBounds=True)
            self._track_labels.append(label)
