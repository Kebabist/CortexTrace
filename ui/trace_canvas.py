"""The trace canvas - a single pyqtgraph plot that renders all visible channels.

Why one plot, not 32?
* A single scene graph redraws in one pass - the only practical way to hit
  60 FPS with 32 channels on a Raspberry Pi.
* Lanes are created by vertical offsets (``raw * gain + lane_offset``), the
  standard clinical-EEG rendering trick.
* Curves are created once and only ``setData`` is called per frame -> the hot
  path allocates nothing.

The canvas:
* Reads the model's :py:meth:`~eeg_viewer.core.model.EEGModel.render_window`
  zero-copy slice every frame.
* Recomputes lane geometry from its live pixel size on every resize -> fully
  scalable, nothing hardcoded.
* Channel labels are rendered in a dedicated left axis bar.
* Skips hidden channels (cost scales with N_visible, not 32).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..config import (
    build_channel_palette,
    detect_platform,
    SWEEP_CURSOR_ALPHA,
    SWEEP_CURSOR_WIDTH_PX,
)
from .theme import Fonts, Palette


class _ChannelLabelAxis(pg.AxisItem):
    """Left axis that displays channel names at lane centre positions.

    Replaces the previous in-scene text-label approach: labels now live
    in the axis margin so they never overlap the waveform data area.
    """

    def __init__(self, model, **kwargs) -> None:
        super().__init__(orientation="left", **kwargs)
        self._model = model
        # Hide tick marks and the axis line — only text is shown.
        self.setPen(None)
        self.setStyle(tickFont=self._label_font(), tickLength=0, tickTextOffset=2)
        self.setTextPen(pg.mkPen(Palette.text_primary))
        # Auto-size to fit text exactly (autoExpandTextSpace=True by default).
        self.setWidth(None)
        # Disable auto-tick generation; we control ticks explicitly.
        self._channel_ticks: list = []

    @staticmethod
    def _label_font() -> QFont:
        f = QFont(Fonts.app_font().family(), -1, QFont.Weight.Bold)
        return f

    def update_labels(
        self, visible_indices: List[int], channel_names: List[str], n_visible: int
    ) -> None:
        """Rebuild the tick list so channel names sit at lane centres."""
        ticks: list = []
        top = float(n_visible)
        for lane_i, ch_idx in enumerate(visible_indices):
            y = top - 0.5 - lane_i
            name = channel_names[ch_idx] if ch_idx < len(channel_names) else str(ch_idx)
            ticks.append((y, name))
        self._channel_ticks = ticks
        # Trigger a redraw by setting ticks (major level only).
        self.setTicks([ticks])
        self.update()


class TraceCanvas(pg.GraphicsLayoutWidget):
    """Real-time multi-channel EEG canvas."""

    # Emitted when the user scrubs the timeline by dragging on the canvas.
    seek_requested = Signal(float)

    def __init__(self, model, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = model
        self._platform = detect_platform()

        self.setBackground(Palette.canvas_bg)
        self.setAntialiasing(not self._platform.is_embedded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # A single PlotItem hosts all lane curves.
        self._plot: pg.PlotItem = self.addPlot(row=0, col=0)
        # Mouse interaction is toggled per-mode by `_apply_interaction_mode()`.
        self._apply_interaction_mode()
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=True, y=True, alpha=1)
        self._plot.setDownsampling(auto=True, mode="peak")
        self._plot.setClipToView(True)

        # Left axis: channel name labels (outside the data area).
        self._channel_axis = _ChannelLabelAxis(self._model)
        # X axis: time in seconds, right-to-left so the latest sample is at the right.
        self._time_axis = _TimeAxisItem(self._model, orientation="bottom")
        self._plot.setAxisItems({"left": self._channel_axis, "bottom": self._time_axis})
        self._time_axis.setLabel("Time", units="s", color=Palette.text_dim)

        # Persistent per-channel curve objects.
        self._curves: Dict[int, pg.PlotDataItem] = {}

        # Pre-allocated scratch buffers (reused each frame -> no allocations).
        self._x_buf: Optional[np.ndarray] = None
        self._y_buf: Optional[np.ndarray] = None

        # Cached lane geometry; recomputed on resize / channel-set change.
        self._lane_height_px: float = 0.0
        self._lane_offsets: Optional[np.ndarray] = None      # per visible index
        self._gain: float = 1.0
        self._n_visible: int = 0

        # Cached palette (rebuilt on metadata change).
        self._palette: Dict[int, QColor] = {}

        # A semi-transparent band drawn per region for clinical grouping.
        self._region_regions: List[pg.LinearRegionItem] = []

        # Sweep cursor: a thin vertical line spanning the wave area (Y range is
        # [0, n_visible], so it does not cover the bottom time axis). Created in
        # _on_metadata once curves exist; hidden by default and shown only while
        # sweeping (playing). Set to None until the first metadata arrives.
        self._cursor: Optional[pg.InfiniteLine] = None

        # Paint coalescing: signals like position_changed and data_changed
        # can fire multiple times within a single event-loop iteration.  Instead
        # of calling paint_frame() immediately from each handler, we set a
        # flag and defer the actual paint to the next Qt idle tick.  This
        # guarantees at most ONE paint per frame instead of 2-3.
        self._paint_scheduled: bool = False

        # Replace the scene's default render to draw lane labels + gridlines.
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Re-render on model signals. Every view-affecting mutation routes
        # through `_relayout_needed`, which calls `paint_frame()` directly so
        # changes are reflected instantly — even while playback is paused.
        # (During playback the timer-driven `frame_ready` path coexists and
        # is idempotent with these connections.)
        self._model.metadata_changed.connect(self._on_metadata)
        self._model.viewport_changed.connect(self._relayout_needed)
        self._model.visibility_changed.connect(self._relayout_needed)
        self._model.filters_changed.connect(self._relayout_needed)
        self._model.position_changed.connect(self._relayout_needed)
        self._model.data_changed.connect(self._relayout_needed)
        self._model.following_changed.connect(self._on_following_changed)
        # Re-render when sweep mode toggles (cursor must be shown/hidden and the
        # paint branch switches between sweep and legacy scroll).
        self._model.sweep_mode_changed.connect(self._relayout_needed)

    # ------------------------------------------------------------------ #
    # Metadata / palette
    # ------------------------------------------------------------------ #
    def _on_metadata(self) -> None:
        names = self._model.channel_names
        rgba = build_channel_palette(names)
        self._palette = {
            i: QColor.fromRgbF(
                float(rgba[i][0]), float(rgba[i][1]), float(rgba[i][2]), float(rgba[i][3])
            )
            for i in range(len(names))
        }
        # Build persistent curve objects.
        self._clear_curves()
        for i, name in enumerate(names):
            color = self._palette.get(i, QColor("white"))
            pen = QPen(color, 1.0)
            pen.setCosmetic(True)
            # skipFiniteCheck=False so NaN columns (un-swept page area on the
            # very first sweep) render as trace breaks instead of being
            # collapsed. The isfinite scan over ~timebase*fs points is cheap
            # relative to the 30 FPS frame budget.
            curve = self._plot.plot(pen=pen, skipFiniteCheck=False)
            curve.setZValue(10)
            self._curves[i] = curve
        # Populate the left axis with channel names.
        self._update_channel_axis()
        # (Re)create the sweep cursor line on top of the curves. It spans the
        # whole wave area because the Y range is [0, n_visible]; the bottom time
        # axis lives outside that range, so the line never crosses it.
        self._build_cursor()
        self._relayout_needed()

    def _build_cursor(self) -> None:
        """Create (or recreate) the sweep-cursor InfiniteLine, hidden by default."""
        if self._cursor is not None:
            try:
                self._plot.removeItem(self._cursor)
            except Exception:  # noqa: BLE001 - defensive
                pass
            self._cursor = None
        accent = QColor(Palette.accent)
        # QColor has no setF(); rebuild via fromRgbF so we can apply the cursor
        # alpha without changing the accent colour used elsewhere.
        accent = QColor.fromRgbF(
            float(accent.redF()),
            float(accent.greenF()),
            float(accent.blueF()),
            float(SWEEP_CURSOR_ALPHA),
        )
        pen = QPen(accent, float(SWEEP_CURSOR_WIDTH_PX))
        pen.setCosmetic(True)
        self._cursor = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pen)
        self._cursor.setZValue(20)   # above the curves (z=10)
        self._cursor.hide()
        self._plot.addItem(self._cursor, ignoreBounds=True)

    # ------------------------------------------------------------------ #
    # Interaction mode (live/auto-follow vs review)
    # ------------------------------------------------------------------ #
    def _apply_interaction_mode(self) -> None:
        """Enable or disable mouse zoom/pan based on the model's follow mode.

        In live/auto-follow mode (the default) the canvas owns the X range and
        the mouse is disabled. In review mode (paused) the user can wheel-zoom,
        drag-rectangle zoom, and right-drag pan freely.
        """
        reviewing = not self._model.following
        vb = self._plot.vb
        vb.setMouseEnabled(x=reviewing, y=reviewing)
        vb.setMouseMode(pg.ViewBox.RectMode if reviewing else pg.ViewBox.PanMode)

    def _on_following_changed(self, following: bool) -> None:
        # When re-entering live mode, snap the range back to the playhead window.
        self._apply_interaction_mode()
        if following:
            self.paint_frame()

    def zoom_out(self, factor: float = 2.0) -> None:
        """Widen the visible X range by ``factor`` (review mode only).

        No-op in live/auto-follow mode, where ``paint_frame`` owns the range.
        """
        if self._model.following:
            return
        (xmin, xmax), _ = self._plot.vb.viewRange()
        center = 0.5 * (xmin + xmax)
        half = 0.5 * (xmax - xmin) * factor
        self._plot.vb.setXRange(center - half, center + half, padding=0.0)

    def zoom_in(self, factor: float = 1.4) -> None:
        """Narrow the visible X range by ``factor`` (review mode only).

        No-op in live/auto-follow mode. ``factor`` ~ sqrt(2) gives smooth,
        predictable keyboard zoom (area halves every two presses), which is
        the convention used in clinical imaging viewers.
        """
        if self._model.following:
            return
        (xmin, xmax), _ = self._plot.vb.viewRange()
        center = 0.5 * (xmin + xmax)
        half = 0.5 * (xmax - xmin) / factor
        self._plot.vb.setXRange(center - half, center + half, padding=0.0)

    def reset_zoom(self) -> None:
        """Reset the X range to the current data window (review mode only).

        Recomputes the default ``[pos - timebase, pos]`` window around the
        current playhead and applies it once. No-op in live/auto-follow mode.
        """
        if self._model.following:
            return
        pos = self._model.position_seconds
        timebase = self._model.viewport.timebase_s
        self._plot.vb.setXRange(max(0.0, pos - timebase), pos, padding=0.0)

    def _clear_curves(self) -> None:
        for c in self._curves.values():
            self._plot.removeItem(c)
        self._curves.clear()
        # Clear the channel label axis.
        self._channel_axis.setTicks([[]])
        # Drop the cursor too; _on_metadata rebuilds it after the new curves.
        if self._cursor is not None:
            try:
                self._plot.removeItem(self._cursor)
            except Exception:  # noqa: BLE001
                pass
            self._cursor = None

    # ------------------------------------------------------------------ #
    # Lane geometry (computed from live widget size -> scalable)
    # ------------------------------------------------------------------ #
    def _schedule_paint(self) -> None:
        """Defer paint_frame to the next idle tick, coalescing duplicates.

        Multiple signals (position_changed, data_changed, viewport_changed,
        etc.) can fire within a single event-loop iteration.  Instead of
        painting on every one, we schedule a single paint via QTimer(0)
        and ignore subsequent calls until that paint completes.  This
        drops redundant work from ~3 paints/frame down to exactly 1.
        """
        if self._paint_scheduled:
            return
        self._paint_scheduled = True
        QTimer.singleShot(0, self._do_deferred_paint)

    def _do_deferred_paint(self) -> None:
        """Actually run the deferred paint and reset the scheduling flag."""
        self._paint_scheduled = False
        self._compute_lane_geometry()
        self.paint_frame()

    def _relayout_needed(self) -> None:
        """Request a repaint (deferred and coalesced)."""
        self._schedule_paint()

    def _compute_lane_geometry(self) -> None:
        """Distribute visible lanes evenly across the current plot height."""
        visible = self._model.visible_channels()
        self._n_visible = len(visible)
        if self._n_visible == 0:
            return
        names = self._model.channel_names
        # Vertical range spans [0, n_visible]; each lane is one unit tall.
        top = float(self._n_visible)
        self._lane_offsets = np.fromiter(
            (top - 0.5 - i for i in range(self._n_visible)),
            dtype=np.float32,
            count=self._n_visible,
        )
        # Gain: map the lane's data (in µV) so the sensitivity range fills
        # ~70% of one lane's vertical extent.
        sensitivity = self._model.viewport.sensitivity_uv
        self._gain = 0.7 / max(sensitivity, 1e-3)

        # Update Y range and refresh the left-axis channel labels.
        self._plot.setYRange(0.0, top, padding=0.0)
        self._update_channel_axis()

    def _update_channel_axis(self) -> None:
        """Push the current visible-channel list to the left axis."""
        visible = self._model.visible_channels()
        names = self._model.channel_names
        self._channel_axis.update_labels(visible, names, self._n_visible)

    # ------------------------------------------------------------------ #
    # Public render entry point (called by the playback controller each frame)
    # ------------------------------------------------------------------ #
    def waveform_plot(self) -> pg.PlotItem:
        """The single waveform ``PlotItem`` that hosts all lane curves.

        Exposed so sibling widgets (e.g. the DSA panel) can X-link to it and
        read its live view range, keeping their time axis in sync with the
        waveform without owning playback/viewport logic.
        """
        return self._plot

    def paint_frame(self) -> None:
        """Repaint the current window. Called ~30x/sec by the controller.

        Dispatches to the sweep-cursor path (stationary page + moving cursor)
        or the legacy scroll path depending on ``model.sweep_mode``.
        """
        if self._model.n_channels == 0:
            return
        if self._model.sweep_mode:
            self._paint_sweep()
        else:
            self._paint_scroll()

    # ------------------------------------------------------------------ #
    # Sweep-cursor paint path (CRT refresh mode)
    # ------------------------------------------------------------------ #
    def _paint_sweep(self) -> None:
        """Draw a stationary page and advance the sweep cursor over it."""
        playing = self._model.is_playing
        window, info = self._model.render_sweep_frame(advance=playing)
        if window.shape[1] == 0 or self._n_visible != info.n_visible:
            self._compute_lane_geometry()
        if window.shape[1] == 0:
            return

        n_samples = window.shape[1]
        fs = info.sample_rate_hz
        start_t = info.page_start_seconds
        end_t = info.page_end_seconds

        # Page X coordinates (reused buffer). The page is a fixed absolute-time
        # window; it does NOT slide, so waves stay put.
        if self._x_buf is None or self._x_buf.shape[0] != n_samples:
            self._x_buf = np.empty(n_samples, dtype=np.float32)
        if n_samples > 1:
            step = (end_t - start_t) / n_samples
            self._x_buf[:] = start_t + (np.arange(n_samples, dtype=np.float32) * step)
        else:
            self._x_buf[:1] = start_t

        if self._lane_offsets is None or self._lane_offsets.shape[0] != window.shape[0]:
            self._compute_lane_geometry()
            if self._lane_offsets is None:
                return
        offsets = self._lane_offsets[: window.shape[0]]
        y = window * self._gain + offsets[:, None]

        # The page is always fixed while sweeping; force the X range every frame
        # so the user can't pan it off-screen mid-sweep. When paused (review)
        # we leave any user zoom/pan alone — except right after a seek, where
        # seek_pending forces a snap to the reseeded page.
        force_range = playing or self._model.seek_pending
        if force_range:
            self._plot.setXRange(start_t, end_t, padding=0.0)
            self._model.seek_pending = False

        if playing:
            # pyqtgraph caches the axis's rendered ticks in AxisItem.picture and
            # only regenerates them when the view range changes. In sweep mode
            # the X-range is stationary for the whole sweep, so tickStrings()
            # would run once (cursor at the left edge -> all ticks "future" ->
            # all hidden) and that empty result would be cached forever. Invalidate
            # the cache each frame so tick labels reveal progressively as the
            # cursor sweeps past them. (Paused mode needs no invalidation: the
            # cursor is at the page end and a seek already changes the range,
            # which invalidates the cache naturally.)
            self._time_axis.picture = None
            self._time_axis.update()

        visible_set = set(info.visible_indices)
        for lane_i, ch_idx in enumerate(info.visible_indices):
            curve = self._curves.get(ch_idx)
            if curve is None:
                continue
            curve.setData(self._x_buf, y[lane_i], _callSync="none", antialias=False)
        # Clear hidden channels.
        for ch_idx, curve in self._curves.items():
            if ch_idx not in visible_set:
                curve.setData([], [])

        # Cursor: visible only while playing (paused -> hidden so the page can
        # be reviewed cleanly). It sits at the swept-data edge in absolute time.
        if self._cursor is not None:
            if playing:
                self._cursor.setPos(info.cursor_seconds)
                self._cursor.show()
            else:
                self._cursor.hide()

    # ------------------------------------------------------------------ #
    # Legacy scroll paint path (waves slide left; playhead at right edge)
    # ------------------------------------------------------------------ #
    def _paint_scroll(self) -> None:
        """Repaint the current window. Called ~60x/sec by the controller."""
        if self._model.n_channels == 0:
            return
        window, info = self._model.render_window()
        if window.shape[1] == 0 or self._n_visible != info.n_visible:
            self._compute_lane_geometry()
        if window.shape[1] == 0:
            return

        n_samples = window.shape[1]
        timebase = info.timebase_s
        fs = info.sample_rate_hz
        # Use data_end_seconds so the X-range end matches the actual data
        # extent (may be less than the playhead when the buffer is shorter
        # than the full timebase, e.g. after seeking near the start).
        end_t = info.data_end_seconds
        actual_duration = (n_samples / fs) if n_samples > 1 else timebase
        start_t = max(0.0, end_t - actual_duration)

        # Time axis buffer (reused). numpy's linspace has no `out=` arg in
        # modern releases, so we compute into our own buffer via arange + scale.
        if self._x_buf is None or self._x_buf.shape[0] != n_samples:
            self._x_buf = np.empty(n_samples, dtype=np.float32)
        if n_samples > 1:
            step = (end_t - start_t) / n_samples
            self._x_buf[:] = start_t + (np.arange(n_samples, dtype=np.float32) * step)
        else:
            self._x_buf[:1] = start_t

        # Offsets per *visible* lane; map visible index -> offset.
        if self._lane_offsets is None or self._lane_offsets.shape[0] != window.shape[0]:
            self._compute_lane_geometry()
            if self._lane_offsets is None:
                return

        offsets = self._lane_offsets[: window.shape[0]]
        gain = self._gain

        # Vectorised: y[lane, t] = data[lane, t] * gain + offset[lane]
        # Doing it in one expression avoids Python-level loops.
        y = window * gain + offsets[:, None]

        # Set the X range BEFORE pushing data into the curves so that
        # pyqtgraph's autoDownsample and clip-to-view use the correct
        # coordinate system on the first pass (avoiding a redundant
        # re-computation that setXRange's synchronous signal would otherwise
        # trigger). In review mode (paused) we leave any user zoom/pan
        # alone so it persists across repaints, except right after a seek
        # where seek_pending forces a snap.
        if self._model.following or self._model.seek_pending:
            self._plot.setXRange(start_t, end_t, padding=0.0)
            self._model.seek_pending = False

        visible_set = set(info.visible_indices)
        for lane_i, ch_idx in enumerate(info.visible_indices):
            curve = self._curves.get(ch_idx)
            if curve is None:
                continue
            curve.setData(self._x_buf, y[lane_i], _callSync="none", antialias=False)

        # Clear data for hidden channels so they disappear from the scene.
        for ch_idx, curve in self._curves.items():
            if ch_idx not in visible_set:
                curve.setData([], [])

        # In scroll mode the cursor is never used — keep it hidden.
        if self._cursor is not None and self._cursor.isVisible():
            self._cursor.hide()

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def _on_mouse_click(self, event) -> None:
        """Mouse interaction on the canvas.

        - Right-click: zoom out one step (review mode only).
        - Left-click: seek the playhead to that time (file sources only).
        """
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return

        # Right-click: zoom out one step. No-op in live/auto-follow mode.
        if event.button() == Qt.MouseButton.RightButton:
            if not self._model.following:
                self.zoom_out(factor=2.0)
            event.accept()
            return

        # Left-click: seek (file sources only).
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._model.is_live or self._model.total_samples <= 0:
            return
        vb = self._plot.vb
        data_pos = vb.mapSceneToView(pos)
        self.seek_requested.emit(float(data_pos.x()))


class _TimeAxisItem(pg.AxisItem):
    """Bottom axis showing absolute time in seconds (never negative).

    In sweep-cursor mode a tick label is revealed only once the sweep cursor has
    swept PAST that tick's column (during playback) — so the time axis updates
    value-by-value as the cursor advances, rather than labelling the whole page
    up front. After a page wrap the cursor resets to column 0, so the labels
    clear and re-reveal progressively. When paused (review) the cursor is at the
    page end and all labels show.
    """

    def __init__(self, model, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self._model = model

    def tickSpacing(self, minVal, maxVal, size):
        # pyqtgraph generates up to 3 tick levels (major / minor / sub-minor)
        # and calls tickStrings() once per level. In sweep mode the sub-minor
        # level (e.g. 70.1, 70.2, 70.3 on a 10s page) would all fall at or
        # before the cursor early in a sweep, producing a cramped stack of
        # decimal labels that only clears once pyqtgraph's density filter
        # suppresses them. Keep ONLY the major level in sweep mode so the
        # progressive reveal in tickStrings() applies to clean major ticks.
        # Legacy scroll mode keeps the full hierarchy (delegates to base).
        levels = super().tickSpacing(minVal, maxVal, size)
        if self._model.sweep_mode and levels:
            return levels[:1]
        return levels

    def tickStrings(self, values, scale, spacing):
        try:
            values = [float(v) for v in values]
        except Exception:  # noqa: BLE001
            return super().tickStrings(values, scale, spacing)

        # Legacy scroll mode: label everything (unchanged behaviour).
        if not self._model.sweep_mode:
            return [f"{max(0.0, v):.2f}s" for v in values]

        # Sweep mode: progressive reveal with CRT persistence across page wraps.
        #
        # A tick at absolute time `v` corresponds to page column `col`. The trace
        # does CRT persistence: after the first sweep, columns ahead of the
        # cursor keep showing stale data from the PREVIOUS sweep until the cursor
        # overwrites them. The labels mirror that — a column's label shows the
        # time of the data actually displayed there:
        #   * reached this sweep (col <= cursor_col)       -> current page time
        #   * not reached but has stale data (mask True)   -> previous page time
        #   * not reached, never drawn (first sweep)        -> hidden ("")
        # This keeps the old page's labels visible through a wrap (no wipe) and
        # updates each label to the current time only when the cursor reaches it
        # (per-value reveal preserved).
        fs = self._model.sample_rate_hz
        cursor = self._model.sweep_cursor_sample()
        mask = self._model.sweep_valid_mask()               # bool per page column
        page_start = self._model._sweep_page_start_samples  # abs sample at col 0
        page_n = self._model._sweep_page_samples
        cursor_col = cursor - page_start                    # cursor's column this sweep
        out = []
        for v in values:
            v_clamped = max(0.0, v)
            tick_sample = v_clamped * fs if fs > 0 else 0.0
            col = int(round(tick_sample)) - page_start
            if fs > 0 and 0 <= col < page_n and col <= cursor_col + 0.5:
                # Cursor has reached this column this sweep -> current page time.
                out.append(f"{v_clamped:.2f}s")
            elif mask is not None and 0 <= col < len(mask) and bool(mask[col]):
                # Stale data from the previous sweep is shown here -> label it
                # with that data's (previous-page) time so old values persist
                # across the wrap instead of blanking.
                prev_time = max(0.0, (page_start - page_n + col) / fs) if fs > 0 else 0.0
                out.append(f"{prev_time:.2f}s")
            else:
                out.append("")   # never drawn (first sweep) -> hide
        return out
