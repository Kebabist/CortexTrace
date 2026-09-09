"""The application model.

A single :class:`EEGModel` instance is the source of truth: it owns the ring
buffer, channel metadata, viewing parameters (timebase, sensitivity, filters),
playback state (position, speed, loop) and the channel-visibility map. All
mutations go through small setters that emit fine-grained Qt signals, so the
view layer only needs to listen - it never reaches into internals.

Design goals:

* Real-time-first: the buffer is fed by a worker-thread producer via
  :py:meth:`ingest_chunk`; rendering reads zero-copy views.
* Hardware-agnostic: file replay and live sources feed the same path.
* Allocation-free hot path: views never allocate during a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import QObject, Qt, Signal

from ..config import (
    DEFAULT_HF_HZ,
    DEFAULT_LF_HZ,
    DEFAULT_NOTCH_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SENSITIVITY_UV,
    DEFAULT_SPEED,
    DEFAULT_TIMEBASE_S,
    DSA_DEFAULT,
    DSA_MODE_DEFAULT,
    RING_BUFFER_SECONDS,
    SWEEP_MODE_DEFAULT,
)
from .filters import FilterChain, FilterSettings
from .montage import Montage, apply_montage, identity_montage


@dataclass
class ViewportState:
    """Snapshot of view-only parameters consumed by the canvas."""
    timebase_s: float = DEFAULT_TIMEBASE_S
    sensitivity_uv: float = DEFAULT_SENSITIVITY_UV
    speed: float = DEFAULT_SPEED
    loop: bool = True
    following: bool = True        # True = live/auto-follow; False = review (zoom/pan)
    dsa_enabled: bool = DSA_DEFAULT     # show the DSA trend panel beneath the waveform
    dsa_mode: str = DSA_MODE_DEFAULT    # DSA track layout ("hemisphere" | "regional")


class EEGModel(QObject):
    """Owns acquisition data, viewing parameters and playback state."""

    # ---- lifecycle ------------------------------------------------------- #
    metadata_changed = Signal()              # channel set / sample rate changed
    data_changed = Signal()                  # new samples ingested (buffer grew)
    source_eof = Signal()                    # producer reached end of file

    # ---- playback -------------------------------------------------------- #
    position_changed = Signal(float)         # seconds (cursor)
    playing_changed = Signal(bool)
    speed_changed = Signal(float)
    loop_changed = Signal(bool)
    following_changed = Signal(bool)         # live/auto-follow vs review mode

    # ---- view ------------------------------------------------------------ #
    viewport_changed = Signal()              # timebase/sensitivity/filters
    visibility_changed = Signal()            # channel on/off map
    filters_changed = Signal()
    montage_changed = Signal()

    # ---- sweep cursor ----------------------------------------------------- #
    # Emitted when sweep mode is toggled so the canvas can rebuild its paint
    # branch and (re)create the InfiniteLine cursor.
    sweep_mode_changed = Signal(bool)

    # ---- DSA trend panel ------------------------------------------------- #
    # Emitted when the DSA panel is shown/hidden or its track layout changes.
    dsa_enabled_changed = Signal(bool)
    dsa_mode_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Acquisition metadata (filled on first load).
        self._channel_names: List[str] = []
        self._fs: float = DEFAULT_SAMPLE_RATE_HZ
        self._total_samples: int = -1         # -1 = unbounded (live)

        # Ring buffer (allocated on load).
        self._buffer = None                   # type: Optional[import('ring_buffer').RingBuffer]

        # Viewport / playback.
        self._view = ViewportState()
        self._position_samples: float = 0.0   # playhead (in samples, float for smoothness)
        self._playing: bool = False
        # One-shot flag: set by seek_seconds(), consumed (cleared) by the next
        # paint_frame() so it knows to force-setXRange even in review mode.
        self.seek_pending: bool = False

        # Channel visibility map (channel index -> bool). Default: all visible.
        self._visible: Dict[int, bool] = {}

        # Filters + montage.
        self._filter_chain = FilterChain(self._fs)
        self._filter_chain.reconfigure(FilterSettings())
        self._montage: Optional[Montage] = None

        # Pre-allocated scratch for the render read.
        self._window_scratch: Optional[np.ndarray] = None

        # ---- Sweep-cursor (CRT refresh) state ------------------------------ #
        # The sweep page is a fixed-length raw buffer (one timebase window) that
        # accumulates incoming data CRT-style: each frame the newly acquired
        # samples are written at the cursor position, and samples to the right of
        # the cursor retain the previous sweep's trace (overwrite, never blank).
        # ``_sweep_page_valid`` marks which page columns hold real data vs. the
        # initial NaN fill (gaps render as breaks on the very first sweep).
        self._sweep_mode: bool = SWEEP_MODE_DEFAULT
        self._sweep_page_raw: Optional[np.ndarray] = None       # (n_ch, page_n)
        self._sweep_page_valid: Optional[np.ndarray] = None     # (page_n,) bool
        self._sweep_page_samples: int = 0                       # columns in the page
        # Absolute sample index of the left edge of the current page. The page
        # always spans absolute [page_start, page_start + page_samples).
        self._sweep_page_start_samples: int = 0
        # Absolute end_index (ring buffer write counter) consumed by the last
        # sweep advance. The delta vs. the buffer's current end_index is the
        # number of new samples to push into the page this frame.
        self._sweep_last_end_index: int = 0
        # Absolute sample index of the sweep cursor's leading edge (the rightmost
        # column written THIS sweep). Used by the time axis to reveal tick labels
        # progressively — a label shows only once the cursor has swept past it.
        # Set to page_end (page_start + page_n) in review mode so all labels show.
        self._sweep_cursor_sample: int = 0

    # ------------------------------------------------------------------ #
    # Metadata / buffer lifecycle
    # ------------------------------------------------------------------ #
    def configure_acquisition(
        self,
        channel_names: List[str],
        sample_rate_hz: float,
        total_samples: int = -1,
    ) -> None:
        """(Re)initialise the buffer for a new acquisition session."""
        self._channel_names = list(channel_names)
        self._fs = float(sample_rate_hz)
        self._total_samples = int(total_samples)
        capacity = max(
            int(self._fs * RING_BUFFER_SECONDS),
            int(self._fs * (self._view.timebase_s + 1.0)),
        )
        from ..backend.ring_buffer import RingBuffer
        self._buffer = RingBuffer(len(self._channel_names), capacity)
        self._window_scratch = None

        # Reset filters to the new sample rate.
        self._filter_chain = FilterChain(self._fs)
        self._filter_chain.reconfigure(
            FilterSettings(
                hf_hz=self._filter_chain.settings.hf_hz,
                lf_hz=self._filter_chain.settings.lf_hz,
                notch_hz=self._filter_chain.settings.notch_hz,
            )
        )
        self._montage = identity_montage(self._channel_names)

        # Default visibility: all channels on; persisted selection is applied
        # by the controller via :py:meth:`set_visible_channels` after load.
        self._visible = {i: True for i in range(len(self._channel_names))}

        self._position_samples = 0.0
        self._playing = False
        self.metadata_changed.emit()
        self.visibility_changed.emit()
        self.viewport_changed.emit()
        self.montage_changed.emit()

        # (Re)build the sweep page for the new acquisition / timebase.
        self._alloc_sweep_page()

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    @property
    def n_channels(self) -> int:
        return len(self._channel_names)

    @property
    def sample_rate_hz(self) -> float:
        return self._fs

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def duration_seconds(self) -> float:
        if self._total_samples < 0:
            return float("inf")
        return self._total_samples / self._fs if self._fs > 0 else 0.0

    @property
    def is_live(self) -> bool:
        return self._total_samples < 0

    @property
    def buffer(self):
        return self._buffer

    # ------------------------------------------------------------------ #
    # Producer ingestion (called from queued signal connection)
    # ------------------------------------------------------------------ #
    def ingest_chunk(self, chunk) -> None:
        """Append a :class:`~eeg_viewer.backend.data_source.SampleChunk`."""
        if self._buffer is None:
            return
        self._buffer.write(chunk.data)
        # Advance the live playhead to the most recent sample when playing
        # realtime; the controller fine-tunes the exact pacing.
        if self._playing and self.is_live:
            self._position_samples = float(self._buffer.filled)
            self.position_changed.emit(self.position_seconds)
        self.data_changed.emit()

    def notify_eof(self) -> None:
        self.source_eof.emit()

    # ------------------------------------------------------------------ #
    # Viewport
    # ------------------------------------------------------------------ #
    @property
    def viewport(self) -> ViewportState:
        return self._view

    def set_timebase(self, seconds: float) -> None:
        seconds = max(0.5, float(seconds))
        if abs(seconds - self._view.timebase_s) < 1e-6:
            return
        self._view.timebase_s = seconds
        # Grow the ring buffer if the new timebase exceeds capacity.
        if self._buffer is not None:
            needed = int(self._fs * (seconds + 1.0))
            if self._buffer.capacity_samples < needed:
                from ..backend.ring_buffer import RingBuffer
                new_cap = max(needed, int(self._fs * RING_BUFFER_SECONDS))
                old = self._buffer
                # Preserve existing data AND its absolute position across the
                # reallocation, so the canvas neither blanks nor jumps back to
                # t≈0 when the timebase grows (e.g. while paused). end_index is
                # the single source of truth for the timeline right edge.
                old_end = old.end_index
                old_view, _ = old.latest_window(old.filled)
                self._buffer = RingBuffer(self.n_channels, new_cap)
                if old_view.shape[1] > 0:
                    self._buffer.write(old_view)
                self._buffer.set_end_index(old_end)
        self.seek_pending = True  # force X-range snap even in review (paused) mode
        # The sweep page width follows the timebase. Reallocate the page at the
        # new size (no-op if unchanged), then reseed from the buffer so the new
        # page is immediately populated and reviewable — no blank frame while
        # playing or paused. (reset_sweep alone doesn't recompute page_samples,
        # so we must call _alloc_sweep_page first when the timebase changes.)
        self._alloc_sweep_page()
        self.reset_sweep()
        self.viewport_changed.emit()

    def set_sensitivity(self, microvolts: float) -> None:
        microvolts = max(1.0, float(microvolts))
        if abs(microvolts - self._view.sensitivity_uv) < 1e-6:
            return
        self._view.sensitivity_uv = microvolts
        self.viewport_changed.emit()

    def set_speed(self, multiplier: float) -> None:
        multiplier = max(0.05, float(multiplier))
        if abs(multiplier - self._view.speed) < 1e-6:
            return
        self._view.speed = multiplier
        self.speed_changed.emit(multiplier)

    def set_loop(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._view.loop:
            return
        self._view.loop = enabled
        self.loop_changed.emit(enabled)

    @property
    def following(self) -> bool:
        return self._view.following

    def set_following(self, following: bool) -> None:
        """Toggle live/auto-follow mode.

        When True the playhead tracks incoming data and the canvas re-applies
        its X range every frame. When False (review mode) the playhead freezes,
        the range override stops, and mouse zoom/pan become active.
        """
        following = bool(following)
        if following == self._view.following:
            return
        self._view.following = following
        self.following_changed.emit(following)

    def set_dsa_enabled(self, enabled: bool) -> None:
        """Show or hide the DSA trend panel beneath the waveform."""
        enabled = bool(enabled)
        if enabled == self._view.dsa_enabled:
            return
        self._view.dsa_enabled = enabled
        self.dsa_enabled_changed.emit(enabled)

    def set_dsa_mode(self, mode: str) -> None:
        """Switch the DSA track layout ("hemisphere" or "regional")."""
        if mode == self._view.dsa_mode:
            return
        self._view.dsa_mode = mode
        self.dsa_mode_changed.emit(mode)

    # ------------------------------------------------------------------ #
    # Filters / montage
    # ------------------------------------------------------------------ #
    def set_filters(self, settings: FilterSettings) -> None:
        self._filter_chain.reconfigure(settings)
        self.filters_changed.emit()

    @property
    def filter_chain(self) -> FilterChain:
        return self._filter_chain

    def set_montage(self, montage: Montage) -> None:
        self._montage = montage
        self.montage_changed.emit()

    @property
    def montage(self) -> Optional[Montage]:
        return self._montage

    # ------------------------------------------------------------------ #
    # Channel visibility
    # ------------------------------------------------------------------ #
    def visible_channels(self) -> List[int]:
        return [i for i in range(self.n_channels) if self._visible.get(i, True)]

    def is_visible(self, channel_index: int) -> bool:
        return self._visible.get(channel_index, True)

    def set_channel_visible(self, channel_index: int, visible: bool) -> None:
        if self._visible.get(channel_index) == visible:
            return
        self._visible[channel_index] = visible
        self.visibility_changed.emit()

    def set_visible_channels(self, indices: List[int]) -> None:
        new_map = {i: (i in indices) for i in range(self.n_channels)}
        self._visible = new_map
        self.visibility_changed.emit()

    # ------------------------------------------------------------------ #
    # Playback state
    # ------------------------------------------------------------------ #
    @property
    def is_playing(self) -> bool:
        return self._playing

    def set_playing(self, playing: bool) -> None:
        if playing == self._playing:
            return
        self._playing = playing
        self.playing_changed.emit(playing)

    @property
    def position_samples(self) -> float:
        return self._position_samples

    @position_samples.setter
    def position_samples(self, value: float) -> None:
        clamped = self._clamp_position(value)
        if abs(clamped - self._position_samples) < 1e-6:
            return
        self._position_samples = clamped
        self.position_changed.emit(self.position_seconds)

    @property
    def position_seconds(self) -> float:
        return self._position_samples / self._fs if self._fs > 0 else 0.0

    def set_position_seconds(self, seconds: float) -> None:
        self.position_samples = seconds * self._fs

    def snap_to_tail(self) -> None:
        """Jump the playhead to the most recent available data.

        For live sources this is the buffer tail. For file sources this is a
        no-op (file playback continues from the paused position). Emits
        position_changed so the canvas re-paints at the new location.
        """
        if self.is_live and self._buffer is not None and self._buffer.filled > 0:
            self.position_samples = float(self._buffer.filled)

    def _clamp_position(self, samples: float) -> float:
        if self._total_samples < 0:
            return max(0.0, float(samples))
        return max(0.0, min(float(samples), float(self._total_samples)))

    # ------------------------------------------------------------------ #
    # Sweep-cursor (CRT refresh) mode
    # ------------------------------------------------------------------ #
    @property
    def sweep_mode(self) -> bool:
        return self._sweep_mode

    def set_sweep_mode(self, enabled: bool) -> None:
        """Toggle between CRT sweep-cursor mode and the legacy scroll mode."""
        enabled = bool(enabled)
        if enabled == self._sweep_mode:
            return
        self._sweep_mode = enabled
        # Reseed the page so the first sweep-mode frame starts clean, regardless
        # of whatever the scroll mode was showing.
        self._alloc_sweep_page()
        self.sweep_mode_changed.emit(enabled)

    def _alloc_sweep_page(self) -> None:
        """(Re)allocate the sweep page buffer for the current timebase.

        The page is one timebase-window wide and holds raw (pre-filter) samples.
        Unfilled columns are NaN so the first sweep renders gaps rather than a
        flat zero line. Page position bookkeeping is reset to the current
        playhead so the cursor lands inside a valid page after a realloc.
        """
        if self._buffer is None or self.n_channels == 0 or self._fs <= 0:
            self._sweep_page_raw = None
            self._sweep_page_valid = None
            self._sweep_page_samples = 0
            self._sweep_page_start_samples = 0
            self._sweep_last_end_index = 0
            return

        page_n = max(1, int(round(self._fs * self._view.timebase_s)))
        if (
            self._sweep_page_raw is not None
            and self._sweep_page_raw.shape == (self.n_channels, page_n)
        ):
            # Same size: just clear contents in place (no realloc).
            self._sweep_page_raw.fill(np.nan)
            if self._sweep_page_valid is not None:
                self._sweep_page_valid.fill(False)
        else:
            self._sweep_page_raw = np.full(
                (self.n_channels, page_n), np.nan, dtype=np.float32
            )
            self._sweep_page_valid = np.zeros(page_n, dtype=bool)
        self._sweep_page_samples = page_n

        # Align the page so the playhead sits at the RIGHT edge with a full
        # timebase window behind it: page_start = max(0, pos - page_n). This
        # matches the review-mode convention (playhead at right edge, full page
        # behind it) and ensures _seed_page_from_buffer fills the whole page.
        pos = int(self._position_samples)
        self._sweep_page_start_samples = max(0, pos - page_n)
        # In review the cursor is conceptually at the page end so all time-axis
        # labels reveal immediately.
        self._sweep_cursor_sample = pos
        # Catch the sweep up to whatever the buffer has already ingested so we
        # don't replay buffered history into the new page on the first frame.
        self._sweep_last_end_index = self._buffer.end_index

    def reset_sweep(self, left_align: bool = False) -> None:
        """Reseed the sweep page at the current playhead (used after seek/loop).

        Clears the page contents, recentres ``page_start`` on the current
        position, and seeds the page from whatever the ring buffer currently
        holds so review mode shows a full, immediately-reviewable waveform
        (fixes the "blank page after seek while paused" bug). Does *not* move
        the playhead itself.

        ``left_align``:
          * ``False`` (default) — ``page_start`` snaps to the page boundary at
            or below the playhead (page-aligned). Used for loop-reset /
            mode-toggle where we want whole-page CRT persistence.
          * ``True`` — ``page_start`` is set exactly to the playhead so the
            cursor resumes writing at the LEFT edge of the page (column 0).
            Used by seek / step buttons while *playing* — the user expects the
            cursor to restart at the left edge over the seeded trace.
        """
        if self._sweep_page_raw is None:
            self._alloc_sweep_page()
            if self._sweep_page_raw is None:
                # No acquisition configured yet (no buffer / no channels / no
                # fs): nothing to reset. Defensive against UI actions (e.g.
                # changing the timebase combo) fired before a file is loaded.
                return
        page_n = self._sweep_page_samples
        pos = max(0, int(self._position_samples))
        # left_align=True  -> page_start = pos (playhead at LEFT edge; cursor
        #                     restarts sweeping from column 0 while playing).
        # left_align=False -> page_start = max(0, pos - page_n) (playhead at the
        #                     RIGHT edge with a full timebase window behind it —
        #                     the review convention, so the seeded page is full).
        if left_align:
            self._sweep_page_start_samples = pos
            self._sweep_cursor_sample = pos           # cursor at left edge
        else:
            self._sweep_page_start_samples = max(0, pos - page_n)
            self._sweep_cursor_sample = pos           # review: cursor at playhead (right edge)
        # Clear, then seed from the buffer so review shows the full waveform at
        # the seek/step target immediately (no waiting for the cursor to sweep).
        self._sweep_page_raw.fill(np.nan)
        if self._sweep_page_valid is not None:
            self._sweep_page_valid.fill(False)
        self._seed_page_from_buffer()
        if self._buffer is not None:
            self._sweep_last_end_index = self._buffer.end_index

    def _seed_page_from_buffer(self) -> None:
        """Copy the ring-buffer tail into the sweep page and mark it valid.

        After a seek/step the ring buffer holds one timebase window ending at
        the playhead (file sources via ``fill_window_into``; live sources via
        ongoing ingestion). We mirror that window into the page so a paused
        review frame is immediately reviewable instead of blank. When
        ``page_start`` equals the playhead (left-aligned seek), only the columns
        up to the playhead are seeded so the cursor still writes fresh data from
        column 0 onward while playing.
        """
        if (
            self._sweep_page_raw is None
            or self._sweep_page_valid is None
            or self._buffer is None
            or self._sweep_page_samples <= 0
        ):
            return
        page_n = self._sweep_page_samples
        page_start = self._sweep_page_start_samples
        end_index = self._buffer.end_index
        # The page covers absolute [page_start, page_end). Only the part of the
        # buffer that falls inside this window should be seeded.
        page_end = page_start + page_n
        data_right = min(end_index, page_end)
        if data_right <= page_start:
            return  # buffer has no data inside this page
        # We want the samples with absolute indices in [page_start, data_right).
        # latest_window(n) returns the most recent n samples ending at end_index,
        # so to land the right edge at data_right we must:
        #   - skip (end_index - data_right) trailing samples that are PAST the page
        #   - take (data_right - page_start) samples
        skip = max(0, end_index - data_right)
        take = data_right - page_start
        if take <= 0:
            return
        # Pull enough trailing samples to include the [page_start, data_right)
        # range, then slice off the skip past-the-page tail.
        pull = min(take + skip, end_index)
        chunk, got = self._buffer.latest_window(pull)
        if got <= 0 or chunk.shape[1] == 0:
            return
        # Drop up to `skip` samples from the right (they're past data_right) and
        # up to any shortfall on the left (when the buffer didn't have enough).
        have_after_skip = max(0, got - skip)
        usable = min(have_after_skip, take)
        if usable <= 0:
            return
        # The usable slice's right edge is at absolute data_right; its left edge
        # is at data_right - usable. Map onto page columns.
        chunk_left_abs = data_right - usable
        page_col_lo = max(0, chunk_left_abs - page_start)
        page_col_hi = min(page_n, data_right - page_start)
        n_cols = page_col_hi - page_col_lo
        if n_cols <= 0:
            return
        self._sweep_page_raw[:, page_col_lo:page_col_hi] = chunk[:, -usable:][:, -n_cols:] if usable != n_cols else chunk[:, -n_cols:]
        self._sweep_page_valid[page_col_lo:page_col_hi] = True

    def sweep_valid_mask(self) -> Optional[np.ndarray]:
        """Return the page validity mask (bool per column) or None.

        Used by the time-axis renderer to suppress labels for columns the sweep
        cursor has not yet reached (point-by-point axis updates).
        """
        if not self._sweep_mode:
            return None
        return self._sweep_page_valid

    def sweep_cursor_sample(self) -> int:
        """Absolute sample index of the sweep cursor's leading edge.

        The time axis reveals a tick label only for columns at or before this
        sample, so labels appear progressively as the cursor sweeps left-to-right
        (during playback) and all at once when paused (review).
        """
        return self._sweep_cursor_sample

    def sweep_page_time_map(self) -> Optional[np.ndarray]:
        """Return absolute time (seconds) per page column, or None.

        Column ``i`` corresponds to absolute sample ``page_start_samples + i``,
        i.e. time ``(page_start_samples + i) / fs``. Used by the time-axis
        renderer to map tick values to page columns.
        """
        if not self._sweep_mode or self._sweep_page_samples <= 0 or self._fs <= 0:
            return None
        page_n = self._sweep_page_samples
        base = self._sweep_page_start_samples
        return (base + np.arange(page_n, dtype=np.float64)) / float(self._fs)

    def reset_sweep_cursor_to_position(self) -> None:
        """Snap the sweep cursor to the current playhead (used on live resume).

        Unlike :meth:`reset_sweep`, this keeps the existing page trace so the
        resume only repositions the cursor over the current page; it does not
        wipe what was already swept. The page is re-aligned to the playhead in
        case time elapsed while paused (live sources keep ingesting).
        """
        if self._sweep_page_raw is None:
            self._alloc_sweep_page()
            return
        page_n = self._sweep_page_samples
        pos = int(self._position_samples)
        new_start = pos - (pos % page_n)
        if new_start != self._sweep_page_start_samples:
            # We've moved to a different page — clear it so the new sweep starts
            # clean over the advanced time window.
            self._sweep_page_start_samples = new_start
            self._sweep_page_raw.fill(np.nan)
            if self._sweep_page_valid is not None:
                self._sweep_page_valid.fill(False)
        if self._buffer is not None:
            self._sweep_last_end_index = self._buffer.end_index

    def render_sweep_frame(self, advance: bool) -> tuple:
        """Return ``(visible_window, info)`` for one sweep-cursor frame.

        When ``advance`` is True (playing), newly ingested samples are pushed
        into the page at the cursor position and the cursor advances. When
        False (paused / review), the page is frozen as-is — the swept trace is
        kept and the cursor position is reported but no new data is written.

        The returned ``visible_window`` is float32 ``(n_visible, page_samples)``
        already filtered + montaged, with unfilled (pre-first-sweep) columns
        set to NaN so they render as gaps. ``info`` is a
        :class:`_SweepRenderInfo` carrying the page time bounds, cursor time
        and visible-channel indices.
        """
        empty = (np.zeros((0, 0), dtype=np.float32), _SweepRenderInfo())
        if (
            self._buffer is None
            or self.n_channels == 0
            or self._sweep_page_raw is None
        ):
            return empty

        page_n = self._sweep_page_samples
        page_start = self._sweep_page_start_samples
        end_index = self._buffer.end_index

        if advance:
            # Number of new samples accumulated since the last sweep advance.
            delta = max(0, end_index - self._sweep_last_end_index)
            # Cursor position (absolute) tracks the buffer's right edge for both
            # live and file sources — this is the single source of truth, the
            # same convention as render_window(). For file sources we also push
            # it into _position_samples so the status bar / scrubber stay in
            # lockstep with the swept waveform.
            cursor_abs = min(end_index, page_start + page_n)
            if delta > 0 and self._buffer is not None:
                # The new samples are the most recent `delta`; push them into the
                # page over the range [prev_cursor, new_cursor], clamped to the
                # current page. Anything past the page edge is dropped here and
                # picked up after a wrap (handled below).
                prev_cursor = max(page_start, self._sweep_last_end_index)
                prev_in_page = max(0, min(page_n, prev_cursor - page_start))
                new_in_page = max(0, min(page_n, cursor_abs - page_start))
                if new_in_page > prev_in_page:
                    take = new_in_page - prev_in_page
                    chunk, _ = self._buffer.latest_window(take)
                    if chunk.shape[1] >= take:
                        self._sweep_page_raw[:, prev_in_page:new_in_page] = (
                            chunk[:, -take:]
                        )
                        if self._sweep_page_valid is not None:
                            self._sweep_page_valid[prev_in_page:new_in_page] = True
            self._sweep_last_end_index = end_index

            # Wrap: once the cursor has passed the right edge of the page,
            # advance page_start by a whole timebase and keep the page buffer
            # contents (CRT persistence — the new sweep overwrites the old trace
            # column by column). Any samples beyond the page are absorbed into
            # the new page on subsequent frames.
            while end_index - page_start >= page_n:
                page_start += page_n
            self._sweep_page_start_samples = page_start
            cursor_abs = min(end_index, page_start + page_n)
            if not self.is_live and end_index > 0:
                # File: keep the playhead locked to the swept data edge so the
                # status-bar clock / scrubber match the waveform, exactly like
                # render_window() does for scroll mode.
                self.position_samples = float(cursor_abs)
            # Track the cursor's leading edge (absolute sample). The time axis
            # uses this to reveal tick labels progressively as the cursor sweeps.
            self._sweep_cursor_sample = cursor_abs
            cursor_seconds = cursor_abs / self._fs if self._fs > 0 else 0.0
        else:
            # Paused (review): cursor conceptually at the playhead (right edge of
            # the seeded page) so all time-axis labels reveal immediately.
            self._sweep_cursor_sample = max(0, int(self._position_samples))
            cursor_seconds = self.position_seconds

        # Build the renderable page: NaN where never swept, value otherwise.
        # Filtering NaNs is fine — scipy propagates them and pyqtgraph breaks
        # the trace at non-finite points (skipFiniteCheck=False), which is the
        # intended "blank right of cursor on first sweep" look.
        raw = self._sweep_page_raw
        if self._sweep_page_valid is not None and not self._sweep_page_valid.all():
            # Mask out columns that have never been written (initial NaN fill is
            # already NaN, but be defensive against any partial in-place writes).
            raw = np.where(self._sweep_page_valid[np.newaxis, :], raw, np.nan)

        # Filter + montage on the whole page (same pipeline as render_window).
        filtered = self._filter_chain.process(raw)
        montage_view = apply_montage(filtered, self._montage)

        # Select visible channels.
        vis_idx = self.visible_channels()
        if vis_idx and montage_view.shape[0] == self.n_channels:
            montage_view = np.take(montage_view, vis_idx, axis=0)
        elif montage_view.shape[0] != self.n_channels:
            vis_idx = list(range(montage_view.shape[0]))
        else:
            montage_view = np.zeros(
                (0, montage_view.shape[1]), dtype=montage_view.dtype
            )

        info = _SweepRenderInfo(
            n_visible=montage_view.shape[0],
            n_samples=montage_view.shape[1],
            sample_rate_hz=self._fs,
            timebase_s=self._view.timebase_s,
            sensitivity_uv=self._view.sensitivity_uv,
            cursor_seconds=cursor_seconds,
            page_start_seconds=(page_start / self._fs) if self._fs > 0 else 0.0,
            page_end_seconds=(
                (page_start + page_n) / self._fs if self._fs > 0 else 0.0
            ),
            visible_indices=vis_idx,
        )
        return montage_view, info

    # ------------------------------------------------------------------ #
    # Render read - zero-copy slice of the visible window
    # ------------------------------------------------------------------ #
    def render_window(self) -> tuple:
        """Return ``(visible_window, info)`` for the current frame.

        ``visible_window`` is a float32 array of shape ``(n_visible, n_samples)``
        already filtered and montaged. ``info`` carries lane / timing metadata
        needed by the canvas. This function is the *only* place the view layer
        pulls data from, keeping the hot path predictable.
        """
        if self._buffer is None or self.n_channels == 0:
            empty = np.zeros((0, 0), dtype=np.float32)
            return empty, _RenderInfo()

        timebase_samples = int(self._fs * self._view.timebase_s)
        # Both branches read the buffer tail: the visible window is always the
        # most recent ``timebase_samples``. The single source of truth for the
        # right-edge time is the buffer's ``end_index`` (absolute sample count
        # ever written), NOT an independently-tracked playhead — so the
        # waveform data and the timeline labels can never drift apart.
        view, _ = self._buffer.latest_window(timebase_samples)
        # Absolute sample index just past the right edge of the returned window.
        data_end_samples = float(self._buffer.end_index)

        if view.shape[1] == 0:
            empty = np.zeros((self.n_channels, 0), dtype=np.float32)
            return empty, _RenderInfo()

        # Filter + montage (montage may change channel count/order).
        filtered = self._filter_chain.process(view)
        montage_view = apply_montage(filtered, self._montage)

        # Single source of truth: derive the right-edge time of the displayed
        # data from the actual buffer content (end_index / fs), never from an
        # independently-tracked playhead. This guarantees the waveform and the
        # timeline labels always correspond.
        data_end_seconds = (
            data_end_samples / self._fs if self._fs > 0 else 0.0
        )
        # For file sources, keep the playhead in lockstep with the data we
        # actually show so the status-bar clock matches the waveform. Live
        # sources manage the playhead via ingest_chunk(); we don't touch it
        # here for live so review mode (paused) keeps a frozen playhead.
        if not self.is_live:
            # Use the property setter (not the raw field) so that
            # ``position_changed`` is emitted and the scrubber slider /
            # status-bar clock stay in sync with the waveform.
            self.position_samples = data_end_samples

        # Select only visible channels.
        vis_idx = self.visible_channels()
        if vis_idx and montage_view.shape[0] == self.n_channels:
            montage_view = np.take(montage_view, vis_idx, axis=0)
        elif montage_view.shape[0] != self.n_channels:
            # Montage output count differs from input; show all outputs.
            vis_idx = list(range(montage_view.shape[0]))
        else:
            # All channels hidden — return an empty window so the canvas
            # clears every curve instead of leaking the last frame's data.
            montage_view = np.zeros(
                (0, montage_view.shape[1]), dtype=montage_view.dtype
            )

        info = _RenderInfo(
            n_visible=montage_view.shape[0],
            n_samples=montage_view.shape[1],
            sample_rate_hz=self._fs,
            timebase_s=self._view.timebase_s,
            sensitivity_uv=self._view.sensitivity_uv,
            position_seconds=self.position_seconds,
            # True time of the rightmost displayed sample, derived from the
            # buffer's absolute write position. The X-range and axis labels
            # both use this, so they can never disagree with the waveform.
            data_end_seconds=data_end_seconds,
            visible_indices=vis_idx,
        )
        return montage_view, info


@dataclass
class _RenderInfo:
    n_visible: int = 0
    n_samples: int = 0
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    timebase_s: float = DEFAULT_TIMEBASE_S
    sensitivity_uv: float = DEFAULT_SENSITIVITY_UV
    position_seconds: float = 0.0
    data_end_seconds: float = 0.0
    visible_indices: List[int] = field(default_factory=list)


@dataclass
class _SweepRenderInfo:
    """Metadata for one sweep-cursor frame (see :meth:`render_sweep_frame`).

    The page is a fixed time window in absolute file/acquisition time. The
    canvas draws the page stationary and places the cursor at ``cursor_seconds``.
    """
    n_visible: int = 0
    n_samples: int = 0
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    timebase_s: float = DEFAULT_TIMEBASE_S
    sensitivity_uv: float = DEFAULT_SENSITIVITY_UV
    cursor_seconds: float = 0.0
    page_start_seconds: float = 0.0
    page_end_seconds: float = 0.0
    visible_indices: List[int] = field(default_factory=list)
