"""Playback controller - orchestrates producer -> buffer -> canvas cadence.

Rendering is driven by a single :class:`QTimer` whose interval sets the *target*
frame rate. The controller no longer tracks a playhead position of its own: the
data source owns the streaming cadence (its own wall-clock-paced tick feeds the
ring buffer at the true acquisition rate), and the model derives the playhead
and timeline from the buffer tail inside ``render_window()``. This gives a
single source of truth for position, so the waveform data and the timeline
labels can never drift apart after seek, loop, speed change, or pause/resume.

The controller connects a :class:`~eeg_viewer.backend.data_source.DataSource`
(the producer) to the :class:`~eeg_viewer.core.model.EEGModel` (the buffer) and
to the GUI canvas (the consumer). Nothing here imports Qt widgets.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from ..config import PlatformProfile
from .model import EEGModel


class PlaybackController(QObject):
    """Orchestrates producer -> buffer -> consumer cadence."""

    # Emitted every frame after the model has ingested/advanced; the canvas
    # connects here to trigger a repaint.
    frame_ready = Signal()

    def __init__(
        self,
        model: EEGModel,
        platform_profile: PlatformProfile,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._platform = platform_profile
        self._source = None          # type: Optional[object]

        # Render timer (single, app-wide). PreciseTimer for the smoothest ticks.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._platform.render_interval_ms)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------ #
    # Source plumbing
    # ------------------------------------------------------------------ #
    def attach_source(self, source) -> None:
        """Connect a :class:`DataSource` to the model and start the cadence."""
        self._source = source
        # Metadata arrives first; once known, size the buffer.
        source.metadata_ready.connect(self._on_metadata)
        source.samples_ready.connect(self._model.ingest_chunk)
        source.state_changed.connect(self._on_source_state)
        source.error.connect(self._on_source_error)

    def detach_source(self) -> None:
        if self._source is None:
            return
        try:
            self._source.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self._source = None
        self.pause()

    @property
    def source(self):
        return self._source

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def play(self) -> None:
        if self._source is None:
            return
        # Live sources: snap the playhead to the newest data on resume so the
        # view doesn't stay stuck at the paused position. File sources keep
        # their paused position (continue like a media player). snap_to_tail()
        # is a no-op for file sources, so this is safe to call unconditionally.
        self._model.snap_to_tail()
        # Sweep-cursor resume: for live sources the cursor must jump to the
        # newest data (the page may have advanced while paused). For file
        # sources the cursor resumes from the status-bar position, drawing over
        # the currently shown page — handled by leaving the page as-is.
        if self._model.sweep_mode and self._model.is_live:
            self._model.reset_sweep_cursor_to_position()
        self._model.set_playing(True)
        self._source.start()
        self._timer.start()

    def pause(self) -> None:
        self._timer.stop()
        self._model.set_playing(False)
        if self._source is not None:
            try:
                self._source.pause()
            except Exception:  # noqa: BLE001
                pass

    def toggle(self) -> None:
        if self._model.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop and fully reset: zero the playhead + clock, clear the buffer and
        reseed an empty sweep page, then repaint. Does NOT auto-restart — the
        app waits for the user to press Play. Distinct from :meth:`pause`,
        which freezes playback in place.
        """
        self._timer.stop()
        self._model.set_playing(False)
        if self._source is not None:
            self._source.stop()        # zeroes the file-replay cursor
        # Drop any streamed data so the next Play starts from an empty buffer.
        if self._model.buffer is not None:
            self._model.buffer.clear()
        # Zero the playhead. The setter emits position_changed(0) which resets
        # the status-bar clock and the scrubber thumb.
        self._model.set_position_seconds(0.0)
        # Reseed the sweep page at t=0 (empty page, ready for the next sweep).
        if self._model.sweep_mode:
            self._model.reset_sweep(left_align=True)
        # Repaint so the UI reflects the reset immediately (blank page + t=0).
        self.frame_ready.emit()

    def seek_seconds(self, seconds: float) -> None:
        """Jump the playhead; works for seekable (file) sources."""
        # Clamp to valid range so we never try to show data before sample 0
        # or past the end of the file.
        duration = self._model.duration_seconds
        if duration > 0:
            seconds = max(0.0, min(seconds, duration))

        # --- Prepare the buffer BEFORE emitting position_changed ---
        # The old order (set_position → clear → fill) was wrong because
        # set_position fires position_changed synchronously, which triggers
        # paint_frame while the buffer still held stale data.  Now we
        # clear + fill first so the canvas sees correct data.
        self._model.seek_pending = True
        if self._model.buffer is not None:
            self._model.buffer.clear()
        if self._source is not None:
            try:
                target = int(seconds * self._model.sample_rate_hz)
                self._source.seek(target)
                if self._model.buffer is not None and hasattr(
                    self._source, "fill_window_into"
                ):
                    window_samples = int(
                        self._model.sample_rate_hz * self._model.viewport.timebase_s
                    )
                    self._source.fill_window_into(self._model.buffer, window_samples)
                    # Seed the buffer's absolute position so the timeline shows
                    # the true file time after the seek. The source's cursor is
                    # the right edge (exclusive) of the data just written, which
                    # is exactly the absolute index the buffer's end_index must
                    # report. Without this, render_window would derive the right
                    # edge from the running write count (≈ window_samples) rather
                    # than the real seek target, and the axis would show ~[0, W]
                    # instead of [target - W, target].
                    cursor = getattr(self._source, "_cursor", None)
                    if cursor is not None:
                        self._model.buffer.set_end_index(int(cursor))
            except Exception:  # noqa: BLE001 - live sources no-op
                pass

        # Now emit the position — paint_frame will find correct data.
        self._model.set_position_seconds(seconds)
        # Sweep mode: reseed the page at the seek target. While PLAYING the
        # cursor must restart at the LEFT edge (left_align=True) so new data is
        # drawn over the seeded trace from column 0. While PAUSED the page is
        # seeded from the buffer so review shows a full waveform immediately.
        if self._model.sweep_mode:
            self._model.reset_sweep(left_align=self._model.is_playing)
        self.frame_ready.emit()

    def step(self, direction: int) -> None:
        """Jump one timebase forward (``direction=+1``) or back (``-1``).

        Bound to the ◀ ▶ buttons and the Left / Right arrow keys. The jump size
        is the current timebase; the cursor lands at the LEFT edge of the new
        page so playback continues cleanly from there. Implemented on top of
        :meth:`seek_seconds`, so file replay re-fills the buffer at the target
        and the sweep page is reseeded.
        """
        if direction == 0:
            return
        timebase = self._model.viewport.timebase_s
        target = self._model.position_seconds + (1 if direction > 0 else -1) * timebase
        self.seek_seconds(target)

    def set_speed(self, multiplier: float) -> None:
        self._model.set_speed(multiplier)
        if self._source is not None:
            try:
                self._source.set_speed(multiplier)
            except Exception:  # noqa: BLE001
                pass

    def set_timebase(self, seconds: float) -> None:
        """Set the timebase, refilling the buffer from the file when it grows.

        ``model.set_timebase`` grows the ring buffer and the sweep page, but
        only reseeds the page from the in-memory buffer — which holds at most
        the *previous* timebase window. When the timebase increases, the wider
        page would end up with NaN gaps (empty viewbox) unless we re-read the
        missing history from the file. So for seekable sources we repeat the
        ``seek_seconds`` refill pattern (clear → seek → ``fill_window_into`` →
        ``set_end_index`` → ``reset_sweep``) at the current playhead. Mirrors
        :meth:`set_speed`'s structure (model call + source follow-up).

        At load time (before a source is attached) this just delegates to the
        model — there is no source to refill from yet.
        """
        self._model.set_timebase(seconds)   # reallocs buffer + page, emits viewport_changed
        if self._source is None:
            return                           # load-time: nothing to refill
        if not hasattr(self._source, "fill_window_into") or self._model.buffer is None:
            return                           # live / non-seekable: keep existing buffer data
        try:
            target = int(self._model.position_samples)
            self._model.buffer.clear()
            self._source.seek(target)
            window_samples = int(
                self._model.sample_rate_hz * self._model.viewport.timebase_s
            )
            self._source.fill_window_into(self._model.buffer, window_samples)
            # Seed the buffer's absolute position so the timeline right edge
            # matches the true file position (same reason as seek_seconds).
            cursor = getattr(self._source, "_cursor", None)
            if cursor is not None:
                self._model.buffer.set_end_index(int(cursor))
            # Reseed the sweep page from the now-full buffer so the wider page
            # is fully populated. (model.set_timebase already reset_sweep'd
            # once from the short buffer; this second reset is idempotent.)
            if self._model.sweep_mode:
                self._model.reset_sweep()
        except Exception:  # noqa: BLE001 - best-effort refill; leave model's state intact
            pass

    # ------------------------------------------------------------------ #
    # Frame tick - wall-clock paced
    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        """Request a repaint each frame.

        The file source owns the data cadence (its own wall-clock-paced
        ``_tick`` feeds the ring buffer at the true acquisition rate), and the
        model derives the playhead and timeline from the buffer tail inside
        ``render_window()``. So this method no longer advances any position of
        its own — it only keeps the canvas repainting at the target frame rate.
        Keeping a redundant wall-clock position tracker here was the root cause
        of data/label drift after seek, loop, speed change, and pause/resume.
        """
        if self._source is None or not self._model.is_playing:
            return
        # Notify the canvas to repaint from the current buffer window.
        self.frame_ready.emit()

    # ------------------------------------------------------------------ #
    # Source signal handlers
    # ------------------------------------------------------------------ #
    @Slot(list, float)
    def _on_metadata(self, channel_names, sample_rate_hz) -> None:
        total = -1
        if self._source is not None:
            try:
                total = self._source.total_samples()
            except Exception:  # noqa: BLE001
                total = -1
        self._model.configure_acquisition(channel_names, sample_rate_hz, total)

    @Slot(object)
    def _on_source_state(self, state) -> None:
        # On EOF, the model decides what to do based on the loop flag.
        name = getattr(state, "value", str(state))
        if name == "eof":
            self._model.notify_eof()
            if not self._model.viewport.loop:
                self.pause()

    @Slot(str)
    def _on_source_error(self, message: str) -> None:
        # Forwarded to the model's listeners via a generic error path.
        # The MainWindow listens to the source directly for message boxes.
        pass
