"""File-backed replay source.

Parses a whitespace-delimited EEG text file (first row = channel header,
subsequent rows = samples, one column per channel) and streams it at the
configured sample rate. A wall-clock-driven scheduler emits chunks of
:data:`~eeg_viewer.config.PRODUCER_CHUNK_SAMPLES` samples so the cadence is
true-to-acquisition regardless of timer jitter - exactly how a live hardware
source will behave.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np

from PySide6.QtCore import QTimer, QElapsedTimer, Qt

from ..config import PRODUCER_CHUNK_SAMPLES
from .data_source import DataSource, SampleChunk, SourceState
from .ring_buffer import RingBuffer


class FileFormatError(ValueError):
    """Raised when a file cannot be parsed as expected."""


class FileReplaySource(DataSource):
    """Streams a recorded file at its original sample rate.

    The source is *seekable* and *speed-adjustable*, so the GUI can scrub the
    timeline and replay faster/slower. The on-disk file is loaded once into a
    numpy matrix; streaming then becomes cheap pointer math.
    """

    def __init__(self, path: str, sample_rate_hz: float, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._fs = float(sample_rate_hz)
        self._channel_names: List[str] = []
        self._data: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._cursor = 0           # next sample to emit
        self._speed = 1.0          # realtime multiplier
        self._playing = False
        self._chunk = max(1, int(PRODUCER_CHUNK_SAMPLES))
        # Wall-clock scheduler (lives on whatever thread the object lives on).
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Parse the file. Emits ``metadata_ready`` on success."""
        self._set_state(SourceState.LOADING)
        try:
            names, data = _parse_text_file(self._path)
        except Exception as exc:  # noqa: BLE001 - surface anything to the GUI
            self._set_state(SourceState.ERROR)
            self.error.emit(f"Failed to read {self._path}: {exc}")
            return
        if data.shape[0] == 0 or data.shape[1] == 0:
            self._set_state(SourceState.ERROR)
            self.error.emit("File contains no samples.")
            return
        self._channel_names = names
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._cursor = 0
        self.metadata_ready.emit(self._channel_names, self._fs)
        self._set_state(SourceState.IDLE)

    # ------------------------------------------------------------------ #
    # DataSource API
    # ------------------------------------------------------------------ #
    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    def sample_rate_hz(self) -> float:
        return self._fs

    def total_samples(self) -> int:
        return int(self._data.shape[1])

    def start(self) -> None:
        if self._data.size == 0:
            return
        if self._cursor >= self.total_samples():
            self._cursor = 0
        self._playing = True
        self._clock.start()
        self._emitted_samples = 0.0  # virtual samples produced since play start
        # Backdate the wall-clock anchor so virtual-time already reflects the
        # cursor position. Without this, _tick() computes target ~= 0 (because
        # elapsed ~= 0 after the reset) while _cursor is at the paused position,
        # so the emit loop fires nothing for several seconds and the waveform
        # appears frozen while the X-axis timeline races ahead. This mirrors
        # the backdating already done in seek() below.
        virtual_offset = self._cursor / max(self._fs * self._speed, 1e-3)
        self._wall_at_play = time.perf_counter() - virtual_offset
        interval_ms = max(1, int(self._chunk / (self._fs * max(self._speed, 1e-3)) * 1000))
        self._timer.start(interval_ms)
        self._set_state(SourceState.STREAMING)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self._set_state(SourceState.PAUSED)

    def seek(self, sample_index: int) -> None:
        total = self.total_samples()
        self._cursor = max(0, min(int(sample_index), total - 1))
        # Backdate the wall-clock anchor so virtual-time already reflects the
        # cursor position. Without this, _tick() computes target ~= 0 (because
        # elapsed ~= 0 after the reset) while _cursor is at the seek target,
        # so the emit loop fires nothing for several seconds and the scrubber
        # appears frozen.
        virtual_offset = self._cursor / max(self._fs * self._speed, 1e-3)
        self._wall_at_play = time.perf_counter() - virtual_offset
        self._emitted_samples = 0.0
        if self._playing:
            self._clock.start()

    def fill_window_into(self, buffer, n_samples: int) -> None:
        """Copy up to ``n_samples`` ending at the current cursor into ``buffer``.

        Used after a paused seek so the visible window is populated
        immediately, without waiting for the streaming timer to start. The
        cursor is treated as the *end* (exclusive) of the window, matching
        the "latest sample at the right edge" convention used by the canvas.
        """
        if self._data.size == 0 or n_samples <= 0:
            return
        end = min(int(self._cursor), self.total_samples())
        start = max(0, end - int(n_samples))
        if end > start:
            buffer.write(self._data[:, start:end])

    def set_speed(self, multiplier: float) -> None:
        # Reject non-finite multipliers up front: NaN/inf would otherwise reach
        # ``int(self._chunk / (self._fs * nan) * 1000)`` in start()/here and
        # raise ``ValueError: cannot convert float NaN to integer`` (or produce
        # a degenerate 1 ms timer for inf). Clamp to the previously-set speed.
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            return
        if not math.isfinite(multiplier):
            return
        self._speed = max(0.05, multiplier)
        if self._playing:
            # Re-anchor wall clock to current cursor position so the
            # virtual-time calculation continues seamlessly from here.
            # Without backdating, _tick() thinks virtual time starts
            # from 0 and re-emits data from near the beginning.
            self._emitted_samples = 0.0
            virtual_offset = self._cursor / max(self._fs * self._speed, 1e-3)
            self._wall_at_play = time.perf_counter() - virtual_offset
            interval_ms = max(1, int(self._chunk / (self._fs * self._speed) * 1000))
            self._timer.start(interval_ms)
            self._cursor = max(0, min(self._cursor, self.total_samples() - 1))

    def stop(self) -> None:
        self._timer.stop()
        self._playing = False
        self._cursor = 0
        self._set_state(SourceState.IDLE)

    # ------------------------------------------------------------------ #
    # Cadence-correct scheduler tick
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """Emit enough chunks to catch up to wall-clock-driven virtual time.

        Because QTimer intervals are not precise, we drive the cursor from
        elapsed wall time: ``expected = speed * fs * elapsed``. This keeps the
        streaming rate exact even if a tick fires late, which is the same
        guarantee a hardware clock gives.
        """
        if not self._playing or self._data.size == 0:
            return
        elapsed = time.perf_counter() - self._wall_at_play
        expected_virtual = self._speed * self._fs * elapsed
        # How many real samples should have been emitted by now.
        target = int(expected_virtual)
        # Catch up in chunk-sized steps to keep signal traffic predictable.
        while self._cursor < target and self._cursor < self.total_samples():
            end = min(self._cursor + self._chunk, self.total_samples())
            block = self._data[:, self._cursor:end]
            chunk = SampleChunk(
                data=block,
                sample_rate_hz=self._fs,
                channel_names=self._channel_names,
                start_index=self._cursor,
            )
            self.samples_ready.emit(chunk)
            self._cursor = end
            if self._cursor >= self.total_samples():
                # EOF: stop the scheduler; GUI decides whether to loop.
                self._timer.stop()
                self._playing = False
                self._set_state(SourceState.EOF)
                return
        # If wall time is well ahead of cursor (slow machine), we already caught
        # up in the loop above. If wall time is behind (early tick), we simply
        # emit nothing this cycle and wait for the next.
        _ = self._clock  # kept for future diagnostics


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _parse_text_file(path: str) -> tuple:
    """Return ``(channel_names, data)`` where ``data`` is ``(ch, n)`` float32.

    Accepts the format of the sample file: one header line of whitespace-
    separated channel names, followed by rows of numeric samples. Comments
    (``#``) and blank lines are skipped.
    """
    names: List[str] = []
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if not names:
                # First non-empty line is the header *if* it is non-numeric.
                if _is_header_line(tokens):
                    names = [t for t in tokens]
                    continue
                # Otherwise there is no header; synthesize names later.
            try:
                row = [float(t) for t in tokens]
            except ValueError:
                if not names and not rows:
                    names = [t for t in tokens]
                    continue
                raise FileFormatError(
                    f"Unexpected non-numeric line: {line[:60]!r}"
                )
            rows.append(row)

    if not rows:
        raise FileFormatError("No numeric samples found in file.")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise FileFormatError("Inconsistent column count across rows.")

    if not names:
        names = [f"CH{i + 1}" for i in range(width)]
    elif len(names) != width:
        # Trust the data width; rename/truncate gracefully.
        if len(names) > width:
            names = names[:width]
        else:
            names = names + [f"CH{i + 1}" for i in range(len(names), width)]

    matrix = np.asarray(rows, dtype=np.float32).T  # (channels, samples)
    return names, matrix


def _is_header_line(tokens: List[str]) -> bool:
    try:
        for t in tokens:
            if t.lower() in ("nan", "inf", "-inf", "+inf"):
                return True
            float(t)
        return False
    except ValueError:
        return True


# Convenience for tests / smoke checks -------------------------------------- #
def quick_probe(path: str) -> tuple:
    """Lightweight loader used for smoke tests (not the streaming path)."""
    return _parse_text_file(path)


# Ring-buffer feed helper, used by the model -------------------------------- #
def feed_chunk_into(buffer: RingBuffer, chunk: SampleChunk) -> None:
    buffer.write(chunk.data)
