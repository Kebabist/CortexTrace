"""Lock-free single-producer / single-consumer ring buffer for sample streams.

The buffer is a contiguous ``(n_channels, n_samples)`` numpy array, allocated
once. The producer advances a write index; the consumer reads zero-copy views.
This keeps the per-frame hot path allocation-free, which is what makes the
viewer smooth on low-power hardware (e.g. Raspberry Pi) and lets a live
hardware source feed the same pipeline as file replay.
"""

from __future__ import annotations

import threading
from typing import Tuple

import numpy as np


class RingBuffer:
    """Fixed-capacity circular sample buffer.

    Capacity is sized in *seconds* so it adapts to the sample rate. The buffer
    holds at most :pyattr:`capacity_samples` samples per channel.

    Thread-safety: exactly one writer (the producer thread) and one reader
    (the GUI thread). The write index is updated atomically via a lock that
    only serializes the index swap - never the memcpy - so the GUI thread can
    take a view at any time without blocking the producer.
    """

    def __init__(self, n_channels: int, capacity_samples: int, dtype=np.float32):
        if n_channels <= 0 or capacity_samples <= 0:
            raise ValueError("n_channels and capacity_samples must be positive")
        self._n_channels = int(n_channels)
        self._capacity = int(capacity_samples)
        self._dtype = np.dtype(dtype)
        # Shape (channels, samples) -> a contiguous memory block per channel.
        self._buf = np.zeros((self._n_channels, self._capacity), dtype=self._dtype)
        self._write_idx = 0                 # next sample slot to be written
        self._filled = 0                    # number of valid samples (<= capacity)
        # Monotonic count of every sample ever written (never wraps, never
        # decreases except on clear()). The absolute index of the most recent
        # sample is ``_total_written - 1``; the window returned by
        # latest_window(n) spans absolute indices
        # [_total_written - take, _total_written). This is the single source
        # of truth for "what time is the right edge of the displayed data":
        # time = end_index / sample_rate.
        self._total_written = 0
        self._lock = threading.Lock()       # guards index bookkeeping only

    # ------------------------------------------------------------------ #
    # Producer side
    # ------------------------------------------------------------------ #
    def write(self, chunk: np.ndarray) -> None:
        """Append ``chunk`` of shape ``(n_channels, n_samples)`` (in-order)."""
        if chunk.ndim != 2 or chunk.shape[0] != self._n_channels:
            raise ValueError(
                f"chunk must be (n_channels={self._n_channels}, n); got {chunk.shape}"
            )
        n = int(chunk.shape[1])
        if n == 0:
            return
        # Wrap by splitting into at most two contiguous copies.
        with self._lock:
            start = self._write_idx
            first = min(n, self._capacity - start)
            self._buf[:, start:start + first] = chunk[:, :first]
            if n > first:
                second = n - first
                self._buf[:, 0:second] = chunk[:, first:first + second]
            self._write_idx = (self._write_idx + n) % self._capacity
            self._filled = min(self._capacity, self._filled + n)
            self._total_written += n

    # ------------------------------------------------------------------ #
    # Consumer side (GUI thread) - zero-copy views
    # ------------------------------------------------------------------ #
    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def capacity_samples(self) -> int:
        return self._capacity

    @property
    def filled(self) -> int:
        """Number of valid samples currently available to read."""
        return self._filled

    @property
    def end_index(self) -> int:
        """Absolute index just past the most recent sample (exclusive).

        Equals the total number of samples ever written (since the last
        ``clear()``). The window returned by :meth:`latest_window` always
        spans absolute indices ``[end_index - take, end_index)``, so callers
        can compute the true time of the displayed data as
        ``end_index / sample_rate``.
        """
        return self._total_written

    def clear(self) -> None:
        with self._lock:
            self._write_idx = 0
            self._filled = 0
            self._total_written = 0
            self._buf.fill(0)

    def set_end_index(self, absolute_index: int) -> None:
        """Seed the absolute write position (used after a seek).

        After :meth:`clear`, ``_total_written`` is 0 and increments naturally
        with every :meth:`write`. But when the source seeks to a non-zero file
        position and pre-fills the buffer via ``fill_window_into``, the data in
        the buffer corresponds to high absolute file indices that the running
        write count does not reflect. This method lets the caller set
        ``_total_written`` to the absolute index of the right edge of the data
        that was just written, so :attr:`end_index` (and thus the timeline)
        matches the true file position. Must be called AFTER the data is
        written, while the buffer is in a consistent state.
        """
        with self._lock:
            self._total_written = int(absolute_index)

    def latest_window(self, n_samples: int) -> Tuple[np.ndarray, int]:
        """Return ``(view, available)`` for the most recent ``n_samples``.

        The returned array is a *contiguous copy* only when the window straddles
        the wrap boundary; otherwise it is a zero-copy view. Callers should
        treat it as read-only.
        """
        with self._lock:
            filled = self._filled
            take = min(n_samples, filled)
            if take == 0:
                empty = np.zeros((self._n_channels, 0), dtype=self._dtype)
                return empty, 0
            end = self._write_idx            # exclusive
            start = end - take
            if start >= 0:
                return self._buf[:, start:end], take
            # Wrapped: assemble in-order from [0, end) and [capacity+start, capacity)
            tail = -start
            part_a = self._buf[:, 0:end]                      # newer (right side)
            part_b = self._buf[:, self._capacity - tail:self._capacity]  # older (left side)
            return np.concatenate((part_b, part_a), axis=1), take
