"""Data-source abstraction.

A :class:`DataSource` is anything that can feed sample chunks into the model.
Concrete producers:

* :class:`eeg_viewer.backend.FileReplaySource` - streams a recorded file at
  the original sample rate (a faithful *simulated* live source).
* :class:`eeg_viewer.backend.LiveLslSource` - placeholder for live LSL
  acquisition, wired into the same pipeline so the GUI never needs to change.

All producers live on a worker :class:`~PySide6.QtCore.QThread` and push
chunks via the thread-safe ``samples_ready`` signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from PySide6.QtCore import QObject, Signal


class SourceState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    STREAMING = "streaming"
    PAUSED = "paused"
    EOF = "eof"
    ERROR = "error"


@dataclass
class SampleChunk:
    """A batch of samples: shape ``(n_channels, n_samples)`` (float32)."""
    data: np.ndarray
    sample_rate_hz: float
    channel_names: List[str]
    # Absolute sample index of the first column in ``data`` (monotonic).
    start_index: int = 0


class DataSource(QObject):
    """Abstract producer.

    Subclasses MUST emit :pyattr:`samples_ready` from the worker thread and
    :pyattr:`state_changed` / :pyattr:`metadata_ready` (once) before the first
    chunk. The model owns the buffer; the source owns the cadence.
    """

    # Emitted once, after load completes, before the first chunk.
    metadata_ready = Signal(list, float)       # (channel_names, sample_rate_hz)
    # Emitted for every chunk on the worker thread (queued connection to GUI).
    samples_ready = Signal(object)             # SampleChunk
    state_changed = Signal(object)             # SourceState
    error = Signal(str)                        # human-readable message

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state = SourceState.IDLE

    # ------------------------------------------------------------------ #
    # API to implement
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> SourceState:
        return self._state

    def channel_names(self) -> List[str]:
        raise NotImplementedError

    def sample_rate_hz(self) -> float:
        raise NotImplementedError

    def total_samples(self) -> int:
        """Total samples available, or -1 for an unbounded live source."""
        return -1

    def start(self) -> None:
        """Begin (or resume) streaming from the current position."""
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def seek(self, sample_index: int) -> None:
        """Jump the read cursor to ``sample_index`` (clamped to valid range)."""
        raise NotImplementedError

    def set_speed(self, multiplier: float) -> None:
        """Set realtime multiplier (1.0 == acquisition rate)."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers for subclasses
    # ------------------------------------------------------------------ #
    def _set_state(self, state: SourceState) -> None:
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state)
