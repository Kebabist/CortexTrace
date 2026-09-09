"""Live LSL (Lab Streaming Layer) source - placeholder.

LSL is the de-facto open standard for synchronised streaming EEG acquisition.
This module is intentionally a *fully wired* placeholder: the interface,
threading, and signal contract mirror :class:`FileReplaySource` so that
swapping in real hardware needs no GUI/controller changes - only an
``lsl`` import and the inlet loop.

To activate: ``pip install pylsl`` and replace :py:meth:`_run` with a real
inlet ``pull_chunk`` loop. The rest of the pipeline (ring buffer, render path,
filters, controls) is already hardware-agnostic.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Signal

from .data_source import DataSource, SampleChunk, SourceState


class LiveLslSource(DataSource):
    """Streams live samples from an LSL inlet (currently simulated).

    The simulated path emits a short synthetic sine per channel so the app
    can be exercised end-to-end without hardware. When ``pylsl`` is available
    and ``stream_name`` resolves, :py:meth:`_run` should be replaced with the
    real pull loop (the surrounding scaffolding stays identical).
    """

    # Re-declared so IDE introspection stays accurate in subclasses.
    samples_ready = Signal(object)

    def __init__(
        self,
        stream_name: Optional[str] = None,
        sample_rate_hz: float = 500.0,
        n_channels: int = 32,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._stream_name = stream_name
        self._fs = float(sample_rate_hz)
        self._n_channels = int(n_channels)
        self._channel_names: List[str] = [
            f"CH{i + 1}" for i in range(self._n_channels)
        ]
        self._stop_event = threading.Event()
        # Cleared while streaming, set while paused. The worker loop blocks on
        # this (with a short timeout so it stays responsive to stop()) and skips
        # emission while it is set - so pause() actually suspends the stream
        # instead of being a state-only no-op.
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # DataSource API
    # ------------------------------------------------------------------ #
    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    def sample_rate_hz(self) -> float:
        return self._fs

    def total_samples(self) -> int:
        return -1  # unbounded live stream

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self.metadata_ready.emit(self._channel_names, self._fs)
        self._thread = threading.Thread(
            target=self._run, name="LSLSource", daemon=True
        )
        self._thread.start()
        self._set_state(SourceState.STREAMING)

    def pause(self) -> None:
        # Live acquisition is not truly pausable, but we suspend emission: the
        # worker loop blocks on ``_pause_event`` while it is set and emits no
        # chunks until start() clears it again. ``_pause_event.wait`` uses a
        # short timeout so stop() remains responsive while paused.
        self._pause_event.set()
        self._set_state(SourceState.PAUSED)

    def seek(self, sample_index: int) -> None:
        # Live streams are not seekable. Implemented as a no-op to satisfy
        # the controller contract; the GUI hides seek when total_samples<0.
        return None

    def set_speed(self, multiplier: float) -> None:
        # Live streams run at 1.0x only; no-op keeps the controller simple.
        return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._set_state(SourceState.IDLE)

    # ------------------------------------------------------------------ #
    # Worker - replace with pylsl.pull_chunk when hardware is connected
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Simulated acquisition loop.

        TODO(hardware): when ``pylsl`` is available, resolve the inlet by
        ``stream_name`` and replace this body with::

            inlet = pylsl.StreamInlet(info)
            while not self._stop_event.is_set():
                chunk, ts = inlet.pull_chunk(timeout=0.1)
                if chunk:
                    arr = np.asarray(chunk, dtype=np.float32).T
                    self.samples_ready.emit(SampleChunk(arr, self._fs, ...))
        """
        import math
        import time

        n = self._n_channels
        fs = self._fs
        chunk = 12
        phases = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        # Distinct per-channel frequencies for visual separation.
        freqs = 0.5 + 0.18 * np.arange(n)
        idx = 0
        period = chunk / fs
        while not self._stop_event.is_set():
            # While paused, block (with a short timeout so stop() stays
            # responsive) and emit nothing. The pause check happens *before*
            # emitting so a pause requested between iterations takes effect on
            # the very next loop turn rather than after one more chunk.
            if self._pause_event.is_set():
                # wait() returns True if the event became set (i.e. we should
                # resume); False on timeout. Either way we loop back and
                # re-check stop()/pause() before emitting.
                self._pause_event.wait(timeout=period)
                continue
            t = (np.arange(idx, idx + chunk) / fs).reshape(1, -1)
            wave = 40.0 * np.sin(2 * math.pi * freqs.reshape(-1, 1) * t + phases.reshape(-1, 1))
            self.samples_ready.emit(
                SampleChunk(
                    data=wave.astype(np.float32),
                    sample_rate_hz=fs,
                    channel_names=self._channel_names,
                    start_index=idx,
                )
            )
            idx += chunk
            time.sleep(period)
