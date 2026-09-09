"""Density Spectral Array (DSA) spectral engine.

Converts a multi-channel EEG window into a colour-mapped spectrogram suitable
for a clinical DSA trend display. Pure numpy/scipy: no Qt, no I/O, no globals.
This mirrors :mod:`eeg_viewer.core.filters` conventions (reusable scratch
buffers, operates on ``(n_channels, n_samples)`` float32 windows along axis 1).

Pipeline (per clinically established DSA practice):

1. **Reduce before FFT.** Raw electrodes are grouped into a few composite
   tracks (hemisphere or regional). A 32-row stack of spectrograms is unreadable
   at a glance, which defeats the purpose of a DSA overview.
2. **Average power in the linear domain**, then convert to dB. Averaging
   already-logged values biases the result toward the loudest channel.
3. **Welch-style windowing** (Hanning) per epoch; each DSA column is one FFT of
   ``DSA_FFT_WINDOW_S`` seconds, stepped by the overlap.
4. **Notch before spectrum** is the caller's responsibility (the model's
   ``FilterChain`` runs first); mains hum would otherwise stripe the DSA.
5. **Percentile auto-range** (5-95%) maps dB to a 0..1 LUT index, robust across
   varying amplifier gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    DSA_FFT_OVERLAP,
    DSA_FFT_WINDOW_S,
    DSA_FREQ_MAX_HZ,
    DSA_FREQ_MIN_HZ,
    DSA_MODE_HEMISPHERE,
    DSA_MODE_REGIONAL,
    DSA_PERCENTILE_HIGH,
    DSA_PERCENTILE_LOW,
)


# Regional track order (top -> bottom in the panel). Prefixes are matched
# case-insensitively against channel names; the first matching prefix wins, so
# "FP" must precede "F".
_REGIONAL_TRACKS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Frontal", ("FP", "F")),
    ("Central", ("C",)),
    ("Temporal", ("T",)),
    ("Posterior", ("P", "O")),
]


@dataclass
class DSAFrame:
    """One computed DSA image, ready for an ``ImageItem``.

    ``image`` is laid out as ``(n_tracks * n_freq_bins, n_epochs)``:

    * rows are grouped by track in display order (track 0 at the top); within
      each track block frequencies run **descending** (highest Hz first) so the
      image maps directly onto a non-inverted Y axis with high frequency at the
      top of each band;
    * columns are epochs in ascending time.

    Values are already normalised to ``[0, 1]`` for the colour lookup table.
    When ``valid`` is False the image is empty and the caller should blank the
    panel (e.g. not enough data yet, or no visible channels in any track).
    """

    image: np.ndarray
    freq_bins: int                      # number of frequency rows per track
    n_tracks: int
    track_names: List[str] = field(default_factory=list)
    epoch_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    valid: bool = False


def _group_channels(names: List[str], mode: str) -> Dict[str, List[int]]:
    """Map each track name to the channel indices that belong to it.

    Channels are partitioned by electrode position; non-electrode channels
    (references, aux) fall out of the regional map and are only included in the
    hemisphere map when their suffix digit indicates a side.
    """
    groups: Dict[str, List[int]] = {}

    if mode == DSA_MODE_HEMISPHERE:
        left: List[int] = []
        right: List[int] = []
        for i, name in enumerate(names):
            up = name.upper()
            # Midline (Fz/Cz/Pz) contributes to BOTH hemispheres.
            if up.endswith("Z"):
                left.append(i)
                right.append(i)
                continue
            # Odd suffix digit -> left; even -> right.
            if up and up[-1].isdigit():
                if int(up[-1]) % 2 == 1:
                    left.append(i)
                else:
                    right.append(i)
        groups["Left"] = left
        groups["Right"] = right
        return groups

    if mode == DSA_MODE_REGIONAL:
        for label, _ in _REGIONAL_TRACKS:
            groups[label] = []
        for i, name in enumerate(names):
            up = name.upper()
            for label, prefixes in _REGIONAL_TRACKS:
                if up.startswith(prefixes):
                    groups[label].append(i)
                    break
        return groups

    # Unknown mode -> single aggregate track so the panel never crashes.
    groups["All"] = list(range(len(names)))
    return groups


class SpectralEngine:
    """Stateful Welch-DSA transform over a sliding multi-channel window.

    Scratch buffers and the Hanning window / frequency-bin index are cached and
    reused across calls to keep the hot path allocation-free. Call
    :meth:`reconfigure` when the sample rate or mode changes.
    """

    def __init__(self, sample_rate_hz: float, mode: str = DSA_MODE_HEMISPHERE) -> None:
        self._fs: float = float(sample_rate_hz)
        self._mode: str = mode
        self._names: List[str] = []
        self._groups: Dict[str, List[int]] = {}

        # Cached FFT plan (rebuilt when nperseg changes with fs).
        self._nperseg: int = 0
        self._step: int = 0
        self._win: Optional[np.ndarray] = None          # Hanning window
        self._freq_idx: Optional[np.ndarray] = None     # bin indices within [fmin, fmax]
        self._freqs: np.ndarray = np.empty(0, dtype=np.float64)

        # Scratch for the assembled image (reused across frames).
        self._img: Optional[np.ndarray] = None

        self._rebuild_fft_plan()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def reconfigure(
        self,
        sample_rate_hz: Optional[float] = None,
        mode: Optional[str] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        if sample_rate_hz is not None and float(sample_rate_hz) != self._fs:
            self._fs = float(sample_rate_hz)
            self._rebuild_fft_plan()
        if mode is not None and mode != self._mode:
            self._mode = mode
            # Mode change can alter grouping even with the same names.
            if self._names:
                self._groups = _group_channels(self._names, self._mode)
        if names is not None and names != self._names:
            self._names = list(names)
            self._groups = _group_channels(self._names, self._mode)

    def _rebuild_fft_plan(self) -> None:
        nperseg = max(1, int(round(self._fs * DSA_FFT_WINDOW_S)))
        step = max(1, int(round(nperseg * (1.0 - DSA_FFT_OVERLAP))))
        if nperseg != self._nperseg or step != self._step:
            self._nperseg = nperseg
            self._step = step
            self._win = np.hanning(nperseg).astype(np.float32)
            freqs = np.fft.rfftfreq(nperseg, d=1.0 / self._fs)
            mask = (freqs >= DSA_FREQ_MIN_HZ) & (freqs <= DSA_FREQ_MAX_HZ)
            self._freq_idx = np.nonzero(mask)[0]
            self._freqs = freqs[mask]
            # Invalidate cached image: row count depends on freq bins.
            self._img = None

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #
    def compute(
        self,
        window: np.ndarray,
        channel_names: List[str],
        visible_indices: List[int],
        start_t: float,
        end_t: float,
    ) -> DSAFrame:
        """Compute a DSA frame for one display window.

        ``window`` is ``(n_channels, n_samples)`` float32, already filtered by
        the caller (notch + band-pass). ``visible_indices`` restricts which
        channels contribute to each track's average; a track whose channels are
        all hidden is omitted.
        """
        # Refresh grouping if the channel set changed since last call.
        if channel_names != self._names:
            self._names = list(channel_names)
            self._groups = _group_channels(self._names, self._mode)

        visible = set(visible_indices)
        n_samples = 0 if window is None else int(window.shape[1])

        # Not enough data for even one FFT segment -> blank.
        if (
            n_samples < self._nperseg
            or self._freq_idx is None
            or self._freq_idx.size == 0
            or self._win is None
        ):
            return DSAFrame(
                image=np.empty((0, 0), dtype=np.float32),
                freq_bins=int(self._freq_idx.size) if self._freq_idx is not None else 0,
                n_tracks=0,
                valid=False,
            )

        # Number of (overlapping) epochs that fit in this window.
        n_epochs = 1 + (n_samples - self._nperseg) // self._step
        if n_epochs < 1:
            return DSAFrame(
                image=np.empty((0, 0), dtype=np.float32),
                freq_bins=int(self._freq_idx.size),
                n_tracks=0,
                valid=False,
            )

        # Build the per-track spectrograms. Each track that has at least one
        # visible channel yields a (n_freq_bins, n_epochs) block; tracks with no
        # visible channels are dropped from the display entirely.
        track_blocks: List[Tuple[str, np.ndarray]] = []
        for track_name, idxs in self._groups.items():
            # Restrict to visible channels present in this track.
            vis_idxs = [i for i in idxs if i in visible and i < window.shape[0]]
            if not vis_idxs:
                continue
            block = self._welch_track(window[vis_idxs])  # (n_freq_bins, n_epochs)
            if block is not None:
                track_blocks.append((track_name, block))

        if not track_blocks:
            return DSAFrame(
                image=np.empty((0, 0), dtype=np.float32),
                freq_bins=int(self._freq_idx.size),
                n_tracks=0,
                valid=False,
            )

        n_freq = self._freq_idx.size
        n_tracks = len(track_blocks)
        n_rows = n_tracks * n_freq

        # Assemble the stacked image, reusing the scratch buffer.
        if self._img is None or self._img.shape != (n_rows, n_epochs):
            self._img = np.empty((n_rows, n_epochs), dtype=np.float32)
        for t, (_name, block) in enumerate(track_blocks):
            self._img[t * n_freq:(t + 1) * n_freq, :] = block

        # Percentile auto-range over finite dB values -> 0..1 for the LUT.
        finite = self._img[np.isfinite(self._img)]
        if finite.size == 0:
            return DSAFrame(
                image=np.empty((0, 0), dtype=np.float32),
                freq_bins=n_freq,
                n_tracks=0,
                valid=False,
            )
        plo, phi = np.percentile(finite, [DSA_PERCENTILE_LOW, DSA_PERCENTILE_HIGH])
        span = phi - plo
        if span < 1e-6:
            span = 1.0
        np.clip((self._img - plo) / span, 0.0, 1.0, out=self._img)

        # Epoch centre times, evenly sampling [start_t, end_t].
        if n_epochs == 1:
            epoch_times = np.array([0.5 * (start_t + end_t)], dtype=np.float64)
        else:
            frac = (np.arange(n_epochs, dtype=np.float64) + 0.5) / n_epochs
            epoch_times = start_t + frac * (end_t - start_t)

        return DSAFrame(
            image=self._img,
            freq_bins=n_freq,
            n_tracks=n_tracks,
            track_names=[name for name, _ in track_blocks],
            epoch_times=epoch_times,
            valid=True,
        )

    def _welch_track(self, track_view: np.ndarray) -> Optional[np.ndarray]:
        """Power spectrum per epoch, averaged across the channels in a track.

        ``track_view`` is ``(k, n_samples)`` for the ``k`` channels in the track.
        Returns ``(n_freq_bins, n_epochs)`` in dB with frequency **descending**
        (highest Hz in row 0) so the block maps onto a non-inverted Y axis.
        """
        n_samples = track_view.shape[1]
        if n_samples < self._nperseg or self._win is None or self._freq_idx is None:
            return None

        # Strided, zero-copy epoch extraction: (k, n_windows, nperseg).
        strided = np.lib.stride_tricks.sliding_window_view(
            track_view, self._nperseg, axis=1
        )
        epochs = strided[:, :: self._step, :]            # (k, n_epochs, nperseg)
        n_epochs = epochs.shape[1]
        if n_epochs < 1:
            return None

        # Hanning-windowed FFT, power per channel.
        windowed = epochs * self._win                    # broadcast over nperseg
        spec = np.fft.rfft(windowed, axis=2)             # (k, n_epochs, nbins)
        power = (spec.real * spec.real + spec.imag * spec.imag)  # |F|^2, linear

        # Average across channels in the LINEAR domain, then dB.
        track_power = power.mean(axis=0)                 # (n_epochs, nbins)
        track_db = 10.0 * np.log10(track_power + 1e-20)  # (n_epochs, nbins)

        # Select the displayed frequency band.
        track_db = track_db[:, self._freq_idx]           # (n_epochs, n_freq_bins)

        # Transpose to (n_freq_bins, n_epochs) and flip freq axis so row 0 is
        # the highest frequency (top of each track band).
        block = np.ascontiguousarray(track_db.T[::-1, :], dtype=np.float32)
        return block

    # ------------------------------------------------------------------ #
    # Read-only accessors (used by the view for axis ticks / labels)
    # ------------------------------------------------------------------ #
    @property
    def sample_rate_hz(self) -> float:
        return self._fs

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def freqs(self) -> np.ndarray:
        """Displayed frequency bins, ascending (Hz)."""
        return self._freqs
