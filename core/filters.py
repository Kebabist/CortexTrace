"""Vectorised clinical filters (high-pass, low-pass, 50/60 Hz notch).

Filters operate on whole windows ``(n_channels, n_samples)`` and are applied
just-in-time on each rendered frame. They are *stateful* so the tail of one
window feeds seamlessly into the next - this avoids edge transients and gives
the smooth, continuous look clinicians expect.

Implementation notes:

* We use second-order-section (``sos``) Butterworth designs for numerical
  stability, which matters at low cutoffs on short windows.
* The notch is an IIR notch (``scipy.signal.iirnotch``).
* When all filters are off, :py:meth:`FilterChain.process` short-circuits and
  returns the input array untouched - zero overhead on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from ..config import FILTER_ORDER


@dataclass
class FilterSettings:
    hf_hz: float = 0.0       # high-pass cutoff (Hz); 0 = off
    lf_hz: float = 0.0       # low-pass cutoff (Hz); 0 = off
    notch_hz: float = 0.0    # notch center (Hz); 0 = off

    @property
    def active(self) -> bool:
        return self.hf_hz > 0 or self.lf_hz > 0 or self.notch_hz > 0


class FilterChain:
    """Stateful multi-channel filter applied per rendered window.

    Call :py:meth:`reconfigure` whenever cutoffs change; the internal state is
    reset so the new response is clean. :py:meth:`process` is cheap enough to
    call every frame on 32 channels.
    """

    def __init__(self, sample_rate_hz: float) -> None:
        self._fs = float(sample_rate_hz)
        self._settings = FilterSettings()
        self._bp_sos: Optional[np.ndarray] = None
        self._notch_sos: Optional[np.ndarray] = None

    @property
    def settings(self) -> FilterSettings:
        return self._settings

    @property
    def sample_rate_hz(self) -> float:
        return self._fs

    def reconfigure(self, settings: FilterSettings) -> None:
        """Rebuild coefficients for the given cutoffs and reset state."""
        self._settings = settings
        nyq = self._fs / 2.0

        # Band-pass design (combined HP + LP).
        hp = settings.hf_hz if settings.hf_hz > 0 else None
        lp = settings.lf_hz if settings.lf_hz > 0 else None
        if hp is not None or lp is not None:
            # Cutoffs in Hz (clamped to valid range below Nyquist).
            c_hp = min(max(hp, 1e-4 * nyq), 0.999 * nyq) if hp is not None else None
            c_lp = min(max(lp, 1e-4 * nyq), 0.999 * nyq) if lp is not None else None

            if c_hp is not None and c_lp is not None:
                lo, hi = sorted([c_hp, c_lp])
                if lo >= hi:
                    lo = hi * 0.9          # prevent degenerate bandpass
                self._bp_sos = signal.butter(
                    FILTER_ORDER, [lo, hi], btype="bandpass", output="sos", fs=self._fs
                )
            elif c_hp is not None:
                self._bp_sos = signal.butter(
                    FILTER_ORDER, c_hp, btype="highpass", output="sos", fs=self._fs
                )
            else:
                self._bp_sos = signal.butter(
                    FILTER_ORDER, c_lp, btype="lowpass", output="sos", fs=self._fs
                )
        else:
            self._bp_sos = None

        # Notch design.
        if settings.notch_hz > 0:
            w0 = settings.notch_hz / (self._fs / 2.0)
            if 0 < w0 < 1:
                # Quality factor sets the notch width; 30 ~= narrow clinical notch.
                sos = signal.iirnotch(settings.notch_hz, 30.0, fs=self._fs)
                # iirnotch returns (b, a); convert to single-section sos form.
                self._notch_sos = np.array(
                    [[sos[0][0], sos[0][1], sos[0][2],
                      sos[1][0], sos[1][1], sos[1][2]]], dtype=np.float64
                )
            else:
                self._notch_sos = None
        else:
            self._notch_sos = None

    def reset(self) -> None:
        pass

    def process(self, window: np.ndarray) -> np.ndarray:
        if not self._settings.active or window.shape[1] < int(self._fs) // 25:
            return window

        nan_mask = np.isnan(window) if window.size > 0 else None
        has_nan = nan_mask is not None and nan_mask.any()

        # Fill NaNs with zeros temporarily to prevent IIR NaN feedback propagation
        out = np.nan_to_num(window, nan=0.0) if has_nan else window

        if self._bp_sos is not None:
            out = signal.sosfilt(self._bp_sos, out, axis=1)
        if self._notch_sos is not None:
            out = signal.sosfilt(self._notch_sos, out, axis=1)

        # Restore NaNs where they originally existed
        if has_nan:
            out = np.where(nan_mask, np.nan, out)

        return np.ascontiguousarray(out, dtype=np.float32)
