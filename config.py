"""Central configuration and platform detection for the EEG viewer.

Everything tunable lives here so the rest of the app never branches on
platform details or magic numbers. Sizes are *never* pixel literals; they
derive from DPI / font-metrics at runtime (see :mod:`eeg_viewer.ui.theme`).
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

# --------------------------------------------------------------------------- #
# Acquisition defaults
# --------------------------------------------------------------------------- #
DEFAULT_SAMPLE_RATE_HZ: float = 500.0          # 0.002 s/sample per spec
RENDER_FPS_DESKTOP: int = 60
RENDER_FPS_RPI: int = 30
RENDER_INTERVAL_MS_DESKTOP: int = 1000 // RENDER_FPS_DESKTOP
RENDER_INTERVAL_MS_RPI: int = 1000 // RENDER_FPS_RPI

# Producer chunk size for the worker thread (samples per emitted batch).
# At 500 Hz this is one chunk every 25 ms - low latency, low signal overhead.
PRODUCER_CHUNK_SAMPLES: int = 12

# Ring buffer capacity in seconds. Must exceed the largest visible timebase
# so seeking/scrubbing and the visible window always resolve from memory.
RING_BUFFER_SECONDS: float = 30.0

# --------------------------------------------------------------------------- #
# View / clinical defaults
# --------------------------------------------------------------------------- #
DEFAULT_TIMEBASE_S: float = 10.0               # seconds per page (horizontal window)
DEFAULT_SENSITIVITY_UV: float = 100.0          # µV vertical range shown per lane
DEFAULT_SPEED: float = 1.0                     # playback rate multiplier
DEFAULT_LINEWIDTH_PX: float = 1.0              # trace pen width (scaled by DPR later)

# Speed presets offered in the UI (multiples of realtime).
SPEED_PRESETS: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# --------------------------------------------------------------------------- #
# Sweep-cursor mode (CRT-style refresh sweep)
# --------------------------------------------------------------------------- #
# When True the waveform page is stationary and a vertical cursor sweeps
# left-to-right writing new data over the previous trace, wrapping at the page
# boundary. When False the legacy scrolling mode (waves slide left, playhead
# implicitly at the right edge) is used.
SWEEP_MODE_DEFAULT: bool = True
# Visual style of the sweep cursor line. ACCENT (#2f81f7) gives a clear but
# non-distracting clinical look; alpha < 1 lets the trace show through.
SWEEP_CURSOR_WIDTH_PX: float = 1.5
SWEEP_CURSOR_ALPHA: float = 0.85

# Relative size of the flat transport buttons (◀ ▶ Stop) in font em units.
TRANSPORT_BTN_EM: float = 2.4

# Sensitivity presets in µV (clinical convention).
SENSITIVITY_PRESETS: List[float] = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]

# Timebase presets in seconds per page.
TIMEBASE_PRESETS: List[float] = [5.0, 10.0, 15.0, 20.0, 30.0]

# Filter defaults (all OFF for raw clinical review).
DEFAULT_HF_HZ: float = 0.0        # high-pass / low-cut (Hz); 0 = off
DEFAULT_LF_HZ: float = 0.0        # low-pass / high-cut (Hz); 0 = off
DEFAULT_NOTCH_HZ: float = 0.0     # notch center (Hz); 0 = off (50 or 60 typical)
NOTCH_OPTIONS_HZ: List[float] = [0.0, 50.0, 60.0]
FILTER_ORDER: int = 4

# HF (high-pass / low-cut) presets — standard IFCN / ACNS clinical EEG values.
# 0.0 = off (raw trace). Values in Hz.
HF_PRESETS_HZ: List[float] = [0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 35.0]

# LF (low-pass / high-cut) presets — standard clinical EEG anti-alias / display
# bandwidths. 0.0 = off (full band). Values in Hz.
LF_PRESETS_HZ: List[float] = [0.0, 15.0, 30.0, 40.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0]

# --------------------------------------------------------------------------- #
# DSA (Density Spectral Array) trend graph
# --------------------------------------------------------------------------- #
# A colour-mapped spectrogram shown beneath the waveform, X-linked to the same
# absolute-time axis. EEG is reduced to a few composite tracks (hemisphere or
# regional) BEFORE the FFT - clinical DSA never shows one spectrogram per raw
# electrode. Power is averaged in the linear domain per track, converted to dB,
# then colour-mapped blue (low) -> red (high) with a percentile auto-range.
DSA_DEFAULT: bool = False                    # toggle off by default
DSA_MODE_HEMISPHERE: str = "hemisphere"      # 2 tracks: Left / Right
DSA_MODE_REGIONAL: str = "regional"          # 4 tracks: Frontal/Central/Temporal/Posterior
DSA_MODE_DEFAULT: str = DSA_MODE_HEMISPHERE
DSA_MODE_LABELS = {
    DSA_MODE_HEMISPHERE: "Hemisphere (L/R)",
    DSA_MODE_REGIONAL: "Regional (4)",
}

# Welch epoch length (s). Frequency resolution ~= 1/length (0.5 Hz here).
DSA_FFT_WINDOW_S: float = 2.0
DSA_FFT_OVERLAP: float = 0.5                 # 0..1 overlap between consecutive epochs
DSA_FREQ_MIN_HZ: float = 0.5                 # displayed band floor
DSA_FREQ_MAX_HZ: float = 30.0                # displayed band ceiling
DSA_UPDATE_INTERVAL_MS: int = 1000           # spectral recompute cadence (own slower timer)
DSA_HEIGHT_RATIO: float = 0.33               # fraction of canvas height given to DSA when on
DSA_PERCENTILE_LOW: float = 5.0              # colour auto-range floor (%)
DSA_PERCENTILE_HIGH: float = 95.0            # colour auto-range ceiling (%)

# Blue (low) -> cyan -> green -> yellow -> red (high). RGBA floats.
# (normalised power, (R, G, B, A))
DSA_COLORMAP_STOPS = [
    (0.00, (0.05, 0.20, 0.55, 1.0)),   # deep blue
    (0.25, (0.10, 0.55, 0.85, 1.0)),   # cyan
    (0.50, (0.20, 0.75, 0.45, 1.0)),   # green
    (0.75, (0.95, 0.75, 0.20, 1.0)),   # yellow
    (1.00, (0.92, 0.25, 0.20, 1.0)),   # red
]

# --------------------------------------------------------------------------- #
# Persistence (QSettings keys)
# --------------------------------------------------------------------------- #
APP_NAME: str = "EEGViewer"
APP_ORG: str = "EEGViewer"
SETTINGS_GROUP_UI: str = "ui"
KEY_WINDOW_STATE: str = "ui/windowState"
KEY_WINDOW_GEO: str = "ui/windowGeometry"
KEY_SPLITTER: str = "ui/splitter"
KEY_LAST_DIR: str = "io/lastDirectory"
KEY_CHANNELS_VISIBLE: str = "channels/visible"   # CSV of channel indices
KEY_TIMEBASE: str = "view/timebase"
KEY_SENSITIVITY: str = "view/sensitivity"
KEY_HF: str = "view/hfHz"
KEY_LF: str = "view/lfHz"
KEY_NOTCH: str = "view/notchHz"
KEY_SPEED: str = "view/speed"
KEY_DSA: str = "view/dsa"                    # DSA panel enabled (bool)
KEY_DSA_MODE: str = "view/dsaMode"           # DSA track layout ("hemisphere" | "regional")


# --------------------------------------------------------------------------- #
# 10-20 channel groupings (for tinted regions + group headers)
# --------------------------------------------------------------------------- #
CHANNEL_REGIONS = {
    "Frontal":   ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"],
    "Temporal":  ["T3", "T4", "T5", "T6"],
    "Central":   ["C3", "CZ", "C4"],
    "Parietal":  ["P3", "PZ", "P4"],
    "Occipital": ["O1", "O2"],
    "Reference": ["A1", "A2"],
    "Auxiliary": [],   # filled at runtime from non-standard names (EX*, ECG, EMG...)
}


# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlatformProfile:
    """Tuning knobs that vary by host hardware.

    Auto-detected in :func:`detect_platform`; the rest of the code reads these
    so there are zero ``if platform == ...`` branches elsewhere.
    """
    is_raspberry_pi: bool
    is_embedded: bool
    render_fps: int
    render_interval_ms: int
    use_opengl: bool
    dense_grid: bool
    label: str


def _running_on_raspberry_pi() -> bool:
    """Detect Raspberry Pi without importing anything platform-specific."""
    if sys.platform.startswith("linux"):
        # /proc/device-tree or /proc/cpuinfo model
        try:
            with open("/proc/device-tree/model", "r", encoding="utf-8") as fh:
                model = fh.read().lower()
                if "raspberry pi" in model:
                    return True
        except OSError:
            pass
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model") and "raspberry pi" in line.lower():
                        return True
                    if "raspberrypi" in line.lower():
                        return True
        except OSError:
            pass
        # aarch64 + common Pi kernel string
        machine = platform.machine().lower()
        if machine.startswith(("arm", "aarch64")) and os.path.isdir("/boot"):
            # Heuristic fallback - not all boards, but conservative.
            return "raspberrypi" in " ".join(os.listdir("/boot") or []).lower()
    return False


def detect_platform() -> PlatformProfile:
    """Return a :class:`PlatformProfile` describing the current host."""
    rpi = _running_on_raspberry_pi()
    machine = platform.machine().lower()
    is_embedded = rpi or machine.startswith(("arm", "aarch64"))

    if rpi or is_embedded:
        return PlatformProfile(
            is_raspberry_pi=rpi,
            is_embedded=True,
            render_fps=RENDER_FPS_RPI,
            render_interval_ms=RENDER_INTERVAL_MS_RPI,
            # Mesa OpenGL on Pi can be unstable; software pyqtgraph is fast enough
            # for 32 channels at 30 FPS and far more reliable clinically.
            use_opengl=True,
            dense_grid=False,
            label=f"{platform.system()} {machine} (embedded)",
        )
    return PlatformProfile(
        is_raspberry_pi=False,
        is_embedded=False,
        render_fps=RENDER_FPS_DESKTOP,
        render_interval_ms=RENDER_INTERVAL_MS_DESKTOP,
        use_opengl=True,
        dense_grid=True,
        label=f"{platform.system()} {machine}",
    )


# --------------------------------------------------------------------------- #
# Color palette - even channels red, odd channels blue.
# --------------------------------------------------------------------------- #
_EVEN_COLOR = (0.94, 0.32, 0.29, 1.0)   # red  (R, G, B, A)
_ODD_COLOR  = (0.27, 0.58, 0.97, 1.0)   # blue
_GREEN_COLOR = (0.30, 0.84, 0.42, 1.0)   # green for midline Fz, Cz, Pz
_MIDLINE_NAMES = {"FZ", "CZ", "PZ"}        # uppercase for case-insensitive match


def build_channel_palette(names: List[str]) -> List[np.ndarray]:
    """Return one ``(R,G,B,A)`` float-array per channel in ``names``.

    Fz, Cz, Pz (case-insensitive) channels are green.
    Even-indexed other channels are red.
    Odd-indexed other channels are blue.
    """
    palette: List[np.ndarray] = []
    for i, name in enumerate(names):
        if name.upper() in _MIDLINE_NAMES:
            rgba = _GREEN_COLOR
        else:
            rgba = _EVEN_COLOR if i % 2 == 0 else _ODD_COLOR
        palette.append(np.array(rgba, dtype=np.float32))
    return palette


# # --------------------------------------------------------------------------- #
# # Color palette - 32 perceptually-uniform colors, hemisphere-aware.
# # Left-hemisphere (odd suffix) cool, right-hemisphere (even suffix) warm,
# # midline (Z) neutral, auxiliary muted.
# # --------------------------------------------------------------------------- #
# def _hsl_to_rgb(h: float, s: float, l: float) -> tuple:
#     import colorsys
#     return colorsys.hls_to_rgb(h, l, s)
#
#
# def _channel_color(name: str, base_hue: float) -> tuple:
#     """Return ``(h, s, l)`` for a channel name, hemisphere-aware.
#
#     Midline (suffix ``Z``) -> neutral amber. Odd-suffix (``1``) -> cool band.
#     Even-suffix (``2``) -> warm band. Auxiliary/non-standard -> muted generic.
#     """
#     upper = name.upper()
#     suffix = upper[-1] if upper else ""
#     if suffix == "Z":
#         return 0.13, 0.65, 0.62
#     if suffix == "1":
#         return (0.58 + 0.10 * base_hue) % 1.0, 0.70, 0.62
#     if suffix == "2":
#         return (0.04 + 0.10 * base_hue) % 1.0, 0.70, 0.62
#     return base_hue, 0.55, 0.60
#
#
# def build_channel_palette(names: List[str]) -> List[np.ndarray]:
#     """Return one ``(R,G,B,A)`` float-array per channel in ``names``.
#
#     Hue is spread around the wheel so adjacent lanes contrast. Left/Right
#     hemisphere channels are biased cool/warm respectively; midline neutral.
#     """
#     n = max(len(names), 1)
#     palette: List[np.ndarray] = []
#     for i, name in enumerate(names):
#         base_hue = (i / n) % 1.0
#         hue, sat, lum = _channel_color(name, base_hue)
#         r, g, b = _hsl_to_rgb(hue, sat, lum)
#         palette.append(np.array((r, g, b, 1.0), dtype=np.float32))
#     return palette


# UI surface colors (QSS-friendly hex strings).
CANVAS_BG = "#0d1117"
CANVAS_GRID_MAJOR = "#1f2733"
CANVAS_GRID_MINOR = "#161b24"
PANEL_BG = "#161b22"
PANEL_BG_ALT = "#1c2330"
TEXT_PRIMARY = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT = "#2f81f7"
ACCENT_WARN = "#f0883e"
LIVE_RED = "#ff5c5c"
BORDER = "#30363d"
