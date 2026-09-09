"""Tests for core/spectral.py."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_viewer.config import DSA_MODE_HEMISPHERE, DSA_MODE_REGIONAL
from eeg_viewer.core.spectral import SpectralEngine, DSAFrame, _group_channels


def test_group_channels_hemisphere():
    names = ["Fp1", "Fp2", "F3", "F4", "Fz", "A1"]
    groups = _group_channels(names, DSA_MODE_HEMISPHERE)
    assert "Left" in groups
    assert "Right" in groups

    # Odd suffix -> Left (Fp1:0, F3:2), Midline Z -> Both (Fz:4)
    # Even suffix -> Right (Fp2:1, F4:3), Midline Z -> Both (Fz:4)
    assert 0 in groups["Left"]
    assert 2 in groups["Left"]
    assert 4 in groups["Left"]

    assert 1 in groups["Right"]
    assert 3 in groups["Right"]
    assert 4 in groups["Right"]


def test_group_channels_regional():
    names = ["FP1", "F3", "C3", "T3", "P3", "O1", "ECG"]
    groups = _group_channels(names, DSA_MODE_REGIONAL)

    assert 0 in groups["Frontal"]  # FP1
    assert 1 in groups["Frontal"]  # F3
    assert 2 in groups["Central"]  # C3
    assert 3 in groups["Temporal"]  # T3
    assert 4 in groups["Posterior"]  # P3
    assert 5 in groups["Posterior"]  # O1


def test_spectral_engine_computation():
    fs = 500.0
    engine = SpectralEngine(sample_rate_hz=fs, mode=DSA_MODE_HEMISPHERE)
    names = ["F3", "F4", "C3", "C4"]

    # 4 seconds of 10Hz sine wave across channels
    t = np.arange(2000) / fs
    wave = np.sin(2 * np.pi * 10.0 * t).reshape(1, -1).astype(np.float32)
    window = np.repeat(wave, 4, axis=0)

    frame: DSAFrame = engine.compute(
        window=window,
        channel_names=names,
        visible_indices=[0, 1, 2, 3],
        start_t=0.0,
        end_t=4.0,
    )

    assert frame.valid is True
    assert frame.n_tracks == 2  # Left and Right
    assert frame.image.ndim == 2
    assert frame.image.shape[0] == frame.n_tracks * frame.freq_bins
    # Check normalized values in range [0, 1]
    assert np.all(frame.image >= 0.0)
    assert np.all(frame.image <= 1.0)


def test_spectral_engine_insufficient_data():
    fs = 500.0
    engine = SpectralEngine(sample_rate_hz=fs, mode=DSA_MODE_HEMISPHERE)
    names = ["F3", "F4"]
    # Only 10 samples (less than FFT window)
    window = np.ones((2, 10), dtype=np.float32)

    frame = engine.compute(
        window=window,
        channel_names=names,
        visible_indices=[0, 1],
        start_t=0.0,
        end_t=0.02,
    )

    assert frame.valid is False
    assert frame.image.size == 0


def test_spectral_engine_nan_window():
    fs = 500.0
    engine = SpectralEngine(sample_rate_hz=fs, mode=DSA_MODE_HEMISPHERE)
    names = ["F3", "F4"]
    window = np.full((2, 2000), np.nan, dtype=np.float32)

    frame = engine.compute(
        window=window,
        channel_names=names,
        visible_indices=[0, 1],
        start_t=0.0,
        end_t=4.0,
    )

    assert frame.valid is False or frame.image.size == 0
