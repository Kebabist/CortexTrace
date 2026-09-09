"""Tests for core/filters.py."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_viewer.core.filters import FilterChain, FilterSettings


def test_filter_chain_inactive_pass_through():
    chain = FilterChain(sample_rate_hz=500.0)
    data = np.ones((2, 100), dtype=np.float32)
    out = chain.process(data)
    np.testing.assert_array_equal(out, data)


def test_highpass_cutoff_frequency_accuracy():
    """Verify that high-pass filter cutoff actually attenuates below HP cutoff.

    If hp_hz = 10 Hz and fs = 500 Hz:
    - Signal at 2 Hz should be heavily attenuated (> 10 dB attenuation).
    - Signal at 50 Hz should pass through (~ 0 dB attenuation).
    """
    fs = 500.0
    chain = FilterChain(sample_rate_hz=fs)
    chain.reconfigure(FilterSettings(hf_hz=10.0, lf_hz=0.0, notch_hz=0.0))

    t = np.arange(5000) / fs
    low_freq_sig = np.sin(2 * np.pi * 2.0 * t).reshape(1, -1).astype(np.float32)
    high_freq_sig = np.sin(2 * np.pi * 50.0 * t).reshape(1, -1).astype(np.float32)

    filtered_low = chain.process(low_freq_sig)
    filtered_high = chain.process(high_freq_sig)

    # Calculate RMS amplitude after steady-state (ignoring initial transient)
    rms_low = np.sqrt(np.mean(filtered_low[:, 1000:] ** 2))
    rms_high = np.sqrt(np.mean(filtered_high[:, 1000:] ** 2))

    # Low frequency (2Hz) must be attenuated compared to high frequency (50Hz)
    assert rms_low < 0.2, f"Low frequency (2Hz) should be attenuated by 10Hz HP filter, got RMS={rms_low}"
    assert rms_high > 0.6, f"High frequency (50Hz) should pass 10Hz HP filter, got RMS={rms_high}"


def test_lowpass_cutoff_frequency_accuracy():
    """Verify that low-pass filter cutoff actually attenuates above LP cutoff.

    If lp_hz = 30 Hz and fs = 500 Hz:
    - Signal at 5 Hz should pass through (~ 0 dB attenuation).
    - Signal at 100 Hz should be heavily attenuated.
    """
    fs = 500.0
    chain = FilterChain(sample_rate_hz=fs)
    chain.reconfigure(FilterSettings(hf_hz=0.0, lf_hz=30.0, notch_hz=0.0))

    t = np.arange(5000) / fs
    low_freq_sig = np.sin(2 * np.pi * 5.0 * t).reshape(1, -1).astype(np.float32)
    high_freq_sig = np.sin(2 * np.pi * 100.0 * t).reshape(1, -1).astype(np.float32)

    filtered_low = chain.process(low_freq_sig)
    filtered_high = chain.process(high_freq_sig)

    rms_low = np.sqrt(np.mean(filtered_low[:, 1000:] ** 2))
    rms_high = np.sqrt(np.mean(filtered_high[:, 1000:] ** 2))

    assert rms_low > 0.6, f"Low frequency (5Hz) should pass 30Hz LP filter, got RMS={rms_low}"
    assert rms_high < 0.2, f"High frequency (100Hz) should be attenuated by 30Hz LP filter, got RMS={rms_high}"


def test_notch_filter_attenuation():
    fs = 500.0
    chain = FilterChain(sample_rate_hz=fs)
    chain.reconfigure(FilterSettings(hf_hz=0.0, lf_hz=0.0, notch_hz=50.0))

    t = np.arange(5000) / fs
    notch_sig = np.sin(2 * np.pi * 50.0 * t).reshape(1, -1).astype(np.float32)
    off_notch_sig = np.sin(2 * np.pi * 20.0 * t).reshape(1, -1).astype(np.float32)

    filtered_notch = chain.process(notch_sig)
    filtered_off = chain.process(off_notch_sig)

    rms_notch = np.sqrt(np.mean(filtered_notch[:, 1000:] ** 2))
    rms_off = np.sqrt(np.mean(filtered_off[:, 1000:] ** 2))

    assert rms_notch < 0.1, f"50Hz notch should attenuate 50Hz signal, got RMS={rms_notch}"
    assert rms_off > 0.6, f"50Hz notch should not attenuate 20Hz signal, got RMS={rms_off}"


def test_nan_handling_in_filters():
    fs = 500.0
    chain = FilterChain(sample_rate_hz=fs)
    chain.reconfigure(FilterSettings(hf_hz=1.0, lf_hz=40.0, notch_hz=50.0))

    data = np.ones((2, 100), dtype=np.float32)
    data[0, 50] = np.nan

    out = chain.process(data)
    assert out.shape == data.shape
    # NaN propagation check
    assert np.isnan(out[0, 50])
