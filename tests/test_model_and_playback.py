"""Tests for core/model.py and core/playback.py."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_viewer.config import detect_platform
from eeg_viewer.core.model import EEGModel
from eeg_viewer.core.playback import PlaybackController
from eeg_viewer.backend.data_source import SampleChunk, SourceState, DataSource


class MockDataSource(DataSource):
    def __init__(self, n_channels=4, fs=500.0, total_samples=1000):
        super().__init__()
        self._n_channels = n_channels
        self._fs = fs
        self._total = total_samples
        self._names = [f"CH{i+1}" for i in range(n_channels)]
        self._cursor = 0

    def channel_names(self):
        return self._names

    def sample_rate_hz(self):
        return self._fs

    def total_samples(self):
        return self._total

    def start(self):
        self._set_state(SourceState.STREAMING)

    def pause(self):
        self._set_state(SourceState.PAUSED)

    def seek(self, sample_index):
        self._cursor = sample_index

    def fill_window_into(self, buffer, n_samples):
        data = np.ones((self._n_channels, min(n_samples, self._total)), dtype=np.float32)
        buffer.write(data)

    def set_speed(self, multiplier):
        pass

    def stop(self):
        self._set_state(SourceState.IDLE)


def test_model_configure_acquisition():
    model = EEGModel()
    names = ["F3", "Fz", "F4", "C3"]
    model.configure_acquisition(names, sample_rate_hz=500.0, total_samples=5000)

    assert model.channel_names == names
    assert model.n_channels == 4
    assert model.sample_rate_hz == 500.0
    assert model.total_samples == 5000
    assert model.duration_seconds == 10.0
    assert model.is_live is False
    assert len(model.visible_channels()) == 4


def test_model_visibility_toggles():
    model = EEGModel()
    model.configure_acquisition(["CH1", "CH2", "CH3"], 500.0)

    model.set_channel_visible(1, False)
    assert model.visible_channels() == [0, 2]
    assert model.is_visible(1) is False

    model.set_visible_channels([0])
    assert model.visible_channels() == [0]


def test_model_sweep_mode_toggle_and_render():
    model = EEGModel()
    model.configure_acquisition(["CH1", "CH2"], 500.0)
    assert model.sweep_mode is True

    # Ingest a chunk
    chunk = SampleChunk(
        data=np.ones((2, 50), dtype=np.float32),
        sample_rate_hz=500.0,
        channel_names=["CH1", "CH2"],
    )
    model.ingest_chunk(chunk)

    frame, info = model.render_sweep_frame(advance=True)
    assert info.n_visible == 2
    assert frame.shape[0] == 2

    # Toggle to scroll mode
    model.set_sweep_mode(False)
    assert model.sweep_mode is False

    frame_scroll, info_scroll = model.render_window()
    assert info_scroll.n_visible == 2


def test_playback_controller_actions():
    profile = detect_platform()
    model = EEGModel()
    controller = PlaybackController(model, profile)

    source = MockDataSource(n_channels=4, fs=500.0, total_samples=5000)
    controller.attach_source(source)

    # Initial metadata emission simulation
    source.metadata_ready.emit(source.channel_names(), source.sample_rate_hz())
    assert model.n_channels == 4

    controller.play()
    assert model.is_playing is True

    controller.pause()
    assert model.is_playing is False

    controller.seek_seconds(5.0)
    assert model.position_seconds == 5.0

    controller.step(+1)  # forward 1 timebase (10s -> clamped to duration 10s)
    assert model.position_seconds == 10.0

    controller.stop()
    assert model.is_playing is False
    assert model.position_seconds == 0.0
