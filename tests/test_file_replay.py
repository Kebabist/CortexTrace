"""Tests for backend/file_replay.py and data_source parsing."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from eeg_viewer.backend.file_replay import (
    FileReplaySource,
    FileFormatError,
    _parse_text_file,
    _is_header_line,
)
from eeg_viewer.backend.data_source import SourceState


def test_is_header_line():
    assert _is_header_line(["FP1", "FP2", "F7"]) is True
    assert _is_header_line(["1.0", "2.0", "3.0"]) is False
    assert _is_header_line(["CH1", "CH2"]) is True


def test_parse_valid_text_file(tmp_path):
    file_path = tmp_path / "test_eeg.txt"
    content = "F3 Fz F4\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
    file_path.write_text(content, encoding="utf-8")

    names, data = _parse_text_file(str(file_path))
    assert names == ["F3", "Fz", "F4"]
    assert data.shape == (3, 3)
    np.testing.assert_array_equal(data[0], [1.0, 4.0, 7.0])
    np.testing.assert_array_equal(data[1], [2.0, 5.0, 8.0])


def test_parse_file_without_header(tmp_path):
    file_path = tmp_path / "no_header.txt"
    content = "10.0 20.0\n30.0 40.0\n"
    file_path.write_text(content, encoding="utf-8")

    names, data = _parse_text_file(str(file_path))
    assert names == ["CH1", "CH2"]
    assert data.shape == (2, 2)


def test_parse_empty_or_invalid_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("# Only comments\n", encoding="utf-8")
    with pytest.raises(FileFormatError):
        _parse_text_file(str(empty_file))

    bad_cols = tmp_path / "bad_cols.txt"
    bad_cols.write_text("F3 F4\n1.0 2.0\n3.0 4.0 5.0\n", encoding="utf-8")
    with pytest.raises(FileFormatError):
        _parse_text_file(str(bad_cols))


def test_file_replay_source_lifecycle(tmp_path):
    file_path = tmp_path / "recording.txt"
    file_path.write_text("CH1 CH2\n1.0 2.0\n3.0 4.0\n5.0 6.0\n7.0 8.0\n", encoding="utf-8")

    source = FileReplaySource(str(file_path), sample_rate_hz=500.0)
    assert source.state == SourceState.IDLE

    metadata = []
    source.metadata_ready.connect(lambda names, fs: metadata.append((names, fs)))
    source.load()

    assert source.state == SourceState.IDLE
    assert len(metadata) == 1
    assert metadata[0][0] == ["CH1", "CH2"]
    assert metadata[0][1] == 500.0
    assert source.total_samples() == 4

    source.seek(2)
    assert source._cursor == 2

    source.set_speed(2.0)
    assert source._speed == 2.0

    source.set_speed(float("nan"))  # Should handle NaN gracefully
    assert source._speed == 2.0

    source.stop()
    assert source.state == SourceState.IDLE
