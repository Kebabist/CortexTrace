"""Tests for backend/ring_buffer.py."""

from __future__ import annotations

import threading
import numpy as np
import pytest

from eeg_viewer.backend.ring_buffer import RingBuffer


def test_ring_buffer_init_validation():
    with pytest.raises(ValueError):
        RingBuffer(n_channels=0, capacity_samples=100)
    with pytest.raises(ValueError):
        RingBuffer(n_channels=32, capacity_samples=0)

    buf = RingBuffer(n_channels=4, capacity_samples=1000)
    assert buf.n_channels == 4
    assert buf.capacity_samples == 1000
    assert buf.filled == 0
    assert buf.end_index == 0


def test_ring_buffer_write_and_latest_window_unwrapped():
    buf = RingBuffer(n_channels=2, capacity_samples=100)
    data = np.arange(20, dtype=np.float32).reshape(2, 10)
    buf.write(data)

    assert buf.filled == 10
    assert buf.end_index == 10

    view, take = buf.latest_window(5)
    assert take == 5
    assert view.shape == (2, 5)
    np.testing.assert_array_equal(view, data[:, 5:10])

    view_all, take_all = buf.latest_window(15)
    assert take_all == 10
    assert view_all.shape == (2, 10)
    np.testing.assert_array_equal(view_all, data)


def test_ring_buffer_wrapping():
    buf = RingBuffer(n_channels=2, capacity_samples=10)
    # Write 8 samples
    data1 = np.ones((2, 8), dtype=np.float32) * 1.0
    buf.write(data1)
    assert buf.filled == 8
    assert buf.end_index == 8

    # Write 5 samples (will wrap: 2 slots at end, 3 slots at beginning)
    data2 = np.ones((2, 5), dtype=np.float32) * 2.0
    buf.write(data2)
    assert buf.filled == 10  # clamped to capacity
    assert buf.end_index == 13

    view, take = buf.latest_window(10)
    assert take == 10
    assert view.shape == (2, 10)
    # Older 5 samples should be 1.0, newer 5 samples should be 2.0
    expected = np.hstack([np.ones((2, 5), dtype=np.float32) * 1.0,
                          np.ones((2, 5), dtype=np.float32) * 2.0])
    np.testing.assert_array_equal(view, expected)


def test_ring_buffer_clear_and_set_end_index():
    buf = RingBuffer(n_channels=3, capacity_samples=50)
    buf.write(np.ones((3, 20), dtype=np.float32))
    assert buf.filled == 20
    assert buf.end_index == 20

    buf.clear()
    assert buf.filled == 0
    assert buf.end_index == 0

    buf.write(np.ones((3, 10), dtype=np.float32))
    buf.set_end_index(1500)
    assert buf.end_index == 1500
    assert buf.filled == 10


def test_ring_buffer_concurrent_writes_and_reads():
    buf = RingBuffer(n_channels=4, capacity_samples=1000)
    stop_event = threading.Event()
    errors = []

    def writer():
        idx = 0
        while not stop_event.is_set():
            chunk = np.full((4, 10), fill_value=idx, dtype=np.float32)
            try:
                buf.write(chunk)
            except Exception as e:
                errors.append(e)
            idx += 1

    def reader():
        while not stop_event.is_set():
            try:
                view, take = buf.latest_window(50)
                assert take <= 50
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()

    import time
    time.sleep(0.2)
    stop_event.set()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent operations caused errors: {errors}"
