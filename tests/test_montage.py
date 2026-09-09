"""Tests for core/montage.py."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_viewer.core.montage import (
    Montage,
    identity_montage,
    average_reference_montage,
    apply_montage,
    default_catalogue,
)


def test_identity_montage():
    labels = ["F3", "Fz", "F4"]
    m = identity_montage(labels)
    assert m.name == "Referential"
    assert m.output_labels == labels
    assert m.n_outputs == 3

    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = apply_montage(data, m)
    np.testing.assert_array_equal(out, data)


def test_average_reference_montage():
    labels = ["C3", "Cz", "C4"]
    m = average_reference_montage(labels)
    assert m.name == "Average Reference"
    assert m.output_labels == labels

    # Data: C3=10, Cz=20, C4=30. Mean = 20.
    # Expected output: C3 = 10-20 = -10, Cz = 20-20 = 0, C4 = 30-20 = 10.
    data = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    out = apply_montage(data, m)
    np.testing.assert_allclose(out[0, 0], -10.0, atol=1e-5)
    np.testing.assert_allclose(out[1, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(out[2, 0], 10.0, atol=1e-5)


def test_apply_montage_none():
    data = np.ones((4, 10), dtype=np.float32)
    out = apply_montage(data, None)
    np.testing.assert_array_equal(out, data)


def test_default_catalogue():
    labels = ["O1", "O2"]
    cat = default_catalogue(labels)
    assert len(cat) == 2
    assert cat[0].name == "Referential"
    assert cat[1].name == "Average Reference"
