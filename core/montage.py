"""Montage transforms (referential / average / bipolar).

A montage is a linear recombination of channels - e.g. average reference or
longitudinal bipolar chains - that clinicians use to localise activity. The
default is *referential* (raw), so this module is mostly a reserved hook: the
GUI exposes a montage selector and the model calls :func:`apply_montage` on
each rendered window. New montages can be added without touching the view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class Montage:
    name: str
    # Output channel labels after the transform.
    output_labels: List[str]
    # Mapping: output_index -> list of (input_index, coefficient) pairs.
    # An empty mapping means "identity for that output".
    coefficients: Dict[int, List[tuple]]

    @property
    def n_outputs(self) -> int:
        return len(self.output_labels)


def identity_montage(labels: List[str]) -> Montage:
    """Pass-through montage: every output equals its input."""
    return Montage(
        name="Referential",
        output_labels=list(labels),
        coefficients={i: [] for i in range(len(labels))},
    )


def average_reference_montage(labels: List[str]) -> Montage:
    """Subtract the across-channel mean from each channel."""
    n = len(labels)
    coeffs: Dict[int, List[tuple]] = {}
    for i in range(n):
        # y_i = x_i - mean(x) = x_i - (1/n)*sum(x_j)
        terms = [(j, -1.0 / n) for j in range(n)]
        terms.append((i, 1.0))  # +x_i
        coeffs[i] = terms
    return Montage(
        name="Average Reference",
        output_labels=list(labels),
        coefficients=coeffs,
    )


def apply_montage(window: np.ndarray, montage: Optional[Montage]) -> np.ndarray:
    """Apply ``montage`` to a ``(n_channels, n_samples)`` window.

    Returns the transformed array. ``None`` or an identity montage returns the
    input unchanged. Implemented with ``np.add.reduceat``-free accumulation for
    clarity and speed; coefficient rows are sparse but few.
    """
    if montage is None or montage.name == "Referential":
        return window
    n_out = montage.n_outputs
    n_samples = window.shape[1]
    out = np.zeros((n_out, n_samples), dtype=window.dtype)
    for out_idx, terms in montage.coefficients.items():
        if not terms:
            if out_idx < window.shape[0]:
                out[out_idx] = window[out_idx]
            continue
        acc = np.zeros(n_samples, dtype=window.dtype)
        for in_idx, coef in terms:
            if 0 <= in_idx < window.shape[0]:
                acc += coef * window[in_idx]
        out[out_idx] = acc
    return out


# Pre-baked catalogue for the UI dropdown ------------------------------- #
def default_catalogue(labels: List[str]) -> List[Montage]:
    return [identity_montage(labels), average_reference_montage(labels)]
