"""Core package: model, controller and signal-processing for the viewer.

This layer is GUI-agnostic. The model owns the ring buffer plus all viewing
parameters; the controller paces rendering from the wall clock; filters and
montages operate on numpy arrays.
"""

from .model import EEGModel
from .playback import PlaybackController
from .filters import FilterChain
from .montage import apply_montage

__all__ = ["EEGModel", "PlaybackController", "FilterChain", "apply_montage"]
