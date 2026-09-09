"""Backend package: data producers and the ring buffer.

Producers run on a worker thread and push fixed-size sample chunks into the
model via a thread-safe Qt signal. File replay and live hardware sources are
interchangeable - they share the same ring-buffer/render pipeline.
"""

from .data_source import DataSource, SampleChunk, SourceState
from .ring_buffer import RingBuffer
from .file_replay import FileReplaySource, FileFormatError
from .live_lsl import LiveLslSource

__all__ = [
    "DataSource",
    "SampleChunk",
    "SourceState",
    "RingBuffer",
    "FileReplaySource",
    "FileFormatError",
    "LiveLslSource",
]
