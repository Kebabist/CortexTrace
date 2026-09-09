"""Status bar - LIVE indicator, sample rate, master clock, channel count, source.

A single horizontal strip at the bottom of the window, styled to look like the
status line on clinical hardware. All values update from model signals.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette as QtPalette
from PySide6.QtWidgets import QLabel, QStatusBar

from ..config import detect_platform
from .theme import Palette


class StatusBar(QStatusBar):
    """Bottom status strip with clinical readouts."""

    def __init__(self, model, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._model = model
        self.setSizeGripEnabled(False)

        self._platform = detect_platform()

        self._live = QLabel("● LIVE")
        self._live.setStyleSheet(f"color: {Palette.live_red}; font-weight: 700; padding-left: 8px;")
        self._live.setVisible(False)
        self.addWidget(self._live)

        self._rate = QLabel("0.0 Hz")
        self.addWidget(self._rate)

        self._clock = QLabel("00:00.000")
        self._clock.setObjectName("Mono")
        self.addWidget(self._clock)

        self._ch = QLabel("0 / 0 ch")
        self.addWidget(self._ch)

        self._src = QLabel("No source")
        self.addPermanentWidget(self._src)

        self._plat = QLabel(self._platform.label)
        self.addPermanentWidget(self._plat)

        self._model.metadata_changed.connect(self._on_metadata)
        self._model.position_changed.connect(self._on_position)
        self._model.visibility_changed.connect(self._on_visibility)
        self._model.playing_changed.connect(self._on_playing)

    def set_source_label(self, text: str) -> None:
        self._src.setText(text)

    def set_live(self, is_live: bool) -> None:
        self._live.setVisible(is_live)
        if is_live:
            self._live.setText("● LIVE")
        else:
            self._live.setText("○ REC")

    def _on_metadata(self) -> None:
        self._rate.setText(f"{self._model.sample_rate_hz:.1f} Hz")
        self._on_visibility()
        self.set_live(self._model.is_live)

    def _on_position(self, seconds: float) -> None:
        # Guard against non-finite positions: live sources report finite
        # position_seconds today, but a NaN/inf emitted from the model would
        # otherwise raise OverflowError on ``int(seconds // 60)``. Mirror the
        # same guard used by playback_controls._fmt_seconds for LIVE sources.
        if seconds == float("inf") or seconds != seconds:  # inf or NaN
            self._clock.setText("LIVE")
            return
        m = int(seconds // 60)
        s = seconds - m * 60
        self._clock.setText(f"{m:02d}:{s:06.3f}")

    def _on_visibility(self) -> None:
        visible = len(self._model.visible_channels())
        total = self._model.n_channels
        self._ch.setText(f"{visible} / {total} ch")

    def _on_playing(self, playing: bool) -> None:
        if self._model.is_live:
            self._live.setStyleSheet(
                f"color: {Palette.live_red if playing else Palette.text_dim};"
                f" font-weight: 700; padding-left: 8px;"
            )
