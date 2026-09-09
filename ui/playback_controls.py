"""Transport bar with clinical controls (sensitivity, timebase, filters, speed).

Every control reads from / writes to the model. No playback logic lives here -
this widget only translates UI gestures into model/controller calls. Controls
are grouped logically left-to-right:

  transport (Back / Play-Pause / Forward / Stop) | timeline scrubber | speed |
  sensitivity | timebase | filters (HF/LF/Notch) | loop

All labels use medical units (µV, Hz, s) - this is what clinicians expect on
EEG hardware, not generic "gain" / "scale".

Transport buttons use the flat `#Transport` QSS style (no rectangular boxes);
the parameter combos are sized to their widest item so the bar stays compact.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from ..config import (
    DSA_MODE_DEFAULT,
    DSA_MODE_LABELS,
    HF_PRESETS_HZ,
    LF_PRESETS_HZ,
    NOTCH_OPTIONS_HZ,
    SENSITIVITY_PRESETS,
    SPEED_PRESETS,
    TIMEBASE_PRESETS,
    TRANSPORT_BTN_EM,
)
from ..core.filters import FilterSettings
from ..core.playback import PlaybackController
from .theme import Fonts, Palette


class PlaybackControls(QWidget):
    """Bottom transport bar wired to a :class:`PlaybackController."""

    def __init__(
        self,
        model,
        controller: PlaybackController,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._controller = controller
        self._was_playing = False   # tracks Play->Pause transitions to enter review mode
        self.setObjectName("Pane")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # ---- Transport cluster ----------------------------------------- #
        # Flat (borderless) buttons: ◀ back | Play/Pause (center) | ▶ forward |
        # ⏹ stop. Back/Forward step by one timebase and land the cursor at the
        # left edge of the new page (bound to Left/Right arrows elsewhere).
        self._back = QPushButton("◀")
        self._back.setObjectName("Transport")
        self._back.setFixedWidth(Fonts.em(TRANSPORT_BTN_EM))
        self._back.setToolTip("Back one page (Left arrow)")
        self._back.clicked.connect(lambda: self._controller.step(-1))
        layout.addWidget(self._back)

        self._play = QPushButton("▶  Play")
        self._play.setObjectName("Transport")
        self._play.setCheckable(True)
        self._play.setToolTip("Play / Pause (Space)")
        self._play.clicked.connect(self._on_play)
        layout.addWidget(self._play)

        self._fwd = QPushButton("▶")
        self._fwd.setObjectName("Transport")
        self._fwd.setFixedWidth(Fonts.em(TRANSPORT_BTN_EM))
        self._fwd.setToolTip("Forward one page (Right arrow)")
        self._fwd.clicked.connect(lambda: self._controller.step(+1))
        layout.addWidget(self._fwd)

        self._stop = QPushButton("⏹")
        self._stop.setObjectName("Transport")
        self._stop.setFixedWidth(Fonts.em(TRANSPORT_BTN_EM))
        self._stop.setToolTip("Stop and reset to start")
        self._stop.clicked.connect(self._controller.stop)
        layout.addWidget(self._stop)

        # ---- Timeline scrubber ----------------------------------------- #
        self._scrubber = Scrubber(model, controller)
        self._scrubber.value_changed.connect(controller.seek_seconds)
        layout.addWidget(self._scrubber, stretch=1)

        # ---- Speed ------------------------------------------------------ #
        layout.addWidget(_dim_label("Speed"))
        self._speed = _preset_combo(SPEED_PRESETS, fmt=lambda v: f"{v:g}×")
        self._speed.setCurrentText(f"{model.viewport.speed:g}×")
        self._speed.currentTextChanged.connect(self._on_speed_text)
        layout.addWidget(self._speed)

        # ---- Sensitivity (µV) ------------------------------------------ #
        layout.addWidget(_dim_label("Sensitivity"))
        self._sens = _preset_combo(SENSITIVITY_PRESETS, fmt=lambda v: f"{int(v)} µV")
        self._sens.setCurrentText(f"{int(model.viewport.sensitivity_uv)} µV")
        self._sens.currentTextChanged.connect(self._on_sens_text)
        layout.addWidget(self._sens)

        # ---- Timebase (s) ---------------------------------------------- #
        layout.addWidget(_dim_label("Timebase"))
        self._tb = _preset_combo(TIMEBASE_PRESETS, fmt=lambda v: f"{int(v)}s")
        self._tb.setCurrentText(f"{int(model.viewport.timebase_s)}s")
        self._tb.currentTextChanged.connect(self._on_timebase_text)
        layout.addWidget(self._tb)

        # ---- Filters --------------------------------------------------- #
        layout.addWidget(_vsep())
        layout.addWidget(_dim_label("HF"))
        self._hf = _preset_combo(
            HF_PRESETS_HZ, fmt=lambda v: "Off" if v == 0 else f"{v:g} Hz"
        )
        self._hf.currentIndexChanged.connect(self._on_filters)
        layout.addWidget(self._hf)

        layout.addWidget(_dim_label("LF"))
        self._lf = _preset_combo(
            LF_PRESETS_HZ, fmt=lambda v: "Off" if v == 0 else f"{v:g} Hz"
        )
        self._lf.currentIndexChanged.connect(self._on_filters)
        layout.addWidget(self._lf)

        layout.addWidget(_dim_label("Notch"))
        self._notch = _preset_combo(
            NOTCH_OPTIONS_HZ, fmt=lambda v: "Off" if v == 0 else f"{int(v)} Hz"
        )
        self._notch.currentIndexChanged.connect(self._on_filters)
        layout.addWidget(self._notch)

        # ---- Loop ------------------------------------------------------ #
        layout.addWidget(_vsep())
        self._loop = QCheckBox("Loop")
        self._loop.setChecked(model.viewport.loop)
        self._loop.toggled.connect(self._model.set_loop)
        layout.addWidget(self._loop)

        # ---- DSA trend panel ------------------------------------------- #
        # Toggle + track-layout dropdown. The dropdown is only enabled while
        # DSA is on. Both persist via the model (restored on next launch).
        self._dsa = QCheckBox("DSA")
        self._dsa.setChecked(model.viewport.dsa_enabled)
        self._dsa.toggled.connect(self._model.set_dsa_enabled)
        layout.addWidget(self._dsa)

        self._dsa_mode = QComboBox()
        self._dsa_mode.setEnabled(model.viewport.dsa_enabled)
        for mode_value, mode_label in DSA_MODE_LABELS.items():
            self._dsa_mode.addItem(mode_label, mode_value)
        # Reflect the persisted mode without emitting (avoids a redundant
        # set_dsa_mode round-trip during initial construction).
        idx = max(
            0,
            self._dsa_mode.findData(model.viewport.dsa_mode or DSA_MODE_DEFAULT),
        )
        self._dsa_mode.setCurrentIndex(idx)
        self._dsa_mode.currentIndexChanged.connect(self._on_dsa_mode)
        layout.addWidget(self._dsa_mode)
        model.dsa_enabled_changed.connect(self._dsa_mode.setEnabled)

        self._sync_state()
        self._model.playing_changed.connect(self._on_playing_changed)

    # ------------------------------------------------------------------ #
    def _on_play(self) -> None:
        # setCheckable(True) lets the button show the playing state visually,
        # but the checked state is driven by _on_playing_changed so playback
        # and the button can never disagree. Prevent the click from also
        # toggling the check (toggle() would fight the model-driven state).
        self._play.blockSignals(True)
        self._play.setChecked(self._model.is_playing)
        self._play.blockSignals(False)
        self._controller.toggle()

    def _on_playing_changed(self, playing: bool) -> None:
        self._play.setText("⏸  Pause" if playing else "▶  Play")
        # Keep the checked state in sync with the model (it is the visual cue
        # for "currently playing" with the flat #Transport style).
        self._play.blockSignals(True)
        self._play.setChecked(playing)
        self._play.blockSignals(False)
        # Review mode is tied to real Play->Pause transitions only. We must not
        # flip `following` on the initial sync (when nothing is playing yet) —
        # otherwise the canvas would enable mouse zoom before any data is loaded.
        if playing:
            self._model.set_following(True)
        elif self._was_playing:
            self._model.set_following(False)
        self._was_playing = playing

    def _on_speed_text(self, text: str) -> None:
        try:
            value = float(text.replace("×", "").strip())
        except ValueError:
            return
        self._controller.set_speed(value)

    def _on_sens_text(self, text: str) -> None:
        try:
            value = float(text.replace("µV", "").strip())
        except ValueError:
            return
        self._model.set_sensitivity(value)

    def _on_timebase_text(self, text: str) -> None:
        try:
            value = float(text.replace("s", "").strip())
        except ValueError:
            return
        # Route through the controller so it can refill the buffer from the file
        # when the timebase grows (otherwise the wider sweep page has NaN gaps).
        self._controller.set_timebase(value)

    def _on_filters(self, *_) -> None:
        hf = float(self._hf.currentData() or 0)
        lf = float(self._lf.currentData() or 0)
        notch = float(self._notch.currentData() or 0)
        self._model.set_filters(FilterSettings(hf_hz=hf, lf_hz=lf, notch_hz=notch))

    def _on_dsa_mode(self, _index: int) -> None:
        mode = self._dsa_mode.currentData()
        if mode:
            self._model.set_dsa_mode(mode)

    def _sync_state(self) -> None:
        self._on_playing_changed(self._model.is_playing)


class Scrubber(QWidget):
    """Timeline slider showing position / duration with click-to-seek."""

    value_changed = Signal(float)   # seconds

    def __init__(self, model, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = model

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._current = QLabel("00:00.000")
        self._current.setObjectName("Mono")
        self._current.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._current)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._dragging = False
        layout.addWidget(self._slider, stretch=1)

        self._total = QLabel("--:--.---")
        self._total.setObjectName("Mono")
        layout.addWidget(self._total)

        self._model.metadata_changed.connect(self._on_metadata)
        self._model.position_changed.connect(self._on_position)

    def _on_metadata(self) -> None:
        if self._model.is_live:
            self._slider.setEnabled(False)
            self._total.setText("LIVE")
        else:
            self._slider.setEnabled(True)
            self._total.setText(_fmt_seconds(self._model.duration_seconds))

    def _on_position(self, seconds: float) -> None:
        self._current.setText(_fmt_seconds(seconds))
        # Don't update the slider while the user is actively dragging —
        # otherwise the playback-driven feedback fights the gesture and
        # causes the thumb to jump or the position to drift.
        if self._dragging:
            return
        if self._model.is_live or self._model.duration_seconds <= 0:
            return
        frac = seconds / self._model.duration_seconds
        self._slider.blockSignals(True)
        self._slider.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._slider.blockSignals(False)

    def _on_slider_pressed(self) -> None:
        self._dragging = True
        self._on_slider_moved(self._slider.value())

    def _on_slider_released(self) -> None:
        self._dragging = False

    def _on_slider_moved(self, value: int) -> None:
        if self._model.duration_seconds <= 0:
            return
        seconds = (value / 1000.0) * self._model.duration_seconds
        self.value_changed.emit(seconds)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _dim_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Dim")
    return lbl


def _vsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setStyleSheet(f"color: {Palette.border}; background: {Palette.border};")
    line.setFixedWidth(1)
    return line


def _preset_combo(values, fmt) -> QComboBox:
    """Build a combobox from preset values, sized to its widest item.

    ``setSizeAdjustPolicy(AdjustToContents)`` + a ``minimumContentsLength`` of
    the longest display string keeps the box tight (no oversized chrome) while
    still wide enough to show every preset without truncation.
    """
    combo = QComboBox()
    for v in values:
        combo.addItem(fmt(v), v)
    # Shrink to fit the widest item; never larger.
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    longest = max((len(fmt(v)) for v in values), default=1)
    combo.setMinimumContentsLength(longest)
    # Drop the up/down arrows on the spinbox-style widgets if we ever reused
    # this helper for one — harmless on QComboBox.
    combo.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, 0)
    return combo


def _fmt_seconds(seconds: float) -> str:
    if seconds == float("inf"):
        return "LIVE"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"
