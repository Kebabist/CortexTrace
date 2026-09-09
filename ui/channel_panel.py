"""Channel panel - per-channel show/hide, color swatch, and Solo.

Features:
* One row per channel: swatch (lane color) + checkbox + Solo button.
* Channels are grouped under collapsible region headers (Frontal, Temporal,
  Central, Parietal, Occipital, Reference, Auxiliary).
* "All" / "None" quick buttons + filter-by-name search.
* The selection is emitted through :class:`EEGModel` and persisted by the
  main window via QSettings, so the doctor's last view is restored on launch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import CHANNEL_REGIONS, build_channel_palette
from .theme import Fonts, Palette


class ChannelRow(QWidget):
    """A single channel: color swatch + visibility checkbox + Solo button."""

    visibility_toggled = Signal(int, bool)   # (channel_index, visible)
    solo_requested = Signal(int)             # channel_index

    def __init__(self, index: int, name: str, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setObjectName("ChannelRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._swatch = _Swatch(color)
        layout.addWidget(self._swatch)

        self._check = QCheckBox(name)
        self._check.setChecked(True)
        self._check.toggled.connect(lambda v: self.visibility_toggled.emit(self._index, v))
        layout.addWidget(self._check, stretch=1)

        self._solo = QPushButton("S")
        self._solo.setFixedWidth(Fonts.em(1.4))
        self._solo.setToolTip(f"Show only {name}")
        self._solo.clicked.connect(lambda: self.solo_requested.emit(self._index))
        layout.addWidget(self._solo)

    def set_color(self, color: QColor) -> None:
        self._swatch.set_color(color)

    def set_checked(self, checked: bool) -> None:
        self._check.blockSignals(True)
        self._check.setChecked(checked)
        self._check.blockSignals(False)


class _Swatch(QFrame):
    """A small colored square showing the channel's trace color."""

    def __init__(self, color: QColor) -> None:
        super().__init__()
        self._color = color
        self.setFixedSize(Fonts.em(0.9), Fonts.em(0.9))
        self._refresh()

    def set_color(self, color: QColor) -> None:
        self._color = color
        self._refresh()

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid {Palette.border};"
            f"border-radius: 2px;"
        )


class ChannelPanel(QWidget):
    """Scrollable panel listing every channel grouped by anatomical region."""

    visibility_changed = Signal()   # coarse notification (mainly for persistence)

    def __init__(self, model, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = model
        self._rows: Dict[int, ChannelRow] = {}
        self.setObjectName("Pane")
        self.setMinimumWidth(Fonts.em(14))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QLabel("Channels")
        header.setObjectName("Heading")
        outer.addWidget(header)

        # Search + All/None.
        tools = QHBoxLayout()
        tools.setSpacing(4)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter...")
        self._search.textChanged.connect(self._apply_filter)
        tools.addWidget(self._search, stretch=1)
        self._all_btn = QPushButton("All")
        self._all_btn.clicked.connect(self._on_all)
        tools.addWidget(self._all_btn)
        self._none_btn = QPushButton("None")
        self._none_btn.clicked.connect(self._on_none)
        tools.addWidget(self._none_btn)
        outer.addLayout(tools)

        # Scroll area holding region groups + rows.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(4)
        self._container_layout.addStretch(1)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

        self._model.metadata_changed.connect(self.rebuild)
        self._model.visibility_changed.connect(self._sync_from_model)

    # ------------------------------------------------------------------ #
    # Build rows grouped by region
    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        # Clear any previous content.
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._rows.clear()

        names = self._model.channel_names
        palette = build_channel_palette(names)
        groups = _group_channels_by_region(names)

        for region_name, idx_list in groups.items():
            if not idx_list:
                continue
            header = QLabel(region_name)
            header.setObjectName("Heading")
            header.setStyleSheet(f"color: {Palette.text_dim}; padding-top: 8px;")
            self._container_layout.insertWidget(self._container_layout.count() - 1, header)
            for idx in idx_list:
                color = QColor.fromRgbF(
                    float(palette[idx][0]),
                    float(palette[idx][1]),
                    float(palette[idx][2]),
                    float(palette[idx][3]),
                )
                row = ChannelRow(idx, names[idx], color)
                row.visibility_toggled.connect(self._on_row_toggled)
                row.solo_requested.connect(self._on_solo)
                self._rows[idx] = row
                self._container_layout.insertWidget(self._container_layout.count() - 1, row)

        self._sync_from_model()

    # ------------------------------------------------------------------ #
    # Selection changes
    # ------------------------------------------------------------------ #
    def _on_row_toggled(self, index: int, visible: bool) -> None:
        self._model.set_channel_visible(index, visible)
        self.visibility_changed.emit()

    def _on_solo(self, index: int) -> None:
        # Show only this channel.
        self._model.set_visible_channels([index])
        self._sync_from_model()
        self.visibility_changed.emit()

    def _on_all(self) -> None:
        self._model.set_visible_channels(list(range(self._model.n_channels)))
        self._sync_from_model()
        self.visibility_changed.emit()

    def _on_none(self) -> None:
        self._model.set_visible_channels([])
        self._sync_from_model()
        self.visibility_changed.emit()

    def _sync_from_model(self) -> None:
        for idx, row in self._rows.items():
            row.set_checked(self._model.is_visible(idx))

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for idx, row in self._rows.items():
            name = self._model.channel_names[idx] if idx < self._model.n_channels else ""
            row.setVisible(not text or text in name.lower())


def _group_channels_by_region(names: List[str]) -> Dict[str, List[int]]:
    """Bucket channel indices into anatomical regions.

    Channels that don't match a known region are collected under
    ``Auxiliary`` so the panel always shows every channel.
    """
    groups: Dict[str, List[int]] = {r: [] for r in CHANNEL_REGIONS}
    for i, name in enumerate(names):
        upper = name.upper()
        placed = False
        for region, members in CHANNEL_REGIONS.items():
            if region == "Auxiliary":
                continue
            if upper in members or any(upper == m for m in members):
                groups[region].append(i)
                placed = True
                break
        if not placed:
            groups["Auxiliary"].append(i)
    return groups
