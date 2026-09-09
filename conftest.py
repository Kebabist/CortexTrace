"""Pytest root configuration."""

import os
import sys

# Set offscreen Qt QPA platform before any Qt/pyqtgraph imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"

here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(here)

if parent not in sys.path:
    sys.path.insert(0, parent)
if here not in sys.path:
    sys.path.insert(0, here)

# Ensure QApplication is initialized before pyqtgraph module imports
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)
