"""Master test runner script for EEG Viewer."""

from __future__ import annotations

import os
import sys

# Ensure project root and parent are on sys.path
here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(here)
if here not in sys.path:
    sys.path.insert(0, here)
if parent not in sys.path:
    sys.path.insert(0, parent)

import pytest

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    print("=" * 70)
    print("Running EEG Viewer Automated Test Suite & Bug Discovery...")
    print("=" * 70)

    args = ["-v", "--tb=short", os.path.join(here, "tests")]
    exit_code = pytest.main(args)
    sys.exit(exit_code)
