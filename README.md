# 🧠 NeuroStream EEG Viewer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52.svg?logo=qt&logoColor=white)](https://pyside.org/)
[![pyqtgraph](https://img.shields.io/badge/Graphics-pyqtgraph-FF6F00.svg)](https://www.pyqtgraph.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey.svg)](#cross-platform--hardware-tuning)

A high-performance, real-time 32-channel EEG signal visualization and DSP analyzer built with **PySide6** and **pyqtgraph**. Designed with clinical-grade standards: smooth sub-sample scrolling, real-time Butterworth & IIR notch filters, Density Spectral Array (DSA) spectrograms, and an ultra-responsive dark interface tailored for medical hardware.

Runs seamlessly across **Windows**, **Linux**, and **Raspberry Pi** with zero platform-specific UI code.

---

## ✨ Key Features

### ⚡ Real-Time Pipeline & Streaming Architecture
- **Lock-Free SPSC Ring Buffer:** Built with a high-performance single-producer / single-consumer `numpy` ring buffer decoupling acquisition hardware/files from GUI rendering.
- **Wall-Clock Playhead Pacing:** Playhead advance is anchored to physical elapsed time (`QElapsedTimer`), maintaining true 500 Hz streaming speed without timing drift even during frame drops.
- **Zero-Allocation Hot Path:** Pre-allocated plot curves updated via in-place `setData()` and zero-copy array slicing to minimize garbage collection overhead.

### 🔬 Clinical Signal Processing & Analysis
- **Digital Filtering:** Stateful real-time High-pass (HP), Low-pass (LP), and 50/60 Hz IIR Notch filters (scipy Butterworth).
- **Density Spectral Array (DSA):** Real-time spectrogram time-frequency waterfall display for spectral monitoring across channels.
- **Montage Support:** Referential and Average Reference transforms for spatial signal re-referencing.
- **Channel Controls:** Toggle 32 individual channels, **Solo** specific channels, or filter by anatomical regions (*Frontal, Temporal, Central, Parietal, Occipital*). Channel selection state persists across sessions.

### 🖥️ Professional Clinical UI & UX
- **Medical Signal Conventions:** Displayed with vertical sensitivity controls (10–1000 µV), timebase adjustment (5–30 s/page), and margin-aligned channel labels.
- **Dark Theme Palette:** Perceptually uniform 32-color hemisphere-aware color palette optimized for high-contrast dark room environments.
- **DPI-Aware & Responsive:** Dynamic layout math scales crisp UI elements on 4K monitors down to 7" touchscreens without hardcoded pixel offsets.

---

## 🏗️ System Architecture

```
PRODUCER (Worker Thread)                   MODEL (GUI Thread)               VIEW (60 FPS / 30 FPS)
┌─────────────────────────┐               ┌──────────────────────┐         ┌───────────────────────┐
│ DataSource (ABC)        │   Qt Signal   │ EEGModel             │  Read   │ TraceCanvas           │
│  ├─ FileReplaySource    │──────────────>│  ├─ SPSC RingBuffer  │────────>│  (pyqtgraph)          │
│  └─ LiveLslSource (stub)│   (queued)    │  └─ Filters & Vis    │         │  Persistent Curves    │
└─────────────────────────┘               └──────────┬───────────┘         └───────────────────────┘
                                                     │ Qt Signals
                                           PlaybackController (Wall-Clock Timer)
```

> **Why decoupled Architecture?**  
> The file streaming source and live hardware sources (LSL / Serial / TCP) share the exact same ring buffer and render path. Swapping from offline file replay to real-time amplifier streaming requires **zero GUI or controller modifications**.

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10+**

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neurostream-eeg.git
   cd neurostream-eeg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

Launch the application:

```bash
# Run as module
python -m eeg_viewer.main

# Or execute main script directly
python main.py
```

### Loading Data
- **Open File:** Navigate to `File -> Open…` (`Ctrl+O`) and select a formatted `.txt` recording.
- **Drag & Drop:** Simply drag and drop any supported EEG file directly into the application window.

### Supported File Format
Whitespace-delimited text format:
- **Line 1:** Space-separated channel names (e.g., `FP1 FP2 F7 F3 C3 ... CH32`)
- **Lines 2+:** Numerical matrix representing voltage samples per timepoint.

*Includes a sample dataset (`EEG3840 Sine.txt` — 32 channels × 8,600 samples at 500 Hz).*

---

## 🎛️ Controls Reference

| Shortcut / Control | Feature / Description |
| :--- | :--- |
| **`▶ Play` / `⏸ Pause`** | Start or pause real-time playback |
| **`■ Stop`** | Reset playhead to beginning |
| **Scrubber Timeline** | Interactive drag-and-seek timeline scrubber |
| **Speed (0.25× – 8×)** | Adjustable playhead playback rate multiplier |
| **Sensitivity (10–1000 µV)** | Microvolt scaling gain adjustment |
| **Timebase (5–30 s)** | Visible time window per display page |
| **HF / LF / Notch** | Toggle High-pass, Low-pass, and 50/60 Hz Notch filters |
| **`Ctrl+O`** | Open EEG file dialog |
| **`Ctrl+P`** | Toggle channel visibility side panel |
| **`F11`** | Toggle Fullscreen mode |
| **`Ctrl+Q`** | Quit application |

---

## 🐧 Cross-Platform & Hardware Tuning

The viewer auto-detects hardware performance profiles at runtime (`config.py:detect_platform()`):

- **Desktop (Windows / Linux):** Runs at 60 FPS with OpenGL hardware acceleration enabled (< 2% CPU usage).
- **Raspberry Pi (ARM / Embedded):** Cap capped at 30 FPS with software rendering enabled to ensure stability against Mesa GL quirks (< 5% CPU usage on Pi 4).

---

## 🧪 Running Tests

Run the automated test suite powered by `pytest`:

```bash
python run_tests.py
```

Covered test suites include filter responsiveness, SPSC ring-buffer thread safety, playback timing, spectral density algorithms, and UI state persistence.

---

## 🔌 Connecting Live EEG Hardware

`LiveLslSource` in `backend/live_lsl.py` is pre-wired for **Lab Streaming Layer (LSL)**. To acquire live data from hardware amplifiers (e.g., OpenBCI, BrainProducts, g.tec):

```bash
pip install pylsl
```

Uncomment the `pylsl.StreamInlet` pull loop in `live_lsl.py`. The canvas, ring buffer, filtering, and DSA will seamlessly render the incoming live stream.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
