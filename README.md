# PinchBrightness

Control your screen brightness in real-time using just your hand and webcam no buttons, no sliders. Pinch your thumb and index finger together to dim the screen, spread them apart to brighten it.

Built with **MediaPipe Tasks API** and **OpenCV** on Python.

---

## Demo

| Gesture | Result |
|---|---|
|  Pinch fingers together | Brightness → low |
|  Spread fingers apart | Brightness → high |
|  No hand visible | Brightness holds at last value |

---

## How It Works

1. Your webcam feed is processed frame-by-frame using MediaPipe's `HandLandmarker`.
2. The distance between your **thumb tip** (landmark 4) and **index finger tip** (landmark 8) is measured.
3. That distance is **normalized by your palm size** — so moving closer or further from the camera doesn't accidentally change brightness.
4. The normalized ratio is mapped to a `1–100%` brightness range.
5. An **exponential moving average** smooths out hand jitter before writing to the system.
6. `brightnessctl` applies the value to your display hardware.

---

## Requirements

### System

- Linux (Ubuntu 20.04+ recommended)
- Python 3.9–3.12
- A working webcam
- `brightnessctl` installed

```bash
sudo apt install brightnessctl

# Allow brightness control without sudo
sudo usermod -aG video $USER
# Log out and back in after this
```

### Python dependencies

```bash
pip install mediapipe opencv-python numpy
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/handtraker.git
cd handtraker
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install mediapipe opencv-python numpy
```

### 4. Download the MediaPipe hand landmark model

```bash
wget -q --show-progress \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

The `.task` file must be in the **same directory as `main.py`**.

### 5. Run

```bash
python main.py
```

> !! Always run from inside the project folder so the model path resolves correctly.

---

## Project Structure

```
handtraker/
├── main.py                  # Main script
├── hand_landmarker.task     # MediaPipe model file (downloaded separately)
├── .venv/                   # Virtual environment (not committed)
└── README.md
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | You have MediaPipe 0.10+, which removed the old API. This project uses the new Tasks API — make sure you're on the latest code. |
| `Unable to open file at .../hand_landmarker.task` | The model file is missing. Re-run the `wget` command in step 4. |
| Brightness doesn't change | Run `brightnessctl` manually to confirm it works. Make sure your user is in the `video` group and you've re-logged in. |
| Black screen / no webcam | Check that no other app is holding the camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`. |
| Brightness changes too fast / jittery | Increase the EMA smoothing factor in `main.py` from `0.75` toward `0.90`. |

---

## Configuration

These values in `main.py` are easy to tune:

```python
# EMA smoothing — higher = smoother but slower response (0.0–1.0)
smooth_brightness = smooth_brightness * 0.75 + target_br * 0.25

# Pinch ratio range — adjust if your hand size triggers wrong range
target_br = int(np.interp(ratio, [0.15, 1.25], [1, 100]))

# Detection confidence
min_hand_detection_confidence=0.7,
min_tracking_confidence=0.5
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection (Tasks API) |
| `opencv-python` | Webcam capture and frame rendering |
| `numpy` | Value interpolation and math |
| `brightnessctl` | Linux display brightness control |

---

## License

MIT License — feel free to use, modify, and share.
