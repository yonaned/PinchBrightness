import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks
import subprocess
import numpy as np
import math
import time


def set_brightness(percent: int):
    """Set screen brightness using brightnessctl (Linux)."""
    percent = max(1, min(100, percent))
    try:
        subprocess.run(
            ['brightnessctl', 's', f'{percent}%'],
            capture_output=True, timeout=1
        )
    except FileNotFoundError:
        print("brightnessctl not found — run: sudo apt install brightnessctl")
    except Exception as e:
        print(f"Brightness error: {e}")

def get_brightness() -> int:
    """Read current brightness as a percentage."""
    try:
        cur  = subprocess.run(['brightnessctl', 'g'], capture_output=True, text=True, timeout=1)
        top  = subprocess.run(['brightnessctl', 'm'], capture_output=True, text=True, timeout=1)
        return int((int(cur.stdout.strip()) / int(top.stdout.strip())) * 100)
    except:
        return 50   # safe fallback



def pixel_dist(a, b, w, h) -> float:
    """Euclidean distance between two normalized landmarks in pixel space."""
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)



HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def draw_hand(frame, detection_result):
    """Draw skeleton and highlight thumb + index tips."""
    if not detection_result.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand in detection_result.hand_landmarks:
        # Skeleton
        for conn in HAND_CONNECTIONS:
            s, e = hand[conn.start], hand[conn.end]
            cv2.line(frame,
                     (int(s.x * w), int(s.y * h)),
                     (int(e.x * w), int(e.y * h)),
                     (0, 0, 200), 2)
        # Landmarks
        for i, lm in enumerate(hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if i in (4, 8):                                # thumb & index tips
                cv2.circle(frame, (cx, cy), 12, (255, 80, 0), cv2.FILLED)
            else:
                cv2.circle(frame, (cx, cy), 4,  (0, 220, 0), cv2.FILLED)

def draw_pinch_line(frame, thumb, index, w, h, brightness: int):
    """Draw the pinch line and percentage between fingers."""
    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index.x * w), int(index.y * h)
    cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 2)
    mid_x, mid_y = (tx + ix) // 2, (ty + iy) // 2
    cv2.putText(frame, f'{brightness}%',
                (mid_x - 20, mid_y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def draw_brightness_bar(frame, brightness: int, active: bool):
    """Vertical brightness bar on the right side of the frame."""
    fh, fw, _ = frame.shape
    bx       = fw - 55
    bar_h    = 280
    bar_top  = (fh - bar_h) // 2
    bar_bot  = bar_top + bar_h

    # Background
    cv2.rectangle(frame, (bx, bar_top), (bx + 28, bar_bot), (40, 40, 40), -1)
    # Fill
    fill     = int(bar_h * brightness / 100)
    color    = (0, 255, 255) if active else (120, 120, 120)
    cv2.rectangle(frame, (bx, bar_bot - fill), (bx + 28, bar_bot), color, -1)
    # Border
    cv2.rectangle(frame, (bx, bar_top), (bx + 28, bar_bot), (220, 220, 220), 2)
    # Labels
    cv2.putText(frame, f'{brightness}%', (bx - 8, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, 'BRT', (bx + 1, bar_bot + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def draw_hud(frame, active: bool):
    """Top-left status hint."""
    status = "CONTROLLING BRIGHTNESS" if active else "Show hand to control brightness"
    color  = (0, 255, 180) if active else (180, 180, 180)
    cv2.putText(frame, status, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, "Pinch: thumb + index  |  Q = quit", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)



options = HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap              = cv2.VideoCapture(0)
smooth_brightness = float(get_brightness())   # start from current system brightness
last_set_value   = -1                         # avoid redundant syscalls
start_time       = time.time()

print("Hand brightness controller started. Pinch thumb + index to adjust. Press Q to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame     = cv2.flip(frame, 1)
        h, w, _   = frame.shape
        ts_ms     = int((time.time() - start_time) * 1000)   # monotonic ms

        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results   = landmarker.detect_for_video(mp_img, ts_ms)

        draw_hand(frame, results)

        hand_active = bool(results.hand_landmarks)

        if hand_active:
            hand       = results.hand_landmarks[0]
            thumb_tip  = hand[4]   # THUMB_TIP
            index_tip  = hand[8]   # INDEX_FINGER_TIP
            wrist      = hand[0]   # WRIST
            mid_base   = hand[9]   # MIDDLE_FINGER_MCP  — good proxy for palm size

            pinch_dist = pixel_dist(thumb_tip, index_tip, w, h)
            hand_size  = pixel_dist(wrist, mid_base, w, h)

            if hand_size > 0:
                # Normalize pinch by palm size so distance to camera doesn't matter.
                # Ratio ≈ 0.1 (full pinch) → 1.3 (fully spread)
                ratio      = pinch_dist / hand_size
                target_br  = int(np.interp(ratio, [0.15, 1.25], [1, 100]))
                target_br  = max(1, min(100, target_br))

                # Exponential moving average — smooths jitter without lag
                smooth_brightness = smooth_brightness * 0.75 + target_br * 0.25

            draw_pinch_line(frame, thumb_tip, index_tip, w, h, int(smooth_brightness))

        # Only write to the system when value actually changes (reduces overhead)
        rounded = int(smooth_brightness)
        if rounded != last_set_value:
            set_brightness(rounded)
            last_set_value = rounded

        draw_brightness_bar(frame, rounded, hand_active)
        draw_hud(frame, hand_active)

        cv2.imshow('Hand Brightness Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()