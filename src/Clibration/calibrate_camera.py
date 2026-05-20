r"""
Camera Calibration Script — WSLg compatible version
=====================================================
Since cv2.imshow + cv2.waitKey has issues in WSLg/Qt,
this version works differently:

  - Saves a preview JPG to Output/preview.jpg (watch this to align)
  - Saves a verification JPG to Output/last_capture.jpg (check this after ENTER)
  - Open these files in Windows Explorer to see the view:
      \\wsl$\Ubuntu\home\jannat\sdc_2026\Output\
  - Type commands in the TERMINAL (not a popup window):
      ENTER = capture current frame
      s     = skip
      q     = quit and compute calibration
"""

import cv2
import numpy as np
import json
import time
import threading
import sys
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
# CAMERA PORT MAPPING (from New_control.py):
#   - RIGHT camera:  index 0   ← Change CAMERA_INDEX to 0
#   - MIDDLE camera: index 2   ← Change CAMERA_INDEX to 2
#   - LEFT camera:   index 4   ← Change CAMERA_INDEX to 4
#
# WHICH CAMERA TO CALIBRATE? (edit line below)
CAMERA_INDEX   = 2             # ← 0=RIGHT, 2=MIDDLE, 4=LEFT

CAMERA_NAMES   = {0: "RIGHT", 2: "MIDDLE", 4: "LEFT"}
CAMERA_NAME    = CAMERA_NAMES.get(CAMERA_INDEX, f"UNKNOWN_{CAMERA_INDEX}")

SQUARE_SIZE_MM = 21.0         # ← 21mm squares (matches printed calibration_checkerboard.pdf)
BOARD_W        = 9            # inner corners horizontally
BOARD_H        = 6            # inner corners vertically
MIN_CAPTURES   = 25
OUTPUT_DIR     = "/home/jannat/sdc_2026/Output"
OUTPUT_FILE    = f"{OUTPUT_DIR}/camera_calibration_{CAMERA_NAME}.json"
PREVIEW_FILE   = f"{OUTPUT_DIR}/preview_{CAMERA_NAME}.jpg"
LAST_SNAP_VIEW = f"{OUTPUT_DIR}/last_capture_{CAMERA_NAME}.jpg"

# ── SETUP ───────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

board_size   = (BOARD_W, BOARD_H)
object_point = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
object_point[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
object_point *= SQUARE_SIZE_MM

obj_points = []
img_points = []
criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ── CAMERA ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"🎥 CALIBRATING: {CAMERA_NAME} CAMERA (index {CAMERA_INDEX})")
print(f"{'='*60}\n")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open {CAMERA_NAME} camera at index {CAMERA_INDEX}\n"
                      f"Check New_control.py:\n"
                      f"  - RIGHT:  index 0\n"
                      f"  - MIDDLE: index 2\n"
                      f"  - LEFT:   index 4")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── SHARED STATE ─────────────────────────────────────────────────────────────
state = {
    "latest_frame":   None,
    "latest_display": None,
    "board_found":    False,
    "corners":        None,
    "lock":           threading.Lock(),
    "n_captures":     0,
    "command":        None,
    "running":        True,
}

# ── INPUT THREAD — reads keyboard from terminal ───────────────────────────
def input_thread(state):
    print("\nTerminal commands (type here, press ENTER):")
    print("  [ENTER] = capture frame")
    print("  s       = skip")
    print("  q       = quit and compute\n")
    while state["running"]:
        try:
            cmd = input()
            cmd = cmd.strip().lower()
            with state["lock"]:
                if cmd == "q":
                    state["command"] = "quit"
                elif cmd == "s":
                    state["command"] = "skip"
                else:
                    state["command"] = "capture"
        except EOFError:
            break

t_input = threading.Thread(target=input_thread, args=(state,), daemon=True)
t_input.start()

# ── CAMERA THREAD — reads frames continuously ────────────────────────────
def camera_thread(state, cap):
    while state["running"]:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        display = frame.copy()
        corners_refined = None
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, board_size, corners_refined, found)
        with state["lock"]:
            state["latest_frame"]   = frame.copy()
            state["latest_display"] = display.copy()
            state["board_found"]    = found
            state["corners"]        = corners_refined

t_cam = threading.Thread(target=camera_thread, args=(state, cap), daemon=True)
t_cam.start()

# ── MAIN LOOP ────────────────────────────────────────────────────────────
print(f"\n📷 Camera: {CAMERA_NAME} (index {CAMERA_INDEX})")
print(f"📏 Resolution: {actual_w}x{actual_h}")
print(f"🎯 Workflow: 1. Watch preview_{CAMERA_NAME}.jpg | 2. Press ENTER | 3. Verify last_capture_{CAMERA_NAME}.jpg")
print(f"💾 Output will be saved to: camera_calibration_{CAMERA_NAME}.json")

last_preview_save = 0
n_captures = 0

while True:
    time.sleep(0.05)

    with state["lock"]:
        display = state["latest_display"]
        frame   = state["latest_frame"]
        found   = state["board_found"]
        corners = state["corners"]
        command = state["command"]
        state["command"] = None

    # 1. Save live preview JPG (approx 2Hz for performance)
    now = time.time()
    if display is not None and now - last_preview_save > 0.4:
        preview = display.copy()
        status  = f"[{n_captures}/{MIN_CAPTURES}] {'READY - Press ENTER' if found else 'SEARCHING FOR BOARD...'}"
        color   = (0, 255, 0) if found else (0, 0, 255)
        # Overlay status bar
        cv2.rectangle(preview, (0, 0), (actual_w, 45), (0, 0, 0), -1)
        cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(PREVIEW_FILE, preview)
        last_preview_save = now

    # 2. Handle terminal commands
    if command == "quit":
        if n_captures >= 4:
            print(f"\nQuitting — computing calibration from {n_captures} captures...")
            break
        else:
            print(f"Need at least 4 captures to calculate anything. You have {n_captures}.")

    elif command == "capture":
        if frame is None:
            print("❌ No camera frame received yet.")
        elif not found:
            print("❌ Capture failed: No checkerboard detected in this frame.")
        else:
            obj_points.append(object_point)
            img_points.append(corners)
            n_captures += 1
            
            # Save high-res numbered version
            snap_path = f"{OUTPUT_DIR}/capture_{n_captures:02d}.jpg"
            cv2.imwrite(snap_path, display)
            
            # Save "Latest" version for quick viewing
            cv2.imwrite(LAST_SNAP_VIEW, display)
            
            print(f"✅ Captured {n_captures}/{MIN_CAPTURES} -> saved to Output folder.")

    elif command == "skip":
        print("Skipped.")

state["running"] = False
cap.release()

# ── COMPUTE CALIBRATION ──────────────────────────────────────────────────
print(f"\nComputing calibration math...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (actual_w, actual_h), None, None)

print(f"\n{'='*50}")
print(f"Reprojection error: {ret:.4f} px")
print(f"Interpretation: {'Excellent' if ret < 0.5 else 'Good' if ret < 1.0 else 'High Error - try again'}")
print(f"{'='*50}")

# Extract focal length and FOV
fx, fy   = K[0, 0], K[1, 1]
cx, cy   = K[0, 2], K[1, 2]
hfov_deg = 2 * np.degrees(np.arctan(actual_w / (2 * fx)))
vfov_deg = 2 * np.degrees(np.arctan(actual_h / (2 * fy)))

# Save to JSON
output = {
    "camera_metadata": "WSL Calibration Session",
    "resolution": [actual_w, actual_h],
    "date": datetime.now().isoformat(),
    "reprojection_error_px": float(ret),
    "K": K.tolist(),
    "dist": dist.ravel().tolist(),
    "derived": {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "hfov_deg": float(hfov_deg),
        "vfov_deg": float(vfov_deg),
    }
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ SUCCESS! {CAMERA_NAME} camera calibration complete")
print(f"{'='*60}")
print(f"\n📁 Output files:")
print(f"   Camera Matrix (K): {OUTPUT_FILE}")
print(f"   Captures: {OUTPUT_DIR}/capture_*.jpg")
print(f"\n📊 Reprojection Error: {ret:.4f} px")
print(f"   {'✓ Excellent' if ret < 0.5 else '✓ Good' if ret < 1.0 else '⚠ High - consider recalibrating'}")
print(f"\n🎯 Next steps:")
print(f"   1. Calibrate MIDDLE camera (if not done): CAMERA_INDEX = 2")
print(f"   2. Calibrate LEFT camera (if needed):    CAMERA_INDEX = 4")
print(f"   3. Calibrate RIGHT camera (if needed):   CAMERA_INDEX = 0")