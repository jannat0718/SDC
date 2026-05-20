"""
Camera Calibration using Checkerboard Pattern
Captures images only when checkerboard is detected
Calculates intrinsic parameters and saves to JSON
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKERBOARD_SIZE = (9, 6)  # Inner corners (9 width, 6 height) - ADJUST THIS!
SQUARE_SIZE = 0.021         # 2.1cm (21mm) per square - matches printed checkerboard
NUM_IMAGES_TO_CAPTURE = 20  # Number of images to capture
OUTPUT_DIR = "/home/jannat/sdc_2026/src/calibration_images"
OUTPUT_JSON = "/home/jannat/sdc_2026/src/camera_calibration.json"

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"📁 Calibration images will be saved to: {OUTPUT_DIR}")

# ============================================================================
# SETUP CALIBRATION CRITERIA
# ============================================================================

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

print("\n" + "=" * 70)
print("CAMERA CALIBRATION - CHECKERBOARD DETECTION")
print("=" * 70)
print(f"\n⚙️  Settings:")
print(f"  Checkerboard size: {CHECKERBOARD_SIZE[0]}×{CHECKERBOARD_SIZE[1]} inner corners")
print(f"  Square size: {SQUARE_SIZE*100:.1f} cm")
print(f"  Target images: {NUM_IMAGES_TO_CAPTURE}")
print(f"\n📸 Instructions:")
print(f"  1. Position the checkerboard in front of the camera")
print(f"  2. When you see 'DETECTED' and green corners, press SPACE to capture")
print(f"  3. Move the checkerboard to a different angle/distance")
print(f"  4. Repeat until {NUM_IMAGES_TO_CAPTURE} images are captured")
print(f"  5. Press 'q' to finish and calibrate")
print("\n" + "=" * 70 + "\n")

# ============================================================================
# OPEN CAMERA
# ============================================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    exit(1)

# Set camera resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Camera opened: {width}×{height} @ {fps} fps\n")

# ============================================================================
# CAPTURE LOOP
# ============================================================================

captured_count = 0
frame_count = 0

cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)

while captured_count < NUM_IMAGES_TO_CAPTURE:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    ret_find, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD_SIZE, None,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Draw frame info
    display_frame = frame.copy()

    if ret_find:
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw detected corners (green)
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD_SIZE, corners_refined, ret_find)

        # Draw status text
        cv2.putText(display_frame, "✓ DETECTED - Press SPACE to capture", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {captured_count}/{NUM_IMAGES_TO_CAPTURE}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = "READY TO CAPTURE"
        status_color = (0, 255, 0)

    else:
        # Draw status text (red)
        cv2.putText(display_frame, "✗ No checkerboard detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Captured: {captured_count}/{NUM_IMAGES_TO_CAPTURE}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        status = "NOT DETECTED"
        status_color = (0, 0, 255)

    cv2.putText(display_frame, f"Frame: {frame_count}", (20, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Show frame
    cv2.imshow("Camera Calibration", display_frame)

    # Handle key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and ret_find:  # SPACE to capture
        # Save image
        img_path = os.path.join(OUTPUT_DIR, f"calibration_{captured_count:02d}.png")
        cv2.imwrite(img_path, frame)

        # Store object and image points
        objpoints.append(objp)
        imgpoints.append(corners_refined)

        captured_count += 1
        print(f"✅ Captured {captured_count}/{NUM_IMAGES_TO_CAPTURE}: {img_path}")

    elif key == ord('q'):  # Q to quit
        if captured_count < NUM_IMAGES_TO_CAPTURE:
            response = input(f"\n⚠️  Only captured {captured_count}/{NUM_IMAGES_TO_CAPTURE}. Proceed anyway? (y/n): ")
            if response.lower() != 'y':
                continue
        break

cap.release()
cv2.destroyAllWindows()

# ============================================================================
# CALIBRATION
# ============================================================================

if len(objpoints) == 0:
    print("\n❌ ERROR: No images captured. Exiting.")
    exit(1)

print("\n" + "=" * 70)
print("RUNNING CALIBRATION...")
print("=" * 70)

# Run calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if not ret:
    print("❌ Calibration failed!")
    exit(1)

print("✅ Calibration successful!\n")

# ============================================================================
# EXTRACT PARAMETERS
# ============================================================================

fx = float(K[0, 0])
fy = float(K[1, 1])
cx = float(K[0, 2])
cy = float(K[1, 2])

# Distortion coefficients
k1 = float(dist[0, 0])
k2 = float(dist[0, 1])
p1 = float(dist[0, 2])
p2 = float(dist[0, 3])
k3 = float(dist[0, 4]) if dist.shape[1] > 4 else 0.0

# ============================================================================
# CALCULATE REPROJECTION ERROR
# ============================================================================

mean_error = 0
total_points = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    total_points += len(imgpoints2)

mean_error /= len(objpoints)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("📊 CALIBRATION RESULTS:")
print(f"\n  Focal length X (fx):  {fx:.2f} px")
print(f"  Focal length Y (fy):  {fy:.2f} px")
print(f"  Principal point X (cx): {cx:.2f} px")
print(f"  Principal point Y (cy): {cy:.2f} px")
print(f"\n  Distortion coefficients:")
print(f"    k1: {k1:.6f}")
print(f"    k2: {k2:.6f}")
print(f"    p1: {p1:.6f}")
print(f"    p2: {p2:.6f}")
print(f"    k3: {k3:.6f}")
print(f"\n  Reprojection error: {mean_error:.4f} pixels")
print(f"  Images used: {len(objpoints)}")

# ============================================================================
# SAVE TO JSON
# ============================================================================

calibration_data = {
    "image_width": width,
    "image_height": height,
    "checkerboard_size": list(CHECKERBOARD_SIZE),
    "square_size_m": SQUARE_SIZE,
    "num_images": len(objpoints),
    "reprojection_error_px": round(mean_error, 4),
    "intrinsic_matrix": {
        "fx": round(fx, 2),
        "fy": round(fy, 2),
        "cx": round(cx, 2),
        "cy": round(cy, 2)
    },
    "distortion_coefficients": {
        "k1": round(k1, 6),
        "k2": round(k2, 6),
        "p1": round(p1, 6),
        "p2": round(p2, 6),
        "k3": round(k3, 6)
    },
    "K_matrix": K.tolist(),
    "dist_matrix": dist.tolist()
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(calibration_data, f, indent=2)

print(f"\n✅ Calibration saved to: {OUTPUT_JSON}")

# ============================================================================
# VERIFY BY SHOWING JSON CONTENT
# ============================================================================

print("\n" + "=" * 70)
print("SAVED CALIBRATION:")
print("=" * 70)
print(json.dumps(calibration_data, indent=2))
print("\n" + "=" * 70)
