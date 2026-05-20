import cv2
import numpy as np
import glob
import json
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
BOARD_W = 9
BOARD_H = 6
SQUARE_SIZE_MM = 21.0         # 21mm squares (matches printed calibration_checkerboard.pdf)
INPUT_DIR = "/home/jannat/sdc_2026/Output/Clibration"
OUTPUT_FILE = f"{INPUT_DIR}/camera_K_compute_only.json"

board_size = (BOARD_W, BOARD_H)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (30,0,0), (60,0,0) ...
objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE_MM

obj_points = [] 
img_points = [] 

# Get all images in the folder
images = glob.glob(f"{INPUT_DIR}/*.jpg")
images = [img for img in images if "preview" not in img and "last_capture" not in img]

print(f"Found {len(images)} images. Processing...")

valid_count = 0
shape = None

for fname in sorted(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        valid_count += 1
        print(f"✅ {os.path.basename(fname)}: Corners found")
    else:
        print(f"❌ {os.path.basename(fname)}: Could not find corners")

if valid_count < 4:
    print("\nError: Not enough valid images left to calibrate!")
else:
    print(f"\nComputing calibration on {valid_count} images...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None)

    # Save results
    output = {
        "reprojection_error": float(ret),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "images_used": valid_count
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Reprojection Error: {ret:.4f}")
    print(f"Results saved to {OUTPUT_FILE}")