# validate_homography_at_home.py
import cv2
import numpy as np
import glob

# ============================================
# LOAD BIRD EYE VIEW CALIBRATION (from bird_eye_view.py)
# ============================================

# Zebra stripe corners in STITCHED CANVAS (from bird_eye_view.py lines 18-22)
src_pts = np.float32([
    [1559, 565],   # Top-Left
    [1693, 573],   # Top-Right
    [1181, 948],   # Bottom-Right
    [739, 907]     # Bottom-Left
])

# Real-world zebra dimensions (from bird_eye_view.py)
STRIPE_WIDTH = 60    # cm
STRIPE_LENGTH = 300  # cm

BEV_WIDTH = 800
BEV_HEIGHT = 1200
X_CENTER = BEV_WIDTH // 2
Y_BOTTOM = 1000

# Destination points (what the zebra SHOULD look like in real world)
# These are in pixel coordinates within the BEV image
dst_pts = np.float32([
    [X_CENTER - (STRIPE_WIDTH // 2), Y_BOTTOM - STRIPE_LENGTH],  # Top-Left
    [X_CENTER + (STRIPE_WIDTH // 2), Y_BOTTOM - STRIPE_LENGTH],  # Top-Right
    [X_CENTER + (STRIPE_WIDTH // 2), Y_BOTTOM],                  # Bottom-Right
    [X_CENTER - (STRIPE_WIDTH // 2), Y_BOTTOM]                   # Bottom-Left
])

# Get the M_bev homography matrix
M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("=" * 60)
print("HOMOGRAPHY VALIDATION AT HOME")
print("=" * 60)
print("\nExpected zebra stripe dimensions:")
print(f"  Width:  {STRIPE_WIDTH} cm")
print(f"  Length: {STRIPE_LENGTH} cm")
print("\nDestination points in BEV image (pixels):")
for i, pt in enumerate(dst_pts):
    print(f"  Corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")

# Calculate pixel distances in BEV image
width_px = dst_pts[1][0] - dst_pts[0][0]
length_px = dst_pts[3][1] - dst_pts[0][1]
print(f"\nBEV image dimensions:")
print(f"  Width (pixels):  {width_px:.1f}")
print(f"  Length (pixels): {length_px:.1f}")

# Scale: cm per pixel in BEV image
scale_cm_per_px_w = STRIPE_WIDTH / width_px
scale_cm_per_px_l = STRIPE_LENGTH / length_px
print(f"\nScale factors:")
print(f"  Width:  {scale_cm_per_px_w:.4f} cm/pixel")
print(f"  Length: {scale_cm_per_px_l:.4f} cm/pixel")

# ============================================
# LOAD A TEST VIDEO
# ============================================

video_files = glob.glob('/home/jannat/sdc_2026/data/Middle20260226133416.mp4')
if not video_files:
    print("\n❌ ERROR: No video files found in /Output/")
    exit(1)

video_path = video_files[0]
print(f"\n✅ Loading video: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Could not open video")
    exit(1)

# Get a frame at a specific timestamp: 1 minute 54 seconds
fps = cap.get(cv2.CAP_PROP_FPS)
timestamp_seconds = 1 * 60 + 54  # 1:54 = 114 seconds
frame_idx = int(timestamp_seconds * fps)

print(f"FPS: {fps}")
print(f"Timestamp: {timestamp_seconds} seconds")
print(f"Frame number: {frame_idx}")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()

# ============================================
# INTERACTIVE: Mark the 4 zebra corners
# ============================================

corners = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        print(f"Marked corner {len(corners)}: ({x}, {y})")
        
        # Draw circles on the image
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
        cv2.imshow('Mark Zebra Stripe Corners', frame)

print("\n" + "=" * 60)
print("INTERACTIVE: Mark the 4 corners of the zebra stripe")
print("=" * 60)
print("\nInstructions:")
print("  1. Look at the video frame below")
print("  2. Find the zebra crossing (white/yellow stripes)")
print("  3. Click on the TOP-LEFT corner")
print("  4. Click on the TOP-RIGHT corner")
print("  5. Click on the BOTTOM-RIGHT corner")
print("  6. Click on the BOTTOM-LEFT corner")
print("  7. Press 'q' when done")
print("\n(If you don't see a zebra stripe, 'q' and pick a different frame)")

cv2.namedWindow('Mark Zebra Stripe Corners')
cv2.setMouseCallback('Mark Zebra Stripe Corners', mouse_callback)
cv2.imshow('Mark Zebra Stripe Corners', frame)

while len(corners) < 4:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n⚠️  You pressed 'q' before marking 4 corners. Exiting.")
        cv2.destroyAllWindows()
        exit(1)

cv2.destroyAllWindows()

corners = np.float32(corners)
print(f"\n✅ Marked 4 corners:")
for i, c in enumerate(corners):
    print(f"  Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")

# ============================================
# APPLY M_bev TO CONVERT TO BEV COORDINATES
# ============================================

print("\n" + "=" * 60)
print("APPLYING HOMOGRAPHY")
print("=" * 60)

# Transform the marked corners using M_bev
bev_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M_bev)
bev_corners = bev_corners.reshape(-1, 2)

print(f"\n✅ Transformed corners (in BEV image):")
for i, c in enumerate(bev_corners):
    print(f"  Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")

# ============================================
# MEASURE DISTANCES IN BEV IMAGE
# ============================================

# Width: distance between top-left and top-right
measured_width_px = bev_corners[1][0] - bev_corners[0][0]
measured_width_cm = measured_width_px * scale_cm_per_px_w

# Length: distance between top-left and bottom-left
measured_length_px = bev_corners[3][1] - bev_corners[0][1]
measured_length_cm = measured_length_px * scale_cm_per_px_l

print(f"\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\n📏 WIDTH (cm):")
print(f"  Expected:   {STRIPE_WIDTH} cm")
print(f"  Measured:   {measured_width_cm:.1f} cm")
print(f"  Error:      {abs(STRIPE_WIDTH - measured_width_cm):.1f} cm ({abs(STRIPE_WIDTH - measured_width_cm) / STRIPE_WIDTH * 100:.1f}%)")

print(f"\n📏 LENGTH (cm):")
print(f"  Expected:   {STRIPE_LENGTH} cm")
print(f"  Measured:   {measured_length_cm:.1f} cm")
print(f"  Error:      {abs(STRIPE_LENGTH - measured_length_cm):.1f} cm ({abs(STRIPE_LENGTH - measured_length_cm) / STRIPE_LENGTH * 100:.1f}%)")

# ============================================
# VERDICT
# ============================================

width_error_pct = abs(STRIPE_WIDTH - measured_width_cm) / STRIPE_WIDTH * 100
length_error_pct = abs(STRIPE_LENGTH - measured_length_cm) / STRIPE_LENGTH * 100

print(f"\n" + "=" * 60)
if width_error_pct < 5 and length_error_pct < 5:
    print("✅ HOMOGRAPHY IS VALID (error < 5%)")
    print("   The M_bev matrix is correctly calibrated!")
elif width_error_pct < 15 and length_error_pct < 15:
    print("⚠️  HOMOGRAPHY IS ROUGHLY CORRECT (error 5-15%)")
    print("   Acceptable for validation, but consider recalibration")
else:
    print("❌ HOMOGRAPHY IS INVALID (error > 15%)")
    print("   Zebra stripe calibration is wrong!")
    print("   → Check if src_pts and/or STRIPE_WIDTH/LENGTH are incorrect")

print("=" * 60)

cap.release()
