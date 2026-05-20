# recalibrate_zebra_corners.py
import cv2
import numpy as np
import glob
import os

# Find the middle camera video
video_files = glob.glob('/home/jannat/sdc_2026/data/Middle20260226133416.mp4')
if not video_files:
    print("❌ No Middle*.mp4 video found in /home/jannat/sdc_2026/Output/")
    exit(1)

video_path = video_files[0]
print(f"📹 Opening: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps

print(f"Video FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration_seconds/60:.2f} minutes")

# ========== USE 30 FPS × 114 SECONDS ==========
ASSUMED_FPS = 30
timestamp_seconds = 1 * 60 + 54  # 114 seconds
frame_idx = ASSUMED_FPS * timestamp_seconds  # 30 × 114 = 3420

print(f"\n🎯 Using assumed FPS: {ASSUMED_FPS}")
print(f"🎯 Timestamp: {timestamp_seconds//60}:{timestamp_seconds%60:02d} ({timestamp_seconds} seconds)")
print(f"🎯 Target frame: {frame_idx} (30 fps × 114 sec)")

# Check if frame index is valid
if frame_idx >= total_frames:
    print(f"⚠️  Frame {frame_idx} exceeds total frames ({total_frames})")
    print(f"📌 Video actual FPS: {fps:.2f}")
    print(f"📌 Using actual FPS calculation instead...")
    frame_idx = int(fps * timestamp_seconds)
    print(f"📌 New target frame: {frame_idx}")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()

# If still can't read, try nearby frames
attempt = 0
while not ret and attempt < 30:
    frame_idx += 1  # Try next frame
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    attempt += 1
    print(f"🔄 Retry {attempt}: trying frame {frame_idx}")

if not ret or frame is None:
    print("❌ Could not read any frame from video")
    cap.release()
    exit(1)

# Get video resolution
h, w = frame.shape[:2]
print(f"\n✅ Loaded frame {frame_idx} | Resolution: {w}×{h}")
print("📸 Click the 4 corners of the zebra stripe (TL → TR → BR → BL)")
print("📸 Press SPACE to pause/zoom, 'r' to reset, 'q' when done\n")

# Store corners
corners = []
paused = False
original_frame = frame.copy()

def mouse_callback(event, x, y, flags, param):
    global corners, frame
    if event == cv2.EVENT_LBUTTONDOWN and not paused:
        corners.append([x, y])
        print(f"✓ Corner {len(corners)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
        cv2.putText(frame, str(len(corners)), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Click Zebra Corners', frame)

# Create window and set callback
window_name = 'Click Zebra Corners (TL, TR, BR, BL)'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

def reset_corners():
    global corners, frame
    corners = []
    frame = original_frame.copy()
    print("\n🔄 Corners reset. Start over.")
    cv2.imshow(window_name, frame)

# Display initial frame
cv2.imshow(window_name, frame)

while True:
    key = cv2.waitKey(1) & 0xFF

    # Show status message every few iterations
    if len(corners) > 0 and len(corners) < 4:
        remaining = 4 - len(corners)
        print(f"[Status] {len(corners)}/4 corners marked. {remaining} more to go...", end='\r')
    elif len(corners) == 4:
        print(f"\n✅ 4 corners marked! Press 'q' to confirm and see results")

    if key == ord('q'):
        if len(corners) < 4:
            print(f"\n⚠️  Only {len(corners)} corners selected. Need 4.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                corners = []
        else:
            print(f"\n✅ Confirmed! Processing...")
        break
    
    elif key == ord('r'):
        reset_corners()
    
    elif key == ord(' '):  # Space bar to pause/resume
        paused = not paused
        status = "PAUSED" if paused else "RESUMED"
        print(f"\n⏸️ {status}")
        if paused and len(corners) > 0:
            # Zoom on last clicked corner
            last = corners[-1]
            zoom_size = 200
            h, w = frame.shape[:2]
            x1 = max(0, last[0] - zoom_size)
            y1 = max(0, last[1] - zoom_size)
            x2 = min(w, last[0] + zoom_size)
            y2 = min(h, last[1] + zoom_size)
            zoom = frame[y1:y2, x1:x2]
            if zoom.size > 0:
                zoom = cv2.resize(zoom, (400, 400))
                cv2.circle(zoom, (200, 200), 5, (0, 0, 255), -1)
                cv2.imshow('Zoom (SPACE to resume)', zoom)
        elif not paused:
            cv2.destroyWindow('Zoom (SPACE to resume)')

cv2.destroyAllWindows()
cap.release()

# Show results
if len(corners) == 4:
    print("\n" + "=" * 60)
    print("✅ YOUR NEW src_pts for bird_eye_view.py:")
    print("=" * 60)
    print()

    # Prepare output
    output_text = "src_pts = np.float32([\n"
    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    print("src_pts = np.float32([")
    for i, c in enumerate(corners):
        line = f"    [{c[0]}, {c[1]}],  # {labels[i]}\n"
        print(line.rstrip())
        output_text += line
    print("])")
    output_text += "])\n"

    # Calculate sanity checks
    tl, tr, br, bl = corners
    top_width = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    bottom_width = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    left_height = np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[1])**2)
    right_height = np.sqrt((br[0]-tr[0])**2 + (br[1]-tr[1])**2)

    print(f"\n📏 Sanity checks:")
    print(f"  Top width:    {top_width:.1f} px")
    print(f"  Bottom width: {bottom_width:.1f} px")
    print(f"  Left height:  {left_height:.1f} px")
    print(f"  Right height: {right_height:.1f} px")

    # Save to file
    output_file = '/home/jannat/sdc_2026/src/nevigation/NEW_SRC_PTS.txt'
    with open(output_file, 'w') as f:
        f.write(output_text)

    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Open bird_eye_view.py (line 23)")
    print("2. Replace the old src_pts with the above")
    print("3. Run: python3 validate_homography_at_home.py")
    print("=" * 60)

else:
    print(f"\n❌ Calibration incomplete. ({len(corners)}/4 corners marked)")