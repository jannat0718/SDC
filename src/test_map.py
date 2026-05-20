#!/usr/bin/env python3
"""
test_map.py — Fixed Landmark Verification
==========================================
Draws fixed landmarks (zebras, start/end lines) from nav_config onto
the track image. All coordinates printed to terminal only.

HOW TO RUN:
  cd /home/jannat/sdc_2026/src
  python3 test_map.py
"""

import os, sys, cv2, numpy as np, json

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.utils  import load_nav_config
from nevigation.av_map import TrackMapper, DetectedObject, Pose, OUTPUT_DIR

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_map_static.png")
CONFIG_PATH = os.path.join(project_root, "nevigation", "nav_config.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────
cfg      = load_nav_config(CONFIG_PATH)
with open(CONFIG_PATH) as f:
    cfg_raw = json.load(f)

landmarks = cfg_raw.get("fixed_landmarks", [])
start_px  = cfg_raw.get("start_px",   [7823, 720])
px_per_m  = cfg_raw.get("px_per_meter", 20.6869)

if not os.path.exists(cfg.track_image_path or ""):
    print(f"ERROR: Track image not found: {cfg.track_image_path}")
    sys.exit(1)

# ── Load track image at FULL resolution ──────────────────────────────────────
track_img = cv2.imread(cfg.track_image_path)
MAP_H, MAP_W = track_img.shape[:2]
print(f"Track image loaded: {MAP_W} x {MAP_H} px")

# ── Draw landmarks directly on full-res track image ──────────────────────────
# Type colours (BGR):
TYPE_COLORS = {
    "line":   (0,   255,   0),   # green  — start/end lines
    "zebra":  (0,   220, 255),   # yellow — zebra crossings
    "point":  (0,   255,   0),   # green  — single points
}
LABEL_COLORS = {
    "line":   (0,   255,   0),
    "zebra":  (0,   200, 220),
    "point":  (0,   255,   0),
}
BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

for lm in landmarks:
    name   = lm["name"]
    ltype  = lm.get("type", "point")
    color  = tuple(reversed(lm.get("color_bgr", [200,200,200])))  # RGB→BGR
    color  = tuple(int(c) for c in lm.get("color_bgr", [200,200,200]))

    if ltype in ("line", "zebra") and "pixel_box" in lm:
        box = lm["pixel_box"]
        x1, y1 = box["x_min"], box["y_min"]
        x2, y2 = box["x_max"], box["y_max"]

        if ltype == "line":
            # Use pixel_center x — guaranteed to align with start/end points
            xc = lm["pixel_center"][0]   # x=6687 for all start/end markers
            cv2.line(track_img, (xc, y1), (xc, y2), color, BOX_THICKNESS + 1)
            # Tick marks at top and bottom
            cv2.line(track_img, (xc-10, y1), (xc+10, y1), color, 2)
            cv2.line(track_img, (xc-10, y2), (xc+10, y2), color, 2)
            label_y = y1 - 6 if y1 > 20 else y2 + 18
            cv2.putText(track_img, name, (xc+8, label_y),
                        FONT, 0.55, color, 1, cv2.LINE_AA)

        elif ltype == "zebra":
            # Zebra crossings — semi-transparent filled box + border
            overlay = track_img.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
            cv2.addWeighted(overlay, 0.25, track_img, 0.75, 0, track_img)
            cv2.rectangle(track_img, (x1,y1), (x2,y2), color, BOX_THICKNESS)
            label_y = y1 - 6 if y1 > 20 else y2 + 18
            cv2.putText(track_img, name, (x1, label_y),
                        FONT, 0.55, color, 1, cv2.LINE_AA)

    elif ltype == "point" and "pixel_center" in lm:
        px, py = lm["pixel_center"]
        cv2.drawMarker(track_img, (px,py), color, cv2.MARKER_STAR, 22, 2)
        cv2.putText(track_img, name, (px+12, py-8),
                    FONT, 0.55, color, 1, cv2.LINE_AA)

# ── Also draw your 6 custom test points ──────────────────────────────────────
custom_pts = [
    ("Ob1", 1681, 686,  (0,255,255)),
    ("Ob2", 1607, 383,  (0,165,255)),
    ("Ob3", 1317, 917,  (255,0,255)),
    ("SS",  1326, 972,  (0,0,255)),
    ("P",   1382, 879,  (255,128,0)),
    ("K",   5353, 610,  (128,255,0)),
]
for name, px, py, color in custom_pts:
    cv2.circle(track_img, (px,py), 12, color, -1)
    cv2.circle(track_img, (px,py), 12, (255,255,255), 1)
    cv2.putText(track_img, name, (px+14, py+5),
                FONT, 0.5, color, 1, cv2.LINE_AA)

# ── Build slim legend strip BELOW track ──────────────────────────────────────
LEGEND_H = 60
legend   = np.full((LEGEND_H, MAP_W, 3), 25, dtype=np.uint8)
cv2.line(legend, (0,0), (MAP_W,0), (80,80,80), 2)

items = [
    ((0,255,0),   "■ Start/End line"),
    ((0,220,255), "■ Zebra crossing"),
    ((0,255,255), "● Ob1"),
    ((0,165,255), "● Ob2"),
    ((255,0,255), "● Ob3"),
    ((0,0,255),   "● SS"),
    ((255,128,0), "● P"),
    ((128,255,0), "● K"),
]
x_off = 30
for color, label in items:
    (tw,_),_ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.putText(legend, label, (x_off, 38), FONT, 0.5, color, 1, cv2.LINE_AA)
    x_off += tw + 35

combined = np.vstack([track_img, legend])
cv2.imwrite(OUTPUT_FILE, combined)

# ── Terminal output — all coordinates ────────────────────────────────────────
def px_to_world(px, py):
    return round((px-start_px[0])/px_per_m,3), round(-(py-start_px[1])/px_per_m,3)

def dist(wx,wy): return round(np.sqrt(wx**2+wy**2),2)

print("\n" + "="*75)
print("  FIXED LANDMARKS  (from nav_config.json)")
print("="*75)
print(f"{'Name':<16} {'Pixel center':>14}  {'World X':>10} {'World Y':>10}  {'Dist from START':>16}  Size(m)")
print("-"*75)
for lm in landmarks:
    pc  = lm.get("pixel_center", [0,0])
    wx, wy = lm.get("world_center", [0,0])
    d   = dist(wx, wy)
    sz  = lm.get("size_m", ["-","-"])
    sz_s= f"{float(sz[0]):.2f}x{float(sz[1]):.2f}" if isinstance(sz,(list,tuple)) and len(sz)==2 else "-"
    print(f"{lm['name']:<16} ({pc[0]:5d},{pc[1]:5d})   {wx:>10.3f} {wy:>10.3f}  {d:>10.2f} m        {sz_s}")

print("\n" + "="*75)
print("  YOUR 6 CUSTOM TEST POINTS")
print("="*75)
print(f"{'Name':<6} {'Pixel (x,y)':>14}  {'World X':>10} {'World Y':>10}  {'Dist from START':>16}")
print("-"*55)
for name, px, py, _ in custom_pts:
    wx, wy = px_to_world(px, py)
    d = dist(wx, wy)
    print(f"{name:<6} ({px:5d},{py:5d})   {wx:>10.3f} {wy:>10.3f}  {d:>10.2f} m")

print(f"\nSaved : {OUTPUT_FILE}")
print(f"Open  : " + r"\\wsl$\Ubuntu" + OUTPUT_FILE)
