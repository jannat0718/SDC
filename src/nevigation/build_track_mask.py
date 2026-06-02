"""
Offline script: read Track.png + nav_config_1080.json,
produce a drivable-area mask and a marking mask aligned to Track.png.

Usage:
    cd /home/jannat/sdc_2026/src/nevigation
    python3 build_track_mask.py [path/to/nav_config_1080.json]

If no config path is supplied, defaults to ./nav_config_1080.json.

Outputs (saved next to Track.png):
    track_drivable_mask.npy   -- bool array, True where kart can drive
    track_markings_mask.npy   -- bool array, True for on-track markings (zebra, dashed lines, etc.)
    track_mask_3class.png     -- uint8 image: 0 = non-drivable, 128 = drivable, 255 = markings
    track_mask_overlay.png    -- BGR debug overlay on Track.png

Reproducible defaults:
    * Drivable seed = nav_config.track_info.start_pixel
    * Wall threshold = white component with max(W,H) >= 200 px in canonical scale
    * White components matching configured DRIVABLE_LINES (8 entries) are
      forced to be markings, not walls.
    * Two known non-drivable rectangles excluded by hand.
    * ROI restricts the mask to the on-track region.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Defaults / config (canonical 9360x1410)
# --------------------------------------------------------------------------
CANON_W, CANON_H = 9360, 1410

# Region of interest: only consider pixels in this bbox as candidate drivable
ROI_CANON = (315, 270, 6795, 1170)

# Known non-drivable rectangles (parking-stall barriers / blocked regions)
NON_DRIVABLE_RECTS_CANON: List[Tuple[int, int, int, int]] = [
    (5130, 450, 5550, 540),
    (5130, 690, 5550, 840),
]

# Wall classification: a white connected component is treated as a wall iff
# max(width, height) >= this many pixels in canonical scale.
WALL_LEN_PX_CANON = 200

# Drivable-line overrides: white components overlapping any of these bboxes
# get reclassified as markings (drivable) regardless of length. Names suffixed
# with _CFG_NAME are looked up from cfg.fixed_landmarks; bare tuples are
# specified directly.
DRIVABLE_LINES_FROM_CFG = [
    'starting_line',
    'upper_line_after_start',
    'lower_line_zebra3_east',
    'lower_line_before_end',
    'end_line',
]
DRIVABLE_LINES_DIRECT_CANON: List[Tuple[str, Tuple[int, int, int, int]]] = [
    ('line4_horizontal', (1550, 688, 1633, 688)),
    ('line5_vertical',   (1820, 785, 1820, 865)),
    ('line6_horizontal', (1640, 1053, 1725, 1053)),
]

# Pad applied to each drivable-line bbox AT CANONICAL SCALE before scaling
# down to the target file. Ensures degenerate (1-px) bboxes still cover the
# actual white pixels.
DRIVABLE_LINE_PAD_CANON = 6   # pixels in canonical (9360x1410)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def landmark_pixel_box(cfg: dict, name: str) -> Optional[Tuple[int, int, int, int]]:
    for L in cfg.get('fixed_landmarks', []):
        if L.get('name') == name:
            b = L.get('pixel_box')
            if not b:
                return None
            return (b['x_min'], b['y_min'], b['x_max'], b['y_max'])
    return None

def make_scaler(W: int, H: int):
    """Return (sx, sy, srect, spt) for converting canonical → file coords."""
    sx = W / CANON_W
    sy = H / CANON_H
    def srect(r: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = r
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return (max(0, int(round(x1 * sx))),
                max(0, int(round(y1 * sy))),
                min(W, int(round(x2 * sx))),
                min(H, int(round(y2 * sy))))
    def spt(p: Tuple[int, int]) -> Tuple[int, int]:
        return (max(0, min(W - 1, int(round(p[0] * sx)))),
                max(0, min(H - 1, int(round(p[1] * sy)))))
    return sx, sy, srect, spt


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------
def build_mask(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    track_path = cfg.get('track_image_path')
    if not track_path or not os.path.exists(track_path):
        raise FileNotFoundError(
            f"track_image_path '{track_path}' from nav_config does not exist.")
    out_dir = os.path.dirname(os.path.abspath(track_path))

    img = cv2.imread(track_path)
    if img is None:
        raise IOError(f"cv2.imread failed on {track_path}")
    H, W = img.shape[:2]
    sx, sy, srect, spt = make_scaler(W, H)
    print(f"[build_track_mask] Track image: {W}x{H}  scale-from-canonical: x={sx:.4f} y={sy:.4f}")

    # ---- Resolve seed ----
    seed_canon = tuple(cfg['track_info']['start_pixel'])
    origin_canon = tuple(cfg['start_px'])   # world origin pixel
    seed = spt(seed_canon)
    origin_px = spt(origin_canon)
    print(f"[build_track_mask] Seed (kart start_pixel): canonical {seed_canon} -> file {seed}")
    print(f"[build_track_mask] World origin: canonical {origin_canon} -> file {origin_px}")

    # ---- Inflate drivable-line bboxes AT CANONICAL SCALE before scaling ----
    pad = DRIVABLE_LINE_PAD_CANON
    def inflate_canon(b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = b
        return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    drv_lines_canon: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for nm in DRIVABLE_LINES_FROM_CFG:
        b = landmark_pixel_box(cfg, nm)
        if b is None:
            print(f"[build_track_mask] WARNING: landmark '{nm}' not found in cfg, skipping")
            continue
        drv_lines_canon.append((nm, inflate_canon(b)))
    for nm, b in DRIVABLE_LINES_DIRECT_CANON:
        drv_lines_canon.append((nm, inflate_canon(b)))
    drv_lines = [(nm, srect(b)) for nm, b in drv_lines_canon]
    print(f"[build_track_mask] Drivable-line overrides ({len(drv_lines)}):")
    for nm, (x1, y1, x2, y2) in drv_lines:
        print(f"   {nm}: ({x1},{y1})-({x2},{y2})  (size {x2-x1}x{y2-y1})")

    # ---- Resolve other regions ----
    roi = srect(ROI_CANON)
    nd = [srect(r) for r in NON_DRIVABLE_RECTS_CANON]

    # ---- Pixel classes: grey (drivable surface) and white (markings/walls) ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey  = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 100) & (hsv[:, :, 2] < 200)).astype(np.uint8) * 255
    white = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] >= 200)).astype(np.uint8) * 255

    # ---- Classify white components into walls vs markings ----
    wall_len_px = max(1, int(round(WALL_LEN_PX_CANON * sx)))
    print(f"[build_track_mask] Wall length threshold (this file): {wall_len_px} px")
    nlbl, lbl, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    walls    = np.zeros_like(white)
    markings = np.zeros_like(white)
    for i in range(1, nlbl):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 15:
            continue
        w_b = stats[i, cv2.CC_STAT_WIDTH]
        h_b = stats[i, cv2.CC_STAT_HEIGHT]
        if max(w_b, h_b) >= wall_len_px:
            walls[lbl == i]    = 255
        else:
            markings[lbl == i] = 255

    # ---- Override: any wall pixels inside drivable-line bboxes become markings ----
    # ---- Override: any wall pixels inside drivable-line bboxes become markings ----
    # ---- Override: any wall pixels inside drivable-line bboxes become markings ----
    override_mask = np.zeros_like(walls)
    bridge_mask = np.zeros_like(walls)
    
    # Define exactly which lines let flood-fill pass. 
    # We include all config lines + line4 & line6, but STRICTLY EXCLUDE line5_vertical.
    safe_bridges = DRIVABLE_LINES_FROM_CFG + ['line4_horizontal', 'line6_horizontal']
    
    for nm, (x1, y1, x2, y2) in drv_lines:
        override_mask[y1:y2+1, x1:x2+1] = 255
        
        if nm in safe_bridges:
            bridge_mask[y1:y2+1, x1:x2+1] = 255
            
    # Isolate the white pixels belonging to the allowed bridge lines
    bridge_white = cv2.bitwise_and(white, bridge_mask)

    demote = cv2.bitwise_and(walls, override_mask)
    walls    = cv2.bitwise_and(walls, cv2.bitwise_not(demote))
    markings = cv2.bitwise_or(markings, demote)

    # Also force the drivable-line bboxes to contribute their *white pixels*
    # to the markings layer even when those white pixels were already in the
    # markings bucket (e.g. line was below the wall threshold). And union the
    # bbox interior with white so the rendered line is visible.
    line_white = cv2.bitwise_and(white, override_mask)
    bridge_white = cv2.bitwise_and(white, bridge_mask)
    markings = cv2.bitwise_or(markings, line_white)
    print(f"[build_track_mask] Wall pixels demoted to markings: {int((demote > 0).sum())}")
    print(f"[build_track_mask] Total marking pixels in drivable-line bboxes: {int((line_white > 0).sum())}")

    # ---- Dilate walls slightly to close 1-2 px anti-aliasing gaps ----
    walls_dil = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ---- Build passable mask + flood fill from seed ----
    # Union grey with only the bridge_white boundaries to prevent inland leaks
    passable = (((grey == 255) | (bridge_white == 255)) & (walls_dil == 0)).astype(np.uint8) * 255
    for (x1, y1, x2, y2) in nd:
        passable[y1:y2, x1:x2] = 0
    roi_mask = np.zeros_like(passable)
    roi_mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
    passable = cv2.bitwise_and(passable, roi_mask)

    # Relocate seed if it landed on a non-passable pixel
    if passable[seed[1], seed[0]] == 0:
        best = None
        best_d = 1e18
        sr = max(40, int(40 * sx))
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                ny, nx = seed[1] + dy, seed[0] + dx
                if 0 <= ny < H and 0 <= nx < W and passable[ny, nx] > 0:
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
        if best is None:
            raise RuntimeError(
                f"Seed {seed} is not passable and no passable pixel found within {sr}px.")
        print(f"[build_track_mask] Seed relocated from {seed} to {best} ({best_d**0.5:.1f}px)")
        seed = best

    ff = passable.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), dtype=np.uint8), seed, 128)
    drivable = ((ff == 128).astype(np.uint8)) * 255
    drivable_pct = 100.0 * (drivable > 0).sum() / (W * H)
    print(f"[build_track_mask] Drivable region: {int((drivable > 0).sum())} px ({drivable_pct:.2f}% of image)")

    # ---- Markings considered "on track" = within a few pixels of drivable ----
    near_drivable = cv2.dilate(drivable,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    markings_on_track = cv2.bitwise_and(markings, near_drivable)

    # ---- 3-class PNG ----
    vis3 = np.zeros((H, W), dtype=np.uint8)
    vis3[drivable == 255]          = 128   # grey = drivable
    vis3[markings_on_track == 255] = 255   # white = on-track markings

    # ---- Colored debug overlay ----
    overlay = (img * 0.35).astype(np.uint8)
    green_layer = np.full_like(img, (0, 200, 0))
    overlay = np.where(drivable[:, :, None] == 255,
                       cv2.addWeighted(img, 0.55, green_layer, 0.45, 0),
                       overlay)
    walls_vis = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    overlay[walls_vis == 255] = (0, 0, 255)           # walls red
    mark_vis = cv2.dilate(markings_on_track, np.ones((3, 3), np.uint8))
    overlay[(mark_vis == 255) & (drivable == 255)] = (0, 255, 255)   # markings yellow
    # Drivable line bboxes (and any white in them) painted BLUE
    ov_vis = cv2.dilate(override_mask, np.ones((3, 3), np.uint8))
    overlay[ov_vis == 255] = (255, 100, 0)            # drivable lines blue
    cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 4)
    for (x1, y1, x2, y2) in nd:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
    for nm, (x1, y1, x2, y2) in drv_lines:
        cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 100, 0), 2)
    # SEED marker
    cv2.circle(overlay, seed, max(8, int(20 * sx)), (255, 255, 255), 3)
    cv2.putText(overlay, "SEED",
                (seed[0] + int(30 * sx), seed[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sx, (255, 255, 255), 2)
    # World origin marker (cyan cross-hair)
    ox, oy = origin_px
    cv2.drawMarker(overlay, (ox, oy), (255, 255, 0),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=max(20, int(40 * sx)), thickness=3)
    cv2.circle(overlay, (ox, oy), max(10, int(25 * sx)), (255, 255, 0), 3)
    cv2.putText(overlay, "WORLD ORIGIN",
                (ox + int(30 * sx), oy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sx, (255, 255, 0), 2)
    # Legend
    legend_items = [
        ("Drivable (green)",           (0, 200, 0)),
        ("Wall (red)",                 (0, 0, 255)),
        ("Drivable line (blue)",       (255, 100, 0)),
        ("Marking on track (yellow)",  (0, 255, 255)),
        ("ROI border (yellow box)",    (0, 255, 255)),
        ("Non-drivable rect (mag)",    (255, 0, 255)),
        ("Seed (white)",               (255, 255, 255)),
        ("World origin (cyan)",        (255, 255, 0)),
    ]
    pad_box = 10
    legend_w = 540
    legend_h = 40 * len(legend_items) + 20
    cv2.rectangle(overlay, (pad_box, pad_box),
                  (pad_box + legend_w, pad_box + legend_h), (255, 255, 255), -1)
    cv2.rectangle(overlay, (pad_box, pad_box),
                  (pad_box + legend_w, pad_box + legend_h), (0, 0, 0), 2)
    for i, (label, color) in enumerate(legend_items):
        cv2.putText(overlay, label, (pad_box + 20, pad_box + 35 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ---- Save outputs ----
    np.save(os.path.join(out_dir, 'track_drivable_mask.npy'), drivable > 0)
    np.save(os.path.join(out_dir, 'track_markings_mask.npy'), markings_on_track > 0)
    cv2.imwrite(os.path.join(out_dir, 'track_mask_3class.png'), vis3)
    # Save downscaled overlay for visual review (full-res file would be huge)
    max_w = 2400
    if W > max_w:
        scale = max_w / W
        ov_small = cv2.resize(overlay, (max_w, int(H * scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        ov_small = overlay
    cv2.imwrite(os.path.join(out_dir, 'track_mask_overlay.png'), ov_small)
    print(f"[build_track_mask] Saved artifacts to {out_dir}:")
    print("    track_drivable_mask.npy   (bool, drivable area)")
    print("    track_markings_mask.npy   (bool, on-track markings)")
    print("    track_mask_3class.png     (uint8, 0/128/255)")
    print("    track_mask_overlay.png    (debug visual)")


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'nav_config_1080.json'
    build_mask(cfg_path)