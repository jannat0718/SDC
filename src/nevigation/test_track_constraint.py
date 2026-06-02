"""
test_track_constraint.py
=========================
Standalone test for track_constraint + track_aiding.

Run AFTER build_track_mask.py has produced the .npy artifacts.

Usage:
    cd /home/jannat/sdc_2026/src/nevigation
    python3 test_track_constraint.py [path/to/nav_config_1080.json]

What it does:
    1. Loads the mask via load_from_nav_config()
    2. Tests a few known points (kart start, world origin, off-track points)
    3. Tests soft_track_pull on synthetic off-track positions
    4. Renders the mask + test query results on a debug image so you can
       visually verify the queries return sensible results.

No side effects on the real pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Allow running this script standalone whether or not the package is on path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from track_constraint import TrackConstraint, load_from_nav_config
from track_aiding import soft_track_pull


@dataclass
class FakeState:
    """Stand-in for an EKF state so we can call soft_track_pull standalone."""
    x: float
    y: float
    theta: float = 0.0


def main(cfg_path: str) -> int:
    with open(cfg_path) as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(cfg.get('track_image_path', '.')))
    track = load_from_nav_config(cfg, base_dir=base_dir)

    px_per_m = float(cfg['px_per_meter'])
    origin   = tuple(cfg['start_px'])
    print(f"Config: px_per_m={px_per_m}, origin_px={origin}")

    # ---- known-position tests -------------------------------------------
    kart_start_px = tuple(cfg['track_info']['start_pixel'])     # (1320, 915)
    kart_end_px   = tuple(cfg['track_info']['end_pixel'])       # (6687, 915)
    print(f"Kart start (px): {kart_start_px}  end (px): {kart_end_px}")

    # Convert kart_start_px to world via inverse of world_to_pixel:
    # px = origin_x + wx*ppm  ->  wx = (px - origin_x)/ppm
    # py = origin_y - wy*ppm  ->  wy = (origin_y - py)/ppm
    def px_to_world(p):
        wx = (p[0] - origin[0]) / px_per_m
        wy = (origin[1] - p[1]) / px_per_m
        return wx, wy

    ks_w = px_to_world(kart_start_px)
    ke_w = px_to_world(kart_end_px)
    print(f"Kart start (m): ({ks_w[0]:.2f}, {ks_w[1]:.2f})")
    print(f"Kart end (m):   ({ke_w[0]:.2f}, {ke_w[1]:.2f})")
    print()

    print("--- is_drivable / is_marking tests ---")
    test_points = [
        ("kart_start",         ks_w),
        ("kart_end",           ke_w),
        ("world_origin",       (0.0, 0.0)),                      # by definition origin_px
        ("100m_above_kart_start", (ks_w[0], ks_w[1] + 100.0)),   # off the map
        ("off-grass_north",    (50.0, 30.0)),                    # likely off-track
        ("center_main_straight",(50.0, 0.0)),                    # ambiguous
    ]
    for name, (x, y) in test_points:
        drv = track.is_drivable(x, y)
        mrk = track.is_marking(x, y)
        d   = track.distance_to_drivable_m(x, y)
        print(f"  {name:25s} world=({x:7.2f},{y:7.2f}) drivable={drv} marking={mrk} dist_to_drivable={d:.2f}m")
    print()

    print("--- nearest_drivable for off-track points ---")
    off_track_probes = [
        ("kart_start + 5m up",    (ks_w[0],     ks_w[1] + 5.0)),
        ("kart_start + 2m left",  (ks_w[0] - 2, ks_w[1])),
        ("middle of trapezoid",   (60.0,        4.0)),
    ]
    for name, (x, y) in off_track_probes:
        res = track.nearest_drivable(x, y)
        if res is None:
            print(f"  {name}: outside image bounds")
        else:
            nx, ny, d = res
            print(f"  {name:30s} from ({x:6.2f},{y:6.2f}) -> ({nx:6.2f},{ny:6.2f}) dist={d:.2f}m")
    print()

    print("--- soft_track_pull tests ---")
    for name, (x, y) in [("on-track (kart_start)", ks_w),
                          ("0.3m above kart_start", (ks_w[0], ks_w[1] + 0.3)),
                          ("1.0m above kart_start", (ks_w[0], ks_w[1] + 1.0)),
                          ("3.0m above kart_start", (ks_w[0], ks_w[1] + 3.0)),
                          ("20m above (too far)",   (ks_w[0], ks_w[1] + 20.0))]:
        s = FakeState(x=x, y=y)
        pull = soft_track_pull(s, track, R_track_m_sq=0.25, max_pull_dist_m=5.0)
        if pull is None:
            print(f"  {name:30s} -> no pull (on-track or out of range)")
        else:
            zx, zy, R = pull
            print(f"  {name:30s} -> pull to ({zx:6.2f},{zy:6.2f}) R={R:.4f}")
    print()

    # ---- visual verification --------------------------------------------
    print("--- saving debug image ---")
    track_img = cv2.imread(cfg['track_image_path'])
    if track_img is None:
        print("  could not load track image for visualisation; skipping")
        return 0
    H, W = track_img.shape[:2]
    drivable_uint8 = (np.load(os.path.join(base_dir, 'track_drivable_mask.npy')).astype(np.uint8) * 255)
    if drivable_uint8.shape != (H, W):
        drivable_uint8 = cv2.resize(drivable_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
    overlay = track_img.copy()
    green = np.full_like(track_img, (0, 200, 0))
    overlay = np.where(drivable_uint8[:, :, None] == 255,
                       cv2.addWeighted(track_img, 0.6, green, 0.4, 0),
                       (track_img * 0.4).astype(np.uint8))
    # Draw test queries: green dot = nearest drivable, red dot = source
    for name, (x, y) in off_track_probes + test_points:
        srcpx = track.world_to_pixel(x, y)
        if 0 <= srcpx[0] < W and 0 <= srcpx[1] < H:
            cv2.circle(overlay, srcpx, 12, (0, 0, 255), 2)
        res = track.nearest_drivable(x, y)
        if res:
            nxm, nym, _ = res
            tgtpx = track.world_to_pixel(nxm, nym)
            if 0 <= tgtpx[0] < W and 0 <= tgtpx[1] < H:
                cv2.circle(overlay, tgtpx, 8, (0, 255, 0), -1)
                cv2.line(overlay, srcpx, tgtpx, (0, 200, 255), 2)
    out_path = os.path.join(base_dir, 'track_constraint_test.png')
    # Downscale for review
    if W > 2400:
        s = 2400 / W
        overlay = cv2.resize(overlay, (2400, int(H * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, overlay)
    print(f"  saved {out_path}")
    return 0


if __name__ == '__main__':
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else 'nav_config_1080.json'
    sys.exit(main(cfg_arg))
