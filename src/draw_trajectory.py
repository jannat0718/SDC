#!/usr/bin/env python3
"""
draw_trajectory.py — Draw saved trajectory + detections onto track map
=======================================================================
Reads session_telemetry_log.csv and draws:
  - Black trajectory line (all kart positions)
  - Coloured dots per detected object type
  - Landmark overlays from nav_config.json
  - HUD summary at bottom

HOW TO RUN:
  cd /home/jannat/sdc_2026/src
  python3 draw_trajectory.py

  # Or specify custom paths:
  python3 draw_trajectory.py \
    --csv  /home/jannat/sdc_2026/logs/session_telemetry_log.csv \
    --out  /home/jannat/sdc_2026/Output/trajectory_map.png
"""

import os, sys, csv, json, argparse
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.dirname(__file__))

DEFAULT_CSV    = os.path.join(project_root, "logs",       "session_telemetry_log.csv")
DEFAULT_OUT    = os.path.join(project_root, "..", "Output", "trajectory_map.png")
CONFIG_PATH    = os.path.join(project_root, "nevigation", "nav_config.json")
TRACK_IMG_PATH = os.path.join(project_root, "nevigation", "Track.png")

# ── Object colours (BGR) ──────────────────────────────────────────────────────
OBJ_COLORS = {
    "stop":                (0,   0,   255),   # red
    "person":              (255, 128,   0),   # sky blue
    "kart":                (0,   255, 255),   # yellow
    "speed_20":            (0,   165, 255),   # orange
    "speed_30":            (0,   165, 255),   # orange
    "traffic_red_light":   (0,   0,   200),   # dark red
    "traffic_green_light": (0,   200,   0),   # green
    "turn_left":           (255,   0, 255),   # magenta
}
DEFAULT_OBJ_COLOR = (180, 180, 180)


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def px_to_world(px, py, origin_px, px_per_m):
    wx = (px - origin_px[0]) / px_per_m
    wy = -(py - origin_px[1]) / px_per_m
    return wx, wy


def world_to_pixel_homography(wx, wy, H):
    if H is None:
        return None, None
    pt  = np.array([[[wx, wy]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return int(res[0,0,0]), int(res[0,0,1])


def world_to_pixel_simple(wx, wy, origin_px, px_per_m, map_w, map_h):
    u = int(origin_px[0] + wx * px_per_m)
    v = int(origin_px[1] - wy * px_per_m)
    return u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=DEFAULT_CSV)
    parser.add_argument("--out",   default=DEFAULT_OUT)
    parser.add_argument("--track", default=TRACK_IMG_PATH)
    parser.add_argument("--config",default=CONFIG_PATH)
    parser.add_argument("--skip",  type=int, default=3,
                        help="Draw every Nth trajectory point (default 3 = every 3rd frame)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    origin_px  = cfg["start_px"]          # [6686, 615]
    px_per_m   = cfg["px_per_meter"]      # 17.6801
    landmarks  = cfg.get("fixed_landmarks", [])

    # Homography
    H = None
    if cfg.get("world_pts") and cfg.get("image_pts"):
        wp = np.float32(cfg["world_pts"])
        ip = np.float32(cfg["image_pts"])
        H, _ = cv2.findHomography(wp, ip)

    # ── Load track image ──────────────────────────────────────────────────────
    track = cv2.imread(args.track)
    if track is None:
        print(f"ERROR: Track image not found: {args.track}")
        sys.exit(1)
    MAP_H, MAP_W = track.shape[:2]
    print(f"Track image: {MAP_W} x {MAP_H}")

    def w2p(wx, wy):
        if H is not None:
            u, v = world_to_pixel_homography(wx, wy, H)
        else:
            u, v = world_to_pixel_simple(wx, wy, origin_px, px_per_m, MAP_W, MAP_H)
        return u, v

    # ── Draw fixed landmarks ──────────────────────────────────────────────────
    for lm in landmarks:
        ltype = lm.get("type", "point")
        color = tuple(int(c) for c in lm.get("color_bgr", [200,200,200]))
        name  = lm["name"]

        if ltype == "line" and "pixel_box" in lm:
            box = lm["pixel_box"]
            xc  = lm["pixel_center"][0]
            y1, y2 = box["y_min"], box["y_max"]
            cv2.line(track, (xc, y1), (xc, y2), color, 4)
            cv2.line(track, (xc-10, y1), (xc+10, y1), color, 2)
            cv2.line(track, (xc-10, y2), (xc+10, y2), color, 2)
            cv2.putText(track, name, (xc+8, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif ltype == "zebra" and "pixel_box" in lm:
            box = lm["pixel_box"]
            x1,y1,x2,y2 = box["x_min"],box["y_min"],box["x_max"],box["y_max"]
            ov = track.copy()
            cv2.rectangle(ov, (x1,y1), (x2,y2), color, -1)
            cv2.addWeighted(ov, 0.2, track, 0.8, 0, track)
            cv2.rectangle(track, (x1,y1), (x2,y2), color, 2)
            cv2.putText(track, name, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        elif ltype == "point":
            pc = lm["pixel_center"]
            cv2.drawMarker(track, tuple(pc), color, cv2.MARKER_STAR, 18, 2)

    # ── Read CSV ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    print(f"CSV rows: {len(rows)}")

    # ── Draw trajectory (BLACK, every Nth frame) ──────────────────────────────
    traj_pts = []
    in_bounds_count = 0

    for i, row in enumerate(rows):
        if i % args.skip != 0:
            continue
        try:
            kx = float(row['Kart_X (m)'])
            ky = float(row['Kart_Y (m)'])
        except (ValueError, KeyError):
            continue

        u, v = w2p(kx, ky)
        if u is None:
            continue
        if 0 <= u < MAP_W and 0 <= v < MAP_H:
            traj_pts.append((u, v))
            in_bounds_count += 1

    print(f"Trajectory points in bounds: {in_bounds_count} / {len(rows)//args.skip}")

    # Draw trajectory as BLACK line
    if len(traj_pts) > 1:
        for i in range(1, len(traj_pts)):
            cv2.line(track, traj_pts[i-1], traj_pts[i], (0, 0, 0), 3)
        # Draw start marker
        cv2.circle(track, traj_pts[0],  12, (0, 0, 0), -1)
        cv2.circle(track, traj_pts[-1], 12, (50, 50, 50), -1)
        cv2.putText(track, "START", (traj_pts[0][0]+14, traj_pts[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(track, "END", (traj_pts[-1][0]+14, traj_pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)

    # ── Draw detected objects ─────────────────────────────────────────────────
    det_rows = [r for r in rows
                if r.get('Detected_Label','None') not in ('None', '') 
                and r.get('World_X','') != ''
                and r.get('World_Y','') != '']

    print(f"Detection rows: {len(det_rows)}")

    det_counts = {}
    det_in_bounds = 0

    for row in det_rows:
        try:
            wx = float(row['World_X'])
            wy = float(row['World_Y'])
            label = row['Detected_Label']
            conf  = float(row.get('Confidence', 0))
            depth = float(row.get('Depth (m)', 0))
        except ValueError:
            continue

        u, v = w2p(wx, wy)
        if u is None or not (0 <= u < MAP_W and 0 <= v < MAP_H):
            continue

        det_in_bounds += 1
        det_counts[label] = det_counts.get(label, 0) + 1
        color = OBJ_COLORS.get(label.lower(), DEFAULT_OBJ_COLOR)

        # Draw dot + label
        cv2.circle(track, (u, v), 10, color, -1)
        cv2.circle(track, (u, v), 10, (255,255,255), 1)
        txt = f"{label} {depth:.1f}m"
        cv2.putText(track, txt, (u+12, v+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    print(f"Detections drawn on map: {det_in_bounds}")
    for label, cnt in sorted(det_counts.items()):
        print(f"  {label}: {cnt}")

    # ── Info panel below track ─────────────────────────────────────────────────
    PANEL_H = 80
    panel   = np.full((PANEL_H, MAP_W, 3), 20, dtype=np.uint8)
    cv2.line(panel, (0,0), (MAP_W,0), (100,100,100), 2)

    # Trajectory stats
    kxs = [float(r['Kart_X (m)']) for r in rows]
    kys = [float(r['Kart_Y (m)']) for r in rows]
    info1 = (f"Trajectory: {len(rows)} frames  "
             f"X:[{min(kxs):.1f},{max(kxs):.1f}]m  "
             f"Y:[{min(kys):.1f},{max(kys):.1f}]m  "
             f"In-bounds: {in_bounds_count}/{len(rows)//args.skip} pts")
    cv2.putText(panel, info1, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,255,180), 1)

    info2 = "Detections: " + "  ".join(
        f"{lbl}={cnt}" for lbl,cnt in sorted(det_counts.items()))
    info2 += f"  (in-bounds: {det_in_bounds})"
    cv2.putText(panel, info2, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,200,255), 1)

    combined = np.vstack([track, panel])

    # ── Save ──────────────────────────────────────────────────────────────────
    cv2.imwrite(args.out, combined)
    print(f"\nSaved: {args.out}")
    wsl = r"\\wsl$\Ubuntu" + os.path.abspath(args.out)
    print(f"Open : {wsl}")

    # ── Terminal summary ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAJECTORY SUMMARY")
    print("="*55)
    print(f"Total frames    : {len(rows)}")
    print(f"In-bounds pts   : {in_bounds_count} ({in_bounds_count/(len(rows)//args.skip)*100:.1f}%)")
    print(f"X range (world) : {min(kxs):.2f} to {max(kxs):.2f} m")
    print(f"Y range (world) : {min(kys):.2f} to {max(kys):.2f} m")
    print(f"\nDetections on map: {det_in_bounds}")
    for lbl, cnt in sorted(det_counts.items()):
        color_name = {
            "stop": "RED", "person": "SKY BLUE",
            "kart": "YELLOW", "speed_20": "ORANGE",
            "speed_30": "ORANGE", "traffic_green_light": "GREEN",
            "traffic_red_light": "DARK RED"
        }.get(lbl, "GREY")
        print(f"  {lbl:<22} {cnt:>4} detections  [{color_name}]")


if __name__ == "__main__":
    main()
