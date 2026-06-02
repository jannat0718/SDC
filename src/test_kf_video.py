#!/usr/bin/env python3
"""
test_kf_video.py - VO + landmark snaps fused with an EKF.

Same video/config inputs as test_map_video.py, but pose comes from the
Kalman fusion module instead of raw VO. The old test_map_video.py and
av_map.py are NOT modified.

USAGE:
  cd /home/jannat/sdc_2026/src
  python3 test_kf_video.py --config videos/camera_14-40-45.json
"""

import os
import sys
import math
import argparse
from datetime import datetime

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.kalman_fusion import KFPipeline


# =========================================================================
# Per-run start pose - edit these when you have a new recording.
# Set to None to use the per-video JSON config's start_pixel /
# initial_heading_rad. Otherwise these override the JSON.
# CLI flags (--start-pixel, --heading-deg) still override these constants.
#
# Heading convention (degrees, kart-body frame on Track.png):
#     0  = facing +X (east, right of map)
#    90  = facing +Y (north, up on map)
#   180  = facing -X (west, left of map)
#   -90  = facing -Y (south, down on map)
# Right turn DECREASES heading (clockwise viewed from above).
# =========================================================================
START_PIXEL = [1300, 925]             # e.g. [1693, 995]
INITIAL_HEADING_DEG = 40          # e.g. 90.0

DEFAULT_CONFIG = os.path.join(project_root, "videos", "Middle20260226133416.json")
DEFAULT_OUTPUT_BASE = os.path.join(project_root, "logs")


def main():
    p = argparse.ArgumentParser(description="VO + landmarks Kalman fusion test")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Per-video config JSON (e.g. videos/camera_14-40-45.json)")
    p.add_argument("--out", default=None,
                   help="Output dir (default: logs/kf_<timestamp>/)")
    p.add_argument("--detector", default="gftt", choices=["gftt", "orb"],
                   help="VO feature detector (default: gftt)")
    p.add_argument("--vo-pos-noise", type=float, default=0.05,
                   help="Std of VO per-frame position increment (m). "
                        "Higher = trust VO less. Default 0.05.")
    p.add_argument("--vo-yaw-noise", type=float, default=0.5,
                   help="Std of VO per-frame heading increment (degrees). "
                        "Higher = trust VO less. Default 0.5 deg.")
    p.add_argument("--landmark-pos-noise", type=float, default=0.30,
                   help="Std of landmark snap position (m). "
                        "Lower = trust snaps more (snap harder). Default 0.30.")
    p.add_argument("--start-pixel", nargs=2, type=int, default=None,
                   metavar=("PX", "PY"),
                   help="Override kart start pixel on Track.png")
    p.add_argument("--heading-deg", type=float, default=None,
                   help="Override initial heading in degrees. "
                        "0=+X east (right), 90=+Y north (up), "
                        "180=-X west (left), -90=-Y south (down). "
                        "Right turn DECREASES heading.")
    p.add_argument("--start-frame", type=int, default=None,
                   help="Skip to this frame")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Stop after this frame (for smoke testing)")
    p.add_argument("--preview-every-n", type=int, default=10,
                   help="Save preview JPG every N frames (default 10)")
    args = p.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(DEFAULT_OUTPUT_BASE, f"kf_{ts}")

    # Precedence: CLI > module-level constants > per-video JSON.
    effective_start_pixel = (args.start_pixel
                             if args.start_pixel is not None
                             else START_PIXEL)
    effective_heading_deg = (args.heading_deg
                             if args.heading_deg is not None
                             else INITIAL_HEADING_DEG)
    heading_rad = (math.radians(effective_heading_deg)
                   if effective_heading_deg is not None else None)

    src_pixel = ("CLI" if args.start_pixel is not None
                 else "constant" if START_PIXEL is not None
                 else "JSON")
    src_hdg = ("CLI" if args.heading_deg is not None
               else "constant" if INITIAL_HEADING_DEG is not None
               else "JSON")
    pose_pixel_str = (str(effective_start_pixel) if effective_start_pixel is not None
                      else f"(see {os.path.basename(args.config)})")
    pose_hdg_str = (f"{effective_heading_deg}°" if effective_heading_deg is not None
                    else f"(see {os.path.basename(args.config)})")

    print("=" * 60)
    print("  VO + landmarks Kalman fusion test")
    print("=" * 60)
    print(f"  config           : {args.config}")
    print(f"  out              : {args.out}")
    print(f"  start_pixel      : {pose_pixel_str}  (from {src_pixel})")
    print(f"  initial_heading  : {pose_hdg_str}  (from {src_hdg})")
    print(f"  feature detector : {args.detector}")
    print(f"  vo pos noise     : {args.vo_pos_noise} m")
    print(f"  vo yaw noise     : {args.vo_yaw_noise} deg")
    print(f"  landmark pos noise: {args.landmark_pos_noise} m")
    print()

    pipe = KFPipeline(
        config_path=args.config,
        output_dir=args.out,
        vo_pos_noise_m=args.vo_pos_noise,
        vo_yaw_noise_rad=math.radians(args.vo_yaw_noise),
        landmark_pos_noise_m=args.landmark_pos_noise,
        feature_detector=args.detector,
        start_pixel_override=effective_start_pixel,
        heading_override_rad=heading_rad,
        preview_every_n_frames=args.preview_every_n,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    pipe.run()


if __name__ == "__main__":
    main()
