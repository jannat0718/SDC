#!/usr/bin/env python3
"""
test_can_motion_video.py - Run CAN-driven trajectory integration on a video

Mirrors test_map_video.py's style but uses CanPipeline (no Visual Odometry).
Existing test_map_video.py and av_map.py are NOT touched.

USAGE:
  cd /home/jannat/sdc_2026/src
  python3 test_can_motion_video.py \
      --config videos/camera_14-40-45.json \
      --csv    /home/jannat/sdc_2026/data/Thursday_28_05_2026/control_data.csv
"""

import os
import sys
import math
import argparse
from datetime import datetime

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.can_motion_test import CanPipeline


DEFAULT_CONFIG = os.path.join(project_root, "videos", "camera_14-40-45.json")
DEFAULT_CSV = "/home/jannat/sdc_2026/data/Thursday_28_05_2026/control_data.csv"
DEFAULT_OUTPUT_BASE = os.path.join(project_root, "logs")


def main():
    p = argparse.ArgumentParser(description="CAN-based trajectory test")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Per-video config JSON (e.g. videos/camera_14-40-45.json)")
    p.add_argument("--csv", default=DEFAULT_CSV,
                   help="control_data.csv path")
    p.add_argument("--out", default=None,
                   help="Output dir (default: logs/can_motion_<timestamp>/)")
    p.add_argument("--wheelbase", type=float, default=1.0,
                   help="Kart wheelbase in metres (default 1.0)")
    p.add_argument("--steering-scale", type=float, default=8.0,
                   help="Road-wheel deg per unit of command_steering "
                        "(default 8 = derived from vendor 5.75m turn radius)")
    p.add_argument("--steering-deadzone", type=float, default=0.02,
                   help="Treat |command_steering| < this as zero")
    p.add_argument("--invert-steering", action="store_true",
                   help="Negate command_steering (use if trajectory is mirrored)")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="Multiply feedback_speed_kmh by this (default 1.0)")
    p.add_argument("--slip-factor", type=float, default=1.5,
                   help="Low-speed slip multiplier on tan(steering) "
                        "(default 1.5; set 1.0 for pure bicycle model)")
    p.add_argument("--slip-speed-threshold", type=float, default=1.4,
                   help="Speed (m/s) below which full slip-factor applies. "
                        "Fades linearly to 1.0 by 5 m/s (default 1.4 m/s ~5 km/h)")
    p.add_argument("--start-pixel", nargs=2, type=int, default=None,
                   metavar=("PX", "PY"),
                   help="Override kart start pixel on Track.png")
    p.add_argument("--heading-deg", type=float, default=None,
                   help="Override initial heading in degrees (0=+X right)")
    p.add_argument("--preview-every-n", type=int, default=1,
                   help="Save preview JPG every N frames (default 1 = every frame)")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Stop after this frame (for smoke testing)")
    args = p.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(DEFAULT_OUTPUT_BASE, f"can_motion_{ts}")

    heading_rad = (math.radians(args.heading_deg)
                   if args.heading_deg is not None else None)

    print("=" * 60)
    print("  CAN-driven trajectory test")
    print("=" * 60)
    print(f"  config  : {args.config}")
    print(f"  csv     : {args.csv}")
    print(f"  out     : {args.out}")
    print(f"  L       : {args.wheelbase} m")
    print(f"  ratio   : {args.steering_scale} deg/unit")
    print(f"  deadz   : {args.steering_deadzone}")
    print(f"  invert  : {args.invert_steering}")
    print(f"  v scale : {args.speed_scale}")
    print(f"  slip    : {args.slip_factor}x below {args.slip_speed_threshold} m/s, "
          f"fades to 1.0 by 5 m/s")
    print()

    pipe = CanPipeline(
        config_path=args.config,
        csv_path=args.csv,
        output_dir=args.out,
        wheelbase_m=args.wheelbase,
        steering_ratio_deg_per_unit=args.steering_scale,
        steering_deadzone=args.steering_deadzone,
        invert_steering=args.invert_steering,
        speed_scale=args.speed_scale,
        slip_factor_low=args.slip_factor,
        slip_speed_threshold_mps=args.slip_speed_threshold,
        start_pixel_override=args.start_pixel,
        heading_override_rad=heading_rad,
        preview_every_n_frames=args.preview_every_n,
        end_frame=args.end_frame,
    )
    pipe.run()


if __name__ == "__main__":
    main()
