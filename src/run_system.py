#!/usr/bin/env python3
r"""
run_system.py — Integrated SDC 2026 Navigation Pipeline
=========================================================
Combines:
  - YOLOv8 OpenVINO object detection  (av_system.py)
  - Visual Odometry + track map        (av_map.py)
  - Output video with map overlay      (Output/result_with_map.mp4)
  - Preview JPG every second           (Output/nav_preview.jpg)

USAGE:
  # Video mode (tuning):
  python3 src/run_system.py --mode video

  # Live camera:
  python3 src/run_system.py --mode live

TERMINAL COMMANDS (type + ENTER):
  q → quit
  r → reset VO pose
  s → save current frame
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
# Detects if script is run from project root or inside src/
script_dir = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(script_dir) == "src":
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
else:
    project_root = script_dir

# Inject the src/ directory into system paths explicitly
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.av_system import ObjectDetector, MODEL_XML, CLASSES_PATH
from nevigation.av_map    import (MapSystem, DetectedObject,
                                  PREVIEW_FILE, OUTPUT_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = (
    "/home/jannat/sdc_2026/data/SDC_March_Sessions/DATA/"
    "20260305_145211/202603051452114/Middle20260305145211.mp4"
)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "result_with_map.mp4")


def main():
    parser = argparse.ArgumentParser(description="SDC 2026 Navigation Pipeline")
    parser.add_argument("--mode", default="video",
                        choices=["video", "live"],
                        help="video = recorded file | live = camera")
    parser.add_argument("--video", default=VIDEO_PATH,
                        help="Path to video file")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SDC 2026 — Integrated Navigation Pipeline")
    print("="*60)

    # ── Step 1: Load detector ─────────────────────────────────────────────────
    print(f"\n[1/3] Loading OpenVINO model: {os.path.basename(MODEL_XML)}")
    print("      Compiling YOLOv8 network (first run is slow ~10s)...")
    detector = ObjectDetector(MODEL_XML, CLASSES_PATH)
    print("      Model ready.")

    # ── Step 2: Init MapSystem ────────────────────────────────────────────────
    print(f"\n[2/3] Initialising MapSystem ({args.mode} mode)...")
    map_system = MapSystem(
        mode=args.mode,
        video_path=args.video if args.mode == "video" else None
    )

    # ── Step 3: Wire detector into MapSystem ──────────────────────────────────
    def get_detections(frame: np.ndarray) -> list:
        raw = detector.detect(frame)
        return [DetectedObject(
                    label=det['label'],
                    bbox=(det['x1'], det['y1'], det['x2'], det['y2']),
                    depth=None,          # ground plane depth computed inside projector
                    class_id=det['class_id'])
                for det in raw]

    map_system.get_detections_from_teammates = get_detections
    print("      Detector wired into MapSystem.")

    # ── Step 4: Set up output video writer ────────────────────────────────────
    print(f"\n[3/3] Output video → {OUTPUT_VIDEO}")
    print(f"      Preview JPG → {PREVIEW_FILE}")
    print(f"\n      Commands: q=quit  r=reset_pose  s=save_frame")
    print("="*60 + "\n")

    _orig_run = map_system.run.__func__  # get unbound method

    def patched_run(self):
        """Runs the original loop but also writes combined frames to video."""
        import threading

        writer     = [None]   # list so closure can mutate
        frame_size = [None]

        _orig_render = self.mapper.render

        def render_and_record(*args, **kwargs):
            combined = _orig_render(*args, **kwargs)

            # Init video writer on first frame
            if writer[0] is None:
                h, w = combined.shape[:2]
                frame_size[0] = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps_out = 15.0 if self.mode == "video" else 20.0
                writer[0] = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_out, (w, h))
                print(f"[VideoWriter] Writing {w}x{h} @ {fps_out}fps → {OUTPUT_VIDEO}")

            writer[0].write(combined)
            return combined

        self.mapper.render = render_and_record

        try:
            _orig_run(self)   # run original loop
        finally:
            if writer[0]:
                writer[0].release()
                print(f"\n[VideoWriter] Saved → {OUTPUT_VIDEO}")

    import types
    map_system.run = types.MethodType(patched_run, map_system)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        map_system.run()
    except KeyboardInterrupt:
        print("\n[run_system] Interrupted.")
    except Exception as e:
        print(f"\n[run_system] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[run_system] Done.")
        print(f"  Preview    → {PREVIEW_FILE}")
        print(f"  Video out → {OUTPUT_VIDEO}")
        final_map = os.path.join(OUTPUT_DIR, "map_final.png")
        if os.path.exists(final_map):
            print(f"  Final map → {final_map}")


if __name__ == "__main__":
    main()