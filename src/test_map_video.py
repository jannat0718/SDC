#!/usr/bin/env python3
"""
test_map_video.py - Thin test harness with CSV logging (using frame_callback)
HOW TO RUN:
  cd /home/jannat/sdc_2026/src
  python3 test_map_video.py --mode video
"""

import os, sys, argparse, csv, math

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.av_map import NavigationPipeline

# -- Video path ----------------------------------------------------------------
VIDEO_PATH = (
    "/home/jannat/sdc_2026/data/SDC_March_Sessions/DATA/"
    "20260305_145211/202603051452114/Middle20260305145211.mp4"
)

# -- Check points ------------------------------------------------------------
CHECK_POINTS = [
    {"name": "kart_start", "track_pixel": (1681, 686), "threshold_m": 2.0, "note": "Video starting position"},
    {"name": "kart_pass",  "track_pixel": (1607, 383), "threshold_m": 2.0, "note": "Kart passes through here"},
    {"name": "stop_sign",  "track_pixel": (1326, 972), "threshold_m": 3.0, "note": "Stop sign location"},
    {"name": "zebra_3",    "track_pixel": (1395, 871), "threshold_m": 2.0, "note": "Zebra crossing 3 - lower lane"},
]

OUTPUT_DIR = os.path.join(project_root, "logs")

# ------------------ CSV helpers (unchanged) ------------------
def log_frame_to_csv(output_dir, frame_num, kart_x, kart_y, heading_deg, stopped, detected_objs):
    csv_path = os.path.join(output_dir, "session_telemetry_log.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Frame", "Kart_X (m)", "Kart_Y (m)", "Heading (deg)",
                "Stopped_Status", "Detected_Label", "Confidence", "Depth (m)", "World_X", "World_Y"
            ])
        if not detected_objs:
            writer.writerow([frame_num, f"{kart_x:.3f}", f"{kart_y:.3f}", f"{heading_deg:.1f}", stopped, "None", "", "", "", ""])
        else:
            for obj in detected_objs:
                writer.writerow([
                    frame_num, f"{kart_x:.3f}", f"{kart_y:.3f}", f"{heading_deg:.1f}",
                    stopped, obj.label, f"{obj.conf:.2f}", f"{obj.depth_m:.2f}",
                    f"{obj.world_x:.2f}", f"{obj.world_y:.2f}"
                ])

def save_checkpoint_summary(output_dir, check_points):
    summary_path = os.path.join(output_dir, "checkpoint_results.csv")
    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Check Point Name", "Status", "Target World Coordinates"])
        for cp in check_points:
            status = "HIT" if cp.get("hit", False) else "MISSED"
            writer.writerow([cp["name"], status, str(cp.get("world", "Unknown"))])
    print(f"[Exporter] Checkpoint summary saved to: {output_dir}")

# ------------------ Frame callback (now takes only the pipeline) ------------------
def make_frame_callback(output_dir):
    """Returns a function that logs the pipeline state after each frame."""
    def callback(pipeline):
        # All needed data is now available directly from the pipeline object
        log_frame_to_csv(
            output_dir,
            pipeline.frame_count,
            pipeline.kart_x,
            pipeline.kart_y,
            pipeline.kart_heading,
            "STOPPED" if pipeline.stopped else "MOVING",
            pipeline.detected_objects
        )
    return callback

def main():
    parser = argparse.ArgumentParser(description="Navigation map test")
    parser.add_argument("--mode",  default="video", choices=["video", "live"])
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--out",   default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frame_cb = make_frame_callback(args.out)

    pipeline = NavigationPipeline(
        mode           = args.mode,
        video_path     = args.video if args.mode == "video" else None,
        check_points   = CHECK_POINTS,
        frame_callback = frame_cb
    )

    pipeline.run()

    # Save checkpoint summary after the video finishes
    save_checkpoint_summary(args.out, pipeline.check_points)

if __name__ == "__main__":
    main()