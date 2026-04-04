#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2

# --- Modules not yet available — commented out ---
# from common.logging_utils import ControlLogger, DetectionLogger
# from perception.lane_detector import LaneDetector
# -------------------------------------------------

from control.manual_xbox_control import XboxCanController
from perception.object_detector import OnnxDetector, load_class_names, find_openvino_model
from perception.sensors import CameraStream  # LidarStream, startup_self_check


def draw_overlay(frame, detections, fps):
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 200, 255), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (10, 200, 255),
            2,
        )
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main runtime – OpenVINO perception + Xbox control")

    parser.add_argument("--model", type=str, default="",
                        help="Path to OpenVINO .xml model file or folder containing one")
    parser.add_argument("--class-names", type=str, default="", help="Optional class names text file")

    parser.add_argument("--camera-index", type=int, default=2, help="OpenCV camera index (default: 2 = middle cam)")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.45)

    parser.add_argument("--save-video", type=str, default="",
                        help="Path to save output video (e.g. ../../Output/realtime_output.mp4)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable-control", action="store_true", help="Disable Xbox+CAN control module")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Resolve model path ----
    model_path = args.model
    if not model_path:
        script_dir = Path(__file__).resolve().parent
        model_dir = script_dir.parent.parent / "Model"
        if model_dir.is_dir():
            model_path = find_openvino_model(str(model_dir))
            print(f"[main] Auto-detected model: {model_path}")
        else:
            raise FileNotFoundError("No --model specified and ../../Model/ not found")
    elif os.path.isdir(model_path):
        model_path = find_openvino_model(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    auto_headless = (os.name != "nt") and (os.environ.get("DISPLAY") is None)
    headless = args.headless or auto_headless

    # ---- Load class names ----
    class_names = load_class_names(args.class_names if args.class_names else None)

    # ---- Create detector ----
    detector = OnnxDetector(
        model_path=model_path,
        class_names=class_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    # ---- Camera (created but NOT started yet — waits for X button) ----
    camera = CameraStream(args.camera_index, args.width, args.height, args.fps)
    camera_running = False

    # ---- Video writer ----
    video_writer = None
    if args.save_video:
        save_path = Path(args.save_video)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Controller (always active for kart driving) ----
    controller = None
    if not args.disable_control:
        controller = XboxCanController()
        controller.start()
        print("[main] Xbox+CAN controller started. Kart is drivable.")

    print("[main] Press X to start camera + detection. Press Y to stop. Press B to kill.")

    prev_t = time.time()
    last_status_t = 0.0

    try:
        while True:
            # ---- Always process controller (kart drives regardless of camera) ----
            if controller:
                control_info = controller.step(timeout_s=0.005)

                # B-kill: stop everything
                if controller.state.emergency_stop:
                    print("[main] EMERGENCY STOP (B pressed). Exiting.")
                    break

                # X pressed: start camera
                if controller.state.camera_active and not camera_running:
                    camera.start()
                    camera_running = True
                    if args.save_video and video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(Path(args.save_video)), fourcc, args.fps,
                            (args.width, args.height)
                        )
                    print("[main] Camera + detection STARTED")

                # Y pressed: stop camera
                if not controller.state.camera_active and camera_running:
                    camera.stop()
                    camera_running = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                        print(f"[main] Video saved to {args.save_video}")
                    if not headless:
                        cv2.destroyAllWindows()
                    print("[main] Camera + detection STOPPED")

            # ---- If camera is not active, just keep polling controller ----
            if not camera_running:
                time.sleep(0.01)
                continue

            # ---- Camera + detection loop ----
            cam_data = camera.get_latest()
            if cam_data is None:
                time.sleep(0.005)
                continue

            timestamp, frame = cam_data

            t0 = time.perf_counter()
            detections = detector.predict(frame)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            # ---- Draw + display + save ----
            if not headless or video_writer:
                draw_overlay(frame, detections, fps)

            if video_writer:
                video_writer.write(frame)

            if headless:
                if now - last_status_t >= 1.0:
                    print(f"[status] fps={fps:.1f} det={len(detections)} infer_ms={inference_ms:.1f}")
                    last_status_t = now
            else:
                cv2.imshow("Main Runtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if camera_running:
            camera.stop()
        if video_writer:
            video_writer.release()
            print(f"[main] Video saved to {args.save_video}")
        if controller:
            controller.stop()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
