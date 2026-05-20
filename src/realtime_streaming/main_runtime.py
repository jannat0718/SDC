#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
import cv2

from control.manual_xbox_control import XboxCanController
from perception.object_detector import OnnxDetector, load_class_names, find_openvino_model
from perception.sensors import CameraStream

def draw_overlay(frame, detections, fps):
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 200, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 200, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--camera-index", type=int, default=0) # Changed to 0 for laptop/USB default
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disable-control", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    model_path = args.model if args.model else find_openvino_model(str(script_dir.parent.parent / "Model"))
    
    detector = OnnxDetector(model_path=model_path, class_names=load_class_names(None))
    camera = CameraStream(args.camera_index, args.width, args.height, args.fps)
    camera_running = False

    controller = None
    if not args.disable_control:
        controller = XboxCanController()
        controller.start()

    print("[Ready] Press X on Xbox Controller to start detection. Press B to Exit.")

    prev_t = time.time()
    try:
        while True:
            if controller:
                ctrl_data = controller.step()
                if controller.state.emergency_stop: break
                
                # Logic to start/stop camera based on X/Y buttons
                if controller.state.camera_active and not camera_running:
                    camera.start()
                    camera_running = True
                elif not controller.state.camera_active and camera_running:
                    camera.stop()
                    camera_running = False

            if not camera_running:
                time.sleep(0.01)
                continue

            cam_data = camera.get_latest()
            if cam_data:
                ts, frame = cam_data
                detections = detector.predict(frame)
                now = time.time()
                fps = 1.0 / (now - prev_t)
                prev_t = now
                
                draw_overlay(frame, detections, fps)
                cv2.imshow("SDC 2026 - Perception", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        if camera_running: camera.stop()
        if controller: controller.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()