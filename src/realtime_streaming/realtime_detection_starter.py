"""
Real-time camera + LiDAR pipeline for NUC3 Ubuntu deployment.

Features:
- ONNX model inference (YOLO-like output support)
- Camera and LiDAR startup self-checks
- Detection logging to CSV and JSONL for training/evaluation
"""

import argparse
import csv
import glob
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None


BBox = Tuple[int, int, int, int]
Detection = Tuple[str, float, BBox]


@dataclass
class LidarSample:
    timestamp: float
    points: np.ndarray  # shape: (N, 3): quality, angle_deg, distance_mm


def resolve_lidar_port(requested_port: str) -> str:
    if requested_port and requested_port.lower() != "auto":
        return requested_port

    linux_candidates = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if linux_candidates:
        print(f"[LiDAR] auto-selected port: {linux_candidates[0]}")
        return linux_candidates[0]

    raise RuntimeError("Could not auto-detect LiDAR port. Pass --lidar-port explicitly.")


def load_class_names(path: Optional[str]) -> List[str]:
    if not path:
        return []

    names_path = Path(path)
    if not names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    names: List[str] = []
    with names_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                names.append(item)
    return names


class CameraStream:
    def __init__(self, camera_index: int = 0, width: int = 848, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest: Optional[Tuple[float, np.ndarray]] = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._latest_lock:
                self._latest = (time.time(), frame)

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._latest_lock:
            if self._latest is None:
                return None
            ts, frame = self._latest
            return ts, frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


class LidarStream:
    def __init__(self, port: str, max_queue_size: int = 3):
        self.port = port
        self.max_queue_size = max_queue_size

        self._lidar: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[LidarSample] = queue.Queue(maxsize=max_queue_size)

    def start(self) -> None:
        if RPLidar is None:
            raise RuntimeError("rplidar package is not installed. Install with: pip install rplidar-roboticia")

        port = resolve_lidar_port(self.port)
        self._lidar = RPLidar(port, timeout=3)
        self._lidar.start_motor()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._lidar is not None
        try:
            for scan in self._lidar.iter_scans(max_buf_meas=1000):
                if not self._running:
                    break

                points = np.array(scan, dtype=np.float32)
                sample = LidarSample(timestamp=time.time(), points=points)

                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

                self._queue.put_nowait(sample)
        except Exception as exc:
            print(f"[LiDAR] Reader stopped: {exc}")

    def get_latest(self) -> Optional[LidarSample]:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass


class OnnxDetector:
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self.model_path = model_path
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_h, self.input_w = self._resolve_model_size(self.input_shape)

    @staticmethod
    def _resolve_model_size(input_shape: List[Any]) -> Tuple[int, int]:
        if len(input_shape) != 4:
            return 640, 640

        h_val = input_shape[2]
        w_val = input_shape[3]

        h = int(h_val) if isinstance(h_val, int) else 640
        w = int(w_val) if isinstance(w_val, int) else 640
        return h, w

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def _decode_common_yolo(self, output: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        if output.ndim == 3:
            preds = output[0]
        elif output.ndim == 2:
            preds = output
        else:
            return []

        boxes_xyxy: List[List[int]] = []
        scores: List[float] = []
        labels: List[str] = []

        for row in preds:
            if row.shape[0] < 6:
                continue

            # Case A: [x1, y1, x2, y2, score, class_id]
            if row.shape[0] == 6:
                x1, y1, x2, y2, score, cls_id = row[:6]
                score = float(score)
                if score < self.conf_threshold:
                    continue

                x1 = int(max(0, min(frame_w - 1, x1 * frame_w / self.input_w)))
                y1 = int(max(0, min(frame_h - 1, y1 * frame_h / self.input_h)))
                x2 = int(max(0, min(frame_w - 1, x2 * frame_w / self.input_w)))
                y2 = int(max(0, min(frame_h - 1, y2 * frame_h / self.input_h)))

                class_idx = int(cls_id)
            else:
                # Case B: YOLO-style [cx, cy, w, h, obj_conf, class_probs...]
                cx, cy, bw, bh, obj_conf = row[:5]
                class_probs = row[5:]
                if class_probs.size == 0:
                    continue

                class_idx = int(np.argmax(class_probs))
                class_conf = float(class_probs[class_idx])
                score = float(obj_conf) * class_conf
                if score < self.conf_threshold:
                    continue

                x1 = int((cx - bw / 2.0) * frame_w / self.input_w)
                y1 = int((cy - bh / 2.0) * frame_h / self.input_h)
                x2 = int((cx + bw / 2.0) * frame_w / self.input_w)
                y2 = int((cy + bh / 2.0) * frame_h / self.input_h)

                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w - 1, x2))
                y2 = max(0, min(frame_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            if self.class_names and 0 <= class_idx < len(self.class_names):
                label = self.class_names[class_idx]
            else:
                label = f"cls_{class_idx}"

            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)

        if not boxes_xyxy:
            return []

        boxes_for_nms = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_xyxy]
        nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, self.conf_threshold, self.iou_threshold)

        detections: List[Detection] = []
        if len(nms_indices) > 0:
            idxs = np.array(nms_indices).reshape(-1).tolist()
            for i in idxs:
                detections.append((labels[i], float(scores[i]), tuple(boxes_xyxy[i])))

        return detections

    def predict(self, frame: np.ndarray, lidar: Optional[LidarSample]) -> List[Detection]:
        _ = lidar
        blob = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        if not outputs:
            return []

        frame_h, frame_w = frame.shape[:2]
        for out in outputs:
            detections = self._decode_common_yolo(np.asarray(out), frame_w, frame_h)
            if detections:
                return detections

        return []


class DetectionLogger:
    def __init__(self, log_dir: Optional[str], save_csv: bool, save_json: bool):
        self.enabled = bool(log_dir) and (save_csv or save_json)
        self.csv_fp = None
        self.csv_writer: Optional[csv.writer] = None
        self.json_fp = None

        if not self.enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        if save_csv:
            csv_path = self.log_dir / f"detections_{ts}.csv"
            self.csv_fp = csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_fp)
            self.csv_writer.writerow(
                [
                    "timestamp",
                    "frame_id",
                    "label",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "lidar_points",
                    "lidar_min_distance_mm",
                    "fps",
                    "inference_ms",
                ]
            )
            self.csv_fp.flush()
            print(f"[log] CSV -> {csv_path}")

        if save_json:
            json_path = self.log_dir / f"detections_{ts}.jsonl"
            self.json_fp = json_path.open("w", encoding="utf-8")
            print(f"[log] JSONL -> {json_path}")

    @staticmethod
    def _lidar_stats(lidar: Optional[LidarSample]) -> Tuple[int, Optional[float]]:
        if lidar is None or lidar.points is None or lidar.points.size == 0:
            return 0, None

        n_points = int(lidar.points.shape[0])
        min_dist = float(np.min(lidar.points[:, 2]))
        return n_points, min_dist

    def log_frame(
        self,
        timestamp: float,
        frame_id: int,
        detections: List[Detection],
        lidar: Optional[LidarSample],
        fps: float,
        inference_ms: float,
    ) -> None:
        if not self.enabled:
            return

        lidar_points, lidar_min_distance_mm = self._lidar_stats(lidar)

        if self.csv_writer:
            if not detections:
                self.csv_writer.writerow(
                    [
                        f"{timestamp:.6f}",
                        frame_id,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        lidar_points,
                        "" if lidar_min_distance_mm is None else f"{lidar_min_distance_mm:.2f}",
                        f"{fps:.3f}",
                        f"{inference_ms:.3f}",
                    ]
                )
            else:
                for label, conf, (x1, y1, x2, y2) in detections:
                    self.csv_writer.writerow(
                        [
                            f"{timestamp:.6f}",
                            frame_id,
                            label,
                            f"{conf:.6f}",
                            x1,
                            y1,
                            x2,
                            y2,
                            lidar_points,
                            "" if lidar_min_distance_mm is None else f"{lidar_min_distance_mm:.2f}",
                            f"{fps:.3f}",
                            f"{inference_ms:.3f}",
                        ]
                    )
            self.csv_fp.flush()

        if self.json_fp:
            payload: Dict[str, Any] = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "fps": fps,
                "inference_ms": inference_ms,
                "lidar_points": lidar_points,
                "lidar_min_distance_mm": lidar_min_distance_mm,
                "detections": [
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                    for (label, conf, (x1, y1, x2, y2)) in detections
                ],
            }
            self.json_fp.write(json.dumps(payload) + "\n")
            self.json_fp.flush()

    def close(self) -> None:
        if self.csv_fp:
            self.csv_fp.close()
        if self.json_fp:
            self.json_fp.close()


class RealTimePerceptionApp:
    def __init__(
        self,
        camera_index: int,
        lidar_port: str,
        width: int,
        height: int,
        fps: int,
        headless: bool,
        model_path: str,
        class_names_path: Optional[str],
        conf_threshold: float,
        iou_threshold: float,
        log_dir: Optional[str],
        log_csv: bool,
        log_json: bool,
    ):
        self.camera = CameraStream(camera_index=camera_index, width=width, height=height, fps=fps)
        self.lidar = LidarStream(port=lidar_port)
        class_names = load_class_names(class_names_path)
        self.detector = OnnxDetector(
            model_path=model_path,
            class_names=class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self.logger = DetectionLogger(log_dir=log_dir, save_csv=log_csv, save_json=log_json)

        self._last_lidar: Optional[LidarSample] = None
        self.headless = headless

    @staticmethod
    def startup_self_check(camera_index: int, lidar_port: str) -> None:
        print("[check] Verifying camera...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Startup check failed: camera index {camera_index} is not available.")

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Startup check failed: camera opened but no frame received.")

        print("[check] Camera OK")

        print("[check] Verifying LiDAR...")
        if RPLidar is None:
            raise RuntimeError("Startup check failed: rplidar package is not installed.")

        resolved_port = resolve_lidar_port(lidar_port)
        lidar = None
        try:
            lidar = RPLidar(resolved_port, timeout=3)
            info = lidar.get_info()
            health = lidar.get_health()
            model = info.get("model", "unknown") if isinstance(info, dict) else "unknown"
            status = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
            print(f"[check] LiDAR OK on {resolved_port} | model={model} | status={status}")
        finally:
            if lidar is not None:
                try:
                    lidar.disconnect()
                except Exception:
                    pass

    @staticmethod
    def _draw_overlay(frame: np.ndarray, detections: List[Detection], lidar: Optional[LidarSample]) -> None:
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

        if lidar is None:
            cv2.putText(frame, "LiDAR: waiting...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 180, 255), 2)
        else:
            n_points = int(lidar.points.shape[0]) if lidar.points is not None else 0
            age_ms = (time.time() - lidar.timestamp) * 1000.0
            cv2.putText(
                frame,
                f"LiDAR points: {n_points} | age: {age_ms:.0f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 180, 255),
                2,
            )

    def run(self) -> None:
        self.camera.start()
        self.lidar.start()

        frame_id = 0
        prev_t = time.time()
        last_status_t = 0.0

        try:
            while True:
                cam_data = self.camera.get_latest()
                if cam_data is None:
                    time.sleep(0.005)
                    continue

                timestamp, frame = cam_data
                lidar_now = self.lidar.get_latest()
                if lidar_now is not None:
                    self._last_lidar = lidar_now

                t0 = time.perf_counter()
                detections = self.detector.predict(frame, self._last_lidar)
                inference_ms = (time.perf_counter() - t0) * 1000.0

                self._draw_overlay(frame, detections, self._last_lidar)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

                self.logger.log_frame(
                    timestamp=timestamp,
                    frame_id=frame_id,
                    detections=detections,
                    lidar=self._last_lidar,
                    fps=fps,
                    inference_ms=inference_ms,
                )
                frame_id += 1

                if self.headless:
                    if now - last_status_t >= 1.0:
                        print(
                            f"[status] fps={fps:.1f} infer_ms={inference_ms:.1f} detections={len(detections)}"
                        )
                        last_status_t = now
                else:
                    cv2.imshow("Real-Time Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.camera.stop()
            self.lidar.stop()
            self.logger.close()
            if not self.headless:
                cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time camera + LiDAR + ONNX object detection")
    parser.add_argument("--onnx-model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--class-names", type=str, default="", help="Optional text file with class names")

    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--lidar-port", type=str, default="auto", help="LiDAR serial port or 'auto'")
    parser.add_argument("--width", type=int, default=848, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")

    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")

    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for detection logs")
    parser.add_argument("--log-csv", action="store_true", help="Enable CSV logging")
    parser.add_argument("--log-json", action="store_true", help="Enable JSONL logging")

    parser.add_argument("--headless", action="store_true", help="Run without OpenCV window")
    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="Skip camera/LiDAR startup checks",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.onnx_model).exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_model}")

    auto_headless = (os.name != "nt") and (os.environ.get("DISPLAY") is None)
    headless = args.headless or auto_headless

    if not args.skip_self_check:
        RealTimePerceptionApp.startup_self_check(args.camera_index, args.lidar_port)

    app = RealTimePerceptionApp(
        camera_index=args.camera_index,
        lidar_port=args.lidar_port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        headless=headless,
        model_path=args.onnx_model,
        class_names_path=args.class_names if args.class_names else None,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        log_dir=args.log_dir,
        log_csv=args.log_csv,
        log_json=args.log_json,
    )
    app.run()


if __name__ == "__main__":
    main()
