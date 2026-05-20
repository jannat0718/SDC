import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception.object_detector import Detection
from perception.sensors import LidarSample


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

        if save_json:
            json_path = self.log_dir / f"detections_{ts}.jsonl"
            self.json_fp = json_path.open("w", encoding="utf-8")

    @staticmethod
    def _lidar_stats(lidar: Optional[LidarSample]) -> Tuple[int, Optional[float]]:
        if lidar is None or lidar.points is None or lidar.points.size == 0:
            return 0, None
        return int(lidar.points.shape[0]), float(np.min(lidar.points[:, 2]))

    def log_frame(
        self,
        timestamp: float,
        frame_id: int,
        detections: List[Detection],
        lidar: Optional[LidarSample],
        fps: float,
        inference_ms: float,
        lane_state: Optional[Dict[str, Any]] = None,
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
            payload = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "fps": fps,
                "inference_ms": inference_ms,
                "lidar_points": lidar_points,
                "lidar_min_distance_mm": lidar_min_distance_mm,
                "lane_state": lane_state,
                "detections": [
                    {"label": label, "confidence": conf, "bbox_xyxy": [x1, y1, x2, y2]}
                    for label, conf, (x1, y1, x2, y2) in detections
                ],
            }
            self.json_fp.write(json.dumps(payload) + "\n")
            self.json_fp.flush()

    def close(self) -> None:
        if self.csv_fp:
            self.csv_fp.close()
        if self.json_fp:
            self.json_fp.close()


class ControlLogger:
    def __init__(self, log_dir: Optional[str], enabled: bool):
        self.enabled = enabled and bool(log_dir)
        self.fp = None
        self.writer = None

        if not self.enabled:
            return

        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = p / f"control_{ts}.csv"
        self.fp = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fp)
        self.writer.writerow(
            [
                "timestamp",
                "cmd_steering",
                "cmd_throttle",
                "feedback_brake",
                "feedback_steering",
                "feedback_steering_speed",
                "feedback_throttle",
                "feedback_steering_sensor",
            ]
        )

    def log(self, timestamp: float, control_state: Any, feedback: Dict[str, Any]) -> None:
        if not self.enabled or self.writer is None:
            return

        self.writer.writerow(
            [
                f"{timestamp:.6f}",
                getattr(control_state, "steering_command", ""),
                getattr(control_state, "throttle_command", ""),
                feedback.get("feedback_brake", ""),
                feedback.get("feedback_steering", ""),
                feedback.get("feedback_steering_speed", ""),
                feedback.get("feedback_throttle", ""),
                feedback.get("feedback_steering_sensor", ""),
            ]
        )
        self.fp.flush()

    def close(self) -> None:
        if self.fp:
            self.fp.close()
