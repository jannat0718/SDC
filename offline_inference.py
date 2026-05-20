#!/usr/bin/env python3
"""
Offline video inference using a YOLO26 OpenVINO model.

Usage examples:
    # Basic usage (auto-finds .xml model in Model/ and class names from metadata.yaml)
    python offline_inference.py --video "data/SDC_March_Sessions/DATA/20260305_150938/Left20260305150938.mp4"

    # Specify model explicitly
    python offline_inference.py --model "Model/best_openvino_model-20260401T075819Z-3-001/best_openvino_model/best.xml" --video "data/SDC_March_Sessions/DATA/20260305_150938/Left20260305150938.mp4"

    # With explicit class names and custom thresholds
    python offline_inference.py --video "data/SDC_March_Sessions/DATA/20260305_150938/Left20260305150938.mp4" --class-names Model/classes.txt --conf 0.4

    # Headless mode (no preview window, just process and save)
    python offline_inference.py --video "data/SDC_March_Sessions/DATA/20260305_150938/Left20260305150938.mp4" --headless
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from openvino import Core
except ImportError:
    Core = None


BBox = Tuple[int, int, int, int]
Detection = Tuple[str, float, BBox]

# Color palette for drawing different classes
COLORS = [
    (10, 200, 255), (255, 50, 50), (50, 255, 50), (255, 100, 200),
    (200, 100, 255), (100, 255, 255), (255, 255, 100), (0, 128, 255),
]


# --------------- LiDAR helpers ---------------

def load_lidar_scans(csv_path: str) -> List[List[Tuple[float, float]]]:
    """Load an RPLidar CSV and split it into per-revolution scans.

    Each CSV row is: quality, angle_deg, distance_mm
    A new revolution starts when the angle wraps (decreases).
    Returns a list of scans, each scan being a list of (angle_deg, distance_mm) tuples.
    """
    scans: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    prev_angle = -1.0

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                angle = float(parts[1])
                dist = float(parts[2])
            except ValueError:
                continue
            if angle < prev_angle - 10:  # wrap-around
                if current:
                    scans.append(current)
                current = []
            current.append((angle, dist))
            prev_angle = angle
    if current:
        scans.append(current)
    return scans


def get_lidar_for_frame(
    lidar_scans: List[List[Tuple[float, float]]],
    frame_idx: int,
    total_frames: int,
) -> List[Tuple[float, float]]:
    """Pick the LiDAR scan closest to the current video frame by linear interpolation."""
    if not lidar_scans:
        return []
    ratio = frame_idx / max(total_frames - 1, 1)
    scan_idx = int(ratio * (len(lidar_scans) - 1))
    scan_idx = max(0, min(scan_idx, len(lidar_scans) - 1))
    return lidar_scans[scan_idx]


def draw_lidar_minimap(
    frame: np.ndarray,
    scan: List[Tuple[float, float]],
    radius: int = 90,
    margin: int = 15,
    max_dist_mm: float = 6000.0,
    forward_offset_deg: float = 0.0,
    camera_fov_deg: float = 62.0,
) -> np.ndarray:
    """Draw a top-down LiDAR minimap in the bottom-left corner of the frame."""
    h, w = frame.shape[:2]
    cx = margin + radius
    cy = h - margin - radius

    # Solid white circle background
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), radius, (180, 180, 180), 1)

    # Draw camera FOV cone (light gray on white)
    half_fov = camera_fov_deg / 2.0
    fov_left_rad = np.deg2rad(-half_fov - 90)   # left edge of camera view
    fov_right_rad = np.deg2rad(half_fov - 90)    # right edge of camera view
    fov_r = radius - 2
    pts = [np.array([cx, cy], dtype=np.int32)]
    for a in np.linspace(fov_left_rad, fov_right_rad, 20):
        pts.append(np.array([int(cx + fov_r * np.cos(a)),
                             int(cy + fov_r * np.sin(a))], dtype=np.int32))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(frame, [pts], (230, 230, 230))

    # Draw points (corrected orientation: forward = up)
    for angle_deg, dist_mm in scan:
        if dist_mm <= 0:
            continue
        r = min(dist_mm / max_dist_mm, 1.0) * (radius - 4)
        # forward_offset_deg - angle_deg gives CW angle from forward,
        # then -90 rotates so forward = up on screen
        rad = np.deg2rad(forward_offset_deg - angle_deg - 90)
        px = int(cx + r * np.cos(rad))
        py = int(cy + r * np.sin(rad))
        # Color by distance: close=red, far=green
        g = int(min(dist_mm / max_dist_mm, 1.0) * 255)
        b_color = (0, g, 255 - g)
        cv2.circle(frame, (px, py), 2, b_color, -1)

    # Car position dot
    cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)
    cv2.putText(frame, "LiDAR", (cx - 22, cy + radius + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def match_lidar_distances(
    detections: List[Detection],
    scan: List[Tuple[float, float]],
    frame_w: int,
    camera_fov_deg: float = 62.0,
    lidar_forward_deg: float = 0.0,
    min_tolerance_deg: float = 8.0,
    min_dist_mm: float = 150.0,
) -> List[Optional[float]]:
    """Estimate distance to each detection using median-filtered LiDAR matching.

    For each bounding box:
      1. Map the horizontal center to a LiDAR angle.
      2. Collect all LiDAR points within the angular window.
      3. Apply ground-plane filter (discard readings < min_dist_mm).
      4. Return the **median** distance of the remaining points.
    """
    if not scan:
        return [None] * len(detections)

    half_w = frame_w / 2.0
    half_fov = camera_fov_deg / 2.0

    distances: List[Optional[float]] = []
    for label, conf, (x1, y1, x2, y2) in detections:
        # Horizontal center of bounding box
        cx = (x1 + x2) / 2.0
        # Relative position: -1 (left) to +1 (right)
        rel_x = (cx - half_w) / half_w
        # Camera angle in degrees (positive = right in image)
        cam_angle = rel_x * half_fov
        # Map to LiDAR angle space (RPLidar angles increase CCW, so negate)
        lidar_angle = (lidar_forward_deg - cam_angle) % 360.0

        # Use bbox angular width as search window (at least min_tolerance_deg)
        bbox_angle_span = ((x2 - x1) / frame_w) * camera_fov_deg
        tolerance = max(bbox_angle_span / 2.0, min_tolerance_deg)

        # Collect all LiDAR readings within the angular window
        candidates: List[float] = []
        for angle_deg, dist_mm in scan:
            if dist_mm <= 0:
                continue
            # Ground-plane filter: ignore very close readings (floor hits)
            if dist_mm < min_dist_mm:
                continue
            diff = abs(angle_deg - lidar_angle)
            if diff > 180:
                diff = 360 - diff
            if diff <= tolerance:
                candidates.append(dist_mm)

        # Median filter: robust against outliers
        if candidates:
            candidates.sort()
            mid = len(candidates) // 2
            if len(candidates) % 2 == 1:
                distances.append(candidates[mid])
            else:
                distances.append((candidates[mid - 1] + candidates[mid]) / 2.0)
        else:
            distances.append(None)
    return distances


class DistanceSmoother:
    """Temporal smoothing using Exponential Moving Average (EMA) per object label.

    Tracks smoothed distances keyed by (label, rough_angle_bin) to handle
    multiple instances of the same class at different positions.
    alpha controls responsiveness: 0.0 = very smooth, 1.0 = no smoothing.
    """

    def __init__(self, alpha: float = 0.35, max_age: int = 10):
        self.alpha = alpha
        self.max_age = max_age
        # key -> (smoothed_dist_mm, frames_since_update)
        self._tracks: dict = {}

    def _make_key(self, label: str, bbox: Tuple[int, int, int, int], frame_w: int) -> str:
        # Bin by label + horizontal position (left/center/right third)
        cx = (bbox[0] + bbox[2]) / 2.0
        region = int(cx / frame_w * 3)  # 0=left, 1=center, 2=right
        return f"{label}_{region}"

    def smooth(
        self,
        detections: List[Detection],
        raw_distances: List[Optional[float]],
        frame_w: int,
    ) -> List[Optional[float]]:
        # Age all existing tracks
        for k in list(self._tracks):
            self._tracks[k] = (self._tracks[k][0], self._tracks[k][1] + 1)
            if self._tracks[k][1] > self.max_age:
                del self._tracks[k]

        smoothed: List[Optional[float]] = []
        for i, (label, conf, bbox) in enumerate(detections):
            raw = raw_distances[i] if i < len(raw_distances) else None
            key = self._make_key(label, bbox, frame_w)

            if raw is not None:
                if key in self._tracks:
                    prev = self._tracks[key][0]
                    val = self.alpha * raw + (1 - self.alpha) * prev
                else:
                    val = raw
                self._tracks[key] = (val, 0)
                smoothed.append(val)
            elif key in self._tracks:
                # No new reading — use last smoothed value
                smoothed.append(self._tracks[key][0])
            else:
                smoothed.append(None)

        return smoothed


# --------------- end LiDAR helpers ---------------


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


def load_metadata_class_names(model_path: str) -> List[str]:
    """Try to load class names from metadata.yaml next to the model."""
    metadata_path = Path(model_path).parent / "metadata.yaml"
    if not metadata_path.exists():
        return []
    try:
        import yaml
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        names_dict = meta.get("names", {})
        if isinstance(names_dict, dict):
            return [names_dict[k] for k in sorted(names_dict.keys())]
    except Exception as e:
        print(f"[WARN] Could not parse metadata.yaml: {e}")
    return []


def find_openvino_model(model_dir: str) -> str:
    # Search current dir and subdirs for .xml files
    xml_files = glob.glob(os.path.join(model_dir, "**", "*.xml"), recursive=True)
    if len(xml_files) == 1:
        return xml_files[0]
    elif len(xml_files) > 1:
        # Prefer "best.xml" if it exists
        for f in xml_files:
            if "best" in os.path.basename(f).lower():
                return f
        return xml_files[0]
    raise FileNotFoundError(f"No .xml model file found in {model_dir} (searched subdirs too)")


class OpenVINODetector:
    """YOLO26 detector using OpenVINO runtime."""

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "AUTO",
    ):
        if Core is None:
            raise RuntimeError(
                "OpenVINO is not installed. Install with: pip install openvino"
            )

        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        core = Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.input_layer = self.compiled_model.input(0)

        # Detect output layers
        self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]
        self.num_outputs = len(self.output_layers)

        # Model input shape: [batch, channels, height, width]
        input_shape = self.input_layer.shape
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        # Detect if end2end model (built-in NMS)
        self.end2end = self.num_outputs > 1 or (
            self.num_outputs == 1 and len(self.output_layers[0].shape) == 3
            and self.output_layers[0].shape[-1] == 6
        )

        print(f"[Model] Loaded: {model_path}")
        print(f"[Model] Input size: {self.input_w}x{self.input_h}")
        print(f"[Model] Outputs: {self.num_outputs} | End2End NMS: {self.end2end}")
        print(f"[Model] Device: {device}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def _decode_end2end(self, outputs: dict, frame_w: int, frame_h: int) -> List[Detection]:
        """Decode end2end YOLO26 output (NMS built into model).
        Output is typically (1, N, 6) with [x1, y1, x2, y2, score, class_id].
        """
        # Get the primary output
        output = np.asarray(outputs[self.output_layers[0]])
        if output.ndim == 3:
            output = output[0]  # Remove batch dim -> (N, 6)

        scale_x = frame_w / self.input_w
        scale_y = frame_h / self.input_h

        detections: List[Detection] = []
        for row in output:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            score = float(score)
            if score < self.conf_threshold:
                continue

            x1 = int(max(0, min(frame_w - 1, x1 * scale_x)))
            y1 = int(max(0, min(frame_h - 1, y1 * scale_y)))
            x2 = int(max(0, min(frame_w - 1, x2 * scale_x)))
            y2 = int(max(0, min(frame_h - 1, y2 * scale_y)))
            class_idx = int(class_id)

            if x2 <= x1 or y2 <= y1:
                continue

            label = self.class_names[class_idx] if self.class_names and 0 <= class_idx < len(self.class_names) else f"class_{class_idx}"
            detections.append((label, score, (x1, y1, x2, y2)))

        return detections

    def _decode_standard(self, output: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        """Decode standard YOLO output (non-end2end): shape (1, 4+num_classes, num_detections)."""
        if output.ndim == 3:
            output = output[0]

        # YOLOv8 output: rows = [cx, cy, w, h, class0_conf, class1_conf, ...]
        if output.shape[0] < output.shape[1]:
            output = output.T

        num_attrs = output.shape[1]
        if num_attrs < 5:
            return []

        boxes_xyxy: List[List[int]] = []
        scores: List[float] = []
        labels: List[str] = []

        scale_x = frame_w / self.input_w
        scale_y = frame_h / self.input_h

        for i in range(output.shape[0]):
            row = output[i]
            class_probs = row[4:]
            class_idx = int(np.argmax(class_probs))
            score = float(class_probs[class_idx])

            if score < self.conf_threshold:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]

            x1 = int((cx - bw / 2.0) * scale_x)
            y1 = int((cy - bh / 2.0) * scale_y)
            x2 = int((cx + bw / 2.0) * scale_x)
            y2 = int((cy + bh / 2.0) * scale_y)

            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            label = self.class_names[class_idx] if self.class_names and 0 <= class_idx < len(self.class_names) else f"class_{class_idx}"
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)

        if not boxes_xyxy:
            return []

        # NMS
        boxes_for_nms = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_xyxy]
        nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, self.conf_threshold, self.iou_threshold)

        detections: List[Detection] = []
        if len(nms_indices) > 0:
            for i in np.array(nms_indices).reshape(-1).tolist():
                detections.append((labels[i], float(scores[i]), tuple(boxes_xyxy[i])))
        return detections

    def predict(self, frame: np.ndarray) -> List[Detection]:
        blob = self._preprocess(frame)
        results = self.compiled_model({self.input_layer: blob})
        frame_h, frame_w = frame.shape[:2]

        if self.end2end:
            return self._decode_end2end(results, frame_w, frame_h)
        else:
            output = np.asarray(results[self.output_layers[0]])
            return self._decode_standard(output, frame_w, frame_h)


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    fps: float,
    distances: Optional[List[Optional[float]]] = None,
) -> np.ndarray:
    for i, (label, conf, (x1, y1, x2, y2)) in enumerate(detections):
        # Pick color based on label hash
        color_idx = hash(label) % len(COLORS)
        color = COLORS[color_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label text (include distance if available)
        text = f"{label} {conf:.2f}"
        dist_mm = distances[i] if distances and i < len(distances) else None
        if dist_mm is not None:
            dist_m = dist_mm / 1000.0
            text += f" {dist_m:.2f}m"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, text,
            (x1, max(th + 4, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.putText(
        frame, f"Detections: {len(detections)}",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline video inference with YOLO26 OpenVINO model"
    )

    parser.add_argument("--model", type=str, default=None,
                        help="Path to OpenVINO .xml model file (auto-detects from Model/ if omitted)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to directory of image frames (sorted alphabetically)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output video (default: Output/<input_name>_detected.mp4)")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Path to class names .txt file (one class per line)")

    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--device", type=str, default="AUTO",
                        help="OpenVINO device: AUTO, CPU, GPU")

    parser.add_argument("--lidar-dir", type=str, default=None,
                        help="Path to dir with Lidar CSVs, or 'auto' to find in same folder as video")
    parser.add_argument("--camera-fov", type=float, default=62.0,
                        help="Horizontal field-of-view of the camera in degrees (default: 62)")
    parser.add_argument("--lidar-forward", type=float, default=180.0,
                        help="LiDAR angle (deg) that corresponds to camera forward direction (default: 180)")

    parser.add_argument("--headless", action="store_true",
                        help="No preview window, just process and save")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output video (preview only)")

    return parser.parse_args()


def load_image_sequence(image_dir: str) -> List[str]:
    """Load sorted image file paths from a directory."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith(extensions):
            files.append(os.path.join(image_dir, f))
    return files


def main() -> None:
    args = parse_args()

    if not args.video and not args.image_dir:
        print("[ERROR] Must provide either --video or --image-dir")
        sys.exit(1)

    # Resolve project root (directory containing this script)
    project_root = Path(__file__).resolve().parent

    # --- Model ---
    if args.model:
        model_path = args.model
    else:
        model_dir = project_root / "Model"
        model_path = find_openvino_model(str(model_dir))
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    # --- Determine input mode ---
    use_image_dir = False
    image_files: List[str] = []

    if args.image_dir:
        img_dir = Path(args.image_dir)
        if not img_dir.is_dir():
            print(f"[ERROR] Image directory not found: {args.image_dir}")
            sys.exit(1)
        image_files = load_image_sequence(str(img_dir))
        if not image_files:
            print(f"[ERROR] No image files found in: {args.image_dir}")
            sys.exit(1)
        use_image_dir = True
        input_name = img_dir.name
        print(f"[Input] Image sequence: {len(image_files)} frames from {img_dir}")
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            sys.exit(1)
        input_name = video_path.stem

    # --- Output path ---
    output_dir = project_root / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"{input_name}_detected.mp4"

    # --- Class names (from arg, or auto from metadata.yaml) ---
    class_names = load_class_names(args.class_names)
    if not class_names:
        class_names = load_metadata_class_names(model_path)
        if class_names:
            print(f"[Classes] Auto-loaded from metadata.yaml")
    if class_names:
        print(f"[Classes] {len(class_names)}: {class_names}")

    # --- Init detector ---
    detector = OpenVINODetector(
        model_path=model_path,
        class_names=class_names,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # --- Setup frame source ---
    cap = None
    if use_image_dir:
        # Read first image to get dimensions
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            print(f"[ERROR] Could not read first image: {image_files[0]}")
            sys.exit(1)
        frame_h, frame_w = first_frame.shape[:2]
        fps_video = args.fps or 30.0
        total_frames = len(image_files)
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            sys.exit(1)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Source] {frame_w}x{frame_h} @ {fps_video:.1f} FPS, {total_frames} frames")

    # --- LiDAR ---
    lidar_scans: List[List[Tuple[float, float]]] = []
    if args.lidar_dir:
        lidar_dir_path = args.lidar_dir
        if lidar_dir_path.lower() == "auto":
            if args.video:
                lidar_dir_path = str(Path(args.video).parent)
            elif args.image_dir:
                lidar_dir_path = args.image_dir
            else:
                lidar_dir_path = None
        if lidar_dir_path and Path(lidar_dir_path).is_dir():
            csv_files = sorted(glob.glob(os.path.join(lidar_dir_path, "Lidar_*.csv")))
            if csv_files:
                print(f"[LiDAR] Found {len(csv_files)} CSV file(s) in {lidar_dir_path}")
                for csv_f in csv_files:
                    lidar_scans.extend(load_lidar_scans(csv_f))
                print(f"[LiDAR] Loaded {len(lidar_scans)} scans total")
            else:
                print(f"[LiDAR] No Lidar_*.csv files found in {lidar_dir_path}")
        else:
            print(f"[LiDAR] Directory not found: {lidar_dir_path}")

    # --- Determine output FPS ---
    if args.fps:
        output_fps = args.fps
    else:
        output_fps = fps_video
    print(f"[Output FPS] {output_fps:.1f}")

    # --- Output writer ---
    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_w, frame_h))
        print(f"[Output] Saving to: {output_path}")

    # --- Process frames ---
    frame_id = 0
    prev_t = time.time()
    total_detections = 0
    dist_smoother = DistanceSmoother(alpha=0.35, max_age=10) if lidar_scans else None

    try:
        while True:
            # Read frame from video or image sequence
            if use_image_dir:
                if frame_id >= len(image_files):
                    break
                frame = cv2.imread(image_files[frame_id])
                if frame is None:
                    frame_id += 1
                    continue
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            t0 = time.perf_counter()
            detections = detector.predict(frame)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            total_detections += len(detections)

            # Estimate distances from LiDAR
            distances: Optional[List[Optional[float]]] = None
            if lidar_scans:
                scan = get_lidar_for_frame(lidar_scans, frame_id, total_frames)
                if scan:
                    raw_distances = match_lidar_distances(
                        detections, scan, frame_w,
                        camera_fov_deg=args.camera_fov,
                        lidar_forward_deg=args.lidar_forward,
                    )
                    distances = dist_smoother.smooth(detections, raw_distances, frame_w)

            annotated = draw_detections(frame.copy(), detections, fps, distances)

            # LiDAR minimap overlay
            if lidar_scans:
                if not distances:  # scan not yet fetched
                    scan = get_lidar_for_frame(lidar_scans, frame_id, total_frames)
                if scan:
                    annotated = draw_lidar_minimap(annotated, scan,
                                                   forward_offset_deg=args.lidar_forward,
                                                   camera_fov_deg=args.camera_fov)

            if writer:
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("Offline Inference - Press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Stopped by user.")
                    break

            frame_id += 1
            if frame_id % 100 == 0:
                pct = (frame_id / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_id}/{total_frames} ({pct:.1f}%) | "
                      f"Inference: {inference_ms:.1f} ms | Detections: {len(detections)}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\n[Done] Processed {frame_id} frames, {total_detections} total detections.")
    if writer and not args.no_save:
        print(f"[Done] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
