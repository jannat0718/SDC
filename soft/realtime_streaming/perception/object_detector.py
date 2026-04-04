import glob
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    from openvino import Core
except ImportError:
    Core = None

# LidarSample import is optional — works without LiDAR hardware
try:
    from perception.sensors import LidarSample
except Exception:
    LidarSample = None


BBox = Tuple[int, int, int, int]
Detection = Tuple[str, float, BBox]


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


def find_openvino_model(search_dir: str) -> str:
    xml_files = glob.glob(os.path.join(search_dir, "**", "*.xml"), recursive=True)
    if len(xml_files) == 1:
        return xml_files[0]
    elif len(xml_files) > 1:
        for f in xml_files:
            if "best" in os.path.basename(f).lower():
                return f
        return xml_files[0]
    raise FileNotFoundError(f"No .xml model file found in {search_dir}")


class OpenVINODetector:
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "AUTO",
    ):
        if Core is None:
            raise RuntimeError("OpenVINO is not installed. Install with: pip install openvino")

        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        core = Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]

        input_shape = self.input_layer.shape
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        self.end2end = (
            len(self.output_layers) > 1
            or (len(self.output_layers) == 1
                and len(self.output_layers[0].shape) == 3
                and self.output_layers[0].shape[-1] == 6)
        )

        if not self.class_names:
            self.class_names = load_metadata_class_names(model_path)
            if self.class_names:
                print(f"[Detector] Auto-loaded {len(self.class_names)} classes from metadata.yaml")

        print(f"[Detector] Loaded: {model_path}")
        print(f"[Detector] Input: {self.input_w}x{self.input_h} | End2End: {self.end2end} | Device: {device}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def predict(self, frame: np.ndarray, lidar=None) -> List[Detection]:
        _ = lidar
        blob = self._preprocess(frame)
        results = self.compiled_model({self.input_layer: blob})
        frame_h, frame_w = frame.shape[:2]

        output = np.asarray(results[self.output_layers[0]])
        if self.end2end:
            return self._decode_end2end(output, frame_w, frame_h)
        else:
            return self._decode_standard(output, frame_w, frame_h)

    def _decode_end2end(self, output: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        if output.ndim == 3:
            output = output[0]
        scale_x = frame_w / self.input_w
        scale_y = frame_h / self.input_h
        detections: List[Detection] = []
        for row in output:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            if float(score) < self.conf_threshold:
                continue
            x1 = int(max(0, min(frame_w - 1, x1 * scale_x)))
            y1 = int(max(0, min(frame_h - 1, y1 * scale_y)))
            x2 = int(max(0, min(frame_w - 1, x2 * scale_x)))
            y2 = int(max(0, min(frame_h - 1, y2 * scale_y)))
            if x2 <= x1 or y2 <= y1:
                continue
            idx = int(class_id)
            label = self.class_names[idx] if self.class_names and 0 <= idx < len(self.class_names) else f"class_{idx}"
            detections.append((label, float(score), (x1, y1, x2, y2)))
        return detections

    def _decode_standard(self, output: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 5:
            return []

        boxes, scores, labels = [], [], []
        sx, sy = frame_w / self.input_w, frame_h / self.input_h

        for row in output:
            probs = row[4:]
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            if score < self.conf_threshold:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int(max(0, min(frame_w - 1, (cx - bw / 2) * sx)))
            y1 = int(max(0, min(frame_h - 1, (cy - bh / 2) * sy)))
            x2 = int(max(0, min(frame_w - 1, (cx + bw / 2) * sx)))
            y2 = int(max(0, min(frame_h - 1, (cy + bh / 2) * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self.class_names[idx] if self.class_names and 0 <= idx < len(self.class_names) else f"class_{idx}"
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)

        if not boxes:
            return []
        nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        nms_idx = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf_threshold, self.iou_threshold)
        dets: List[Detection] = []
        if len(nms_idx) > 0:
            for i in np.array(nms_idx).reshape(-1).tolist():
                dets.append((labels[i], scores[i], tuple(boxes[i])))
        return dets


# Backward-compatible alias
OnnxDetector = OpenVINODetector
