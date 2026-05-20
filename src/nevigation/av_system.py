import cv2
import numpy as np
from rplidar import RPLidar
import openvino.runtime as ov
import threading
import time
import os

# Suppress Qt font warnings (harmless — Qt fonts not bundled in this venv)
os.environ["QT_QPA_FONTDIR"] = ""
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# --- Configuration ---
CAMERA_INDEX = 0          # confirmed working index
PORT_NAME    = '/dev/ttyUSB0'
MAP_SIZE     = 400
MAX_DIST     = 4000       # mm — LiDAR max range shown on minimap
CONF_THRESHOLD = 0.25     # lowered — end2end models output lower raw scores than standard YOLOv8
NMS_THRESHOLD  = 0.45     # overlap IOU threshold for NMS

# Paths
MODEL_XML    = "/home/jannat/sdc_2026/Model/best_openvino_model-20260401T075819Z-3-001/best_openvino_model/best.xml"
CLASSES_PATH = "/home/jannat/sdc_2026/Model/classes.txt"

# Colour palette per class index (BGR)
PALETTE = [
    (0, 255, 255),   (0, 165, 255),  (0, 255, 0),
    (255, 0, 0),     (255, 0, 255),  (0, 128, 255),
    (128, 255, 0),   (255, 128, 0),  (0, 0, 255),
    (128, 0, 255),
]


# ─────────────────────────────────────────────
# MODULE 1 — THREADED CAMERA
# ─────────────────────────────────────────────

class CameraStream:
    """
    Reads camera frames in a background thread so the main loop
    never blocks waiting for the next frame.

    Root cause of blank screen:
      OpenVINO inference took longer than one frame interval.
      cv2.VideoCapture.read() was blocking, so imshow() was
      never reached in time. Threading fixes this completely.
    """
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera opened but first frame failed")
        self.frame   = frame
        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"[Camera] Started on index {index} — {self.frame_w}x{self.frame_h}")

    def _reader(self):
        while self.running:
            self.cap.grab()          # discard stale buffer — fixes WSL black frames
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


# ─────────────────────────────────────────────
# MODULE 2 — OBJECT DETECTOR  (YOLOv8 OpenVINO)
# ─────────────────────────────────────────────

class ObjectDetector:
    """
    Fixes vs previous version:
      1. BGR→RGB conversion before inference (model trained on RGB)
         — this was causing the tiny corner box (wrong colours = garbage coords)
      2. device="AUTO" — OpenVINO picks best available (CPU/GPU/NPU)
      3. Coordinate clamping — boxes stay within frame bounds
      4. End-to-end output support — handles both [1,84,8400] and [N,6] outputs
      5. Debug: prints raw output shape on first frame so you can verify
    """

    def __init__(self, model_path, classes_path):
        with open(classes_path, 'r') as f:
            self.classes = [l.strip() for l in f.readlines() if l.strip()]
        print(f"[Detector] {len(self.classes)} classes: {self.classes}")

        core  = ov.Core()
        model = core.read_model(model=model_path)

        # AUTO lets OpenVINO choose GPU/NPU/CPU — faster than forcing CPU
        self.compiled    = core.compile_model(model=model, device_name="AUTO")
        self.input_layer = self.compiled.input(0)
        self.out_layer   = self.compiled.output(0)

        _, _, self.model_h, self.model_w = self.input_layer.shape

        # Detect if model is end-to-end (output shape [N,6]: x1,y1,x2,y2,score,class_id)
        # vs standard YOLOv8 (output shape [1,84,8400])
        out_shape = tuple(self.out_layer.shape)
        self.end2end = (len(out_shape) == 3 and out_shape[-1] == 6)
        print(f"[Detector] Input: {self.model_w}x{self.model_h} | "
              f"Output shape: {out_shape} | End2End: {self.end2end}")

        self._first_frame = True   # for one-time debug print

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize → BGR to RGB → normalise to [0,1] → CHW → add batch dim.
        RGB conversion is critical — model trained on RGB, OpenCV reads BGR.
        Skipping this was the root cause of the tiny corner box.
        """
        resized = cv2.resize(frame, (self.model_w, self.model_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)           # ← KEY FIX
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))                      # HWC → CHW
        return np.expand_dims(blob, axis=0)                       # → [1,3,H,W]

    def detect(self, frame: np.ndarray):
        """
        Returns list of dicts: { class_id, label, conf, x1, y1, x2, y2 }
        Handles both standard YOLOv8 [1,84,8400] and end-to-end [N,6] outputs.
        """
        img_h, img_w = frame.shape[:2]
        blob   = self._preprocess(frame)
        raw    = np.asarray(self.compiled({self.input_layer: blob})[self.out_layer])

        # One-time debug — helps confirm output format
        if self._first_frame:
            print(f"[Detector] Raw output shape: {raw.shape}  "
                  f"min={raw.min():.3f}  max={raw.max():.3f}")
            self._first_frame = False

        if self.end2end:
            return self._decode_end2end(raw, img_w, img_h)
        else:
            return self._decode_standard(raw, img_w, img_h)

    def _decode_end2end(self, raw: np.ndarray, img_w: int, img_h: int):
        """
        End-to-end export: [1, 300, 6] → x1, y1, x2, y2, score, class_id
        Coords are in model-input pixel space (0 to model_w/h).
        NMS is already baked into the model — do NOT run NMSBoxes again.
        Scale to original frame size with sx, sy.
        """
        data   = raw[0] if raw.ndim == 3 else raw   # [300, 6]
        sx, sy = img_w / self.model_w, img_h / self.model_h
        results = []
        for row in data:
            x1, y1, x2, y2, score, class_id = row[:6]
            conf = float(score)
            if conf < CONF_THRESHOLD:
                continue
            rx1 = int(max(0, min(img_w - 1, x1 * sx)))
            ry1 = int(max(0, min(img_h - 1, y1 * sy)))
            rx2 = int(max(0, min(img_w - 1, x2 * sx)))
            ry2 = int(max(0, min(img_h - 1, y2 * sy)))
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            cid   = int(class_id)
            label = self.classes[cid] if 0 <= cid < len(self.classes) else f"cls{cid}"
            results.append({'class_id': cid, 'label': label,
                            'conf': conf,
                            'x1': rx1, 'y1': ry1, 'x2': rx2, 'y2': ry2})
        return results

    def _decode_standard(self, raw: np.ndarray, img_w: int, img_h: int):
        """
        Output format: [1, 84, 8400]
          cols 0-3  = cx, cy, w, h  (in model-input pixel space)
          cols 4-83 = class scores
        """
        data = raw[0]                        # [84, 8400]
        if data.shape[0] < data.shape[1]:
            data = data.T                    # → [8400, 84]

        sx, sy = img_w / self.model_w, img_h / self.model_h

        boxes, scores, labels, class_ids = [], [], [], []
        for row in data:
            probs  = row[4:]
            cid    = int(np.argmax(probs))
            score  = float(probs[cid])
            if score < CONF_THRESHOLD:
                continue
            cx, cy, bw, bh = row[:4]
            # Clamp to frame bounds
            x1 = int(max(0, min(img_w - 1, (cx - bw / 2) * sx)))
            y1 = int(max(0, min(img_h - 1, (cy - bh / 2) * sy)))
            x2 = int(max(0, min(img_w - 1, (cx + bw / 2) * sx)))
            y2 = int(max(0, min(img_h - 1, (cy + bh / 2) * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self.classes[cid] if 0 <= cid < len(self.classes) else f"cls{cid}"
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)
            class_ids.append(cid)

        if not boxes:
            return []

        # NMS — remove overlapping duplicate boxes
        nms_input = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
        idxs = cv2.dnn.NMSBoxes(nms_input, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) == 0:
            return []

        results = []
        for i in np.array(idxs).flatten():
            b = boxes[i]
            results.append({'class_id': class_ids[i], 'label': labels[i],
                            'conf': scores[i],
                            'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]})
        return results

    def draw(self, frame: np.ndarray, detections: list):
        for d in detections:
            color = PALETTE[d['class_id'] % len(PALETTE)]
            cv2.rectangle(frame, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
            txt = f"{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                          (d['x1'], d['y1'] - th - 6),
                          (d['x1'] + tw + 4, d['y1']), color, -1)
            cv2.putText(frame, txt, (d['x1'] + 2, d['y1'] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────
# MODULE 3 — LIDAR
# ─────────────────────────────────────────────

class LidarStream:
    """
    Minimap dot colours:
      RED   = < 1 m   — danger
      AMBER = 1–2 m   — caution
      GREEN = > 2 m   — clear

    LIDAR_ANGLE_OFFSET:
      RPLidar cable side = 180° by hardware convention.
      Adjust this until the minimap front matches your kart front.

        0   = cable at back  (laser front = kart front) ← try this first
       180  = cable at front
        90  = cable on left side
       -90  = cable on right side
    """

    LIDAR_ANGLE_OFFSET = 0   # ← tune after running direction test

    def __init__(self):
        try:
            self.lidar = RPLidar(PORT_NAME)
        except Exception as e:
            raise RuntimeError(f"LiDAR connection failed: {e}")
        self.current_scan = []
        self.running = True
        self.lock    = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        print("[LiDAR] Thread started.")

    def _worker(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                with self.lock:
                    self.current_scan = scan
        except Exception as e:
            print(f"[LiDAR] Error: {e}")
        finally:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()

    def get_scan(self):
        with self.lock:
            return list(self.current_scan)

    def get_minimap(self, detections=None, frame_w=640, hfov_deg=70.0):
        mini   = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        cx = cy = MAP_SIZE // 2

        # Distance rings
        for r_mm, lbl in [(1000, "1m"), (2000, "2m"), (3000, "3m")]:
            r_px = int((r_mm / MAX_DIST) * cx)
            cv2.circle(mini, (cx, cy), r_px, (55, 55, 55), 1)
            cv2.putText(mini, lbl, (cx + r_px + 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

        cv2.putText(mini, "FRONT", (cx - 18, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

        # Pre-compute angle windows per detection for LiDAR matching
        det_windows = []
        if detections:
            for d in detections:
                box_cx       = (d['x1'] + d['x2']) / 2
                angle_offset = ((box_cx / frame_w) - 0.5) * hfov_deg
                half_w       = ((d['x2'] - d['x1']) / frame_w) * hfov_deg / 2
                det_windows.append({
                    'label':    d['label'],
                    'class_id': d['class_id'],
                    'amin':     angle_offset - half_w,
                    'amax':     angle_offset + half_w,
                    'min_dist': None,
                })

        for _, angle, dist in self.get_scan():
            if not (0 < dist < MAX_DIST):
                continue

            adj = (angle + self.LIDAR_ANGLE_OFFSET) % 360

            # Colour by distance
            if dist < 1000:
                color = (0, 0, 255)
            elif dist < 2000:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            rad = np.deg2rad(adj - 90)
            dpx = (dist / MAX_DIST) * cx
            px  = int(cx + dpx * np.cos(rad))
            py  = int(cy + dpx * np.sin(rad))
            if not (0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE):
                continue

            # Match LiDAR angle to camera detection angle window
            rel = adj if adj <= 180 else adj - 360
            for dw in det_windows:
                if dw['amin'] <= rel <= dw['amax']:
                    color = PALETTE[dw['class_id'] % len(PALETTE)]
                    if dw['min_dist'] is None or dist < dw['min_dist']:
                        dw['min_dist'] = dist
                    break

            cv2.circle(mini, (px, py), 2, color, -1)

        # Per-object distance legend (top left)
        y = 15
        for dw in det_windows:
            if dw['min_dist'] is not None:
                txt = f"{dw['label']}: {dw['min_dist']/1000:.2f}m"
                cv2.putText(mini, txt, (5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            PALETTE[dw['class_id'] % len(PALETTE)], 1, cv2.LINE_AA)
                y += 14

        # Kart cross
        cv2.line(mini, (cx-6, cy), (cx+6, cy), (255,255,255), 1)
        cv2.line(mini, (cx, cy-6), (cx, cy+6), (255,255,255), 1)

        # Colour key (bottom)
        for i, (txt, col) in enumerate([
            ("< 1m DANGER",  (0, 0, 255)),
            ("1-2m CAUTION", (0, 165, 255)),
            ("> 2m CLEAR",   (0, 255, 0)),
        ]):
            cv2.putText(mini, txt, (5, MAP_SIZE - 42 + i*14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
        return mini

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print("Initialising systems...")

    camera   = CameraStream(index=CAMERA_INDEX)
    lidar    = LidarStream()
    detector = ObjectDetector(MODEL_XML, CLASSES_PATH)  # loads last — slowest

    print("All systems ready. Press 'q' to quit.")

    fps_t, fps_count, fps = time.time(), 0, 0.0

    try:
        while True:
            # 1. Grab latest frame — never blocks (background thread)
            frame = camera.read()

            # 2. Detect + draw
            detections = detector.detect(frame)
            detector.draw(frame, detections)

            # 3. FPS counter
            fps_count += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_t, fps_count = time.time(), 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. LiDAR minimap — coloured dots + per-object distance labels
            mini = lidar.get_minimap(
                detections=detections,
                frame_w=camera.frame_w,
                hfov_deg=78.0    # Logitech StreamCam measured HFOV's real horizontal FOV
            )

            # 5. Show both windows
            cv2.imshow("Camera — Object Detection", frame)
            cv2.imshow("LiDAR Minimap", mini)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        lidar.stop()
        cv2.destroyAllWindows()
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()
