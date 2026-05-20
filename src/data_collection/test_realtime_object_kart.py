#!/usr/bin/env python3
"""
Kart variant of test_realtime_object.py.

Combines:
  - Joystick + CAN control loop from src/data_collection/New_control.py
    (Xbox controller → steering / throttle / brake on can0)
  - Threaded camera + OpenVINO object detection from test_realtime_object.py
  - LiDAR minimap with vehicle-frame conversion:
        θ_vehicle = θ_lidar - 180°   (forward = +X, top of minimap = kart front)

Camera ports (from New_control.py):
    RIGHT = 0    MIDDLE = 2    LEFT = 4
"""

import os
import sys
import time
import struct
import select
import threading

import cv2
import can
import numpy as np
import openvino.runtime as ov
from rplidar import RPLidar


# ─────────────────────────────────────────────
# CONFIG  —  ports & control (from New_control.py)
# ─────────────────────────────────────────────

# Joystick
JOYSTICK_DEV   = '/dev/input/js0'

# CAN bus
CAN_INTERFACE  = 'socketcan'
CAN_CHANNEL    = 'can0'
CAN_BITRATE    = 500000
SENDING_SPEED  = 0.04

CONTROL_CAN_IDS = {
    'brake':    0x110,
    'steering': 0x220,
    'throttle': 0x330,
}

# Xbox controller mapping
AXIS_LEFT_X       = 0
AXIS_LEFT_TRIGGER = 2
AXIS_RIGHT_Y      = 3
BTN_B_KILL        = 1
BTN_X_RECORD      = 3   # unused here, kept for parity
BTN_Y_STOP_REC    = 4   # unused here, kept for parity
EVENT_BUTTON      = 0x01
EVENT_AXIS        = 0x02
EVENT_INIT        = 0x80

# LiDAR
LIDAR_PORT = '/dev/ttyUSB0'

# Cameras (RIGHT=0, MIDDLE=2, LEFT=4 per New_control.py:300)
CAMERA_INDEX = 2

# Perception
MAP_SIZE       = 400
MAX_DIST       = 4000           # mm
CONF_THRESHOLD = 0.45
NMS_THRESHOLD  = 0.40
HFOV_DEG       = 70.0

# Kart hardware: cable side = 180° → forward in vehicle frame is at raw 180°.
# Verified from 23-04-2026 bench test (car at 0.75m in front → returns near 190°).
LIDAR_ANGLE_OFFSET = 180

MODEL_XML    = "/home/jannat/sdc_2026/Model/best_openvino_model-20260401T075819Z-3-001/best_openvino_model/best.xml"
CLASSES_PATH = "/home/jannat/sdc_2026/Model/classes.txt"

PALETTE = [
    (0, 255, 255), (0, 165, 255), (0, 255, 0),
    (255, 0, 0),   (255, 0, 255), (0, 128, 255),
    (128, 255, 0), (255, 128, 0), (0, 0, 255),
    (128, 0, 255),
]


# ─────────────────────────────────────────────
# CAMERA  (threaded)
# ─────────────────────────────────────────────

class CameraStream:
    def __init__(self, index=CAMERA_INDEX):
        # V4L2 + MJPG so the USB bus can sustain the camera's native
        # resolution (1920x1080 @ 30fps on Logitech StreamCam).
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera opened but first frame failed")
        self.frame   = frame
        self.frame_w = frame.shape[1]
        self.frame_h = frame.shape[0]
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"[Camera] index {index} — native {self.frame_w}x{self.frame_h}")

    def _reader(self):
        while self.running:
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
# OBJECT DETECTOR  (OpenVINO YOLOv8)
# ─────────────────────────────────────────────

class ObjectDetector:
    def __init__(self, model_path, classes_path):
        with open(classes_path, 'r') as f:
            self.classes = [l.strip() for l in f.readlines() if l.strip()]
        print(f"[Detector] {len(self.classes)} classes")

        core  = ov.Core()
        model = core.read_model(model=model_path)
        self.compiled    = core.compile_model(model=model, device_name="AUTO")
        self.input_layer = self.compiled.input(0)
        self.out_layer   = self.compiled.output(0)
        _, _, self.model_h, self.model_w = self.input_layer.shape

        out_shape = tuple(self.out_layer.shape)
        self.end2end = (len(out_shape) == 3 and out_shape[-1] == 6)
        print(f"[Detector] In {self.model_w}x{self.model_h} | Out {out_shape} | End2End={self.end2end}")

    def _preprocess(self, frame):
        resized = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        blob = self._preprocess(frame)
        raw  = np.asarray(self.compiled({self.input_layer: blob})[self.out_layer])
        if self.end2end:
            return self._decode_end2end(raw, img_w, img_h)
        return self._decode_standard(raw, img_w, img_h)

    def _decode_end2end(self, raw, img_w, img_h):
        data   = raw[0] if raw.ndim == 3 else raw
        sx, sy = img_w / self.model_w, img_h / self.model_h
        results = []
        for row in data:
            x1, y1, x2, y2, score, cid = row[:6]
            if float(score) < CONF_THRESHOLD: continue
            x1 = int(max(0, min(img_w - 1, x1 * sx)))
            y1 = int(max(0, min(img_h - 1, y1 * sy)))
            x2 = int(max(0, min(img_w - 1, x2 * sx)))
            y2 = int(max(0, min(img_h - 1, y2 * sy)))
            if x2 <= x1 or y2 <= y1: continue
            cid = int(cid)
            label = self.classes[cid] if 0 <= cid < len(self.classes) else f"cls{cid}"
            results.append({'class_id': cid, 'label': label, 'conf': float(score),
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        return results

    def _decode_standard(self, raw, img_w, img_h):
        data = raw[0]
        if data.shape[0] < data.shape[1]:
            data = data.T
        sx, sy = img_w / self.model_w, img_h / self.model_h
        boxes, scores, labels, class_ids = [], [], [], []
        for row in data:
            probs = row[4:]
            cid   = int(np.argmax(probs))
            score = float(probs[cid])
            if score < CONF_THRESHOLD: continue
            cx, cy, bw, bh = row[:4]
            x1 = int(max(0, min(img_w - 1, (cx - bw / 2) * sx)))
            y1 = int(max(0, min(img_h - 1, (cy - bh / 2) * sy)))
            x2 = int(max(0, min(img_w - 1, (cx + bw / 2) * sx)))
            y2 = int(max(0, min(img_h - 1, (cy + bh / 2) * sy)))
            if x2 <= x1 or y2 <= y1: continue
            label = self.classes[cid] if 0 <= cid < len(self.classes) else f"cls{cid}"
            boxes.append([x1, y1, x2, y2]); scores.append(score)
            labels.append(label); class_ids.append(cid)
        if not boxes: return []
        nms_in = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
        idxs = cv2.dnn.NMSBoxes(nms_in, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) == 0: return []
        out = []
        for i in np.array(idxs).flatten():
            b = boxes[i]
            out.append({'class_id': class_ids[i], 'label': labels[i], 'conf': scores[i],
                        'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]})
        return out

    def draw(self, frame, detections):
        for d in detections:
            color = PALETTE[d['class_id'] % len(PALETTE)]
            cv2.rectangle(frame, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
            txt = f"{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (d['x1'], d['y1'] - th - 6),
                          (d['x1'] + tw + 4, d['y1']), color, -1)
            cv2.putText(frame, txt, (d['x1'] + 2, d['y1'] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────
# LIDAR  (with vehicle-frame conversion)
# ─────────────────────────────────────────────

class LidarStream:
    """
    Raw rplidar angle convention (verified 23-04-2026 bench test):
        cable side / kart rear  ≈ 0° / 360°
        kart front              ≈ 180°
    Vehicle frame is obtained by adding LIDAR_ANGLE_OFFSET (=180), so the
    front of the kart ends up at 0° / top of the minimap, with forward = +X.
    """
    def __init__(self):
        try:
            self.lidar = RPLidar(LIDAR_PORT)
        except Exception as e:
            raise RuntimeError(f"LiDAR connection failed: {e}")
        self.current_scan = []
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        print("[LiDAR] thread started")

    def _worker(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self.running: break
                with self.lock:
                    self.current_scan = scan
        except Exception as e:
            print(f"[LiDAR] error: {e}")
        finally:
            try: self.lidar.stop(); self.lidar.stop_motor(); self.lidar.disconnect()
            except Exception: pass

    def get_scan(self):
        with self.lock:
            return list(self.current_scan)

    def get_minimap(self, detections=None, frame_w=640, hfov_deg=HFOV_DEG):
        mini = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        cx = cy = MAP_SIZE // 2

        for r_mm, lbl in [(1000, "1m"), (2000, "2m"), (3000, "3m")]:
            r_px = int((r_mm / MAX_DIST) * cx)
            cv2.circle(mini, (cx, cy), r_px, (55, 55, 55), 1)
            cv2.putText(mini, lbl, (cx + r_px + 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        cv2.putText(mini, "FRONT", (cx - 18, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

        det_windows = []
        if detections:
            for d in detections:
                box_cx       = (d['x1'] + d['x2']) / 2
                angle_offset = ((box_cx / frame_w) - 0.5) * hfov_deg
                half_w       = ((d['x2'] - d['x1']) / frame_w) * hfov_deg / 2
                det_windows.append({'label': d['label'], 'class_id': d['class_id'],
                                    'amin': angle_offset - half_w,
                                    'amax': angle_offset + half_w,
                                    'min_dist': None})

        for _, angle, dist in self.get_scan():
            if not (0 < dist < MAX_DIST): continue

            # Vehicle frame: θ_v = θ_lidar - 180  (equivalently + 180 mod 360)
            adj = (angle + LIDAR_ANGLE_OFFSET) % 360

            color = (0, 0, 255) if dist < 1000 else (0, 165, 255) if dist < 2000 else (0, 255, 0)

            rad = np.deg2rad(adj - 90)
            dpx = (dist / MAX_DIST) * cx
            px  = int(cx + dpx * np.cos(rad))
            py  = int(cy + dpx * np.sin(rad))
            if not (0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE): continue

            rel = adj if adj <= 180 else adj - 360
            for dw in det_windows:
                if dw['amin'] <= rel <= dw['amax']:
                    color = PALETTE[dw['class_id'] % len(PALETTE)]
                    if dw['min_dist'] is None or dist < dw['min_dist']:
                        dw['min_dist'] = dist
                    break

            cv2.circle(mini, (px, py), 2, color, -1)

        y = 15
        for dw in det_windows:
            if dw['min_dist'] is not None:
                txt = f"{dw['label']}: {dw['min_dist']/1000:.2f}m"
                cv2.putText(mini, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            PALETTE[dw['class_id'] % len(PALETTE)], 1, cv2.LINE_AA)
                y += 14

        cv2.line(mini, (cx-6, cy), (cx+6, cy), (255,255,255), 1)
        cv2.line(mini, (cx, cy-6), (cx, cy+6), (255,255,255), 1)

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
# CAN CONTROL  (joystick → kart, from New_control.py)
# ─────────────────────────────────────────────

class KartControl:
    def __init__(self):
        self.bus = can.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
        self.steer_msg    = can.Message(arbitration_id=CONTROL_CAN_IDS['steering'],
                                        data=[0]*8, is_extended_id=False)
        self.throttle_msg = can.Message(arbitration_id=CONTROL_CAN_IDS['throttle'],
                                        data=[0, 0, 1, 0, 0, 0, 0, 0], is_extended_id=False)
        self.brake_msg    = can.Message(arbitration_id=CONTROL_CAN_IDS['brake'],
                                        data=[0]*8, is_extended_id=False)
        self.steer_task    = self.bus.send_periodic(self.steer_msg, SENDING_SPEED)
        self.throttle_task = self.bus.send_periodic(self.throttle_msg, SENDING_SPEED)
        self.brake_task    = self.bus.send_periodic(self.brake_msg, SENDING_SPEED)

        self.cur_steer    = 0.0
        self.cur_throttle = 0.0
        self.cur_brake    = 0.0
        self.cur_gear     = 'neutral'
        print("[CAN] periodic tasks running")

    def set_steering(self, value):
        self.cur_steer = round(value / 32767.0, 4)
        self.steer_msg.data = list(struct.pack("<f", self.cur_steer)) + [0]*4
        self.steer_task.modify_data(self.steer_msg)

    def set_throttle(self, value):
        self.cur_throttle = round(-(value / 32767.0), 4)
        speed = int(abs(self.cur_throttle) * 45) if abs(value) > 5000 else 0
        gear  = 1 if self.cur_throttle > 0.15 else (2 if self.cur_throttle < -0.15 else 0)
        self.cur_gear = {0: 'neutral', 1: 'forward', 2: 'reverse'}[gear]
        self.throttle_msg.data = [speed, 0, gear, 0, 0, 0, 0, 0]
        self.throttle_task.modify_data(self.throttle_msg)

    def set_brake(self, value):
        self.cur_brake = round(max(0, (value + 32767) / 65534.0), 4)
        brake_val = int(self.cur_brake * 100)
        self.brake_msg.data = [brake_val, 0, 0, 0, 0, 0, 0, 0]
        self.brake_task.modify_data(self.brake_msg)

    def shutdown(self):
        try:
            self.steer_task.stop(); self.throttle_task.stop(); self.brake_task.stop()
            self.bus.shutdown()
        except Exception: pass


def joystick_thread(ctrl: 'KartControl', stop_event: threading.Event):
    try:
        js = open(JOYSTICK_DEV, 'rb')
    except OSError as e:
        print(f"[Joystick] cannot open {JOYSTICK_DEV}: {e}")
        return
    print("[Joystick] ready  —  B = kill")
    try:
        while not stop_event.is_set():
            ready, _, _ = select.select([js], [], [], 0.05)
            if not ready: continue
            event_data = js.read(8)
            if not event_data: break
            _, value, type_, number = struct.unpack('IhBB', event_data)
            if type_ & EVENT_INIT: continue

            if type_ & EVENT_BUTTON and value == 1:
                if number == BTN_B_KILL:
                    stop_event.set()
                    return

            elif type_ & EVENT_AXIS:
                if   number == AXIS_LEFT_X:       ctrl.set_steering(value)
                elif number == AXIS_RIGHT_Y:      ctrl.set_throttle(value)
                elif number == AXIS_LEFT_TRIGGER: ctrl.set_brake(value)
    finally:
        js.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Initialising kart systems...")

    try:
        ctrl = KartControl()
    except OSError as e:
        print(f"CAN init failed ({e}). Is can0 up?  sudo ip link set can0 up type can bitrate 500000")
        sys.exit(1)

    camera   = CameraStream(index=CAMERA_INDEX)
    lidar    = LidarStream()
    detector = ObjectDetector(MODEL_XML, CLASSES_PATH)

    stop_event = threading.Event()
    threading.Thread(target=joystick_thread, args=(ctrl, stop_event), daemon=True).start()

    print("All systems ready.  q in window or B on pad to quit.")
    fps_t, fps_count, fps = time.time(), 0, 0.0

    try:
        while not stop_event.is_set():
            frame = camera.read()

            detections = detector.detect(frame)
            detector.draw(frame, detections)

            fps_count += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_t, fps_count = time.time(), 0

            hud = (f"FPS:{fps:4.1f}  "
                   f"str:{ctrl.cur_steer:+.2f}  "
                   f"thr:{ctrl.cur_throttle:+.2f}  "
                   f"brk:{ctrl.cur_brake:.2f}  "
                   f"gear:{ctrl.cur_gear}")
            cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            mini = lidar.get_minimap(detections=detections, frame_w=camera.frame_w, hfov_deg=HFOV_DEG)

            cv2.imshow("Kart — Object Detection", frame)
            cv2.imshow("LiDAR Minimap (vehicle frame, front=top)", mini)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

    finally:
        stop_event.set()
        camera.stop()
        lidar.stop()
        ctrl.shutdown()
        cv2.destroyAllWindows()
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()
