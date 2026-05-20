#!/usr/bin/env python3
import os
import struct
import sys
import can
import threading
import cv2
import time
import select
import csv
from datetime import datetime
from rplidar import RPLidar
import numpy as np
from queue import Queue

# Format for session naming
dttstr_int = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")

# --- CONFIGURATION ---
JOYSTICK_DEV = '/dev/input/js0'
CAN_INTERFACE = 'socketcan'
CAN_CHANNEL = 'can0'
CAN_BITRATE = 500000
SENDING_SPEED = 0.04
BASE_DATA_DIR = "/home/sdc/SDC.ExampleCode/data"
LIDAR_PORT = '/dev/ttyUSB0'

# Mapping from All.py
CAN_ID_MAP = {
    0x110: 'brake',
    0x220: 'steering',
    0x330: 'throttle',
    0x1e5: 'steering_sensor'
}

# Input Mapping
AXIS_LEFT_X = 0
AXIS_RIGHT_Y = 3
BTN_B_KILL = 1
BTN_X_RECORD = 3
BTN_Y_STOP_REC = 4
EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02
EVENT_INIT = 0x80

# --- CONTROL DATA WORKER (Saves CSV) ---
class ControlDataWorker:
    def __init__(self):
        self.recording = False
        self.running = True
        self.queue = Queue()
        self.session_dir = ""
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self, session_path):
        self.session_dir = os.path.join(session_path, "Control")
        os.makedirs(self.session_dir, exist_ok=True)
        self.recording = True

    def stop_rec(self):
        self.recording = False

    def put(self, data):
        if self.recording:
            self.queue.put(data)

    def cleanup(self):
        self.running = False
        self.thread.join()

    def _worker(self):
        while self.running:
            if not self.queue.empty() and self.recording:
                session_file = os.path.join(self.session_dir, "control_data.csv")
                file_exists = os.path.isfile(session_file)
                
                with open(session_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'brake', 'steering', 'throttle', 'steering_sensor'])
                    
                    while not self.queue.empty():
                        writer.writerow(self.queue.get())
            else:
                time.sleep(0.1)

# --- CAN LISTENER (From All.py) ---
class CanListener:
    def __init__(self, bus):
        self.bus = bus
        self.data = {name: 0 for name in CAN_ID_MAP.values()}
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            msg = self.bus.recv(0.1)
            if msg and msg.arbitration_id in CAN_ID_MAP:
                name = CAN_ID_MAP[msg.arbitration_id]
                # Simple extraction (adjust if your All.py used specific byte offsets)
                self.data[name] = msg.data[0] 

    def get_values(self):
        return [self.data['brake'], self.data['steering'], self.data['throttle'], self.data['steering_sensor']]

# --- LIDAR LOGIC ---
class LidarRecorder:
    def __init__(self):
        self.recording = False
        self.running = True
        self.lidar = None
        self.scan_count = 0
        self.session_dir = ""
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self, session_path):
        self.session_dir = os.path.join(session_path, "Lidar")
        os.makedirs(self.session_dir, exist_ok=True)
        self.scan_count = 0
        self.recording = True

    def stop_rec(self):
        self.recording = False

    def cleanup(self):
        self.running = False
        self.recording = False
        self.thread.join()

    def _worker(self):
        while self.running:
            if self.recording:
                try:
                    self.lidar = RPLidar(LIDAR_PORT, timeout=3)
                    self.lidar.start_motor()
                    time.sleep(2) 
                    for scan in self.lidar.iter_scans(max_buf_meas=1000):
                        if not self.recording or not self.running: break
                        filename = os.path.join(self.session_dir, f"scan_{self.scan_count:05d}.csv")
                        np.savetxt(filename, np.array(scan), delimiter=",", fmt='%.2f')
                        self.scan_count += 1
                except: self.recording = False
                finally:
                    if self.lidar:
                        self.lidar.stop(); self.lidar.stop_motor(); self.lidar.disconnect(); self.lidar = None
            else: time.sleep(0.1)

# --- CAMERA LOGIC ---
class CameraRecorder:
    def __init__(self):
        self.recording = False
        self.running = True
        self.img_count = 0
        self.last_capture_time = 0
        self.session_path = ""
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self):
        if not self.recording:
            session_id = dttstr_int()
            self.session_path = os.path.join(BASE_DATA_DIR, f"session_{session_id}")
            os.makedirs(os.path.join(self.session_path, "Video"), exist_ok=True)
            os.makedirs(os.path.join(self.session_path, "Image"), exist_ok=True)
            print(f"\n>>> STARTING SESSION: {session_id}")
            self.recording = True
            self.img_count = 0
            return self.session_path

    def stop_rec(self):
        if self.recording:
            print(f"\n>>> STOPPED SESSION")
            self.recording = False

    def cleanup(self):
        self.running = False
        self.thread.join()

    def _worker(self):
        def try_open(idx):
            c = cv2.VideoCapture(idx)
            if not c.isOpened(): c.release(); return None
            return c

        cam_right, cam_mid, cam_left = try_open(0), try_open(2), try_open(4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_l = out_m = out_r = None

        while self.running:
            ret_r, frame_r = cam_right.read() if cam_right else (False, None)
            ret_m, frame_m = cam_mid.read() if cam_mid else (False, None)
            ret_l, frame_l = cam_left.read() if cam_left else (False, None)

            if self.recording:
                if out_r is None:
                    active = cam_right or cam_mid or cam_left
                    w = int(active.get(cv2.CAP_PROP_FRAME_WIDTH)) if active else 848
                    h = int(active.get(cv2.CAP_PROP_FRAME_HEIGHT)) if active else 480
                    v_dir = os.path.join(self.session_path, "Video")
                    if cam_right: out_r = cv2.VideoWriter(os.path.join(v_dir, 'Right.mp4'), fourcc, 30.0, (w, h))
                    if cam_mid:   out_m = cv2.VideoWriter(os.path.join(v_dir, 'Middle.mp4'), fourcc, 30.0, (w, h))
                    if cam_left:  out_l = cv2.VideoWriter(os.path.join(v_dir, 'Left.mp4'), fourcc, 30.0, (w, h))
                
                if ret_r and out_r: out_r.write(frame_r)
                if ret_m and out_m: out_m.write(frame_m)
                if ret_l and out_l: out_l.write(frame_l)

                if time.time() - self.last_capture_time >= 0.04:
                    if ret_r and frame_r is not None:
                        cv2.imwrite(os.path.join(self.session_path, "Image", f"img_{self.img_count:05d}.png"), frame_r, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        self.img_count += 1
                        self.last_capture_time = time.time()
            else:
                if out_r: out_r.release(); out_r = None
                if out_m: out_m.release(); out_m = None
                if out_l: out_l.release(); out_l = None

        for c in [cam_left, cam_mid, cam_right]:
            if c: c.release()

# --- MAIN ---
def main():
    recorder = CameraRecorder()
    lidar_rec = LidarRecorder()
    control_worker = ControlDataWorker()

    try:
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
        can_listener = CanListener(bus)
    except OSError:
        print("CAN Error: Interface not found."); sys.exit(1)

    steer_msg = can.Message(arbitration_id=0x220, data=[0]*8, is_extended_id=False)
    motor_msg = can.Message(arbitration_id=0x330, data=[0, 0, 1, 0, 0, 0, 0, 0], is_extended_id=False)
    steer_task = bus.send_periodic(steer_msg, SENDING_SPEED)
    motor_task = bus.send_periodic(motor_msg, SENDING_SPEED)

    print("\nALL-IN-ONE SYSTEM READY.")
    print("X: Start Record | Y: Stop | B: Kill\n")

    try:
        with open(JOYSTICK_DEV, 'rb') as js_file:
            while True:
                # Log Control data if recording
                if recorder.recording:
                    control_worker.put([time.time()] + can_listener.get_values())

                ready, _, _ = select.select([js_file], [], [], 0.01)
                if ready:
                    event_data = js_file.read(8)
                    if not event_data: break
                    _, value, type_, number = struct.unpack('IhBB', event_data)
                    
                    if type_ & EVENT_BUTTON and value == 1:
                        if number == BTN_X_RECORD:
                            path = recorder.start_rec()
                            lidar_rec.start_rec(path)
                            control_worker.start_rec(path)
                        elif number == BTN_Y_STOP_REC:
                            recorder.stop_rec(); lidar_rec.stop_rec(); control_worker.stop_rec()
                        elif number == BTN_B_KILL:
                            recorder.cleanup(); lidar_rec.cleanup(); control_worker.cleanup()
                            steer_task.stop(); motor_task.stop(); bus.shutdown(); sys.exit(0)

                    elif type_ & EVENT_AXIS:
                        if number == AXIS_LEFT_X:
                            steer_msg.data = list(struct.pack("<f", value / 32767.0)) + [0]*4
                            steer_task.modify_data(steer_msg)
                        elif number == AXIS_RIGHT_Y:
                            speed = int((abs(value) / 32767.0) * 45) if abs(value) > 5000 else 0## Fix the maximum speed 50% now
                            gear = 1 if value < -5000 else (2 if value > 5000 else 0)
                            motor_msg.data = [speed, 0, gear, 0, 0, 0, 0, 0]
                            motor_task.modify_data(motor_msg)
    except KeyboardInterrupt:
        recorder.cleanup(); lidar_rec.cleanup(); control_worker.cleanup(); bus.shutdown()

if __name__ == "__main__":
    main()