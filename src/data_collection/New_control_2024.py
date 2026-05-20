#!/usr/bin/env python3
import os, struct, sys, can, threading, cv2, time, select, csv
from datetime import datetime
from rplidar import RPLidar
import numpy as np
from queue import Queue

# --- CONFIGURATION ---
JOYSTICK_DEV = '/dev/input/js0'
CAN_INTERFACE = 'socketcan'
CAN_CHANNEL = 'can0'
BASE_DATA_DIR = "/home/sdc/SDC.ExampleCode/data"
LIDAR_PORT = '/dev/ttyUSB0'

# Axis Mappings (Confirmed Correct)
AXIS_STEER = 0     # Left Stick X
AXIS_THROTTLE = 3  # Right Stick Y
BTN_B_KILL = 1
BTN_X_RECORD = 3
BTN_Y_STOP_REC = 4

EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02

# --- WORKERS ---

class VideoWorker:
    def __init__(self):
        self.recording = False
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.session_dir = ""
        threading.Thread(target=self._worker, daemon=True).start()

    def start_rec(self, session_path):
        self.session_dir = os.path.join(session_path, "Camera")
        os.makedirs(self.session_dir, exist_ok=True)
        self.recording = True

    def stop_rec(self): self.recording = False

    def _worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and self.recording and self.session_dir:
                ts = time.time()
                cv2.imwrite(os.path.join(self.session_dir, f"{ts}.jpg"), frame)
            time.sleep(0.05)

class LidarWorker:
    def __init__(self, port=LIDAR_PORT):
        self.recording = False
        self.running = True
        try:
            self.lidar = RPLidar(port)
        except:
            print("Lidar not found on", port); self.lidar = None
        self.session_dir = ""
        threading.Thread(target=self._worker, daemon=True).start()

    def start_rec(self, session_path):
        self.session_dir = os.path.join(session_path, "Lidar")
        os.makedirs(self.session_dir, exist_ok=True)
        self.recording = True

    def stop_rec(self): self.recording = False

    def _worker(self):
        while self.running and self.lidar:
            if self.recording and self.session_dir:
                try:
                    points = []
                    for scan in self.lidar.iter_scans():
                        if not self.recording: break
                        points.extend(scan)
                        if len(points) > 500: break
                    np.save(os.path.join(self.session_dir, f"{time.time()}.npy"), np.array(points))
                except: continue
            time.sleep(0.1)

class ControlDataWorker:
    def __init__(self):
        self.recording = False
        self.queue = Queue()
        self.session_dir = ""
        threading.Thread(target=self._worker, daemon=True).start()

    def start_rec(self, session_path):
        self.session_dir = os.path.join(session_path, "Control")
        os.makedirs(self.session_dir, exist_ok=True)
        self.recording = True

    def stop_rec(self): self.recording = False

    def put(self, t, steer, throttle, brake=0, speed=0, steer_sensor=0):
        if self.recording:
            self.queue.put((t, steer, throttle, brake, speed, steer_sensor))

    def _worker(self):
        while True:
            if not self.queue.empty() and self.session_dir:
                csv_path = os.path.join(self.session_dir, "recording.csv")
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    if not file_exists:
                        writer.writerow(['Timestamp', 'Steering', 'SteeringSpeed', 'Throttle', 'Brake', 'SpeedSensor', 'SteeringSensor'])
                    
                    while not self.queue.empty():
                        t, s, th, b, sp, ss = self.queue.get()
                        writer.writerow([t, s, 0, th, b, sp, ss])
            time.sleep(0.01)

# --- MAIN LOOP ---

def main():
    try:
        bus = can.ThreadSafeBus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=500000)
    except Exception as e:
        print(f"CAN Error: {e}"); return

    # Setup Periodic Tasks for Watchdog Safety
    steer_msg = can.Message(arbitration_id=0x220, data=[0]*8, is_extended_id=False)
    motor_msg = can.Message(arbitration_id=0x330, data=[0]*8, is_extended_id=False)
    steer_task = bus.send_periodic(steer_msg, 0.02)
    motor_task = bus.send_periodic(motor_msg, 0.02)

    # Initialize Workers
    recorder = ControlDataWorker()
    vid_worker = VideoWorker()
    lidar_rec = LidarWorker()

    cur_steer, cur_throttle = 0.0, 0.0

    print("\n[READY] X: Record | Y: Stop | B: Emergency Stop")

    with open(JOYSTICK_DEV, 'rb') as js:
        while True:
            r, _, _ = select.select([js], [], [], 0.01)
            if r:
                ev = js.read(8)
                _, value, type_, number = struct.unpack('IhBB', ev)
                
                if type_ & EVENT_AXIS:
                    norm = round(value / 32767.0, 4)
                    if number == AXIS_STEER:
                        cur_steer = norm
                        steer_msg.data = list(struct.pack("<f", cur_steer)) + [0]*4
                        steer_task.modify_data(steer_msg)
                    
                    elif number == AXIS_THROTTLE:
                        cur_throttle = round(-norm, 4)
                        speed = int(abs(cur_throttle) * 45) if abs(value) > 5000 else 0
                        gear = 1 if cur_throttle > 0.15 else (2 if cur_throttle < -0.15 else 0)
                        # FIXED: Gear is index 1
                        motor_msg.data = [speed, gear, 0, 0, 0, 0, 0, 0]
                        motor_task.modify_data(motor_msg)

                elif type_ & EVENT_BUTTON and value == 1:
                    if number == BTN_X_RECORD:
                        sess_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = os.path.join(BASE_DATA_DIR, f"session_{sess_id}")
                        recorder.start_rec(path)
                        vid_worker.start_rec(path)
                        lidar_rec.start_rec(path)
                        print(f"\n>>> RECORDING: {sess_id}")
                    
                    elif number == BTN_Y_STOP_REC:
                        recorder.stop_rec(); vid_worker.stop_rec(); lidar_rec.stop_rec()
                        print("\n>>> STOPPED")

                    elif number == BTN_B_KILL:
                        steer_task.stop(); motor_task.stop(); sys.exit(0)

            # High-frequency data logging
            if recorder.recording:
                recorder.put(time.time(), cur_steer, cur_throttle)

if __name__ == "__main__":
    main()