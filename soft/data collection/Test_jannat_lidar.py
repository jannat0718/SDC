#!/usr/bin/env python3
import os
import struct
import sys
import can
import threading
import cv2
import time
import select
from datetime import datetime
from rplidar import RPLidar
import numpy as np
from queue import Queue

# Format with NO spaces or colons: YYYYMMDDHHMMSS
dttstr_int = lambda: int(datetime.now().strftime("%Y%m%d%H%M%S"))

# --- CONFIGURATION ---
JOYSTICK_DEV = '/dev/input/js0'
CAN_INTERFACE = 'socketcan'
CAN_CHANNEL = 'can0'
CAN_BITRATE = 500000
SENDING_SPEED = 0.04
SAVE_DIR = "/home/sdc/SDC.ExampleCode/Image" 
SAVE_DIR_LIDAR = "/home/sdc/SDC.ExampleCode/Lidar"
SAVE_DIR_VIDEO = "/home/sdc/SDC.ExampleCode/Video"
SAVE_DIR_CONTROL = "/home/sdc/SDC.ExampleCode/Control Data"
LIDAR_PORT = '/dev/ttyUSB0'

# Ensure directories exist
for folder in [SAVE_DIR, SAVE_DIR_LIDAR, SAVE_DIR_VIDEO, SAVE_DIR_CONTROL]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Input Mapping
AXIS_LEFT_X = 0
AXIS_RIGHT_Y = 3
BTN_B_KILL = 1
BTN_X_RECORD = 3
BTN_Y_STOP_REC = 4

EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02
EVENT_INIT = 0x80

# --- CONTROL DATA LOGGING ---
class ControlDataWorker:
    """Worker that writes control data (steering, throttle, brake) to a CSV file."""
    
    def __init__(self, data_queue: Queue, session_dir: str):
        self.queue = data_queue
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.file_pointer = open(os.path.join(session_dir, 'control_data.csv'), 'w')
        self.file_pointer.write('Timestamp|Steering|Throttle|Brake\n')
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.running = True
        self.thread.start()
    
    def put(self, data):
        self.queue.put(data)
    
    def stop(self):
        self.running = False
        self.queue.join()
        self.file_pointer.close()
    
    def _process(self):
        while self.running:
            timestamp, steering, throttle, brake = self.queue.get()
            self.file_pointer.write(f'{timestamp}|{steering:.6f}|{throttle:.2f}|{brake:.2f}\n')
            self.file_pointer.flush()
            self.queue.task_done()

# --- LIDAR LOGIC ---
class LidarRecorder:
    def __init__(self):
        self.recording = False
        self.running = True
        self.lidar = None
        self.session_id = 0
        self.scan_count = 0
        self.session_dir = None
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self, session_id):
        self.session_id = session_id
        self.scan_count = 0
        self.recording = True
        self.session_dir = os.path.join(SAVE_DIR_LIDAR, f"session_{session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

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
                        if not self.recording or not self.running:
                            break
                        filename = os.path.join(self.session_dir, f"Lidar_{self.session_id}_{self.scan_count:05d}.csv")
                        np.savetxt(filename, np.array(scan), delimiter=",", fmt='%.2f')
                        self.scan_count += 1
                except Exception as e:
                    print(f"\n[LiDAR Error] {e}")
                    self.recording = False
                finally:
                    if self.lidar:
                        self.lidar.stop()
                        self.lidar.stop_motor()
                        self.lidar.disconnect()
                        self.lidar = None
            else:
                time.sleep(0.1)

# --- CAMERA LOGIC (UNIFIED CAPTURE MODE) ---
class CameraRecorder:
    def __init__(self):
        self.recording = False
        self.running = True
        self.counter = 1 + dttstr_int()
        self.img_count = 0
        self.last_capture_time = 0
        self.session_dir = None
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self):
        if not self.recording:
            print(f"STARTING: VIDEO + IMAGES (3-Cam Video + Middle Camera Images @ 25 FPS)")
            self.recording = True
            self.session_dir = os.path.join(SAVE_DIR_VIDEO, f"session_{self.counter}")
            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(os.path.join(self.session_dir, "images"), exist_ok=True)
            self.img_count = 0

    def stop_rec(self):
        if self.recording:
            print(f"STOPPED Session {self.counter}")
            self.recording = False
            self.counter += 1

    def cleanup(self):
        self.running = False
        self.thread.join()

    def _worker(self):
        # --- SAFE INITIALIZATION ---
        def try_open(idx):
            c = cv2.VideoCapture(idx)
            if not c.isOpened():
                c.release()
                return None
            return c

        # Attempt to open cameras. If hardware is missing, it returns None instead of crashing.
        cam_right = try_open(0)
        cam_mid   = try_open(2)
        cam_left  = try_open(4)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_l = out_m = out_r = None

        while self.running:
            # Capture only if camera was successfully opened
            ret_r, frame_r = cam_right.read() if cam_right else (False, None)
            ret_m, frame_m = cam_mid.read() if cam_mid else (False, None)
            ret_l, frame_l = cam_left.read() if cam_left else (False, None)

            if self.recording:
                # Always record video from all 3 cameras
                # Initialize VideoWriters only for cameras that exist
                if out_r is None and out_m is None and out_l is None:
                    if frame_r is not None:
                        h, w = frame_r.shape[:2]
                        if cam_right: out_r = cv2.VideoWriter(os.path.join(self.session_dir, f'Right{self.counter}.mp4'), fourcc, 30.0, (w, h))
                        if cam_mid:   out_m = cv2.VideoWriter(os.path.join(self.session_dir, f'Middle{self.counter}.mp4'), fourcc, 30.0, (w, h))
                        if cam_left:  out_l = cv2.VideoWriter(os.path.join(self.session_dir, f'Left{self.counter}.mp4'), fourcc, 30.0, (w, h))
                
                if ret_r and out_r: out_r.write(frame_r)
                if ret_m and out_m: out_m.write(frame_m)
                if ret_l and out_l: out_l.write(frame_l)

                # Always capture images from middle camera at 25 FPS
                current_time = time.time()
                if current_time - self.last_capture_time >= 0.04:  # 25 FPS
                    if ret_m and frame_m is not None:
                        filename = os.path.join(self.session_dir, "images", f"MidCam_{self.counter}_{self.img_count:05d}.png")
                        cv2.imwrite(filename, frame_m, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        self.img_count += 1
                        self.last_capture_time = current_time
            else:
                # Close writers if not recording
                for o in [out_l, out_m, out_r]:
                    if o is not None: o.release()
                out_l = out_m = out_r = None

        for c in [cam_left, cam_mid, cam_right]:
            if c: c.release()

# --- MAIN ---
def main():
    recorder = CameraRecorder()
    lidar_rec = LidarRecorder()
    steering_log = None 
    control_data_worker = None
    control_queue = Queue() 

    try:
        bus = can.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
    except OSError:
        print("CAN Error: Interface not found.")
        recorder.cleanup(); lidar_rec.cleanup(); sys.exit(1)

    steer_msg = can.Message(arbitration_id=0x220, data=[0]*8, is_extended_id=False)
    motor_msg = can.Message(arbitration_id=0x330, data=[0, 0, 1, 0, 0, 0, 0, 0], is_extended_id=False)
    steer_task = bus.send_periodic(steer_msg, SENDING_SPEED)
    motor_task = bus.send_periodic(motor_msg, SENDING_SPEED)

    print("READY. X: Record | Y: Stop | B: Kill")

    try:
        with open(JOYSTICK_DEV, 'rb') as js_file:
            while True:
                ready, _, _ = select.select([js_file], [], [], 0.01)
                if ready:
                    event_data = js_file.read(8)
                    if not event_data: break
                    _, value, type_, number = struct.unpack('IhBB', event_data)
                    
                    if type_ & EVENT_INIT: continue
                    if type_ & EVENT_BUTTON and value == 1:
                        if number == BTN_X_RECORD:
                            recorder.start_rec()
                            lidar_rec.start_rec(recorder.counter)
                            steering_log = open(os.path.join(recorder.session_dir, "control_log.csv"), "w")
                            steering_log.write("timestamp,type,value1,value2\n")
                            control_data_worker = ControlDataWorker(control_queue, os.path.join(SAVE_DIR_CONTROL, f"session_{recorder.counter}"))
                        elif number == BTN_Y_STOP_REC:
                            recorder.stop_rec()
                            lidar_rec.stop_rec()
                            if steering_log:
                                steering_log.close()
                                steering_log = None
                            if control_data_worker:
                                control_data_worker.stop()
                                control_data_worker = None
                        elif number == BTN_B_KILL:
                            if steering_log:
                                steering_log.close()
                            if control_data_worker:
                                control_data_worker.stop()
                            recorder.cleanup(); lidar_rec.cleanup()
                            steer_task.stop(); motor_task.stop(); bus.shutdown(); sys.exit(0)

                    elif type_ & EVENT_AXIS:
                        if number == AXIS_LEFT_X:
                            angle = value / 32767.0
                            steer_msg.data = list(struct.pack("<f", angle)) + [0]*4
                            steer_task.modify_data(steer_msg)
                            if steering_log:
                                steering_log.write(f"{time.time()},steering,{angle},\n")
                            if control_data_worker:
                                control_data_worker.put((time.time(), angle, 0, 0))
                        elif number == AXIS_RIGHT_Y:
                            speed = int((abs(value) / 32767.0) * 50) if abs(value) > 5000 else 0
                            gear = 1 if value < -5000 else (2 if value > 5000 else 0)
                            throttle = speed / 100.0
                            brake = 0
                            motor_msg.data = [speed, 0, gear, 0, 0, 0, 0, 0]
                            motor_task.modify_data(motor_msg)
                            if steering_log:
                                steering_log.write(f"{time.time()},motor,{speed},{gear}\n")
                            if control_data_worker:
                                control_data_worker.put((time.time(), 0, throttle, brake))
    except KeyboardInterrupt:
        if steering_log:
            steering_log.close()
        if control_data_worker:
            control_data_worker.stop()
        recorder.cleanup(); lidar_rec.cleanup(); bus.shutdown()

if __name__ == "__main__":
    main()