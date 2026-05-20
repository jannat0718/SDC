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
dttstr_int = lambda: int(datetime.now().strftime("%Y%m%d%H%M%S"))

# --- CONFIGURATION ---
JOYSTICK_DEV = '/dev/input/js0'
CAN_INTERFACE = 'socketcan'
CAN_CHANNEL = 'can0'
CAN_BITRATE = 500000
SENDING_SPEED = 0.04
SAVE_DIR = "/home/sdc/SDC.ExampleCode/data/SDC_March_Sessions/DATA/"
SAVE_DIR_LIDAR = "/home/sdc/SDC.ExampleCode/data/SDC_March_Sessions/DATA/"
SAVE_DIR_VIDEO = "/home/sdc/SDC.ExampleCode/data/SDC_March_Sessions/DATA/"
SAVE_DIR_CONTROL = "/home/sdc/SDC.ExampleCode/data/SDC_March_Sessions/DATA/"
LIDAR_PORT = '/dev/ttyUSB0'

# Official SDC control CAN identifiers
CONTROL_CAN_IDS = {
    'brake': 0x110,
    'steering': 0x220,
    'throttle': 0x330,
}

# Official SDC feedback CAN identifiers
FEEDBACK_CAN_ID_MAP = {
    0x710: 'brake_ecu',
    0x720: 'steering_ecu',
    0x730: 'motor_ecu',
    0x1e5: 'steering_sensor',
    0x440: 'speed_sensor',
}

# Input Mapping
AXIS_LEFT_X = 0
AXIS_LEFT_TRIGGER = 2
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

    def start_rec(self, session_id):
        self.session_dir = os.path.join(SAVE_DIR_CONTROL, f"session_{session_id}")
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
                        writer.writerow([
                            'timestamp',
                            'command_steering',
                            'command_throttle',
                            'command_brake',
                            'command_gear',
                            'feedback_brake_position',
                            'feedback_brake_target',
                            'feedback_brake_direction',
                            'feedback_brake_speed',
                            'feedback_brake_error',
                            'feedback_steering_last_angle',
                            'feedback_steering_target_angle',
                            'feedback_steering_direction',
                            'feedback_steering_error',
                            'feedback_motor_throttle_pct',
                            'feedback_motor_braking',
                            'feedback_motor_gear',
                            'feedback_motor_idle',
                            'feedback_steering_sensor',
                            'feedback_speed_kmh'
                        ])
                    
                    while not self.queue.empty():
                        writer.writerow(self.queue.get())
            else:
                time.sleep(0.1)


def decode_feedback(can_data):
    brake_raw = can_data['brake_ecu']
    steering_raw = can_data['steering_ecu']
    motor_raw = can_data['motor_ecu']
    steering_sensor_raw = can_data['steering_sensor']
    speed_sensor_raw = can_data['speed_sensor']

    if brake_raw and len(brake_raw) >= 7:
        feedback_brake_position = struct.unpack('>H', bytes(brake_raw[:2]))[0]
        feedback_brake_target = struct.unpack('>H', bytes(brake_raw[2:4]))[0]
        feedback_brake_direction = 'extending' if brake_raw[4] else 'retracting'
        feedback_brake_speed = brake_raw[5]
        feedback_brake_error = bool(brake_raw[6])
    else:
        feedback_brake_position = ''
        feedback_brake_target = ''
        feedback_brake_direction = ''
        feedback_brake_speed = ''
        feedback_brake_error = ''

    if steering_raw and len(steering_raw) >= 6:
        feedback_steering_last_angle = struct.unpack('>h', bytes(steering_raw[:2]))[0]
        feedback_steering_target_angle = struct.unpack('>h', bytes(steering_raw[2:4]))[0]
        feedback_steering_direction = 'counterclockwise' if steering_raw[4] else 'clockwise'
        feedback_steering_error = bool(steering_raw[5])
    else:
        feedback_steering_last_angle = ''
        feedback_steering_target_angle = ''
        feedback_steering_direction = ''
        feedback_steering_error = ''

    if motor_raw and len(motor_raw) >= 4:
        feedback_motor_throttle_pct = round((motor_raw[0] / 255.0) * 100.0, 4)
        feedback_motor_braking = bool(motor_raw[1])
        feedback_motor_gear = {0: 'neutral', 1: 'forward', 2: 'reverse'}.get(motor_raw[2], f'invalid_{motor_raw[2]}')
        feedback_motor_idle = bool(motor_raw[3])
    else:
        feedback_motor_throttle_pct = ''
        feedback_motor_braking = ''
        feedback_motor_gear = ''
        feedback_motor_idle = ''

    if steering_sensor_raw and len(steering_sensor_raw) >= 2:
        feedback_steering_sensor = struct.unpack('>h', bytes(steering_sensor_raw[:2]))[0]
    else:
        feedback_steering_sensor = ''

    if speed_sensor_raw and len(speed_sensor_raw) >= 2:
        feedback_speed_kmh = round(struct.unpack('>H', bytes(speed_sensor_raw[:2]))[0] / 10.0, 2)
    else:
        feedback_speed_kmh = ''

    return {
        'feedback_brake_position': feedback_brake_position,
        'feedback_brake_target': feedback_brake_target,
        'feedback_brake_direction': feedback_brake_direction,
        'feedback_brake_speed': feedback_brake_speed,
        'feedback_brake_error': feedback_brake_error,
        'feedback_steering_last_angle': feedback_steering_last_angle,
        'feedback_steering_target_angle': feedback_steering_target_angle,
        'feedback_steering_direction': feedback_steering_direction,
        'feedback_steering_error': feedback_steering_error,
        'feedback_motor_throttle_pct': feedback_motor_throttle_pct,
        'feedback_motor_braking': feedback_motor_braking,
        'feedback_motor_gear': feedback_motor_gear,
        'feedback_motor_idle': feedback_motor_idle,
        'feedback_steering_sensor': feedback_steering_sensor,
        'feedback_speed_kmh': feedback_speed_kmh,
    }

# --- CAN LISTENER ---
class CanListener:
    def __init__(self, bus):
        self.bus = bus
        self.data = {name: None for name in FEEDBACK_CAN_ID_MAP.values()}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            msg = self.bus.recv(0.1)
            if msg and msg.arbitration_id in FEEDBACK_CAN_ID_MAP:
                name = FEEDBACK_CAN_ID_MAP[msg.arbitration_id]
                with self.lock:
                    self.data[name] = list(msg.data)

    def get_values(self):
        with self.lock:
            return dict(self.data)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

# --- LIDAR LOGIC ---
class LidarRecorder:
    def __init__(self):
        self.recording = False
        self.running = True
        self.lidar = None
        self.scan_count = 0
        self.session_id = 0
        self.session_dir = ""
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self, session_id):
        self.session_id = session_id
        self.session_dir = os.path.join(SAVE_DIR_LIDAR, str(session_id))
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
                        filename = os.path.join(self.session_dir, f"Lidar_{self.session_id}_{self.scan_count:05d}.csv")
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
        self.counter = 0
        self.session_dir = ""
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def start_rec(self):
        if not self.recording:
            self.counter = 1 + dttstr_int()
            self.session_dir = os.path.join(SAVE_DIR_VIDEO, str(self.counter))
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"\n>>> STARTING SESSION: {self.counter}")
            self.recording = True
            self.img_count = 0
            return self.counter

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
                    if cam_right: out_r = cv2.VideoWriter(os.path.join(self.session_dir, f'Right{self.counter}.mp4'), fourcc, 30.0, (w, h))
                    if cam_mid:   out_m = cv2.VideoWriter(os.path.join(self.session_dir, f'Middle{self.counter}.mp4'), fourcc, 30.0, (w, h))
                    if cam_left:  out_l = cv2.VideoWriter(os.path.join(self.session_dir, f'Left{self.counter}.mp4'), fourcc, 30.0, (w, h))
                
                if ret_r and out_r: out_r.write(frame_r)
                if ret_m and out_m: out_m.write(frame_m)
                if ret_l and out_l: out_l.write(frame_l)

                if time.time() - self.last_capture_time >= 0.04:
                    if ret_m and frame_m is not None:
                        filename = os.path.join(self.session_dir, f"MidCam_{self.counter}_{self.img_count:05d}.png")
                        cv2.imwrite(filename, frame_m, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        self.img_count += 1
                        self.last_capture_time = time.time()
            else:
                for o in [out_l, out_m, out_r]:
                    if o is not None: o.release()
                out_l = out_m = out_r = None

        for c in [cam_left, cam_mid, cam_right]:
            if c: c.release()

# --- MAIN ---
def main():
    recorder = CameraRecorder()
    lidar_rec = LidarRecorder()
    control_worker = ControlDataWorker()
    steering_log = None

    try:
        bus = can.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
        bus.set_filters([
            {'can_id': 0x710, 'can_mask': 0x7FF, 'extended': False},
            {'can_id': 0x720, 'can_mask': 0x7FF, 'extended': False},
            {'can_id': 0x730, 'can_mask': 0x7FF, 'extended': False},
            {'can_id': 0x1e5, 'can_mask': 0x7FF, 'extended': False},
            {'can_id': 0x440, 'can_mask': 0x7FF, 'extended': False},
        ])
        can_listener = CanListener(bus)
    except OSError:
        print("CAN Error: Interface not found.")
        recorder.cleanup(); lidar_rec.cleanup(); control_worker.cleanup(); sys.exit(1)

    steer_msg = can.Message(arbitration_id=CONTROL_CAN_IDS['steering'], data=[0]*8, is_extended_id=False)
    throttle_msg = can.Message(arbitration_id=CONTROL_CAN_IDS['throttle'], data=[0, 0, 1, 0, 0, 0, 0, 0], is_extended_id=False)
    brake_msg = can.Message(arbitration_id=CONTROL_CAN_IDS['brake'], data=[0]*8, is_extended_id=False)
    steer_task = bus.send_periodic(steer_msg, SENDING_SPEED)
    throttle_task = bus.send_periodic(throttle_msg, SENDING_SPEED)
    brake_task = bus.send_periodic(brake_msg, SENDING_SPEED)

    # Initialize signed values for logging
    cur_steer = 0.0
    cur_throttle = 0.0
    cur_brake = 0.0
    cur_gear = 'neutral'

    print("READY. X: Record | Y: Stop | B: Kill")

    try:
        with open(JOYSTICK_DEV, 'rb') as js_file:
            while True:
                # Log Control data if recording
                if recorder.recording:
                    feedback = decode_feedback(can_listener.get_values())
                    control_worker.put([
                        time.time(),
                        cur_steer,
                        cur_throttle,
                        cur_brake,
                        cur_gear,
                        feedback['feedback_brake_position'],
                        feedback['feedback_brake_target'],
                        feedback['feedback_brake_direction'],
                        feedback['feedback_brake_speed'],
                        feedback['feedback_brake_error'],
                        feedback['feedback_steering_last_angle'],
                        feedback['feedback_steering_target_angle'],
                        feedback['feedback_steering_direction'],
                        feedback['feedback_steering_error'],
                        feedback['feedback_motor_throttle_pct'],
                        feedback['feedback_motor_braking'],
                        feedback['feedback_motor_gear'],
                        feedback['feedback_motor_idle'],
                        feedback['feedback_steering_sensor'],
                        feedback['feedback_speed_kmh'],
                    ])

                ready, _, _ = select.select([js_file], [], [], 0.01)
                if ready:
                    event_data = js_file.read(8)
                    if not event_data: break
                    _, value, type_, number = struct.unpack('IhBB', event_data)

                    if type_ & EVENT_INIT: continue

                    if type_ & EVENT_BUTTON and value == 1:
                        if number == BTN_X_RECORD:
                            counter = recorder.start_rec()
                            lidar_rec.start_rec(counter)
                            control_worker.start_rec(counter)
                            steering_log = open(os.path.join(recorder.session_dir, "control_log.csv"), "w")
                            steering_log.write("timestamp,type,value1,value2\n")
                        elif number == BTN_Y_STOP_REC:
                            recorder.stop_rec()
                            lidar_rec.stop_rec()
                            control_worker.stop_rec()
                            if steering_log:
                                steering_log.close()
                                steering_log = None
                        elif number == BTN_B_KILL:
                            if steering_log:
                                steering_log.close()
                            recorder.cleanup(); lidar_rec.cleanup(); control_worker.cleanup()
                            can_listener.stop(); steer_task.stop(); throttle_task.stop(); brake_task.stop(); bus.shutdown(); sys.exit(0)

                    elif type_ & EVENT_AXIS:
                        if number == AXIS_LEFT_X:
                            cur_steer = round(value / 32767.0, 4)
                            steer_msg.data = list(struct.pack("<f", cur_steer)) + [0]*4
                            steer_task.modify_data(steer_msg)
                            if steering_log:
                                steering_log.write(f"{time.time()},steering,{cur_steer},\n")

                        elif number == AXIS_RIGHT_Y:
                            cur_throttle = round(-(value / 32767.0), 4)
                            speed = int(abs(cur_throttle) * 45) if abs(value) > 5000 else 0
                            gear = 1 if cur_throttle > 0.15 else (2 if cur_throttle < -0.15 else 0)
                            cur_gear = {0: 'neutral', 1: 'forward', 2: 'reverse'}[gear]
                            throttle_msg.data = [speed, 0, gear, 0, 0, 0, 0, 0]
                            throttle_task.modify_data(throttle_msg)
                            if steering_log:
                                steering_log.write(f"{time.time()},throttle,{speed},{gear}\n")

                        elif number == AXIS_LEFT_TRIGGER:
                            # Left trigger: -32767 (released) to +32767 (fully pressed)
                            cur_brake = round(max(0, (value + 32767) / 65534.0), 4)
                            brake_val = int(cur_brake * 100)
                            brake_msg.data = [brake_val, 0, 0, 0, 0, 0, 0, 0]
                            brake_task.modify_data(brake_msg)
                            if steering_log:
                                steering_log.write(f"{time.time()},brake,{cur_brake},\n")

    except KeyboardInterrupt:
        if steering_log:
            steering_log.close()
        recorder.cleanup(); lidar_rec.cleanup(); control_worker.cleanup(); can_listener.stop(); bus.shutdown()

if __name__ == "__main__":
    main()