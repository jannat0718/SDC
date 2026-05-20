import select
import struct
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import can

CAN_ID_MAP = {
    0x110: "brake",
    0x220: "steering",
    0x330: "throttle",
    0x1e5: "steering_sensor",
}

AXIS_LEFT_X = 0
AXIS_RIGHT_Y = 3
BTN_B_KILL = 1
BTN_X_RECORD = 3
BTN_Y_STOP_REC = 4
EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02

@dataclass
class ControlState:
    steering_command: float = 0.0
    throttle_command: float = 0.0
    emergency_stop: bool = False
    camera_active: bool = False

class CanListener:
    def __init__(self, bus):
        self.bus = bus
        self.data = {name: None for name in CAN_ID_MAP.values()}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                msg = self.bus.recv(0.1)
                if msg and msg.arbitration_id in CAN_ID_MAP:
                    name = CAN_ID_MAP[msg.arbitration_id]
                    with self.lock:
                        self.data[name] = list(msg.data)
            except:
                continue

    def get_values(self):
        with self.lock:
            return dict(self.data)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

def decode_feedback(can_data: Dict[str, Optional[list]]) -> Dict[str, object]:
    steering_raw = can_data.get("steering")
    throttle_raw = can_data.get("throttle")
    brake_raw = can_data.get("brake")
    steering_sensor_raw = can_data.get("steering_sensor")

    if steering_raw and len(steering_raw) >= 8:
        feedback_steering = round(struct.unpack("<f", bytes(steering_raw[:4]))[0], 4)
        feedback_steering_speed = struct.unpack(">I", bytes(steering_raw[4:8]))[0]
    else:
        feedback_steering = ""
        feedback_steering_speed = ""

    feedback_throttle = round(throttle_raw[0] / 100.0, 4) if throttle_raw else ""
    feedback_brake = round(brake_raw[0] / 100.0, 4) if brake_raw else ""

    if steering_sensor_raw and len(steering_sensor_raw) >= 3:
        feedback_steering_sensor = (steering_sensor_raw[1] << 8) | steering_sensor_raw[2]
        if feedback_steering_sensor > 32767:
            feedback_steering_sensor -= 65536
    else:
        feedback_steering_sensor = ""

    return {
        "feedback_brake": feedback_brake,
        "feedback_steering": feedback_steering,
        "feedback_steering_speed": feedback_steering_speed,
        "feedback_throttle": feedback_throttle,
        "feedback_steering_sensor": feedback_steering_sensor,
    }

class XboxCanController:
    def __init__(
        self,
        joystick_dev: str = "/dev/input/js0",
        can_interface: str = "socketcan",
        can_channel: str = "can0",
        can_bitrate: int = 500000,
        sending_speed: float = 0.04,
    ):
        self.joystick_dev = joystick_dev
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.can_bitrate = can_bitrate
        self.sending_speed = sending_speed

        self.bus = None
        self.can_listener: Optional[CanListener] = None
        self.js_file = None
        self.steer_task = None
        self.motor_task = None
        self.state = ControlState()

    def start(self) -> None:
        # CAN Bus with Fallback
        try:
            self.bus = can.Bus(interface=self.can_interface, channel=self.can_channel, bitrate=self.can_bitrate)
            print(f"[INFO] Connected to physical CAN: {self.can_channel}")
        except Exception as e:
            print(f"[WARN] CAN Hardware not found ({e}). Using Virtual Bus.")
            self.bus = can.Bus(interface='virtual', channel='test')

        self.bus.set_filters([
            {"can_id": 0x110, "can_mask": 0x7FF, "extended": False},
            {"can_id": 0x220, "can_mask": 0x7FF, "extended": False},
            {"can_id": 0x330, "can_mask": 0x7FF, "extended": False},
            {"can_id": 0x1e5, "can_mask": 0x7FF, "extended": False},
        ])
        
        self.can_listener = CanListener(self.bus)
        self.steer_msg = can.Message(arbitration_id=0x220, data=[0] * 8, is_extended_id=False)
        self.motor_msg = can.Message(arbitration_id=0x330, data=[0, 0, 1, 0, 0, 0, 0, 0], is_extended_id=False)

        self.steer_task = self.bus.send_periodic(self.steer_msg, self.sending_speed)
        self.motor_task = self.bus.send_periodic(self.motor_msg, self.sending_speed)

        try:
            self.js_file = open(self.joystick_dev, "rb")
            print(f"[INFO] Controller opened at {self.joystick_dev}")
        except Exception as e:
            print(f"[ERROR] Could not open joystick: {e}")

    def step(self, timeout_s: float = 0.001) -> Dict[str, object]:
        if self.js_file is None: return {}
        ready, _, _ = select.select([self.js_file], [], [], timeout_s)
        if not ready:
            fb = decode_feedback(self.can_listener.get_values()) if self.can_listener else {}
            return {"state": self.state, "feedback": fb}

        event_data = self.js_file.read(8)
        if not event_data: return {"state": self.state}
        _, value, type_, number = struct.unpack("IhBB", event_data)

        if type_ & EVENT_BUTTON and value == 1:
            if number == BTN_B_KILL:
                self.state.emergency_stop = True
                print("[STOP] Emergency stop triggered!")
            elif number == BTN_X_RECORD:
                self.state.camera_active = True
                print("[RUN] Camera Active")
            elif number == BTN_Y_STOP_REC:
                self.state.camera_active = False
                print("[STOP] Camera Deactivated")

        elif type_ & EVENT_AXIS:
            if number == AXIS_LEFT_X and self.steer_task:
                self.state.steering_command = round(value / 32767.0, 4)
                m = can.Message(arbitration_id=0x220, data=list(struct.pack("<f", self.state.steering_command)) + [0]*4)
                self.steer_task.modify_data(m)
            elif number == AXIS_RIGHT_Y and self.motor_task:
                self.state.throttle_command = round(-(value / 32767.0), 4)
                gear = 1 if self.state.throttle_command > 0.1 else (2 if self.state.throttle_command < -0.1 else 0)
                speed = int(abs(self.state.throttle_command) * 45)
                m = can.Message(arbitration_id=0x330, data=[speed, 0, gear, 0, 0, 0, 0, 0])
                self.motor_task.modify_data(m)

        fb = decode_feedback(self.can_listener.get_values()) if self.can_listener else {}
        return {"state": self.state, "feedback": fb}

    def stop(self) -> None:
        if self.steer_task: self.steer_task.stop()
        if self.motor_task: self.motor_task.stop()
        if self.can_listener: self.can_listener.stop()
        if self.js_file: self.js_file.close()
        if self.bus: self.bus.shutdown()