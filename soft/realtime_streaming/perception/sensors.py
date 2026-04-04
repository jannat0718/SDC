import glob
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None


@dataclass
class LidarSample:
    timestamp: float
    points: np.ndarray


def resolve_lidar_port(requested_port: str) -> str:
    if requested_port and requested_port.lower() != "auto":
        return requested_port

    linux_candidates = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if linux_candidates:
        return linux_candidates[0]

    raise RuntimeError("Could not auto-detect LiDAR port. Pass --lidar-port explicitly.")


class CameraStream:
    def __init__(self, camera_index: int = 0, width: int = 848, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest: Optional[Tuple[float, np.ndarray]] = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._latest_lock:
                self._latest = (time.time(), frame)

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._latest_lock:
            if self._latest is None:
                return None
            ts, frame = self._latest
            return ts, frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


class LidarStream:
    def __init__(self, port: str, max_queue_size: int = 3):
        self.port = port
        self.max_queue_size = max_queue_size

        self._lidar: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[LidarSample] = queue.Queue(maxsize=max_queue_size)

    def start(self) -> None:
        if RPLidar is None:
            raise RuntimeError("rplidar package is not installed. Install with: pip install rplidar-roboticia")

        port = resolve_lidar_port(self.port)
        self._lidar = RPLidar(port, timeout=3)
        self._lidar.start_motor()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._lidar is not None
        try:
            for scan in self._lidar.iter_scans(max_buf_meas=1000):
                if not self._running:
                    break

                points = np.array(scan, dtype=np.float32)
                sample = LidarSample(timestamp=time.time(), points=points)

                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

                self._queue.put_nowait(sample)
        except Exception as exc:
            print(f"[LiDAR] Reader stopped: {exc}")

    def get_latest(self) -> Optional[LidarSample]:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass


def startup_self_check(camera_index: int, lidar_port: str) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Startup check failed: camera index {camera_index} is not available.")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Startup check failed: camera opened but no frame received.")

    if RPLidar is None:
        raise RuntimeError("Startup check failed: rplidar package is not installed.")

    resolved_port = resolve_lidar_port(lidar_port)
    lidar = None
    try:
        lidar = RPLidar(resolved_port, timeout=3)
        _ = lidar.get_info()
        _ = lidar.get_health()
    finally:
        if lidar is not None:
            try:
                lidar.disconnect()
            except Exception:
                pass
