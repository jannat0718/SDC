r"""
perception.py - Object Projection + Tracking
=============================================
Handles everything between raw detection bbox and world coordinates.

Modules:
  1. GroundPlaneProjector  - converts bbox pixels -> world (x,y) metres
                             using camera intrinsics + known camera height
                             No tilt assumed (middle camera is level)

  2. ObjectTracker         - manages per-frame detections:
                             * rolling average for stable positions
                             * 0-6m visibility window
                             * stationary freeze (VO drift prevention)
                             * terminal print of detections

UPGRADE PATH:
  When BEV homography is available from teammate:
    Replace GroundPlaneProjector.project() with:
      bev_pt = cv2.perspectiveTransform([[(u,v)]], H_bev)
      rel_x, rel_y = bev_pt[0,0]
    Everything else stays the same.

CAMERA HEIGHT:
  1.5 feet = 0.4572m  (no tilt, level mount)
"""

import cv2
import numpy as np
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DetectedObject:
    """Single detection with world position."""
    label:     str
    class_id:  int
    conf:      float
    bbox:      Tuple[int,int,int,int]   # x1,y1,x2,y2 in image pixels
    depth_m:   float = 0.0              # metres ahead of kart
    lateral_m: float = 0.0             # metres left(-)/right(+) of kart centre
    world_x:   float = 0.0             # absolute world X (metres)
    world_y:   float = 0.0             # absolute world Y (metres)
    frame_num: int   = 0
    timestamp: float = 0.0


@dataclass
class TrackedObject:
    """Tracked object with rolling position average."""
    label:     str
    class_id:  int
    world_x:   float
    world_y:   float
    depth_m:   float
    conf:      float
    last_seen: int    = 0               # frame number
    positions: deque  = field(default_factory=lambda: deque(maxlen=5))

    def update(self, wx: float, wy: float, depth: float, frame: int):
        self.positions.append((wx, wy))
        self.last_seen = frame
        self.depth_m   = depth
        # Stable position = average of last N frames
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        self.world_x = float(np.mean(xs))
        self.world_y = float(np.mean(ys))


# ==============================================================================
# MODULE 1 - GROUND PLANE PROJECTOR
# ==============================================================================

class GroundPlaneProjector:
    """
    Projects bounding box bottom-centre from image pixels to world (x,y).

    Coordinate mapping (no camera tilt, level mount):

      Image v (vertical, bbox bottom)
          -> depth = cam_height * fy / (v - cy)
          -> world forward distance (metres ahead of kart)

      Image u (horizontal, bbox centre)
          -> lateral = (u - cx) / fx * depth
          -> world lateral offset (metres left/right of kart)

    Then rotate by kart heading to get absolute world (x,y):
      world_x = kart_x + depth * cos(heading) - lateral * sin(heading)
      world_y = kart_y + depth * sin(heading) + lateral * cos(heading)

    UPGRADE: replace project() body with BEV homography when available.
    """

    CAMERA_HEIGHT_M = 0.4572   # 1.5 feet, level mount, no tilt

    def __init__(self, fx: float, fy: float,
                 cx: float, cy: float,
                 camera_height_m: float = CAMERA_HEIGHT_M):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h  = camera_height_m
        print(f"[GroundPlaneProjector] "
              f"h={self.h:.4f}m  fx={fx:.1f}  fy={fy:.1f}  "
              f"cx={cx:.1f}  cy={cy:.1f}")

    def project(self, det: DetectedObject,
                kart_x: float, kart_y: float,
                heading: float) -> DetectedObject:
        """
        Fill det.depth_m, det.lateral_m, det.world_x, det.world_y.
        kart_x, kart_y  : kart world position from VO (metres)
        heading         : kart heading from VO (radians, pi = facing left)
        """
        x1, y1, x2, y2 = det.bbox

        # -- u: horizontal centre -> lateral offset
        u = (x1 + x2) / 2.0

        # -- v: bottom edge -> depth
        v    = float(y2)
        dv   = v - self.cy          # pixels below principal point
        if dv < 5.0:
            # Object near or above horizon - too far to estimate reliably
            det.depth_m   = 15.0
            det.lateral_m = 0.0
            det.world_x   = kart_x + 15.0 * math.cos(heading)
            det.world_y   = kart_y + 15.0 * math.sin(heading)
            return det

        depth   = float(np.clip((self.h * self.fy) / dv, 0.3, 15.0))
        lateral = (u - self.cx) / self.fx * depth   # + = right of camera

        det.depth_m   = depth
        det.lateral_m = lateral

        # -- rotate relative offset by kart heading -> world coords
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        det.world_x = kart_x + depth * cos_h - lateral * sin_h
        det.world_y = kart_y + depth * sin_h + lateral * cos_h

        return det


# ==============================================================================
# MODULE 2 - OBJECT TRACKER
# ==============================================================================

class ObjectTracker:
    """
    Manages detected objects:
      - Projects each bbox to world coords via GroundPlaneProjector
      - Rolling 5-frame average per object for stable map position
      - Visibility window: only show objects 0-6m ahead of kart
      - Stationary detection: freeze VO when optical flow is near zero
      - Terminal logging: prints detections per frame
    """

    VISIBILITY_MIN_M   =  0.0   # metres - don't show objects behind kart
    VISIBILITY_MAX_M   =  6.0   # metres - don't show objects too far ahead
    STATIONARY_THRESH  =  0.35  # Tailored for 320x180 downsampled resolution flow scaling
    DEBOUNCE_FRAMES    =  5     # Consecutively logged stopped frames required to trigger lock
    STALE_FRAMES       = 10     # remove tracked object if not seen for N frames

    def __init__(self, projector: GroundPlaneProjector):
        self.proj         = projector
        self.tracked:     Dict[str, TrackedObject] = {}
        self.is_stationary = False
        self.stationary_frames = 0
        self._frame_num   = 0
        self._prev_gray   = None

    # -- public API ------------------------------------------------------------

    def update(self, raw_detections: list,
               kart_x: float, kart_y: float,
               heading: float,
               frame: np.ndarray) -> List[TrackedObject]:
        """
        Call once per frame.
        raw_detections: list of dicts from ObjectDetector.detect()
        Returns: list of TrackedObject currently visible (within 0-6m ahead)
        """
        self._frame_num += 1
        now = time.time()

        # -- Check if kart is stationary (suppress VO drift)
        # now called before VO in av_map.py to ensure timing is correct

        # -- Project each detection to world coords
        for d in raw_detections:
            det = DetectedObject(
                label    = d['label'],
                class_id = d['class_id'],
                conf     = d['conf'],
                bbox     = (d['x1'], d['y1'], d['x2'], d['y2']),
                frame_num = self._frame_num,
                timestamp = now,
            )
            det = self.proj.project(det, kart_x, kart_y, heading)

            # -- Update or create tracked object
            key = det.label   # one track per class for simplicity
                              # extend to per-instance ID when needed
            if key in self.tracked:
                self.tracked[key].update(
                    det.world_x, det.world_y, det.depth_m, self._frame_num)
                self.tracked[key].conf = det.conf
            else:
                tr = TrackedObject(
                    label    = det.label,
                    class_id = det.class_id,
                    world_x  = det.world_x,
                    world_y  = det.world_y,
                    depth_m  = det.depth_m,
                    conf     = det.conf,
                    last_seen= self._frame_num,
                )
                tr.positions.append((det.world_x, det.world_y))
                self.tracked[key] = tr

        # -- Remove stale tracks
        stale = [k for k,v in self.tracked.items()
                 if self._frame_num - v.last_seen > self.STALE_FRAMES]
        for k in stale:
            del self.tracked[k]

        # -- Filter to visible objects only (0-6m ahead)
        visible = self._filter_visible(kart_x, kart_y, heading)

        # -- Terminal print
        self._print_frame(visible, kart_x, kart_y, heading)

        return visible

    def is_kart_stopped(self) -> bool:
        return self.is_stationary

    # -- internals -------------------------------------------------------------

    def _check_stationary(self, frame: np.ndarray):
        """Detect if kart is stopped using optical flow magnitude with tracking persistence."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))   # small for speed

        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 2, 8, 2, 5, 1.1, 0)
            mag = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
            was_stationary = self.is_stationary
            
            # Apply downscaled threshold criteria and debounce counter
            if mag < self.STATIONARY_THRESH:
                self.stationary_frames += 1
                if self.stationary_frames >= self.DEBOUNCE_FRAMES:
                    self.is_stationary = True
            else:
                self.stationary_frames = 0
                self.is_stationary = False

            if self.is_stationary and not was_stationary:
                print(f"\n  [Tracker] Kart STOPPED  (flow={mag:.2f} < {self.STATIONARY_THRESH})")
            elif not self.is_stationary and was_stationary:
                print(f"\n  [Tracker] Kart MOVING   (flow={mag:.2f})")

        self._prev_gray = gray

    def _forward_dist(self, kart_x: float, kart_y: float,
                      heading: float, wx: float, wy: float) -> float:
        """Signed forward distance from kart to world point. + = ahead."""
        dx = wx - kart_x
        dy = wy - kart_y
        return dx * math.cos(heading) + dy * math.sin(heading)

    def _filter_visible(self, kart_x: float, kart_y: float,
                        heading: float) -> List[TrackedObject]:
        visible = []
        for obj in self.tracked.values():
            fd = self._forward_dist(kart_x, kart_y, heading,
                                    obj.world_x, obj.world_y)
            if self.VISIBILITY_MIN_M <= fd <= self.VISIBILITY_MAX_M:
                visible.append(obj)
        return visible

    def _print_frame(self, visible: List[TrackedObject],
                     kart_x: float, kart_y: float, heading: float):
        """Print detections to terminal - only when objects present."""
        if not visible:
            return
        print(f"\n  [F{self._frame_num:05d}] "
              f"Kart({kart_x:.2f},{kart_y:.2f})m  "
              f"hdg={math.degrees(heading):.1f}°  "
              f"stopped={self.is_stationary}")
        for obj in visible:
            fd = self._forward_dist(kart_x, kart_y, heading,
                                    obj.world_x, obj.world_y)
            
            # Formats safely to bypass value type interpretation errors
            if hasattr(obj, 'lateral_m') and obj.lateral_m is not None:
                lateral_str = f"{obj.lateral_m:.2f}m"
            else:
                lateral_str = "??m"
                     
            print(f"    {obj.label:<20} "
                  f"conf={obj.conf:.2f}  "
                  f"depth={obj.depth_m:.2f}m  "
                  f"lateral={lateral_str:<7} "  
                  f"forward={fd:.2f}m  "
                  f"world=({obj.world_x:.2f},{obj.world_y:.2f})m")