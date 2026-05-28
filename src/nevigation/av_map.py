r"""
av_map.py - Navigation Map System
===================================

Modules:
  1. CameraCalibration   - auto-loads K from camera_K.json
  2. VisualOdometry      - Lucas-Kanade + Essential Matrix -> pose (x, y, heading)
  3. ObjectProjector     - ground plane depth + world coordinate projection
  4. HomographyHelper    - interactive tool to click 4 points on track image
  5. TrackMapper         - renders kart + trajectory + objects on track image
  6. MapSystem           - main loop, supports VIDEO and LIVE modes

USAGE:
  # Video mode (for tuning and testing):
  python3 av_map.py --mode video --video /path/to/Middle.mp4

  # Live camera mode:
  python3 av_map.py --mode live

  # Run homography calibration helper first:
  python3 av_map.py --calibrate-homography

CONTROLS (terminal - WSLg compatible, no imshow keypresses needed):
  q + ENTER  -> quit
  r + ENTER  -> reset VO pose to origin
  s + ENTER  -> save current map frame to Output/

CONFIG:
  Edit nevigation/nav_config.json to set:
    - track_image_path
    - scale_factor (tune by driving known distance)
    - camera_height_m (measure on real kart)
    - world_pts / image_pts (from homography calibration)
"""

import cv2
import math
import numpy as np
import json
import os
import sys
import time
import threading
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ── Path setup — works from any working directory ────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nevigation.utils import load_camera_params, load_nav_config, load_nav_config_merged, CameraParams, NavConfig
from nevigation.landmark_localizer import LandmarkLocalizer

# ── Paths ─────────────────────────────────────────────────────────────────────
CALIB_PATH  = os.path.join(os.path.dirname(__file__), "camera_K.json")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "nav_config.json")
# All output artefacts (map_final.png, snap_*.png, nav_preview.jpg, manual map
# saves) land in <project>/logs/.  Callers can override the location by setting
# VO_SDC_OUTPUT_DIR before importing this module (test_map_video.py does this).
_DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
OUTPUT_DIR  = os.environ.get("VO_SDC_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preview JPG path — open in Windows Explorer, press F5 to refresh
PREVIEW_FILE = os.path.join(OUTPUT_DIR, "nav_preview.jpg")

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
PALETTE = [
    (0, 255, 255), (0, 165, 255), (0, 255, 0),
    (255, 0, 0),   (255, 0, 255), (0, 128, 255),
    (128, 255, 0), (255, 128, 0), (0, 0, 255),
    (128, 0, 255),
]

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Pose:
    x:       float = 0.0
    y:       float = 0.0
    #heading: float = math.pi/2     
    heading: float = 0.0     # radians
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))

@dataclass
class DetectedObject:
    label:    str
    bbox:     tuple                # (x1, y1, x2, y2) image pixels
    depth:    Optional[float]      # metres — from ground plane or LiDAR
    world_x:  float = 0.0
    world_y:  float = 0.0
    class_id: int   = 0

# ─────────────────────────────────────────────
# MODULE 1 — CAMERA CALIBRATION LOADER
# ─────────────────────────────────────────────

class CameraCalibration:
    """
    Auto-loads K matrix and distortion from camera_K.json.
    Falls back to StreamCam spec values if file not found.
    """

    def __init__(self, calib_path: str = CALIB_PATH):
        try:
            self.cam = load_camera_params(calib_path)
            self.cam.print_summary()
        except FileNotFoundError:
            print(f"[CameraCalibration] {calib_path} not found — using StreamCam defaults.")
            # Fallback: values from your actual calibration
            self.cam = CameraParams({
                "K": [[959.57, 0.0, 636.88],
                      [0.0, 959.39, 352.32],
                      [0.0, 0.0, 1.0]],
                "dist": [0.0479, -0.1929, 0.0008, -0.0019, 0.1923],
                "derived": {"fx": 959.57, "fy": 959.39,
                            "cx": 636.88, "cy": 352.32,
                            "hfov_deg": 67.4, "vfov_deg": 41.1},
                "resolution": [1280, 720],
                "reprojection_error_px": 0.9486
            })

        self.K    = self.cam.K
        self.dist = self.cam.dist

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.undistort(frame, self.K, self.dist)

# ─────────────────────────────────────────────
# MODULE 2 — VISUAL ODOMETRY
# ─────────────────────────────────────────────
class VisualOdometry:
    """
    Estimates kart motion from consecutive camera frames.
    Uses Lucas-Kanade optical flow + Essential Matrix decomposition.

    scale_factor:
      Monocular VO has no metric scale — tune this by driving a known
      distance (e.g. 10m) and adjusting until map shows ~10m.
      Start with 0.05 and adjust up/down by 0.01 increments.

    max_heading_rate_deg:
      Maximum yaw rotation per frame (degrees). Clamped to realistic
      kart dynamics (default 5°/frame). Prevents wild heading flips.

    lateral_scale:
      Separate scale for lateral (camera right) motion.
      Set very low (0.01 – 0.02) because monocular VO cannot estimate
      lateral displacement reliably. Too high → Y drift off map.
    """

    def __init__(self, K: np.ndarray, scale_factor: float = 0.05,
                 max_heading_rate_deg: float = 1.0,
                 lateral_scale: float = 0.02,
                 feature_detector=None):
        from nevigation.feature_detectors import make_detector
        self.K     = K
        self.scale_factor      = scale_factor
        self.lateral_scale     = lateral_scale
        self.max_heading_rate  = math.radians(max_heading_rate_deg)
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.detector   = make_detector(feature_detector)   # ORB or GFTT
        self.pose       = Pose()
        self.trajectory = [(0.0, 0.0)]
        self._R_cum     = np.eye(3)            # cumulative rotation (camera→world)
        self._prev_gray = None
        self._prev_pts  = None
        self._frame_num = 0
        self._road_mask = None
        self.n_features = 0
        print(f"[VO] feature detector = {self.detector.name}  "
              f"max_heading_rate = {max_heading_rate_deg}°/frame")

    def reset(self, heading: float = None):
        """Reset VO tracking buffers. Preserves kart position and heading.

        Only the optical-flow state (_prev_gray, _prev_pts) is truly reset.
        X, Y, and heading are preserved — the kart didn't teleport or spin
        just because tracking was briefly lost.

        Args:
            heading: Override heading (radians). If None, keeps current heading.
        """
        h        = heading if heading is not None else self.pose.heading
        saved_x  = self.pose.x
        saved_y  = self.pose.y

        self.pose           = Pose()
        self.pose.x         = saved_x          # restore position
        self.pose.y         = saved_y
        self.pose.heading   = h                # restore / override heading

        # Reseed _R_cum consistent with heading h.
        # heading = raw_yaw + π/2  →  raw_yaw = h - π/2
        # Do NOT use np.eye(3) — identity encodes VO-neutral = 90°.
        _raw = h - math.pi / 2
        _c, _s = math.cos(_raw), math.sin(_raw)
        self._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])

        self.trajectory = [(saved_x, saved_y)]
        self._prev_gray = None
        self._prev_pts  = None
        print(f"[VO] reset — heading={math.degrees(h):.1f}°  pos=({saved_x:.2f},{saved_y:.2f})")

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        if self._road_mask is None or self._road_mask.shape != gray.shape:
            self._road_mask = self._build_road_mask(gray.shape)
        return self.detector.detect(gray, mask=self._road_mask)

    @staticmethod
    def _build_road_mask(shape: tuple, top_skip: float = 0.35,
                         side_skip: float = 0.10) -> np.ndarray:
        """Restrict feature detection to the road surface.

        Excludes the top `top_skip` fraction (sky/horizon) and the outer
         `side_skip` fraction on each side. Returns uint8 mask where 
        255 = look here, 0 = ignore.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        y0 = int(h * top_skip)
        x0 = int(w * side_skip)
        x1 = int(w * (1.0 - side_skip))
        mask[y0:, x0:x1] = 255
        return mask

    @staticmethod
    def _clamp_rotation_yaw_only(R: np.ndarray, max_yaw: float) -> np.ndarray:
        yaw = math.atan2(R[0, 2], R[2, 2])
        yaw_clamped = max(-max_yaw, min(yaw, max_yaw))
        c = math.cos(yaw_clamped)
        s = math.sin(yaw_clamped)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def update(self, frame: np.ndarray) -> Pose:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_num += 1

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_gray = gray
            self._prev_pts  = self._detect(gray)
            return self.pose

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self.lk_params)

        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if self._frame_num % 100 == 0:
            print(f"[VO] Frame {self._frame_num}: features = {len(good_prev)}")

        if len(good_prev) < 8:
            self._prev_pts  = self._detect(gray)
            self._prev_gray = gray
            return self.pose

        E, mask = cv2.findEssentialMat(
            good_curr, good_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            return self.pose

        _, R_inc, t, _ = cv2.recoverPose(E, good_curr, good_prev, self.K, mask=mask)

        # Suppress heading change when forward motion is weak.
        # recoverPose always returns unit-norm t; |t[2]|<0.3 means the kart
        # is mostly stopped or moving sideways → heading measurement is noise.
        if abs(float(t[2])) < 0.3:
            R_inc = np.eye(3)

        # 1. Clamp incremental rotation
        R_inc = self._clamp_rotation_yaw_only(R_inc, self.max_heading_rate)

        # 2. Update heading
        self._R_cum = R_inc @ self._R_cum
        raw_yaw = np.arctan2(self._R_cum[0, 2], self._R_cum[2, 2])
        #self.pose.heading = (raw_yaw + math.pi) % (2 * math.pi)  # map forward = -X
        #self.pose.heading = (raw_yaw + math.pi / 2) % (2 * math.pi)  # map forward = +Y
        raw = raw_yaw + math.pi / 2
        self.pose.heading = math.atan2(math.sin(raw), math.cos(raw))  # normalise to [-π, π]
        # then shift to [0, 2π]
        if self.pose.heading < 0:
            self.pose.heading += 2 * math.pi

        self.pose.R = self._R_cum.copy()

        # 3. Local translation with separate lateral scale
        step_forward = float(t[2]) * self.scale_factor   # camera forward
        step_lateral = float(t[0]) * self.lateral_scale  # camera right (small)

        # 4. Project into world coordinates
        theta = self.pose.heading
        self.pose.x += (step_forward * math.cos(theta) - step_lateral * math.sin(theta))
        self.pose.y += (step_forward * math.sin(theta) + step_lateral * math.cos(theta))

        self.trajectory.append((self.pose.x, self.pose.y))

        # Refresh features
        self._prev_pts  = good_curr.reshape(-1, 1, 2) if len(good_curr) > 150 \
                          else self._detect(gray)
        self._prev_gray = gray
        self.n_features = len(good_curr)   # exposed for pipeline diagnostics
        return self.pose

    def speed_kmh(self, prev_pose: Pose, dt: float) -> float:
        if dt <= 0:
            return 0.0
        dx = self.pose.x - prev_pose.x
        dy = self.pose.y - prev_pose.y
        return round((math.sqrt(dx*dx + dy*dy) / dt) * 3.6, 1)

# ─────────────────────────────────────────────
# MODULE 3 — OBJECT PROJECTOR
# ─────────────────────────────────────────────

class ObjectProjector:
    """
    Projects detected objects into world (x, y) coordinates.

    Primary method: Ground Plane Assumption
      - Works in full sunlight, no LiDAR needed
      - Assumes object base touches the ground
      - Formula: depth = (cam_height * fy) / (pixel_y_bottom - cy)
      - Accurate to ±0.3m for objects at 1-8m range

    Secondary method: LiDAR angle matching (when available)
      - Overrides ground plane depth if LiDAR confidence is high
    """

    def __init__(self, cam: CameraParams, camera_height_m: float = 0.50):
        self.cam    = cam
        self.height = camera_height_m
        self.fx     = cam.fx
        self.fy     = cam.fy
        self.cx     = cam.cx
        self.cy     = cam.cy

    def ground_plane_depth(self, bbox: tuple) -> float:
        """Estimate depth from bottom edge of bounding box using ground plane."""
        x1, y1, x2, y2 = bbox
        pixel_y_bottom  = y2    # bottom of bbox = object base on ground
        dy = pixel_y_bottom - self.cy
        if abs(dy) < 1.0:
            return 8.0          # too close to horizon — cap at 8m
        depth = (self.height * self.fy) / abs(dy)
        return float(np.clip(depth, 0.3, 15.0))

    def project(self, obj: DetectedObject, pose: Pose) -> DetectedObject:
        """Back-project object from image to world coordinates."""
        x1, y1, x2, y2 = obj.bbox
        bx = (x1 + x2) / 2.0    # bbox centre x

        depth = obj.depth if (obj.depth and obj.depth > 0) \
                else self.ground_plane_depth(obj.bbox)
        obj.depth = depth

        # Back-project to camera frame
        cam_x = (bx - self.cx) * depth / self.fx
        cam_z = depth

        cam_pt   = np.array([[cam_x], [0.0], [cam_z]])
        world_pt = pose.R.T @ cam_pt - pose.R.T @ pose.t

        obj.world_x = float(world_pt[0]) + pose.x
        obj.world_y = float(world_pt[2]) + pose.y
        return obj

# ─────────────────────────────────────────────
# MODULE 4 — HOMOGRAPHY HELPER
# ─────────────────────────────────────────────

class HomographyHelper:
    """
    Interactive tool — saves clicked pixel coordinates from track image.
    Run once: python3 av_map.py --calibrate-homography

    Saves results into nav_config.json so they load automatically next time.

    WSLg note: since imshow is unreliable, this saves the track image with
    numbered markers to Output/homography_setup.jpg for you to click coordinates
    using any image viewer, then enter them manually in nav_config.json.
    """

    def __init__(self, track_image_path: str, config_path: str = CONFIG_PATH):
        self.track_path  = track_image_path
        self.config_path = config_path

    def run(self):
        if not os.path.exists(self.track_path):
            print(f"Track image not found: {self.track_path}")
            return

        img = cv2.imread(self.track_path)
        h, w = img.shape[:2]

        print("\n" + "="*55)
        print("  HOMOGRAPHY CALIBRATION HELPER")
        print("="*55)
        print(f"\nTrack image size: {w} x {h} pixels")
        print("\nStep 1: Measure 4 landmarks on your REAL track (in meters)")
        print("  Example: corner cones, start line markings")
        print("  Use kart start position as origin (0, 0)\n")

        world_pts = []
        image_pts = []

        for i in range(4):
            print(f"--- Landmark {i+1} of 4 ---")
            wx = float(input(f"  Real-world X (meters): "))
            wy = float(input(f"  Real-world Y (meters): "))
            world_pts.append([wx, wy])

            print(f"  Now open your track image and find the pixel coordinates")
            print(f"  of this same landmark.")
            print(f"  Track image saved at: {OUTPUT_DIR}/homography_setup.jpg")
            print(f"  Open in Windows Explorer: \\\\wsl$\\Ubuntu{OUTPUT_DIR}\\homography_setup.jpg")

            # Draw marker on image for reference
            marker_img = img.copy()
            for j, (px_done) in enumerate(image_pts):
                cv2.circle(marker_img, tuple(map(int, px_done)), 8, (0, 255, 0), -1)
                cv2.putText(marker_img, str(j+1), (int(px_done[0])+10, int(px_done[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Draw crosshair guide for current point
            cv2.putText(marker_img, f"Click point {i+1}: world({wx},{wy})m",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "homography_setup.jpg"), marker_img)

            pu = float(input(f"  Pixel U (x, horizontal): "))
            pv = float(input(f"  Pixel V (y, vertical):   "))
            image_pts.append([pu, pv])
            print(f"  ✓ Landmark {i+1}: world=({wx},{wy})m  pixel=({pu},{pv})\n")

        # Verify homography makes sense
        wp = np.float32(world_pts)
        ip = np.float32(image_pts)
        H, _ = cv2.findHomography(wp, ip)
        print(f"Homography matrix computed:")
        print(H)

        # Save to nav_config.json
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        cfg["world_pts"] = world_pts
        cfg["image_pts"] = image_pts
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n✓ Saved to {self.config_path}")
        print("  Run av_map.py normally — homography loads automatically.\n")

# ─────────────────────────────────────────────
# MODULE 5 — TRACK MAPPER
# ─────────────────────────────────────────────

class TrackMapper:
    """
    Renders kart position, trajectory, speed, and detected objects
    onto the track image.

    If homography is calibrated → accurate world→pixel mapping.
    If not → centred scale transform (still useful for relative motion).
    """

    def __init__(self, cfg: NavConfig, map_size: tuple = (800, 800)):
        self.map_w, self.map_h = map_size
        self.world_range = cfg.map_world_range_m
        self.H: Optional[np.ndarray] = None

        # ── Load track image (read once; derive scale factors for everything) ──
        orig_img_w, orig_img_h = self.map_w, self.map_h   # fallback if no image
        if cfg.track_image_path and os.path.exists(cfg.track_image_path):
            _img = cv2.imread(cfg.track_image_path)
            orig_img_h, orig_img_w = _img.shape[:2]
            self.track_bg = cv2.resize(_img, (self.map_w, self.map_h))
            print(f"[TrackMapper] Loaded track image: {cfg.track_image_path}")
        else:
            print(f"[TrackMapper] Track image not found at '{cfg.track_image_path}' — using grid map.")
            print(f"  Set 'track_image_path' in nav_config.json when ready.")
            self.track_bg = self._make_grid()

        # Scale factors: full-res image coords → display map coords
        sx = self.map_w / orig_img_w
        sy = self.map_h / orig_img_h

        # ── World-origin pixel ────────────────────────────────────────────────
        # Use cfg.start_px (the pixel on Track.png that corresponds to world (0,0)).
        # Scale it to the display map. Fall back to map centre if not set.
        if cfg.start_px and (cfg.start_px[0] != 0 or cfg.start_px[1] != 0):
            self.origin_x = int(cfg.start_px[0] * sx)
            self.origin_y = int(cfg.start_px[1] * sy)
            self.scale    = cfg.px_per_meter * sx   # px/m at display resolution
            print(f"[TrackMapper] origin=({self.origin_x},{self.origin_y})px  "
                  f"scale={self.scale:.2f}px/m  (cfg.start_px={cfg.start_px})")
        else:
            self.scale    = min(self.map_w, self.map_h) / self.world_range
            self.origin_x = self.map_w // 2
            self.origin_y = self.map_h // 2
            print(f"[TrackMapper] origin=map-centre ({self.origin_x},{self.origin_y})px  "
                  f"(no start_px in config — fallback)")

        # ── Homography (image_pts are in full-res coords; scale them) ─────────
        if cfg.world_pts and cfg.image_pts:
            wp = np.float32(cfg.world_pts)
            ip = np.float32(cfg.image_pts)
            ip[:, 0] *= sx
            ip[:, 1] *= sy
            self.H, _ = cv2.findHomography(wp, ip)
            print("[TrackMapper] Homography loaded from config.")
        else:
            print("[TrackMapper] No homography — using scaled pixel-origin fallback.")

    def world_to_pixel(self, wx: float, wy: float) -> tuple:
        if self.H is not None:
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            px = cv2.perspectiveTransform(pt, self.H)
            return int(px[0, 0, 0]), int(px[0, 0, 1])
        # Fallback: centred scale
        u = int(self.origin_x + wx * self.scale)
        v = int(self.origin_y - wy * self.scale)
        return u, v

    def render(self, pose: Pose, trajectory: list,
               objects: list, speed_kmh: float = 0.0) -> np.ndarray:

        frame = self.track_bg.copy()

        # Trajectory trail — navy blue
        if len(trajectory) > 1:
            pts = [self.world_to_pixel(x, y) for x, y in trajectory]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (128, 0, 0), 3)

        # Detected objects
        for obj in objects:
            px, py = self.world_to_pixel(obj.world_x, obj.world_y)
            if not (0 <= px < self.map_w and 0 <= py < self.map_h):
                continue
            color = PALETTE[obj.class_id % len(PALETTE)]
            cv2.circle(frame, (px, py), 9, color, -1)
            dist_txt = f"{obj.depth:.1f}m" if obj.depth else ""
            cv2.putText(frame, f"{obj.label} {dist_txt}",
                        (px + 11, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Kart icon — filled circle + heading arrow, deep red
        kx, ky = self.world_to_pixel(pose.x, pose.y)
        if 0 <= kx < self.map_w and 0 <= py < self.map_h:
            ex = int(kx + 22 * np.sin(pose.heading))
            ey = int(ky - 22 * np.cos(pose.heading))
            cv2.circle(frame, (kx, ky), 11, (0, 0, 139), -1)
            cv2.circle(frame, (kx, ky), 11, (0, 0, 80), 2)
            cv2.arrowedLine(frame, (kx, ky), (ex, ey),
                            (0, 0, 139), 3, tipLength=0.4)

        # HUD
        hud = [
            f"Pos : ({pose.x:.2f}, {pose.y:.2f}) m",
            f"Hdg : {np.degrees(pose.heading):.1f} deg",
            f"Spd : {speed_kmh:.1f} km/h",
            f"Obj : {len(objects)}",
        ]
        cv2.rectangle(frame, (0, 0), (270, 16 + len(hud)*26), (0, 0, 0), -1)
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (8, 22 + i*26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _make_grid(self) -> np.ndarray:
        img = np.full((self.map_h, self.map_w, 3), 28, dtype=np.uint8)
        sp  = max(1, int(self.scale))
        for x in range(0, self.map_w, sp):
            cv2.line(img, (x, 0), (x, self.map_h), (50, 50, 50), 1)
        for y in range(0, self.map_h, sp):
            cv2.line(img, (0, y), (self.map_w, y), (50, 50, 50), 1)
        cv2.line(img, (self.origin_x, 0), (self.origin_x, self.map_h), (70,70,70), 2)
        cv2.line(img, (0, self.origin_y), (self.map_w, self.origin_y),  (70,70,70), 2)
        return img

# ─────────────────────────────────────────────
# MODULE 6 — MAP SYSTEM (Main Loop)
# ─────────────────────────────────────────────

class MapSystem:
    """
    Main loop — supports VIDEO and LIVE modes.

    VIDEO mode: feed a recorded .mp4 for algorithm tuning
    LIVE mode:  use real camera (index 0, CAP_V4L2)

    WSLg display: saves nav_preview.jpg every second.
    Open in Windows Explorer and press F5 to refresh.

    Terminal commands (press ENTER after each):
      q → quit
      r → reset VO pose
      s → save current map frame
    """

    def __init__(self, mode: str = "video",
                 video_path: Optional[str] = None):

        self.mode = mode

        # Load configs
        self.calib  = CameraCalibration(CALIB_PATH)
        self.cfg    = load_nav_config(CONFIG_PATH)
        self.cfg.print_summary()

        # Init modules
        self.vo        = VisualOdometry(self.calib.K, self.cfg.scale_factor)
        self.projector = ObjectProjector(self.calib.cam, self.cfg.camera_height_m)
        self.mapper    = TrackMapper(self.cfg)

        # Camera / video source
        if mode == "video":
            path = video_path or self.cfg.__dict__.get("video_path", None)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"Video not found: {path}\n"
                    f"Pass --video /path/to/video.mp4")
            self.cap = cv2.VideoCapture(path)
            print(f"[MapSystem] VIDEO mode — {path}")
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[MapSystem] LIVE mode — camera index 0")

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video/camera source")

        # Shared terminal command state
        self._cmd     = None
        self._cmd_lock = threading.Lock()
        self._running  = True

        # Start terminal input thread
        threading.Thread(target=self._input_thread, daemon=True).start()

        print(f"\nPreview saves to: {PREVIEW_FILE}")
        print(f"Open in Windows Explorer: \\\\wsl$\\Ubuntu{PREVIEW_FILE}")
        print(f"\nTerminal commands: q=quit  r=reset  s=save\n")

    def _input_thread(self):
        while self._running:
            try:
                cmd = input().strip().lower()
                with self._cmd_lock:
                    self._cmd = cmd
            except EOFError:
                break

    def get_detections_from_teammates(self, frame: np.ndarray) -> list:
        """
        ── REPLACE THIS with real detection input ──
        Currently returns empty list — plug in your av_system.py detections here.
        """
        return []

    def run(self):
        fps_t     = time.time()
        fps_count = 0
        fps       = 0.0
        prev_pose = Pose()
        prev_t    = time.time()
        speed     = 0.0
        frame_num = 0
        last_save = 0.0

        # Warmup for live mode
        if self.mode == "live":
            for _ in range(10):
                self.cap.grab()
            time.sleep(0.5)

        try:
            while True:
                if self.mode == "live":
                    self.cap.grab()
                ret, frame = self.cap.read()

                if not ret:
                    if self.mode == "video":
                        print("\n[MapSystem] Video ended.")
                    break

                frame_num += 1
                now = time.time()

                frame_ud = self.calib.undistort(frame)
                pose = self.vo.update(frame_ud)

                dt    = now - prev_t
                speed = self.vo.speed_kmh(prev_pose, dt) if dt > 0 else speed
                prev_pose = Pose(x=pose.x, y=pose.y, heading=pose.heading)
                prev_t    = now

                raw_dets  = self.get_detections_from_teammates(frame_ud)
                world_obj = [self.projector.project(obj, pose) for obj in raw_dets]

                map_frame = self.mapper.render(
                    pose, self.vo.trajectory, world_obj, speed)

                fps_count += 1
                if now - fps_t >= 1.0:
                    fps   = fps_count / (now - fps_t)
                    fps_t, fps_count = now, 0
                cv2.putText(frame_ud, f"FPS:{fps:.1f}  SPD:{speed:.1f}km/h",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                if now - last_save >= 1.0:
                    cam_small = cv2.resize(frame_ud, (640, 360))
                    map_small = cv2.resize(map_frame, (640, 640))
                    pad = np.zeros((640 - 360, 640, 3), dtype=np.uint8)
                    combined = np.vstack([cam_small, pad, map_small])
                    cv2.imwrite(PREVIEW_FILE, combined)
                    last_save = now

                with self._cmd_lock:
                    cmd = self._cmd
                    self._cmd = None

                if cmd == "q":
                    print("[MapSystem] Quit command received.")
                    break
                elif cmd == "r":
                    self.vo.reset()
                elif cmd == "s":
                    fname = os.path.join(OUTPUT_DIR,
                                         f"map_{datetime.now().strftime('%H%M%S')}.png")
                    cv2.imwrite(fname, map_frame)
                    print(f"[MapSystem] Saved {fname}")

                if self.mode == "video":
                    time.sleep(0.01)

        finally:
            self._running = False
            self.cap.release()
            if self.vo.trajectory:
                final_map = self.mapper.render(
                    self.vo.pose, self.vo.trajectory, [], speed)
                fname = os.path.join(OUTPUT_DIR, "map_final.png")
                cv2.imwrite(fname, final_map)
                print(f"[MapSystem] Final map saved → {fname}")
            print("[MapSystem] Stopped.")

# ─────────────────────────────────────────────
# NAVIGATION PIPELINE — main integration class
# ─────────────────────────────────────────────

class NavigationPipeline:
    """
    Top-level class called by test_map_video.py.
    Wires together: VO (kart position) + perception.py (object projection)
    + TrackMapper (rendering) + check point matching.
    """

    OBJ_COLORS = {
        "stop":                (0,   0,   255),
        "person":              (255, 128,   0),
        "kart":                (0,   255, 255),
        "speed_20":            (0,   165, 255),
        "speed_30":            (0,   165, 255),
        "traffic_red_light":   (0,   0,   200),
        "traffic_green_light": (0,   200,   0),
        "turn_left":           (255,   0, 255),
    }
    DEFAULT_COLOR = (200, 200, 200)

    def __init__(self, mode: str = "video",
                 video_path: Optional[str] = None,
                 check_points: list = None,
                 frame_callback=None,
                 feature_detector=None,
                 config_path: Optional[str] = None,
                 calib_path: Optional[str] = None,
                 start_pixel: Optional[list] = None,
                 initial_heading_rad: Optional[float] = None,
                 start_frame: Optional[int] = None,
                 end_frame: Optional[int] = None):

        from nevigation.perception import GroundPlaneProjector, ObjectTracker
        from nevigation.av_system  import ObjectDetector, MODEL_XML, CLASSES_PATH

        # Allow per-run overrides; default to the module-level constants so
        # existing callers are unaffected.
        _cfg_path   = config_path or CONFIG_PATH
        _calib_path = calib_path  or CALIB_PATH

        self.mode         = mode
        self.check_points = check_points or []
        self.frame_callback = frame_callback
        self.start_frame  = start_frame
        self.end_frame    = end_frame

        # Public state attributes – updated every frame for the callback
        self.frame_count      = 0
        self.kart_x           = 0.0
        self.kart_y           = 0.0
        self.kart_heading     = 0.0   # degrees
        self.stopped          = False
        self.detected_objects = []
        self.snap_events      = []    # list of dicts: {frame, name, world_x, world_y}
        self.start_time            = time.time()
        # VO health — updated each frame after vo.update()
        self.vo_last_features      = 0
        self.vo_last_dx_m          = 0.0
        self.vo_last_dy_m          = 0.0
        self.vo_last_dheading_deg  = 0.0
        # Snap event — set by snap handler each frame (None if no snap occurred)
        self.last_snap             = None

        print("\n" + "="*60)
        print("  Navigation Pipeline — starting")
        print("="*60)

        self.calib = CameraCalibration(_calib_path)
        self.cfg_raw = load_nav_config_merged(_cfg_path)
        # caller/CLI overrides (highest priority)
        if start_pixel is not None:
            self.cfg_raw.setdefault("track_info", {})["start_pixel"] = list(start_pixel)
        if initial_heading_rad is not None:
            self.cfg_raw["initial_heading_rad"] = float(initial_heading_rad)
        self.cfg = NavConfig(self.cfg_raw)
        _src = self.cfg_raw.get("_per_video_source")
        if _src:
            print(f"[Config] per-video override: {_src}")

        self.px_per_m  = self.cfg_raw["px_per_meter"]
        self.start_px  = self.cfg_raw["start_px"]
        self.landmarks = self.cfg_raw.get("fixed_landmarks", [])

        # ── Convert check_point track pixels → world coords ───────────
        for cp in self.check_points:
            px, py = cp["track_pixel"]
            wx = (px - self.start_px[0]) / self.px_per_m
            wy = -(py - self.start_px[1]) / self.px_per_m
            cp["world"]     = (round(wx, 3), round(wy, 3))
            cp["hit"]       = False
            cp["closest_m"] = float("inf")
        print(f"\nCheck points ({len(self.check_points)}):")
        for cp in self.check_points:
            print(f"  {cp['name']:<16} track_px={cp['track_pixel']}  "
                  f"world={cp['world']}m  threshold={cp['threshold_m']}m")

        # ── Open video / camera ───────────────────────────────────────
        if mode == "video":
            video_path = video_path or self.cfg_raw.get("video_path")
            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            self.cap     = cv2.VideoCapture(video_path)
            self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.fps_src = 30.0
            for _ in range(10): self.cap.grab()

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video/camera")

        # ── K matrix scaling to match actual video resolution ────────
        video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        baseline_w, baseline_h = self.calib.cam.resolution

        scale_x = video_w / baseline_w
        scale_y = video_h / baseline_h

        self.calib.K[0, 0] *= scale_x   # fx
        self.calib.K[1, 1] *= scale_y   # fy
        self.calib.K[0, 2] *= scale_x   # cx
        self.calib.K[1, 2] *= scale_y   # cy

        self.calib.cam.fx = self.calib.K[0, 0]
        self.calib.cam.fy = self.calib.K[1, 1]
        self.calib.cam.cx = self.calib.K[0, 2]
        self.calib.cam.cy = self.calib.K[1, 2]

        print(f"[Pipeline] Video: {video_w}x{video_h} | "
              f"Baseline: {baseline_w}x{baseline_h} | "
              f"Scaling: x={scale_x:.3f} y={scale_y:.3f}")

        # ── Init VO ──────────────────────────────────────────────────
        self.vo = VisualOdometry(self.calib.K,
                                 scale_factor=self.cfg.scale_factor,
                                 feature_detector=feature_detector)

        # Seed VO start pose from nav_config track_info.start_pixel + world origin
        _sp  = self.cfg_raw["track_info"]["start_pixel"]   # e.g. [1320, 915]
        _ox  = self.start_px                                # e.g. [6795, 270]
        if "initial_heading_rad" not in self.cfg_raw:
            raise KeyError(
                "initial_heading_rad missing from config — "
                "set to 0.0 (+X, video) or 3.14159265 (-X, real-time)")
        _h0  = float(self.cfg_raw["initial_heading_rad"])

        start_wx =  (_sp[0] - _ox[0]) / self.px_per_m
        start_wy = -(_sp[1] - _ox[1]) / self.px_per_m

        self.vo.pose.x       = start_wx
        self.vo.pose.y       = start_wy
        self.vo.pose.heading = _h0
        self.vo.trajectory   = [(start_wx, start_wy)]

        # Seed _R_cum so VO heading is consistent with _h0.
        # VO computes heading = raw_yaw + π/2, so raw_yaw = _h0 - π/2.
        _raw_yaw = _h0 - math.pi / 2
        _c, _s = math.cos(_raw_yaw), math.sin(_raw_yaw)
        self.vo._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])

        print(f"[Pipeline] VO seeded: pos=({start_wx:.3f}, {start_wy:.3f})m  "
              f"heading={math.degrees(_h0):.1f}°")

        # ── Init perception ──────────────────────────────────────────
        self.projector = GroundPlaneProjector(
            fx=self.calib.cam.fx, fy=self.calib.cam.fy,
            cx=self.calib.cam.cx, cy=self.calib.cam.cy,
            camera_height_m=self.cfg.camera_height_m   # from nav_config (measured)
        )
        self.tracker = ObjectTracker(self.projector)

        print("\nLoading detector...")
        self.detector = ObjectDetector(MODEL_XML, CLASSES_PATH)

        # ── Init mapper (full track image resolution) ─────────────────
        track_img = cv2.imread(self.cfg.track_image_path)
        MAP_H, MAP_W = track_img.shape[:2]
        self.mapper   = TrackMapper(self.cfg, map_size=(MAP_W, MAP_H))
        self.MAP_W    = MAP_W
        self.MAP_H    = MAP_H

        self._landmark_base = self._draw_landmarks(track_img.copy())

        # ── Landmark-based pose corrector ─────────────────────────────
        # Snaps VO (x,y) to the nearest known zebra/start-line when one is
        # detected ahead of the kart. Heading correction stays disabled until
        # a validated BEV homography is added to nav_config (key "H_bev").
        H_bev = None
        H_raw = self.cfg_raw.get("H_bev")
        if H_raw is not None:
            try:
                H_bev = np.array(H_raw, dtype=np.float32).reshape(3, 3)
                print("[Localizer] BEV homography loaded — heading correction ENABLED")
            except Exception as e:
                print(f"[Localizer] H_bev in config is malformed, ignoring: {e}")
        self.localizer = LandmarkLocalizer(
            landmarks=self.landmarks,
            bev_homography=H_bev,
            px_per_meter_bev=self.cfg_raw.get("px_per_meter_bev"),
        )

        # Terminal input thread
        self._cmd     = None
        self._running = True
        import threading as _t
        _t.Thread(target=self._input_thread, daemon=True).start()

        print(f"\nPreview  → {PREVIEW_FILE}")
        print(f"Commands : q=quit  r=reset  s=save\n")

    # ── Track-map scale helpers ───────────────────────────────────────────
    # Verified against zebra_3 pixel_box (123x178 px ≈ 400x592 cm) → 30 px/m.
    # Track.png is a top-down map at a uniform scale, so 1 px = (1/px_per_m) m.
    def track_px_to_m(self, dist_px: float) -> float:
        """Convert a distance in track-map pixels to metres."""
        return dist_px / self.px_per_m

    def track_m_to_px(self, dist_m: float) -> float:
        """Convert a distance in metres to track-map pixels."""
        return dist_m * self.px_per_m

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        import time as _t
        prev_x, prev_y = self.vo.pose.x, self.vo.pose.y
        prev_t   = _t.time()
        speed    = 0.0
        last_save= 0.0
        frame_num= 0
        fps_t    = _t.time()
        fps_cnt  = 0
        fps      = 0.0

        try:
            while True:
                if self.mode == "live": self.cap.grab()
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video.")
                    break

                frame_num += 1

                # Skip frames before start_frame
                if self.start_frame is not None and frame_num < self.start_frame:
                    continue

                # Stop after end_frame
                if self.end_frame is not None and frame_num > self.end_frame:
                    print(f"\nReached end frame {self.end_frame}.")
                    break

                now = _t.time()
                frame_ud = self.calib.undistort(frame)

                # Check stationary FIRST so is_kart_stopped() reflects the current frame
                self.tracker._check_stationary(frame_ud)

                # Update VO — freeze heading if stopped to prevent noise accumulation
                if not self.tracker.is_kart_stopped():
                    pose = self.vo.update(frame_ud)
                else:
                    _saved_x, _saved_y = self.vo.pose.x, self.vo.pose.y
                    _saved_heading = float(pose.heading)  # Save as scalar BEFORE vo.update() aliases it
                    _ = self.vo.update(frame_ud)          # run to refresh features for next movement
                    # Restore BOTH heading and position — VO ran but kart didn't actually move
                    self.vo.pose.heading = _saved_heading  # Now actually restores pre-VO heading
                    self.vo.pose.x       = _saved_x
                    self.vo.pose.y       = _saved_y
                    # Re-seed _R_cum to stay consistent with the frozen heading.
                    # Without this, _R_cum diverges from pose.heading during stopped frames,
                    # causing a heading jump when movement resumes.
                    _h = _saved_heading                    # Seeds _R_cum from true saved heading
                    _raw = _h - math.pi / 2
                    _c, _s = math.cos(_raw), math.sin(_raw)
                    self.vo._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
                    pose = self.vo.pose

                # Capture VO health for logger
                _prev_heading = getattr(self, '_prev_heading', self.vo.pose.heading)
                self.vo_last_features     = self.vo.n_features
                self.vo_last_dx_m         = self.vo.pose.x - prev_x
                self.vo_last_dy_m         = self.vo.pose.y - prev_y
                self.vo_last_dheading_deg = math.degrees(self.vo.pose.heading - _prev_heading)
                self._prev_heading        = self.vo.pose.heading

                # ── Landmark snap (drift correction) ─────────────────
                # If we see a known zebra/start-line and VO says we're close
                # to it, overwrite VO (x,y) with the landmark's world_center.
                _MAX_SNAP_M = 3.0
                self.last_snap = None   # reset each frame
                pre_x, pre_y = self.vo.pose.x, self.vo.pose.y
                pre_hdg = math.degrees(self.vo.pose.heading)

                snap = self.localizer.process(frame_ud, self.vo.pose)
                if snap is not None:
                    snap_dist = math.hypot(snap.world_x - pre_x, snap.world_y - pre_y)
                    if snap_dist > _MAX_SNAP_M:
                        print(f"[Snap REJECTED] {snap.landmark_name}  "
                              f"distance={snap_dist:.1f}m > {_MAX_SNAP_M}m threshold")
                        self.last_snap = {
                            "name": snap.landmark_name,
                            "type": getattr(snap, "landmark_type", ""),
                            "dist_m": snap_dist, "conf": snap.confidence,
                            "pre_x": pre_x, "pre_y": pre_y,
                            "post_x": pre_x, "post_y": pre_y,
                            "pre_hdg": pre_hdg, "post_hdg": pre_hdg, "dhdg": 0.0,
                            "accepted": False,
                            "reject_reason": f"dist {snap_dist:.1f}m > {_MAX_SNAP_M}m",
                        }
                    else:
                        post_hdg = pre_hdg
                        print(f"[Snap] {snap.landmark_name}  "
                              f"VO=({pre_x:.2f},{pre_y:.2f}) -> "
                              f"({snap.world_x:.2f},{snap.world_y:.2f})  "
                              f"dist={snap_dist:.2f}m  conf={snap.confidence:.2f}")
                        self.vo.pose.x = snap.world_x
                        self.vo.pose.y = snap.world_y
                        if snap.heading_rad is not None:
                            self.vo.pose.heading = snap.heading_rad
                            post_hdg = math.degrees(snap.heading_rad)
                        # Reseed _R_cum from the current (preserved) heading so future frames
                        # continue in the correct direction.
                        # DO NOT use np.eye(3) — identity encodes heading=90° (VO neutral),
                        # corrupting the heading on the very next frame.
                        _h = self.vo.pose.heading
                        _raw = _h - math.pi / 2
                        _c, _s = math.cos(_raw), math.sin(_raw)
                        self.vo._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
                        # Re-anchor trajectory so the corrected jump doesn't
                        # render as a wild line on the map.
                        self.vo.trajectory.append((snap.world_x, snap.world_y))
                        pose = self.vo.pose
                        self.last_snap = {
                            "name": snap.landmark_name,
                            "type": getattr(snap, "landmark_type", ""),
                            "dist_m": snap_dist, "conf": snap.confidence,
                            "pre_x": pre_x, "pre_y": pre_y,
                            "post_x": snap.world_x, "post_y": snap.world_y,
                            "pre_hdg": pre_hdg, "post_hdg": post_hdg,
                            "dhdg": post_hdg - pre_hdg,
                            "accepted": True, "reject_reason": "",
                        }
                        self.snap_events.append({
                            "frame":   frame_num,
                            "name":    snap.landmark_name,
                            "world_x": snap.world_x,
                            "world_y": snap.world_y,
                        })

                # Proximity snap — triggers on position alone (no visual needed)
                prox_snap = self.localizer.check_proximity_snaps(self.vo.pose)
                if prox_snap is not None:
                    self.vo.pose.x = prox_snap.world_x
                    self.vo.pose.y = prox_snap.world_y
                    if prox_snap.heading_rad is not None:
                        self.vo.pose.heading = prox_snap.heading_rad
                    _h = self.vo.pose.heading
                    _raw = _h - math.pi / 2
                    _c, _s = math.cos(_raw), math.sin(_raw)
                    self.vo._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
                    self.vo.trajectory.append((prox_snap.world_x, prox_snap.world_y))
                    pose = self.vo.pose
                    self.snap_events.append({
                        "frame":   frame_num,
                        "name":    prox_snap.landmark_name,
                        "world_x": prox_snap.world_x,
                        "world_y": prox_snap.world_y,
                    })

                # Speed
                dt    = max(now - prev_t, 1e-6)
                dx, dy = pose.x-prev_x, pose.y-prev_y
                speed  = round(math.sqrt(dx*dx+dy*dy)/dt*3.6, 1)
                prev_x, prev_y, prev_t = pose.x, pose.y, now

                # Debug: world → pixel every 30 frames
                if frame_num % 30 == 0:
                    px, py = self.mapper.world_to_pixel(pose.x, pose.y)
                    print(f"[DEBUG] Frame {frame_num}: world=({pose.x:.2f}, {pose.y:.2f}) -> pixel=({px}, {py})  map_size=({self.MAP_W},{self.MAP_H})")

                # ── Detect objects ────────────────────────────────────
                raw_dets = self.detector.detect(frame_ud)
                frame_display = frame_ud.copy()                 # Separate display buffer
                self.detector.draw(frame_display, raw_dets)     # Only draws on the copy

                # ── Track + project objects (perception.py) ───────────
                visible_objs = self.tracker.update(
                    raw_dets, pose.x, pose.y, pose.heading, frame_display)

                # ── Check point matching ──────────────────────────────
                self._check_points(pose, frame_num)

                # ── Update public state ──────────────────────────────
                self.frame_count      = frame_num
                self.kart_x           = pose.x
                self.kart_y           = pose.y
                self.kart_heading     = math.degrees(pose.heading)
                self.stopped          = self.tracker.is_kart_stopped()
                self.detected_objects = visible_objs

                if self.frame_callback:
                    self.frame_callback(self)

                # FPS
                fps_cnt += 1
                if now - fps_t >= 1.0:
                    fps = fps_cnt/(now-fps_t)
                    fps_t, fps_cnt = now, 0

                cv2.putText(frame_display,
                    f"FPS:{fps:.0f} SPD:{speed:.1f}km/h "
                    f"Pos:({pose.x:.1f},{pose.y:.1f})m "
                    f"{'STOPPED' if self.tracker.is_kart_stopped() else 'MOVING'}",
                    (10,28), cv2.FONT_HERSHEY_DUPLEX,
                    0.65, (0, 0, 0), 2)

                # ── Render track map ──────────────────────────────────
                if now - last_save >= 1.0:
                    map_frame = self._render_map(pose, visible_objs, speed)
                    cam_s = cv2.resize(frame_display, (1280, 720))
                    map_s = cv2.resize(map_frame, (1280, int(self.MAP_H * 1280/self.MAP_W)))
                    cv2.imwrite(PREVIEW_FILE, np.vstack([cam_s, map_s]))
                    last_save = now

                # Commands
                cmd = self._cmd; self._cmd = None
                if cmd == "q": break
                elif cmd == "r":
                    self.vo.reset()
                    print("VO reset.")
                elif cmd == "s":
                    ts = datetime.now().strftime("%H%M%S")
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"snap_{ts}.png"),
                                self._render_map(pose, visible_objs, speed))
                    print(f"Saved snap_{ts}.png")

                if self.mode == "video":
                    _t.sleep(1.0/self.fps_src * 0.5)

        finally:
            self._running = False
            self.cap.release()
            self._print_checkpoint_summary()
            final = self._render_map(self.vo.pose, [], 0.0)
            cv2.imwrite(os.path.join(OUTPUT_DIR,"map_final.png"), final)
            print(f"\nFinal map → {os.path.join(OUTPUT_DIR,'map_final.png')}")

    # ── Rendering ─────────────────────────────────────────────────────────
    def _render_map(self, pose, visible_objs, speed):
        frame = self._landmark_base.copy()

        # Trajectory — navy blue, full path
        if len(self.vo.trajectory) > 1:
            pts = [self.mapper.world_to_pixel(x, y) for x, y in self.vo.trajectory]
            for i in range(1, len(pts)):
                if (0 <= pts[i-1][0] < self.MAP_W and 0 <= pts[i-1][1] < self.MAP_H
                        and 0 <= pts[i][0] < self.MAP_W and 0 <= pts[i][1] < self.MAP_H):
                    cv2.line(frame, pts[i-1], pts[i], (128, 0, 0), 3)

        # Snap correction markers (yellow stars)
        for ev in self.snap_events:
            ex, ey = self.mapper.world_to_pixel(ev["world_x"], ev["world_y"])
            if 0 <= ex < self.MAP_W and 0 <= ey < self.MAP_H:
                cv2.drawMarker(frame, (ex, ey), (0, 255, 255),
                               cv2.MARKER_STAR, 22, 3)
                cv2.putText(frame, f"snap:{ev['name']}", (ex + 14, ey + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Detected objects
        for obj in visible_objs:
            px, py = self.mapper.world_to_pixel(obj.world_x, obj.world_y)
            if 0 <= px < self.MAP_W and 0 <= py < self.MAP_H:
                col = self.OBJ_COLORS.get(obj.label, self.DEFAULT_COLOR)
                cv2.circle(frame, (px, py), 10, col, -1)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 1)
                cv2.putText(frame, obj.label, (px+12, py+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Check point markers
        for cp in self.check_points:
            px, py = cp["track_pixel"]
            col = (0, 255, 0) if cp["hit"] else (100, 100, 255)
            cv2.drawMarker(frame, (px, py), col, cv2.MARKER_DIAMOND, 20, 2)
            cv2.putText(frame, cp["name"], (px+12, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Kart icon — deep red
        kx, ky = self.mapper.world_to_pixel(pose.x, pose.y)
        if 0 <= kx < self.MAP_W and 0 <= ky < self.MAP_H:
            ex = int(kx + 20 * math.cos(pose.heading))
            ey = int(ky - 20 * math.sin(pose.heading))
            cv2.circle(frame, (kx, ky), 12, (0, 0, 139), -1)
            cv2.circle(frame, (kx, ky), 12, (0, 0, 80), 2)
            cv2.arrowedLine(frame, (kx, ky), (ex, ey), (0, 0, 139), 3, tipLength=0.4)

        # HUD — dark bar with white text, split into two lines for readability
        hud = np.full((60, self.MAP_W, 3), 20, dtype=np.uint8)
        line1 = (f"Pos: ({pose.x:.1f}, {pose.y:.1f}) m    "
                 f"Hdg: {math.degrees(pose.heading):.0f} deg")
        line2 = (f"Speed: {speed:.1f} km/h    "
                 f"Obj: {len(visible_objs)}    "
                 f"{'STOPPED' if self.tracker.is_kart_stopped() else 'MOVING'}")
        cv2.putText(hud, line1, (10, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, line2, (10, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([frame, hud])

    def _draw_landmarks(self, img):
        for lm in self.landmarks:
            ltype = lm.get("type", "point")
            color = tuple(int(c) for c in lm.get("color_bgr", [200, 200, 200]))
            if ltype == "line" and "pixel_box" in lm:
                xc = lm["pixel_center"][0]
                y1 = lm["pixel_box"]["y_min"]
                y2 = lm["pixel_box"]["y_max"]
                cv2.line(img, (xc, y1), (xc, y2), color, 4)
                cv2.line(img, (xc-10, y1), (xc+10, y1), color, 2)
                cv2.line(img, (xc-10, y2), (xc+10, y2), color, 2)
                cv2.putText(img, lm["name"], (xc+8, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif ltype == "zebra" and "pixel_box" in lm:
                box = lm["pixel_box"]
                x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
                ov = img.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(ov, 0.2, img, 0.8, 0, img)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, lm["name"], (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            elif ltype == "point" and "pixel_center" in lm:
                px, py = lm["pixel_center"]
                cv2.drawMarker(img, (px, py), color, cv2.MARKER_STAR, 18, 2)
        return img

    def _check_points(self, pose, frame_num):
        for cp in self.check_points:
            wx, wy = cp["world"]
            dist = math.sqrt((pose.x - wx) ** 2 + (pose.y - wy) ** 2)
            if dist < cp["closest_m"]:
                cp["closest_m"] = dist
            if cp["hit"]:
                continue
            if dist <= cp["threshold_m"]:
                cp["hit"] = True
                print(f"\n  *** CHECK POINT HIT: {cp['name']} ***")
                print(f"      note      : {cp.get('note','')}")
                print(f"      kart world: ({pose.x:.2f},{pose.y:.2f})m")
                print(f"      cp world  : ({wx:.2f},{wy:.2f})m")
                print(f"      distance  : {dist:.3f}m  (threshold={cp['threshold_m']}m)")
                print(f"      frame     : {frame_num}")

    def _print_checkpoint_summary(self):
        print("\n" + "="*55)
        print("  CHECK POINT SUMMARY")
        print("="*55)
        for cp in self.check_points:
            status = "HIT ✓" if cp["hit"] else "MISSED ✗"
            print(f"  {cp['name']:<16} {status}  world={cp['world']}")

    def _input_thread(self):
        while self._running:
            try:
                c = input().strip().lower()
                self._cmd = c
            except EOFError:
                break

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AV Navigation Map")
    parser.add_argument("--mode",   default="video",
                        choices=["video", "live"],
                        help="'video' for recorded file, 'live' for camera")
    parser.add_argument("--video",  default=None,
                        help="Path to video file (required for --mode video)")
    parser.add_argument("--calibrate-homography", action="store_true",
                        help="Run interactive homography calibration tool")
    args = parser.parse_args()

    if args.calibrate_homography:
        cfg = load_nav_config(CONFIG_PATH)
        if not cfg.track_image_path:
            print("Set 'track_image_path' in nav_config.json first.")
            sys.exit(1)
        helper = HomographyHelper(cfg.track_image_path)
        helper.run()
        return

    default_video = "/home/jannat/sdc_2026/data/Middle20260305154429.mp4"
    video_path = args.video or default_video

    system = MapSystem(
        mode=args.mode,
        video_path=video_path if args.mode == "video" else None,
    )
    system.run()

if __name__ == "__main__":
    main()
