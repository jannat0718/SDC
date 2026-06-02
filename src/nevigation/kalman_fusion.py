"""
kalman_fusion.py - Extended Kalman Filter fusing VO motion with landmark snaps.

Wraps the existing VisualOdometry + LandmarkLocalizer (imported read-only,
not edited) with an EKF that tracks position uncertainty over time. Between
landmark snaps, P (covariance) grows; at each snap, P shrinks. The result is
a smoother trajectory than raw VO, and drift is bounded by landmark spacing.

State vector x = [px, py, theta] (3D).
  px, py  - kart position (metres)
  theta   - heading (radians, normalised to [-pi, pi])

Predict step: VO supplies the per-frame motion increment (dpx, dpy, dtheta)
as the control input u. We add it to the state and inflate covariance by
process noise Q + a portion of measurement noise R_vo (since the input itself
is noisy).

Update step (only when a snap fires): the landmark snap provides absolute
(px, py, optional theta) as a measurement z. Standard Kalman gain blends.

The OLD VO pipeline (test_map_video.py, av_map.py, landmark_localizer.py)
stays untouched.
"""

from __future__ import annotations
import os
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Read-only imports. Both modules have no module-level mutable state and we
# never assign back into them.
from nevigation.av_map import (
    VisualOdometry, CameraCalibration, TrackMapper, Pose
)
from nevigation.landmark_localizer import LandmarkLocalizer
from nevigation.utils import load_nav_config_merged, NavConfig


# ──────────────────────────────────────────────────────────────────────────
# EKF state
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class EKFState:
    """3D filter state with 3x3 covariance."""
    x:    float = 0.0   # metres, world frame
    y:    float = 0.0
    theta: float = 0.0  # radians, normalised to [-pi, +pi]
    P:    np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)

    def vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def sigma(self) -> tuple:
        """Standard deviations of (x, y, theta) for HUD display."""
        return (math.sqrt(max(0.0, self.P[0, 0])),
                math.sqrt(max(0.0, self.P[1, 1])),
                math.sqrt(max(0.0, self.P[2, 2])))


def _wrap_pi(angle: float) -> float:
    """Wrap any radian value to [-pi, +pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _heading_deg_360(theta_rad: float) -> float:
    """Display-only: convert heading from [-pi, +pi] rad to [0, 360) deg.

    EKF math always uses signed [-pi, +pi]; this helper is for HUD / CSV
    output only so humans don't have to mentally fold negative angles.
    """
    return (math.degrees(theta_rad) + 360.0) % 360.0


# Chi-squared 95% threshold, 2 degrees of freedom. Used as the Mahalanobis
# gate cutoff for visual landmark snaps: a snap is accepted iff the kart's
# position is within this many "standard deviations" of the landmark, where
# the metric is defined by P_xy + R_landmark_xy. Tighten to 9.210 (99%) if
# false-positive bindings show up; loosen to 4.605 (90%) if real snaps are
# being rejected.
MAHAL_CHI2_95_2D = 5.991


# ──────────────────────────────────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────────────────────────────────

class VOLandmarkEKF:
    """EKF fusing VO motion increments with absolute landmark measurements."""

    def __init__(
        self,
        initial_pose: Pose,
        vo_pos_noise_m:   float = 0.05,   # std of (dpx, dpy) per frame
        vo_yaw_noise_rad: float = math.radians(0.5),  # std of dtheta per frame
        process_noise_m:  float = 0.01,   # extra "always-on" position drift
        process_noise_rad: float = math.radians(0.05),  # extra heading drift
        landmark_pos_noise_m:   float = 0.30,  # how much to trust a snap's position
        landmark_yaw_noise_rad: float = math.radians(5.0),  # snap heading noise (if provided)
    ):
        self.state = EKFState(
            x=initial_pose.x,
            y=initial_pose.y,
            theta=_wrap_pi(initial_pose.heading),
            P=np.diag([0.25, 0.25, math.radians(2.0) ** 2]),
        )

        # VO measurement noise (per-frame increment)
        self.R_vo = np.diag([
            vo_pos_noise_m ** 2,
            vo_pos_noise_m ** 2,
            vo_yaw_noise_rad ** 2,
        ])
        # Always-on process noise
        self.Q = np.diag([
            process_noise_m ** 2,
            process_noise_m ** 2,
            process_noise_rad ** 2,
        ])
        # Landmark measurement noise (absolute position and optional heading)
        self.R_landmark_xy = np.diag([
            landmark_pos_noise_m ** 2,
            landmark_pos_noise_m ** 2,
        ])
        self.R_landmark_yaw = landmark_yaw_noise_rad ** 2

        self.trajectory = [(self.state.x, self.state.y)]
        self.n_predicts = 0
        self.n_vo_updates = 0
        self.n_landmark_updates = 0

        # EMA of world-frame velocity (m/frame) for snap-reversal detection.
        self._recent_vel = (0.0, 0.0)
        self._vel_alpha = 0.3

    def predict_with_vo(self, dpx: float, dpy: float, dtheta: float) -> None:
        """Advance the state by a VO-supplied increment. dpx/dpy are in WORLD
        frame (av_map.VisualOdometry already rotates by current heading), so
        the transition is a simple sum. Covariance inflates by Q + R_vo."""
        self.state.x += float(dpx)
        self.state.y += float(dpy)
        self.state.theta = _wrap_pi(self.state.theta + float(dtheta))
        # F = I (additive model in world frame). P_new = F·P·F^T + Q + R_vo.
        self.state.P = self.state.P + self.Q + self.R_vo
        self.n_predicts += 1
        self.trajectory.append((self.state.x, self.state.y))
        # Update velocity EMA for the snap-reversal detector.
        a = self._vel_alpha
        self._recent_vel = (
            a * float(dpx) + (1.0 - a) * self._recent_vel[0],
            a * float(dpy) + (1.0 - a) * self._recent_vel[1],
        )

    def update_landmark(
        self,
        z_x: float,
        z_y: float,
        z_theta: Optional[float] = None,
    ) -> dict:
        """Apply an absolute landmark measurement. Returns a dict summarising
        the update (pre/post pose, Kalman gain trace, residual) for logging."""
        x_pre = self.state.vec().copy()
        P_pre = self.state.P.copy()

        if z_theta is None:
            # 2D measurement (x, y only). H picks out the first two rows.
            H = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
            z = np.array([z_x, z_y])
            R = self.R_landmark_xy
        else:
            # 3D measurement (x, y, theta).
            H = np.eye(3)
            z = np.array([z_x, z_y, _wrap_pi(z_theta)])
            R = np.diag([self.R_landmark_xy[0, 0],
                         self.R_landmark_xy[1, 1],
                         self.R_landmark_yaw])

        innov = z - H @ x_pre
        if z_theta is not None:
            innov[2] = _wrap_pi(innov[2])  # angular residual must wrap

        S = H @ P_pre @ H.T + R
        K = P_pre @ H.T @ np.linalg.inv(S)

        x_post = x_pre + K @ innov
        x_post[2] = _wrap_pi(x_post[2])
        I = np.eye(3)
        P_post = (I - K @ H) @ P_pre

        self.state.x, self.state.y, self.state.theta = (
            float(x_post[0]), float(x_post[1]), float(x_post[2]))
        self.state.P = P_post
        self.n_landmark_updates += 1
        # Re-anchor the trajectory: the last point should reflect the snap
        self.trajectory.append((self.state.x, self.state.y))

        # Snap-reversal diagnostic: warn when the snap pushes the kart
        # opposite to its recent heading of travel. Real corrections usually
        # move the kart slightly sideways or along the velocity vector; a
        # large displacement *against* it suggests a wrong-landmark binding.
        SNAP_REVERSAL_M = 2.0
        snap_dx = float(x_post[0] - x_pre[0])
        snap_dy = float(x_post[1] - x_pre[1])
        snap_mag = math.hypot(snap_dx, snap_dy)
        vx, vy = self._recent_vel
        vel_mag = math.hypot(vx, vy)
        if snap_mag > SNAP_REVERSAL_M and vel_mag > 0.05:
            dot = (snap_dx * vx + snap_dy * vy) / vel_mag
            if dot < -SNAP_REVERSAL_M:
                print(f"[WARN] Snap reversal: moved {snap_mag:.2f}m, "
                      f"{dot:.2f}m AGAINST recent velocity "
                      f"({vx:+.2f},{vy:+.2f}). "
                      f"Possible wrong-landmark binding.")

        return {
            "pre_xyz":  (float(x_pre[0]), float(x_pre[1]), math.degrees(float(x_pre[2]))),
            "post_xyz": (float(x_post[0]), float(x_post[1]), math.degrees(float(x_post[2]))),
            "innov":    tuple(round(float(v), 3) for v in innov),
            "gain_trace": float(np.trace(K)),
        }

    @property
    def pose(self) -> Pose:
        """Adapter so callers expecting an av_map.Pose can use the EKF state."""
        return Pose(x=self.state.x, y=self.state.y, heading=self.state.theta)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────

class KFPipeline:
    """End-to-end runner: video -> VO -> EKF predict; landmark detection ->
    EKF update; render trajectory on Track.png.

    Reuses VisualOdometry, CameraCalibration, TrackMapper, LandmarkLocalizer
    from the existing nevigation modules (all read-only imports). No edits
    to any existing .py file.
    """

    PREVIEW_W = 1280

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        vo_pos_noise_m:   float = 0.05,
        vo_yaw_noise_rad: float = math.radians(0.5),
        landmark_pos_noise_m: float = 0.30,
        start_pixel_override: Optional[list] = None,
        heading_override_rad: Optional[float] = None,
        feature_detector: str = "gftt",
        preview_every_n_frames: int = 1,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        self.cfg_raw = load_nav_config_merged(config_path)
        if start_pixel_override is not None:
            self.cfg_raw.setdefault("track_info", {})["start_pixel"] = list(start_pixel_override)
        if heading_override_rad is not None:
            self.cfg_raw["initial_heading_rad"] = float(heading_override_rad)
        self.cfg = NavConfig(self.cfg_raw)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.preview_path = os.path.join(output_dir, "preview.jpg")
        self.final_map_path = os.path.join(output_dir, "map_final.png")
        self.trajectory_csv_path = os.path.join(output_dir, "trajectory.csv")

        # Camera calibration
        calib_path = os.path.join(os.path.dirname(__file__), "camera_K_real.json")
        self.calib = CameraCalibration(calib_path)

        # Track image + landmark coords for initial pose
        self.px_per_m = float(self.cfg_raw["px_per_meter"])
        self.world_origin_px = self.cfg_raw["start_px"]
        sp = self.cfg_raw["track_info"]["start_pixel"]
        start_wx =  (sp[0] - self.world_origin_px[0]) / self.px_per_m
        start_wy = -(sp[1] - self.world_origin_px[1]) / self.px_per_m
        h0 = float(self.cfg_raw["initial_heading_rad"])
        init_pose = Pose(x=start_wx, y=start_wy, heading=h0)
        print(f"[KFPipeline] start pixel=({sp[0]},{sp[1]}) "
              f"world=({start_wx:.2f},{start_wy:.2f})m  "
              f"heading={_heading_deg_360(h0):.1f}deg")

        # Video
        self.video_path = self.cfg_raw["video_path"]
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[KFPipeline] video={os.path.basename(self.video_path)} "
              f"{video_w}x{video_h} {self.fps:.1f}fps n={self.n_frames}")

        # Scale K to match video resolution (mirrors NavigationPipeline)
        baseline_w, baseline_h = self.calib.cam.resolution
        sx = video_w / baseline_w
        sy = video_h / baseline_h
        self.calib.K[0, 0] *= sx
        self.calib.K[1, 1] *= sy
        self.calib.K[0, 2] *= sx
        self.calib.K[1, 2] *= sy

        # VisualOdometry (reused as-is; we own this instance, not a shared one)
        self.vo = VisualOdometry(
            self.calib.K,
            scale_factor=self.cfg.scale_factor,
            feature_detector=feature_detector,
        )
        # Seed VO pose to match initial pose so first frame returns sane
        # increments. The EKF will subtract VO state to get the increment.
        self.vo.pose.x = start_wx
        self.vo.pose.y = start_wy
        self.vo.pose.heading = h0
        #_raw = h0 - math.pi / 2
        _raw = math.pi / 2 - h0
        _c, _s = math.cos(_raw), math.sin(_raw)
        self.vo._R_cum = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
        self.vo.trajectory = [(start_wx, start_wy)]

        # LandmarkLocalizer (reused as-is)
        landmarks = self.cfg_raw.get("fixed_landmarks", [])
        H_bev = None
        H_raw = self.cfg_raw.get("H_bev")
        if H_raw is not None:
            try:
                H_bev = np.array(H_raw, dtype=np.float32).reshape(3, 3)
            except Exception:
                H_bev = None
        # snap_gate_fn is a bound method that reads self.ekf at call time —
        # it's safe to pass here even though self.ekf is built below.
        self.localizer = LandmarkLocalizer(
            landmarks=landmarks,
            bev_homography=H_bev,
            px_per_meter_bev=self.cfg_raw.get("px_per_meter_bev"),
            snap_gate_fn=self._mahalanobis_gate,
        )

        # TrackMapper for rendering
        self.mapper = TrackMapper(self.cfg, map_size=None) if False else None
        # TrackMapper init signature in av_map needs (cfg, map_size=(W,H)).
        # Use track image dims so output matches existing pipeline.
        track_img = cv2.imread(self.cfg.track_image_path)
        self.map_h, self.map_w = track_img.shape[:2]
        self.mapper = TrackMapper(self.cfg, map_size=(self.map_w, self.map_h))
        self.track_bg = track_img
        self.landmarks = landmarks

        # EKF
        self.ekf = VOLandmarkEKF(
            initial_pose=init_pose,
            vo_pos_noise_m=vo_pos_noise_m,
            vo_yaw_noise_rad=vo_yaw_noise_rad,
            landmark_pos_noise_m=landmark_pos_noise_m,
        )

        # Trajectory output
        self._traj_rows = []
        self.preview_every_n = max(1, int(preview_every_n_frames))
        self.start_frame = start_frame
        self.end_frame = end_frame

    # ── world<->pixel helpers ────────────────────────────────────────────
    def world_to_pixel(self, wx: float, wy: float) -> tuple:
        u = int(self.world_origin_px[0] + wx * self.px_per_m)
        v = int(self.world_origin_px[1] - wy * self.px_per_m)
        return u, v

    # ── snap gating ──────────────────────────────────────────────────────
    def _mahalanobis_gate(self, x_kart: float, y_kart: float, lm: dict) -> bool:
        """Accept a visual landmark snap iff the kart's position is within
        MAHAL_CHI2_95_2D of the landmark under the EKF's own covariance.

        Adapts the gate to filter confidence: tight when sigma is small
        (rejects wrong-zebra parallels), wider when sigma has grown.

        S = P_xy + R_landmark_xy. Adding R prevents the gate from shrinking
        to nothing right after a snap, when P_xy can be ~0.09 m^2 per axis.
        """
        wx, wy = lm["world_center"]
        dx = np.array([wx - x_kart, wy - y_kart])
        Pxy = self.ekf.state.P[:2, :2]
        S = Pxy + self.ekf.R_landmark_xy
        try:
            d2 = float(dx @ np.linalg.solve(S, dx))
        except np.linalg.LinAlgError:
            return False
        return d2 <= MAHAL_CHI2_95_2D

    # ── render ───────────────────────────────────────────────────────────
    def _render_map(self) -> np.ndarray:
        frame = self.track_bg.copy()

        # Trajectory: navy
        traj = self.ekf.trajectory
        if len(traj) > 1:
            pts = [self.world_to_pixel(x, y) for x, y in traj]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (128, 0, 0), 3)

        # Kart marker
        p = self.ekf.pose
        kx, ky = self.world_to_pixel(p.x, p.y)
        if 0 <= kx < self.map_w and 0 <= ky < self.map_h:
            ex = int(kx + 25 * math.cos(p.heading))
            ey = int(ky - 25 * math.sin(p.heading))
            cv2.circle(frame, (kx, ky), 13, (0, 0, 139), -1)
            cv2.arrowedLine(frame, (kx, ky), (ex, ey),
                            (0, 0, 139), 3, tipLength=0.4)

        # HUD bar
        sigma = self.ekf.state.sigma()
        hud = np.full((60, self.map_w, 3), 20, dtype=np.uint8)
        line1 = (f"EKF Pos: ({p.x:.2f}, {p.y:.2f}) m   "
                 f"Hdg: {_heading_deg_360(p.heading):.1f} deg")
        line2 = (f"sigma_x={sigma[0]:.2f}m  sigma_y={sigma[1]:.2f}m  "
                 f"sigma_th={math.degrees(sigma[2]):.1f}deg   "
                 f"VO upd={self.ekf.n_vo_updates}  "
                 f"LM upd={self.ekf.n_landmark_updates}")
        cv2.putText(hud, line1, (10, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, line2, (10, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([frame, hud])

    def _save_preview(self, camera_frame: np.ndarray, frame_num: int):
        map_frame = self._render_map()
        ch, cw = camera_frame.shape[:2]
        scale = self.PREVIEW_W / cw
        cam_s = cv2.resize(camera_frame, (self.PREVIEW_W, int(ch * scale)))

        p = self.ekf.pose
        kx, ky = self.world_to_pixel(p.x, p.y)
        sigma = self.ekf.state.sigma()
        hud_lines = [
            f"Frame {frame_num:04d}",
            f"World ({p.x:+.2f}, {p.y:+.2f}) m",
            f"Pixel ({kx}, {ky})",
            f"Heading {_heading_deg_360(p.heading):6.1f} deg",
            f"sigma_x {sigma[0]:.2f} m   sigma_th {math.degrees(sigma[2]):.1f} deg",
            f"VO upd {self.ekf.n_vo_updates}  LM upd {self.ekf.n_landmark_updates}",
        ]
        bar_h = 22 * len(hud_lines) + 12
        cv2.rectangle(cam_s, (0, 0), (440, bar_h), (0, 0, 0), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(cam_s, line, (8, 20 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)

        mh, mw = map_frame.shape[:2]
        map_s = cv2.resize(map_frame, (self.PREVIEW_W,
                                       int(mh * self.PREVIEW_W / mw)))
        cv2.imwrite(self.preview_path, np.vstack([cam_s, map_s]))

    # ── main loop ────────────────────────────────────────────────────────
    def run(self):
        frame_num = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[KFPipeline] end of video.")
                    break
                frame_num += 1
                if self.start_frame is not None and frame_num < self.start_frame:
                    continue
                if self.end_frame is not None and frame_num > self.end_frame:
                    print(f"[KFPipeline] reached end_frame={self.end_frame}")
                    break

                frame_ud = self.calib.undistort(frame)

                # Capture pre-VO pose, run VO, derive increment
                vo_pre_x = self.vo.pose.x
                vo_pre_y = self.vo.pose.y
                vo_pre_h = self.vo.pose.heading
                self.vo.update(frame_ud)
                dpx = self.vo.pose.x - vo_pre_x
                dpy = self.vo.pose.y - vo_pre_y
                dth = _wrap_pi(self.vo.pose.heading - vo_pre_h)

                # EKF predict using VO increment
                self.ekf.predict_with_vo(dpx, dpy, dth)
                self.ekf.n_vo_updates += 1

                # Landmark detection on the EKF pose (not VO pose), then snap.
                # LandmarkLocalizer's process(frame, pose) is read-only.
                snap_info = None
                snap = self.localizer.process(frame_ud, self.ekf.pose)
                if snap is not None:
                    snap_info = self.ekf.update_landmark(
                        z_x=snap.world_x,
                        z_y=snap.world_y,
                        z_theta=getattr(snap, "heading_rad", None),
                    )
                    print(f"[F{frame_num:04d}] LM snap '{snap.landmark_name}' "
                          f"pre={snap_info['pre_xyz']} "
                          f"post={snap_info['post_xyz']} "
                          f"innov={snap_info['innov']}")

                # Proximity snaps (no frame analysis needed)
                prox = self.localizer.check_proximity_snaps(self.ekf.pose)
                if prox is not None:
                    snap_info = self.ekf.update_landmark(
                        z_x=prox.world_x,
                        z_y=prox.world_y,
                        z_theta=getattr(prox, "heading_rad", None),
                    )
                    print(f"[F{frame_num:04d}] PROX snap '{prox.landmark_name}' "
                          f"pre={snap_info['pre_xyz']} "
                          f"post={snap_info['post_xyz']}")

                # Per-frame log row
                p = self.ekf.pose
                kx, ky = self.world_to_pixel(p.x, p.y)
                sx, sy, sth = self.ekf.state.sigma()
                self._traj_rows.append((
                    frame_num,
                    round(p.x, 3), round(p.y, 3),
                    kx, ky,
                    round(_heading_deg_360(p.heading), 2),
                    round(sx, 3), round(sy, 3),
                    round(math.degrees(sth), 2),
                    1 if snap_info else 0,
                ))
                if frame_num == 1 or frame_num % 5 == 0:
                    print(f"[F{frame_num:04d}] EKF world=({p.x:+.2f},{p.y:+.2f})m "
                          f"px=({kx},{ky}) hdg={_heading_deg_360(p.heading):6.1f}deg "
                          f"sigma_x={sx:.2f}m sigma_th={math.degrees(sth):.1f}deg")

                if frame_num % self.preview_every_n == 0 or frame_num == 1:
                    self._save_preview(frame_ud, frame_num)
        finally:
            self.cap.release()
            cv2.imwrite(self.final_map_path, self._render_map())
            import pandas as pd
            traj_df = pd.DataFrame(
                self._traj_rows,
                columns=["frame", "x_m", "y_m", "pixel_x", "pixel_y",
                         "heading_deg", "sigma_x_m", "sigma_y_m",
                         "sigma_heading_deg", "snap_event"],
            )
            traj_df.to_csv(self.trajectory_csv_path, index=False)
            print()
            print("[KFPipeline] === Summary ===")
            print(f"  frames_processed     = {frame_num}")
            print(f"  VO updates           = {self.ekf.n_vo_updates}")
            print(f"  landmark updates     = {self.ekf.n_landmark_updates}")
            print(f"  trajectory points    = {len(self.ekf.trajectory)}")
            p = self.ekf.pose
            print(f"  final pose           = ({p.x:.2f}, {p.y:.2f}) m, "
                  f"heading={_heading_deg_360(p.heading):.1f}deg")
            sx, sy, sth = self.ekf.state.sigma()
            print(f"  final sigma          = ({sx:.2f}, {sy:.2f}) m, "
                  f"{math.degrees(sth):.2f} deg")
            print(f"  map -> {self.final_map_path}")
            print(f"  trajectory -> {self.trajectory_csv_path}")
            print(f"  preview -> {self.preview_path}")
