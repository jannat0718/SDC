"""
Landmark-based pose correction for SDC 2026.

Detects bright horizontal stripes (zebra crossings, start/end lines) in the
lower part of the front-camera frame. When a stripe is visible AND the kart's
current VO position is within SNAP_RADIUS_M of a known landmark from
nav_config.fixed_landmarks, the VO pose (x, y) is snapped to that landmark's
world_center. This corrects accumulated VO drift every time the kart drives
over a known crosswalk or line.

Heading correction (optional): if a bird's-eye-view homography is provided,
the orientation of the detected stripe in the BEV gives an absolute heading
observation. Disabled by default until BEV homography is validated on track.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np


# ── Tunables ───────────────────────────────────────────────────────────────
SNAP_RADIUS_M       = 8.0      # only snap if VO pose is within this of a landmark
SNAP_COOLDOWN_S     = 2.0      # min interval between snaps (debounce)
LOWER_FRAC          = 0.45     # process bottom 45% of the frame (road region)
WHITE_THRESHOLD     = 195      # grayscale brightness for "white paint"
MIN_WHITE_ROW_FRAC  = 0.18     # row counts as bright if >18% of its pixels are white
ZEBRA_MIN_STRIPES   = 3        # >=3 bright bands separated by gaps = zebra
LINE_MIN_LEN_FRAC   = 0.35     # Hough line must span at least this fraction of width
LINE_MAX_ANGLE_DEG  = 25       # |angle from horizontal| in image, before warp


@dataclass
class SnapResult:
    landmark_name: str
    world_x: float
    world_y: float
    confidence: float           # 0..1
    heading_rad: Optional[float] = None   # absolute heading observation, if computed


class LandmarkLocalizer:
    """
    Lightweight classical-CV stripe detector + nearest-landmark snap.

    Usage (in av_map main loop, after vo.update()):
        snap = self.localizer.process(frame_ud, self.vo.pose)
        if snap is not None:
            self.vo.pose.x = snap.world_x
            self.vo.pose.y = snap.world_y
            if snap.heading_rad is not None:
                self.vo.pose.heading = snap.heading_rad
    """

    def __init__(self,
                 landmarks: List[dict],
                 bev_homography: Optional[np.ndarray] = None,
                 px_per_meter_bev: Optional[float] = None):
        # Keep only landmarks that have a usable world_center.
        self.landmarks = [lm for lm in landmarks if "world_center" in lm]
        self.H_bev = bev_homography
        self.px_per_m_bev = px_per_meter_bev
        self._last_snap_t = 0.0
        self._last_snap_name: Optional[str] = None

    # ───────────────────────────────────────────────────────────────────
    # Stripe detection
    # ───────────────────────────────────────────────────────────────────
    def _detect_stripe(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Returns {'kind': 'zebra'|'line', 'angle_deg': float, 'roi_y': int,
                 'mask': np.ndarray} or None.
        Operates on the lower LOWER_FRAC of the frame.
        """
        h, w = frame_bgr.shape[:2]
        y0 = int(h * (1.0 - LOWER_FRAC))
        roi = frame_bgr[y0:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Suppress speckle, keep horizontal structures.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Row whiteness profile.
        row_white = mask.mean(axis=1) / 255.0     # fraction white per row
        bright_rows = row_white > MIN_WHITE_ROW_FRAC
        if bright_rows.sum() < 5:
            return None

        # Count bright bands separated by ≥3 dark rows → zebra signature.
        bands = []
        in_band = False
        run_start = 0
        gap = 0
        for i, b in enumerate(bright_rows):
            if b and not in_band:
                in_band, run_start, gap = True, i, 0
            elif not b and in_band:
                gap += 1
                if gap >= 3:
                    bands.append((run_start, i - gap))
                    in_band = False
        if in_band:
            bands.append((run_start, len(bright_rows) - 1))

        if not bands:
            return None

        kind = "zebra" if len(bands) >= ZEBRA_MIN_STRIPES else "line"

        # Estimate orientation via Hough on the mask.
        angle_deg = self._hough_angle(mask, w)
        if angle_deg is None:
            return None
        if abs(angle_deg) > LINE_MAX_ANGLE_DEG:
            return None  # not a roughly-horizontal stripe → ignore

        return {"kind": kind, "angle_deg": angle_deg, "roi_y": y0, "mask": mask,
                "n_bands": len(bands)}

    @staticmethod
    def _hough_angle(mask: np.ndarray, w: int) -> Optional[float]:
        edges = cv2.Canny(mask, 50, 150)
        min_len = int(w * LINE_MIN_LEN_FRAC)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=min_len, maxLineGap=15)
        if lines is None:
            return None
        angles = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Fold to [-90, 90] then near 0 = horizontal.
            if a > 90:  a -= 180
            if a < -90: a += 180
            angles.append(a)
        if not angles:
            return None
        return float(np.median(angles))

    # ───────────────────────────────────────────────────────────────────
    # Nearest landmark
    # ───────────────────────────────────────────────────────────────────
    def _nearest_landmark(self, x: float, y: float) -> Optional[Tuple[dict, float]]:
        best, best_d = None, float("inf")
        for lm in self.landmarks:
            wx, wy = lm["world_center"]
            d = math.hypot(x - wx, y - wy)
            if d < best_d:
                best, best_d = lm, d
        if best is None:
            return None
        return best, best_d

    # ───────────────────────────────────────────────────────────────────
    # Public entry point
    # ───────────────────────────────────────────────────────────────────
    def process(self, frame_bgr: np.ndarray, pose) -> Optional[SnapResult]:
        """
        Run detection and decide whether to emit a snap correction.
        Returns SnapResult or None.
        """
        # Debounce.
        now = time.time()
        if now - self._last_snap_t < SNAP_COOLDOWN_S:
            return None

        det = self._detect_stripe(frame_bgr)
        if det is None:
            return None

        nearest = self._nearest_landmark(pose.x, pose.y)
        if nearest is None:
            return None
        lm, dist_m = nearest
        if dist_m > SNAP_RADIUS_M:
            return None

        # Confidence: stronger for zebra (more bands) and tighter VO proximity.
        prox  = max(0.0, 1.0 - dist_m / SNAP_RADIUS_M)
        kind_w = 1.0 if det["kind"] == "zebra" else 0.6
        conf  = float(np.clip(0.5 * prox + 0.5 * kind_w, 0.0, 1.0))

        # Optional absolute heading from BEV — only if homography supplied.
        heading_rad: Optional[float] = None
        if self.H_bev is not None:
            heading_rad = self._heading_from_bev(det, pose.heading)

        self._last_snap_t = now
        self._last_snap_name = lm["name"]

        return SnapResult(
            landmark_name=lm["name"],
            world_x=float(lm["world_center"][0]),
            world_y=float(lm["world_center"][1]),
            confidence=conf,
            heading_rad=heading_rad,
        )

    def _heading_from_bev(self, det: dict, prior_heading: float) -> Optional[float]:
        """
        Stripe in BEV is perpendicular to kart travel direction.
        Heading observation = (stripe_angle_in_bev + 90°) → world frame.
        Resolves the 180° ambiguity by picking the candidate closest to
        the VO prior heading. Returns None if BEV is unconfigured.
        """
        if self.H_bev is None:
            return None
        # Detect a strong line in the original ROI and warp its endpoints.
        mask = det["mask"]
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                minLineLength=int(mask.shape[1] * 0.4),
                                maxLineGap=20)
        if lines is None:
            return None
        # Pick the longest line.
        best_len, best_line = 0, None
        for x1, y1, x2, y2 in lines[:, 0, :]:
            L = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if L > best_len:
                best_len, best_line = L, (x1, y1, x2, y2)
        if best_line is None:
            return None
        # Lift back to full-frame coords (add roi_y offset on the y component).
        x1, y1, x2, y2 = best_line
        y1 += det["roi_y"]; y2 += det["roi_y"]
        pts = np.float32([[[x1, y1]], [[x2, y2]]])
        warped = cv2.perspectiveTransform(pts, self.H_bev)[:, 0, :]
        dx = warped[1, 0] - warped[0, 0]
        dy = warped[1, 1] - warped[0, 1]
        stripe_angle = math.atan2(dy, dx)            # in BEV pixel frame
        # Kart heading is perpendicular to the stripe; two candidates 180° apart.
        cand_a = stripe_angle + math.pi / 2
        cand_b = stripe_angle - math.pi / 2
        # Choose the candidate nearer the VO prior (to resolve the flip).
        def _wrap(a):
            return math.atan2(math.sin(a), math.cos(a))
        prior = _wrap(prior_heading)
        da = abs(_wrap(cand_a - prior))
        db = abs(_wrap(cand_b - prior))
        return _wrap(cand_a if da <= db else cand_b)
