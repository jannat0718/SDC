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
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np


# ── Tunables ───────────────────────────────────────────────────────────────
CAMERA_TO_FRONT_M   = 0.5      # camera is this far behind the kart's front bumper
SNAP_RADIUS_M       = 15.0     # only snap if VO pose is within this of a landmark
LOG_EVERY_N_DETECTS = 10       # throttle [Localizer] detection log lines
LOWER_FRAC          = 0.45     # process bottom 45% of the frame (road region)
WHITE_THRESHOLD     = 170      # grayscale brightness for "white paint" (tuned for 640x480 compressed video)
MIN_WHITE_ROW_FRAC  = 0.18     # row counts as bright if >18% of its pixels are white
ZEBRA_MIN_STRIPES   = 3        # >=3 bright bands separated by gaps = zebra
LINE_MIN_LEN_FRAC   = 0.15     # Hough line must span at least this fraction of width (tuned for 640x480)
LINE_MAX_ANGLE_DEG  = 25       # |angle from horizontal| in image, before warp
HOUGH_VOTE_THRESH   = 40       # Hough votes (lowered from 60 for 640x480)
HOUGH_MAX_GAP_PX    = 25       # max gap between collinear segments (raised from 15 for 640x480)
ZEBRA_CONFIRM_FRAMES = 3       # require N consecutive zebra frames on the same nearest landmark before snapping


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
        # Snap-eligible landmarks: stripe types (zebra/line) detected visually,
        # plus proximity_snap types triggered by pose distance alone.
        # Point landmarks (yield_triangles, start_point, end_point) are display-only.
        self.landmarks = [lm for lm in landmarks
                          if "world_center" in lm and lm.get("type") in ("zebra", "line")]
        self.proximity_landmarks = [lm for lm in landmarks
                                    if lm.get("type") == "proximity_snap"]
        self.H_bev = bev_homography
        self.px_per_m_bev = px_per_meter_bev
        self._last_snap_name: Optional[str] = None
        self._detect_count = 0          # increments every time a stripe is detected
        self._consec_zebra_frames  = 0
        # Entry-based snap gate: snap fires once per zone crossing for visual snaps too.
        # Landmark name added on snap; removed when stripe disappears for 2+ frames.
        self._visual_snap_inside: set = set()
        self._no_stripe_frames = 0      # consecutive frames with no stripe detected
        self._consec_landmark_name: Optional[str] = None
        # Track which proximity zones the kart is currently inside.
        # Snap fires once on entry; suppressed until kart exits and re-enters.
        self._prox_inside: set = set()

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
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_VOTE_THRESH,
                                minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP_PX)
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
    # Heading from config candidates
    # ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _snap_heading_from_candidates(lm: dict, prior_heading: float) -> Optional[float]:
        """Pick the heading candidate nearest to the VO prior heading.

        Unidirectional landmark (one candidate):
            Always returns that heading — no ambiguity.

        Bidirectional landmark (two candidates, e.g. zebra_3 with [0, π]):
            Picks whichever candidate the kart is currently heading towards.
            Works because the two candidates are always 180° apart: as long as
            VO drift is < 90°, the wrong candidate is always further away.

        Returns None if the landmark has no 'heading_candidates_rad' key.
        """
        candidates = lm.get("heading_candidates_rad")
        if not candidates:
            return None
        if len(candidates) == 1:
            return float(candidates[0])
        # Multiple candidates — pick nearest to VO prior
        def _adist(a: float, b: float) -> float:
            return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
        best = min(candidates, key=lambda c: _adist(float(c), prior_heading))
        return float(best)

    @staticmethod
    def _get_snap_point(lm: dict, heading_rad: Optional[float]) -> list:
        """
        Return the world [x, y] position to snap the camera to.

        Looks up 'snap_points' in the landmark config (keyed by heading string).
        The snap_points store where the KART FRONT hits the stripe entry edge.
        The camera is CAMERA_TO_FRONT_M behind the front, so:
            camera_x = entry_x - CAMERA_TO_FRONT_M * cos(heading)
            camera_y = entry_y - CAMERA_TO_FRONT_M * sin(heading)

        For this track all stripes are perpendicular to ±X travel, so
        sin(heading)=0 and only X is offset. Falls back to world_center if
        snap_points is absent or heading is None.
        """
        snap_pts = lm.get("snap_points")
        if snap_pts and heading_rad is not None:
            # Find the snap_point key closest to the selected heading
            def _adist(a: float, b: float) -> float:
                return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
            best_key = min(snap_pts.keys(),
                           key=lambda k: _adist(float(k), heading_rad))
            pt = list(snap_pts[best_key])
            # Apply camera-to-front offset along heading direction
            pt[0] -= CAMERA_TO_FRONT_M * math.cos(heading_rad)
            pt[1] -= CAMERA_TO_FRONT_M * math.sin(heading_rad)
            return pt
        return list(lm["world_center"])

    # ───────────────────────────────────────────────────────────────────
    # Proximity snap (position-based, no visual detection required)
    # ───────────────────────────────────────────────────────────────────
    def check_proximity_snaps(self, pose) -> Optional[SnapResult]:
        """
        Check all proximity_snap landmarks. Fires once when the kart enters the
        radius; suppressed every subsequent frame until the kart exits and
        re-enters. This prevents the snap from repeatedly pulling the kart back
        while it is travelling through the zone.
        """
        best_snap: Optional[SnapResult] = None
        best_dist = float("inf")

        for lm in self.proximity_landmarks:
            name = lm["name"]
            wx, wy = lm["world_center"]
            dist = math.hypot(pose.x - wx, pose.y - wy)
            radius = float(lm.get("snap_radius_m", 2.0))

            inside_now = dist <= radius

            if not inside_now:
                # Kart has left the zone — allow snap again on next entry
                self._prox_inside.discard(name)
                continue

            if name in self._prox_inside:
                # Already snapped this entry — do nothing until kart exits
                continue

            # First frame inside: mark as inside and queue snap
            self._prox_inside.add(name)
            if dist < best_dist:
                best_dist = dist
                heading_rad = self._snap_heading_from_candidates(lm, pose.heading)
                snap_pt = self._get_snap_point(lm, heading_rad)
                best_snap = SnapResult(
                    landmark_name=name,
                    world_x=snap_pt[0],
                    world_y=snap_pt[1],
                    confidence=1.0,
                    heading_rad=heading_rad,
                )

        if best_snap is not None:
            print(f"[Localizer] PROXIMITY SNAP → {best_snap.landmark_name} "
                  f"pos=({best_snap.world_x:.2f},{best_snap.world_y:.2f}) "
                  f"hdg={math.degrees(best_snap.heading_rad or 0):.1f}°  "
                  f"kart_dist={best_dist:.2f}m")

        return best_snap

    # ───────────────────────────────────────────────────────────────────
    # Public entry point
    # ───────────────────────────────────────────────────────────────────
    def process(self, frame_bgr: np.ndarray, pose) -> Optional[SnapResult]:
        """
        Run detection and decide whether to emit a snap correction.
        Returns SnapResult or None.

        Emits throttled `[Localizer]` log lines whenever a stripe is detected,
        regardless of whether a snap is ultimately produced. This is the
        primary diagnostic for whether the detector is firing at all.
        """
        det = self._detect_stripe(frame_bgr)
        if det is None:
            # No stripe this frame — streak broken.
            self._consec_zebra_frames  = 0
            self._consec_landmark_name = None
            # After 2+ consecutive no-stripe frames, allow the same landmark to
            # snap again (kart has clearly passed it or moved away).
            self._no_stripe_frames += 1
            if self._no_stripe_frames >= 2:
                self._visual_snap_inside.clear()
            return None
        self._no_stripe_frames = 0

        self._detect_count += 1

        nearest = self._nearest_landmark(pose.x, pose.y)
        lm, dist_m = (None, float("inf")) if nearest is None else nearest
        lm_name = lm["name"] if lm is not None else None

        # Update consecutive-frame streak for the same nearest landmark.
        # A streak requires both kind=zebra AND a non-None nearest landmark.
        if det["kind"] == "zebra" and lm_name is not None \
                and lm_name == self._consec_landmark_name:
            self._consec_zebra_frames += 1
        elif det["kind"] == "zebra" and lm_name is not None:
            self._consec_zebra_frames  = 1
            self._consec_landmark_name = lm_name
        else:
            self._consec_zebra_frames  = 0
            self._consec_landmark_name = None

        # Decide whether to snap.
        suppress_reason = None
        if det["kind"] != "zebra":
            # Single-band "line" detections are too generic (road edges, lane lines).
            # Only multi-band zebra signatures (n_bands >= ZEBRA_MIN_STRIPES) trigger snap.
            suppress_reason = f"not_zebra(bands={det['n_bands']})"
        elif lm is None:
            suppress_reason = "no_landmarks"
        elif dist_m > SNAP_RADIUS_M:
            suppress_reason = f"out_of_range({dist_m:.1f}>{SNAP_RADIUS_M:.0f}m)"
        elif self._consec_zebra_frames < ZEBRA_CONFIRM_FRAMES:
            suppress_reason = (f"confirming({self._consec_zebra_frames}"
                               f"/{ZEBRA_CONFIRM_FRAMES})")
        elif lm_name in self._visual_snap_inside:
            suppress_reason = f"already_snapped({lm_name})"

        # Throttled diagnostic log.
        if self._detect_count % LOG_EVERY_N_DETECTS == 1:
            tag = "SNAP" if suppress_reason is None else f"SKIP[{suppress_reason}]"
            nm  = lm["name"] if lm is not None else "—"
            print(f"[Localizer] det#{self._detect_count} kind={det['kind']} "
                  f"bands={det['n_bands']} angle={det['angle_deg']:.1f}° "
                  f"nearest={nm} dist={dist_m:.1f}m  → {tag}")

        if suppress_reason is not None:
            return None

        # Confidence: stronger for zebra (more bands) and tighter VO proximity.
        prox  = max(0.0, 1.0 - dist_m / SNAP_RADIUS_M)
        kind_w = 1.0 if det["kind"] == "zebra" else 0.6
        conf  = float(np.clip(0.5 * prox + 0.5 * kind_w, 0.0, 1.0))

        # Heading — config candidates first (primary), BEV fallback (secondary).
        #
        # Config candidates (heading_candidates_rad in nav_config):
        #   Encodes the track geometry explicitly. For bidirectional zebra_3
        #   the list is [0, π]; for others it's a single value. The nearest
        #   candidate to the VO prior is selected — this robustly disambiguates
        #   which lane the kart is in even with moderate VO drift.
        #
        # BEV fallback (used only when no candidates are defined in config):
        #   Older behaviour — derives heading from the warped stripe angle.
        #   Kept for backward compatibility with landmarks that predate the
        #   heading_candidates_rad field.
        heading_rad: Optional[float] = self._snap_heading_from_candidates(lm, pose.heading)
        if heading_rad is None and self.H_bev is not None:
            heading_rad = self._heading_from_bev(det, pose.heading)

        self._last_snap_name = lm["name"]
        self._visual_snap_inside.add(lm["name"])   # gate: no repeat until stripe gone

        # Use entry-edge snap point (with camera offset) instead of world_center
        snap_pt = self._get_snap_point(lm, heading_rad)

        return SnapResult(
            landmark_name=lm["name"],
            world_x=snap_pt[0],
            world_y=snap_pt[1],
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
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_VOTE_THRESH,
                                minLineLength=int(mask.shape[1] * LINE_MIN_LEN_FRAC),
                                maxLineGap=HOUGH_MAX_GAP_PX)
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
        stripe_angle = math.atan2(dy, dx)            # in kart-relative BEV frame
        # H_bev outputs kart-relative metres (X=kart's right, Y=kart's forward),
        # so stripe_angle is a kart-frame angle. Convert to world frame by adding
        # (prior_heading - π/2). The kart heading is perpendicular to the stripe
        # (stripe_angle ± π/2 in kart frame); substituting cancels the π/2 and
        # collapses to:
        cand_a = stripe_angle + prior_heading
        cand_b = stripe_angle + prior_heading - math.pi
        # Choose the candidate nearer the VO prior (to resolve the flip).
        def _wrap(a):
            return math.atan2(math.sin(a), math.cos(a))
        prior = _wrap(prior_heading)
        da = abs(_wrap(cand_a - prior))
        db = abs(_wrap(cand_b - prior))
        return _wrap(cand_a if da <= db else cand_b)
