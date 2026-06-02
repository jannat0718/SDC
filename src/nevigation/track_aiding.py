"""
track_aiding.py
================
Pure helper functions that turn a TrackConstraint query into EKF measurements.

These are stateless functions; the caller (KFPipeline) owns the EKF and
decides whether to feed the result into ekf.update_landmark.

Functions
---------
soft_track_pull(ekf_state, track, R_track_m_sq, max_pull_dist_m)
    If the EKF's current (x, y) is off the drivable mask, return a position
    measurement at the nearest drivable point with a noise floor of R.
    Returns None when:
      - position is already drivable
      - position is too far from the mask (>max_pull_dist_m) - probably a
        sensor failure or coordinate mismatch, refuse to pull
      - position is outside the image bounds (no info)
"""
from __future__ import annotations

from typing import Optional, Tuple

# Forward-declare a duck type for the EKF state so this module stays
# independent of kalman_fusion's actual class. The caller passes anything
# with .x, .y, .theta float attributes.
class _EKFStateLike:
    x: float
    y: float
    theta: float


def soft_track_pull(ekf_state: _EKFStateLike,
                    track,                     # TrackConstraint
                    R_track_m_sq: float = 0.5**2,
                    max_pull_dist_m: float = 5.0
                    ) -> Optional[Tuple[float, float, float]]:
    """If off-track, return (z_x, z_y, R) for a position measurement; else None.

    The returned R is the variance to use for both x and y in update_landmark
    (caller is expected to pass it as np.array([[R, 0], [0, R]])).
    The R inflates with distance from the mask so closer corrections are
    stronger and far-away ones (likely bad data) are softer.
    """
    res = track.nearest_drivable(ekf_state.x, ekf_state.y)
    if res is None:
        return None
    nx, ny, dist_m = res
    if dist_m <= 0.05:           # tiny tolerance; treat as on-track
        return None
    if dist_m > max_pull_dist_m:
        # Too far - refuse to pull. Likely a coordinate mismatch.
        return None
    # Inflate R proportionally to how far we are - far errors are noisier signals
    # R_eff = R_base * (1 + dist/dist_scale)
    dist_scale = max(max_pull_dist_m, 1.0)
    R_eff = R_track_m_sq * (1.0 + dist_m / dist_scale)
    return float(nx), float(ny), float(R_eff)
