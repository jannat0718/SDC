"""
INTEGRATION PATCH v2 for kalman_fusion.py
==========================================

This version matches the ACTUAL signatures found in your kalman_fusion.py:
  * update_landmark takes z_x, z_y, z_theta (kwargs), not z + R
  * R is hard-coded inside update_landmark via self.R_landmark_xy
  * frame counter is local `frame_num`, not self.frame_idx
  * EKF state has self.ekf.state.x / .y / .theta directly

Three edits, all additive (~20 lines), no existing lines modified.

----------------------------------------------------------------------------
EDIT 1: imports near the top of kalman_fusion.py
----------------------------------------------------------------------------

Find the existing imports block (around lines 1-25). Append:

    # --- track-mask soft constraint (optional) ---
    try:
        from track_constraint import load_from_nav_config
        from track_aiding import soft_track_pull
        _TRACK_AIDING_AVAILABLE = True
    except ImportError:
        _TRACK_AIDING_AVAILABLE = False

----------------------------------------------------------------------------
EDIT 2: at the END of KFPipeline.__init__
----------------------------------------------------------------------------

The current __init__ ends around line 463 (after all the other state setup).
Find the final statement before the closing of __init__ and append AFTER it:

    # --- track-mask aiding (optional, fails silently if files missing) ---
    self.track = None
    self._track_R_m_sq = float(self.cfg_raw.get('track_constraint_R_m_sq', 0.25))
    self._track_max_pull_m = float(self.cfg_raw.get('track_constraint_max_pull_m', 5.0))
    if _TRACK_AIDING_AVAILABLE and self.cfg_raw.get('use_track_mask', True):
        try:
            self.track = load_from_nav_config(self.cfg_raw)
            print(f"[KFPipeline] Track mask aiding ENABLED "
                  f"(R={self._track_R_m_sq:.4f}, max_pull={self._track_max_pull_m:.1f}m)")
        except Exception as e:
            print(f"[KFPipeline] Track mask aiding DISABLED ({type(e).__name__}: {e})")
            self.track = None

----------------------------------------------------------------------------
EDIT 3: inside KFPipeline.run(), AFTER the proximity-snap block
----------------------------------------------------------------------------

Locate this block in run() (around lines 510-520):

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

Add AFTER it (before the "# Per-frame log row" comment):

    # --- track-mask soft pull (gentle correction back to drivable area) ---
    if self.track is not None:
        pull = soft_track_pull(
            self.ekf.state, self.track,
            R_track_m_sq=self._track_R_m_sq,
            max_pull_dist_m=self._track_max_pull_m,
        )
        if pull is not None:
            zx, zy, _R_eff = pull        # R unused: update_landmark uses R_landmark_xy
            snap_info = self.ekf.update_landmark(z_x=zx, z_y=zy)
            print(f"[F{frame_num:04d}] track-pull -> ({zx:+.2f},{zy:+.2f})m  "
                  f"pre={snap_info['pre_xyz']} post={snap_info['post_xyz']}")

Note on R: the existing update_landmark uses self.R_landmark_xy internally,
so we ignore the R from soft_track_pull. This means every track-pull has
the same noise variance as a landmark snap. If you want a softer pull (so
the EKF trusts it less than a real landmark), see "Optional EDIT 4" below.

----------------------------------------------------------------------------
OPTIONAL EDIT 4: per-call R override (only if track-pulls feel too aggressive)
----------------------------------------------------------------------------

If the soft pull behaves more like a hard snap than a gentle correction,
add an R_override parameter to VOLandmarkEKF.update_landmark.

Find the existing signature (around line 159):

    def update_landmark(
        self,
        z_x: float,
        z_y: float,
        z_theta: Optional[float] = None,
    ) -> dict:

Change to:

    def update_landmark(
        self,
        z_x: float,
        z_y: float,
        z_theta: Optional[float] = None,
        R_xy_override: Optional[float] = None,
    ) -> dict:

Find the R assignment block (line 175):

            R = self.R_landmark_xy

Change to:

            if R_xy_override is not None:
                R = np.eye(2) * float(R_xy_override)
            else:
                R = self.R_landmark_xy

And in the 3D measurement branch (line 180):

            R = np.diag([self.R_landmark_xy[0, 0],
                         self.R_landmark_xy[1, 1],
                         self.R_landmark_yaw])

Change to:

            xy_var = (R_xy_override if R_xy_override is not None
                      else self.R_landmark_xy[0, 0])
            R = np.diag([xy_var, xy_var, self.R_landmark_yaw])

Then in EDIT 3, change:

            snap_info = self.ekf.update_landmark(z_x=zx, z_y=zy)

to:

            snap_info = self.ekf.update_landmark(
                z_x=zx, z_y=zy, R_xy_override=_R_eff)

----------------------------------------------------------------------------
CONFIG additions (nav_config_1080.json)
----------------------------------------------------------------------------

Add to the top-level config object (alongside existing keys):

    "use_track_mask": true,
    "track_constraint_R_m_sq": 0.25,
    "track_constraint_max_pull_m": 5.0,
    "track_mask_artifacts": {
        "drivable": "track_drivable_mask.npy",
        "markings": "track_markings_mask.npy"
    }

All four keys are optional; defaults match these values.

----------------------------------------------------------------------------
ROLLBACK
----------------------------------------------------------------------------

To disable without removing code, set in config:

    "use_track_mask": false

----------------------------------------------------------------------------
EXPECTED RUNTIME BEHAVIOUR
----------------------------------------------------------------------------

Per-frame:
  Most frames     -> no pull, no log line
  Off-track frame -> "[Fxxxx] track-pull -> (+x,+y)m  pre=... post=..."
  Catastrophic    -> no pull (silently refused; correct behaviour)

If you see track-pull every frame, R is too low (or the mask is wrong).
If you see no track-pull ever during a known-bad trajectory, R is too high
or the mask is missing the area where the trajectory leaves the track.
"""