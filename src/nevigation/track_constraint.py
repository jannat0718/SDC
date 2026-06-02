"""
track_constraint.py
====================
Runtime module for querying the precomputed track mask.

Loads the artifacts produced by build_track_mask.py:
    track_drivable_mask.npy   bool[H, W] - True where kart can drive
    track_markings_mask.npy   bool[H, W] - True for on-track markings

Provides O(1) queries:
    is_drivable(x_m, y_m)        : is this world position on the track?
    nearest_drivable(x_m, y_m)   : closest on-track world position + distance

World coordinates are in metres. Conversion to pixel uses the same convention
as KFPipeline.world_to_pixel:
    px = origin_px[0] + wx * px_per_m
    py = origin_px[1] - wy * px_per_m         (y flipped)

For nearest-on-track, we precompute a distance transform + labels image so
the runtime query is a single array lookup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Geometry:
    """Cached geometry for nearest-drivable queries."""
    drivable: np.ndarray       # bool[H, W]
    markings: np.ndarray       # bool[H, W]
    dist_from_drivable: np.ndarray  # float32[H, W]
    nearest_labels: np.ndarray      # int32[H, W]
    drivable_coords: np.ndarray     # int32[N, 2] (px_x, px_y) for N drivable pixels


class TrackConstraint:
    """Query a precomputed track mask in world coordinates."""

    # Expected canonical resolution; the .npy mask MUST match this so that
    # the world_to_pixel math (using config px_per_meter + start_px) is valid.
    CANONICAL_W = 9360
    CANONICAL_H = 1410

    def __init__(self,
                 drivable_npy_path: str,
                 markings_npy_path: str,
                 px_per_m: float,
                 world_origin_px: Tuple[int, int],
                 expect_canonical: bool = True):
        self.px_per_m = float(px_per_m)
        self.origin_px = (int(world_origin_px[0]), int(world_origin_px[1]))

        if not os.path.exists(drivable_npy_path):
            raise FileNotFoundError(f"Drivable mask not found: {drivable_npy_path}")
        drivable = np.load(drivable_npy_path)
        if drivable.dtype != bool:
            drivable = drivable.astype(bool)
        if os.path.exists(markings_npy_path):
            markings = np.load(markings_npy_path)
            if markings.dtype != bool:
                markings = markings.astype(bool)
        else:
            markings = np.zeros_like(drivable, dtype=bool)
        if drivable.shape != markings.shape:
            raise ValueError(
                f"Drivable/markings shape mismatch: {drivable.shape} vs {markings.shape}")
        self.H, self.W = drivable.shape
        if expect_canonical and (self.W, self.H) != (self.CANONICAL_W, self.CANONICAL_H):
            print(f"[TrackConstraint] WARNING: mask is {self.W}x{self.H}, expected "
                  f"canonical {self.CANONICAL_W}x{self.CANONICAL_H}. "
                  f"World->pixel math assumes canonical scale; results may be off. "
                  f"Pass expect_canonical=False to silence this warning if you've "
                  f"already rescaled origin_px and px_per_m to match this mask.")

        # Precompute distance transform from drivable pixels, plus the
        # label image identifying the nearest drivable pixel for any off-track
        # query. distanceTransformWithLabels needs the 0-pixels to be the
        # ones we want labelled-by; we want labels for "what's the nearest TRUE
        # pixel in drivable", so we invert.
        not_drivable = (~drivable).astype(np.uint8) * 255
        # Where not_drivable is 0 (i.e. drivable pixels themselves), label = own index.
        # We need labels for *every* pixel pointing at the nearest drivable pixel.
        # The cleanest way: distanceTransformWithLabels on the inverted mask with
        # PIXEL labelType gives each pixel the index of the nearest "zero" pixel.
        # Zero pixels in `255 - not_drivable` (i.e. non_drivable pixels) -- that's
        # the opposite of what we want.
        # Use a different formulation: feed the inverse so zeros = drivable.
        drivable_as_zero = (~drivable).astype(np.uint8) * 255
        dist, labels = cv2.distanceTransformWithLabels(
            drivable_as_zero, cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        # labels[y, x] is the index of the nearest drivable pixel.
        # Build the inverse lookup: label -> (px, py).
        # The label values run 1..N where N = number of drivable pixels.
        ys, xs = np.where(drivable)
        # The labels assigned to the drivable pixels themselves tell us the mapping.
        # labels[ys, xs] gives each drivable pixel its own label.
        pixel_labels = labels[ys, xs]
        # Sort by label so we can index by label later.
        # Pre-allocate a lookup table sized to max label + 1.
        max_label = int(pixel_labels.max()) if pixel_labels.size > 0 else 0
        lookup = np.zeros((max_label + 1, 2), dtype=np.int32)
        lookup[pixel_labels, 0] = xs
        lookup[pixel_labels, 1] = ys

        self._geom = _Geometry(
            drivable=drivable,
            markings=markings,
            dist_from_drivable=dist.astype(np.float32),
            nearest_labels=labels.astype(np.int32),
            drivable_coords=lookup,
        )
        n_drivable = int(drivable.sum())
        print(f"[TrackConstraint] Loaded {self.W}x{self.H} mask, "
              f"{n_drivable} drivable px ({100*n_drivable/(self.W*self.H):.2f}% of image), "
              f"px_per_m={self.px_per_m}, origin_px={self.origin_px}")

    # ---- coordinate conversion -------------------------------------------
    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(round(self.origin_px[0] + x_m * self.px_per_m))
        py = int(round(self.origin_px[1] - y_m * self.px_per_m))
        return px, py

    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        x_m = (px - self.origin_px[0]) / self.px_per_m
        y_m = (self.origin_px[1] - py) / self.px_per_m
        return x_m, y_m

    # ---- queries ----------------------------------------------------------
    def is_drivable(self, x_m: float, y_m: float) -> bool:
        """True iff the world position is inside the drivable mask."""
        px, py = self.world_to_pixel(x_m, y_m)
        if 0 <= px < self.W and 0 <= py < self.H:
            return bool(self._geom.drivable[py, px])
        return False

    def is_marking(self, x_m: float, y_m: float) -> bool:
        """True iff the world position is on an on-track marking."""
        px, py = self.world_to_pixel(x_m, y_m)
        if 0 <= px < self.W and 0 <= py < self.H:
            return bool(self._geom.markings[py, px])
        return False

    def distance_to_drivable_m(self, x_m: float, y_m: float) -> float:
        """Distance from the world position to the nearest drivable pixel, in metres.
        Zero if the position is already drivable."""
        px, py = self.world_to_pixel(x_m, y_m)
        if not (0 <= px < self.W and 0 <= py < self.H):
            return float('inf')
        return float(self._geom.dist_from_drivable[py, px]) / self.px_per_m

    def nearest_drivable(self, x_m: float, y_m: float
                         ) -> Optional[Tuple[float, float, float]]:
        """Return (nx_m, ny_m, dist_m) of the nearest drivable position, or None
        if (x_m, y_m) is outside the image bounds."""
        px, py = self.world_to_pixel(x_m, y_m)
        if not (0 <= px < self.W and 0 <= py < self.H):
            return None
        if self._geom.drivable[py, px]:
            return x_m, y_m, 0.0
        lbl = int(self._geom.nearest_labels[py, px])
        if lbl == 0 or lbl >= len(self._geom.drivable_coords):
            return None
        nx_px, ny_px = self._geom.drivable_coords[lbl]
        if nx_px == 0 and ny_px == 0:
            return None
        nx_m, ny_m = self.pixel_to_world(int(nx_px), int(ny_px))
        dist_m = float(self._geom.dist_from_drivable[py, px]) / self.px_per_m
        return nx_m, ny_m, dist_m


# ---- factory helper for use from kalman_fusion ---------------------------

def load_from_nav_config(cfg: dict, base_dir: Optional[str] = None) -> TrackConstraint:
    """Construct a TrackConstraint from a nav_config dict.

    Expects (with defaults):
        cfg['track_image_path']       -> used to locate the artifact dir
        cfg['px_per_meter']           -> required
        cfg['start_px']               -> required (world origin pixel)
        cfg['track_mask_artifacts']   -> optional dict with paths
    """
    track_path = cfg.get('track_image_path')
    if base_dir is None:
        if track_path:
            base_dir = os.path.dirname(os.path.abspath(track_path))
        else:
            base_dir = os.getcwd()
    artifacts = cfg.get('track_mask_artifacts', {}) or {}
    drivable_name = artifacts.get('drivable', 'track_drivable_mask.npy')
    markings_name = artifacts.get('markings', 'track_markings_mask.npy')
    drivable_path = drivable_name if os.path.isabs(drivable_name) else os.path.join(base_dir, drivable_name)
    markings_path = markings_name if os.path.isabs(markings_name) else os.path.join(base_dir, markings_name)
    px_per_m = float(cfg['px_per_meter'])
    origin = tuple(cfg['start_px'])
    return TrackConstraint(drivable_path, markings_path, px_per_m, origin)
