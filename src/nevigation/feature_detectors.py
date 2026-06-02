"""Feature detectors for Visual Odometry.

VisualOdometry (av_map.py) calls `make_detector(method)` and expects an object
exposing:
  - .name                       : str (for logging)
  - .detect(gray, mask=None)    : -> np.ndarray of shape (N, 1, 2) float32,
                                    suitable as prevPts for cv2.calcOpticalFlowPyrLK
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_METHOD = "gftt"


class GFTTDetector:
    name = "gftt"

    def __init__(self, max_corners: int = 500, quality_level: float = 0.01,
                 min_distance: int = 7, block_size: int = 7):
        self.params = dict(maxCorners=max_corners, qualityLevel=quality_level,
                           minDistance=min_distance, blockSize=block_size)

    def detect(self, gray: np.ndarray, mask=None) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.params)
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)


class ORBDetector:
    name = "orb"

    def __init__(self, n_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def detect(self, gray: np.ndarray, mask=None) -> np.ndarray:
        kps = self.orb.detect(gray, mask)
        if not kps:
            return np.empty((0, 1, 2), dtype=np.float32)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts.reshape(-1, 1, 2)


def make_detector(method: str | None = None):
    method = (method or DEFAULT_METHOD).lower()
    if method == "orb":
        return ORBDetector()
    if method == "gftt":
        return GFTTDetector()
    raise ValueError(f"Unknown feature detector '{method}' (expected 'orb' or 'gftt')")
