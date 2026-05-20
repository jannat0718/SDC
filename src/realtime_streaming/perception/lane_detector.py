from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class LaneDetector:
    """
    Lightweight lane detector placeholder.

    Replace this implementation with your trained lane model later.
    """

    def __init__(self, canny_low: int = 60, canny_high: int = 150):
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        roi = np.zeros_like(edges)
        polygon = np.array([[(0, h), (w, h), (int(0.7 * w), int(0.55 * h)), (int(0.3 * w), int(0.55 * h))]], dtype=np.int32)
        cv2.fillPoly(roi, polygon, 255)
        masked = cv2.bitwise_and(edges, roi)

        lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=80)
        if lines is None:
            return {"left": None, "right": None, "center_offset_px": None}

        left_lines = []
        right_lines = []

        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

        left = self._average_line(left_lines, h)
        right = self._average_line(right_lines, h)

        center_offset_px = None
        if left and right:
            left_x_bottom = left[0]
            right_x_bottom = right[0]
            lane_center = (left_x_bottom + right_x_bottom) / 2.0
            center_offset_px = int(lane_center - (w / 2.0))

        return {"left": left, "right": right, "center_offset_px": center_offset_px}

    @staticmethod
    def _average_line(lines, h: int):
        if not lines:
            return None

        xs = []
        ys = []
        for x1, y1, x2, y2 in lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        m, b = np.polyfit(xs, ys, 1)
        y_bottom = h - 1
        y_top = int(0.6 * h)

        if abs(m) < 1e-6:
            return None

        x_bottom = int((y_bottom - b) / m)
        x_top = int((y_top - b) / m)
        return (x_bottom, y_bottom, x_top, y_top)
