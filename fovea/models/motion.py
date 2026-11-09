"""Frame-difference motion saliency: where on the X axis is the action happening?

Incremental: call :meth:`MotionAnalyzer.reset` at each shot start and
:meth:`MotionAnalyzer.update` once per frame (fed by the single decode loop).
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

MotionResults = Dict[int, Dict[str, float]]


class MotionAnalyzer:
    def __init__(self, width: int, height: int, scale: float = 0.5,
                 diff_thresh: int = 25, density_thresh: float = 0.12, decay_rate: float = 0.92):
        self.full_w = width
        self.scale = scale
        self.small = (max(16, int(width * scale)), max(16, int(height * scale)))
        k = max(1, int(min(self.small) * 0.12))
        self.kernel = np.ones((k, k), dtype=np.float32) / (k * k)
        self.diff_thresh = diff_thresh
        self.density_thresh = density_thresh
        self.decay_rate = decay_rate
        self.reset()

    def reset(self) -> None:
        """Call on hard cuts."""
        self.prev = None
        self.anchor_cx = self.full_w / 2.0
        self.anchor_weight = 0.0

    def update(self, frame: np.ndarray) -> Dict[str, float]:
        small = cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        if self.prev is None:
            self.prev = gray
            return {"motion_cx": round(self.anchor_cx, 2), "motion_weight": 0.0, "instant_density": 0.0}

        diff = cv2.absdiff(gray, self.prev)
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        density_map = cv2.filter2D(mask.astype(np.float32), -1, self.kernel)
        _, max_val, _, max_loc = cv2.minMaxLoc(density_map)
        instant_density = float(np.clip(max_val / 255.0, 0.0, 1.0))

        if instant_density >= self.density_thresh:
            self.anchor_cx = float(max_loc[0]) / self.scale  # back to full-res coords
            self.anchor_weight = min(1.0, instant_density * 1.5)
        else:
            self.anchor_weight = max(0.0, self.anchor_weight * self.decay_rate)

        self.prev = gray
        return {
            "motion_cx": round(self.anchor_cx, 2),
            "motion_weight": round(self.anchor_weight, 3),
            "instant_density": round(instant_density, 3),
        }
