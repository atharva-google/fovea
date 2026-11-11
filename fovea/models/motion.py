"""Frame-difference motion saliency: where on the X axis is the action happening?

Incremental: call :meth:`MotionAnalyzer.reset` at each shot start and
:meth:`MotionAnalyzer.update` once per frame. ``analyze_shots`` runs many shots in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from ..utils.video import ShotReader, split_contiguous

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


def analyze_shots(video_path: str, shots: Sequence[Dict[str, Any]], width: int, height: int,
                  scale: float = 0.5, workers: int = 2) -> MotionResults:
    """Motion saliency for many shots at once.

    Shots are split into runs of consecutive shots; a thread pool processes the runs, each task
    with its own video reader and analyzer. The analyzer is reset at every cut anyway, so the
    results are identical to processing the shots one after another. OpenCV releases the GIL,
    so threads use several CPU cores.
    """
    shots = sorted(shots, key=lambda s: int(s["start_frame"]))
    groups = split_contiguous(shots, max(1, workers) * 4)

    def run_group(group: List[Dict[str, Any]]) -> MotionResults:
        reader = ShotReader(video_path)
        analyzer = MotionAnalyzer(width, height, scale=scale)
        out: MotionResults = {}
        try:
            for shot in group:
                analyzer.reset()
                for idx, frame in reader.frames(int(shot["start_frame"]), int(shot["end_frame"])):
                    out[idx] = analyzer.update(frame)
        finally:
            reader.close()
        return out

    results: MotionResults = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="motion") as pool:
        for part in pool.map(run_group, groups):
            results.update(part)
    return dict(sorted(results.items()))
