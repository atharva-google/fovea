"""Pass 3: Kinematic camera director.

Turns raw per-frame focal points into camera moves: stationary holds, smooth
cosine-eased pans, or instant cuts. Works for any crop width (any aspect ratio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import CropGeometry


def plan_camera(
    focal_df: pd.DataFrame,
    geometry: CropGeometry,
    pan_window_frames: int = 12,
    cut_threshold_px: float = 450.0,
    still_deadband_px: float = 45.0,
    spike_threshold_px: float = 200.0,
) -> pd.DataFrame:
    min_cx, max_cx = geometry.min_cx, geometry.max_cx
    records = []

    for shot_id, group in focal_df.groupby("shot_id", sort=False):
        group = group.sort_values("frame_idx")
        raw = group["target_cx"].to_numpy(dtype=np.float64)
        tags = group["tag"].to_numpy()
        frames = group["frame_idx"].to_numpy()
        N = len(raw)

        # Step 1: remove isolated 1-2 frame spikes (5-frame median filter)
        filtered = raw.copy()
        for i in range(N):
            med = np.median(raw[max(0, i - 2):min(N, i + 3)])
            if abs(raw[i] - med) > spike_threshold_px:
                filtered[i] = med

        # Step 2: decide camera action per frame
        cx = np.zeros(N, dtype=np.float64)
        modes = [""] * N
        cx[0] = float(np.clip(filtered[0], min_cx, max_cx))
        modes[0] = "shot_start_cut"

        i = 1
        while i < N:
            prev = cx[i - 1]
            target = float(np.clip(filtered[i], min_cx, max_cx))
            delta = target - prev

            if abs(delta) <= still_deadband_px:
                # A: deadband hold (no micro-wobble)
                cx[i] = prev
                modes[i] = "camera_hold"
                i += 1
            elif abs(delta) > cut_threshold_px and tags[i] != tags[i - 1] and "speaker" in str(tags[i]):
                # B: instant cut on a dialogue swap
                cx[i] = target
                modes[i] = "instant_cut"
                i += 1
            else:
                # C: smooth cosine ease-in/out pan
                t_window = min(pan_window_frames, N - i)
                if t_window <= 1:
                    cx[i] = target
                    modes[i] = "pan_step"
                    i += 1
                else:
                    steps = np.arange(1, t_window + 1)
                    ease = 0.5 * (1.0 - np.cos(np.pi * steps / t_window))
                    cx[i:i + t_window] = prev + delta * ease
                    modes[i:i + t_window] = ["smooth_pan"] * t_window
                    i += t_window

        cx = np.clip(cx, min_cx, max_cx)
        for idx in range(N):
            records.append({
                "shot_id": shot_id,
                "frame_idx": int(frames[idx]),
                "target_cx": round(float(raw[idx]), 2),
                "smoothed_cx": round(float(cx[idx]), 2),
                "camera_mode": modes[idx],
                "tag": tags[idx],
            })
    return pd.DataFrame(records)
