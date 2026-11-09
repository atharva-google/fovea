"""Debug visualisations: annotated tracking boxes and motion-saliency overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import pandas as pd


def render_tracking_debug(video_path: str, shot_tracks: Dict, shots_df: pd.DataFrame, output_path: str | Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for _, shot in shots_df.iterrows():
        shot_id = int(shot["shot_id"])
        start_frame, end_frame = int(shot["start_frame"]), int(shot["end_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_tracks = shot_tracks.get(shot_id, {})

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Shot: #{shot_id} | Frame: {frame_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            for trk in frame_tracks.get(frame_idx, []):
                x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
                cname, tid = trk["cname"], trk["track_id"]
                is_speaking = trk.get("speaking", False)
                asd_score = trk.get("asd_score")

                # Speaking person = green, silent person = orange, objects = magenta
                if cname == "person":
                    color = (0, 255, 0) if is_speaking else (0, 140, 255)
                    label = f"Speaker #{tid} ({asd_score:+.2f})" if is_speaking else f"Person #{tid}"
                else:
                    color = (255, 0, 255)
                    label = f"{cname} #{tid}" if tid is not None else cname

                if cname == "person" and trk.get("face_bbox") is not None:
                    fx1, fy1, fx2, fy2 = [int(v) for v in trk["face_bbox"]]
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0) if is_speaking else (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            out.write(frame)

    cap.release()
    out.release()
    return Path(output_path)


def render_motion_debug(video_path: str, motion_results: Dict, shots_df: pd.DataFrame, output_path: str | Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_box = max(15, int(min(w, h) * 0.12)) // 2  # matches the 12% motion kernel
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for _, shot in shots_df.iterrows():
        shot_id = int(shot["shot_id"])
        start_frame, end_frame = int(shot["start_frame"]), int(shot["end_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            m = motion_results.get(frame_idx, {"motion_cx": w / 2.0, "motion_weight": 0.0, "instant_density": 0.0})
            cx = int(round(m["motion_cx"]))
            weight, density = m["motion_weight"], m["instant_density"]

            # Active movement = cyan, decaying hold = orange
            color = (0, 255, 255) if density > 0.12 else (0, 165, 255)
            cv2.line(frame, (cx, 0), (cx, h), color, 2)
            cv2.circle(frame, (cx, h // 2), 10, color, -1)
            cv2.rectangle(frame, (max(0, cx - half_box), max(0, h // 2 - half_box)),
                          (min(w, cx + half_box), min(h, h // 2 + half_box)), color, 2)
            cv2.putText(frame, f"Shot #{shot_id} | F: {frame_idx} | Cx: {cx} | Weight: {weight:.2f} | Density: {density:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out.write(frame)

    cap.release()
    out.release()
    return Path(output_path)
