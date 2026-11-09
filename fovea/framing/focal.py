"""Pass 1 + 2: find the focal X coordinate per frame with a Viterbi path search.

Each frame produces *candidates* (speakers, persons, OCR text, motion). A
dynamic-programming search picks the sequence of candidates that balances
"interesting" (high score) against "stable" (small jumps, few subject switches).
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

OcrByFrame = Dict[int, List[Dict[str, float]]]


def parse_ocr_entries(ocr_df: pd.DataFrame, fps: float, video_w: int) -> OcrByFrame:
    """Expand per-shot OCR rows into a {frame_idx: [boxes]} lookup."""

    def safe_parse_vertices(val):
        if isinstance(val, (list, tuple)):
            return val
        if isinstance(val, str):
            val_clean = val.strip()
            try:
                return ast.literal_eval(val_clean)
            except (ValueError, SyntaxError):
                matches = re.findall(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", val_clean)
                if matches:
                    return [(float(x), float(y)) for x, y in matches]
        return []

    ocr_by_frame: OcrByFrame = {}
    for _, row in ocr_df.iterrows():
        f_start = int(round(float(row["start_s"]) * fps))
        f_end = int(round(float(row["end_s"]) * fps))

        vertices = safe_parse_vertices(row["text_box_vertices"])
        if not vertices or len(vertices) < 2:
            continue

        xs = [v[0] * video_w for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        box_w = max_x - min_x
        box_h = max(ys) - min(ys)

        entry = {
            "cx": (min_x + max_x) / 2.0,
            "box_w": box_w,
            "area": box_w * box_h,
            "confidence": float(row.get("confidence", 1.0)),
        }
        for f in range(f_start, f_end + 1):
            ocr_by_frame.setdefault(f, []).append(entry)
    return ocr_by_frame


def extract_frame_candidates(
    f_idx: int,
    shot_rel_idx: int,
    frames_data: Dict[int, List[Dict[str, Any]]],
    ocr_by_frame: OcrByFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_w: int,
    video_h: int,
) -> List[Dict[str, Any]]:
    candidates = []
    frame_area = float(video_w * video_h)

    # 1. Active speakers & visible persons
    for det in frames_data.get(f_idx, []):
        if det.get("cname") != "person":
            continue
        tid = det["track_id"]
        box = det.get("face_bbox") or det["bbox"]
        cx = (box[0] + box[2]) / 2.0
        area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
        area_ratio = area / frame_area

        if det.get("speaking") is True:
            score = 2.0 + max(0.0, det.get("asd_score") or 0.0)
            tag = f"speaker_{tid}"
        else:
            score = 0.60 + 0.30 * min(1.0, area_ratio / 0.25)
            tag = f"person_{tid}"
        candidates.append({"cx": cx, "score": score, "tag": tag, "type": "person"})

    # 2. Prominent OCR (high for the first ~0.5s of a shot, then decays once read)
    if f_idx in ocr_by_frame:
        read_decay = 1.0 if shot_rel_idx < 12 else max(0.15, 1.0 - (shot_rel_idx - 12) * 0.08)
        for box in ocr_by_frame[f_idx]:
            candidates.append({"cx": box["cx"], "score": 0.85 * read_decay, "tag": "ocr_text", "type": "ocr"})

    # 3. Motion saliency (action peak)
    motion = motion_results.get(f_idx, {})
    density = motion.get("instant_density", 0.0)
    weight = motion.get("motion_weight", 0.0)
    mcx = motion.get("motion_cx")

    if mcx is not None and (density >= 0.10 or weight >= 0.15):
        # Ignore edge artifacts (keep inside 5% - 95% width)
        if 0.05 * video_w <= mcx <= 0.95 * video_w:
            has_speaker = any(c["type"] == "person" and "speaker" in c["tag"] for c in candidates)
            # Physical action in zero-person shots commands the highest saliency
            motion_score = 0.50 if has_speaker else 1.50 * (density + 0.5 * weight)
            candidates.append({"cx": mcx, "score": motion_score, "tag": "action_motion", "type": "motion"})

    if not candidates:
        candidates.append({"cx": video_w / 2.0, "score": 0.20, "tag": "frame_center", "type": "fallback"})
    return candidates


def optimize_shot_focal_path(
    shot_frames: List[int],
    frames_data: Dict[int, List[Dict[str, Any]]],
    ocr_by_frame: OcrByFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_w: int,
    video_h: int,
    max_jump_cost: float = 4.0,
    switch_penalty: float = 1.2,
) -> List[Dict[str, Any]]:
    T = len(shot_frames)
    if T == 0:
        return []

    frame_candidates = [
        extract_frame_candidates(f, t, frames_data, ocr_by_frame, motion_results, video_w, video_h)
        for t, f in enumerate(shot_frames)
    ]

    dp_cost = [[-np.log(max(1e-4, c["score"])) for c in frame_candidates[0]]]
    backpointer = [[-1] * len(frame_candidates[0])]

    for t in range(1, T):
        prev_cands, curr_cands = frame_candidates[t - 1], frame_candidates[t]
        t_costs, t_bp = [], []
        for curr_c in curr_cands:
            obs_cost = -np.log(max(1e-4, curr_c["score"]))
            best_prev_idx, best_total = -1, float("inf")
            for prev_idx, prev_c in enumerate(prev_cands):
                # Linear transition cost capped at max_jump_cost
                norm_dx = abs(curr_c["cx"] - prev_c["cx"]) / float(video_w)
                trans_cost = min(max_jump_cost, 6.0 * norm_dx)
                if curr_c["tag"] != prev_c["tag"]:
                    trans_cost += switch_penalty
                total = dp_cost[t - 1][prev_idx] + obs_cost + trans_cost
                if total < best_total:
                    best_total, best_prev_idx = total, prev_idx
            t_costs.append(best_total)
            t_bp.append(best_prev_idx)
        dp_cost.append(t_costs)
        backpointer.append(t_bp)

    # Backtrack
    path = [int(np.argmin(dp_cost[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(backpointer[t][path[-1]])
    path.reverse()

    return [
        {
            "frame_idx": shot_frames[t],
            "target_cx": round(frame_candidates[t][ci]["cx"], 2),
            "tag": frame_candidates[t][ci]["tag"],
            "type": frame_candidates[t][ci]["type"],
        }
        for t, ci in enumerate(path)
    ]


def find_focal_points_viterbi(
    shots_df: pd.DataFrame,
    shot_tracks: Dict[int, Dict[int, List[Dict[str, Any]]]],
    ocr_df: pd.DataFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_width: int,
    video_height: int,
    fps: float = 25.0,
    max_jump_cost: float = 4.0,
    switch_penalty: float = 1.2,
) -> pd.DataFrame:
    ocr_by_frame = parse_ocr_entries(ocr_df, fps, video_width)
    records = []
    for _, shot in shots_df.iterrows():
        s_id = int(shot["shot_id"])
        shot_frames = list(range(int(shot["start_frame"]), int(shot["end_frame"])))
        for entry in optimize_shot_focal_path(
            shot_frames, shot_tracks.get(s_id, {}), ocr_by_frame, motion_results,
            video_width, video_height, max_jump_cost, switch_penalty,
        ):
            entry["shot_id"] = s_id
            records.append(entry)
    return pd.DataFrame(records)
