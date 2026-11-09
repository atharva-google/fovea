"""Alternative focal strategy: hand-written priority rules (the notebook's first approach).

Select with ``--focal-strategy rules``. Produces the same columns as the Viterbi
strategy (``frame_idx, target_cx, tag, shot_id``) so the camera planner works
with either.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .focal import parse_ocr_entries


def _cx(det: Dict[str, Any]) -> float:
    box = det.get("face_bbox") or det["bbox"]
    return (box[0] + box[2]) / 2.0


def find_focal_points_rules(
    shots_df: pd.DataFrame,
    shot_tracks: Dict[int, Dict[int, List[Dict[str, Any]]]],
    ocr_df: pd.DataFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_width: int,
    fps: float = 25.0,
    hangover_frames: int = 10,
    motion_hold_frames: int = 8,
) -> pd.DataFrame:
    records = []
    ocr_by_frame = parse_ocr_entries(ocr_df, fps, video_width)
    default_cx = video_width / 2.0

    for _, shot in shots_df.iterrows():
        s_id = int(shot["shot_id"])
        frames_data = shot_tracks.get(s_id, {})

        last_speaker_tid = None
        hangover_remaining = 0
        last_valid_cx = default_cx
        last_action_cx = None
        action_hold_remaining = 0

        for f_idx in range(int(shot["start_frame"]), int(shot["end_frame"])):
            persons = [d for d in frames_data.get(f_idx, []) if d.get("cname") == "person"]
            active_speakers = [p for p in persons if p.get("speaking") is True]
            target_cx, reason, speaker_tid = None, "", None

            # Level 1: exactly one person on screen
            if len(persons) == 1:
                p = persons[0]
                target_cx, reason = _cx(p), "single_person"
                if p.get("speaking") is True:
                    last_speaker_tid = speaker_tid = p["track_id"]
                    hangover_remaining = hangover_frames
                elif hangover_remaining > 0:
                    hangover_remaining -= 1

            # Level 2: multiple persons on screen
            elif len(persons) > 1:
                if active_speakers:
                    spk = max(active_speakers, key=lambda x: x.get("asd_score") or 0.0)
                    last_speaker_tid = speaker_tid = spk["track_id"]
                    hangover_remaining = hangover_frames
                    target_cx, reason = _cx(spk), "active_speaker"
                elif hangover_remaining > 0 and last_speaker_tid is not None:
                    held = next((p for p in persons if p["track_id"] == last_speaker_tid), None)
                    if held is not None:
                        hangover_remaining -= 1
                        target_cx, reason, speaker_tid = _cx(held), "speaker_hangover", last_speaker_tid
                    else:
                        hangover_remaining = 0

                if target_cx is None:  # no speech -> largest person
                    largest = max(persons, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))
                    target_cx, reason = _cx(largest), "salient_person_fallback"

            # Level 3: zero persons (motion > OCR > inertia)
            else:
                hangover_remaining, last_speaker_tid = 0, None
                motion = motion_results.get(f_idx, {})
                density = motion.get("instant_density", 0.0)
                m_weight = motion.get("motion_weight", 0.0)
                mcx = motion.get("motion_cx", default_cx)

                if density >= 0.10 or m_weight >= 0.20:
                    target_cx = last_action_cx = mcx
                    action_hold_remaining = motion_hold_frames
                    reason = "action_takeover"
                elif action_hold_remaining > 0 and last_action_cx is not None:
                    target_cx = last_action_cx
                    action_hold_remaining -= 1
                    reason = "action_hold"
                elif f_idx in ocr_by_frame:
                    target_cx = max(ocr_by_frame[f_idx], key=lambda b: b["area"])["cx"]
                    reason = "zero_person_ocr"
                else:
                    target_cx, reason = last_valid_cx, "zero_person_inertia"

            last_valid_cx = target_cx
            records.append({
                "shot_id": s_id,
                "frame_idx": f_idx,
                "target_cx": round(target_cx, 2),
                "speaker_tid": speaker_tid,
                "reason": reason,
                # stable tag so the camera planner can detect subject switches
                "tag": f"speaker_{speaker_tid}" if speaker_tid is not None else reason,
            })
    return pd.DataFrame(records)
