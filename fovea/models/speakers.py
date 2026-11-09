"""Active speaker detection: build TalkNet inputs from face tracks and merge scores back.

Flow:  face tracks  ->  ``build_talknet_inputs``  ->  ``talknet.score_tracks``
       ->  ``attach_speaking_scores`` (adds ``speaking`` / ``asd_score`` to detections)
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import python_speech_features
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from .tracker import ShotTracks


def build_talknet_inputs(
    video_path: str,
    audio_path: str,
    shot_tracks: ShotTracks,
    shots_df: pd.DataFrame,
    min_track_len: int = 10,
    crop_scale: float = 0.40,
) -> List[Dict[str, Any]]:
    # 1. Load audio once and resample to 16 kHz
    sr, audio = wavfile.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype not in (np.float32, np.float64):
        audio = audio.astype(np.float32)

    target_sr = 16000
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd)

    # 2. Global MFCC over the whole stream (100 Hz = 4 MFCC rows per 25fps video frame)
    global_mfcc = python_speech_features.mfcc(audio, samplerate=target_sr, numcep=13, winlen=0.025, winstep=0.010)

    processed: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))

    for _, shot in shots_df.iterrows():
        s_id = int(shot["shot_id"])
        if s_id not in shot_tracks:
            continue
        frame_tracks = shot_tracks[s_id]

        # Group face boxes by track id
        buckets: Dict[int, Dict[str, list]] = {}
        for frame_idx, detections in frame_tracks.items():
            for det in detections:
                if det.get("cname") != "person" or det.get("face_bbox") is None:
                    continue
                b = buckets.setdefault(det["track_id"], {"frames": [], "face_bboxes": []})
                b["frames"].append(frame_idx)
                b["face_bboxes"].append(det["face_bbox"])

        # Interpolate gaps for tracks long enough to score
        valid_tracks = {}
        needed_frames = set()
        for tid, data in buckets.items():
            if len(data["frames"]) < min_track_len:
                continue
            f_arr = np.array(data["frames"])
            b_arr = np.array(data["face_bboxes"])
            interp_frames = np.arange(f_arr[0], f_arr[-1] + 1)
            interp_bboxes = np.stack([
                interp1d(f_arr, b_arr[:, c], kind="linear", fill_value="extrapolate")(interp_frames)
                for c in range(4)
            ], axis=1)
            valid_tracks[tid] = {"frames": interp_frames, "bboxes": interp_bboxes}
            needed_frames.update(interp_frames.tolist())

        if not valid_tracks:
            continue

        # Read the shot once, keeping only frames we need
        s_start_f, s_end_f = int(shot["start_frame"]), int(shot["end_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_start_f)
        frame_cache = {}
        for curr_f in range(s_start_f, s_end_f):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if curr_f in needed_frames:
                frame_cache[curr_f] = frame

        for tid, data in valid_tracks.items():
            video_features = _crop_face_sequence(data["frames"], data["bboxes"], frame_cache, crop_scale)

            # 3. Frame-indexed slice into the global MFCC (strict 4:1 alignment)
            frames_seq = data["frames"]
            a_start, a_end = int(frames_seq[0]) * 4, (int(frames_seq[-1]) + 1) * 4
            mfcc_features = global_mfcc[a_start:a_end, :]

            v_len = min(video_features.shape[0], mfcc_features.shape[0] // 4)
            if v_len < min_track_len:
                continue

            processed.append({
                "shot_id": s_id,
                "track_id": tid,
                "frames": frames_seq[:v_len],
                "video_features": video_features[:v_len, :, :],
                "audio_features": mfcc_features[:v_len * 4, :],
            })

    cap.release()
    return processed


def _crop_face_sequence(frames_seq, bboxes_seq, frame_cache, crop_scale) -> np.ndarray:
    """Median-smoothed, padded 112x112 grayscale face crops (TalkNet input format)."""
    dets = {"x": [], "y": [], "s": []}
    for det in bboxes_seq:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2.0)
        dets["y"].append((det[1] + det[3]) / 2.0)
        dets["x"].append((det[0] + det[2]) / 2.0)

    k_size = 13 if len(bboxes_seq) >= 13 else (len(bboxes_seq) // 2 * 2 + 1)
    if k_size >= 3:
        for key in ("s", "x", "y"):
            dets[key] = signal.medfilt(dets[key], kernel_size=k_size)

    out = []
    for i, f_idx in enumerate(frames_seq):
        img = frame_cache.get(f_idx)
        if img is None:
            out.append(np.zeros((112, 112), dtype=np.uint8))
            continue

        bs = dets["s"][i]
        bsi = int(bs * (1 + 2 * crop_scale))
        padded = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = dets["y"][i] + bsi, dets["x"][i] + bsi

        face = padded[
            int(my - bs): int(my + bs * (1 + 2 * crop_scale)),
            int(mx - bs * (1 + crop_scale)): int(mx + bs * (1 + crop_scale)),
        ]
        face_224 = np.zeros((224, 224, 3), dtype=np.uint8) if face.size == 0 else cv2.resize(face, (224, 224))
        out.append(cv2.cvtColor(face_224, cv2.COLOR_BGR2GRAY)[56:168, 56:168])
    return np.array(out)


def attach_speaking_scores(shot_tracks: ShotTracks, scored_tracks: List[Dict[str, Any]]) -> ShotTracks:
    """Mark each person detection with ``speaking`` (bool) and ``asd_score`` (float|None)."""
    speech_lookup = {}
    for trk in scored_tracks:
        for f_idx, score in zip(trk["frames"], trk["scores"]):
            val = float(score)
            if val > 0.0:
                speech_lookup[(trk["shot_id"], int(f_idx), trk["track_id"])] = round(val, 3)

    for s_id, frames_data in shot_tracks.items():
        for f_idx, detections in frames_data.items():
            for det in detections:
                if det.get("cname") != "person":
                    continue
                key = (s_id, int(f_idx), det["track_id"])
                det["speaking"] = key in speech_lookup
                det["asd_score"] = speech_lookup.get(key)
    return shot_tracks
