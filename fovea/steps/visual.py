"""STEP 2 - visual: understand what's on screen, for one shard (group of shots).

Each proxy frame is decoded ONCE (in a background thread) and handed to:
  * motion saliency  (every frame, downscaled)          -> models/motion.py
  * person + face tracker  (every Nth frame, batched)    -> models/tracker.py
  * OCR  (a few sampled frames per shot)                 -> models/ocr.py
Then TalkNet scores which face is speaking               -> models/speakers.py, models/talknet/

Reads:  prep/*
Writes: visual/shard_<i>.pkl   (several shards can run in parallel on different GPUs)
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import PipelineConfig
from ..models.motion import MotionAnalyzer
from ..models.ocr import TextReader, sample_frame_indices
from ..models.tracker import PersonTracker, ShotTracks, fill_skipped_frames
from ..utils.device import clean_gpu_memory
from ..utils.jobdir import JobDir
from ..utils.video import iter_frames
from ..utils.weights import ensure_for


def run(job: JobDir, cfg: PipelineConfig, device: str, shard: int, log=print) -> None:
    out = job.shard_result(shard)
    if out.exists() and not cfg.force:
        log(f"  shard {shard} already done, skipping")
        return

    _, proxy, _ = job.video_infos()
    plan = job.read_json(job.shards)["shards"][shard]
    shots = pd.read_csv(job.shots)
    shots = shots[shots["shot_id"].isin(plan["shot_ids"])]

    result = analyze_shard(str(job.proxy), str(job.audio), shots, proxy.width, proxy.height, proxy.fps,
                           cfg, device, log)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.rename(out)


def analyze_shard(proxy_path: str, audio_path: str, shots_df: pd.DataFrame, proxy_w: int, proxy_h: int,
                  fps: float, cfg: PipelineConfig, device: str, log=print) -> Dict[str, Any]:
    shots = [s for s in shots_df.sort_values("start_frame").to_dict("records")
             if int(s["end_frame"]) > int(s["start_frame"])]
    if not shots:
        return {"tracks": {}, "motion": {}, "ocr": [], "speaker_scores": []}

    # ---- load models
    if cfg.face_backend == "pose":
        tracker = PersonTracker(str(ensure_for(cfg, "yolo_pose")), device, fps, "pose",
                                conf=cfg.yolo_conf, iou=cfg.yolo_iou, detect_every=cfg.detect_every)
    else:
        tracker = PersonTracker(str(ensure_for(cfg, "yolo")), device, fps, "mediapipe",
                                face_model_path=str(ensure_for(cfg, "blazeface")), conf=cfg.yolo_conf,
                                iou=cfg.yolo_iou, face_min_conf=cfg.face_min_conf, detect_every=cfg.detect_every)
    motion = MotionAnalyzer(proxy_w, proxy_h, scale=cfg.motion_scale) if cfg.enable_motion else None
    reader = TextReader(str(ensure_for(cfg, "easyocr")), device, cfg.ocr_langs) if cfg.enable_ocr else None

    tracks: ShotTracks = {}
    motion_res: Dict[int, Dict[str, float]] = {}
    ocr_records: List[Dict[str, Any]] = []

    # ---- per-shot state
    shot_i = -1
    shot: Optional[Dict[str, Any]] = None
    batch: list = []
    shot_frames: Dict[int, list] = {}
    ocr_targets: set = set()
    ocr_frames: list = []

    def flush():
        if batch:
            shot_frames.update(tracker.process_batch(batch))
            batch.clear()

    def finish_shot():
        if shot is None:
            return
        flush()
        sid, s0, s1 = int(shot["shot_id"]), int(shot["start_frame"]), int(shot["end_frame"])
        tracks[sid] = fill_skipped_frames(shot_frames, s0, s1) if cfg.detect_every > 1 else dict(shot_frames)
        if reader is not None and ocr_frames:
            ocr_records.extend(reader.read_shot(shot, ocr_frames))

    # ---- the single decode loop
    for idx, frame in iter_frames(proxy_path, int(shots[0]["start_frame"]), int(shots[-1]["end_frame"])):
        while shot is None or idx >= int(shot["end_frame"]):  # moved into the next shot?
            finish_shot()
            shot_i += 1
            if shot_i >= len(shots):
                shot = None
                break
            shot = shots[shot_i]
            tracker.new_shot()
            if motion is not None:
                motion.reset()
            shot_frames, ocr_frames = {}, []
            ocr_targets = set(sample_frame_indices(int(shot["start_frame"]), int(shot["end_frame"]),
                                                   cfg.ocr_samples_per_shot))
        if shot is None:
            break
        if idx < int(shot["start_frame"]):
            continue

        if motion is not None:
            motion_res[idx] = motion.update(frame)
        if idx in ocr_targets:
            ocr_frames.append(frame)
        if (idx - int(shot["start_frame"])) % cfg.detect_every == 0:
            batch.append((idx, frame))
            if len(batch) >= cfg.detect_batch:
                flush()
    finish_shot()
    log(f"  tracked {len(tracks)} shots, {len(motion_res)} motion frames, {len(ocr_records)} text items")

    del tracker, reader
    clean_gpu_memory()

    # ---- active speaker detection
    speaker_scores: List[Dict[str, Any]] = []
    if cfg.enable_speaker_detection:
        from ..models.speakers import build_talknet_inputs
        from ..models.talknet import load_talknet, score_tracks

        inputs = build_talknet_inputs(proxy_path, audio_path, tracks, shots_df)
        model = load_talknet(ensure_for(cfg, "talknet"), device)
        speaker_scores = score_tracks(model, inputs, device)
        del model
        clean_gpu_memory()
        log(f"  scored {len(speaker_scores)} face tracks with TalkNet")

    return {"tracks": tracks, "motion": motion_res, "ocr": ocr_records, "speaker_scores": speaker_scores}
