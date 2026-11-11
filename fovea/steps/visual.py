"""STEP 2 - visual: understand what's on screen, for one shard (group of shots).

The shard is processed STAGE BY STAGE. Each stage works on many shots at the same time and
finishes all of them before the next stage starts; its GPU model is freed before the next loads:

  1. tracking   person + face tracker (YOLO + ByteTrack + face)   -> models/tracker.py
                several decoder threads, each on a different run of shots, feed ONE shared YOLO
                model with batches that mix shots; detections go to a ByteTrack per shot
  2. speakers   TalkNet active speaker scores                      -> models/speakers.py, models/talknet/
                face crops built by a thread pool (one reader per thread), scored in big GPU batches
  3. OCR        on-screen text on a few sampled frames per shot    -> models/ocr.py
                thread pool over shots, one shared EasyOCR model
  4. motion     frame-difference motion saliency                   -> models/motion.py
                thread pool over runs of shots (CPU)

Reads:  prep/*
Writes: visual/shard_<i>.pkl   (several shards can run in parallel on different GPUs)
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import pandas as pd

from ..config import PipelineConfig
from ..models.motion import analyze_shots as analyze_motion
from ..models.ocr import TextReader
from ..models.tracker import PersonTracker, ShotTracks, fill_skipped_frames
from ..utils.device import clean_gpu_memory
from ..utils.jobdir import JobDir
from ..utils.progress import Counter, Progress, noop
from ..utils.timing import Timings
from ..utils.video import ThreadLocalReaders, auto_workers
from ..utils.weights import ensure_for


def run(job: JobDir, cfg: PipelineConfig, device: str, shard: int, log=print,
        progress: Optional[Progress] = None) -> None:
    out = job.shard_result(shard)
    if out.exists() and not cfg.force:
        log(f"  shard {shard} already done, skipping")
        return

    _, proxy, _ = job.video_infos()
    plan = job.read_json(job.shards)["shards"][shard]
    shots = pd.read_csv(job.shots)
    shots = shots[shots["shot_id"].isin(plan["shot_ids"])]

    result = analyze_shard(str(job.proxy), str(job.audio), shots, proxy.width, proxy.height, proxy.fps,
                           cfg, device, log, progress)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.rename(out)


def analyze_shard(proxy_path: str, audio_path: str, shots_df: pd.DataFrame, proxy_w: int, proxy_h: int,
                  fps: float, cfg: PipelineConfig, device: str, log=print,
                  progress: Optional[Progress] = None) -> Dict[str, Any]:
    """``progress(stage, done=, total=, unit=, index=, count=)`` is called as each stage advances."""
    shots = [s for s in shots_df.sort_values("start_frame").to_dict("records")
             if int(s["end_frame"]) > int(s["start_frame"])]
    if not shots:
        return {"tracks": {}, "motion": {}, "ocr": [], "speaker_scores": []}

    timings = Timings()
    t_start = time.perf_counter()
    workers = auto_workers(cfg.visual_workers)
    ocr_workers = auto_workers(cfg.ocr_workers)
    motion_workers = auto_workers(cfg.motion_workers, cap=8)
    log(f"  {len(shots)} shots | workers: visual={workers} ocr={ocr_workers} motion={motion_workers}")
    progress = progress or noop
    stages = ["tracking"] + (["speakers"] if cfg.enable_speaker_detection else []) \
        + (["ocr"] if cfg.enable_ocr else []) + (["motion"] if cfg.enable_motion else [])

    def counter(stage: str, total: int, unit: str) -> Counter:
        return Counter(progress, stage, total, unit, stages.index(stage) + 1, len(stages))

    # ---- 1. tracking
    with _stage("1/4 tracking", timings, log):
        every = cfg.detect_every
        n_detect = sum(-(-(int(s["end_frame"]) - int(s["start_frame"])) // every) for s in shots)
        tracks = _track(proxy_path, shots, fps, cfg, device, workers, timings,
                        counter("tracking", n_detect, "frames").add)
        log(f"    {sum(1 for t in tracks.values() for d in t.values() if d)} frames with people")

    # ---- 2. active speaker detection
    speaker_scores: List[Dict[str, Any]] = []
    if cfg.enable_speaker_detection:
        with _stage("2/4 speakers", timings, log):
            speaker_scores = _score_speakers(proxy_path, audio_path, shots, tracks, cfg, device, workers, timings,
                                             counter("speakers", len(shots), "shots").add)
            log(f"    scored {len(speaker_scores)} face tracks")

    # ---- 3. OCR
    ocr_records: List[Dict[str, Any]] = []
    if cfg.enable_ocr:
        with _stage("3/4 ocr", timings, log):
            with timings("ocr_load"):
                reader = TextReader(str(ensure_for(cfg, "easyocr")), device, cfg.ocr_langs,
                                    serialize=cfg.ocr_serialize)
            ocr_records = reader.read_shots(proxy_path, shots, cfg.ocr_samples_per_shot, ocr_workers,
                                            counter("ocr", len(shots), "shots").add)
            del reader
            clean_gpu_memory()
            log(f"    {len(ocr_records)} text items")

    # ---- 4. motion
    motion_res: Dict[int, Dict[str, float]] = {}
    if cfg.enable_motion:
        with _stage("4/4 motion", timings, log):
            n_frames = sum(int(s["end_frame"]) - int(s["start_frame"]) for s in shots)
            motion_res = analyze_motion(proxy_path, shots, proxy_w, proxy_h, cfg.motion_scale, motion_workers,
                                        counter("motion", n_frames, "frames").add)
            log(f"    {len(motion_res)} motion frames")

    total = time.perf_counter() - t_start
    log(f"  visual timings ({total:.1f}s total; per-part times inside a stage overlap): {timings.summary()}")
    return {"tracks": tracks, "motion": motion_res, "ocr": ocr_records, "speaker_scores": speaker_scores,
            "timings": {**timings.as_dict(), "total": round(total, 2)}}


class _stage:
    """``with _stage("1/4 tracking", timings, log):`` -> logs and records the stage's wall time."""

    def __init__(self, name: str, timings: Timings, log):
        self.name, self.timings, self.log = name, timings, log

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        self.timings.add(f"stage_{self.name.split(' ', 1)[1]}", secs)
        if exc[0] is None:
            self.log(f"  ✓ {self.name} ({secs:.1f}s)")
        return False


# --------------------------------------------------------------------------- 1. tracking
def _track(proxy_path: str, shots: List[Dict[str, Any]], fps: float, cfg: PipelineConfig, device: str,
           workers: int, timings: Timings, on_frames: Optional[Callable[[int], None]] = None) -> ShotTracks:
    with timings("tracker_load"):
        if cfg.face_backend == "pose":
            tracker = PersonTracker(str(ensure_for(cfg, "yolo_pose")), device, fps, "pose",
                                    conf=cfg.yolo_conf, iou=cfg.yolo_iou, detect_every=cfg.detect_every)
        else:
            tracker = PersonTracker(str(ensure_for(cfg, "yolo")), device, fps, "mediapipe",
                                    face_model_path=str(ensure_for(cfg, "blazeface")), conf=cfg.yolo_conf,
                                    iou=cfg.yolo_iou, face_min_conf=cfg.face_min_conf, detect_every=cfg.detect_every)
    try:
        raw = tracker.track_shots(proxy_path, shots, batch_size=cfg.detect_batch, decoders=workers,
                                  face_workers=workers, on_frames=on_frames)
    finally:
        for name, secs in tracker.seconds.items():
            timings.add(name, secs)
        del tracker
        clean_gpu_memory()

    tracks: ShotTracks = {}
    for shot in shots:
        sid, s0, s1 = int(shot["shot_id"]), int(shot["start_frame"]), int(shot["end_frame"])
        frames = raw.get(sid, {})
        tracks[sid] = fill_skipped_frames(frames, s0, s1) if cfg.detect_every > 1 else dict(frames)
    return tracks


# --------------------------------------------------------------------------- 2. speakers
def _score_speakers(proxy_path: str, audio_path: str, shots: List[Dict[str, Any]], tracks: ShotTracks,
                    cfg: PipelineConfig, device: str, workers: int, timings: Timings,
                    on_shot: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
    """Face crops for many shots at once (thread pool), scored by TalkNet in large GPU batches.

    Crops are scored once ``talknet_batch_frames * 8`` face frames are waiting, so memory stays
    bounded on long shards; the pool keeps building the next shots' crops meanwhile.
    """
    from ..models.speakers import build_shot_inputs, load_mfcc
    from ..models.talknet import load_talknet, score_tracks

    with timings("talknet_setup"):
        mfcc = load_mfcc(audio_path)
        model = load_talknet(ensure_for(cfg, "talknet"), device)

    readers = ThreadLocalReaders(proxy_path)
    flush_frames = cfg.talknet_batch_frames * 8
    scores: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    pending_frames = 0

    def crops(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        sid = int(shot["shot_id"])
        t0 = time.perf_counter()
        try:
            return build_shot_inputs(readers.get(), mfcc, sid, int(shot["start_frame"]), int(shot["end_frame"]),
                                     tracks.get(sid, {}))
        finally:
            timings.add("speaker_crops_cpu", time.perf_counter() - t0)

    def score() -> None:
        nonlocal pending, pending_frames
        if pending:
            with timings("talknet"):
                scores.extend(score_tracks(model, pending, device, cfg.talknet_batch_frames))
        pending, pending_frames = [], 0

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="talknet-crops") as pool:
            for inputs in _bounded_map(pool, crops, shots, window=workers * 2):
                pending.extend(inputs)
                pending_frames += sum(len(t["frames"]) for t in inputs)
                if on_shot:
                    on_shot()
                if pending_frames >= flush_frames:
                    score()
        score()
    finally:
        readers.close()
        del model
        clean_gpu_memory()
    return scores


def _bounded_map(pool: ThreadPoolExecutor, fn: Callable, items: Iterable, window: int) -> Iterator:
    """Like ``pool.map`` (results in order) but with at most ``window`` tasks in flight."""
    it = iter(items)
    futures = []
    for item in it:
        futures.append(pool.submit(fn, item))
        if len(futures) >= window:
            break
    while futures:
        result = futures.pop(0).result()
        nxt = next(it, None)
        if nxt is not None:
            futures.append(pool.submit(fn, nxt))
        yield result
