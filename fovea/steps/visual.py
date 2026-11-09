"""STEP 2 - visual: understand what's on screen, for one shard (group of shots).

Each proxy frame is decoded ONCE (in a background thread) and handed to:
  * motion saliency  (every frame, downscaled)          -> models/motion.py
  * person + face tracker  (every Nth frame, batched)    -> models/tracker.py
  * OCR  (a few sampled frames per shot)                 -> models/ocr.py
Then TalkNet scores which face is speaking               -> models/speakers.py, models/talknet/
  This runs in a second thread NEXT TO tracking: as soon as a shot is tracked it is handed
  over, so speaker scoring of earlier shots overlaps with tracking of later ones, and face
  tracks from several shots are scored together in large GPU batches.

Reads:  prep/*
Writes: visual/shard_<i>.pkl   (several shards can run in parallel on different GPUs)
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import PipelineConfig
from ..models.motion import MotionAnalyzer
from ..models.ocr import TextReader, sample_frame_indices
from ..models.tracker import PersonTracker, ShotTracks, fill_skipped_frames
from ..utils.device import clean_gpu_memory
from ..utils.jobdir import JobDir
from ..utils.timing import Timings
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

    timings = Timings()
    t_start = time.perf_counter()

    # ---- load models
    with timings("load_models"):
        if cfg.face_backend == "pose":
            tracker = PersonTracker(str(ensure_for(cfg, "yolo_pose")), device, fps, "pose",
                                    conf=cfg.yolo_conf, iou=cfg.yolo_iou, detect_every=cfg.detect_every)
        else:
            tracker = PersonTracker(str(ensure_for(cfg, "yolo")), device, fps, "mediapipe",
                                    face_model_path=str(ensure_for(cfg, "blazeface")), conf=cfg.yolo_conf,
                                    iou=cfg.yolo_iou, face_min_conf=cfg.face_min_conf, detect_every=cfg.detect_every)
        motion = MotionAnalyzer(proxy_w, proxy_h, scale=cfg.motion_scale) if cfg.enable_motion else None
        reader = TextReader(str(ensure_for(cfg, "easyocr")), device, cfg.ocr_langs) if cfg.enable_ocr else None
    # Speaker detection runs NEXT TO tracking: each finished shot is handed to a background
    # thread, which scores it while the main loop tracks the following shots.
    speakers = (SpeakerScorer(proxy_path, audio_path, cfg, device, timings, log)
                if cfg.enable_speaker_detection else None)

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
            with timings("ocr"):
                ocr_records.extend(reader.read_shot(shot, ocr_frames))
        if speakers is not None:
            speakers.add_shot(sid, s0, s1, tracks[sid])

    # ---- the single decode loop
    frames = timings.timed_iter("decode_wait", iter_frames(proxy_path, int(shots[0]["start_frame"]),
                                                           int(shots[-1]["end_frame"])))
    try:
        for idx, frame in frames:
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
                with timings("motion"):
                    motion_res[idx] = motion.update(frame)
            if idx in ocr_targets:
                ocr_frames.append(frame)
            if (idx - int(shot["start_frame"])) % cfg.detect_every == 0:
                batch.append((idx, frame))
                if len(batch) >= cfg.detect_batch:
                    flush()
        finish_shot()
    except BaseException:
        if speakers is not None:
            speakers.abort()
        raise
    finally:
        frames.close()
    for name, secs in tracker.seconds.items():
        timings.add(name, secs)
    log(f"  tracked {len(tracks)} shots, {len(motion_res)} motion frames, {len(ocr_records)} text items "
        f"({time.perf_counter() - t_start:.1f}s)")

    del tracker, reader
    clean_gpu_memory()

    # ---- active speaker detection: wait for the shots still being scored
    speaker_scores: List[Dict[str, Any]] = []
    if speakers is not None:
        t_wait = time.perf_counter()
        speaker_scores = speakers.finish()
        log(f"  scored {len(speaker_scores)} face tracks with TalkNet "
            f"(+{time.perf_counter() - t_wait:.1f}s after tracking finished)")
        clean_gpu_memory()

    total = time.perf_counter() - t_start
    log(f"  visual timings ({total:.1f}s total; speaker work overlaps tracking): {timings.summary()}")
    return {"tracks": tracks, "motion": motion_res, "ocr": ocr_records, "speaker_scores": speaker_scores,
            "timings": {**timings.as_dict(), "total": round(total, 2)}}


class SpeakerScorer:
    """Active speaker detection (TalkNet) for finished shots, in ONE background thread.

    ``add_shot`` returns immediately; the thread builds the shot's face crops (with its own video
    reader) and collects them. Once enough face frames are waiting, they are scored together as
    big GPU batches, so tracks from several shots share the same TalkNet passes.
    """

    def __init__(self, proxy_path: str, audio_path: str, cfg: PipelineConfig, device: str,
                 timings: Timings, log=print):
        self.cfg, self.device, self.timings, self.log = cfg, device, timings, log
        self.flush_frames = cfg.talknet_batch_frames * 8  # score once this many face frames are waiting
        self.pending: List[Dict[str, Any]] = []
        self.pending_frames = 0
        self.scores: List[Dict[str, Any]] = []
        self.futures: List[Future] = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="talknet")
        self.futures.append(self.pool.submit(self._setup, proxy_path, audio_path))

    # ----------------------------------------------------------- main-thread API
    def add_shot(self, shot_id: int, start: int, end: int, frame_tracks: Dict[int, list]) -> None:
        self._raise_if_failed()
        self.futures.append(self.pool.submit(self._add_shot, shot_id, start, end, frame_tracks))

    def finish(self) -> List[Dict[str, Any]]:
        self.futures.append(self.pool.submit(self._finish))
        try:
            for f in self.futures:
                f.result()  # re-raises any error from the background thread
        finally:
            self.pool.shutdown(wait=True)
        return self.scores

    def abort(self) -> None:
        for f in self.futures:
            f.cancel()
        self.pool.shutdown(wait=True)

    def _raise_if_failed(self) -> None:
        for f in self.futures:
            if f.done() and f.exception() is not None:
                raise f.exception()

    # ------------------------------------------------------- background thread
    def _setup(self, proxy_path: str, audio_path: str) -> None:
        import cv2

        from ..models.speakers import load_mfcc
        from ..models.talknet import load_talknet

        with self.timings("talknet_setup"):
            self.cap = cv2.VideoCapture(proxy_path)
            self.mfcc = load_mfcc(audio_path)
            self.model = load_talknet(ensure_for(self.cfg, "talknet"), self.device)

    def _add_shot(self, shot_id: int, start: int, end: int, frame_tracks: Dict[int, list]) -> None:
        from ..models.speakers import build_shot_inputs

        with self.timings("speaker_crops"):
            inputs = build_shot_inputs(self.cap, self.mfcc, shot_id, start, end, frame_tracks)
        self.pending.extend(inputs)
        self.pending_frames += sum(len(t["frames"]) for t in inputs)
        if self.pending_frames >= self.flush_frames:
            self._score_pending()

    def _score_pending(self) -> None:
        if not self.pending:
            return
        from ..models.talknet import score_tracks

        with self.timings("talknet"):
            self.scores.extend(score_tracks(self.model, self.pending, self.device, self.cfg.talknet_batch_frames))
        self.pending, self.pending_frames = [], 0

    def _finish(self) -> None:
        try:
            self._score_pending()
        finally:
            if getattr(self, "cap", None) is not None:
                self.cap.release()
            self.model = None
