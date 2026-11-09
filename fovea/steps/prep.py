"""STEP 1 - prep: get the video ready for analysis.

  * probe the original video
  * make a small 720p / 25fps silent copy ("proxy") that every later step analyses
  * extract 16 kHz audio
  * detect shots (scene cuts)
  * split the shots into groups ("shards") so the visual step can run on several GPUs

Reads:  input/<video>
Writes: prep/video_info.json, prep/proxy.mp4, prep/audio.wav, prep/shots.csv, prep/shards.json
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import pandas as pd

from ..config import PipelineConfig
from ..utils.jobdir import JobDir
from ..utils.video import extract_audio, has_audio_stream, make_proxy, make_silent_audio, probe_video


def run(job: JobDir, cfg: PipelineConfig, log=print) -> int:
    """Returns the number of shards."""
    if job.shards.exists() and not cfg.force:
        log("  prep already done, skipping")
        return job.read_json(job.shards)["num_shards"]

    source = job.source
    src = probe_video(source)
    log(f"  source {src.width}x{src.height} @ {src.fps:.2f}fps, {src.duration or 0:.1f}s")
    if cfg.max_duration_s and src.duration and src.duration > cfg.max_duration_s:
        raise ValueError(f"Video is {src.duration:.0f}s long; the limit is {cfg.max_duration_s:.0f}s")

    t = time.time()
    make_proxy(source, job.proxy, cfg.proxy_height, cfg.proxy_fps, cfg.use_gpu_video)
    has_audio = has_audio_stream(source)
    if has_audio:
        extract_audio(source, job.audio)
    else:
        make_silent_audio(job.audio, src.duration or 1.0)
    proxy = probe_video(job.proxy)
    log(f"  proxy {proxy.width}x{proxy.height} @ {proxy.fps:.0f}fps ({time.time() - t:.1f}s)")

    shots = detect_shots(str(job.proxy), cfg.shot_threshold)
    shots.to_csv(job.shots, index=False)
    shards = plan_shards(shots, cfg.max_shards, cfg.min_frames_per_shard)
    log(f"  {len(shots)} shots -> {len(shards)} shard(s)")

    job.write_json(job.video_info, {"source": src.to_dict(), "proxy": proxy.to_dict(), "has_audio": has_audio})
    job.write_json(job.shards, {"num_shards": len(shards), "shards": shards})  # written last = "prep done"
    return len(shards)


def detect_shots(video_path: str, threshold: float = 27.0) -> pd.DataFrame:
    """Scene cuts with PySceneDetect."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list() or [(video.base_timecode, video.duration)]

    return pd.DataFrame([{
        "shot_id": i,
        "start_frame": start.frame_num,
        "end_frame": end.frame_num,
        "duration_frame": end.frame_num - start.frame_num,
        "start_sec": start.seconds,
        "end_sec": end.seconds,
        "duration_sec": end.seconds - start.seconds,
    } for i, (start, end) in enumerate(scenes)])


def plan_shards(shots: pd.DataFrame, max_shards: int, min_frames_per_shard: int) -> List[Dict[str, Any]]:
    """Split consecutive shots into ~equal-frame groups (a shot is never split)."""
    rows = shots.sort_values("start_frame").to_dict("records")
    total = int(sum(r["end_frame"] - r["start_frame"] for r in rows))
    n = max(1, min(max_shards, len(rows), math.ceil(total / max(1, min_frames_per_shard))))
    target = total / n

    groups, current, acc = [], [], 0
    for r in rows:
        current.append(r)
        acc += r["end_frame"] - r["start_frame"]
        if acc >= target * (len(groups) + 1) and len(groups) < n - 1:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    return [{
        "index": i,
        "shot_ids": [int(r["shot_id"]) for r in g],
        "start_frame": int(g[0]["start_frame"]),
        "end_frame": int(g[-1]["end_frame"]),
    } for i, g in enumerate(groups)]
