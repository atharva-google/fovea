"""Run the steps for one video.

    video ──┬── sound ── transcript (Whisper) ──┐
            └── proxy ── shots (PySceneDetect) ─┴── base analysis (one Gemini call)

The two branches run at the same time. A step whose output is already there is skipped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import analysis, media, shots, transcript
from .config import Settings
from .job import Job, read_json, write_json


def analyze(video: str | Path, settings: Optional[Settings] = None, api_key: Optional[str] = None,
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Shots + transcript, then the base analysis. Returns what ``analysis.run`` returns, plus ``"job"``."""
    settings = settings or Settings()
    job = Job.open(video, settings.data_dir)
    log(f"Fovea | {job.source.name} | folder: {job.root}")
    if settings.force or not job.video_info.exists():
        write_json(job.video_info, media.probe(job.source))
    info = read_json(job.video_info)
    log(f"  {info['width']}x{info['height']} @ {info['fps']:g} fps, {info['duration']:.1f}s"
        f"{'' if info['has_audio'] else ', no sound'}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        words_job = pool.submit(_transcript, job, settings, info["has_audio"], log)
        shots_job = pool.submit(_shots, job, settings, log)
        words, shot_list = words_job.result(), shots_job.result()

    log("▶ base analysis (Gemini)")
    return {**analysis.run(job, shot_list, words, settings, api_key, log), "job": job}


def _shots(job: Job, settings: Settings, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    if settings.force or not job.shots_csv.exists():
        if settings.force or not job.proxy.exists():
            media.make_proxy(job.source, job.proxy, settings.proxy_height, settings.proxy_fps)
        shots.write_csv(job.shots_csv, shots.detect(job.proxy, settings.shot_threshold))
    found = shots.read_csv(job.shots_csv)
    log(f"  [shots] {len(found)} shots -> {job.shots_csv}")
    return found


def _transcript(job: Job, settings: Settings, has_audio: bool, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    if settings.force or not job.transcript_csv.exists():
        words: List[Dict[str, Any]] = []
        if has_audio:
            media.extract_audio(job.source, job.audio_wav)
            words = transcript.transcribe(job.audio_wav, settings.whisper_model, settings.whisper_device,
                                          settings.language, settings.weights_dir)
        transcript.write_csv(job.transcript_csv, words)
    words = transcript.read_csv(job.transcript_csv)
    reach = f", reaching {words[-1]['end_s']:.1f}s" if words else ""
    log(f"  [transcript] {len(words)} words{reach} -> {job.transcript_csv}")
    for problem in transcript.problems(words):
        log(f"  [transcript] WARNING, it looks unreliable: {problem}")
    return words
