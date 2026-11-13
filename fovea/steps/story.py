"""STEP story (only for cutdowns): units + story map for this video, from the store or from Gemini.

  * fingerprint the video file and look in the store: if units + story map are there, use them
    (no analysis outputs, no Gemini call and no key needed)
  * otherwise build the units from the analysis (shots, transcript, speakers, people, on-screen
    text), ask Gemini ONCE for the story map, and store both under the fingerprint

Reads:  input/<video>, and on a store miss: prep/*, visual/*, audio/transcript.json
Writes: <story dir>/<fingerprint>/units.json, story_map.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..config import PipelineConfig
from ..story.store import StoryStore, default_root, fingerprint
from ..story.story_map import request_story_map
from ..story.units import FactsAt, Unit, build_units
from ..utils.jobdir import JobDir


class StoryKeyRequired(RuntimeError):
    """No story map is stored for this video yet, and no Gemini key was given to make one."""


def story_store(cfg: PipelineConfig) -> StoryStore:
    return StoryStore(default_root(cfg))


def lookup(video_path: str | Path, cfg: PipelineConfig) -> Tuple[str, Optional[List[Unit]], Optional[Dict[str, Any]]]:
    """``(fingerprint, units, record)``; units and record are None when nothing usable is stored
    (or ``cfg.refresh_story`` is set)."""
    fp = fingerprint(video_path)
    found = None if cfg.refresh_story else story_store(cfg).load(fp)
    return (fp, found[0], found[1]) if found else (fp, None, None)


def run(job: JobDir, cfg: PipelineConfig, api_key: Optional[str] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Returns ``{"fingerprint", "units", "record", "cached"}``; ``record["story_map"]`` is the map."""
    fp, units, record = lookup(job.source, cfg)
    if units is not None:
        log(f"  story map found in the store ({fp[:12]}), no Gemini call")
        return {"fingerprint": fp, "units": units, "record": record, "cached": True}
    if not api_key:
        raise StoryKeyRequired("Gemini key required for the first cutdown of this video "
                               "(no story map is stored for it yet)")

    shots, words, facts_at = load_analysis(job, cfg)
    units = build_units(shots, words, facts_at)
    if not units:
        raise RuntimeError("Could not build any units from this video (no shots found)")
    log(f"  {len(units)} units ({sum(u.kind == 'speech' for u in units)} speech, "
        f"{sum(u.kind == 'picture' for u in units)} picture) from {len(shots)} shots and {len(words)} words")

    result = request_story_map(job.source, units, api_key, cfg.gemini_model, cfg.gemini_video_processing, log)
    folder = story_store(cfg).save(fp, units, result, job.source.name)
    log(f"  story map stored in {folder}")
    record = {"fingerprint": fp, "source": job.source.name, **result}
    return {"fingerprint": fp, "units": units, "record": record, "cached": False}


def load_analysis(job: JobDir, cfg: PipelineConfig) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]], FactsAt]:
    """Shots, transcript words, and a ``facts_at(start, end, shot)`` lookup, from the finished
    prep / visual / audio steps."""
    from ..models.speakers import attach_speaking_scores
    from .render import merge_shards

    transcript = job.audio_dir / "transcript.json"
    if not job.shards.exists():
        raise FileNotFoundError("The prep step has not run for this job")
    if not transcript.exists():
        raise FileNotFoundError("No transcript: the audio step (transcribe) must run before the story step")

    _, proxy, _ = job.video_infos()
    fps = proxy.fps
    sdf = pd.read_csv(job.shots).sort_values("start_sec")
    shots = list(zip(sdf.start_sec.astype(float), sdf.end_sec.astype(float)))
    shot_ids = list(sdf.shot_id.astype(int))            # position -> real shot id

    words: List[Dict[str, Any]] = []
    for seg in json.loads(transcript.read_text()):
        for w in seg.get("words_data") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({"start": float(w["start"]), "end": float(w["end"]), "word": str(w["word"])})

    data = merge_shards(job)                            # raises if a visual shard is missing
    tracks = attach_speaking_scores(data["tracks"], data["speaker_scores"])
    people: Dict[int, int] = {}
    speaking = set()
    for frames in tracks.values():
        for f, dets in frames.items():
            persons = [d for d in dets if d.get("cname") == "person"]
            people[int(f)] = len(persons)
            if any(d.get("speaking") for d in persons):
                speaking.add(int(f))
    ocr: Dict[int, set] = {}
    for r in data["ocr"]:
        ocr.setdefault(int(r["shot_id"]), set()).add(str(r["text"]))
    speakers_known = cfg.enable_speaker_detection       # off -> "unknown", not "nobody speaks"

    def facts_at(a: float, b: float, shot_pos: int) -> Dict[str, Any]:
        fr = range(int(a * fps), max(int(a * fps) + 1, int(b * fps)))
        return {"speaking": sum(f in speaking for f in fr) / len(fr) if speakers_known else None,
                "people": max((people.get(f, 0) for f in fr), default=0),
                "ocr": sorted(ocr.get(shot_ids[shot_pos], []))}

    return shots, words, facts_at
