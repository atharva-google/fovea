"""Shots: where the video cuts (PySceneDetect), stored as prep/shots.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

COLUMNS = ["shot_id", "start_s", "end_s"]


def detect(video_path: str | Path, threshold: float = 27.0) -> List[Dict[str, Any]]:
    """``[{"shot_id", "start_s", "end_s"}]`` covering the whole video, in order."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list() or [(video.base_timecode, video.duration)]
    return [{"shot_id": i, "start_s": round(_seconds(a), 3), "end_s": round(_seconds(b), 3)}
            for i, (a, b) in enumerate(scenes)]


def _seconds(timecode) -> float:
    return float(timecode.seconds) if hasattr(timecode, "seconds") else float(timecode.get_seconds())


def write_csv(path: str | Path, shots: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(shots)


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        return [{"shot_id": int(r["shot_id"]), "start_s": float(r["start_s"]), "end_s": float(r["end_s"])}
                for r in csv.DictReader(f)]
