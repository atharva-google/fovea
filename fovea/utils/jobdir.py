"""The folder layout of one job, and which files each step reads and writes.

The SAME relative layout is used on local disk and in the cloud bucket
(``gs://<bucket>/users/<user_id>/<job_id>/...``), so the cloud worker only
has to copy a step's inputs down, run it, and copy its outputs up.

::

    job.json                    request: targets + config overrides
    input/<video file>          the original upload
    prep/video_info.json        {"source": {...}, "proxy": {...}, "has_audio": bool}
    prep/proxy.mp4              720p / 25fps silent copy used for all analysis
    prep/audio.wav              16 kHz audio
    prep/shots.csv              shot list
    prep/shards.json            how shots are split across visual workers
    visual/shard_000.pkl        per-shard analysis (tracks, motion, OCR, speaker scores)
    audio/                      vocals.wav, no_vocals.wav, transcript.json, transcript.csv
    framing/                    focal_points.csv, trajectory_<label>.csv
    outputs/<label>.mp4         final videos
    debug/                      optional annotated videos
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .video import VideoInfo

STEPS = ("prep", "visual", "audio", "render")


def step_files(step: str, shard: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Return ``(inputs, outputs)`` for a step. Entries ending in "/" are folders."""
    if step == "prep":
        return ["job.json", "input/"], ["prep/"]
    if step == "visual":
        if shard is None:
            raise ValueError("visual step needs a shard index")
        return ["job.json", "prep/"], [f"visual/shard_{shard:03d}.pkl"]
    if step == "audio":
        return ["job.json", "prep/audio.wav", "prep/video_info.json"], ["audio/"]
    if step == "render":
        return ["job.json", "input/", "prep/", "visual/"], ["framing/", "outputs/", "debug/"]
    raise ValueError(f"Unknown step: {step}")


class JobDir:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def folder(self, rel: str) -> Path:
        p = self.root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------------------------------------------------------- files
    @property
    def job_json(self) -> Path: return self.path("job.json")
    @property
    def video_info(self) -> Path: return self.path("prep/video_info.json")
    @property
    def proxy(self) -> Path: return self.path("prep/proxy.mp4")
    @property
    def audio(self) -> Path: return self.path("prep/audio.wav")
    @property
    def shots(self) -> Path: return self.path("prep/shots.csv")
    @property
    def shards(self) -> Path: return self.path("prep/shards.json")
    @property
    def audio_dir(self) -> Path: return self.folder("audio")
    @property
    def framing_dir(self) -> Path: return self.folder("framing")
    @property
    def outputs_dir(self) -> Path: return self.folder("outputs")
    @property
    def debug_dir(self) -> Path: return self.folder("debug")

    def shard_result(self, index: int) -> Path:
        return self.path(f"visual/shard_{index:03d}.pkl")

    @property
    def source(self) -> Path:
        files = sorted(p for p in (self.root / "input").glob("*") if not p.name.startswith("."))
        if not files:
            raise FileNotFoundError(f"No input video in {self.root / 'input'}")
        return files[0]

    def set_source(self, video_path: str | Path) -> Path:
        """Place a local video at ``input/<name>`` (symlinked to avoid copying)."""
        src = Path(video_path).resolve()
        d = self.folder("input")
        dest = d / src.name
        if dest.is_symlink() or dest.exists():
            if dest.resolve() == src:
                return dest
            dest.unlink()
        for old in d.glob("*"):
            old.unlink()
        try:
            dest.symlink_to(src)
        except OSError:
            shutil.copy2(src, dest)
        return dest

    # ----------------------------------------------------------------- json
    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(Path(path).read_text())

    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def video_infos(self) -> Tuple[VideoInfo, VideoInfo, bool]:
        """(source info, proxy info, has_audio) as written by the prep step."""
        info = self.read_json(self.video_info)
        return VideoInfo(**info["source"]), VideoInfo(**info["proxy"]), info.get("has_audio", True)

    def job_config(self) -> Dict[str, Any]:
        """Config overrides stored in job.json (empty if there is none)."""
        if not self.job_json.exists():
            return {}
        return self.read_json(self.job_json).get("config", {})
