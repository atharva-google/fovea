"""One folder per video, named after the file, holding everything about it.

    fovea_data/<video name>/
        job.json                 which file this folder belongs to (name + SHA-256)
        input/<video file>       a copy of the video
        prep/video_info.json     size, frame rate, length
        prep/proxy.mp4           720p copy with sound: analysed, and shown to Gemini
        prep/audio.wav           16 kHz mono sound for the transcript
        prep/shots.csv           the shots
        prep/units.json          the video cut into video units (shots) and audio units
        prep/log.json            Gemini's description of the cast and of every unit
        audio/transcript.csv     one row per word, with its times
        framing/framing.json     where the picture is centred, per shot over time
        edit/edit_synced_15s.json  a shorter version: which units are kept
        outputs/                 the rendered videos, one per format
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def fingerprint(path: str | Path, chunk: int = 4 * 1024 * 1024) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @classmethod
    def open(cls, video: str | Path, data_dir: str | Path) -> "Job":
        """The folder for ``video`` (created if needed, with a copy of the video in ``input/``).
        A folder left by a DIFFERENT file with the same name is emptied first."""
        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        job = cls(Path(data_dir) / video.stem)
        sha = fingerprint(video)
        if job.root.exists() and job.fingerprint != sha:
            if video.resolve().is_relative_to(job.root.resolve()):
                raise ValueError(f"{job.root} belongs to a different video; pass the original file, not its copy")
            shutil.rmtree(job.root)
        dest = job.root / "input" / video.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, dest)
        write_json(job.root / "job.json", {"source": video.name, "sha256": sha})
        return job

    # ---------------------------------------------------------------- what the folder belongs to
    @property
    def fingerprint(self) -> str:
        try:
            return read_json(self.root / "job.json")["sha256"]
        except (OSError, ValueError, KeyError):
            return ""

    @property
    def source(self) -> Path:
        return self.root / "input" / read_json(self.root / "job.json")["source"]

    # ---------------------------------------------------------------- files
    def _file(self, rel: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def video_info(self) -> Path: return self._file("prep/video_info.json")
    @property
    def proxy(self) -> Path: return self._file("prep/proxy.mp4")
    @property
    def audio_wav(self) -> Path: return self._file("prep/audio.wav")
    @property
    def shots_csv(self) -> Path: return self._file("prep/shots.csv")
    @property
    def units_json(self) -> Path: return self._file("prep/units.json")
    @property
    def log_json(self) -> Path: return self._file("prep/log.json")
    @property
    def transcript_csv(self) -> Path: return self._file("audio/transcript.csv")
    @property
    def framing_json(self) -> Path: return self._file("framing/framing.json")

    def edit_json(self, name: str) -> Path:
        return self._file(f"edit/edit_{name}.json")

    def output(self, name: str) -> Path:
        return self._file(f"outputs/{name}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Any) -> None:
    """Written through a temp file, so a half-written file is never read."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)
