"""ffmpeg / ffprobe helpers: probe a video, make the analysis copy, pull out the sound."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list) -> subprocess.CompletedProcess:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"{cmd[0]} not found. Install ffmpeg (apt install ffmpeg / brew install ffmpeg).")
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{res.stderr.decode(errors='ignore')[-1500:]}")
    return res


def probe(path: str | Path) -> Dict[str, Any]:
    """``{"width", "height", "fps", "duration", "has_audio"}`` of a video file."""
    out = _run(["ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", str(path)]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {path}")
    num, _, den = str(video.get("r_frame_rate", "25/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 25.0
    return {"width": int(video["width"]), "height": int(video["height"]), "fps": round(fps, 3),
            "duration": round(float(data.get("format", {}).get("duration") or 0.0), 3),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def make_proxy(source: str | Path, output: str | Path, height: int = 720, fps: float = 25.0) -> Path:
    """Small constant-frame-rate copy WITH sound: shots are found on it and Gemini watches it."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", f"fps={fps},scale=-2:'min({int(height)},ih)'",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", str(int(round(fps))), "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output)])
    return Path(output)


def extract_audio(source: str | Path, output: str | Path) -> Path:
    """16 kHz mono wav, what Whisper takes."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "pcm_s16le", str(output)])
    return Path(output)


def load_audio(path: str | Path):
    """A wav / video file's sound as 16 kHz mono float32 samples (numpy array)."""
    import numpy as np

    raw = _run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1",
                "-ar", "16000", "-"]).stdout
    return np.frombuffer(raw, np.float32)
