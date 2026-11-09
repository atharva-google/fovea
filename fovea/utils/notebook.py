"""Notebook helpers (Colab / Jupyter)."""

from __future__ import annotations

import subprocess
from base64 import b64encode
from pathlib import Path


def show_video(video_path: str | Path, width: int = 360):
    """Embed an mp4 inline in a notebook (re-encodes to H.264 if needed)."""
    from IPython.display import HTML

    video_path = Path(video_path)
    data = video_path.read_bytes()
    if len(data) > 50 * 1024 * 1024:
        print(f"Video is large ({len(data) // 2**20} MB); embedding may be slow.")
    url = "data:video/mp4;base64," + b64encode(data).decode()
    return HTML(f'<video width="{width}" controls><source src="{url}" type="video/mp4"></video>')


def to_h264(video_path: str | Path) -> Path:
    """Convert e.g. mp4v debug renders to browser-playable H.264."""
    video_path = Path(video_path)
    out = video_path.with_name(f"h264_{video_path.name}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return out
