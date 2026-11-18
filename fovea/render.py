"""Render the video in other shapes: a crop of the full height that moves left and right along the
stored focus line (framing/framing.json). All shapes are made in ONE ffmpeg run, from the original
file, with the sound untouched in length.

    outputs/<video name>_9x16.mp4, _1x1.mp4, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .framing import crop_track
from .job import Job
from .media import _run

DEFAULT_FORMATS = ("9:16", "1:1", "4:3", "3:4")


def parse_format(text: str) -> Tuple[str, float]:
    """``"9:16"`` -> ``("9x16", 0.5625)``. Any WIDTH:HEIGHT works."""
    try:
        w, h = (float(x) for x in text.lower().replace("x", ":").split(":"))
        if w <= 0 or h <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"Not a format: {text!r}. Write it as WIDTH:HEIGHT, e.g. 9:16") from None
    return f"{w:g}x{h:g}", w / h


def render(job: Job, video_units: Sequence[Any], framing: Dict[str, Any], info: Dict[str, Any],
           formats: Sequence[str] = DEFAULT_FORMATS, dead_zone: float = 3.0, crf: int = 18,
           log: Callable[[str], None] = print) -> List[Path]:
    """One file per format that is narrower than the video. Returns the files written."""
    width, height = int(info["width"]), int(info["height"]) // 2 * 2
    points = {s["id"]: s["points"] for s in framing["shots"]}
    shots = [{"start": v.start, "end": v.end, "points": points[v.id]} for v in video_units]

    jobs: List[Tuple[Path, int, str]] = []                    # (file, crop width, where its left edge is)
    for name, ratio in dict(parse_format(f) for f in formats).items():
        crop_w = int(round(height * ratio)) // 2 * 2
        if crop_w > width - 2:
            log(f"  {name}: as wide as the video or wider, so there is nothing to crop; skipped")
            continue
        track = crop_track(shots, width, crop_w, dead_zone)
        jobs.append((job.output(f"{job.source.stem}_{name}.mp4"), crop_w, _expression(track, float(info["fps"]))))
    if not jobs:
        return []

    graph = [f"[0:v]split={len(jobs)}" + "".join(f"[s{i}]" for i in range(len(jobs)))]
    graph += [f"[s{i}]crop=w={crop_w}:h={height}:x='{x}':y=0[o{i}]" for i, (_, crop_w, x) in enumerate(jobs)]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(job.source), "-filter_complex", ";".join(graph)]
    for i, (path, _, _) in enumerate(jobs):
        cmd += ["-map", f"[o{i}]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(path)]
    _run(cmd)
    for path, crop_w, _ in jobs:
        log(f"  {crop_w}x{height} -> {path}")
    return [path for path, _, _ in jobs]


def _expression(track: List[Tuple[float, float, float, float]], fps: float) -> str:
    """The crop's left edge as an ffmpeg expression of the time ``t``. Each piece starts half a frame
    early, so the frame AT a boundary always falls in the piece that starts there."""
    early = 0.5 / fps if fps > 0 else 0.0
    terms = []
    for i, (a, b, x0, x1) in enumerate(track):
        value = f"{x0:g}" if x0 == x1 else f"({x0:g}+{x1 - x0:g}*clip((t-{a:.3f})/{b - a:.3f},0,1))"
        after = "" if i == 0 else f"*gte(t,{a - early:.4f})"
        before = "" if i == len(track) - 1 else f"*lt(t,{b - early:.4f})"
        terms.append(f"{value}{after}{before}")
    return "+".join(terms)
