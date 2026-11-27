"""Render the video in other shapes: a crop of the full height that moves left and right along the
stored focus line (framing/framing.json). All shapes are made in ONE ffmpeg run, from the original
file. Given an edit, only the kept parts are rendered, back to back.

    outputs/<video name>_9x16.mp4, _1x1.mp4, ...
    outputs/<video name>_synced_15s.mp4, _synced_15s_9x16.mp4, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .framing import crop_track
from .job import Job
from .media import _run

DEFAULT_FORMATS = ("9:16", "1:1", "4:3", "3:4")
Span = Tuple[float, float]


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
           log: Callable[[str], None] = print, segments: Optional[Sequence[Span]] = None, tag: str = "",
           crossfade: float = 0.06) -> List[Path]:
    """One file per format that is narrower than the video. Returns the files written.

    ``segments`` = ``[(start_s, end_s)]``: only these parts of the video, back to back (an edit). The
    edit is then also written in the video's own shape. ``tag`` goes into the file names.
    ``crossfade``: seconds over which the sound of one part is blended into the next at every cut
    (0 = hard cuts). The blend is centred on the cut, so sound and picture stay together."""
    width, height, fps = int(info["width"]), int(info["height"]) // 2 * 2, float(info["fps"])
    points = {s["id"]: s["points"] for s in framing["shots"]}
    shots = [{"start": v.start, "end": v.end, "points": points[v.id]} for v in video_units]
    stem = f"{job.source.stem}{tag}"

    jobs: List[Tuple[Path, int, str]] = []                    # (file, width, the crop filter or "")
    if segments:
        jobs.append((job.output(f"{stem}.mp4"), width, ""))
    for name, ratio in dict(parse_format(f) for f in formats).items():
        crop_w = int(round(height * ratio)) // 2 * 2
        if crop_w > width - 2:
            log(f"  {name}: as wide as the video or wider, so there is nothing to crop; skipped")
            continue
        x = _expression(crop_track(shots, width, crop_w, dead_zone), fps)
        jobs.append((job.output(f"{stem}_{name}.mp4"), crop_w, f"crop=w={crop_w}:h={height}:x='{x}':y=0"))
    if not jobs:
        return []

    n = len(jobs)
    spans = [(round(a * fps) / fps, round(b * fps) / fps) for a, b in segments or []]     # on whole frames
    pick = ("select='" + "+".join(f"gte(t,{a - 0.5 / fps:.4f})*lt(t,{b - 0.5 / fps:.4f})" for a, b in spans)
            + "',setpts=N/FRAME_RATE/TB") if spans else ""
    graph = [f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n))]
    graph += [f"[s{i}]{','.join(f for f in (crop, pick) if f) or 'null'}[o{i}]" for i, (_, _, crop) in enumerate(jobs)]
    sound = ["0:a?"] * n
    if spans and info.get("has_audio"):
        # Each part takes half a crossfade more sound on the sides where it meets another part; blending
        # the two overlaps away again leaves the sound exactly as long as the picture.
        half, end, last = max(0.0, crossfade) / 2, float(info.get("duration") or spans[-1][1]), len(spans) - 1
        pre = [0.0 if k == 0 else min(half, a) for k, (a, _) in enumerate(spans)]
        post = [0.0 if k == last else min(half, max(0.0, end - b)) for k, (_, b) in enumerate(spans)]
        graph += [f"[0:a]atrim=start={a - pre[k]:.4f}:end={b + post[k]:.4f},asetpts=PTS-STARTPTS[t{k}]"
                  for k, (a, b) in enumerate(spans)]
        joined = "[t0]"
        for k in range(1, len(spans)):
            overlap = post[k - 1] + pre[k]
            how = f"acrossfade=d={overlap:.4f}:c1=tri:c2=tri" if overlap > 0.002 else "concat=n=2:v=0:a=1"
            graph.append(f"{joined}[t{k}]{how}[x{k}]")
            joined = f"[x{k}]"
        graph.append(f"{joined}asplit={n}" + "".join(f"[a{i}]" for i in range(n)))
        sound = [f"[a{i}]" for i in range(n)]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(job.source), "-filter_complex", ";".join(graph)]
    for i, (path, _, _) in enumerate(jobs):
        cmd += ["-map", f"[o{i}]", "-map", sound[i], "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(path)]
    _run(cmd)
    for path, w, _ in jobs:
        log(f"  {w}x{height} -> {path}")
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
