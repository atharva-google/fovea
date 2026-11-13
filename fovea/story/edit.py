"""Turn a cut (a list of pieces) into exact frame ranges and an ffmpeg audio graph.

A piece is "``length`` seconds of picture from ``video_start`` and sound from ``audio_start``".
Everything is rounded to whole source frames here, once, and the sound is cut to exactly the same
lengths, so picture and sound cannot drift apart however many pieces there are.

No third-party imports: used by fovea/steps/render.py and tested on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

AUDIO_FADE_S = 0.015  # tiny fade at both ends of every piece, so joins don't click


def frame_plan(pieces: Sequence[Dict[str, Any]], fps: float, total_frames: int = 0) -> List[Tuple[int, int, float]]:
    """``[(first_frame, frame_count, audio_start_s)]`` per piece, in output order.

    ``total_frames`` (0 = unknown) keeps every range inside the source."""
    plan: List[Tuple[int, int, float]] = []
    for p in pieces:
        first = max(0, int(round(float(p["video_start"]) * fps)))
        count = max(1, int(round(float(p["length"]) * fps)))
        if total_frames > 0:
            first = min(first, total_frames - 1)
            count = min(count, total_frames - first)
        plan.append((first, count, max(0.0, float(p["audio_start"]))))
    return plan


def plan_seconds(plan: Sequence[Tuple[int, int, float]], fps: float) -> float:
    return sum(count for _, count, _ in plan) / fps


def audio_graph(plan: Sequence[Tuple[int, int, float]], fps: float, audio_input: str = "1:a:0",
                out_label: str = "aout") -> str:
    """ffmpeg ``-filter_complex`` graph: cut the source sound into the pieces (sample-exact), fade
    each end for a few milliseconds, and join them. Output pad: ``[out_label]``."""
    if not plan:
        raise ValueError("a cut needs at least one piece")
    k = len(plan)
    parts = [f"[{audio_input}]asplit={k}" + "".join(f"[s{i}]" for i in range(k))]
    for i, (_, count, start) in enumerate(plan):
        dur = count / fps
        fade = min(AUDIO_FADE_S, dur / 2)
        parts.append(f"[s{i}]atrim=start={start:.6f}:duration={dur:.6f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d={fade:.6f},afade=t=out:st={dur - fade:.6f}:d={fade:.6f}[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(k)) + f"concat=n={k}:v=0:a=1[{out_label}]")
    return ";".join(parts)


def decode_args(source: Any, first_frame: int, count: int, fps: float, hwaccel: Sequence[str] = ()) -> List[str]:
    """ffmpeg command that writes exactly frames ``[first_frame, first_frame + count)`` of ``source``
    to stdout as raw BGR. Seeking a quarter of a frame early makes ffmpeg start ON ``first_frame``:
    it drops everything before the seek time, so seeking exactly on the frame time risks losing it to
    rounding, while seeking half a frame early makes ffmpeg duplicate the first frame to fill the gap
    (tests/test_edit.py checks the frames against a plain decode)."""
    seek = ["-ss", f"{(first_frame - 0.25) / fps:.6f}"] if first_frame > 0 else []
    return ["ffmpeg", "-v", "error", "-nostdin", *hwaccel, *seek, "-i", str(source), "-map", "0:v:0",
            "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def encode_args(source: Any, width: int, height: int, fps: float, plan: Sequence[Tuple[int, int, float]],
                output: Any, video_codec: Sequence[str], has_audio: bool = True) -> List[str]:
    """ffmpeg command that reads the cut's frames (raw BGR, ``width`` x ``height``) from stdin, takes the
    sound of every piece straight from ``source``, and writes the finished video."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
           "-r", f"{fps}", "-i", "-"]
    if has_audio:
        cmd += ["-i", str(source), "-filter_complex", audio_graph(plan, fps), "-map", "0:v:0", "-map", "[aout]"]
    cmd += [*video_codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    return cmd + [str(output)]
