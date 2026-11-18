"""Where the picture is centred at a given time, from the stored focus points of one shot.

A shot's points are ``[{"t", "move", "x", "slot"}]`` in time order (``t`` in seconds in the video,
``x`` as a percentage of the frame width). ``move`` says how a point is reached:

    "cut"    the picture jumps there at ``t`` (a new shot, or someone starts / stops speaking)
    "glide"  the picture moves there evenly from the point before it (a subject that moves)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def position_at(points: Sequence[Dict[str, Any]], t: float, key: str = "x") -> float:
    """The value of ``key`` at time ``t`` within the shot."""
    k = max((i for i, p in enumerate(points) if p["t"] <= t + 1e-9), default=0)
    here = points[k]
    if k + 1 < len(points) and points[k + 1]["move"] == "glide":
        nxt = points[k + 1]
        span = nxt["t"] - here["t"]
        share = min(1.0, max(0.0, (t - here["t"]) / span)) if span > 0 else 1.0
        return here[key] + (nxt[key] - here[key]) * share
    return float(here[key])


def crop_track(shots: Sequence[Dict[str, Any]], width: int, crop_w: int, dead_zone: float = 3.0
               ) -> List[Tuple[float, float, float, float]]:
    """Where the LEFT edge of a crop ``crop_w`` wide is over the whole video, following the focus line:
    ``[(start_s, end_s, left_at_start, left_at_end)]`` in pixels, back to back. A piece with two
    different values moves evenly.

    ``shots`` = ``[{"start", "end", "points"}]``. A change of the line smaller than ``dead_zone``
    (percent of the frame width) is not followed, so the picture does not twitch.
    """
    def left(x: float) -> float:
        return float(round(min(max(x / 100 * width - crop_w / 2, 0), width - crop_w)))

    track: List[Tuple[float, float, float, float]] = []
    for shot in shots:
        kept: List[Dict[str, Any]] = []
        for p in shot["points"]:
            if kept and 0 < abs(p["x"] - kept[-1]["x"]) < dead_zone:
                continue
            kept.append(p)
        for p, nxt in zip(kept, kept[1:] + [None]):
            start = max(p["t"], shot["start"]) if p is not kept[0] else shot["start"]
            end = nxt["t"] if nxt else shot["end"]
            if end - start <= 1e-6:
                continue
            a = left(p["x"])
            b = left(nxt["x"]) if nxt and nxt["move"] == "glide" else a
            if track and a == b and track[-1][2] == track[-1][3] == a:       # still in the same place
                track[-1] = (track[-1][0], end, a, a)
            else:
                track.append((start, end, a, b))
    return track
