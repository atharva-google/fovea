"""Where the picture is centred at a given time, from the stored focus points of one shot.

A shot's points are ``[{"t", "move", "x", "left", "right", "on", "at"}]`` in time order (seconds in
the video; positions as a percentage of the frame width). ``move`` says how a point is reached:

    "cut"    the picture jumps there at ``t`` (a new shot, or the focus changes to another subject
             when a sound starts)
    "glide"  the picture moves there evenly from the point before it (a subject that moves)
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


def position_at(points: Sequence[Dict[str, Any]], t: float, key: str = "x") -> float:
    """The value of ``key`` ("x", "left" or "right") at time ``t`` within the shot."""
    k = max((i for i, p in enumerate(points) if p["t"] <= t + 1e-9), default=0)
    here = points[k]
    if k + 1 < len(points) and points[k + 1]["move"] == "glide":
        nxt = points[k + 1]
        span = nxt["t"] - here["t"]
        share = min(1.0, max(0.0, (t - here["t"]) / span)) if span > 0 else 1.0
        return here[key] + (nxt[key] - here[key]) * share
    return float(here[key])
