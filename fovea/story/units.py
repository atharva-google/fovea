"""Units: small, exact pieces of the video that Gemini refers to by id.

Speech units are phrases cut on word boundaries (never across a shot cut); picture units are
stretches without speech, split into short chunks. Because Gemini only ever names unit ids, every
time used later comes from here, not from the model.

Ported from notebooks/experiments/story_map_experiment.ipynb (branch exp/story-map).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

UNITS_VERSION = 1  # bump when build_units changes in a way that makes stored units stale


@dataclass
class Unit:
    id: str
    start: float
    end: float
    shot: int                 # position of the shot in the (time-sorted) shot list
    kind: str                 # "speech" | "picture"
    text: str = ""
    speaking: Optional[float] = None   # share of frames where TalkNet sees a person talking (None = unknown)
    people: Optional[int] = None       # most persons detected at once in this unit
    ocr: List[str] = field(default_factory=list)  # on-screen text in this shot

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Unit":
        return cls(id=str(d["id"]), start=float(d["start"]), end=float(d["end"]), shot=int(d["shot"]),
                   kind=str(d["kind"]), text=str(d.get("text") or ""), speaking=d.get("speaking"),
                   people=d.get("people"), ocr=list(d.get("ocr") or []))


FactsAt = Callable[[float, float, int], Dict[str, Any]]


def build_units(shots: List[Tuple[float, float]], words: List[Dict[str, Any]],
                facts_at: Optional[FactsAt] = None, max_gap: float = 0.35, max_speech: float = 6.0,
                min_picture: float = 0.5, max_picture: float = 2.5) -> List[Unit]:
    """``shots`` = [(start_s, end_s)] sorted by time; ``words`` = [{"start", "end", "word"}].

    ``facts_at(start, end, shot)`` -> {"speaking", "people", "ocr"} from the pipeline's analysis.
    """
    if not shots:
        return []

    def shot_of(t: float) -> int:
        for i, (a, b) in enumerate(shots):
            if a <= t < b:
                return i
        return len(shots) - 1

    words = sorted((w for w in words if w["end"] > w["start"]), key=lambda w: w["start"])
    phrases: List[List[Dict[str, Any]]] = []
    for w in words:
        if phrases:
            cur = phrases[-1]
            prev = cur[-1]
            gap = w["start"] - prev["end"]
            dur = prev["end"] - cur[0]["start"]
            new = (gap > max_gap
                   or re.search(r"[.?!]$", prev["word"].strip()) is not None
                   or (re.search(r"[,;:]$", prev["word"].strip()) and dur > 2.5)
                   or (dur > max_speech and gap > 0.12)
                   or shot_of((w["start"] + w["end"]) / 2) != shot_of((prev["start"] + prev["end"]) / 2))
            if not new:
                cur.append(w)
                continue
        phrases.append([w])

    raw: List[Tuple[float, float, int, str, str]] = []
    for k, ph in enumerate(phrases):
        s = shot_of((ph[0]["start"] + ph[0]["end"]) / 2)
        a0, b0 = shots[s]
        prev_end = phrases[k - 1][-1]["end"] if k else 0.0
        next_start = phrases[k + 1][0]["start"] if k + 1 < len(phrases) else 1e9
        start = max(a0, ph[0]["start"] - 0.08, (prev_end + ph[0]["start"]) / 2 if k else 0.0)
        end = min(b0, ph[-1]["end"] + 0.12, (ph[-1]["end"] + next_start) / 2)
        if end - start < 0.05:
            continue  # not in the experiment: a phrase squeezed to nothing by its shot's edges can't be a unit
        raw.append((start, end, s, "speech", " ".join(w["word"].strip() for w in ph)))

    # picture units: uncovered stretches inside each shot
    for s, (a, b) in enumerate(shots):
        covered = sorted((r[0], r[1]) for r in raw if r[2] == s and r[3] == "speech")
        t = a
        holes = []
        for c0, c1 in covered + [(b, b)]:
            if c0 - t >= min_picture:
                holes.append((t, c0))
            t = max(t, c1)
        for h0, h1 in holes:
            n = max(1, math.ceil((h1 - h0) / max_picture))
            step = (h1 - h0) / n
            for i in range(n):
                raw.append((h0 + i * step, h0 + (i + 1) * step, s, "picture", ""))

    raw.sort()
    units = []
    for i, (a, b, s, kind, text) in enumerate(raw):
        f = facts_at(a, b, s) if facts_at else {}
        units.append(Unit(f"U{i + 1:02d}", round(a, 3), round(b, 3), s, kind, text,
                          f.get("speaking"), f.get("people"), list(f.get("ocr") or [])))
    return units


def units_table(units: List[Unit]) -> str:
    """One line per unit: what Gemini reads, and what the CLI prints."""
    lines = []
    for u in units:
        sp = "?" if u.speaking is None else f"{u.speaking:.0%}"
        ppl = "?" if u.people is None else str(u.people)
        txt = f'"{u.text}"' if u.text else "(no speech)"
        ocr = f" | on-screen text: {'; '.join(u.ocr)}" if u.ocr else ""
        lines.append(f"{u.id} | {u.start:.2f}-{u.end:.2f}s ({u.dur:.2f}s) | shot {u.shot} | {u.kind} | "
                     f"people: {ppl} | visible speaker: {sp} | {txt}{ocr}")
    return "\n".join(lines)
