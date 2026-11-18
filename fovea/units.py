"""Cut one video into units, on two tracks that each cover the whole video. Code only.

    video units  V01, V02, ...  one per shot.
    audio units  A01, A02, ...  a few spoken words, or the sound between them ("gap", in chunks of at
                                most a second). Split at every audible pause and at punctuation. A
                                phrase that runs across a shot cut stays ONE unit (a word cannot be
                                split).
    slots                       every audio unit, split at the shot cuts it runs across: the part of
                                a sound that plays over ONE shot. Gemini describes the video shot by
                                shot, with the slots of each shot under it.

Gemini only ever names these ids; every time used later comes from here.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

UNITS_VERSION = 3   # bump when the cutting rules change in a way that makes stored units stale

_PUNCT = re.compile(r"[.?!,;:\u0964]$")   # \u0964 = the Devanagari full stop

Span = Tuple[float, float]


@dataclass
class VideoUnit:
    id: str
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoUnit":
        return cls(str(d["id"]), float(d["start"]), float(d["end"]))


@dataclass
class AudioUnit:
    id: str
    start: float
    end: float
    kind: str                                                   # "speech" | "gap"
    text: str = ""                                              # rough transcript; "" for a gap
    words: List[Dict[str, Any]] = field(default_factory=list)   # [{"start", "end", "word"}]
    over: List[str] = field(default_factory=list)               # video units it plays over

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioUnit":
        return cls(str(d["id"]), float(d["start"]), float(d["end"]), str(d["kind"]), str(d.get("text") or ""),
                   list(d.get("words") or []), list(d.get("over") or []))


def cut_units(shots: Sequence[Span], words: Sequence[Dict[str, Any]]) -> Tuple[List[AudioUnit], List[VideoUnit]]:
    """Both tracks for one video. ``shots`` = [(start_s, end_s)] in order; ``words`` = transcript
    rows with ``word`` and ``start_s`` / ``end_s`` (or ``start`` / ``end``)."""
    audio = build_audio_units(words, shots)
    video = build_video_units(shots)
    for a in audio:   # the video units each sound plays over, in time order
        laps = [(min(a.end, v.end) - max(a.start, v.start), v.id) for v in video]
        a.over = [vid for lap, vid in laps if lap > 0.1] or ([max(laps)[1]] if laps else [])
    return audio, video


# =========================================================================== video units
def build_video_units(shots: Sequence[Span]) -> List[VideoUnit]:
    spans = [(float(a), float(b)) for a, b in shots if float(b) > float(a)]
    return [VideoUnit(f"V{i + 1:02d}", round(a, 3), round(b, 3)) for i, (a, b) in enumerate(spans)]


def slots(audio: Sequence[AudioUnit], video: Sequence[VideoUnit], min_part: float = 0.25) -> List[Dict[str, Any]]:
    """Every audio unit, split at the shot cuts it runs across: ``[{"id", "audio", "shot", "start",
    "end", "part"}]`` in time order. A unit over one shot is one slot with the unit's own id; otherwise
    the id plus a letter (A05a, A05b). Slivers shorter than ``min_part`` stay with their neighbour."""
    out: List[Dict[str, Any]] = []
    for a in audio:
        parts = [[max(a.start, v.start), min(a.end, v.end), v.id] for v in video
                 if min(a.end, v.end) - max(a.start, v.start) > 1e-6]
        while len(parts) > 1:
            k = min(range(len(parts)), key=lambda j: parts[j][1] - parts[j][0])
            if parts[k][1] - parts[k][0] >= min_part:
                break
            j = k - 1 if k > 0 else k + 1
            lo, hi = min(j, k), max(j, k)
            parts[lo:hi + 1] = [[parts[lo][0], parts[hi][1], parts[j][2]]]
        for i, (start, end, shot) in enumerate(parts):
            out.append({"id": a.id if len(parts) == 1 else f"{a.id}{chr(97 + i)}", "audio": a.id, "shot": shot,
                        "start": round(start, 3), "end": round(end, 3), "part": i})
    return out


# =========================================================================== audio units
def build_audio_units(words: Sequence[Dict[str, Any]], shots: Sequence[Span], min_gap: float = 0.25,
                      max_gap: float = 1.0, pad: float = 0.06) -> List[AudioUnit]:
    """Tile ``[0, end of the last shot]`` with speech and gap units.

    ``min_gap``: a pause at least this long is its own unit (so it can be cut); shorter pauses are
    shared between the words around them. ``max_gap``: longer silences are split into chunks.
    ``pad``: a little air kept around speech when the pause next to it becomes its own unit.
    """
    if not shots:
        return []
    total = float(shots[-1][1])
    ws = sorted(({"start": max(0.0, float(w.get("start_s", w.get("start")))),
                  "end": min(total, float(w.get("end_s", w.get("end")))), "word": str(w["word"])}
                 for w in words), key=lambda w: w["start"])
    ws = [w for w in ws if w["end"] > w["start"]]

    phrases: List[List[Dict[str, Any]]] = []
    for w in ws:
        if phrases:
            prev = phrases[-1][-1]
            if w["start"] - prev["end"] < min_gap and not _PUNCT.search(prev["word"].strip()):
                phrases[-1].append(w)
                continue
        phrases.append([w])

    spans: List[List[Any]] = []          # [start, end, words] per phrase, in time order without overlaps
    for ph in phrases:
        s, e = ph[0]["start"], max(w["end"] for w in ph)
        if spans:
            s = max(s, spans[-1][1])
            if e - s < 0.02:             # nothing left of it after its neighbour: same phrase
                spans[-1][2].extend(ph)
                continue
        spans.append([s, e, list(ph)])

    raw: List[Tuple[float, float, str, List[Dict[str, Any]]]] = []
    cursor = 0.0                         # where the next unit starts
    for i, (s, e, ph) in enumerate(spans):
        gap_before = s - (0.0 if i == 0 else spans[i - 1][1])
        if gap_before >= min_gap:        # the pause before this phrase is its own unit
            start = s - min(pad, gap_before / 4)
            raw.extend((a, b, "gap", []) for a, b in _split_gap(cursor, start, shots, min_gap, max_gap))
        else:
            start = cursor
        nxt = spans[i + 1][0] if i + 1 < len(spans) else total
        gap_after = nxt - e
        if gap_after >= min_gap:
            end = e + min(pad, gap_after / 4)
        else:                            # share a short pause with the next phrase / run to the end
            end = (e + nxt) / 2 if i + 1 < len(spans) else total
        raw.append((start, end, "speech", ph))
        cursor = end
    if total - cursor > 1e-6:
        raw.extend((a, b, "gap", []) for a, b in _split_gap(cursor, total, shots, min_gap, max_gap))

    return [AudioUnit(f"A{i + 1:02d}", round(a, 3), round(b, 3), kind, " ".join(w["word"].strip() for w in ph), ph)
            for i, (a, b, kind, ph) in enumerate(raw)]


def _split_gap(a: float, b: float, shots: Sequence[Span], min_gap: float, max_gap: float) -> List[Span]:
    """A silence, split at the shot cuts inside it and then into chunks of at most ``max_gap``."""
    cuts = [a] + [float(s) for s, _ in shots if a + min_gap <= float(s) <= b - min_gap] + [b]
    out: List[Span] = []
    for x, y in zip(cuts, cuts[1:]):
        n = max(1, math.ceil((y - x) / max_gap - 1e-9))
        step = (y - x) / n
        out.extend((x + k * step, x + (k + 1) * step) for k in range(n))
    return out
