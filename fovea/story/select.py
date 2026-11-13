"""Selector: pick units for a target duration from the stored story map. Code only, no Gemini.

The story comes first, the target second:

  * units that need each other (``needs``) are glued into BUNDLES, kept or dropped together
  * bundles holding an essential unit (importance >= ``arc_importance``) are ALWAYS kept, even when
    that alone runs past the target
  * the rest of the time is filled with the most valuable bundles (0/1 knapsack; ``cut_penalty``
    per separate run favours fewer, longer pieces), aiming for the target
  * in cutaway mode, picture units that can cover a chosen line are shown OVER it instead of
    taking their own time; the line's sound is untouched

Ported from notebooks/experiments/story_map_experiment.ipynb (branch exp/story-map). Differences
from the experiment are marked "changed:" below.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from .units import Unit

MODES = ("synced", "cutaway")


@dataclass
class Piece:
    """One output piece: ``length`` seconds of picture from video_start and sound from audio_start."""
    video_start: float
    audio_start: float
    length: float
    units: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def unit_value(importance: float, dur: float, imp_power: float = 1.5, dur_power: float = 0.5) -> float:
    """How much a unit is worth keeping: importance matters most, length a little (so a short punchline
    beats a long so-so stretch)."""
    return (max(0.0, importance) ** imp_power) * (max(dur, 0.05) ** dur_power)


def select_units(units: List[Unit], smap: Dict[str, Any], target: float, mode: str = "synced",
                 cut_penalty: float = 3.0, tolerance: float = 0.1, over_tolerance: float = 0.15,
                 use_hook: bool = True, min_cutaway: float = 0.8, imp_power: float = 1.5,
                 dur_power: float = 0.5, min_importance: float = 1.0, arc_importance: float = 8.0,
                 bridge_gap: float = 0.5) -> Tuple[List[Piece], Dict[str, Any]]:
    """Returns ``(pieces, stats)``. ``stats["ok"]`` is False when the result is not meaningfully
    shorter than the source; ``stats["notes"]`` says what happened in plain words."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not units:
        return [], {"target": target, "mode": mode, "total": 0.0, "units": [], "arc_units": [],
                    "notes": ["the video has no units"], "ok": False}

    info = {m["id"]: m for m in smap["units"]}
    by_id = {u.id: u for u in units}
    pos = {u.id: i for i, u in enumerate(units)}
    imp = {u.id: float(info[u.id]["importance"]) for u in units}

    overlay_ok: Dict[str, List[Tuple[float, str]]] = {}   # speech unit -> usable overlay pictures
    for c in smap["cutaways"]:
        for p in c["pictures"]:
            pu = by_id[p]
            if (pu.speaking or 0) < 0.2 and info[p]["kind"] not in ("dialogue", "voiceover"):
                overlay_ok.setdefault(c["speech"], []).append((c["fit"], p))
    overlays = {p for lst in overlay_ok.values() for _, p in lst}

    # bundles = connected groups of `needs` links (union-find)
    parent = {u.id: u.id for u in units}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for n in smap["needs"]:
        for r in n["requires"]:
            parent[find(n["unit"])] = find(r)
    bundles: Dict[str, List[str]] = {}
    for u in units:
        bundles.setdefault(find(u.id), []).append(u.id)
    # changed: the experiment tested "is this unit its own root", which also dropped a filler or overlay
    # unit that other units need. Only units that stand alone are dropped here.
    groups: List[List[str]] = []
    for ids in bundles.values():
        if len(ids) == 1:
            if imp[ids[0]] < min_importance:
                continue   # plain filler is never picked
            if mode == "cutaway" and ids[0] in overlays:
                continue   # used as an overlay instead
        groups.append(ids)

    def runs(ids: List[str]) -> int:
        idx = sorted(pos[i] for i in ids)
        return 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)

    step = 0.1

    def weight(ids: List[str]) -> int:
        return max(1, int(round(sum(by_id[i].dur for i in ids) / step)))

    # changed: the story (arc) is kept first; the target is an aim, not a ceiling.
    arc = [ids for ids in groups if any(imp[i] >= arc_importance for i in ids)]
    rest = [ids for ids in groups if not any(imp[i] >= arc_importance for i in ids)]
    arc_w = sum(weight(ids) for ids in arc)
    cap = int(round(target / step))
    hi = int(round(target * (1 + over_tolerance) / step))
    lo = int(math.floor((1 - tolerance) * cap))
    room = max(0, hi - arc_w)

    items = []
    for ids in rest:
        w = weight(ids)
        v = sum(unit_value(imp[i], by_id[i].dur, imp_power, dur_power) for i in ids) - cut_penalty * runs(ids)
        if w <= room and v > 0:
            items.append((w, v, ids))

    best = [0.0] + [-1e18] * room
    keep: List[List[List[str]]] = [[] for _ in range(room + 1)]
    for w, v, ids in items:
        for t in range(room, w - 1, -1):
            if best[t - w] > -1e17 and best[t - w] + v > best[t]:
                best[t] = best[t - w] + v
                keep[t] = keep[t - w] + [ids]
    reachable = [t for t in range(room + 1) if best[t] > -1e17]
    in_range = [t for t in reachable if lo <= arc_w + t <= cap]
    if in_range:
        t_best = max(in_range, key=lambda t: (best[t], t))
    else:   # nothing lands in [target - tolerance, target]: take what comes closest to the target
        t_best = min(reachable, key=lambda t: (abs(arc_w + t - cap), -best[t]))
    chosen = {i for ids in arc for i in ids} | {i for ids in keep[t_best] for i in ids}

    notes: List[str] = []
    total_w = arc_w + t_best
    if arc_w > cap:
        notes.append(f"the story needs {arc_w * step:.1f}s (units scored >= {arc_importance:g} and what they "
                     f"need), which is over the {target:g}s target")
    elif total_w > cap:
        notes.append(f"{(total_w - cap) * step:.1f}s over the target to keep whole bundles")
    elif total_w < lo:
        notes.append(f"could not reach {(1 - tolerance) * target:.1f}s with units worth keeping")

    order = [u for u in units if u.id in chosen]
    if use_hook:
        required = {r for n in smap["needs"] for r in n["requires"]}
        hook_ids = [h for h in smap["hook"]["units"] if h in chosen and h not in required
                    and not any(n["unit"] == h and n["requires"] for n in smap["needs"])]
        if hook_ids:
            order = [u for u in order if u.id in hook_ids] + [u for u in order if u.id not in hook_ids]
            notes.append(f"hook first: {', '.join(hook_ids)}")
        elif smap["hook"]["units"]:
            notes.append("hook not moved (not chosen, or it depends on a setup)")

    pieces: List[Piece] = []
    used = set()
    for u in order:
        if mode == "cutaway" and u.kind == "speech" and u.id in overlay_ok:
            options = [(f, p) for f, p in sorted(overlay_ok[u.id], reverse=True)
                       if p not in used and by_id[p].dur >= min_cutaway]
            if options:
                _, p = options[0]
                pu = by_id[p]
                used.add(p)
                cut_len = min(pu.dur, u.dur)
                own = u.dur - cut_len
                label = f"cutaway: {info[p]['shows']}"
                if own >= 0.3:   # speaker first, then cut to the picture while the line finishes
                    pieces.append(Piece(u.start, u.start, own, u.id, "speaker (synced)"))
                    pieces.append(Piece(pu.start, u.start + own, cut_len, f"{u.id}<-{p}", label))
                else:
                    pieces.append(Piece(pu.start, u.start, u.dur, f"{u.id}<-{p}", label))
                continue
        pieces.append(Piece(u.start, u.start, u.dur, u.id, info[u.id]["note"][:60]))

    pieces = merge_pieces(pieces, bridge_gap)
    total = round(sum(p.length for p in pieces), 2)
    source = max(u.end for u in units)
    ok = bool(pieces) and total < 0.95 * source
    if not pieces:
        notes.append("nothing worth keeping was found")
    elif not ok:
        notes.append(f"not meaningfully shorter than the source ({total:.1f}s of {source:.1f}s)")
    stats = {"target": target, "mode": mode, "total": total, "units": [u.id for u in order],
             "arc_units": [i for ids in arc for i in ids], "notes": notes, "ok": ok}
    return pieces, stats


def merge_pieces(pieces: List[Piece], bridge_gap: float = 0.5) -> List[Piece]:
    """Join pieces that continue each other in the source, so a join is a real cut and nothing else.

    changed (not in the experiment): two kept pieces separated by a pause shorter than ``bridge_gap``
    seconds are joined THROUGH the pause. Units leave pauses under 0.5 s unassigned; cutting those
    out would be a tiny jump cut inside a continuous shot.
    """
    out: List[Piece] = []
    for p in pieces:
        if out:
            prev = out[-1]
            v_gap = p.video_start - (prev.video_start + prev.length)
            a_gap = p.audio_start - (prev.audio_start + prev.length)
            # strictly shorter: a dropped picture unit is never shorter than 0.5 s, so it is never bridged back in
            if abs(v_gap - a_gap) < 2e-3 and -2e-3 <= v_gap < bridge_gap:
                gap = max(0.0, v_gap)
                note = prev.note if p.note in ("", prev.note) else (f"{prev.note}; {p.note}" if prev.note else p.note)
                out[-1] = Piece(prev.video_start, prev.audio_start, round(prev.length + gap + p.length, 3),
                                f"{prev.units} {p.units}", note[:200])
                continue
        out.append(Piece(round(p.video_start, 3), round(p.audio_start, 3), round(p.length, 3), p.units, p.note))
    return out
