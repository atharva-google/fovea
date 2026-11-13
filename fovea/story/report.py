"""Plain-text report of the units, Gemini's story map and the cuts picked from it (CLI / notebooks)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .select import select_units
from .units import Unit


def make_cuts(units: List[Unit], story_map: Dict[str, Any], durations: Sequence[float], modes: Sequence[str],
              **params: Any) -> Dict[str, Dict[str, Any]]:
    """``{"15s_synced": {"pieces": [...], "stats": {...}}}`` for every duration x mode.
    A duration that is not shorter than the video is reported, not cut."""
    source = max((u.end for u in units), default=0.0)
    cuts: Dict[str, Dict[str, Any]] = {}
    for target in durations:
        for mode in modes:
            name = f"{target:g}s_{mode}"
            if target >= source:
                cuts[name] = {"pieces": [], "stats": {
                    "target": target, "mode": mode, "total": 0.0, "units": [], "arc_units": [], "ok": False,
                    "notes": [f"skipped: the video is only {source:.1f}s long"]}}
                continue
            pieces, stats = select_units(units, story_map, target, mode, **params)
            cuts[name] = {"pieces": [p.to_dict() for p in pieces], "stats": stats}
    return cuts


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_story(units: List[Unit], story_map: Dict[str, Any]) -> str:
    info = {m["id"]: m for m in story_map["units"]}
    by_id = {u.id: u for u in units}
    lines = [f"SUMMARY: {story_map.get('summary', '')}"]
    hook = story_map.get("hook") or {}
    lines.append(f"HOOK: {' '.join(hook.get('units') or []) or '(none)'}  {_clip(hook.get('why', ''), 120)}")
    lines.append("")
    lines.append("UNITS (imp = Gemini's importance 0-10)")
    for u in units:
        m = info[u.id]
        sp = "?" if u.speaking is None else f"{u.speaking:.0%}"
        said = f'"{_clip(u.text, 60)}"' if u.text else "(no speech)"
        lines.append(f"  {u.id}  {u.start:6.2f}-{u.end:6.2f}s {u.dur:5.2f}s  shot {u.shot:<2} imp {m['importance']:4.1f}  "
                     f"{m['kind']:<9} speaker {sp:>4}  {said}")
        if m.get("shows") or m.get("note"):
            lines.append(f"        sees: {_clip(m.get('shows', ''), 70)} | why: {_clip(m.get('note', ''), 90)}")
    if story_map.get("needs"):
        lines.append("")
        lines.append("NEEDS (kept or dropped together)")
        for n in story_map["needs"]:
            lines.append(f"  {n['unit']} needs {' '.join(n['requires']) or '-'}: {_clip(n.get('why', ''), 100)}")
    if story_map.get("cutaways"):
        lines.append("")
        lines.append("CUTAWAYS (pictures that can cover a line)")
        for c in story_map["cutaways"]:
            shown = ", ".join(f"{p} ({_clip(info[p].get('shows', ''), 30)})" if p in by_id else p for p in c["pictures"])
            lines.append(f"  {c['speech']} <- {shown}  fit {c['fit']:g}: {_clip(c.get('why', ''), 90)}")
    if story_map.get("moments"):
        lines.append("")
        lines.append("MOMENTS")
        for m in story_map["moments"]:
            lines.append(f"  {m.get('unit')} @ {m.get('peak_s')}s: {_clip(m.get('note', ''), 100)}")
    return "\n".join(lines)


def format_cuts(cuts: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for name, cut in cuts.items():
        stats = cut["stats"]
        notes = f"  [{'; '.join(stats['notes'])}]" if stats["notes"] else ""
        flag = "" if stats["ok"] or not cut["pieces"] else "  NOT USABLE"
        lines.append(f"=== {name}: {stats['total']:g}s in {len(cut['pieces'])} piece(s){flag}{notes}")
        for p in cut["pieces"]:
            where = f"{p['video_start']:6.2f}s"
            if abs(p["video_start"] - p["audio_start"]) > 1e-6:
                where += f" (sound from {p['audio_start']:.2f}s)"
            lines.append(f"   {p['length']:5.2f}s @ {where}  {p['units']}  {_clip(p['note'], 80)}")
    return "\n".join(lines)
