"""The edit: ONE Gemini call, in text only, that picks what a shorter version keeps.

Gemini reads the log (it does not see the video) and names the audio units to keep. Audio units
cover the whole video back to back, so that list IS the edit: each sound plays with its own picture
("synced"). Code owns everything mechanical: times, the total length, pulling in what a kept unit
depends on.

The prompt here gives the task and the rules that span units; the fields are explained in schema.py.

    edit/edit_synced_15s.json    the units kept, and the stretches of the video they make
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import gemini
from .analysis import ANALYSIS_VERSION, KeyRequired, sound_tags
from .config import Settings
from .job import Job, read_json, write_json
from .units import UNITS_VERSION, AudioUnit, VideoUnit

EDIT_VERSION = 2       # bump when the prompt or schema changes in a way that makes stored edits stale
SHORT_RUN_S = 0.3      # a kept stretch shorter than this is reported: it is seen as a flash

PROMPT = """You are a senior video editor cutting a shorter version of an ad. You work from the log below,
which a colleague wrote while watching the video; you do not see the video yourself.

UNITS
The ad is logged shot by shot (V01, V02, ...). Under each shot are its audio units (A01, A02, ...) with
their length in seconds: a few spoken words, or up to a second of the sound between words. A unit that
runs on into the next shot is listed again there, marked (continued); it is still ONE unit.
You build the short version by choosing AUDIO UNITS. Each one brings the picture that plays with it, so
keeping the units under a shot keeps that part of the shot, also when nothing is said in it.
Kept units always play in their original order.

THE CUT
- Aim for about {seconds} seconds. That is a guide, not a limit: a little longer is fine when the story
  needs it. Never cut the story short to reach the number.
- The short version must work for someone who never saw the long one: it needs a setup and its payoff,
  and the product or brand must still be seen or named.
- Cut whole beats, not single words. Keep a sentence complete, and prefer runs of neighbouring units to
  many small pieces: every gap between kept units is a cut that is seen and heard.
- A unit, or its shot, can list what it "needs": keep those too, or drop the unit.
- MUSIC. A unit marked "music: song/peak" has that kind of music under it, in that part of the track. The
  music is cut together with everything else, so every gap between kept units is also a jump in the music.
  Put your cuts where it survives them: where there is no music, or exactly where one part of the track ends
  and the next begins (jumping from the end of a build to the start of the peak works; leaving the middle of
  a part is heard as a stumble). If you keep a build, keep the peak it leads to.
- A sung line, and any unit marked "timed to the music", is kept whole or dropped whole, together with the
  units of the same line next to it.
- A unit marked "sound:" holds a sound made by something seen. Keep it with the action it belongs to.
- Silence and music between lines are where time is saved most cheaply, but a reaction or a pause that a
  joke depends on is part of that joke.

THE LOG
{log}
"""


def name(seconds: float) -> str:
    """How an edit of about ``seconds`` is called in file names."""
    return f"synced_{seconds:g}s"


# =========================================================================== the step
def run(job: Job, audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], seconds: float,
        settings: Settings, api_key: Optional[str] = None, reuse: bool = True,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The stored edit of about ``seconds``, or a new one from Gemini. ``reuse=False`` when the log
    has just been written again: an edit made from the old log no longer fits."""
    path = job.edit_json(name(seconds))
    stamp = {"fingerprint": job.fingerprint, "version": [UNITS_VERSION, ANALYSIS_VERSION, EDIT_VERSION]}
    if reuse and not (settings.refresh or settings.force):
        try:
            stored = read_json(path)
            if {k: stored.get(k) for k in stamp} == stamp:
                log(f"  edit already stored ({path}), no Gemini call")
                return stored
        except (OSError, ValueError):
            pass
    if not api_key:
        raise KeyRequired("This edit has not been made yet, so a Gemini key is needed: "
                          "pass --gemini-key or set GEMINI_API_KEY")
    raw, took = gemini.ask(build_prompt(audio, video, described, seconds), api_key, settings.gemini_model,
                           settings.gemini_api_version, schema="Edit", thinking=settings.thinking, log=log)
    edit, issues = clean(raw, audio, described)
    for issue in issues[:15]:
        log(f"  fixed: {issue}")
    if not edit["keep"]:
        raise gemini.GeminiError("Gemini's edit keeps nothing; run again with --refresh")
    edit = {**stamp, "target": seconds, "model": settings.gemini_model, "thinking": settings.thinking,
            "took": took, "issues": issues, **edit}
    write_json(path, edit)
    log(f"  {edit['seconds']:.1f}s in {len(edit['segments'])} stretches -> {path}")
    return edit


# =========================================================================== the prompt
def build_prompt(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], seconds: float) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    lines = [f"Story: {described['summary']}", "Cast:"]
    for c in described["cast"]:
        names = " / ".join(n for n in (c["ad_name"], c["real_name"]) if n)
        lines.append(f"  {c['id']}{f' ({names})' if names else ''}: {c['description']}")
    lines.append(f"The whole ad is {video[-1].end:.2f}s long.\n")
    for v in video:
        d = dv[v.id]
        extra = "".join(f" | {label}: {'; '.join(d[key])}" for key, label in (("text", "text"), ("needs", "needs"))
                        if d[key])
        lines.append(f"{v.id} | {v.dur:.2f}s | cast: {' '.join(d['cast']) or '-'} | {d['shows']}{extra}")
        for a in audio:
            if v.id not in a.over:
                continue
            if a.over[0] != v.id:
                lines.append(f"    {a.id} | (continued)")
                continue
            u = da[a.id]
            who = f"{u['speaker']}: " if u["speaker"] else ""
            need = f" | needs: {' '.join(u['needs'])}" if u["needs"] else ""
            lines.append(f"    {a.id} | {a.dur:.2f}s | {u['kind']} | {who}{u['heard']}{sound_tags(u)}{need}")
    return PROMPT.replace("{seconds}", f"{seconds:g}").replace("{log}", "\n".join(lines))


# =========================================================================== checking the answer
def clean(raw: Dict[str, Any], audio: List[AudioUnit], described: Dict[str, Any]
          ) -> Tuple[Dict[str, Any], List[str]]:
    """``(edit, issues)``: ``{"story", "must_keep", "keep", "segments", "seconds", "cuts"}``. Only known
    units, in their original order, with everything they need and with sung lines whole; ``segments``
    are the stretches of the video the kept units make, as ``[start_s, end_s]``; ``cuts`` says for
    every gap between kept units how the music takes it."""
    issues: List[str] = []
    order = {a.id: i for i, a in enumerate(audio)}
    needs = {u["id"]: u["needs"] for u in [*described["audio_units"], *described["video_units"]]}

    def known(value: Any, where: str) -> str:
        uid = str(value or "").strip()
        uid = uid if uid in order else uid.rstrip("abcdefgh")          # "A05b" names a part of A05
        if uid not in order:
            issues.append(f"{where} names unknown unit {str(value)!r}: ignored")
            return ""
        return uid

    keep: List[str] = []
    for k in raw.get("keep") or []:
        uid = known(k.get("id") if isinstance(k, dict) else k, "keep")
        if not uid or uid in keep:
            continue
        keep.append(uid)
        try:        # Gemini copies each unit's length; a wrong copy means it lost its place
            if abs(float(k.get("seconds")) - audio[order[uid]].dur) > 0.05:
                issues.append(f"{uid} was taken to be {float(k.get('seconds')):g}s but is {audio[order[uid]].dur:.2f}s")
        except (TypeError, ValueError, AttributeError):
            pass
    if keep != sorted(keep, key=order.get):
        issues.append("the kept units were not listed in their original order: the original order is used")

    must = []
    for m in raw.get("must_keep") or []:
        uid = known(m.get("id") if isinstance(m, dict) else m, "must_keep")
        if uid:
            must.append({"id": uid, "why": str(m.get("why") or "").strip() if isinstance(m, dict) else ""})
            if uid not in keep:
                issues.append(f"{uid} is in must_keep but was not kept: added")
                keep.append(uid)

    # ---- what the kept units (and the shots they play over) need
    todo = list(keep)
    while todo:
        uid = todo.pop()
        for n in [*needs.get(uid, []), *(n for shot in audio[order[uid]].over for n in needs.get(shot, []))]:
            if n in order and n not in keep:
                issues.append(f"{uid} needs {n}: added")
                keep.append(n)
                todo.append(n)
    # ---- a sung line, or words timed to the music, is one piece: all of it or none of it
    sound = {u["id"]: u for u in described["audio_units"]}
    locked = [sound[a.id]["kind"] == "lyrics" or bool(sound[a.id].get("with_music")) for a in audio]
    for uid in list(keep):
        lo = hi = order[uid]
        if not locked[lo]:
            continue
        while lo > 0 and locked[lo - 1]:
            lo -= 1
        while hi + 1 < len(audio) and locked[hi + 1]:
            hi += 1
        rest = [a.id for a in audio[lo:hi + 1] if a.id not in keep]
        if rest:
            issues.append(f"{uid} is part of a line that goes with the music ({audio[lo].id}-{audio[hi].id}): "
                          f"the rest of it is kept too ({' '.join(rest)})")
            keep += rest
    keep.sort(key=order.get)

    # ---- every gap between kept units is a cut in the music too: how does the music take it?
    cuts = []
    for x, y in zip(keep, keep[1:]):
        i, j = order[x], order[y]
        if j == i + 1:
            continue
        part = lambda k: sound[audio[k].id].get("music_part", "none")                        # noqa: E731
        if "none" in (sound[x].get("music", "none"), sound[y].get("music", "none")):
            verdict = "no music"                         # nothing to break on at least one side
        elif part(i + 1) != part(i) and part(j - 1) != part(j):
            verdict = "between music parts"              # leaves at the end of a part, lands at the start of one
        else:
            verdict = "inside the music"
            issues.append(f"the cut from {x} to {y} falls inside the music ({part(i)} > {part(j)}): it may be "
                          f"heard as a jump")
        cuts.append({"after": x, "before": y, "music": verdict})
    shown = {shot for uid in keep for shot in audio[order[uid]].over}
    for uid in [*keep, *sorted(shown)]:
        for n in needs.get(uid, []):
            if n not in order and n not in shown:
                issues.append(f"{uid} needs {n}, of which nothing is kept")

    segments: List[List[float]] = []
    for uid in keep:
        a = audio[order[uid]]
        if segments and abs(segments[-1][1] - a.start) < 1e-6:
            segments[-1][1] = a.end
        else:
            segments.append([a.start, a.end])
    issues += [f"the stretch {a:.2f}-{b:.2f}s is only {b - a:.2f}s long: it will be seen as a flash"
               for a, b in segments if b - a < SHORT_RUN_S]
    return {"story": str(raw.get("story") or "").strip(), "must_keep": must, "keep": keep, "segments": segments,
            "seconds": round(sum(b - a for a, b in segments), 3), "cuts": cuts}, issues


# =========================================================================== for people
def report(edit: Dict[str, Any], audio: Sequence[AudioUnit], described: Dict[str, Any]) -> str:
    da = {u["id"]: u for u in described["audio_units"]}
    why = {m["id"]: m["why"] for m in edit["must_keep"]}
    lines = [f"EDIT (synced): {edit['seconds']:.1f}s of {audio[-1].end:.1f}s, aiming for about {edit['target']:g}s",
             f"STORY: {edit['story']}", "", "UNITS (+ kept, ! kept as essential)"]
    for a in audio:
        u = da[a.id]
        mark = "!" if a.id in why else "+" if a.id in edit["keep"] else " "
        who = f"{u['speaker']}: " if u["speaker"] else ""
        reason = f"   <- {why[a.id]}" if why.get(a.id) else ""
        lines.append(f"  {mark} {a.id:<4} {a.start:6.2f}-{a.end:6.2f}s  {who}{u['heard']}{sound_tags(u, '  [', ']')}{reason}")
    cuts = edit.get("cuts") or []
    bad = sum(c["music"] == "inside the music" for c in cuts)
    lines += ["", f"CUTS: {len(cuts)}, of which {bad} inside the music (may be heard as a jump)"]
    lines += [f"  {c['after']:<4} > {c['before']:<4}  {c['music']}" for c in cuts]
    return "\n".join(lines)
