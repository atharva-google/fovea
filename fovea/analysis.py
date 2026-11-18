"""The base analysis: ONE Gemini call that watches the video and returns, in one answer,

  * the cast
  * for every video unit (shot): what is seen, who is in it, the setting, objects, on-screen text,
    what it depends on, and WHERE THE PICTURE SHOULD BE CENTRED at each moment code asks about
  * for every audio unit: what is really heard, what kind of sound it is, who says it, and what it
    depends on

The answer is checked and stored as three files, so that each can change on its own later:

    prep/units.json          the units cut by code (times, rough transcript, focus moments)
    prep/log.json            the cast and the description of every unit
    framing/framing.json     per shot: the focus moments, the position at each, and what it is on
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import gemini
from .config import Settings
from .job import Job, read_json, write_json
from .units import UNITS_VERSION, AudioUnit, VideoUnit, cut_units

ANALYSIS_VERSION = 1   # bump when the prompt or schema changes in a way that makes stored answers stale
KINDS = ("dialogue_onscreen", "dialogue_offscreen", "voiceover", "lyrics", "other")

PROMPT = """You are a senior video editor logging the footage of an ad before it is cut down and reframed.
Watch and listen to the whole video carefully. Later steps work from your log WITHOUT seeing the video,
so everything they need to know has to be in it.

The video has already been cut into units with exact times (listed below):
  VIDEO UNITS  one per shot (V01, V02, ...)
  AUDIO UNITS  a few spoken words, or the sound between words (A01, A02, ...). "over" names the shot(s)
               the sound plays over.
Refer to units ONLY by their id. Never write timestamps.
The rough transcript comes from speech-to-text and can have wrong words, miss sounds or invent words.
Trust what you hear.

Give:
  summary  one or two sentences: the story of the ad
  cast     everyone who speaks or is clearly the subject of at least one shot, as P1, P2, ... in order of first
           appearance. NOT background people or crowds. The same person is ONE entry however often they appear,
           also if their clothes or the lighting change. For each:
             real_name    their real name ONLY if this is a well-known person you recognise with confidence;
                          otherwise empty. Never guess.
             ad_name      the name the video itself gives them (spoken or on screen); otherwise empty
             description  how they LOOK, so they can be told apart: apparent age and gender, hair, build,
                          clothing. Nothing about what they do in the story.

For EVERY video unit give:
  shows    what is seen, in one sentence: who, doing what. People who are not in the cast list (a crowd,
           passers-by) are mentioned here.
  cast     the ids of the cast members recognisably visible at ANY point in this shot, even briefly, partly
           or from behind
  setting  where it is, as a short label ("living room", "beach", "studio, white background"). Use exactly the
           same wording for every shot in the same place.
  objects  things that matter to the story or that someone would name: the product, props that are handled,
           vehicles, animals. A few words each, and the same wording in every shot they appear in.
  text     readable on-screen text that belongs to THIS shot (a price, a caption, a sign, end-card wording),
           corrected. Leave out a logo or disclaimer that sits in the same place throughout the video.
  needs    earlier units this shot makes NO SENSE without (a reaction needs what it reacts to; a reveal needs
           its setup). A shot does NOT need another one just because it comes after it, shows the same scene or
           continues the same action. Most shots need nothing. Only the DIRECT ones.
  focus    The video will be cropped to a narrower picture (for example vertical) that uses the full height and
           can only move left and right. For EACH moment listed after "focus at" give one entry, in the same
           order and the same number of entries:
             x   where the cropped picture should be centred at that moment, left to right, as a percentage of
                 the frame width (0 = left edge, 50 = middle, 100 = right edge)
             on  what is at that position, in a few words ("P1's face", "the pack"). Fill it in for the first
                 moment of a shot and whenever it changes; otherwise leave it empty.
           Centre on what the viewer must see at that moment: the person speaking on screen, otherwise the
           subject of the shot or the product. While the subject stays where it is, repeat the SAME x: do not
           wobble by a few points. When two people talk in turn, x goes to whoever is speaking at that moment.

For EVERY audio unit give:
  heard    what is REALLY heard. Write speech out exactly, in its own language and script, correcting the rough
           transcript. Put any other sound in round brackets: (laughs), (door slams), (music), (silence).
  kind     dialogue_onscreen   spoken by someone whose face is SEEN saying it during this unit
           dialogue_offscreen  spoken by someone in the scene who is NOT seen saying it (out of frame, back to
                               the camera, or the picture shows someone or something else)
           voiceover           a narrator who is not seen saying it
           lyrics              sung words that are part of the music
           other               everything else: music, effects, a breath, a laugh, silence
  speaker  who says it: a cast id; "voiceover" for a narrator who is not in the cast; "other" for any other
           unseen or unlisted voice; empty when nothing is spoken
  needs    earlier units this sound makes NO SENSE without (a punchline needs its setup line). A sound does NOT
           need the one before it just because it follows it, and the parts of one sentence do not need each
           other. Most sounds need nothing. Only the DIRECT ones.
{json_note}
{table}
"""

JSON_NOTE = """
Think it through first if you like, then answer with ONE ```json block in exactly this shape:
```json
{"summary": "...",
 "cast": [{"id": "P1", "real_name": "", "ad_name": "", "description": "..."}],
 "video_units": [{"id": "V01", "shows": "...", "cast": ["P1"], "setting": "...", "objects": [], "text": [],
                  "needs": [], "focus": [{"x": 50, "on": "..."}, {"x": 50, "on": ""}]}],
 "audio_units": [{"id": "A01", "heard": "...", "kind": "other", "speaker": "", "needs": []}]}
```
Include EVERY video unit and EVERY audio unit.
"""


class KeyRequired(RuntimeError):
    """This video has not been analysed yet, and no Gemini key was given."""


# =========================================================================== the step
def run(job: Job, shots: Sequence[Dict[str, Any]], words: Sequence[Dict[str, Any]], settings: Settings,
        api_key: Optional[str] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The stored analysis of this video, or a new one from Gemini.
    Returns ``{"audio", "video", "log", "framing", "cached"}``."""
    stored = None if (settings.refresh or settings.force) else load(job)
    if stored:
        log(f"  analysis already stored ({job.log_json}), no Gemini call")
        return {**stored, "cached": True}
    if not api_key:
        raise KeyRequired("This video has not been analysed yet, so a Gemini key is needed: "
                          "pass --gemini-key or set GEMINI_API_KEY")
    audio, video = cut_units([(s["start_s"], s["end_s"]) for s in shots], words)
    if not video:
        raise RuntimeError("No shots found in this video")
    log(f"  {len(video)} video units (shots), {len(audio)} audio units "
        f"({sum(a.kind == 'speech' for a in audio)} with speech)")
    raw, seconds = gemini.ask_about_video(
        job.proxy, build_prompt(audio, video, settings.agentic), api_key, settings.gemini_model,
        settings.gemini_api_version, schema="Analysis", agentic=settings.agentic, log=log)
    described, framing, issues = clean(raw, audio, video)
    for issue in issues[:15]:
        log(f"  fixed: {issue}")
    save(job, audio, video, described, framing, {
        "model": settings.gemini_model, "mode": "agentic" if settings.agentic else "standard",
        "seconds": seconds, "issues": issues})
    log(f"  stored: {job.units_json.name}, {job.log_json.name} (prep/) and framing/{job.framing_json.name}")
    return {"audio": audio, "video": video, "log": described, "framing": framing, "cached": False}


# =========================================================================== the prompt
def build_prompt(audio: List[AudioUnit], video: List[VideoUnit], agentic: bool = False) -> str:
    lines = ["VIDEO UNITS"]
    for v in video:
        lines.append(f"{v.id} | {v.start:.2f}-{v.end:.2f}s | focus at: {', '.join(f'{t:.2f}' for t in v.focus_times)}")
    lines += ["", "AUDIO UNITS"]
    for a in audio:
        said = f'rough transcript: "{a.text}"' if a.kind == "speech" else "(no words transcribed)"
        lines.append(f"{a.id} | {a.start:.2f}-{a.end:.2f}s | over {' '.join(a.over)} | {said}")
    return PROMPT.replace("{json_note}", JSON_NOTE if agentic else "").replace("{table}", "\n".join(lines))


# =========================================================================== checking the answer
def clean(raw: Dict[str, Any], audio: List[AudioUnit], video: List[VideoUnit]
          ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """``(log, framing, issues)``: only known ids, every unit present, ``needs`` pointing backwards,
    and exactly one focus position per focus moment."""
    issues: List[str] = []
    start = {u.id: u.start for u in [*audio, *video]}

    def by_id(items: Any) -> Dict[str, Dict[str, Any]]:
        return {str(x.get("id", "")).strip(): x for x in items or [] if isinstance(x, dict)}

    def words(value: Any) -> List[str]:
        out: List[str] = []
        for x in value if isinstance(value, (list, tuple)) else []:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        return out

    def needs(uid: str, value: Any) -> List[str]:
        out = []
        for n in words(value):
            if n not in start:
                issues.append(f"{uid} needs unknown unit {n!r}: ignored")
            elif n == uid or start[n] > start[uid] + 1e-6:
                issues.append(f"{uid} needs {n}, which is not earlier: ignored")
            else:
                out.append(n)
        return out

    cast: List[Dict[str, str]] = []
    for c in raw.get("cast") or []:
        cid = str(c.get("id") or "").strip() if isinstance(c, dict) else ""
        if not cid or cid in {x["id"] for x in cast}:
            issues.append(f"cast entry with a missing or repeated id {cid!r} ignored")
            continue
        cast.append({"id": cid, **{k: str(c.get(k) or "").strip() for k in ("real_name", "ad_name", "description")}})
    cast_ids = {c["id"] for c in cast}

    got_v, got_a = by_id(raw.get("video_units")), by_id(raw.get("audio_units"))
    issues += [f"unknown unit {uid!r} ignored" for uid in [*got_v, *got_a] if uid not in start]

    video_out, shots_out = [], []
    for v in video:
        g = got_v.get(v.id)
        if g is None:
            issues.append(f"{v.id} missing from the answer")
            g = {}
        seen = words(g.get("cast"))
        issues += [f"{v.id} lists {c!r}, who is not in the cast: ignored" for c in seen if c not in cast_ids]
        video_out.append({"id": v.id, "shows": str(g.get("shows") or ""), "cast": [c for c in seen if c in cast_ids],
                          "setting": str(g.get("setting") or "").strip(), "objects": words(g.get("objects")),
                          "text": words(g.get("text")), "needs": needs(v.id, g.get("needs"))})
        points, note = _focus(g.get("focus"), len(v.focus_times))
        if note:
            issues.append(f"{v.id} focus: {note}")
        shots_out.append({"id": v.id, "times": list(v.focus_times), "x": [x for x, _ in points],
                          "on": [on for _, on in points]})

    audio_out = []
    for a in audio:
        g = got_a.get(a.id)
        if g is None:
            issues.append(f"{a.id} missing from the answer")
            g = {}
        kind = str(g.get("kind") or "").strip()
        if kind not in KINDS:
            # unknown: speech is assumed to be said on screen (the cautious choice for cutaways)
            kind = "dialogue_onscreen" if a.kind == "speech" else "other"
        speaker = str(g.get("speaker") or "").strip()
        if speaker and speaker not in cast_ids and speaker not in ("voiceover", "other"):
            issues.append(f"{a.id} is said by {speaker!r}, who is not in the cast: recorded as 'other'")
            speaker = "other"
        audio_out.append({"id": a.id, "heard": str(g.get("heard") or "") or (a.text if a.kind == "speech" else "(silence)"),
                          "kind": kind, "speaker": speaker, "needs": needs(a.id, g.get("needs"))})

    described = {"summary": str(raw.get("summary") or ""), "cast": cast, "video_units": video_out,
                 "audio_units": audio_out}
    return described, {"shots": shots_out}, issues


def _focus(entries: Any, n: int) -> Tuple[List[Tuple[int, str]], str]:
    """Exactly ``n`` ``(x, on)`` pairs from Gemini's entries, with ``on`` carried forward where it was
    left empty. A wrong number of entries is stretched evenly over the shot; none means the middle."""
    points: List[Tuple[int, str]] = []
    on = ""
    for e in entries if isinstance(entries, (list, tuple)) else []:
        try:
            x = float(e.get("x"))
        except (AttributeError, TypeError, ValueError):
            continue
        on = str(e.get("on") or "").strip() or on
        points.append((max(0, min(100, int(round(x)))), on))
    if not points:
        return [(50, "")] * n, "no positions given: the middle of the frame is used"
    if len(points) == n:
        return points, ""
    got = len(points)
    stretched = [points[int(round(k * (got - 1) / (n - 1)))] if n > 1 else points[0] for k in range(n)]
    return stretched, f"{got} positions for {n} moments: spread evenly over the shot"


# =========================================================================== storage
def save(job: Job, audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any],
         framing: Dict[str, Any], info: Dict[str, Any]) -> None:
    stamp = {"fingerprint": job.fingerprint, "version": [UNITS_VERSION, ANALYSIS_VERSION]}
    write_json(job.units_json, {**stamp, "video_units": [v.to_dict() for v in video],
                                "audio_units": [a.to_dict() for a in audio]})
    write_json(job.framing_json, {**stamp, "source": "gemini", **framing})
    write_json(job.log_json, {**stamp, **info, **described})          # written last = "the analysis is done"


def load(job: Job) -> Optional[Dict[str, Any]]:
    """``{"audio", "video", "log", "framing"}`` if all three files are there for this exact file and
    this version of the code."""
    stamp = {"fingerprint": job.fingerprint, "version": [UNITS_VERSION, ANALYSIS_VERSION]}
    try:
        units, described, framing = read_json(job.units_json), read_json(job.log_json), read_json(job.framing_json)
        if any({k: f.get(k) for k in stamp} != stamp for f in (units, described, framing)):
            return None
        audio = [AudioUnit.from_dict(d) for d in units["audio_units"]]
        video = [VideoUnit.from_dict(d) for d in units["video_units"]]
        if [u["id"] for u in described["video_units"]] != [v.id for v in video]:
            return None
        return {"audio": audio, "video": video, "log": described, "framing": framing}
    except (OSError, ValueError, KeyError, TypeError):
        return None


# =========================================================================== for people
def report(result: Dict[str, Any]) -> str:
    audio, video, described = result["audio"], result["video"], result["log"]
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    fr = {s["id"]: s for s in result["framing"]["shots"]}
    lines = [f"SUMMARY: {described['summary']}", "", "CAST"]
    for c in described["cast"]:
        names = " / ".join(n for n in (c["ad_name"], c["real_name"]) if n)
        lines.append(f"  {c['id']}{f' ({names})' if names else ''}: {c['description']}")
    lines += ["", "VIDEO UNITS (one per shot)"]
    for v in video:
        d, f = dv[v.id], fr[v.id]
        need = f"  needs {' '.join(d['needs'])}" if d["needs"] else ""
        lines.append(f"  {v.id}  {v.start:6.2f}-{v.end:6.2f}s{need}")
        lines.append(f"        shows: {d['shows']}")
        lines.append(f"        cast: {' '.join(d['cast']) or '-'}  |  setting: {d['setting'] or '-'}")
        if d["objects"]:
            lines.append(f"        objects: {'; '.join(d['objects'])}")
        if d["text"]:
            lines.append(f"        text: {'; '.join(d['text'])}")
        marks, last = [], None
        for t, x, on in zip(f["times"], f["x"], f["on"]):
            marks.append(f"{t:.1f}s {x}%" + (f" [{on}]" if on and on != last else ""))
            last = on
        lines.append(f"        focus: {'  '.join(marks)}")
    lines += ["", "AUDIO UNITS"]
    for a in audio:
        d = da[a.id]
        need = f"  needs {' '.join(d['needs'])}" if d["needs"] else ""
        who = f"{d['speaker']}: " if d["speaker"] else ""
        lines.append(f"  {a.id}  {a.start:6.2f}-{a.end:6.2f}s  over {' '.join(a.over):<8} {d['kind']:<18} "
                     f"{who}{d['heard']}{need}")
    return "\n".join(lines)
