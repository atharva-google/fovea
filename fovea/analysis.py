"""The base analysis: ONE Gemini call that watches the video and returns, in one answer,

  * the cast
  * for every video unit (shot): what is seen, who is in it, the setting, objects, on-screen text
    and what it depends on
  * under each shot, for every sound that plays over it: what is really heard, what kind of sound
    it is, who says it, what it depends on, and WHERE THE PICTURE SHOULD BE CENTRED while it plays

Gemini is asked shot by shot, with the sounds of each shot under it. Because every focus position
belongs to a sound, code knows exactly when it applies: Gemini never has to say when.

The answer is checked and stored as three files, so that each can change on its own later:

    prep/units.json          the units cut by code (times, rough transcript)
    prep/log.json            the cast and the description of every video and audio unit
    framing/framing.json     per shot: focus points (when, where, cut or glide); see framing.py
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import gemini
from .config import Settings
from .framing import position_at
from .job import Job, read_json, write_json
from .units import UNITS_VERSION, AudioUnit, VideoUnit, cut_units, slots

ANALYSIS_VERSION = 3   # bump when the prompt or schema changes in a way that makes stored answers stale
KINDS = ("dialogue_onscreen", "dialogue_offscreen", "voiceover", "lyrics", "other")
TALK = "dialogue_onscreen"

PROMPT = """You are a senior video editor logging the footage of an ad before it is cut down and reframed.
Watch and listen to the whole video carefully. Later steps work from your log WITHOUT seeing the video,
so everything they need to know has to be in it.

The video has already been cut into units with exact times (listed below):
  VIDEO UNITS  one per shot (V01, V02, ...)
  AUDIO UNITS  listed under the shot they play over (A01, A02, ...): a few spoken words, or up to a second of
               the sound between words. A sound that runs on into the next shot appears again there, marked
               (continued).
Refer to units ONLY by their id. Never write timestamps.
The rough transcript comes from speech-to-text and can have wrong words, miss sounds or invent words.
Trust what you hear.

Give:
  summary  one or two sentences: the story of the ad
  cast     ONLY the people who play a part in the story: they speak, are spoken to, or do something the story
           depends on. As P1, P2, ... in order of first appearance. Someone who is merely present (sitting in the
           room, part of a crowd, passing by) is NOT cast, however clearly they are seen. The same person is ONE
           entry however often they appear, also if their clothes or the lighting change. For each:
             real_name    their real name ONLY if this is a well-known person you recognise with confidence;
                          otherwise empty. Never guess.
             ad_name      the name the video itself gives them (spoken or on screen); otherwise empty
             description  how they LOOK, so they can be told apart: apparent age and gender, hair, build,
                          clothing. Nothing about what they do in the story.

For EVERY video unit give:
  shows    what is seen, in one sentence: who, doing what. People who are not in the cast are mentioned here.
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
  audio    every audio unit listed under this shot, in the same order. For each:
    heard    what is REALLY heard. Write speech out exactly, in its own language and script, correcting the
             rough transcript. Put any other sound in round brackets: (laughs), (door slams), (music), (silence).
             Leave it empty for a unit marked (continued).
    kind     dialogue_onscreen   spoken by someone whose face is SEEN saying it during this unit
             dialogue_offscreen  spoken by someone in the scene who is NOT seen saying it (out of frame, back
                                 to the camera, or the picture shows someone or something else)
             voiceover           a narrator who is not seen saying it
             lyrics              sung words that are part of the music
             other               everything else: music, effects, a breath, a laugh, silence
    speaker  the cast id of who says it. Decide by whose mouth is moving, not by who is most prominent in the
             shot. Empty when nothing is spoken, or the voice is not a cast member.
    needs    earlier units this sound makes NO SENSE without (a punchline needs its setup line). A sound does
             NOT need the one before it just because it follows it. Most sounds need nothing.
    focus_x  The video will be cropped to a narrow picture that uses the full height and only moves left and
             right. For a vertical video it is only about ONE THIRD of the frame's width; whatever is outside
             it is never seen. Give where that picture should be centred WHILE this sound plays, left to right,
             as a percentage of the frame's width (0 = left edge, 50 = middle, 100 = right edge).
             It is the centre of ONE person or thing, never the space between two. Choose, in this order:
               1. the action or movement that is the punchline or the point of the shot
               2. the main subject of the shot
               3. in a wide shot with several people: the people the cropped picture has already shown in
                  earlier shots, not someone its viewer has never seen
               4. when someone speaks on screen: that person
             If the subject moves during a shot, each unit gets the position where it is at that time.
             While nothing changes, repeat exactly the same number.
{json_note}
{table}
"""

JSON_NOTE = """
Think it through first if you like, then answer with ONE ```json block in exactly this shape:
```json
{"summary": "...",
 "cast": [{"id": "P1", "real_name": "", "ad_name": "", "description": "..."}],
 "video_units": [{"id": "V01", "shows": "...", "cast": ["P1"], "setting": "...", "objects": [], "text": [], "needs": [],
                  "audio": [{"id": "A01", "heard": "...", "kind": "other", "speaker": "", "needs": [], "focus_x": 50}]}]}
```
Include EVERY video unit, and under each one EVERY audio unit listed there.
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
    sounds = {a.id: a for a in audio}
    under: Dict[str, List[Dict[str, Any]]] = {}
    for s in slots(audio, video):
        under.setdefault(s["shot"], []).append(s)
    lines = []
    for v in video:
        lines.append(f"{v.id} | {v.start:.2f}-{v.end:.2f}s")
        for s in under.get(v.id, []):
            a = sounds[s["audio"]]
            if s["part"]:
                said = "(continued)"
            else:
                said = f'rough transcript: "{a.text}"' if a.kind == "speech" else "(no words transcribed)"
            lines.append(f"    {s['id']} | {s['start']:.2f}-{s['end']:.2f}s | {said}")
    return PROMPT.replace("{json_note}", JSON_NOTE if agentic else "").replace("{table}", "\n".join(lines))


# =========================================================================== checking the answer
def clean(raw: Dict[str, Any], audio: List[AudioUnit], video: List[VideoUnit]
          ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """``(log, framing, issues)``: only known ids, every unit present, ``needs`` pointing backwards,
    and a focus position for every slot, turned into points in time (see ``framing.py``)."""
    issues: List[str] = []
    start = {u.id: u.start for u in [*audio, *video]}
    sounds = {a.id: a for a in audio}
    all_slots = slots(audio, video)

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
            n = n if n in start else n.rstrip("abcdefgh")      # "A05b" names a part of A05
            if n not in start:
                issues.append(f"{uid} needs unknown unit {n!r}: ignored")
            elif n == uid or start[n] > start[uid] + 1e-6:
                issues.append(f"{uid} needs {n}, which is not earlier: ignored")
            elif n not in out:
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

    got_v = by_id(raw.get("video_units"))
    issues += [f"unknown unit {uid!r} ignored" for uid in got_v if uid not in start]
    got_s: Dict[str, Dict[str, Any]] = {}          # Gemini's answer per slot, wherever it put it
    for g in got_v.values():
        got_s.update(by_id(g.get("audio")))
    known_slots = {s["id"] for s in all_slots}
    issues += [f"unknown unit {sid!r} ignored" for sid in got_s if sid not in known_slots]

    video_out = []
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

    # ---- every slot: kind, speaker and focus position
    rows: List[Dict[str, Any]] = []
    first: Dict[str, Dict[str, Any]] = {}          # audio unit -> the row of its first part
    last_x: Dict[str, int] = {}                    # shot -> the last position given in it
    for s in all_slots:
        a = sounds[s["audio"]]
        g = got_s.get(s["id"])
        if g is None:
            issues.append(f"{s['id']} missing from the answer")
            g = {}
        head = first.get(a.id)
        kind = str(g.get("kind") or "").strip()
        if kind not in KINDS:
            # unknown: speech is assumed to be said on screen (the cautious choice for cutaways)
            kind = head["kind"] if head else (TALK if a.kind == "speech" else "other")
        speaker = head["speaker"] if head else str(g.get("speaker") or "").strip()
        if speaker and speaker not in cast_ids:
            if speaker not in ("voiceover", "other"):
                issues.append(f"{s['id']} is said by {speaker!r}, who is not in the cast: left empty")
            speaker = ""
        try:
            x = max(0, min(100, int(round(float(g.get("focus_x"))))))
        except (TypeError, ValueError):
            x = last_x.get(s["shot"], 50)
            issues.append(f"{s['id']} has no focus position: {x}% is used")
        last_x[s["shot"]] = x
        row = {**s, "kind": kind, "speaker": speaker, "x": x,
               "heard": str(g.get("heard") or "").strip(), "needs": needs(a.id, g.get("needs"))}
        rows.append(row)
        first.setdefault(a.id, row)

    audio_out = []
    for a in audio:
        parts = [r for r in rows if r["audio"] == a.id]
        head = parts[0]
        audio_out.append({"id": a.id, "heard": head["heard"] or (a.text if a.kind == "speech" else "(silence)"),
                          "kind": head["kind"], "speaker": head["speaker"], "needs": head["needs"],
                          "parts": [{"id": r["id"], "shot": r["shot"], "kind": r["kind"]} for r in parts]})

    shots_out = [{"id": v.id, "points": _points(v, [r for r in rows if r["shot"] == v.id])} for v in video]
    described = {"summary": str(raw.get("summary") or ""), "cast": cast, "video_units": video_out,
                 "audio_units": audio_out}
    return described, {"shots": shots_out}, issues


def _points(shot: VideoUnit, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The focus positions of one shot's slots as points in time.

    The picture JUMPS to a position when someone starts or stops speaking on screen, or the speaker
    changes: that happens at the slot's start, which is a time code knows exactly. Otherwise the
    subject has moved, and the picture GLIDES: a position is where the subject is in the MIDDLE of
    its slot, and the picture moves evenly from one middle to the next."""
    if not rows:
        return [{"t": shot.start, "move": "cut", "x": 50, "slot": ""}]

    def middle(r: Dict[str, Any]) -> float:
        return round((r["start"] + r["end"]) / 2, 3)

    points = [{"t": shot.start, "move": "cut", "x": rows[0]["x"], "slot": rows[0]["id"]}]
    for prev, r in zip(rows, rows[1:]):
        same_voice = r["kind"] == prev["kind"] == TALK and r["speaker"] == prev["speaker"]
        if r["x"] == prev["x"]:
            continue
        if (r["kind"] == TALK or prev["kind"] == TALK) and not same_voice:
            points.append({"t": r["start"], "move": "cut", "x": r["x"], "slot": r["id"]})
        else:
            if points[-1]["t"] < middle(prev):       # stay put until the middle of the slot before
                points.append({"t": middle(prev), "move": "glide", "x": prev["x"], "slot": prev["id"]})
            points.append({"t": middle(r), "move": "glide", "x": r["x"], "slot": r["id"]})
    return points


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
    kind_of = {p["id"]: p["kind"] for u in described["audio_units"] for p in u["parts"]}
    x_at = {s["id"]: s["points"] for s in result["framing"]["shots"]}
    lines = [f"SUMMARY: {described['summary']}", "", "CAST"]
    for c in described["cast"]:
        names = " / ".join(n for n in (c["ad_name"], c["real_name"]) if n)
        lines.append(f"  {c['id']}{f' ({names})' if names else ''}: {c['description']}")
    lines += ["", "SHOTS, with the sounds over each (focus = where the picture is centred; 40% > 60% = it glides)"]
    under: Dict[str, List[Dict[str, Any]]] = {}
    for s in slots(audio, video):
        under.setdefault(s["shot"], []).append(s)
    for v in video:
        d = dv[v.id]
        need = f"  needs {' '.join(d['needs'])}" if d["needs"] else ""
        lines.append(f"{v.id}  {v.start:.2f}-{v.end:.2f}s{need}")
        lines.append(f"    shows: {d['shows']}")
        lines.append(f"    cast: {' '.join(d['cast']) or '-'}  |  setting: {d['setting'] or '-'}")
        if d["objects"]:
            lines.append(f"    objects: {'; '.join(d['objects'])}")
        if d["text"]:
            lines.append(f"    text: {'; '.join(d['text'])}")
        for s in under.get(v.id, []):
            a = da[s["audio"]]
            x0, x1 = (round(position_at(x_at[v.id], t)) for t in (s["start"], s["end"] - 0.001))
            focus = f"{x0:3d}%" + (f" > {x1:3d}%" if x1 != x0 else "       ")
            who = f"{a['speaker']}: " if a["speaker"] else ""
            heard = "(continued)" if s["part"] else f"{who}{a['heard']}"
            need = f"  needs {' '.join(a['needs'])}" if a["needs"] and not s["part"] else ""
            lines.append(f"    {s['id']:<5} {s['start']:6.2f}-{s['end']:6.2f}s  focus {focus}  "
                         f"{kind_of[s['id']]:<18} {heard}{need}")
    return "\n".join(lines)
