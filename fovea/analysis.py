"""The log (the base analysis): ONE Gemini call that watches the video and returns, in one answer,

  * the cast
  * for every video unit (shot): what is seen, who is in it, the setting, objects, on-screen text
    and what it depends on
  * under each shot, for every sound that plays over it: what is really heard, what kind of sound
    it is, who says it, what it depends on, and WHERE THE PICTURE SHOULD BE CENTRED while it plays

Gemini is asked shot by shot, with the sounds of each shot under it. Because every focus position
belongs to a sound, code knows exactly when it applies: Gemini never has to say when.

The prompt here gives the task and the rules that span units. What each field means is written in
ONE place, schema.py, which Gemini is given alongside the prompt.

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

ANALYSIS_VERSION = 5   # bump when the prompt or schema changes in a way that makes stored answers stale
KINDS = ("dialogue_onscreen", "dialogue_offscreen", "voiceover", "lyrics", "other")
TALK = "dialogue_onscreen"
MUSIC = ("none", "score", "song")
MUSIC_PARTS = ("none", "intro", "build", "peak", "break", "outro")

PROMPT = """You are a senior video editor logging the footage of an ad before it is cut down and reframed.
Watch and listen to the whole video carefully. Later steps work from your log WITHOUT seeing the video,
so everything they need to know has to be in it.

UNITS
The video has already been cut into units with exact times, listed at the end (seconds from the start):
  video units  one per shot (V01, V02, ...)
  audio units  listed under the shot they play over (A01, A02, ...): a few spoken words, or up to a second of
               the sound between words. A sound that runs on into the next shot appears again there, marked
               (continued).
Describe every unit, in the order listed. For each one, first copy its id and time range exactly as listed,
then look at and listen to THAT stretch of the video, and nothing else. Everywhere else, refer to units by
their id and never by time.
The rough transcript comes from speech-to-text and can have wrong words, miss sounds or invent words.
Trust what you hear.

LOOKING AND LISTENING
- Judge every shot by its own frames. A shot shorter than a second is often a close-up of an object or a
  silent reaction: never carry people or actions over from the shot before it.
- With several people in a shot, check hands and mouths separately. One person often speaks while another
  does the thing that matters, and two people can say one line each within the same shot: for every audio
  unit, decide afresh whose mouth is moving and whether the voice has changed.

SOUND
The ad will also be cut shorter, and every cut is heard. So for EVERY audio unit, also when someone speaks:
- Say whether music plays under it and which part of the track that is. Music runs across many units: give
  neighbouring units in the same part the same label, and change the label exactly where the music changes.
- Note a sound made by something that is seen (a wrapper torn, a door, a bottle opened) on the unit where
  it is heard.

REFRAMING
The video will be cropped to a narrow picture that uses the full height and only moves left and right. For a
vertical video it is only about ONE THIRD of the frame's width; whatever is outside it is never seen.
focus_x says where that picture is centred while each sound plays. It follows what must be SEEN, which is not
always the person who is heard.
{fields}
THE UNITS
{table}
"""

FIELDS_NOTE = """
ANSWER
Think it through first if you like, then answer with ONE ```json block holding these fields, in this order:
{outline}
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
        settings.gemini_api_version, schema="Analysis", agentic=settings.agentic,
        thinking=settings.thinking, fps=settings.gemini_fps, log=log)
    described, framing, issues = clean(raw, audio, video)
    for issue in issues[:15]:
        log(f"  fixed: {issue}")
    save(job, audio, video, described, framing, {
        "model": settings.gemini_model, "mode": "agentic" if settings.agentic else "standard", "thinking": settings.thinking, "fps": settings.gemini_fps,
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
        lines.append(f"{v.id} | {_span(v.start, v.end)}")
        for s in under.get(v.id, []):
            a = sounds[s["audio"]]
            if s["part"]:
                said = "(continued)"
            else:
                said = f'rough transcript: "{a.text}"' if a.kind == "speech" else "(no words transcribed)"
            lines.append(f"    {s['id']} | {_span(s['start'], s['end'])} | {said}")
    fields = ""
    if agentic:                       # that mode takes no schema, so the fields are explained in the prompt
        from .schema import outline

        fields = FIELDS_NOTE.replace("{outline}", outline())
    return PROMPT.replace("{fields}", fields).replace("{table}", "\n".join(lines))


def _span(start: float, end: float) -> str:
    return f"{start:.2f}-{end:.2f}s"


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

    def anchored(uid: str, g: Dict[str, Any], start_s: float, end_s: float) -> None:
        """Gemini copies each unit's time range before describing it. A wrong copy means it lost its place."""
        echo = str(g.get("time_range") or "").replace(" ", "")
        if echo and echo.rstrip("s") != _span(start_s, end_s).rstrip("s"):
            issues.append(f"{uid} was described as {echo} but is {_span(start_s, end_s)}: "
                          f"it may describe another part of the video")

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
        anchored(v.id, g, v.start, v.end)
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
        anchored(s["id"], g, s["start"], s["end"])
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
        music = str(g.get("music") or "").strip()
        music = music if music in MUSIC else "none"
        part = str(g.get("music_part") or "").strip()
        row = {**s, "kind": kind, "speaker": speaker, "x": x,
               "heard": str(g.get("heard") or "").strip(), "needs": needs(a.id, g.get("needs")),
               "music": music, "music_part": part if part in MUSIC_PARTS and music != "none" else "none",
               "with_music": g.get("with_music") is True and kind != "other",
               "sfx": str(g.get("sfx") or "").strip()}
        rows.append(row)
        first.setdefault(a.id, row)

    audio_out = []
    for a in audio:
        parts = [r for r in rows if r["audio"] == a.id]
        head = parts[0]
        audio_out.append({"id": a.id, "heard": head["heard"] or (a.text if a.kind == "speech" else "(silence)"),
                          "kind": head["kind"], "speaker": head["speaker"], "needs": head["needs"],
                          "with_music": any(r["with_music"] for r in parts), "music": head["music"],
                          "music_part": head["music_part"], "sfx": next((r["sfx"] for r in parts if r["sfx"]), ""),
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
def sound_tags(unit: Dict[str, Any], before: str = " | ", after: str = "") -> str:
    """What is heard in a unit besides its words, as short text ("" when there is nothing to say)."""
    tags = []
    if unit.get("music", "none") != "none":
        tags.append(f"music: {unit['music']}/{unit.get('music_part', 'none')}")
    if unit.get("with_music"):
        tags.append("timed to the music")
    if unit.get("sfx"):
        tags.append(f"sound: {unit['sfx']}")
    return "".join(f"{before}{t}{after}" for t in tags)


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
                         f"{kind_of[s['id']]:<18} {heard}{need}{'' if s['part'] else sound_tags(a, '  [', ']')}")
    return "\n".join(lines)
