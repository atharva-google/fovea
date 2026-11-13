"""The story map: ONE Gemini call that watches the video and scores / links the units by id.

Gemini never writes timestamps (except optional "moments"), so all timing stays exact. The answer
is free text with one ```json block; if that can't be parsed, a cheap structured call reformats it.

Ported from notebooks/experiments/story_map_experiment.ipynb (branch exp/story-map).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .units import Unit, units_table

PROMPT_VERSION = 1  # bump when STORY_PROMPT changes in a way that makes stored story maps stale
REFORMAT_MODEL = "gemini-3.8-flash"
REQUEST_TIMEOUT_S = 300     # no single request to Gemini may hang longer than this
PROCESSING_TIMEOUT_S = 180  # how long Gemini may take to make an uploaded video usable

STORY_PROMPT = """You are a senior video editor. Watch this video (rewatch any part as often as you need)
and build a STORY MAP that a program will use to cut shorter versions of it (e.g. 20 s, 15 s, 10 s).

The video has been cut into UNITS with exact times (below). Refer to units ONLY by their id (U01, U02, ...);
never invent timestamps except in "moments". A "speech" unit is a phrase someone says; a "picture" unit is a
stretch without speech. Measured facts per unit, from our analysis software:
  people          most persons detected on screen at once
  visible speaker share of the unit's frames in which active speaker detection sees an on-screen person talking
                  (speech with a low share is likely voiceover or off-screen)
  on-screen text  text read by OCR in that shot
Use them, but trust what you see and hear when they disagree.

For every unit decide:
  importance 0-10  how much the short version needs it to tell the story (0 = pure filler / dead time)
  kind             one of: dialogue (someone in the scene talks), voiceover (narration / off-screen speech),
                   reaction (a visible reaction, no talking), visual (an action, object or scene worth seeing),
                   filler (nothing worth keeping)
  shows            what is SEEN in this unit (short)
  note             why it matters (or why it doesn't)

Also give:
  needs     units that only make sense together (a punchline needs its setup; an answer needs its question)
  cutaways  for speech units: picture or reaction units that could be shown INSTEAD of the speaker while the line
            plays (e.g. show the gift while she names it). Never propose a unit where someone is visibly talking.
  hook      the most gripping unit(s) to open a short version with, if moving them first helps
  moments   optional exact peaks (seconds in the source video) inside a unit, e.g. the frame the reaction lands

Answer with your reasoning first if you like, then ONE ```json block exactly in this shape:
```json
{
  "summary": "one or two sentences: the story of the video",
  "units": [{"id": "U01", "importance": 0, "kind": "filler", "shows": "...", "note": "..."}],
  "needs": [{"unit": "U07", "requires": ["U03"], "why": "..."}],
  "cutaways": [{"speech": "U07", "pictures": ["U11"], "fit": 9, "why": "..."}],
  "hook": {"units": ["U08"], "why": "..."},
  "moments": [{"unit": "U08", "peak_s": 21.4, "note": "..."}]
}
```
Include EVERY unit in "units".

UNITS:
{units}
"""


def story_prompt(units: List[Unit]) -> str:
    return STORY_PROMPT.replace("{units}", units_table(units))


# ----------------------------------------------------------------------------- parsing
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Last ```json block; else the last {...} that parses."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    start = (text or "").rfind("{\n")
    while start != -1:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.rfind("{\n", 0, start)
    return None


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_story_map(raw: Dict[str, Any], units: List[Unit]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only known unit ids, clamp numbers, fill units Gemini forgot (importance 0)."""
    ids = {u.id for u in units}
    issues: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for u in raw.get("units", []) or []:
        if not isinstance(u, dict):
            continue
        uid = str(u.get("id", "")).strip()
        if uid not in ids:
            issues.append(f"unknown unit {uid!r} ignored")
            continue
        by_id[uid] = {"id": uid, "importance": max(0.0, min(10.0, _num(u.get("importance", 0), 0.0))),
                      "kind": str(u.get("kind", "filler")), "shows": str(u.get("shows", "")),
                      "note": str(u.get("note", ""))}
    for u in units:
        if u.id not in by_id:
            issues.append(f"{u.id} missing from the answer -> importance 0")
            by_id[u.id] = {"id": u.id, "importance": 0.0, "kind": "filler", "shows": "", "note": "(missing)"}

    needs = [{"unit": n["unit"], "requires": [r for r in n.get("requires") or [] if r in ids],
              "why": str(n.get("why", ""))}
             for n in raw.get("needs", []) or [] if isinstance(n, dict) and n.get("unit") in ids]
    cutaways = []
    for c in raw.get("cutaways", []) or []:
        if isinstance(c, dict) and c.get("speech") in ids:
            pics = [p for p in c.get("pictures") or [] if p in ids]
            if pics:
                cutaways.append({"speech": c["speech"], "pictures": pics, "fit": _num(c.get("fit", 5), 5.0),
                                 "why": str(c.get("why", ""))})
    hook = raw.get("hook") if isinstance(raw.get("hook"), dict) else {}
    hook = {"units": [h for h in hook.get("units") or [] if h in ids], "why": str(hook.get("why", ""))}
    moments = [m for m in raw.get("moments", []) or [] if isinstance(m, dict) and m.get("unit") in ids]
    return {"summary": str(raw.get("summary", "")), "units": [by_id[u.id] for u in units], "needs": needs,
            "cutaways": cutaways, "hook": hook, "moments": moments}, issues


# ----------------------------------------------------------------------------- the Gemini call
class GeminiError(RuntimeError):
    """A Gemini call failed. The message never contains the API key."""


def _safe(message: str, api_key: str) -> str:
    return message.replace(api_key, "***") if api_key else message


def request_story_map(video_path: str | Path, units: List[Unit], api_key: str,
                      model: str = "gemini-3.8-flash", processing: str = "AGENTIC",
                      log: Callable[[str], None] = print, api_version: str = "v1beta") -> Dict[str, Any]:
    """Upload the video, ask for the story map, return it cleaned.

    Returns ``{"story_map", "issues", "raw_answer", "model", "processing", "seconds", "parsed_by"}``.
    ``processing`` is the video processing mode ("AGENTIC" lets Gemini rewatch parts; "" or "STATIC"
    = the standard one). It is ignored, with a log line, if the installed google-genai has no such option.
    ``api_version``: "v1beta" is what the experiment ran on (newer google-genai releases default to
    "v1"). "" = the SDK's default.
    """
    if not api_key:
        raise GeminiError("A Gemini API key is required to build the story map")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise GeminiError("google-genai is not installed: pip install -U google-genai") from e

    http = {"timeout": REQUEST_TIMEOUT_S * 1000}   # milliseconds
    if api_version:
        http["api_version"] = api_version
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http))
    uploaded = None
    try:
        log(f"  uploading {Path(video_path).name} to Gemini")
        uploaded = client.files.upload(file=str(video_path))
        log("  uploaded; waiting for Gemini to make the video usable")
        waited = 0.0
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if waited > PROCESSING_TIMEOUT_S:
                raise GeminiError(f"Gemini is still processing the uploaded video after {PROCESSING_TIMEOUT_S}s; "
                                  "run the command again")
            time.sleep(2)
            waited += 2
            uploaded = client.files.get(name=uploaded.name)
        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise GeminiError(f"Gemini could not process the video (state: {getattr(uploaded.state, 'name', '?')})")

        file_data = types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
        used = (processing or "").upper()
        if used and used != "STATIC" and not hasattr(types, "MediaProcessing"):
            log("  this google-genai version has no agentic video option; using standard video processing")
            used = ""
        video_part = types.Part(file_data=file_data, media_processing=used) if used else types.Part(file_data=file_data)

        log(f"  asking {model} for the story map ({used or 'standard'} video processing; "
            f"gives up after {REQUEST_TIMEOUT_S}s)")
        t0 = time.time()
        resp = client.models.generate_content(model=model, contents=[video_part, story_prompt(units)])
        seconds = round(time.time() - t0, 1)
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        log(f"  {model} / {used or 'standard'}: {seconds}s | tokens in={getattr(usage, 'prompt_token_count', None)} "
            f"out={getattr(usage, 'candidates_token_count', None)} "
            f"thinking={getattr(usage, 'thoughts_token_count', None)}")

        raw = extract_json(text)
        parsed_by = "json block"
        if raw is None:
            parsed_by = "reformat call"
            log("  the answer had no parsable json block; asking for a structured reformat")
            raw = _reformat(client, types, text)
        story, issues = clean_story_map(raw, units)
        for issue in issues[:10]:
            log(f"  story map: {issue}")
        return {"story_map": story, "issues": issues, "raw_answer": text, "model": model,
                "processing": used or "STATIC", "seconds": seconds, "parsed_by": parsed_by}
    except GeminiError:
        raise
    except Exception as e:  # noqa: BLE001 - SDK errors can echo request details; never leak the key
        if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
            raise GeminiError(f"Gemini did not answer within {REQUEST_TIMEOUT_S}s; run the command again "
                              "(everything analysed so far is kept)") from None
        raise GeminiError(_safe(f"Gemini call failed: {type(e).__name__}: {e}", api_key)) from None
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001 - uploaded files expire on their own
                pass


def _reformat(client, types, text: str) -> Dict[str, Any]:
    """Fallback: a cheap structured call that only converts the free-text answer into JSON."""
    from typing import List as L

    from pydantic import BaseModel

    class SUnit(BaseModel):
        id: str
        importance: float
        kind: str
        shows: str
        note: str

    class SNeed(BaseModel):
        unit: str
        requires: L[str]
        why: str

    class SCut(BaseModel):
        speech: str
        pictures: L[str]
        fit: float
        why: str

    class SHook(BaseModel):
        units: L[str]
        why: str

    class SMoment(BaseModel):
        unit: str
        peak_s: float
        note: str

    class SMap(BaseModel):
        summary: str
        units: L[SUnit]
        needs: L[SNeed]
        cutaways: L[SCut]
        hook: SHook
        moments: L[SMoment]

    fix = client.models.generate_content(
        model=REFORMAT_MODEL,
        contents=["Convert this video editor's answer into the JSON schema. Keep every value; invent nothing.\n\n"
                  + text],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SMap))
    return json.loads(fix.text)
