"""One question to Gemini: about a video (``ask_about_video``) or about text only (``ask``)."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

REQUEST_TIMEOUT_S = 600      # no single request may hang longer than this
PROCESSING_TIMEOUT_S = 180   # how long Gemini may take to make an uploaded video usable


class GeminiError(RuntimeError):
    """A Gemini call failed. The message never contains the API key."""


def ask(prompt: str, api_key: str, model: str, api_version: str = "v1beta", schema: Optional[str] = None,
        thinking: str = "default", log: Callable[[str], None] = print) -> Tuple[Dict[str, Any], float]:
    """Ask one question in text, return ``(the answer as a dict, seconds it took)``. The answer is
    forced into ``schema`` (a class name in ``schema.py``)."""
    client, types = _client(api_key, api_version)
    with _errors(api_key):
        log(f"  asking {model} (text only, thinking: {thinking}; gives up after {REQUEST_TIMEOUT_S}s)")
        return _generate(client, model, [prompt], _config(types, schema, thinking), log)


def ask_about_video(video_path: str | Path, prompt: str, api_key: str, model: str, api_version: str = "v1beta",
                    schema: Optional[str] = None, agentic: bool = False, thinking: str = "default",
                    fps: Optional[float] = None,
                    log: Callable[[str], None] = print) -> Tuple[Dict[str, Any], float]:
    """Upload a video, ask one question, return ``(the answer as a dict, seconds it took)``.

    Standard: the answer is forced into ``schema`` (a class name in ``schema.py``).
    ``agentic``: agentic video processing (Gemini may rewatch parts). That mode takes no output
    schema, so the prompt must ask for one ```json block, which is parsed here.
    ``thinking``: how hard the model thinks before it answers: "minimal", "low", "medium" or "high";
    "default" leaves it to the model.
    ``fps``: how many frames per second of the video the model looks at (standard mode; its own
    default is 1, which misses shots shorter than a second).
    The upload is deleted afterwards."""
    client, types = _client(api_key, api_version)
    if agentic and not hasattr(types, "MediaProcessing"):
        raise GeminiError("this google-genai version has no agentic video processing: pip install -U google-genai")
    uploaded = None
    try:
        with _errors(api_key):
            log(f"  uploading {Path(video_path).name} to Gemini")
            uploaded = client.files.upload(file=str(video_path))
            waited = 0.0
            while uploaded.state and uploaded.state.name == "PROCESSING":
                if waited > PROCESSING_TIMEOUT_S:
                    raise GeminiError(f"Gemini is still processing the upload after {PROCESSING_TIMEOUT_S}s; run again")
                time.sleep(2)
                waited += 2
                uploaded = client.files.get(name=uploaded.name)
            if not uploaded.state or uploaded.state.name != "ACTIVE":
                raise GeminiError(f"Gemini could not process the video (state: {getattr(uploaded.state, 'name', '?')})")

            file_data = types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
            if agentic:
                part = types.Part(file_data=file_data, media_processing="AGENTIC")
            else:
                part = types.Part(file_data=file_data,
                                  video_metadata=types.VideoMetadata(fps=fps) if fps else None)
            log(f"  asking {model} ({'agentic' if agentic else 'standard'} video processing, "
                f"{f'{fps:g} frames/s, ' if fps and not agentic else ''}thinking: {thinking}; "
                f"gives up after {REQUEST_TIMEOUT_S}s)")
            return _generate(client, model, [part, prompt],
                             _config(types, None if agentic else schema, thinking, json_only=not agentic), log)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001 - uploads expire on their own
                pass


# =========================================================================== shared
def _client(api_key: str, api_version: str):
    if not api_key:
        raise GeminiError("A Gemini API key is required: pass --gemini-key or set GEMINI_API_KEY")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise GeminiError(f"google-genai is not installed: pip install -U google-genai ({e})") from e
    http = types.HttpOptions(api_version=api_version, timeout=REQUEST_TIMEOUT_S * 1000)
    return genai.Client(api_key=api_key, http_options=http), types


def _config(types, schema: Optional[str], thinking: str, json_only: bool = True):
    options: Dict[str, Any] = {}
    if thinking != "default":
        options["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    if json_only:
        from . import schema as schemas

        options.update(response_mime_type="application/json",
                       response_schema=getattr(schemas, schema) if schema else None)
    return types.GenerateContentConfig(**options) if options else None


def _generate(client, model: str, contents: list, config, log: Callable[[str], None]) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    seconds = round(time.time() - t0, 1)
    usage = getattr(resp, "usage_metadata", None)
    log(f"  answered in {seconds}s | tokens in={getattr(usage, 'prompt_token_count', None)} "
        f"out={getattr(usage, 'candidates_token_count', None)} "
        f"thinking={getattr(usage, 'thoughts_token_count', None)}")
    answer = extract_json(resp.text or "")
    if answer is None:
        raise GeminiError("Gemini's answer held no valid JSON (it may have been cut off); run again")
    return answer, seconds


@contextmanager
def _errors(api_key: str) -> Iterator[None]:
    """Any failure becomes a GeminiError whose message never contains the key."""
    try:
        yield
    except GeminiError:
        raise
    except Exception as e:  # noqa: BLE001 - SDK errors can echo request details
        if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
            raise GeminiError(f"Gemini did not answer within {REQUEST_TIMEOUT_S}s; run again") from None
        raise GeminiError(f"Gemini call failed: {type(e).__name__}: {e}".replace(api_key, "***")) from None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The whole text if it is JSON; else the last ```json block; else the last {...} that parses."""
    candidates = [text] + list(reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)))
    start = text.rfind("{\n")
    while start != -1:
        candidates.append(text[start:])
        start = text.rfind("{\n", 0, start)
    for c in candidates:
        try:
            value = json.loads(c)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
