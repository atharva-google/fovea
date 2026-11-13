"""Where units + story map are kept: in the video's own folder, next to the rest of its analysis.

    <data folder>/<video name>/prep/units.json       the units (times + measured facts)
    <data folder>/<video name>/prep/story_map.json   Gemini's cleaned answer, its raw answer, model, versions

Both files record the fingerprint (SHA-256) of the video file they were made from. They are reused
only for that exact file, so the same video always gets the same data (no second Gemini call, no
key), and a different video that happens to have the same name never inherits them. They are
replaced when a refresh is asked for, or when the unit builder / prompt version changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .story_map import PROMPT_VERSION
from .units import UNITS_VERSION, Unit

UNITS_FILE = "units.json"
STORY_FILE = "story_map.json"


def fingerprint(path: str | Path, chunk: int = 4 * 1024 * 1024) -> str:
    """SHA-256 of the file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load(folder: str | Path, fp: str) -> Optional[Tuple[List[Unit], Dict[str, Any]]]:
    """``(units, record)`` if ``folder`` holds a usable entry made from the file with fingerprint
    ``fp``; ``record["story_map"]`` is the map."""
    units_file, story_file = Path(folder) / UNITS_FILE, Path(folder) / STORY_FILE
    if not (units_file.is_file() and story_file.is_file()):
        return None
    try:
        u = json.loads(units_file.read_text())
        record = json.loads(story_file.read_text())
        units = [Unit.from_dict(d) for d in u["units"]]
    except (ValueError, KeyError, TypeError):
        return None  # unreadable entry = no entry
    if u.get("fingerprint") != fp or record.get("fingerprint") != fp:
        return None  # made from a different file
    if u.get("units_version") != UNITS_VERSION or record.get("prompt_version") != PROMPT_VERSION:
        return None
    if [m.get("id") for m in record.get("story_map", {}).get("units", [])] != [x.id for x in units]:
        return None
    return units, record


def save(folder: str | Path, fp: str, units: List[Unit], result: Dict[str, Any], source_name: str = "") -> Path:
    """``result`` is what ``request_story_map`` returned. The story map is written last, through a
    temp file, so a half-written entry is never read as valid."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / UNITS_FILE, {"fingerprint": fp, "source": source_name, "units_version": UNITS_VERSION,
                                 "units": [u.to_dict() for u in units]})
    _write(folder / STORY_FILE, {"fingerprint": fp, "source": source_name,
                                 "prompt_version": PROMPT_VERSION, **result})
    return folder


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)
