"""Where units + story map are kept: one folder per video FILE, named by the file's fingerprint.

    <root>/<sha256>/units.json       the units (times + measured facts)
    <root>/<sha256>/story_map.json   Gemini's cleaned answer, its raw answer, model, versions

The same file always gives the same fingerprint, so it always gets the same stored data: no second
Gemini call, no key. A stored entry is only replaced when a refresh is asked for, or when the unit
builder / prompt version changed (then the old entry no longer matches the code).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .story_map import PROMPT_VERSION
from .units import UNITS_VERSION, Unit


def default_root(cfg) -> Path:
    """Where a PipelineConfig keeps the store: ``cfg.story_dir``, else ``<work_dir>/_story``."""
    return Path(cfg.story_dir) if cfg.story_dir is not None else Path(cfg.work_dir) / "_story"


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


class StoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def folder(self, fp: str) -> Path:
        return self.root / fp

    def load(self, fp: str) -> Optional[Tuple[List[Unit], Dict[str, Any]]]:
        """``(units, record)`` if a usable entry exists; ``record["story_map"]`` is the map."""
        units_file, story_file = self.folder(fp) / "units.json", self.folder(fp) / "story_map.json"
        if not (units_file.is_file() and story_file.is_file()):
            return None
        try:
            u = json.loads(units_file.read_text())
            record = json.loads(story_file.read_text())
            units = [Unit.from_dict(d) for d in u["units"]]
        except (ValueError, KeyError, TypeError):
            return None  # unreadable entry = no entry
        if u.get("units_version") != UNITS_VERSION or record.get("prompt_version") != PROMPT_VERSION:
            return None
        if [m.get("id") for m in record.get("story_map", {}).get("units", [])] != [x.id for x in units]:
            return None
        return units, record

    def save(self, fp: str, units: List[Unit], result: Dict[str, Any], source_name: str = "") -> Path:
        """``result`` is what ``request_story_map`` returned. The story map is written last, through a
        temp file, so a half-written entry is never read as valid."""
        folder = self.folder(fp)
        folder.mkdir(parents=True, exist_ok=True)
        _write(folder / "units.json", {"fingerprint": fp, "source": source_name, "units_version": UNITS_VERSION,
                                       "units": [u.to_dict() for u in units]})
        _write(folder / "story_map.json", {"fingerprint": fp, "source": source_name,
                                           "prompt_version": PROMPT_VERSION, **result})
        return folder


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)
