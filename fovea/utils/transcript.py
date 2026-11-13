"""Spot a broken speech transcript.

Whisper sometimes gets stuck repeating one character or word ("।।।।।।।…", "the the the the …").
When that happens everything after that point in the audio chunk is lost, so the transcript simply
stops early. These helpers detect that, so the audio step can try again in a safer mode and say
clearly when the transcript is still not usable.

No third-party imports (tested in tests/test_transcript.py).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_RUN = re.compile(r"(.)\1{5,}", flags=re.S)        # the same character 6+ times in a row
MAX_WORD_CHARS = 40
MAX_SAME_WORD = 6


def transcript_problems(rows: List[Dict[str, Any]]) -> List[str]:
    """Reasons to distrust a transcript (``rows`` as written by the audio step); empty = looks fine."""
    problems: List[str] = []
    words = [str(w.get("word", "")) for r in rows for w in (r.get("words_data") or [])]
    texts = [str(r.get("segment_transcript", "")) for r in rows]
    for t in words + texts:
        m = _RUN.search(t)
        if m:
            problems.append(f"the character {m.group(1)!r} repeats {len(m.group(0))} times in a row")
            break
    long_word = next((w for w in words if len(w) > MAX_WORD_CHARS), None)
    if long_word:
        problems.append(f"a single 'word' is {len(long_word)} characters long")
    run, prev = 1, None
    for w in (x.strip().lower() for x in words):
        run = run + 1 if w and w == prev else 1
        prev = w
        if run > MAX_SAME_WORD:
            problems.append(f"the word {w!r} repeats more than {MAX_SAME_WORD} times in a row")
            break
    return problems


def last_word_end(rows: List[Dict[str, Any]]) -> float:
    """When the last transcribed word ends, in seconds (0 if there are no words)."""
    ends = [float(w["end"]) for r in rows for w in (r.get("words_data") or []) if w.get("end") is not None]
    return max(ends, default=0.0)
