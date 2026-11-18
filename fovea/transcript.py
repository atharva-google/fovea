"""Transcript: every word with its times (faster-whisper), stored as audio/transcript.csv.

The words themselves may be wrong; Gemini corrects them later. What matters here is the TIMES:
every cut lands on them.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

COLUMNS = ["word", "start_s", "end_s", "probability", "segment"]


def transcribe(audio_path: str | Path, model: str = "small", device: str = "cpu", language: Optional[str] = None,
               weights_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """``[{"word", "start_s", "end_s", "probability", "segment"}]`` in order.

    Whisper can get stuck repeating one character or word, losing the rest of that stretch of audio.
    The decoding options below (retries at higher temperatures, repeats penalised) are the ones that
    got it unstuck on real footage."""
    from faster_whisper import WhisperModel

    from .media import load_audio

    whisper = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8",
                           download_root=str(weights_dir) if weights_dir else None)
    segments, _ = whisper.transcribe(
        load_audio(audio_path), language=language, vad_filter=True, word_timestamps=True,
        condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        no_repeat_ngram_size=3, repetition_penalty=1.1)
    words = []
    for i, seg in enumerate(segments):
        for w in seg.words or []:
            if w.end > w.start and w.word.strip():
                words.append({"word": w.word.strip(), "start_s": round(float(w.start), 3),
                              "end_s": round(float(w.end), 3), "probability": round(float(w.probability), 3),
                              "segment": i})
    return words


_RUN = re.compile(r"(.)\1{5,}", flags=re.S)     # the same character 6+ times in a row


def problems(words: List[Dict[str, Any]]) -> List[str]:
    """Reasons to distrust a transcript (signs that Whisper got stuck); empty = looks fine."""
    found: List[str] = []
    texts = [str(w["word"]) for w in words]
    run = next((m for m in (_RUN.search(t) for t in texts) if m), None)
    if run:
        found.append(f"the character {run.group(1)!r} repeats {len(run.group(0))} times in a row")
    long = next((t for t in texts if len(t) > 40), None)
    if long:
        found.append(f"a single 'word' is {len(long)} characters long")
    streak, prev = 1, None
    for t in (x.lower() for x in texts):
        streak = streak + 1 if t == prev else 1
        prev = t
        if streak > 6:
            found.append(f"the word {t!r} repeats more than 6 times in a row")
            break
    return found


def write_csv(path: str | Path, words: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(words)


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{"word": r["word"], "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                 "probability": float(r["probability"]), "segment": int(r["segment"])} for r in csv.DictReader(f)]
