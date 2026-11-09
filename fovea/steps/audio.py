"""STEP 3 - audio (optional, runs in PARALLEL with visual): vocals + transcript.

  * Demucs splits the audio into vocals.wav and no_vocals.wav
  * faster-whisper transcribes vocals.wav with word timestamps

Nothing in framing depends on this step, so the videos never wait for it.

Reads:  prep/audio.wav, prep/video_info.json
Writes: audio/vocals.wav, audio/no_vocals.wav, audio/transcript.json, audio/transcript.csv
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

from ..config import PipelineConfig
from ..utils.device import clean_gpu_memory
from ..utils.jobdir import JobDir
from ..utils.weights import ensure_for


def run(job: JobDir, cfg: PipelineConfig, device: str, log=print) -> None:
    out_json = job.audio_dir / "transcript.json"
    if out_json.exists() and not cfg.force:
        log("  transcript already done, skipping")
        return

    _, _, has_audio = job.video_infos()
    if not has_audio:
        job.write_json(out_json, [])
        return

    ensure_for(cfg, "demucs")
    whisper_dir = ensure_for(cfg, "whisper")

    t = time.time()
    vocals, _ = separate_vocals(job.audio, job.audio_dir, device)
    log(f"  demucs done ({time.time() - t:.1f}s)")

    t = time.time()
    transcript = transcribe(vocals, whisper_dir, device)
    log(f"  whisper done ({time.time() - t:.1f}s), {len(transcript)} segments")
    transcript.to_csv(job.audio_dir / "transcript.csv", index=False)
    job.write_json(out_json, transcript.to_dict("records"))


def separate_vocals(audio_path: Path, out_dir: Path, device: str) -> Tuple[Path, Path]:
    temp_dir = out_dir / "stems_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
         "-d", device, "-o", str(temp_dir), str(audio_path)],
        check=True, capture_output=True, text=True,
    )
    stem_dir = temp_dir / "htdemucs" / Path(audio_path).stem
    vocals, no_vocals = out_dir / "vocals.wav", out_dir / "no_vocals.wav"
    shutil.move(str(stem_dir / "vocals.wav"), vocals)
    shutil.move(str(stem_dir / "no_vocals.wav"), no_vocals)
    shutil.rmtree(temp_dir)
    return vocals, no_vocals


def transcribe(audio_path: Path, model_dir: Path, device: str) -> pd.DataFrame:
    from faster_whisper import WhisperModel

    wdevice = "cuda" if device == "cuda" else "cpu"  # CTranslate2 has no Apple MPS support
    model = WhisperModel(str(model_dir), device=wdevice, compute_type="float16" if wdevice == "cuda" else "int8")
    segments, _ = model.transcribe(str(audio_path), vad_filter=True, word_timestamps=True,
                                   temperature=0.0, condition_on_previous_text=False)
    rows = [{
        "audio_segment_id": i,
        "start_s": float(seg.start),
        "end_s": float(seg.end),
        "duration_s": float(seg.end - seg.start),
        "segment_transcript": seg.text.strip(),
        "words_data": [{"start": w.start, "end": w.end, "word": w.word.strip(),
                        "probability": round(w.probability, 5)} for w in (seg.words or [])],
    } for i, seg in enumerate(segments)]
    del model
    clean_gpu_memory()
    return pd.DataFrame(rows)
