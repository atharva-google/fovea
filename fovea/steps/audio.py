"""STEP 3 - audio (optional, runs in PARALLEL with visual): vocals + transcript.

  * Demucs (in-process, model loaded once) splits the audio into vocals.wav and no_vocals.wav
  * faster-whisper transcribes vocals.wav with word timestamps. ``BatchedInferencePipeline``
    cuts the speech into <=30 s chunks with VAD and decodes several chunks per GPU pass, so
    long videos are not limited by Whisper's 30 s window and are much faster than the
    sequential sliding window.

Nothing in framing depends on this step, so the videos never wait for it.

Reads:  prep/audio.wav, prep/video_info.json
Writes: audio/vocals.wav, audio/no_vocals.wav, audio/transcript.json, audio/transcript.csv
"""

from __future__ import annotations

import inspect
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
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
    job.audio_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    vocals, _ = separate_vocals(job.audio, job.audio_dir, device)
    log(f"  demucs done ({time.time() - t:.1f}s)")

    t = time.time()
    transcript = transcribe(vocals, whisper_dir, device, cfg.whisper_batch_size)
    log(f"  whisper done ({time.time() - t:.1f}s), {len(transcript)} segments")
    transcript.to_csv(job.audio_dir / "transcript.csv", index=False)
    job.write_json(out_json, transcript.to_dict("records"))


def separate_vocals(audio_path: Path, out_dir: Path, device: str) -> Tuple[Path, Path]:
    """Same processing as ``demucs --two-stems vocals -n htdemucs`` (default shifts/overlap/clip),
    but in this process: no Python start-up, no model reload, no temp folders."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    # decoded with ffmpeg ourselves: demucs.audio needs torchaudio's I/O backends, which newer torchaudio dropped
    wav = torch.from_numpy(load_audio(audio_path, model.samplerate, model.audio_channels).T.copy())

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    if not torch.isfinite(std) or float(std) < 1e-8:  # silent track: avoid dividing by zero
        std = torch.tensor(1.0)
    wav = (wav - mean) / std
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25,
                              progress=False, num_workers=0)[0]
    sources = sources * std + mean

    stems = list(sources)
    vocals_t = stems.pop(model.sources.index("vocals"))
    no_vocals_t = torch.stack(stems).sum(0)

    vocals, no_vocals = out_dir / "vocals.wav", out_dir / "no_vocals.wav"
    write_wav(vocals, vocals_t.cpu().numpy().T, model.samplerate)
    write_wav(no_vocals, no_vocals_t.cpu().numpy().T, model.samplerate)
    del model, sources, stems, wav
    clean_gpu_memory()
    return vocals, no_vocals


def transcribe(audio_path: Path, model_dir: Path, device: str, batch_size: int = 0) -> pd.DataFrame:
    from faster_whisper import WhisperModel

    wdevice = "cuda" if device == "cuda" else "cpu"  # CTranslate2 has no Apple MPS support
    model = WhisperModel(str(model_dir), device=wdevice, compute_type="float16" if wdevice == "cuda" else "int8")
    options = dict(vad_filter=True, word_timestamps=True, temperature=0.0, condition_on_previous_text=False)
    # 16 kHz mono float32 decoded with ffmpeg: passing a file path makes faster-whisper decode with PyAV,
    # and newer PyAV releases break its decoder ("open() got an unexpected keyword argument 'metadata_errors'")
    audio = load_audio(audio_path, 16000, 1)[:, 0]
    try:
        from faster_whisper import BatchedInferencePipeline  # faster-whisper >= 1.1
    except ImportError:
        BatchedInferencePipeline = None

    pipe = None
    if BatchedInferencePipeline is not None:
        pipe = BatchedInferencePipeline(model=model)
        bs = batch_size or (16 if wdevice == "cuda" else 4)
        accepted = inspect.signature(pipe.transcribe).parameters
        kwargs = {k: v for k, v in options.items() if k in accepted}
        segments, _ = pipe.transcribe(audio, batch_size=bs, **kwargs)
    else:  # old faster-whisper: sequential 30 s sliding window
        segments, _ = model.transcribe(audio, **options)

    rows = [{
        "audio_segment_id": i,
        "start_s": float(seg.start),
        "end_s": float(seg.end),
        "duration_s": float(seg.end - seg.start),
        "segment_transcript": seg.text.strip(),
        "words_data": [{"start": w.start, "end": w.end, "word": w.word.strip(),
                        "probability": round(w.probability, 5)} for w in (seg.words or [])],
    } for i, seg in enumerate(segments)]
    del pipe, model
    clean_gpu_memory()
    return pd.DataFrame(rows)


def load_audio(path: Path, samplerate: int, channels: int) -> np.ndarray:
    """Decode any audio/video file with ffmpeg -> float32 array of shape (samples, channels)."""
    res = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-f", "f32le",
                          "-ac", str(channels), "-ar", str(samplerate), "-"], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}:\n{res.stderr.decode(errors='ignore')[-1500:]}")
    return np.frombuffer(res.stdout, np.float32).reshape(-1, channels)


def write_wav(path: Path, audio: np.ndarray, samplerate: int) -> Path:
    """16-bit PCM wav, scaled down if it would clip (same as Demucs' default ``clip="rescale"``)."""
    from scipy.io import wavfile

    audio = np.asarray(audio, np.float32)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    audio = audio / max(1.01 * peak, 1.0)
    wavfile.write(str(path), samplerate, (audio * 32767).astype(np.int16))
    return Path(path)
