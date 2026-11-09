"""Run the whole pipeline on this machine (CLI / Colab / Python).

    prep ──┬── visual (each shard) ── render ──> videos
           └── audio (optional, background thread) ──> transcript

The backend (backend/) calls exactly the same ``steps.*.run()`` functions.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

from . import steps
from .config import PipelineConfig
from .framing.geometry import OutputTarget
from .utils.device import get_device
from .utils.jobdir import JobDir


@contextmanager
def _step(name: str):
    print(f"▶ {name} ...", flush=True)
    t0 = time.time()
    yield
    print(f"  ✓ {name} ({time.time() - t0:.1f}s)", flush=True)


def run(video_path: str | Path, config: Optional[PipelineConfig] = None, **overrides: Any) -> List[Path]:
    """Reframe ``video_path`` into every format in ``config.targets``.

    Example::

        from fovea import run
        run("input.mp4", targets=["9:16", "1:1"])
    """
    cfg = (config or PipelineConfig()).update(overrides)
    for t in cfg.targets:
        OutputTarget.parse(t)  # fail early on a typo
    device = get_device(cfg.device)

    video_path = Path(video_path)
    job = JobDir(cfg.work_dir / video_path.stem)
    job.set_source(video_path)
    job.write_json(job.job_json, {"config": {"targets": cfg.targets}})
    print(f"Fovea | device={device} | job folder: {job.root}")

    with _step("1/4 prep"):
        n_shards = steps.prep.run(job, cfg)

    pool = ThreadPoolExecutor(max_workers=1)
    audio_future = None
    if cfg.transcribe:
        print("▶ 3/4 audio started in the background", flush=True)
        audio_future = pool.submit(steps.audio.run, job, cfg, device, lambda m: print(f"  [audio]{m}", flush=True))

    for i in range(n_shards):
        with _step(f"2/4 visual (shard {i + 1}/{n_shards})"):
            steps.visual.run(job, cfg, device, i)

    with _step("4/4 render"):
        rendered = steps.render.run(job, cfg)

    outputs = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for label, path in rendered.items():
        dest = cfg.output_dir / f"{video_path.stem}_{label}.mp4"
        shutil.copy2(path, dest)
        outputs.append(dest)

    if audio_future is not None:
        print("… waiting for the audio step", flush=True)
        try:
            audio_future.result()
            print(f"  ✓ transcript: {job.audio_dir / 'transcript.csv'}")
        except Exception as e:  # noqa: BLE001 - the videos are done; don't fail the run
            print(f"  ✗ audio step failed: {e}")
    pool.shutdown(wait=False)
    return outputs
