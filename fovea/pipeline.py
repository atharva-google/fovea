"""Run the whole pipeline on this machine (CLI / Colab / Python).

    prep ──┬── visual (each shard) ── render ──> videos
           └── audio (optional, background thread) ──> transcript
    cutdowns (``cut_durations``): ... ── story (units + Gemini story map, or the stored one) ── render the cuts

The backend (backend/) calls exactly the same ``steps.*.run()`` functions.
"""

from __future__ import annotations

import os
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


def run(video_path: str | Path, config: Optional[PipelineConfig] = None, gemini_api_key: Optional[str] = None,
        **overrides: Any) -> List[Path]:
    """Reframe ``video_path`` into every format in ``config.targets``.

    Example::

        from fovea import run
        run("input.mp4", targets=["9:16", "1:1"])
        run("input.mp4", targets=["9:16"], cut_durations=[15, 10])   # + 15 s and 10 s cutdowns

    Cutdowns need a Gemini key (``gemini_api_key`` or $GEMINI_API_KEY) only the first time a video
    file is seen; after that its stored story map is used.
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

    key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    need_story = False
    if cfg.cut_durations:
        fp, stored_units, _ = steps.story.lookup(job, cfg)
        need_story = stored_units is None
        # The folder is named after the file name: never cut a video using another file's analysis.
        if job.video_info.exists() and job.read_json(job.video_info).get("sha256") not in (None, fp):
            print("  this folder holds the analysis of a different file with the same name: re-running it")
            cfg.force = True
        if need_story and not key:   # fail now, not after minutes of analysis
            raise steps.story.StoryKeyRequired(
                "Cutdowns need a Gemini key the first time a video is seen: pass --gemini-key or set GEMINI_API_KEY")
        if need_story:
            cfg.transcribe = True    # the units are built from the transcript

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

    if cfg.cut_durations:
        with _step("cutdowns"):
            rendered.update(_render_cuts(job, cfg, key, audio_future if need_story else None))

    # The final videos stay in the video's own folder (<video name>/outputs/<format>.mp4).
    # Only when an output folder was asked for are they also copied there.
    outputs = [Path(path) for path in rendered.values()]
    if cfg.output_dir is not None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
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


def _render_cuts(job: JobDir, cfg: PipelineConfig, key: Optional[str], audio_future) -> dict:
    """Story map (stored or new) -> cuts picked by code -> rendered. A failure here is reported and
    leaves the full-length videos untouched."""
    from .story.report import format_cuts, make_cuts

    try:
        if audio_future is not None:
            print("  waiting for the transcript", flush=True)
            audio_future.result()
        story = steps.story.run(job, cfg, key)
        cuts = make_cuts(story["units"], story["record"]["story_map"], cfg.cut_durations, cfg.cut_modes,
                         **cfg.cut_params)
        job.write_json(job.framing_dir / "cuts.json", cuts)
        print(format_cuts(cuts), flush=True)
        return steps.render.run_cuts(job, cfg, cuts)
    except Exception as e:  # noqa: BLE001 - the full-length videos are done; don't fail the run
        print(f"  ✗ cutdowns failed: {type(e).__name__}: {e}", flush=True)
        return {}
