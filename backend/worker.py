"""Runs pipeline steps for one job. Used in BOTH modes:

    MODE=gcp:    each Cloud Run Job task runs ``python -m backend worker --step X --job-path ...``
    MODE=local:  the API's background runner calls ``run_job()`` directly

Every step is:  copy inputs from storage  ->  fovea.steps.<step>.run()  ->  copy outputs to storage
(in local mode the job folder IS the storage, so the copies are skipped).
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from fovea import steps
from fovea.config import PipelineConfig
from fovea.utils.device import get_device
from fovea.utils.jobdir import JobDir, step_files

from .jobs import step_event
from .settings import get_settings
from .storage import get_storage


def log(msg: str) -> None:
    print(msg, flush=True)


def _job_id(prefix: str) -> str:
    return prefix.rstrip("/").rsplit("/", 1)[-1]


def _config(job: JobDir) -> PipelineConfig:
    return PipelineConfig(work_dir=job.root.parent, output_dir=job.outputs_dir).update(job.job_config())


def _outputs_field(rendered: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    return {"outputs": {label: f"outputs/{Path(p).name}" for label, p in rendered.items()}}


def _sync(direction: str, prefix: str, job: JobDir, entries) -> None:
    store, t = get_storage(), time.time()
    if direction == "down":
        store.download(prefix, entries, job.root)
    else:
        store.upload(job.root, entries, prefix)
    if not get_settings().is_local:
        log(f"  {'↓' if direction == 'down' else '↑'} {entries} ({time.time() - t:.1f}s)")


# =========================================================================== one step (cloud)
def run_step(step: str, prefix: str, shard: int = 0) -> None:
    job_id = _job_id(prefix)
    job = JobDir(get_storage().job_workdir(prefix))
    inputs, outputs = step_files(step, shard if step == "visual" else None)
    step_event(job_id, step, "running")

    _sync("down", prefix, job, inputs)
    cfg, device = _config(job), get_device()

    extra: Dict = {}
    if step == "prep":
        n = steps.prep.run(job, cfg, log)
        extra = {"num_shards": n, "steps.visual.total": n}
    elif step == "visual":
        steps.visual.run(job, cfg, device, shard, log)
    elif step == "audio":
        steps.audio.run(job, cfg, device, log)
    elif step == "render":
        extra = _outputs_field(steps.render.run(job, cfg, log))
    else:
        raise ValueError(f"Unknown step {step}")

    _sync("up", prefix, job, outputs)
    step_event(job_id, step, "done", shard=shard if step == "visual" else None, extra=extra)


# =========================================================================== whole job
def run_job(prefix: str, on_rendered=None) -> None:
    """All steps in one process; audio runs in a background thread next to visual.
    Used by local mode, and by the cloud when SPLIT_STEPS=false (single GPU)."""
    job_id = _job_id(prefix)
    job = JobDir(get_storage().job_workdir(prefix))
    _sync("down", prefix, job, ["job.json", "input/"])
    cfg, device = _config(job), get_device()

    step_event(job_id, "prep", "running")
    n = steps.prep.run(job, cfg, log)
    _sync("up", prefix, job, ["prep/"])
    step_event(job_id, "prep", "done", extra={"num_shards": n, "steps.visual.total": n})

    pool = ThreadPoolExecutor(max_workers=1)
    audio_future = None
    if cfg.transcribe:
        step_event(job_id, "audio", "running")
        audio_future = pool.submit(steps.audio.run, job, cfg, device, lambda m: log(f"[audio]{m}"))

    step_event(job_id, "visual", "running")
    for i in range(n):
        steps.visual.run(job, cfg, device, i, log)
        step_event(job_id, "visual", "done", shard=i)
    _sync("up", prefix, job, ["visual/"])
    step_event(job_id, "visual", "done")

    step_event(job_id, "render", "running")
    rendered = steps.render.run(job, cfg, log)
    _sync("up", prefix, job, ["framing/", "outputs/", "debug/"])
    step_event(job_id, "render", "done", extra=_outputs_field(rendered))
    if on_rendered:
        on_rendered()  # videos are ready even if audio is still running

    if audio_future is not None:
        try:
            audio_future.result()
            _sync("up", prefix, job, ["audio/"])
            step_event(job_id, "audio", "done")
        except Exception as e:  # noqa: BLE001 - videos are delivered even if audio fails
            traceback.print_exc()
            step_event(job_id, "audio", "failed", error=f"{type(e).__name__}: {e}")
    pool.shutdown(wait=False)


# =========================================================================== CLI entry (cloud task)
def main(step: str, job_path: str) -> int:
    prefix = job_path.strip("/")
    shard = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    log(f"[worker] step={step} job={prefix} task={shard}")
    try:
        if step == "all":
            run_job(prefix)
        else:
            run_step(step, prefix, shard)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        step_event(_job_id(prefix), step, "failed", error=f"{type(e).__name__}: {e}")
        return 1  # non-zero exit -> task fails -> Workflows marks the job failed
    log("[worker] done")
    return 0
