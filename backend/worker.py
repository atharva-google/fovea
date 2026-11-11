"""Runs pipeline steps for one job. Used in BOTH modes:

    MODE=gcp:    each Cloud Run Job task runs ``python -m backend worker --step X --job-path ...``
    MODE=local:  the API's background runner calls ``run_job()`` directly

Every step is:  copy inputs from storage  ->  fovea.steps.<step>.run()  ->  copy outputs to storage
(in local mode the job folder IS the storage, so the copies are skipped).
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from fovea import steps
from fovea.config import PipelineConfig
from fovea.utils.device import get_device
from fovea.utils.jobdir import JobDir, step_files

from .jobs import get_jobs, step_event
from .settings import get_settings
from .storage import get_storage


def log(msg: str) -> None:
    print(msg, flush=True)


class StageReporter:
    """Progress callback for ``fovea.steps.*.run(progress=...)``.

    Writes the step's current stage into the job record, e.g.::

        steps.visual.stage       = {"name": "ocr", "index": 3, "count": 4, "done": 40, "total": 120,
                                    "unit": "shots", "elapsed_s": 6.2, "shard": 0}
        steps.visual.stage_times = {"tracking": 31.4, "speakers": 12.0}   # finished stages, seconds

    Stage changes are written (and logged) at once; counter updates at most every ``min_interval`` s.
    """

    def __init__(self, job_id: str, step: str, shard: Optional[int] = None, min_interval: float = 2.0):
        self.job_id, self.step, self.shard, self.min_interval = job_id, step, shard, min_interval
        self.prefix = f"shard{shard}/" if shard else ""  # keep stage times of parallel shards apart
        self.stage: Optional[str] = None
        self.t_stage = self.t_write = 0.0
        self.times: Dict[str, float] = {}
        self.lock = threading.Lock()

    def __call__(self, stage: str, done: Optional[int] = None, total: Optional[int] = None,
                 unit: Optional[str] = None, index: Optional[int] = None, count: Optional[int] = None) -> None:
        with self.lock:
            now_t = time.monotonic()
            if stage != self.stage:
                self._close_stage(now_t)
                self.stage, self.t_stage = stage, now_t
                pos = f" {index}/{count}" if index and count else ""
                log(f"  [{self.step}{'' if self.shard is None else f' shard {self.shard}'}] stage{pos}: {stage}")
            elif now_t - self.t_write < self.min_interval and not (total and done is not None and done >= total):
                return
            self.t_write = now_t
            info: Dict[str, Any] = {"name": stage, "index": index, "count": count, "done": done, "total": total,
                                    "unit": unit, "elapsed_s": round(now_t - self.t_stage, 1)}
            if self.shard is not None:
                info["shard"] = self.shard
            self._write({f"steps.{self.step}.stage": info, f"steps.{self.step}.stage_times": dict(self.times)})

    def finish(self) -> None:
        with self.lock:
            self._close_stage(time.monotonic())
            self.stage = None
            self._write({f"steps.{self.step}.stage": None, f"steps.{self.step}.stage_times": dict(self.times)})

    def _close_stage(self, now_t: float) -> None:
        if self.stage is not None:
            secs = round(now_t - self.t_stage, 1)
            self.times[self.prefix + self.stage] = secs
            log(f"  [{self.step}] ✓ {self.stage} ({secs:.1f}s)")

    def _write(self, fields: Dict[str, Any]) -> None:
        try:
            get_jobs().update(self.job_id, fields)
        except Exception as e:  # noqa: BLE001 - progress must never break the pipeline
            log(f"  (could not record progress: {e})")


def _run_with_stages(job_id: str, step: str, fn, *args, shard: Optional[int] = None, **kwargs):
    reporter = StageReporter(job_id, step, shard)
    try:
        return fn(*args, progress=reporter, **kwargs)
    finally:
        reporter.finish()


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
        n = _run_with_stages(job_id, "prep", steps.prep.run, job, cfg, log)
        extra = {"num_shards": n, "steps.visual.total": n}
    elif step == "visual":
        _run_with_stages(job_id, "visual", steps.visual.run, job, cfg, device, shard, log, shard=shard)
    elif step == "audio":
        steps.audio.run(job, cfg, device, log)
    elif step == "render":
        extra = _outputs_field(_run_with_stages(job_id, "render", steps.render.run, job, cfg, log))
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
    n = _run_with_stages(job_id, "prep", steps.prep.run, job, cfg, log)
    _sync("up", prefix, job, ["prep/"])
    step_event(job_id, "prep", "done", extra={"num_shards": n, "steps.visual.total": n})

    pool = ThreadPoolExecutor(max_workers=1)
    audio_future = None
    if cfg.transcribe:
        step_event(job_id, "audio", "running")
        audio_future = pool.submit(steps.audio.run, job, cfg, device, lambda m: log(f"[audio]{m}"))

    step_event(job_id, "visual", "running")
    for i in range(n):
        _run_with_stages(job_id, "visual", steps.visual.run, job, cfg, device, i, log, shard=i if n > 1 else None)
        step_event(job_id, "visual", "done", shard=i)
    _sync("up", prefix, job, ["visual/"])
    step_event(job_id, "visual", "done")

    step_event(job_id, "render", "running")
    rendered = _run_with_stages(job_id, "render", steps.render.run, job, cfg, log)
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
