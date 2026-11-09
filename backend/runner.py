"""Who RUNS a job once it leaves the queue.

    LocalRunner      (MODE=local)  a background thread in the API process runs every step
                                   on this machine (worker.run_job)
    WorkflowsRunner  (MODE=gcp)    starts a Cloud Workflows execution, which runs each step
                                   as a Cloud Run GPU job task (backend/gcp/pipeline.yaml)

``dispatch_queued()`` starts queued jobs while fewer than MAX_CONCURRENT_JOBS are running.
``handle_event()`` records "rendered" / "complete" / "failed" and frees the slot.
"""

from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Optional

from .jobs import get_jobs, now
from .settings import get_settings

log = logging.getLogger("fovea.runner")


class LocalRunner:
    def __init__(self, max_workers: int):
        self.pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="fovea-job")

    def start(self, job: Dict[str, Any]) -> None:
        self.pool.submit(self._run, job["job_id"], job["prefix"])

    @staticmethod
    def _run(job_id: str, prefix: str) -> None:
        from .worker import run_job

        try:
            run_job(prefix, on_rendered=lambda: handle_event(job_id, "rendered"))
            handle_event(job_id, "complete")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            handle_event(job_id, "failed", f"{type(e).__name__}: {e}")


class WorkflowsRunner:
    def start(self, job: Dict[str, Any]) -> None:
        from google.cloud.workflows import executions_v1

        s = get_settings()
        client = executions_v1.ExecutionsClient()
        args = {
            "job_id": job["job_id"],
            "job_path": job["prefix"],
            "bucket": s.bucket,
            "api_url": s.self_url,
            "worker_job": s.worker_job_name,
            "split_steps": s.split_steps,
            "transcribe": bool(job.get("config", {}).get("transcribe", False)),
        }
        execution = client.create_execution(
            parent=client.workflow_path(s.project, s.region, s.workflow_name),
            execution=executions_v1.Execution(argument=json.dumps(args)),
        )
        get_jobs().update(job["job_id"], {"workflow_execution": execution.name})


@lru_cache(maxsize=1)
def get_runner():
    s = get_settings()
    return LocalRunner(s.max_concurrent_jobs) if s.is_local else WorkflowsRunner()


def dispatch_queued() -> int:
    """Start as many queued jobs as there are free slots. Returns how many were started."""
    s, store, started = get_settings(), get_jobs(), 0
    while True:
        job = store.claim_next(s.max_concurrent_jobs, s.stale_job_minutes)
        if job is None:
            return started
        try:
            get_runner().start(job)
            started += 1
        except Exception as e:  # noqa: BLE001
            log.exception("could not start job %s", job["job_id"])
            store.update(job["job_id"], {"status": "failed", "error": f"Could not start: {e}", "finished_at": now()})


def handle_event(job_id: str, event: str, error: Optional[str] = None) -> None:
    store = get_jobs()
    job = store.get(job_id)
    if job is None:
        return
    if event == "rendered":
        store.update(job_id, {"status": "rendered", "rendered_at": now()})
    elif event == "complete":
        store.update(job_id, {"status": "complete", "finished_at": now()})
    elif event == "failed":
        # prefer the step's readable error (e.g. "Video is 900s long...") over a raw stack/workflow error
        step_errors = [st.get("error") for st in (job.get("steps") or {}).values() if isinstance(st, dict) and st.get("error")]
        msg = step_errors[0] if step_errors else (error or "pipeline failed")
        store.update(job_id, {"status": "failed", "error": str(msg)[-2000:], "finished_at": now()})
    if event in ("complete", "failed"):
        dispatch_queued()  # a slot just freed up
