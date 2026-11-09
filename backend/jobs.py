"""Where job RECORDS (status, progress, outputs) live.

    LocalJobStore      (MODE=local)  one JSON file per job in ./data/_jobs/  (open them to inspect!)
    FirestoreJobStore  (MODE=gcp)    one Firestore document per job

A job record looks like::

    job_id, user_id, prefix ("users/<uid>/<job_id>"), filename, input_key
    status:   awaiting_upload -> queued -> running -> rendered -> complete     (or failed)
    config:   {targets, ...options}
    steps:    {prep|visual|audio|render: {status, started_at, finished_at, error, done: [shards], total}}
    outputs:  {"9x16": "outputs/9x16.mp4", ...}
    created_at, queued_at, started_at, rendered_at, finished_at, error
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import get_settings

ACTIVE = ("running", "rendered")  # statuses that occupy a job slot (GPU)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _set_dotted(doc: Dict[str, Any], dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    for p in parents:
        doc = doc.setdefault(p, {})
    doc[last] = value


# =========================================================================== local JSON files
class LocalJobStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "_jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quota_file = Path(data_dir) / "_quotas.json"
        self.lock = threading.RLock()

    def _file(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _read(self, path: Path) -> Dict[str, Any]:
        d = json.loads(path.read_text())
        for k, v in d.items():  # timestamps back to datetime
            if k.endswith("_at") and isinstance(v, str):
                d[k] = datetime.fromisoformat(v)
        return d

    def _write(self, job: Dict[str, Any]) -> None:
        tmp = self._file(job["job_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o)))
        tmp.replace(self._file(job["job_id"]))

    def _all(self) -> List[Dict[str, Any]]:
        return [self._read(p) for p in self.dir.glob("*.json")]

    def create(self, job: Dict[str, Any]) -> None:
        with self.lock:
            self._write(job)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        p = self._file(job_id)
        return self._read(p) if p.exists() else None

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        with self.lock:
            job = self.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                _set_dotted(job, k, v)
            self._write(job)

    def add_done_shard(self, job_id: str, step: str, shard: int) -> None:
        with self.lock:
            job = self.get(job_id)
            done = job.setdefault("steps", {}).setdefault(step, {}).setdefault("done", [])
            if shard not in done:
                done.append(shard)
            self._write(job)

    def list_for_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        jobs = [j for j in self._all() if j.get("user_id") == user_id]
        jobs.sort(key=lambda j: j.get("created_at") or now(), reverse=True)
        return jobs[:limit]

    def consume_daily_quota(self, user_id: str, limit: int) -> bool:
        with self.lock:
            quotas = json.loads(self.quota_file.read_text()) if self.quota_file.exists() else {}
            key = f"{user_id}_{now():%Y%m%d}"
            if quotas.get(key, 0) >= limit:
                return False
            quotas[key] = quotas.get(key, 0) + 1
            self.quota_file.write_text(json.dumps(quotas, indent=2))
            return True

    def claim_next(self, max_concurrent: int, stale_minutes: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            jobs = self._all()
            cutoff = now() - timedelta(minutes=stale_minutes)
            live = 0
            for j in jobs:
                if j.get("status") in ACTIVE:
                    if j.get("started_at") and j["started_at"] < cutoff:
                        self.update(j["job_id"], {"status": "failed", "error": "timed out", "finished_at": now()})
                    else:
                        live += 1
            queued = [j for j in jobs if j.get("status") == "queued"]
            if live >= max_concurrent or not queued:
                return None
            job = min(queued, key=lambda j: j.get("queued_at") or now())
            self.update(job["job_id"], {"status": "running", "started_at": now()})
            return self.get(job["job_id"])

    def queued_jobs(self) -> List[Dict[str, Any]]:
        return [j for j in self._all() if j.get("status") == "queued"]

    def fail_unfinished(self, reason: str) -> None:
        """On a local restart, jobs that were mid-run can't resume; mark them failed."""
        for j in self._all():
            if j.get("status") in ACTIVE:
                self.update(j["job_id"], {"status": "failed", "error": reason, "finished_at": now()})


# =========================================================================== Firestore
class FirestoreJobStore:
    def __init__(self, project: str, database: str, collection: str):
        from google.cloud import firestore

        self.fs = firestore
        self.db = firestore.Client(project=project or None, database=database)
        self.jobs = self.db.collection(collection)
        self.quotas = self.db.collection(collection + "_quotas")

    def _where(self, field: str, op: str, value: Any):
        from google.cloud.firestore_v1.base_query import FieldFilter

        return self.jobs.where(filter=FieldFilter(field, op, value))

    def create(self, job: Dict[str, Any]) -> None:
        self.jobs.document(job["job_id"]).create(job)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        snap = self.jobs.document(job_id).get()
        return snap.to_dict() if snap.exists else None

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        self.jobs.document(job_id).update(fields)  # Firestore understands dotted paths natively

    def add_done_shard(self, job_id: str, step: str, shard: int) -> None:
        self.update(job_id, {f"steps.{step}.done": self.fs.ArrayUnion([shard])})

    def list_for_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        jobs = [d.to_dict() for d in self._where("user_id", "==", user_id).limit(200).stream()]
        jobs.sort(key=lambda j: j.get("created_at") or now(), reverse=True)
        return jobs[:limit]

    def consume_daily_quota(self, user_id: str, limit: int) -> bool:
        ref = self.quotas.document(f"{user_id}_{now():%Y%m%d}")

        @self.fs.transactional
        def _txn(txn) -> bool:
            snap = ref.get(transaction=txn)
            count = (snap.to_dict() or {}).get("count", 0) if snap.exists else 0
            if count >= limit:
                return False
            txn.set(ref, {"user_id": user_id, "count": count + 1, "updated_at": now()})
            return True

        return _txn(self.db.transaction())

    def claim_next(self, max_concurrent: int, stale_minutes: int) -> Optional[Dict[str, Any]]:
        @self.fs.transactional
        def _txn(txn) -> Optional[Dict[str, Any]]:
            active = list(self._where("status", "in", list(ACTIVE)).get(transaction=txn))
            queued = list(self._where("status", "==", "queued").get(transaction=txn))
            cutoff = now() - timedelta(minutes=stale_minutes)
            live = 0
            for snap in active:
                started = snap.to_dict().get("started_at")
                if started and started < cutoff:  # pipeline died without reporting back
                    txn.update(snap.reference, {"status": "failed", "error": "timed out", "finished_at": now()})
                else:
                    live += 1
            if live >= max_concurrent or not queued:
                return None
            nxt = min(queued, key=lambda s: s.to_dict().get("queued_at") or now())
            txn.update(nxt.reference, {"status": "running", "started_at": now()})
            return {**nxt.to_dict(), "status": "running"}

        return _txn(self.db.transaction())

    def queued_jobs(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._where("status", "==", "queued").stream()]

    def fail_unfinished(self, reason: str) -> None:
        pass  # in the cloud, running jobs live on in Workflows; nothing to clean up


@lru_cache(maxsize=1)
def get_jobs():
    s = get_settings()
    if s.is_local:
        return LocalJobStore(s.data_dir)
    return FirestoreJobStore(s.project, s.firestore_database, s.jobs_collection)


# =========================================================================== shared helpers
def step_event(job_id: str, step: str, status: str, shard: Optional[int] = None,
               error: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Record progress of one step. A single visual shard finishing only adds to ``done``."""
    store = get_jobs()
    if shard is not None and status == "done":
        store.add_done_shard(job_id, step, shard)
        fields: Dict[str, Any] = {}
    else:
        key = f"steps.{step}"
        fields = {f"{key}.status": status}
        if status == "running":
            fields[f"{key}.started_at"] = now()
        if status in ("done", "failed"):
            fields[f"{key}.finished_at"] = now()
        if error:
            fields[f"{key}.error"] = error[-2000:]
    fields.update(extra or {})
    if fields:
        store.update(job_id, fields)


def queue_position(job: Dict[str, Any]) -> Optional[int]:
    if job.get("status") != "queued":
        return None
    mine = job.get("queued_at") or now()
    return 1 + sum(1 for q in get_jobs().queued_jobs() if (q.get("queued_at") or now()) < mine)
