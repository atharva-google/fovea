"""All backend settings, read from environment variables.

The one setting that matters most is ``MODE``:

    MODE=local  (default)  files and job records in DATA_DIR (default: <fovea data folder>/backend,
                           outside the code folder; see fovea.config.fovea_home), jobs run in
                           a background thread on THIS machine's GPU/CPU. No cloud needed.
    MODE=gcp               files in a Cloud Storage bucket, job records in Firestore,
                           jobs run as Cloud Run GPU jobs started by Cloud Workflows.

Everything else has a sensible default for each mode. Put overrides in an env file
and start with ``python -m backend serve --env path/to/file.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fovea.config import fovea_home


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


@dataclass(frozen=True)
class Settings:
    mode: str                      # "local" | "gcp"

    # ---- who may call the API
    auth_mode: str                 # "none" (local) | "iam" (private Cloud Run) | "api_key"
    api_key: Optional[str]
    allowed_origins: List[str]     # browser origins allowed to upload/download files (CORS)

    # ---- limits (only enforced when demo_mode is on)
    demo_mode: bool
    max_upload_mb: int
    max_duration_s: int
    daily_jobs_per_user: int
    max_concurrent_jobs: int       # always enforced: how many jobs run at once
    stale_job_minutes: int

    # ---- pipeline
    max_shards: int
    split_steps: bool              # gcp only: each step as its own GPU task
    signed_url_minutes: int

    # ---- local mode
    data_dir: Path
    public_url: str                # how browsers reach this API (for upload/download links)

    # ---- gcp mode
    project: str
    region: str
    bucket: str
    firestore_database: str
    jobs_collection: str
    workflow_name: str
    worker_job_name: str
    self_url: str
    workflow_sa_email: str
    scratch_dir: Path              # where a cloud worker keeps a job's files while running

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    @property
    def max_upload_bytes(self) -> Optional[int]:
        return self.max_upload_mb * 1024 * 1024 if self.demo_mode else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    mode = _env("MODE", "local").lower()
    if mode not in ("local", "gcp"):
        raise ValueError("MODE must be 'local' or 'gcp'")
    local = mode == "local"
    port = _int("PORT", 8000)
    origins = _env("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000" if local else "")

    return Settings(
        mode=mode,
        auth_mode=_env("AUTH_MODE", "none" if local else "iam"),
        api_key=_env("API_KEY") or None,
        allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        demo_mode=_bool("DEMO_MODE", not local),
        max_upload_mb=_int("MAX_UPLOAD_MB", 500),
        max_duration_s=_int("MAX_DURATION_S", 300),
        daily_jobs_per_user=_int("DAILY_JOBS_PER_USER", 3),
        max_concurrent_jobs=_int("MAX_CONCURRENT_JOBS", 1 if local else 2),
        stale_job_minutes=_int("STALE_JOB_MINUTES", 24 * 60 if local else 90),
        max_shards=_int("MAX_SHARDS", 1),
        split_steps=_bool("SPLIT_STEPS", True),
        signed_url_minutes=_int("SIGNED_URL_MINUTES", 60),
        data_dir=Path(_env("DATA_DIR") or fovea_home() / "backend"),
        public_url=_env("PUBLIC_URL", f"http://localhost:{port}").rstrip("/"),
        project=_env("GOOGLE_CLOUD_PROJECT"),
        region=_env("REGION", "us-central1"),
        bucket=_env("BUCKET"),
        firestore_database=_env("FIRESTORE_DATABASE", "(default)"),
        jobs_collection=_env("JOBS_COLLECTION", "fovea_jobs"),
        workflow_name=_env("WORKFLOW_NAME", "fovea-pipeline"),
        worker_job_name=_env("WORKER_JOB_NAME", "fovea-worker"),
        self_url=_env("SELF_URL").rstrip("/"),
        workflow_sa_email=_env("WORKFLOW_SA_EMAIL"),
        scratch_dir=Path(_env("SCRATCH_DIR", "/tmp/fovea")),
    )
