"""Fovea backend API. Same endpoints in local and cloud mode.

Only small JSON goes through these endpoints. Video files go browser <-> storage
through the upload/download links the API hands out.

    POST   /v1/uploads              -> {job_id, upload: {url, method, headers}}
    POST   /v1/jobs/{job_id}/start  -> job (queued / running)
    GET    /v1/jobs/{job_id}        -> status, progress, download links
    GET    /v1/jobs?user_id=...     -> a user's jobs
    DELETE /v1/jobs/{job_id}        -> delete a finished job and its files
    GET    /v1/config               -> limits, presets, allowed options (for the UI)

    PUT/GET /v1/files/{key}                 local mode only: the upload/download links
    POST    /internal/jobs/{job_id}/events  cloud mode only: called by Cloud Workflows

Run it:  python -m backend serve        (interactive docs at http://localhost:8000/docs)
"""

from __future__ import annotations

import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fovea.config import PipelineConfig
from fovea.framing.geometry import PRESETS, OutputTarget

from .jobs import ACTIVE, get_jobs, now, queue_position
from .runner import dispatch_queued, handle_event
from .settings import get_settings
from .storage import get_storage, job_prefix, safe_filename

logging.basicConfig(level=logging.INFO)
settings = get_settings()

# Options a client may set per job; everything else uses server defaults
ALLOWED_OPTIONS = {
    "short_side", "focal_strategy", "face_backend", "transcribe", "enable_ocr", "enable_motion",
    "enable_speaker_detection", "pan_window_frames", "cut_threshold_px", "still_deadband_px",
    "detect_every", "debug_renders",
}
MAX_TARGETS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.is_local:
        get_jobs().fail_unfinished("backend restarted while the job was running")
        dispatch_queued()
    yield


app = FastAPI(title="Fovea backend", version="1.0.0", lifespan=lifespan)
if settings.allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                       allow_methods=["*"], allow_headers=["*"])


# =========================================================================== auth
def require_client(x_api_key: Optional[str] = Header(default=None)) -> None:
    """none: anyone (local) | iam: Cloud Run already verified the caller | api_key: check header."""
    if settings.auth_mode == "api_key":
        if not settings.api_key or not hmac.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def require_workflow(authorization: Optional[str] = Header(default=None)) -> None:
    """Only the pipeline workflow's service account may report job events."""
    from google.auth.transport.requests import Request as GRequest
    from google.oauth2 import id_token

    if settings.is_local:
        raise HTTPException(status_code=404)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        claims = id_token.verify_oauth2_token(authorization.split(" ", 1)[1], GRequest(), audience=settings.self_url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e
    if claims.get("email") != settings.workflow_sa_email:
        raise HTTPException(status_code=403, detail="Caller is not the pipeline workflow")


# =========================================================================== schemas
class UploadRequest(BaseModel):
    user_id: str = Field(..., examples=["user-123"])
    filename: str = Field(..., examples=["podcast.mp4"])
    size_bytes: int = Field(..., gt=0)


class StartRequest(BaseModel):
    user_id: str
    targets: List[str] = Field(default_factory=lambda: ["9:16"], examples=[["9:16", "1:1"]])
    options: Dict[str, Any] = Field(default_factory=dict, examples=[{"transcribe": True}])


class EventRequest(BaseModel):
    event: Literal["rendered", "complete", "failed"]
    error: Optional[Any] = None


# =========================================================================== helpers
def _owned_job(job_id: str, user_id: str) -> Dict[str, Any]:
    job = get_jobs().get(job_id)
    if job is None or job.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _progress(job: Dict[str, Any]) -> float:
    if job.get("status") in ("rendered", "complete"):
        return 1.0
    st = job.get("steps", {})
    p = 0.15 if st.get("prep", {}).get("status") == "done" else 0.0
    vis = st.get("visual", {})
    if vis.get("status") == "done" or st.get("render", {}).get("status") in ("running", "done"):
        p += 0.6
    else:
        p += 0.6 * len(vis.get("done", [])) / max(1, int(vis.get("total") or job.get("num_shards") or 1))
    return round(min(p, 0.99), 2)


def _view(job: Dict[str, Any], with_links: bool = True) -> Dict[str, Any]:
    out = {k: job.get(k) for k in ("job_id", "user_id", "status", "filename", "error", "created_at",
                                   "queued_at", "started_at", "rendered_at", "finished_at", "steps")}
    out["targets"] = (job.get("config") or {}).get("targets")
    out["progress"] = _progress(job)
    out["queue_position"] = queue_position(job)
    out["outputs"], out["transcript_url"] = {}, None
    if with_links:
        store = get_storage()
        out["outputs"] = {label: store.download_url(f"{job['prefix']}/{rel}")
                          for label, rel in (job.get("outputs") or {}).items()}
        if (job.get("steps") or {}).get("audio", {}).get("status") == "done":
            out["transcript_url"] = store.download_url(f"{job['prefix']}/audio/transcript.json")
    return out


# =========================================================================== public API
@app.get("/healthz")
def healthz():
    return {"ok": True, "mode": settings.mode}


@app.get("/v1/config", dependencies=[Depends(require_client)])
def get_config():
    s = settings
    return {
        "mode": s.mode,
        "demo_mode": s.demo_mode,
        "max_upload_mb": s.max_upload_mb if s.demo_mode else None,
        "max_duration_s": s.max_duration_s if s.demo_mode else None,
        "daily_jobs_per_user": s.daily_jobs_per_user if s.demo_mode else None,
        "max_targets": MAX_TARGETS,
        "presets": PRESETS,
        "options": sorted(ALLOWED_OPTIONS),
    }


@app.post("/v1/uploads", dependencies=[Depends(require_client)])
def create_upload(req: UploadRequest):
    s = settings
    if s.max_upload_bytes and req.size_bytes > s.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than {s.max_upload_mb} MB")

    job_id = f"{now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    try:
        prefix = job_prefix(req.user_id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    input_key = f"{prefix}/input/{safe_filename(req.filename)}"

    get_jobs().create({
        "job_id": job_id, "user_id": req.user_id, "prefix": prefix, "filename": safe_filename(req.filename),
        "input_key": input_key, "status": "awaiting_upload", "created_at": now(), "steps": {}, "outputs": {},
    })
    return {"job_id": job_id, "upload": get_storage().upload_url(input_key, s.max_upload_bytes)}


@app.post("/v1/jobs/{job_id}/start", dependencies=[Depends(require_client)])
def start_job(job_id: str, req: StartRequest):
    s, store, jobs = settings, get_storage(), get_jobs()
    job = _owned_job(job_id, req.user_id)
    if job["status"] != "awaiting_upload":
        raise HTTPException(status_code=409, detail=f"Job is already {job['status']}")

    size = store.size(job["input_key"])
    if size is None:
        raise HTTPException(status_code=400, detail="Upload not found: send the file to the upload URL first")
    if s.max_upload_bytes and size > s.max_upload_bytes:
        store.delete(job["input_key"])
        raise HTTPException(status_code=413, detail=f"File is larger than {s.max_upload_mb} MB")

    if not req.targets or len(req.targets) > MAX_TARGETS:
        raise HTTPException(status_code=400, detail=f"Give 1-{MAX_TARGETS} targets")
    unknown = set(req.options) - ALLOWED_OPTIONS
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported options: {sorted(unknown)}")
    config = {"targets": req.targets, **req.options, "max_shards": s.max_shards}
    if s.demo_mode:
        config["max_duration_s"] = s.max_duration_s
    try:
        for t in req.targets:
            OutputTarget.parse(t)
        PipelineConfig().update(config)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if s.demo_mode and not jobs.consume_daily_quota(req.user_id, s.daily_jobs_per_user):
        raise HTTPException(status_code=429, detail=f"Daily limit of {s.daily_jobs_per_user} videos reached")

    store.write_json(f"{job['prefix']}/job.json", {"job_id": job_id, "user_id": req.user_id, "config": config})
    jobs.update(job_id, {"status": "queued", "queued_at": now(), "config": config, "size_bytes": size})
    dispatch_queued()
    return _view(jobs.get(job_id), with_links=False)


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_client)])
def get_job(job_id: str, user_id: str = Query(...)):
    return _view(_owned_job(job_id, user_id))


@app.get("/v1/jobs", dependencies=[Depends(require_client)])
def list_jobs(user_id: str = Query(...), limit: int = Query(20, le=100)):
    return [_view(j, with_links=False) for j in get_jobs().list_for_user(user_id, limit)]


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_client)])
def delete_job(job_id: str, user_id: str = Query(...)):
    job = _owned_job(job_id, user_id)
    if job["status"] in ACTIVE or job["status"] == "queued":
        raise HTTPException(status_code=409, detail="Job is still processing")
    get_storage().delete_prefix(job["prefix"])
    get_jobs().update(job_id, {"status": "deleted", "outputs": {}})
    return {"deleted": job_id}


# =========================================================================== local file links
@app.put("/v1/files/{key:path}")
async def put_file(key: str, request: Request, expires: int, sig: str, max_bytes: Optional[int] = None):
    if not settings.is_local:
        raise HTTPException(status_code=404)
    store = get_storage()
    if not store.verify("PUT", key, expires, sig, max_bytes):
        raise HTTPException(status_code=403, detail="Invalid or expired upload link")
    dest = store.path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp, size = dest.with_suffix(dest.suffix + ".part"), 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if max_bytes and size > max_bytes:
                f.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)
    tmp.replace(dest)
    return {"ok": True, "size_bytes": size}


@app.get("/v1/files/{key:path}")
def get_file(key: str, expires: int, sig: str):
    if not settings.is_local:
        raise HTTPException(status_code=404)
    store = get_storage()
    if not store.verify("GET", key, expires, sig, None):
        raise HTTPException(status_code=403, detail="Invalid or expired link")
    path = store.path(key)
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path)


# =========================================================================== cloud callbacks
@app.post("/internal/jobs/{job_id}/events", dependencies=[Depends(require_workflow)])
def job_event(job_id: str, req: EventRequest):
    if get_jobs().get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    handle_event(job_id, req.event, req.error if isinstance(req.error, str) else (str(req.error) if req.error else None))
    return {"ok": True}
