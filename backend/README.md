# backend/

The Fovea backend: an HTTP API, a job queue, and a worker that runs the four `fovea/steps`. It's all Python, and it runs **on your own machine or on Google Cloud**, depending on one setting.

| | `MODE=local` (default) | `MODE=gcp` |
|---|---|---|
| Video files | `./data/users/<user_id>/<job_id>/` | `gs://<bucket>/users/<user_id>/<job_id>/` |
| Job records | `./data/_jobs/<job_id>.json` | Firestore |
| Who runs a job | a background thread in the API process, on this machine's GPU (or CPU) | Cloud Workflows starts Cloud Run GPU job tasks |
| Upload/download links | the API's own `/v1/files/...` (signed) | signed Cloud Storage URLs |
| Start it | `python -m backend serve` | `./backend/gcp/deploy.sh` |

The API, the job records and the links look **identical** to the frontend in both modes.

## Run it on your machine
```bash
pip install -r requirements.txt -r backend/requirements.txt   # from the repo root
python -m backend serve
# → http://localhost:8000/docs   (try every endpoint in the browser)
```
To change settings: `cp backend/local.env.example backend/local.env`, edit it, then run `python -m backend serve --env backend/local.env`.

## Run it on Colab (free GPU, public URL)
Open [`notebooks/colab_backend.ipynb`](../notebooks/colab_backend.ipynb) in Colab with a GPU runtime and run all cells. It installs everything, downloads the weights, starts the backend (`MODE=local`, `AUTH_MODE=api_key`) and exposes it through a Cloudflare quick tunnel. It then prints the public URL and API key. The URL changes on every start, and uploads are limited to ~100 MB by the tunnel.

## Deploy it to Google Cloud
```bash
./backend/gcp/deploy.sh
```
The script checks gcloud, your login and billing, asks a few questions (project, region, GPU quota, who may call the API), saves the answers to `backend/gcp/config.env`, deploys everything, and prints the URL plus a test command.

Before running it:
1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud auth login`.
2. Create a Google Cloud project with billing turned on.
3. [Request Cloud Run L4 GPU quota](https://console.cloud.google.com/iam-admin/quotas) for jobs in your region. With a quota of 1, each job runs in a single GPU task; with 2 or more, the steps run as separate parallel tasks.

## Files
```
backend/
├── api.py            the HTTP endpoints (start here)
├── worker.py         runs fovea steps for a job: copy inputs in → fovea.steps.*.run() → copy outputs out
├── runner.py         starts queued jobs: local thread  OR  Cloud Workflows
├── storage.py        job files: local folder  OR  Cloud Storage bucket
├── jobs.py           job records: JSON files  OR  Firestore
├── settings.py       every setting (env vars) and the local/gcp defaults
├── __main__.py       `python -m backend serve | worker`
├── local.env.example
├── requirements.txt
└── gcp/              only needed for Google Cloud
    ├── deploy.sh         guided one-command deploy
    ├── pipeline.yaml     the step order on Cloud Workflows
    ├── Dockerfile.api    CPU image for the API
    ├── Dockerfile.worker GPU image for the steps (model weights baked in from HF_REPO)
    └── cloudbuild.yaml
```

## How a job flows
```
POST /v1/uploads ──> job "awaiting_upload" + upload link
browser uploads the file
POST /v1/jobs/{id}/start ──> checks limits, writes job.json, status "queued"
runner starts it when a slot is free (MAX_CONCURRENT_JOBS) ──> "running"
   prep ──┬── visual (shards) ── render ──> "rendered"   (videos ready)
          └── audio (if transcribe) ──────> "complete"   (transcript ready)
GET /v1/jobs/{id} ──> progress, then download links
```

## API reference
All `/v1` endpoints take and return JSON. Interactive docs are at `/docs`.

**`POST /v1/uploads`**
```json
{"user_id": "user-123", "filename": "podcast.mp4", "size_bytes": 104857600}
→ {"job_id": "20261109103000-1a2b3c4d",
   "upload": {"url": "...", "method": "PUT", "headers": {"x-goog-content-length-range": "0,524288000"}}}
```
The browser sends the file to `upload.url`, using `upload.method` and `upload.headers` exactly as given.

**`POST /v1/jobs/{job_id}/start`**
```json
{"user_id": "user-123", "targets": ["9:16", "1:1"], "options": {"transcribe": true}}
```
- `targets`: ratios (`9:16`), sizes (`1080x1920`) or presets (`vertical`, `square`, `portrait`, …), up to 4.
- `options`: `transcribe`, `short_side`, `focal_strategy`, `face_backend`, `enable_ocr`, `enable_motion`, `enable_speaker_detection`, `pan_window_frames`, `cut_threshold_px`, `still_deadband_px`, `detect_every`, `debug_renders`.

**`GET /v1/jobs/{job_id}?user_id=user-123`**
```json
{"status": "rendered", "progress": 1.0, "queue_position": null, "targets": ["9:16", "1:1"],
 "steps": {"prep": {"status": "done"}, "visual": {"done": [0], "total": 1},
           "render": {"status": "done"}, "audio": {"status": "running"}},
 "outputs": {"9x16": "<link>", "1x1": "<link>"}, "transcript_url": null, "error": null}
```
`status` is one of: `awaiting_upload` → `queued` → `running` → `rendered` → `complete`, or `failed`.

While a step runs, `steps.<step>.stage` shows its current stage (refreshed every ~2 s). Finished stages go into `steps.<step>.stage_times` (seconds):
```json
"visual": {"status": "running",
           "stage": {"name": "ocr", "index": 3, "count": 4, "done": 40, "total": 120, "unit": "shots", "elapsed_s": 6.2},
           "stage_times": {"tracking": 31.4, "speakers": 12.0}}
```
Stages: prep = `proxy + audio`, `shot detection`; visual = `tracking`, `speakers`, `ocr`, `motion`; render = `focal points`, `encode <label>` (one per target).

**Also available:**
- `GET /v1/jobs?user_id=`: list a user's jobs
- `DELETE /v1/jobs/{id}?user_id=`: delete a finished job's files
- `GET /v1/config`: limits and presets, for building the UI

## Limits (`DEMO_MODE=true`, the default on Google Cloud)
| Limit | Setting | Default |
|---|---|---|
| Upload size | `MAX_UPLOAD_MB` | 500 |
| Video length | `MAX_DURATION_S` | 300 |
| Jobs per user per day | `DAILY_JOBS_PER_USER` | 3 |
| Jobs running at once | `MAX_CONCURRENT_JOBS` | 1 locally / set by deploy.sh (always enforced) |
