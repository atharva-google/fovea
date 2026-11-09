# frontend/

The Fovea web app: the UI, plus the code that talks to the backend. **Not built yet.**

## What it will do
1. Let a user sign in, and later enter their Gemini API key (kept on the frontend side).
2. Upload a video and pick output formats (9:16, 1:1, 4:5, …).
3. Show progress, then play and download the results.

## How it talks to the backend
The frontend only needs one setting, `BACKEND_URL`. The same code works whether the backend runs on your laptop or on Google Cloud.

| Backend runs… | `BACKEND_URL` | Auth |
|---|---|---|
| on this machine (`python -m backend serve`) | `http://localhost:8000` | none |
| on Google Cloud, `AUTH_MODE=iam` | `https://fovea-api-xxxx.run.app` | the frontend **server** sends a Google identity token (it runs as the service account named `FRONTEND_SA`) |
| anywhere, `AUTH_MODE=api_key` | that backend's URL | send `X-API-Key: <key>` from the frontend **server** (never expose it to the browser) |

The flow (full API reference in [`backend/README.md`](../backend/README.md)):
```
frontend server → POST /v1/uploads                 ← {job_id, upload: {url, method, headers}}
browser         → PUT the file to upload.url (with upload.headers)
frontend server → POST /v1/jobs/{job_id}/start     {targets: ["9:16", "1:1"]}
frontend server → GET  /v1/jobs/{job_id}           (poll every ~3 s)
browser         → play/download outputs["9x16"]    (a ready-to-use link)
```
- Video bytes never pass through the frontend server or the backend API. The browser uploads straight to the link it was given: a Cloud Storage bucket in the cloud, or the backend's `/v1/files` route locally.
- In local mode, the backend's `ALLOWED_ORIGINS` must include the frontend's address (default `http://localhost:3000`).

## Running it (once built)
- **Locally:** run it on port 3000 with `BACKEND_URL=http://localhost:8000`.
- **On Cloud Run:** deploy it as a separate service whose service account is listed as `FRONTEND_SA` in the backend's `backend/gcp/config.env`.
