# frontend/

The Fovea Studio web app: a zero-build HTML/CSS/ES6 Single-Page App served by a lightweight Python FastAPI server (`python -m frontend`). Runs anywhere Python runs—locally, on Google Cloud Run (CPU-only), or inside a Colab notebook—with no Node.js or GPU required.

## Features
- **3-Pane Gemini / ChatGPT-Style Studio Layout:**
  - **Left Sidebar (Collapsible):** Video library grouped into **In Progress** (`queued`, `running`, `rendered`), **Uploaded · Not Started** (`awaiting_upload`), and **Processed** (`complete`, `failed`).
  - **Center Main Workspace:** Drag-and-drop video uploader, target aspect-ratio selector (`9:16`, `1:1`, `4:5`, `16:9`, `21:9`, custom `WxH`), pipeline options, and a live 4-step execution timeline (`prep`, `visual` with shard progress, `audio` with interactive click-to-seek Whisper transcript, and `render`).
  - **Right Video Player Drawer (Collapsible):** Preview the uploaded source video immediately or switch between rendered aspect ratios (`9x16`, `1x1`, etc.) with synchronized playback timestamps, frame stepping (`-1f` / `+1f`), speed controls, and direct MP4 downloads.

## Running Locally
Start the backend in one terminal and the frontend in another:
```bash
# Terminal 1: Backend API (http://localhost:8000)
pip install -r backend/requirements.txt
python -m backend serve

# Terminal 2: Frontend UI (http://localhost:3000)
python -m frontend
```
Then open `http://localhost:3000` in your browser.

## How it talks to the backend
The frontend only needs one setting, `BACKEND_URL` (default `http://localhost:8000`). The same code works whether the backend runs on your laptop, in Colab, or on Google Cloud:

| Backend runs… | `BACKEND_URL` | Auth |
|---|---|---|
| on this machine (`python -m backend serve`) | `http://localhost:8000` | none |
| on Google Cloud, `AUTH_MODE=iam` | `https://fovea-api-xxxx.run.app` | the frontend **server** sends a Google identity token (it runs as the service account named `FRONTEND_SA`) |
| anywhere, `AUTH_MODE=api_key` | that backend's URL | set `BACKEND_API_KEY=<key>`; the frontend **server** sends `X-API-Key` (never exposed to the browser) |

The flow (full API reference in [`backend/README.md`](../backend/README.md)):
```
browser → frontend server (/api/v1/uploads) → POST /v1/uploads  ← {job_id, upload: {url, method, headers}}
browser → PUT the file directly to upload.url (with upload.headers)
browser → frontend server (/api/v1/jobs/{job_id}/start)         → {targets: ["9:16", "1:1"], options: {...}}
browser → frontend server (/api/v1/jobs/{job_id})               → poll every 3s for step & shard status
browser → play/download outputs["9x16"] in the right-hand video player
```

## Deploying to Cloud Run (CPU-only)
```bash
gcloud run deploy fovea-frontend \
  --source . \
  --dockerfile frontend/Dockerfile \
  --region us-central1 \
  --service-account "$FRONTEND_SA" \
  --set-env-vars "BACKEND_URL=$API_URL" \
  --allow-unauthenticated
```
