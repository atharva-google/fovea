# Fovea

Turn horizontal videos into **vertical (9:16)**, **square (1:1)**, or **any aspect ratio**, with smart framing that follows whoever is speaking, on-screen action, and on-screen text.

> The name comes from the *fovea centralis*, the tiny pit in the retina responsible for sharp central vision. Your eye keeps the main subject sharp and ignores the periphery, and Fovea does the same for video: it keeps the subject in frame and crops away the unused wide-screen space.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fovea/blob/main/notebooks/colab_quickstart.ipynb)

---

## Pick how you want to use it

### 1. Try it on Colab (no setup)
Open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) in Colab, choose a GPU runtime, and run the cells.

### 2. Run it on your computer
```bash
git clone https://github.com/YOUR_USERNAME/fovea.git && cd fovea
pip install -r requirements.txt            # needs Python 3.9+ and ffmpeg (brew/apt install ffmpeg)
python -m fovea download                   # model weights, one time
python -m fovea run my_video.mp4 -t 9:16 1:1
# → outputs/my_video_9x16.mp4, outputs/my_video_1x1.mp4
```
To run it **as a web service** on your machine (backend API + a frontend later):
```bash
pip install -r backend/requirements.txt
python -m backend serve              # → http://localhost:8000/docs
```

### 3. Deploy it to Google Cloud
```bash
./backend/gcp/deploy.sh              # guided: checks your setup, asks a few questions, deploys
```
See [`backend/README.md`](backend/README.md) for what you need first (gcloud, a project, GPU quota).

---

## Project layout

```
fovea/                     THE VIDEO ENGINE (Python package)
├── config.py              every setting, in one place
├── pipeline.py            runs the 4 steps in order on this machine
├── steps/                 the 4 steps, one file each:
│   ├── prep.py              proxy video, audio, shot detection
│   ├── visual.py            tracking -> speakers -> OCR -> motion, each stage parallel over shots
│   ├── audio.py             Demucs vocals + Whisper transcript (optional, parallel)
│   └── render.py            camera path + final videos
├── models/                the AI models the steps use (tracker, ocr, motion, speakers, talknet/)
├── framing/               camera logic: aspect ratios, focal path, smooth pans/cuts
└── utils/                 video/ffmpeg helpers, job folders, model weights, device

backend/             THE BACKEND: API + job queue + worker (runs locally OR on Google Cloud)
└── gcp/                   only for Google Cloud: deploy.sh, Dockerfiles, workflow

frontend/            THE WEB APP (not built yet: see its README for how it talks to the backend)

notebooks/                 Colab quickstart + the original research notebook
scripts/                   upload_weights_to_hf.py (publish your model weights)
weights/                   model files (downloaded, not in git)
```

### How the pieces fit
```
                 ┌──────────────── backend ────────────────┐
frontend ──┤ api.py → runner.py → worker.py → fovea/steps  │
 (UI)   JSON     │      storage.py (files)   jobs.py (status)    │
                 └───────────────────────────────────────────────┘
                     MODE=local: folder + thread on your machine
                     MODE=gcp:   bucket + Firestore + Cloud Run GPU jobs
```
The **same `fovea/steps`** run everywhere: on Colab, from the CLI, in the local backend, and in Cloud Run GPU tasks.

### Where to change things
| I want to… | Edit |
|---|---|
| change a default (speed, thresholds, models) | `fovea/config.py` |
| change how the camera moves | `fovea/framing/camera.py` |
| change what the camera prefers to look at | `extract_frame_candidates` in `fovea/framing/focal.py` |
| add a new per-frame analysis | a new file in `fovea/models/`, run as a new stage in `fovea/steps/visual.py` |
| add or change an API endpoint | `backend/api.py` |
| change cloud resources or limits | `backend/gcp/config.env` (created by deploy.sh) |

---

## Output formats
```bash
python -m fovea run in.mp4 -t 9:16                 # vertical (default)
python -m fovea run in.mp4 -t 1:1 4:5              # several formats; the analysis runs once
python -m fovea run in.mp4 -t 720x1280             # exact pixel size
python -m fovea run in.mp4 -t square --short-side 720
```
| You pass | Meaning |
|---|---|
| `W:H` (`9:16`, `4:5`, `2.35:1`) | aspect ratio; output short side = `--short-side` (default 1080) |
| `WxH` (`1080x1920`) | exact output size |
| preset | `vertical`/`shorts`/`reels`/`tiktok` = 9:16, `square` = 1:1, `portrait` = 4:5, `classic` = 4:3, `landscape` = 16:9, `cinema` = 21:9 |

From Python:
```python
from fovea import run
run("in.mp4", targets=["9:16", "1:1"], transcribe=True)
```

## Speed options
| Option | Default | Effect |
|---|---|---|
| `--proxy-height` | 720 | resolution the analysis runs at (output always uses the original) |
| `--detect-every` | 2 | run the detector every N frames and interpolate the rest |
| `--face-backend` | `mediapipe` | `pose` = YOLO-pose head keypoints on the GPU (faster; check accuracy first) |
| `--max-shards` | 1 | split the visual step into N groups of shots |
| `--visual-workers` | auto | shots decoded / processed at once in the tracking and speaker stages |
| `--ocr-workers` | auto | OCR threads (one shared EasyOCR model) |
| `--motion-workers` | auto | motion saliency threads (CPU) |
| `--no-gpu-video` | off | disable NVDEC/NVENC |

Each video gets a job folder in `working/<video>/`. Steps whose outputs already exist are skipped, so re-rendering a new format only re-runs `render`. Use `--force` to start over.

## Model weights
Weights are not stored in git. They're fetched into `weights/` with `python -m fovea download`, from your Hugging Face repo if `FOVEA_HF_REPO` is set (publish one with `scripts/upload_weights_to_hf.py`), otherwise from the official sources. See [`weights/README.md`](weights/README.md).

| Model | License |
|---|---|
| YOLO26 (Ultralytics) | AGPL-3.0 |
| BlazeFace (MediaPipe) | Apache-2.0 |
| TalkNet | MIT |
| EasyOCR | Apache-2.0 |
| faster-whisper, Demucs (optional) | MIT |

## License
Choose a license for your code. Ultralytics YOLO is AGPL-3.0. TalkNet model code is adapted from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) (MIT).
