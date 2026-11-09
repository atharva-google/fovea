"""Model weight registry + downloader.

Weights are NOT stored in git. On first use each model is fetched into
``weights/`` (or ``$FOVEA_WEIGHTS_DIR``).

Two sources are supported, tried in this order:

1. **Your Hugging Face mirror** (optional, recommended): set
   ``FOVEA_HF_REPO=<user>/<repo>`` (or pass ``hf_repo=``) after uploading your
   ``weights/`` folder with ``scripts/upload_weights_to_hf.py``. One reliable,
   fast source for everything.
2. **Official upstream sources** (fallback): Ultralytics GitHub releases,
   Google's MediaPipe bucket, the TalkNet Google Drive file, Systran on HF.

Run ``python -m fovea download`` to fetch everything up-front.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)
TALKNET_GDRIVE_ID = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"  # from TalkNet-ASD demoTalkNet.py

# After running scripts/upload_weights_to_hf.py, put your repo id here
# (e.g. "yourname/fovea-weights") so every user downloads from your mirror.
DEFAULT_HF_REPO: Optional[str] = None


@dataclass
class WeightSpec:
    key: str
    path: str          # relative to weights_dir (file or folder)
    description: str
    used_by: str       # which pipeline feature needs it
    license: str


def _specs(yolo_weights: str = "yolo26s.pt", whisper_model: str = "medium",
           pose_weights: str = "yolo26s-pose.pt") -> Dict[str, WeightSpec]:
    return {
        "yolo": WeightSpec("yolo", yolo_weights, "Ultralytics YOLO person detector", "tracking", "AGPL-3.0"),
        "yolo_pose": WeightSpec("yolo_pose", pose_weights, "Ultralytics YOLO-pose (face_backend=pose)", "tracking", "AGPL-3.0"),
        "blazeface": WeightSpec("blazeface", "blaze_face_full_range.tflite", "MediaPipe face detector", "tracking", "Apache-2.0"),
        "talknet": WeightSpec("talknet", "pretrain_TalkSet.model", "TalkNet active-speaker model", "speaker", "MIT"),
        "easyocr": WeightSpec("easyocr", "easyocr", "EasyOCR detector + recognizer", "ocr", "Apache-2.0"),
        "whisper": WeightSpec("whisper", f"whisper-{whisper_model}", "faster-whisper (CTranslate2)", "transcribe", "MIT"),
        "demucs": WeightSpec("demucs", "", "Demucs htdemucs (stored in torch hub cache)", "transcribe", "MIT"),
    }


# --------------------------------------------------------------------------- helpers
def _log(msg: str) -> None:
    print(f"[weights] {msg}", flush=True)


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _log(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def _try_hf_mirror(spec: WeightSpec, weights_dir: Path, hf_repo: Optional[str]) -> bool:
    """Try to fetch ``spec.path`` from the user's HF mirror. Returns True on success."""
    if not hf_repo or not spec.path:
        return False
    try:
        from huggingface_hub import snapshot_download

        _log(f"trying HF mirror {hf_repo} for {spec.path}")
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[spec.path, f"{spec.path}/*"],
            local_dir=str(weights_dir),
        )
        return (weights_dir / spec.path).exists()
    except Exception as e:  # noqa: BLE001
        _log(f"HF mirror failed ({e}); falling back to upstream source")
        return False


# --------------------------------------------------------------------------- upstream downloaders
def _dl_yolo(spec: WeightSpec, weights_dir: Path, **_) -> None:
    from ultralytics import YOLO

    dest = weights_dir / spec.path
    # Ultralytics auto-downloads official weights when the file is missing.
    YOLO(str(dest))
    if not dest.exists():  # some versions download to the CWD instead
        stray = Path.cwd() / Path(spec.path).name
        if stray.exists():
            shutil.move(str(stray), dest)


def _dl_blazeface(spec: WeightSpec, weights_dir: Path, **_) -> None:
    _http_download(BLAZEFACE_URL, weights_dir / spec.path)


def _dl_talknet(spec: WeightSpec, weights_dir: Path, **_) -> None:
    import gdown

    dest = weights_dir / spec.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=TALKNET_GDRIVE_ID, output=str(dest), quiet=False)


def _dl_easyocr(spec: WeightSpec, weights_dir: Path, ocr_langs: List[str] = ("en",), **_) -> None:
    import easyocr

    (weights_dir / spec.path).mkdir(parents=True, exist_ok=True)
    easyocr.Reader(list(ocr_langs), gpu=False, model_storage_directory=str(weights_dir / spec.path),
                   download_enabled=True, verbose=False)


def _dl_whisper(spec: WeightSpec, weights_dir: Path, whisper_model: str = "medium", **_) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=f"Systran/faster-whisper-{whisper_model}", local_dir=str(weights_dir / spec.path))


def _dl_demucs(spec: WeightSpec, weights_dir: Path, **_) -> None:
    from demucs.pretrained import get_model

    get_model("htdemucs")


_DOWNLOADERS: Dict[str, Callable] = {
    "yolo": _dl_yolo,
    "yolo_pose": _dl_yolo,
    "blazeface": _dl_blazeface,
    "talknet": _dl_talknet,
    "easyocr": _dl_easyocr,
    "whisper": _dl_whisper,
    "demucs": _dl_demucs,
}


def _is_present(spec: WeightSpec, weights_dir: Path) -> bool:
    if spec.key == "demucs":
        return False  # managed by torch hub; get_model() is a no-op when cached
    p = weights_dir / spec.path
    if p.is_dir():
        return any(p.iterdir())
    return p.exists() and p.stat().st_size > 0


# --------------------------------------------------------------------------- public API
def ensure_weight(
    key: str,
    weights_dir: Path,
    *,
    yolo_weights: str = "yolo26s.pt",
    pose_weights: str = "yolo26s-pose.pt",
    whisper_model: str = "medium",
    ocr_langs: List[str] = ("en",),
    hf_repo: Optional[str] = None,
) -> Path:
    """Make sure the weight ``key`` exists locally and return its path."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = hf_repo or os.environ.get("FOVEA_HF_REPO") or DEFAULT_HF_REPO
    spec = _specs(yolo_weights, whisper_model, pose_weights)[key]
    path = weights_dir / spec.path

    if _is_present(spec, weights_dir):
        return path
    if not _try_hf_mirror(spec, weights_dir, hf_repo):
        _DOWNLOADERS[key](spec, weights_dir, whisper_model=whisper_model, ocr_langs=ocr_langs)
    if spec.key != "demucs" and not _is_present(spec, weights_dir):
        raise RuntimeError(f"Failed to obtain weight '{key}' at {path}")
    return path


def download_all(weights_dir: Path, keys: Optional[List[str]] = None, **kwargs) -> None:
    keys = keys or ["yolo", "blazeface", "talknet", "easyocr"]
    for key in keys:
        _log(f"ensuring {key} ...")
        path = ensure_weight(key, weights_dir, **kwargs)
        _log(f"  ok -> {path}")


def list_weights(weights_dir: Path, yolo_weights: str = "yolo26s.pt", whisper_model: str = "medium",
                 pose_weights: str = "yolo26s-pose.pt") -> List[dict]:
    rows = []
    for spec in _specs(yolo_weights, whisper_model, pose_weights).values():
        rows.append({
            "key": spec.key, "path": str(Path(weights_dir) / spec.path) if spec.path else "(torch hub cache)",
            "present": _is_present(spec, Path(weights_dir)), "used_by": spec.used_by, "license": spec.license,
        })
    return rows


def ensure_for(cfg, key: str) -> Path:
    """``ensure_weight`` using the model names/paths from a PipelineConfig."""
    return ensure_weight(key, cfg.weights_dir, yolo_weights=cfg.yolo_weights, pose_weights=cfg.pose_weights,
                         whisper_model=cfg.whisper_model, ocr_langs=cfg.ocr_langs)
