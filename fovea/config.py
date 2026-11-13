"""Central configuration for the Fovea pipeline.

Every tunable knob lives in :class:`PipelineConfig`. The CLI (``cli.py``), the
Python API (``pipeline.py``) and the backend worker (``backend/worker.py``) all build
one of these, so this is the single place to look when you want to change a default.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


def _code_root() -> Optional[Path]:
    """The checkout folder (the one with pyproject.toml), or None when installed as a package."""
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").exists() else None


def fovea_home() -> Path:
    """The data folder: where every working file goes on a local machine or Colab. Never the code folder.

        $FOVEA_HOME                 if set
        <code folder>/../fovea_data when running from a checkout (e.g. /content/fovea -> /content/fovea_data)
        ~/fovea_data                when installed as a package

    One folder per video, named after the video file, holds EVERYTHING about that video::

        fovea_data/<video name>/input/ prep/ visual/ audio/ framing/ outputs/

    (layout: fovea/utils/jobdir.py). The local backend keeps its files in ``fovea_data/backend/``.
    On Google Cloud the bucket plays the role of this folder.

    Model weights are NOT working files: they stay with the code, in ``<code folder>/weights/``.
    """
    env = os.environ.get("FOVEA_HOME")
    if env:
        return Path(env).expanduser()
    root = _code_root()
    return root.parent / "fovea_data" if root else Path.home() / "fovea_data"


def default_weights_dir() -> Path:
    """Models live with the code: ``<code folder>/weights/`` (FOVEA_WEIGHTS_DIR moves them)."""
    env = os.environ.get("FOVEA_WEIGHTS_DIR")
    if env:
        return Path(env)
    root = _code_root()
    return root / "weights" if root else fovea_home() / "weights"


def default_work_dir() -> Path:
    """Parent of the per-video folders: the data folder itself (FOVEA_WORK_DIR moves it)."""
    return Path(os.environ.get("FOVEA_WORK_DIR") or fovea_home())


def default_output_dir() -> Optional[Path]:
    """None = final videos stay in the video's own folder (``<video name>/outputs/``).
    Set FOVEA_OUTPUT_DIR (or ``output_dir``) to ALSO get copies named ``<video name>_<format>.mp4`` there."""
    env = os.environ.get("FOVEA_OUTPUT_DIR")
    return Path(env) if env else None


_PATH_FIELDS = ("weights_dir", "work_dir")


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------ paths
    weights_dir: Path = field(default_factory=default_weights_dir)
    work_dir: Path = field(default_factory=default_work_dir)      # holds one folder per video (everything about it)
    output_dir: Optional[Path] = field(default_factory=default_output_dir)  # extra copies of the final videos (None = no copies)

    # ---------------------------------------------------------------- outputs
    # Ratios ("9:16", "1:1"), presets ("vertical", "square") or sizes ("1080x1920").
    # The expensive analysis runs once and is reused for every target.
    targets: List[str] = field(default_factory=lambda: ["9:16"])
    short_side: int = 1080

    # ----------------------------------------------------------------- device
    device: Optional[str] = None  # "cuda" | "mps" | "cpu" | None (auto)

    # ---------------------------------------------------------------- speed
    proxy_height: int = 720   # all analysis runs on a downscaled copy of this height
    proxy_fps: float = 25.0   # constant fps for analysis (TalkNet expects 25)
    detect_every: int = 2     # run the detector on every Nth frame, interpolate the rest
    detect_batch: int = 16    # frames per detector forward pass
    motion_scale: float = 0.5 # motion saliency runs at this fraction of proxy size
    use_gpu_video: bool = True  # NVDEC decode / NVENC encode when available

    # ------------------------------------------------- parallelism (0 = auto)
    # The visual step runs stage by stage (tracking -> speakers -> OCR -> motion); each stage
    # works on several shots at once with this many threads.
    visual_workers: int = 0   # decoder threads feeding YOLO + face / TalkNet crop threads
    ocr_workers: int = 0      # threads calling EasyOCR (one shared model)
    motion_workers: int = 0   # threads computing motion saliency (CPU)
    ocr_serialize: bool = False  # put a lock around EasyOCR calls if concurrent use misbehaves

    # ---------------------------------------------------------- sharding
    max_shards: int = 1              # split visual analysis across this many workers
    min_frames_per_shard: int = 3000 # don't create shards smaller than this (~2 min @25fps)

    # ---------------------------------------------------------- shot detection
    shot_threshold: float = 27.0

    # ---------------------------------------------------- person/face tracking
    yolo_weights: str = "yolo26s.pt"
    pose_weights: str = "yolo26s-pose.pt"
    face_backend: str = "mediapipe"  # "mediapipe" (CPU, original) | "pose" (GPU, faster)
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
    face_min_conf: float = 0.5

    # -------------------------------------------------------------------- OCR
    enable_ocr: bool = True
    ocr_langs: List[str] = field(default_factory=lambda: ["en"])
    ocr_samples_per_shot: int = 1

    # ----------------------------------------------------------------- motion
    enable_motion: bool = True

    # --------------------------------------------- active speaker (TalkNet)
    enable_speaker_detection: bool = True
    talknet_batch_frames: int = 1200  # video frames per TalkNet GPU pass (lower if the GPU runs out of memory)

    # ------------------------------ audio branch (runs in parallel, not used for framing)
    transcribe: bool = False  # Demucs vocals split + faster-whisper transcript
    whisper_model: str = "medium"
    whisper_batch_size: int = 0  # batched faster-whisper; 0 = auto (16 on CUDA, 4 on CPU)

    # ---------------------------------------------------------------- framing
    focal_strategy: str = "viterbi"  # "viterbi" | "rules"
    pan_window_frames: int = 12
    cut_threshold_px: float = 450.0  # in source pixels
    still_deadband_px: float = 45.0  # in source pixels

    # ------------------------------------------- story map (cutdowns; see fovea/story/)
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_version: str = "v1beta"  # what the experiment ran on; newer google-genai releases default to "v1"
    gemini_video_processing: str = "AGENTIC"  # "AGENTIC" (Gemini may rewatch parts) | "STATIC"
    refresh_story: bool = False  # ask Gemini again even if this video's folder already holds its story map

    # ------------------------------------------------------------------ limits
    max_duration_s: Optional[float] = None  # reject longer videos (demo mode)

    # ------------------------------------------------------------------ misc
    debug_renders: bool = False
    force: bool = False  # recompute steps even if their outputs exist

    def __post_init__(self) -> None:
        for name in _PATH_FIELDS:
            setattr(self, name, Path(getattr(self, name)))
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        if self.focal_strategy not in ("viterbi", "rules"):
            raise ValueError("focal_strategy must be 'viterbi' or 'rules'")
        if self.face_backend not in ("mediapipe", "pose"):
            raise ValueError("face_backend must be 'mediapipe' or 'pose'")
        self.detect_every = max(1, int(self.detect_every))
        self.max_shards = max(1, int(self.max_shards))
        for name in ("visual_workers", "ocr_workers", "motion_workers", "whisper_batch_size"):
            setattr(self, name, max(0, int(getattr(self, name))))
        self.talknet_batch_frames = max(150, int(self.talknet_batch_frames))  # >= one 6 s window

    # ------------------------------------------------------------ helpers
    def update(self, overrides: Dict[str, Any]) -> "PipelineConfig":
        valid = {f.name for f in fields(self)}
        for k, v in overrides.items():
            if k not in valid:
                raise TypeError(f"Unknown config option: {k}")
            setattr(self, k, v)
        self.__post_init__()
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for name in _PATH_FIELDS:
            d[name] = str(d[name])
        if d.get("output_dir") is not None:
            d["output_dir"] = str(d["output_dir"])
        return d
