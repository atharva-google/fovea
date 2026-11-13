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


def _project_root() -> Path:
    """Return the repo root if running from a git checkout, else the CWD."""
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "pyproject.toml").exists():
        return repo_root
    return Path.cwd()


PROJECT_ROOT = _project_root()


def default_weights_dir() -> Path:
    return Path(os.environ.get("FOVEA_WEIGHTS_DIR", PROJECT_ROOT / "weights"))


def default_work_dir() -> Path:
    return Path(os.environ.get("FOVEA_WORK_DIR", PROJECT_ROOT / "working"))


def default_output_dir() -> Path:
    return Path(os.environ.get("FOVEA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


_PATH_FIELDS = ("weights_dir", "work_dir", "output_dir")


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------ paths
    weights_dir: Path = field(default_factory=default_weights_dir)
    work_dir: Path = field(default_factory=default_work_dir)      # per-video job folders
    output_dir: Path = field(default_factory=default_output_dir)  # final videos (local runs)

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
    story_dir: Optional[Path] = None  # stored units + story maps, one folder per video file (None = <work_dir>/_story)
    gemini_model: str = "gemini-flash-latest"
    gemini_video_processing: str = "AGENTIC"  # "AGENTIC" (Gemini may rewatch parts) | "STATIC"
    refresh_story: bool = False  # ask Gemini again even if a story map is stored for this video file

    # ------------------------------------------------------------------ limits
    max_duration_s: Optional[float] = None  # reject longer videos (demo mode)

    # ------------------------------------------------------------------ misc
    debug_renders: bool = False
    force: bool = False  # recompute steps even if their outputs exist

    def __post_init__(self) -> None:
        for name in _PATH_FIELDS:
            setattr(self, name, Path(getattr(self, name)))
        if self.story_dir is not None:
            self.story_dir = Path(self.story_dir)
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
        if d.get("story_dir") is not None:
            d["story_dir"] = str(d["story_dir"])
        return d
