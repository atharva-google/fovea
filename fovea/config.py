"""Every setting, in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent     # the project folder
THINKING_LEVELS = ("default", "minimal", "low", "medium", "high")


@dataclass
class Settings:
    # ---- where things go (both inside the project folder, both ignored by git)
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("FOVEA_DATA", ROOT / "fovea_data")))
    weights_dir: Path = field(default_factory=lambda: Path(os.environ.get("FOVEA_WEIGHTS", ROOT / "weights")))

    # ---- analysis copy of the video
    proxy_height: int = 720
    proxy_fps: Optional[float] = None      # None = the video's own frame rate, so shot times fall exactly on its frames

    # ---- shots (PySceneDetect)
    shot_threshold: float = 27.0

    # ---- transcript (faster-whisper)
    whisper_model: str = "small"
    whisper_device: str = "cpu"            # "cpu" | "cuda"
    language: Optional[str] = None         # e.g. "hi"; None = detect

    # ---- Gemini
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_version: str = "v1beta"
    gemini_fps: float = 4.0                # frames per second Gemini looks at (its own default is 1: too few for short shots)
    thinking: str = "default"              # "default" (the model decides) | "minimal" | "low" | "medium" | "high"
    agentic: bool = False                  # agentic video processing: no output schema, the answer is parsed from a JSON block

    # ---- rendering
    dead_zone: float = 3.0                 # a move of the focus line smaller than this (% of the width) is not followed
    crf: int = 18                          # x264 quality of the rendered videos (lower = better, bigger)

    # ---- re-running
    refresh: bool = False                  # ask Gemini again even if an answer is stored
    force: bool = False                    # redo every step

    def __post_init__(self) -> None:
        self.data_dir, self.weights_dir = Path(self.data_dir), Path(self.weights_dir)
