"""Video probing, job-folder layout, and ffmpeg helpers (proxy, audio, GPU encode/decode)."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- probing
@dataclass
class VideoInfo:
    path: str
    filename: str  # stem, e.g. "mmt" for "mmt.mp4"
    extension: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def probe_video(video_path: str | Path) -> VideoInfo:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = total_frames / fps if fps > 0 and total_frames > 0 else None
    return VideoInfo(str(video_path), video_path.stem, video_path.suffix, fps or 25.0,
                     total_frames, width, height, duration)


# --------------------------------------------------------------------------- ffmpeg capabilities
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it (macOS: `brew install ffmpeg`, Ubuntu: `apt install ffmpeg`).")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True if ffmpeg can actually encode with NVENC on this machine (tested once)."""
    if shutil.which("ffmpeg") is None or not _cuda_available():
        return False
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return res.returncode == 0


@lru_cache(maxsize=1)
def cuda_hwaccel_available() -> bool:
    if shutil.which("ffmpeg") is None or not _cuda_available():
        return False
    res = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True)
    return "cuda" in res.stdout


def h264_encoder_args(use_gpu: bool = True, quality: int = 21) -> List[str]:
    if use_gpu and nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]


def hwaccel_args(use_gpu: bool = True) -> List[str]:
    return ["-hwaccel", "cuda"] if use_gpu and cuda_hwaccel_available() else []


def run_ffmpeg(args: list) -> None:
    _require_ffmpeg()
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{res.stderr[-2000:]}")


# --------------------------------------------------------------------------- prep helpers
def make_proxy(source: Path, output: Path, height: int = 720, fps: float = 25.0, use_gpu: bool = True) -> Path:
    """Silent, constant-fps, downscaled H.264 copy used for all analysis.

    Keyframe every second so seeking to shot starts is cheap.
    """
    vf = f"fps={fps},scale=-2:'min({int(height)},ih)'"
    run_ffmpeg([
        *hwaccel_args(use_gpu), "-i", str(source), "-an", "-vf", vf,
        *h264_encoder_args(use_gpu, quality=23), "-g", str(int(round(fps))),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ])
    return output


def extract_audio(source: Path, output: Path) -> Path:
    """16 kHz stereo PCM wav (what TalkNet / Whisper / Demucs consume)."""
    run_ffmpeg(["-i", str(source), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2", str(output)])
    return output


def make_silent_audio(output: Path, duration_s: float) -> Path:
    """For videos without sound, so downstream audio steps still have a file to read."""
    run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo", "-t", f"{max(duration_s, 0.1):.3f}",
                "-acodec", "pcm_s16le", str(output)])
    return output


def has_audio_stream(source: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return True
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(source)], capture_output=True, text=True)
    return bool(res.stdout.strip())


# --------------------------------------------------------------------------- frame IO
def iter_frames(video_path: str | Path, start_frame: int = 0, end_frame: Optional[int] = None,
                prefetch: int = 32) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, frame)`` sequentially, decoding in a background thread.

    Decoding overlaps with whatever the consumer does (GPU inference), which is
    a free speedup since OpenCV releases the GIL while decoding.
    """
    q: "queue.Queue" = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    _END = object()

    def _producer():
        cap = cv2.VideoCapture(str(video_path))
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        try:
            while not stop.is_set() and (end_frame is None or idx < end_frame):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                q.put((idx, frame))
                idx += 1
        finally:
            cap.release()
            q.put(_END)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            yield item
    finally:
        stop.set()
        # drain so the producer can exit
        while t.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(timeout=0.05)


class ShotReader:
    """One OpenCV reader that decodes frame ranges ``[start, end)`` on demand.

    Each worker thread owns one. It only seeks when the requested range does not start where
    the previous read stopped, so consecutive shots are read without seeking. Frames that the
    caller does not want are only grabbed (decoded, but no colour conversion / copy).
    """

    def __init__(self, video_path: str | Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        self.pos: Optional[int] = 0

    def frames(self, start: int, end: int, want: Optional[Callable[[int], bool]] = None
               ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(idx, frame)`` for every frame in ``[start, end)`` for which ``want(idx)`` is true."""
        if end <= start:
            return
        if self.pos != start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.pos = start
        for idx in range(start, end):
            if want is None or want(idx):
                ret, frame = self.cap.read()
                ok = ret and frame is not None
            else:
                ok, frame = self.cap.grab(), None
            if not ok:
                self.pos = None  # unknown position -> seek next time
                return
            self.pos = idx + 1
            if frame is not None:
                yield idx, frame

    def frame(self, idx: int) -> Optional[np.ndarray]:
        for _, f in self.frames(idx, idx + 1):
            return f
        return None

    def close(self) -> None:
        self.cap.release()


class ThreadLocalReaders:
    """A ``ShotReader`` per thread (for thread pools); ``close()`` releases all of them."""

    def __init__(self, video_path: str | Path):
        self.video_path = video_path
        self._local = threading.local()
        self._all: List[ShotReader] = []
        self._lock = threading.Lock()

    def get(self) -> ShotReader:
        r = getattr(self._local, "reader", None)
        if r is None:
            r = self._local.reader = ShotReader(self.video_path)
            with self._lock:
                self._all.append(r)
        return r

    def close(self) -> None:
        with self._lock:
            for r in self._all:
                r.close()
            self._all.clear()


def auto_workers(requested: int, cap: int = 4) -> int:
    """``requested`` if > 0, else a default from the CPU count (at most ``cap``)."""
    if requested and requested > 0:
        return int(requested)
    return max(1, min(cap, (os.cpu_count() or 2) // 2))


def split_contiguous(shots: Sequence[dict], n_groups: int) -> List[List[dict]]:
    """Split shots (sorted by start) into up to ``n_groups`` runs of consecutive shots with
    ~equal frame counts. A worker reading one run seeks once instead of once per shot."""
    shots = list(shots)
    if not shots:
        return []
    n = max(1, min(n_groups, len(shots)))
    total = sum(int(s["end_frame"]) - int(s["start_frame"]) for s in shots)
    target = total / n
    groups: List[List[dict]] = []
    current: List[dict] = []
    acc = 0
    for s in shots:
        current.append(s)
        acc += int(s["end_frame"]) - int(s["start_frame"])
        if acc >= target * (len(groups) + 1) and len(groups) < n - 1:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


class FFmpegFrameReader:
    """Sequential raw BGR frames from ffmpeg (uses NVDEC when available)."""

    def __init__(self, path: str | Path, width: int, height: int, use_gpu: bool = True):
        _require_ffmpeg()
        self.w, self.h = width, height
        self.nbytes = width * height * 3
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", *hwaccel_args(use_gpu), "-i", str(path),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.nbytes * 4,
        )

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.nbytes)
            if len(buf) < self.nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
