"""STEP 4 - render: decide where the camera looks, then write the output videos.

  1. merge all visual shards and mark who is speaking
  2. focal point per frame (Viterbi search, or simple rules)       -> framing/focal.py, framing/rules.py
  3. for each output format: plan the camera (hold / pan / cut)    -> framing/camera.py
  4. crop the ORIGINAL video along that path and encode it (one pass, NVENC when available)

Reads:  input/<video>, prep/*, visual/*
Writes: framing/*.csv, outputs/<label>.mp4, debug/*.mp4 (optional)
"""

from __future__ import annotations

import pickle
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..framing import CropGeometry, OutputTarget, find_focal_points_rules, find_focal_points_viterbi, plan_camera
from ..models.ocr import records_to_df
from ..models.speakers import attach_speaking_scores
from ..utils.jobdir import JobDir
from ..utils.progress import Counter, Progress, noop
from ..utils.video import FFmpegFrameReader, VideoInfo, h264_encoder_args


def run(job: JobDir, cfg: PipelineConfig, log=print, progress: Optional[Progress] = None) -> Dict[str, str]:
    """Returns {format label: output path}, e.g. {"9x16": ".../outputs/9x16.mp4"}."""
    progress = progress or noop
    n_stages = 1 + len(cfg.targets)
    progress("focal points", index=1, count=n_stages)
    src, proxy, _ = job.video_infos()
    shots = pd.read_csv(job.shots)
    data = merge_shards(job)
    tracks = attach_speaking_scores(data["tracks"], data["speaker_scores"])
    ocr = records_to_df(data["ocr"])

    # Focal points are found in proxy pixels, then scaled to source pixels
    common = dict(shots_df=shots, shot_tracks=tracks, ocr_df=ocr, motion_results=data["motion"],
                  video_width=proxy.width, fps=proxy.fps)
    if cfg.focal_strategy == "rules":
        focal = find_focal_points_rules(**common)
    else:
        focal = find_focal_points_viterbi(**common, video_height=proxy.height)
    focal["target_cx"] = focal["target_cx"] * (src.width / proxy.width)
    focal.to_csv(job.framing_dir / "focal_points.csv", index=False)

    outputs: Dict[str, str] = {}
    for i, spec in enumerate(cfg.targets):
        target = OutputTarget.parse(spec)
        geom = CropGeometry.build(src.width, src.height, target, cfg.short_side)
        t = time.time()
        traj = plan_camera(focal, geom, cfg.pan_window_frames, cfg.cut_threshold_px, cfg.still_deadband_px)
        traj.to_csv(job.framing_dir / f"trajectory_{target.label}.csv", index=False)
        out = job.outputs_dir / f"{target.label}.mp4"
        counter = Counter(progress, f"encode {target.label}", src.total_frames or 0, "frames", i + 2, n_stages)
        render_reframed(job.source, src, traj, geom, out, analysis_fps=proxy.fps, use_gpu=cfg.use_gpu_video,
                        on_frames=counter.add)
        outputs[target.label] = str(out)
        log(f"  {target.label}: {geom.out_w}x{geom.out_h} ({time.time() - t:.1f}s) -> {out}")

    if cfg.debug_renders:
        from ..utils.debug_video import render_motion_debug, render_tracking_debug

        render_tracking_debug(str(job.proxy), tracks, shots, job.debug_dir / "tracking.mp4")
        if data["motion"]:
            render_motion_debug(str(job.proxy), data["motion"], shots, job.debug_dir / "motion.mp4")
    return outputs


def merge_shards(job: JobDir) -> Dict[str, Any]:
    n = job.read_json(job.shards)["num_shards"]
    merged = {"tracks": {}, "motion": {}, "ocr": [], "speaker_scores": []}
    for i in range(n):
        path = job.shard_result(i)
        if not path.exists():
            raise FileNotFoundError(f"Missing visual result for shard {i}: {path}")
        with open(path, "rb") as f:
            r = pickle.load(f)
        merged["tracks"].update(r["tracks"])
        merged["motion"].update(r["motion"])
        merged["ocr"].extend(r["ocr"])
        merged["speaker_scores"].extend(r["speaker_scores"])
    return merged


def render_reframed(source: str | Path, src: VideoInfo, trajectory_df: pd.DataFrame, geometry: CropGeometry,
                    output_path: str | Path, analysis_fps: float = 25.0, use_gpu: bool = True,
                    on_frames: Optional[Callable[[int], None]] = None) -> Path:
    """source --decode--> crop + resize --encode (+ original audio)--> output, in one pass.

    The trajectory is in analysis (proxy, 25fps) frames; each source frame is mapped to
    the analysis frame at the same timestamp, so sources at any fps work.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = trajectory_df["frame_idx"].astype(int).to_numpy()
    n = int(frames.max()) + 1 if len(frames) else 1
    cx_lookup = np.full(n, geometry.src_w / 2.0)
    cx_lookup[frames] = trajectory_df["smoothed_cx"].to_numpy(dtype=np.float64)
    ratio = analysis_fps / (src.fps or analysis_fps)
    out_w, out_h = geometry.out_w, geometry.out_h

    with tempfile.TemporaryFile() as err:
        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{src.fps}", "-i", "-",
             "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
             *h264_encoder_args(use_gpu), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)],
            stdin=subprocess.PIPE, stderr=err,
        )
        reader = FFmpegFrameReader(source, src.width, src.height, use_gpu=use_gpu)
        try:
            for i, frame in enumerate(reader):
                x1, y1, x2, y2 = geometry.window(cx_lookup[min(int(round(i * ratio)), n - 1)])
                crop = frame[y1:y2, x1:x2]
                encoder.stdin.write(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4).tobytes())
                if on_frames and (i + 1) % 25 == 0:
                    on_frames(25)
        except BrokenPipeError:
            pass
        finally:
            reader.close()
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            rc = encoder.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"FFmpeg encode failed:\n{err.read().decode(errors='ignore')[-2000:]}")
    return output_path
