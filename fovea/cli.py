"""Command line interface.

    python -m fovea run input.mp4                       # 9:16 vertical
    python -m fovea run input.mp4 -t 9:16 1:1 4:5       # several formats, one analysis
    python -m fovea run input.mp4 -t 720x720            # exact pixel size
    python -m fovea download [--all]                    # fetch model weights
    python -m fovea weights                             # show which weights are present
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .framing.geometry import PRESETS

CORE_WEIGHTS = ["yolo", "blazeface", "talknet", "easyocr"]
ALL_WEIGHTS = CORE_WEIGHTS + ["yolo_pose", "whisper", "demucs"]


def _build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(prog="fovea", description="Smart video reframing (vertical / square / any ratio).")
    sub = p.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- run
    r = sub.add_parser("run", help="Reframe a video")
    r.add_argument("video", help="Path to the input (horizontal) video")
    r.add_argument("-t", "--target", nargs="+", default=d.targets, dest="targets",
                   help=f"Output formats: ratios (9:16, 1:1, 4:5), sizes (1080x1920) or presets "
                        f"({', '.join(PRESETS)}). Default: 9:16")
    r.add_argument("--short-side", type=int, default=d.short_side,
                   help="Output short side in px when a ratio is given (default 1080)")
    r.add_argument("-o", "--output-dir", type=Path, default=d.output_dir)
    r.add_argument("--work-dir", type=Path, default=d.work_dir)
    r.add_argument("--weights-dir", type=Path, default=d.weights_dir)
    r.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)

    s = r.add_argument_group("speed")
    s.add_argument("--proxy-height", type=int, default=d.proxy_height, help="Analysis resolution (default 720)")
    s.add_argument("--detect-every", type=int, default=d.detect_every, help="Run detector every N frames")
    s.add_argument("--face-backend", choices=["mediapipe", "pose"], default=d.face_backend,
                   help="mediapipe = original (CPU); pose = YOLO-pose head keypoints (GPU, faster)")
    s.add_argument("--max-shards", type=int, default=d.max_shards)
    s.add_argument("--no-gpu-video", action="store_true", help="Disable NVDEC/NVENC")

    g = r.add_argument_group("analysis")
    g.add_argument("--shot-threshold", type=float, default=d.shot_threshold)
    g.add_argument("--no-ocr", action="store_true", help="Skip OCR text detection")
    g.add_argument("--ocr-langs", nargs="+", default=d.ocr_langs)
    g.add_argument("--no-motion", action="store_true", help="Skip motion saliency")
    g.add_argument("--no-speaker", action="store_true", help="Skip TalkNet active speaker detection")
    g.add_argument("--transcribe", action="store_true", help="Also run Demucs + Whisper (in parallel)")
    g.add_argument("--whisper-model", default=d.whisper_model)

    f = r.add_argument_group("framing")
    f.add_argument("--focal-strategy", choices=["viterbi", "rules"], default=d.focal_strategy)
    f.add_argument("--pan-window", type=int, default=d.pan_window_frames, dest="pan_window_frames")
    f.add_argument("--cut-threshold", type=float, default=d.cut_threshold_px, dest="cut_threshold_px")
    f.add_argument("--deadband", type=float, default=d.still_deadband_px, dest="still_deadband_px")

    m = r.add_argument_group("misc")
    m.add_argument("--debug-renders", action="store_true", help="Write annotated tracking/motion videos")
    m.add_argument("--force", action="store_true", help="Recompute every stage even if outputs exist")

    # ----------------------------------------------------------- download
    dl = sub.add_parser("download", help="Download model weights")
    dl.add_argument("--weights-dir", type=Path, default=d.weights_dir)
    dl.add_argument("--all", action="store_true", help="Also YOLO-pose, Whisper and Demucs")
    dl.add_argument("--hf-repo", default=None, help="Hugging Face mirror repo id (or set FOVEA_HF_REPO)")
    dl.add_argument("--whisper-model", default=d.whisper_model)

    # ------------------------------------------------------------ weights
    w = sub.add_parser("weights", help="List model weights and whether they are downloaded")
    w.add_argument("--weights-dir", type=Path, default=d.weights_dir)
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "download":
        from .utils.weights import download_all

        download_all(args.weights_dir, ALL_WEIGHTS if args.all else CORE_WEIGHTS,
                     whisper_model=args.whisper_model, hf_repo=args.hf_repo)
        return

    if args.command == "weights":
        from .utils.weights import list_weights

        for row in list_weights(args.weights_dir):
            mark = "✓" if row["present"] else "✗"
            print(f"{mark} {row['key']:<10} {row['used_by']:<11} {row['license']:<11} {row['path']}")
        return

    from .pipeline import run

    cfg = PipelineConfig(
        weights_dir=args.weights_dir, work_dir=args.work_dir, output_dir=args.output_dir,
        targets=args.targets, short_side=args.short_side, device=args.device,
        proxy_height=args.proxy_height, detect_every=args.detect_every, face_backend=args.face_backend,
        max_shards=args.max_shards, use_gpu_video=not args.no_gpu_video,
        shot_threshold=args.shot_threshold, enable_ocr=not args.no_ocr, ocr_langs=args.ocr_langs,
        enable_motion=not args.no_motion, enable_speaker_detection=not args.no_speaker,
        transcribe=args.transcribe, whisper_model=args.whisper_model,
        focal_strategy=args.focal_strategy, pan_window_frames=args.pan_window_frames,
        cut_threshold_px=args.cut_threshold_px, still_deadband_px=args.still_deadband_px,
        debug_renders=args.debug_renders, force=args.force,
    )
    outputs = run(args.video, cfg)
    print("\nDone:")
    for o in outputs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
