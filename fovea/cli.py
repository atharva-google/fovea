"""Command line interface.

    python -m fovea run input.mp4                       # 9:16 vertical
    python -m fovea run input.mp4 -t 9:16 1:1 4:5       # several formats, one analysis
    python -m fovea run input.mp4 -t 720x720            # exact pixel size
    python -m fovea story input.mp4 -d 15 10            # story map + proposed cuts (prints, renders nothing)
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
    r.add_argument("-o", "--output-dir", type=Path, default=d.output_dir,
                   help="Also copy the final videos here (default: they stay in <work-dir>/<video name>/outputs/)")
    r.add_argument("--work-dir", type=Path, default=d.work_dir,
                   help="Data folder: each video gets ONE folder in it, <video name>/, with everything about it")
    r.add_argument("--weights-dir", type=Path, default=d.weights_dir)
    r.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)

    s = r.add_argument_group("speed")
    s.add_argument("--proxy-height", type=int, default=d.proxy_height, help="Analysis resolution (default 720)")
    s.add_argument("--detect-every", type=int, default=d.detect_every, help="Run detector every N frames")
    s.add_argument("--face-backend", choices=["mediapipe", "pose"], default=d.face_backend,
                   help="mediapipe = original (CPU); pose = YOLO-pose head keypoints (GPU, faster)")
    s.add_argument("--max-shards", type=int, default=d.max_shards)
    s.add_argument("--no-gpu-video", action="store_true", help="Disable NVDEC/NVENC")
    s.add_argument("--visual-workers", type=int, default=d.visual_workers,
                   help="Shots decoded/processed at once in the tracking + speaker stages (0 = auto)")
    s.add_argument("--ocr-workers", type=int, default=d.ocr_workers, help="OCR threads (0 = auto)")
    s.add_argument("--motion-workers", type=int, default=d.motion_workers, help="Motion threads (0 = auto)")

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

    # -------------------------------------------------------------- story
    st = sub.add_parser("story", help="Story map for cutdowns: units, Gemini's map, and the cuts picked from it "
                                      "(prints only, renders nothing)")
    st.add_argument("video", help="Path to the input video")
    st.add_argument("-d", "--durations", nargs="+", type=float, default=[15.0],
                    help="Target lengths in seconds to pick cuts for (default: 15)")
    st.add_argument("--modes", nargs="+", choices=["synced", "cutaway"], default=["synced", "cutaway"],
                    help="synced = sound stays with its picture; cutaway = other pictures may cover a line")
    st.add_argument("--gemini-key", default=None,
                    help="Gemini API key (or set GEMINI_API_KEY). Only needed the first time a video file is seen")
    st.add_argument("--gemini-model", default=d.gemini_model)
    st.add_argument("--video-processing", choices=["AGENTIC", "STATIC"], default=d.gemini_video_processing,
                    dest="gemini_video_processing")
    st.add_argument("--refresh", action="store_true", help="Ask Gemini again even if a story map is stored")
    st.add_argument("--json", type=Path, default=None, dest="json_out",
                    help="Also write units, story map and cuts to this JSON file")
    st.add_argument("--work-dir", type=Path, default=d.work_dir,
                    help="Data folder: each video gets ONE folder in it, <video name>/, with everything about it")
    st.add_argument("--weights-dir", type=Path, default=d.weights_dir)
    st.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    st.add_argument("--whisper-model", default=d.whisper_model)
    k = st.add_argument_group("selector (change and re-run: no Gemini call, no re-analysis)")
    k.add_argument("--cut-penalty", type=float, default=3.0, help="Higher = fewer, longer pieces (default 3)")
    k.add_argument("--arc-importance", type=float, default=8.0,
                   help="Units Gemini scores at or above this are always kept (default 8)")
    k.add_argument("--min-importance", type=float, default=1.0, help="Units below this are never picked (default 1)")
    k.add_argument("--imp-power", type=float, default=1.5, help="Higher = importance outweighs length more (default 1.5)")
    k.add_argument("--dur-power", type=float, default=0.5)
    k.add_argument("--tolerance", type=float, default=0.1, help="How far under the target still counts as on target")
    k.add_argument("--over-tolerance", type=float, default=0.15,
                   help="How far over the target optional material may go (the story itself may go further)")
    k.add_argument("--no-hook", action="store_true", help="Never move Gemini's hook to the front")

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

    if args.command == "story":
        _story(args)
        return

    from .pipeline import run

    cfg = PipelineConfig(
        weights_dir=args.weights_dir, work_dir=args.work_dir, output_dir=args.output_dir,
        targets=args.targets, short_side=args.short_side, device=args.device,
        proxy_height=args.proxy_height, detect_every=args.detect_every, face_backend=args.face_backend,
        max_shards=args.max_shards, use_gpu_video=not args.no_gpu_video,
        visual_workers=args.visual_workers, ocr_workers=args.ocr_workers, motion_workers=args.motion_workers,
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


def _story(args) -> None:
    """``fovea story``: the stored story map if this video has one, else analysis + one Gemini call."""
    import json
    import os

    from .story import store
    from .story.report import format_cuts, format_story, make_cuts

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"Input video not found: {video}")
    cfg = PipelineConfig(
        weights_dir=args.weights_dir, work_dir=args.work_dir, device=args.device,
        transcribe=True, whisper_model=args.whisper_model, gemini_model=args.gemini_model,
        gemini_video_processing=args.gemini_video_processing, refresh_story=args.refresh,
    )
    fp = store.fingerprint(video)
    job_root = cfg.work_dir / video.stem          # the video's own folder: everything about it lives here
    found = None if cfg.refresh_story else store.load(job_root / "prep", fp)
    if found:
        units, record = found
        print(f"Story map for {video.name} already stored in {job_root / 'prep'}: no analysis, no Gemini call\n")
    else:
        key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("No story map is stored for this video yet, so a Gemini key is needed: "
                             "pass --gemini-key or set GEMINI_API_KEY")
        from . import steps
        from .steps import story
        from .utils.device import get_device
        from .utils.jobdir import JobDir

        device = get_device(cfg.device)
        job = JobDir(job_root)
        job.set_source(video)
        # The folder is named after the file name. If it holds the analysis of a DIFFERENT file with the
        # same name, those results must not end up in this file's story map: redo the analysis.
        if job.video_info.exists() and job.read_json(job.video_info).get("sha256") not in (None, fp):
            print("  this folder holds the analysis of a different file with the same name: re-running it")
            cfg.force = True
        if not job.job_json.exists():
            job.write_json(job.job_json, {"config": {"targets": cfg.targets}})
        print(f"Fovea story | device={device} | job folder: {job.root}")
        print("▶ analysis: prep, visual, audio (steps already done for this job folder are skipped)", flush=True)
        for i in range(steps.prep.run(job, cfg)):
            steps.visual.run(job, cfg, device, i)
        steps.audio.run(job, cfg, device)
        print("▶ story map", flush=True)
        try:
            out = story.run(job, cfg, key)
        except Exception as e:  # noqa: BLE001 - show the reason, not a stack trace
            raise SystemExit(f"Story map failed: {e}") from None
        units, record = out["units"], out["record"]
        print()

    cuts = make_cuts(units, record["story_map"], args.durations, args.modes,
                     cut_penalty=args.cut_penalty, arc_importance=args.arc_importance,
                     min_importance=args.min_importance, imp_power=args.imp_power, dur_power=args.dur_power,
                     tolerance=args.tolerance, over_tolerance=args.over_tolerance, use_hook=not args.no_hook)
    print(f"Gemini: {record.get('model')} / {record.get('processing')}, answered in {record.get('seconds')}s, "
          f"read from its {record.get('parsed_by')}")
    if record.get("issues"):
        print(f"Fixed while reading the answer: {'; '.join(record['issues'][:10])}")
    print()
    print(format_story(units, record["story_map"]))
    print("\nCUTS (picked by code from the story map)")
    print(format_cuts(cuts))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"video": str(video), "fingerprint": fp, "units": [u.to_dict() for u in units],
             "story_map": record["story_map"], "raw_answer": record.get("raw_answer"), "cuts": cuts},
            indent=1, ensure_ascii=False))
        print(f"\nWritten to {args.json_out}")


if __name__ == "__main__":
    main()
