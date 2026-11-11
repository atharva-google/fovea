#!/usr/bin/env python3
"""Compare the current code against a baseline git revision: per-stage speed + output parity.

    python scripts/bench_visual.py input.mp4                       # baseline = HEAD
    python scripts/bench_visual.py input.mp4 --baseline-ref 8b434c7 --transcribe
    python scripts/bench_visual.py input.mp4 --set face_backend=pose --set visual_workers=4

What it does:
  1. checks out the baseline revision in a temporary ``git worktree``
  2. prep: runs the baseline and the current prep step (each in its own job folder) and times them
  3. visual: runs ``steps.visual.analyze_shard`` for every shard with BOTH versions on the SAME
     prep output (the current one), each in its own Python process
  4. audio (``--transcribe``): Demucs + Whisper with both versions
  5. prints wall times and compares tracks / speaker scores / OCR / motion / transcript

Needs the same environment as the pipeline (a GPU box for realistic numbers).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]

RUNNER = textwrap.dedent('''
    import json, pickle, sys, time
    from pathlib import Path
    args = json.load(open(sys.argv[1]))
    import pandas as pd
    from fovea.config import PipelineConfig
    from fovea.utils.device import get_device
    from fovea.utils.jobdir import JobDir
    from fovea import steps

    cfg = PipelineConfig(**args["config"])
    device = get_device(cfg.device)
    out = {}

    if args["mode"] == "prep":
        job = JobDir(args["job_root"])
        job.set_source(args["video"])
        cfg.force = True
        t = time.perf_counter()
        steps.prep.run(job, cfg, log=lambda m: None)
        out["prep_wall"] = time.perf_counter() - t

    elif args["mode"] == "analyze":
        job = JobDir(args["job_root"])
        _, proxy, _ = job.video_infos()
        plan = job.read_json(job.shards)["shards"]
        shots = pd.read_csv(job.shots)
        results = []
        t0 = time.perf_counter()
        for p in plan:
            s = shots[shots["shot_id"].isin(p["shot_ids"])]
            t = time.perf_counter()
            r = steps.visual.analyze_shard(str(job.proxy), str(job.audio), s, proxy.width, proxy.height,
                                           proxy.fps, cfg, device, log=lambda m: print("   ", m, flush=True))
            r["wall"] = time.perf_counter() - t
            results.append(r)
        out["visual"] = results
        out["visual_wall"] = time.perf_counter() - t0
        if args.get("transcribe"):
            from fovea.utils.weights import ensure_for
            adir = Path(args["audio_dir"])
            adir.mkdir(parents=True, exist_ok=True)
            ensure_for(cfg, "demucs")
            wdir = ensure_for(cfg, "whisper")
            t = time.perf_counter()
            vocals, _ = steps.audio.separate_vocals(job.audio, adir, device)
            out["demucs_wall"] = time.perf_counter() - t
            t = time.perf_counter()
            out["transcript"] = steps.audio.transcribe(vocals, wdir, device).to_dict("records")
            out["whisper_wall"] = time.perf_counter() - t

    with open(args["out"], "wb") as f:
        pickle.dump(out, f)
''')


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def run_variant(code_root: Path, args: Dict[str, Any], tmp: Path, name: str) -> Dict[str, Any]:
    runner, arg_file, out = tmp / "runner.py", tmp / f"{name}_{args['mode']}.json", tmp / f"{name}_{args['mode']}.pkl"
    runner.write_text(RUNNER)
    arg_file.write_text(json.dumps({**args, "out": str(out)}))
    env = {**os.environ, "PYTHONPATH": str(code_root)}
    print(f"▶ {name}: {args['mode']}", flush=True)
    subprocess.run([sys.executable, str(runner), str(arg_file)], env=env, cwd=str(tmp), check=True)
    with open(out, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- comparisons
def compare_tracks(a: Dict, b: Dict) -> str:
    frames = mism = 0
    max_px = 0
    for sid in set(a) | set(b):
        fa, fb = a.get(sid, {}), b.get(sid, {})
        for f in set(fa) | set(fb):
            frames += 1
            da = {(d["track_id"], d["cname"]): d for d in fa.get(f, [])}
            db = {(d["track_id"], d["cname"]): d for d in fb.get(f, [])}
            if da.keys() != db.keys():
                mism += 1
                continue
            for k in da:
                for key in ("bbox", "face_bbox"):
                    if da[k].get(key) is not None and db[k].get(key) is not None:
                        max_px = max(max_px, max(abs(x - y) for x, y in zip(da[k][key], db[k][key])))
    return f"{frames} frames, {mism} with different detections, max box diff {max_px}px"


def compare_speakers(a: List[Dict], b: List[Dict]) -> str:
    def index(scores):
        return {(s["shot_id"], s["track_id"]): s for s in scores}
    ia, ib = index(a), index(b)
    common = ia.keys() & ib.keys()
    max_diff = 0.0
    for k in common:
        sa, sb = ia[k]["scores"], ib[k]["scores"]
        if len(sa) == len(sb) and len(sa):
            max_diff = max(max_diff, max(abs(float(x) - float(y)) for x, y in zip(sa, sb)))
    return f"{len(ia)} vs {len(ib)} tracks, {len(common)} shared, max score diff {max_diff:.4f}"


def compare_ocr(a: List[Dict], b: List[Dict]) -> str:
    sa = {(r["shot_id"], r["text"]) for r in a}
    sb = {(r["shot_id"], r["text"]) for r in b}
    return f"{len(sa)} vs {len(sb)} texts, {len(sa & sb)} identical, {len(sa ^ sb)} differ"


def compare_motion(a: Dict, b: Dict) -> str:
    keys = a.keys() & b.keys()
    diff = max((abs(a[k]["motion_cx"] - b[k]["motion_cx"]) for k in keys), default=0.0)
    return f"{len(a)} vs {len(b)} frames, max motion_cx diff {diff:.2f}"


def compare_transcript(a: List[Dict], b: List[Dict]) -> str:
    ta = " ".join(r["segment_transcript"] for r in a)
    tb = " ".join(r["segment_transcript"] for r in b)
    ratio = difflib.SequenceMatcher(None, ta.split(), tb.split()).ratio()
    return f"{len(a)} vs {len(b)} segments, word similarity {ratio:.3f}"


def merge_shards(results: List[Dict], key: str):
    if key in ("tracks", "motion"):
        merged: Dict = {}
        for r in results:
            merged.update(r.get(key, {}))
        return merged
    return [x for r in results for x in r.get(key, [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--baseline-ref", default="HEAD", help="git revision to compare against (default HEAD)")
    ap.add_argument("--work-dir", type=Path, default=None, help="where to put job folders (default: temp dir)")
    ap.add_argument("--transcribe", action="store_true", help="also benchmark Demucs + Whisper")
    ap.add_argument("--skip-prep-bench", action="store_true", help="don't time the baseline prep")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="PipelineConfig override for both versions (keys the baseline lacks are dropped for it)")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from fovea.config import PipelineConfig

    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: _parse_value(v) for k, v in overrides.items()}
    base_cfg = PipelineConfig()
    config = {"weights_dir": str(base_cfg.weights_dir), **overrides}

    tmp = Path(tempfile.mkdtemp(prefix="fovea_bench_"))
    work = (args.work_dir or tmp / "work").resolve()
    worktree = tmp / "baseline"
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(worktree), args.baseline_ref],
                   check=True, capture_output=True)
    try:
        # the baseline may not know the new config keys -> keep only the ones it has
        base_keys = subprocess.run(
            [sys.executable, "-c", "import json,dataclasses;from fovea.config import PipelineConfig as P;"
             "print(json.dumps([f.name for f in dataclasses.fields(P)]))"],
            env={**os.environ, "PYTHONPATH": str(worktree)}, capture_output=True, text=True, check=True)
        base_config = {k: v for k, v in config.items() if k in json.loads(base_keys.stdout)}

        video = str(args.video.resolve())
        new_job, base_job = work / "current" / args.video.stem, work / "baseline" / args.video.stem
        report = {}
        prep_new = run_variant(REPO, {"mode": "prep", "config": config, "job_root": str(new_job),
                                      "video": video}, tmp, "current")
        report["prep"] = {"current": prep_new["prep_wall"]}
        if not args.skip_prep_bench:
            prep_base = run_variant(worktree, {"mode": "prep", "config": base_config, "job_root": str(base_job),
                                               "video": video}, tmp, "baseline")
            report["prep"]["baseline"] = prep_base["prep_wall"]

        common = {"mode": "analyze", "job_root": str(new_job), "transcribe": args.transcribe}
        base = run_variant(worktree, {**common, "config": base_config, "audio_dir": str(tmp / "audio_base")},
                           tmp, "baseline")
        cur = run_variant(REPO, {**common, "config": config, "audio_dir": str(tmp / "audio_cur")}, tmp, "current")

        print("\n================ timings (s) ================")
        print(f"{'part':<14}{'baseline':>12}{'current':>12}{'speed-up':>10}")
        rows = [("prep", report["prep"].get("baseline"), report["prep"]["current"]),
                ("visual", base["visual_wall"], cur["visual_wall"])]
        if args.transcribe:
            rows += [("demucs", base["demucs_wall"], cur["demucs_wall"]),
                     ("whisper", base["whisper_wall"], cur["whisper_wall"])]
        for name, b, c in rows:
            sb = f"{b:12.1f}" if b is not None else f"{'-':>12}"
            su = f"{b / c:9.2f}x" if b and c else f"{'-':>10}"
            print(f"{name:<14}{sb}{c:12.1f}{su}")
        for i, (rb, rc) in enumerate(zip(base["visual"], cur["visual"])):
            print(f"\nshard {i} baseline parts: {rb.get('timings', {})}")
            print(f"shard {i} current  parts: {rc.get('timings', {})}")

        print("\n================ parity ================")
        bv, cv = base["visual"], cur["visual"]
        print("tracks   :", compare_tracks(merge_shards(bv, "tracks"), merge_shards(cv, "tracks")))
        print("speakers :", compare_speakers(merge_shards(bv, "speaker_scores"), merge_shards(cv, "speaker_scores")))
        print("ocr      :", compare_ocr(merge_shards(bv, "ocr"), merge_shards(cv, "ocr")))
        print("motion   :", compare_motion(merge_shards(bv, "motion"), merge_shards(cv, "motion")))
        if args.transcribe:
            print("transcript:", compare_transcript(base["transcript"], cur["transcript"]))
        print(f"\nartifacts kept in {tmp}")
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(worktree)], check=False)
        if args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
