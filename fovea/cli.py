"""Command line.

    python -m fovea analyze my_ad.mp4            # shots + transcript + base analysis (one Gemini call)
    python -m fovea analyze my_ad.mp4 --refresh  # ask Gemini again
    python -m fovea render my_ad.mp4             # the video as 9:16, 1:1, 4:3 and 3:4 (analyses it first if needed)
    python -m fovea render my_ad.mp4 --formats 9:16 4:5
"""

from __future__ import annotations

import argparse
import os

from .config import THINKING_LEVELS, Settings
from .render import DEFAULT_FORMATS


def main(argv=None) -> None:
    d = Settings()
    parser = argparse.ArgumentParser(prog="fovea", description="Gemini-led reframing and cutdowns of ad videos.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("video", help="Path to the video")
    common.add_argument("--gemini-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    common.add_argument("--gemini-model", default=d.gemini_model)
    common.add_argument("--gemini-fps", type=float, default=d.gemini_fps,
                        help="Frames per second Gemini looks at (default: %(default)g)")
    common.add_argument("--thinking", choices=THINKING_LEVELS, default=d.thinking,
                        help="How hard Gemini thinks before it answers (default: the model decides)")
    common.add_argument("--agentic", action="store_true",
                        help="Agentic video processing (Gemini may rewatch parts; no output schema)")
    common.add_argument("--whisper-model", default=d.whisper_model)
    common.add_argument("--whisper-device", choices=["cpu", "cuda"], default=d.whisper_device)
    common.add_argument("--language", default=None, help="Spoken language, e.g. hi (default: detect)")
    common.add_argument("--refresh", action="store_true", help="Ask Gemini again even if an answer is stored")
    common.add_argument("--force", action="store_true", help="Redo every step")

    sub.add_parser("analyze", parents=[common], help="Shots, transcript and the base analysis; prints the result")
    r = sub.add_parser("render", parents=[common], help="The video in other shapes, cropped along the focus line")
    r.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), metavar="W:H",
                   help="Shapes to render (default: %(default)s)")
    r.add_argument("--dead-zone", type=float, default=d.dead_zone,
                   help="Moves of the focus line smaller than this, in %% of the width, are not followed "
                        "(default: %(default)g)")
    args = parser.parse_args(argv)

    from . import analysis, gemini, pipeline

    settings = Settings(gemini_model=args.gemini_model, gemini_fps=args.gemini_fps, thinking=args.thinking,
                        agentic=args.agentic, whisper_model=args.whisper_model, whisper_device=args.whisper_device,
                        language=args.language, refresh=args.refresh, force=args.force,
                        dead_zone=getattr(args, "dead_zone", d.dead_zone))
    key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    try:
        if args.command == "analyze":
            result = pipeline.analyze(args.video, settings, key)
            print()
            print(analysis.report(result))
        else:
            pipeline.render(args.video, args.formats, settings, key)
    except (FileNotFoundError, ValueError, analysis.KeyRequired, gemini.GeminiError) as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
