"""Command line.

    python -m fovea analyze my_ad.mp4            # shots + transcript + base analysis (one Gemini call)
    python -m fovea analyze my_ad.mp4 --refresh  # ask Gemini again
"""

from __future__ import annotations

import argparse
import os

from .config import Settings


def main(argv=None) -> None:
    d = Settings()
    parser = argparse.ArgumentParser(prog="fovea", description="Gemini-led reframing and cutdowns of ad videos.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Shots, transcript and the base analysis; prints the result")
    a.add_argument("video", help="Path to the video")
    a.add_argument("--gemini-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    a.add_argument("--gemini-model", default=d.gemini_model)
    a.add_argument("--agentic", action="store_true",
                   help="Agentic video processing (Gemini may rewatch parts; no output schema)")
    a.add_argument("--whisper-model", default=d.whisper_model)
    a.add_argument("--whisper-device", choices=["cpu", "cuda"], default=d.whisper_device)
    a.add_argument("--language", default=None, help="Spoken language, e.g. hi (default: detect)")
    a.add_argument("--refresh", action="store_true", help="Ask Gemini again even if an answer is stored")
    a.add_argument("--force", action="store_true", help="Redo every step")
    args = parser.parse_args(argv)

    from . import analysis, gemini, pipeline

    settings = Settings(gemini_model=args.gemini_model, agentic=args.agentic, whisper_model=args.whisper_model,
                        whisper_device=args.whisper_device, language=args.language, refresh=args.refresh,
                        force=args.force)
    try:
        result = pipeline.analyze(args.video, settings, args.gemini_key or os.environ.get("GEMINI_API_KEY"))
    except (FileNotFoundError, analysis.KeyRequired, gemini.GeminiError) as e:
        raise SystemExit(str(e)) from None
    print()
    print(analysis.report(result))


if __name__ == "__main__":
    main()
