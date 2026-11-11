"""CLI entry point: `python -m frontend` starts the UI server on port 3000."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fovea web frontend")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    parser.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", "http://localhost:8000"))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (requires uvicorn)")
    args = parser.parse_args()

    os.environ["BACKEND_URL"] = args.backend_url.rstrip("/")
    print(f"Fovea Studio UI running at http://localhost:{args.port}  (BACKEND_URL={os.environ['BACKEND_URL']})")

    try:
        import uvicorn
        import fastapi  # noqa: F401

        uvicorn.run("frontend.server:app", host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        from .server import serve_stdlib

        serve_stdlib(args.host, args.port)


if __name__ == "__main__":
    main()
