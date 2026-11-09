"""Entry point.

    python -m backend serve                       # start the API (local mode by default)
    python -m backend serve --env my.env          # with settings from an env file
    python -m backend worker --step prep --job-path users/<uid>/<job_id>   # one step (cloud tasks)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_env_file(path: str) -> None:
    """Minimal KEY=VALUE reader. Real environment variables win over the file."""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def main() -> None:
    p = argparse.ArgumentParser(prog="backend", description="Fovea backend")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Run the API")
    s.add_argument("--env", help="Env file with settings (see backend/local.env.example)")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))

    w = sub.add_parser("worker", help="Run one pipeline step (used by Cloud Run Job tasks)")
    w.add_argument("--step", required=True, choices=["prep", "visual", "audio", "render", "all"])
    w.add_argument("--job-path", required=True, help="users/<user_id>/<job_id>")
    w.add_argument("--env", help="Env file with settings")

    args = p.parse_args()
    if getattr(args, "env", None):
        load_env_file(args.env)

    if args.command == "serve":
        import uvicorn

        os.environ.setdefault("PORT", str(args.port))
        from .settings import get_settings

        st = get_settings()
        print(f"Fovea backend | mode={st.mode} | http://{args.host}:{args.port}  (docs: /docs)")
        if st.is_local:
            print(f"  files + job records: {st.data_dir}")
        # one process: local mode keeps job state in memory-locked files and runs jobs in threads
        uvicorn.run("backend.api:app", host=args.host, port=args.port, workers=1)
    else:
        from .worker import main as worker_main

        sys.exit(worker_main(args.step, args.job_path))


if __name__ == "__main__":
    main()
