"""What the Fovea icon runs: get this computer ready, then start the local server (server.py).

Installed by the setup file, Fovea lives in one folder:

    Fovea/
        app/        the code (replaced when the setup file is run again)
        venv/       its own Python and libraries
        bin/        uv, which installed them
        data/       your projects            } kept when the code is replaced
        weights/    the speech model         }
        settings/   your Gemini key          }
        fovea.log   what the last runs printed

    python launch.py --prepare      download what the first video would otherwise wait for, then stop
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = HERE.parent.parent                                    # Fovea/ when installed by the setup file
INSTALLED = HERE.parent.name == "app" and (HOME / "venv").is_dir()

if INSTALLED:                                                # keep what is yours outside the code folder
    os.environ.setdefault("FOVEA_DATA", str(HOME / "data"))
    os.environ.setdefault("FOVEA_WEIGHTS", str(HOME / "weights"))
    os.environ.setdefault("FOVEA_LOCAL", str(HOME / "settings"))
    if "--prepare" not in sys.argv and (sys.stdout is None or not sys.stdout.isatty()):   # started from an icon: no window to print in
        sys.stdout = sys.stderr = open(HOME / "fovea.log", "a", buffering=1, encoding="utf-8")

if INSTALLED or not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    try:                                                     # the copy of ffmpeg that comes with the libraries
        import static_ffmpeg                                 # (fetched the first time, by --prepare)

        static_ffmpeg.add_paths()
    except Exception as e:  # noqa: BLE001 - not installed, or no internet: use what the computer has
        print(f"The bundled ffmpeg is not available ({type(e).__name__}: {e})", flush=True)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def prepare() -> None:
    from faster_whisper import download_model

    from fovea.config import Settings

    settings = Settings()
    print(f"Getting the speech model ({settings.whisper_model}); this is the big download ...", flush=True)
    download_model(settings.whisper_model, cache_dir=str(settings.weights_dir))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is missing; setup did not finish")
    print("Everything Fovea needs is here.", flush=True)


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    else:
        import server

        server.main()
