"""Bring this copy of Fovea up to date. Used by the Update button (server.py).

Two kinds of copy:
    installed by the setup file   the code in Fovea/app is replaced by the latest from GitHub; the
                                  previous code is kept in Fovea/app.old; projects, the speech model
                                  and the key live outside app/ and are not touched
    a git checkout                ``git pull``
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, List

REPO = "atharva-google/fovea"
BRANCH = os.environ.get("FOVEA_BRANCH", "feat/frontend")       # change to "main" once the web app is merged

APP = Path(__file__).resolve().parent.parent                   # the code
HOME = APP.parent                                              # Fovea/ when installed by the setup file
INSTALLED = APP.name == "app" and (HOME / "venv").is_dir()


def version() -> str:
    """A short name for the code that is running, or "" when it is not known."""
    try:
        if INSTALLED:
            return (HOME / "version").read_text().strip()[:7]
        return _git("rev-parse", "--short", "HEAD")
    except (OSError, RuntimeError):
        return ""


def update(log: Callable[[str], None]) -> bool:
    """Get the latest code. True when something changed (Fovea must then be started again)."""
    return _update_installed(log) if INSTALLED else _update_checkout(log)


# =========================================================================== a git checkout
def _git(*args: str) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=APP, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except OSError as e:
        raise RuntimeError(f"git could not be run: {e}") from None
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout).strip().splitlines()[-1] if (res.stderr or res.stdout).strip()
                           else "git did not work")
    return res.stdout.strip()


def _update_checkout(log: Callable[[str], None]) -> bool:
    if not (APP / ".git").exists():
        raise RuntimeError("This copy of Fovea was not installed by the setup file, so it cannot update itself")
    before = _git("rev-parse", "HEAD")
    log("Getting the latest code (git pull) ...")
    _git("pull", "--ff-only")
    after = _git("rev-parse", "HEAD")
    log("Already up to date." if after == before else f"Updated to {after[:7]}.")
    return after != before


# =========================================================================== installed by the setup file
def _fetch(url: str, timeout: float = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fovea-updater"}), timeout=timeout)


def _run(cmd: List[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if res.returncode != 0:
        raise RuntimeError(f"{what} did not work: {(res.stderr or res.stdout).strip()[-400:]}")


def _update_installed(log: Callable[[str], None]) -> bool:
    try:                                        # which version is the latest? (skipped when GitHub will not say)
        with _fetch(f"https://api.github.com/repos/{REPO}/commits/{BRANCH}", timeout=15) as r:
            latest = str(json.load(r)["sha"])
    except Exception:  # noqa: BLE001
        latest = ""
    try:
        current = (HOME / "version").read_text().strip()
    except OSError:
        current = ""
    if latest and latest == current:
        log("Already up to date.")
        return False

    python = HOME / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    new, old = HOME / "app.new", HOME / "app.old"
    log("Downloading the latest Fovea ...")
    with tempfile.TemporaryDirectory() as tmp:
        with _fetch(f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.tar.gz") as r, \
                open(Path(tmp) / "fovea.tar.gz", "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(Path(tmp) / "fovea.tar.gz") as archive:
            archive.extractall(Path(tmp) / "code", filter="data")
        (top,) = list((Path(tmp) / "code").iterdir())
        shutil.rmtree(new, ignore_errors=True)
        shutil.move(str(top), str(new))

    try:
        wanted, have = (new / "requirements.txt").read_text(), (APP / "requirements.txt").read_text()
        if wanted != have:                      # only when the libraries it needs have changed
            log("Updating the libraries ...")
            uv = HOME / "bin" / ("uv.exe" if os.name == "nt" else "uv")
            _run([str(uv), "pip", "install", "--python", str(python), "-r", str(new / "requirements.txt"),
                  "static-ffmpeg"], "Updating the libraries")
        log("Checking the new version ...")     # it must at least start, before it replaces the one that works
        _run([str(python), "-c", "import sys; sys.path[:0] = sys.argv[1:]; import server",
              str(new / "frontend"), str(new)], "The new version")
    except Exception:
        shutil.rmtree(new, ignore_errors=True)
        raise

    shutil.rmtree(old, ignore_errors=True)
    APP.rename(old)                             # the version that worked is kept, in app.old
    new.rename(APP)
    if latest:
        (HOME / "version").write_text(latest)
    log(f"Updated{f' to {latest[:7]}' if latest else ''}.")
    return True
