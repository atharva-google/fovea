"""Fovea's local web app: one small server on this computer that serves the page and runs the steps.

    python frontend/server.py            # starts it and opens the page in the browser

It only uses Python's standard library and calls the ``fovea`` package as it is. Everything stays on
this computer: projects are the folders in ``fovea_data/``, and what the page adds to a project
(unchecked shots, focus changes, formats) is kept in ``fovea_data/<video name>/frontend.json``.

This one program is all of Fovea that runs: it serves the page AND does the work. It stops by itself
when the page has been closed, or nobody has touched it, for the idle time set in Settings (never
during a job). After that nothing of Fovea is running; an open tab is only a picture of the page.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fovea import analysis, framing, media  # noqa: E402
from fovea import edit as editor  # noqa: E402
from fovea import render as renderer  # noqa: E402
from fovea.config import THINKING_LEVELS, Settings  # noqa: E402
from fovea.job import Job, read_json, write_json  # noqa: E402
from fovea.units import UNITS_VERSION, slots  # noqa: E402

PORT = int(os.environ.get("FOVEA_PORT", "8765"))
IDLE_MINUTES = float(os.environ.get("FOVEA_IDLE_MINUTES", "10"))     # until the user picks another in Settings
DATA = Settings().data_dir
LOCAL = Path(os.environ.get("FOVEA_LOCAL", HERE / ".local"))   # this computer's settings (the Gemini key); not in git
STATIC = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/sw.js": "sw.js",
          "/icon.png": "icon.png"}

JOBS: Dict[str, Dict[str, Any]] = {}         # project -> the job running (or last run) for it
LOCK = threading.Lock()
last_seen = time.time()                      # when the page last asked for anything


class Problem(Exception):
    """Something to tell the page, with an HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# =========================================================================== this computer's settings
def local_settings() -> Dict[str, Any]:
    try:
        stored = read_json(LOCAL / "settings.json")
    except (OSError, ValueError):
        stored = {}
    try:
        idle = min(240.0, max(0.1, float(stored["idle_minutes"])))
    except (KeyError, TypeError, ValueError):
        idle = IDLE_MINUTES
    return {"gemini_key": str(stored.get("gemini_key") or ""), "thinking": stored.get("thinking") or "low",
            "idle_minutes": idle}


def gemini_key() -> Optional[str]:
    return local_settings()["gemini_key"] or os.environ.get("GEMINI_API_KEY")


# =========================================================================== projects
def project(pid: str) -> Job:
    root = DATA / pid
    if not pid or Path(pid).name != pid or pid.startswith(".") or not (root / "job.json").is_file():
        raise Problem(404, f"No project called {pid!r}")
    return Job(root)


def page_state(job: Job) -> Dict[str, Any]:
    """What the page added to a project."""
    try:
        s = read_json(job.root / "frontend.json")
    except (OSError, ValueError):
        s = {}
    return {"skipped": list(s.get("skipped") or []), "focus": dict(s.get("focus") or {}),
            "formats": list(s.get("formats") or renderer.DEFAULT_FORMATS)}


def framed(stored: Dict[str, Any], state: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """The focus line for one format: Gemini's, with the shots the user moved held where they put them."""
    moved = {**state["focus"].get("*", {}), **state["focus"].get(fmt, {})}
    start = {v.id: v.start for v in stored["video"]}
    return {"shots": [{"id": s["id"], "points": [{"t": start[s["id"]], "move": "cut", "x": moved[s["id"]], "slot": ""}]
                       if s["id"] in moved else s["points"]} for s in stored["framing"]["shots"]]}


def crop_width(info: Dict[str, Any], fmt: str) -> Optional[int]:
    """How wide the crop is for a format, or None when the format is as wide as the video."""
    height = int(info["height"]) // 2 * 2
    width = int(round(height * renderer.parse_format(fmt)[1])) // 2 * 2
    return width if width <= int(info["width"]) - 2 else None


def stretches(spans: List[Tuple[float, float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for a, b in spans:
        if out and abs(out[-1][1] - a) < 1e-6:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def summary_of(job: Job) -> Dict[str, Any]:
    try:
        info = read_json(job.root / "prep" / "video_info.json")
    except (OSError, ValueError):
        info = None
    running = JOBS.get(job.root.name, {})
    return {"id": job.root.name, "info": info, "ready": (job.root / "prep" / "log.json").is_file(),
            "busy": running.get("status") == "running"}


def describe(job: Job) -> Dict[str, Any]:
    """Everything the page shows about one project."""
    out = {**summary_of(job), "name": job.source.name, "state": page_state(job), "job": JOBS.get(job.root.name),
           "has_video": job.source.is_file(), "has_proxy": (job.root / "prep" / "proxy.mp4").is_file(),
           "outputs": sorted(({"name": f.name, "size": f.stat().st_size, "time": f.stat().st_mtime}
                              for f in (job.root / "outputs").glob("*.mp4")), key=lambda f: -f["time"])
           if (job.root / "outputs").is_dir() else []}
    stored = analysis.load(job) if out["info"] else None
    out["ready"] = bool(stored)
    if not stored:
        return out
    info, state, log = out["info"], out["state"], stored["log"]
    seen = {u["id"]: u for u in log["video_units"]}
    heard = {u["id"]: u for u in log["audio_units"]}
    kinds = {p["id"]: p["kind"] for u in log["audio_units"] for p in u["parts"]}
    out["summary"], out["cast"] = log["summary"], log["cast"]
    out["shots"] = [{**v.to_dict(), **seen[v.id]} for v in stored["video"]]
    out["audio"] = [{"id": a.id, "start": a.start, "end": a.end, "shots": a.over, "heard": heard[a.id]["heard"],
                     "kind": heard[a.id]["kind"], "speaker": heard[a.id]["speaker"]} for a in stored["audio"]]
    out["slots"] = [{**s, "kind": kinds.get(s["id"], "other")} for s in slots(stored["audio"], stored["video"])]
    out["tracks"] = {}
    for fmt in state["formats"]:
        crop_w = crop_width(info, fmt)
        if crop_w is None:
            out["tracks"][fmt] = None
            continue
        shots = [{"start": v.start, "end": v.end, "points": s["points"]}
                 for v, s in zip(stored["video"], framed(stored, state, fmt)["shots"])]
        width = int(info["width"])
        out["tracks"][fmt] = {"w": crop_w / width, "pieces": [[a, b, x0 / width, x1 / width] for a, b, x0, x1 in
                                                              framing.crop_track(shots, width, crop_w, Settings().dead_zone)]}
    out["edits"] = []
    for f in sorted((job.root / "edit").glob("edit_*.json")) if (job.root / "edit").is_dir() else []:
        try:
            e = read_json(f)
            out["edits"].append({"name": f.stem[len("edit_"):], **{k: e[k] for k in
                                 ("target", "seconds", "story", "must_keep", "keep", "segments", "issues")}})
        except (OSError, ValueError, KeyError):
            pass
    return out


# =========================================================================== jobs (run in the background)
def start(pid: str, kind: str, work: Callable[[Callable[[str], None]], None]) -> None:
    with LOCK:
        if JOBS.get(pid, {}).get("status") == "running":
            raise Problem(409, "This project is still busy; wait for it to finish")
        job = JOBS[pid] = {"kind": kind, "status": "running", "lines": [], "error": ""}

    def run() -> None:
        try:
            work(lambda message: job["lines"].append(str(message)))
            job["status"] = "done"
        except BaseException as e:  # noqa: BLE001 - whatever went wrong is shown on the page
            traceback.print_exc()
            job["error"] = str(e) or type(e).__name__
            job["status"] = "error"

    threading.Thread(target=run, daemon=True).start()


def run_log(job: Job, refresh: bool) -> Callable[[Callable[[str], None]], None]:
    def work(log: Callable[[str], None]) -> None:
        from fovea import pipeline

        try:
            pipeline.analyze(job.source, Settings(thinking=local_settings()["thinking"], refresh=refresh),
                             gemini_key(), log)
        except analysis.KeyRequired:
            raise RuntimeError("Fovea needs a Gemini key to watch the video.") from None
    return work


def run_render(job: Job, edit_name: Optional[str]) -> Callable[[Callable[[str], None]], None]:
    def work(log: Callable[[str], None]) -> None:
        stored, state, settings = analysis.load(job), page_state(job), Settings()
        if not stored:
            raise RuntimeError("This video has no log yet")
        info = read_json(job.video_info)
        if edit_name:
            segments, tag = read_json(job.edit_json(edit_name))["segments"], f"_{edit_name}"
        else:
            kept = [(v.start, v.end) for v in stored["video"] if v.id not in state["skipped"]]
            segments, tag = (None, "") if len(kept) == len(stored["video"]) else (stretches(kept), "_trim")
        if segments is not None and not segments:
            raise RuntimeError("Nothing is selected, so there is nothing to render")
        groups: Dict[str, List[str]] = {}          # formats that share one focus line are made in one ffmpeg run
        for fmt in state["formats"]:
            groups.setdefault(json.dumps(framed(stored, state, fmt)), []).append(fmt)
        for line, formats in groups.items():
            log(f"▶ render ({', '.join(formats)})")
            renderer.render(job, stored["video"], json.loads(line), info, formats, settings.dead_zone, settings.crf,
                            log, segments=segments, tag=tag)
    return work


def run_edit(job: Job, seconds: float) -> Callable[[Callable[[str], None]], None]:
    def work(log: Callable[[str], None]) -> None:
        from fovea import gemini

        stored, state, settings = analysis.load(job), page_state(job), Settings()
        if not stored:
            raise RuntimeError("This video has no log yet")
        if not gemini_key():
            raise RuntimeError("A Gemini key is needed: add it in Settings")
        # Unchecked shots are not shown to Gemini; neither is a sound that runs into one.
        skipped = set(state["skipped"])
        video = [v for v in stored["video"] if v.id not in skipped]
        audio = [a for a in stored["audio"] if a.over and not skipped & set(a.over)]
        if not video or not audio:
            raise RuntimeError("Nothing is selected, so there is nothing to edit")
        ids = {u.id for u in [*video, *audio]}
        described = {**stored["log"], "video_units": [u for u in stored["log"]["video_units"] if u["id"] in ids],
                     "audio_units": [u for u in stored["log"]["audio_units"] if u["id"] in ids]}
        log(f"▶ edit (synced, about {seconds:g}s{f', without {len(skipped)} unchecked shots' if skipped else ''})")
        raw, took = gemini.ask(editor.build_prompt(audio, video, described, seconds), gemini_key(),
                               settings.gemini_model, settings.gemini_api_version, schema="Edit", log=log)
        edit, issues = editor.clean(raw, audio, described)
        if not edit["keep"]:
            raise RuntimeError("Gemini's edit keeps nothing; try again")
        write_json(job.edit_json(editor.name(seconds)), {
            "fingerprint": job.fingerprint, "version": [UNITS_VERSION, analysis.ANALYSIS_VERSION, editor.EDIT_VERSION],
            "target": seconds, "model": settings.gemini_model, "thinking": settings.thinking, "took": took,
            "issues": issues, "skipped": sorted(skipped), **edit})
        log(f"  {edit['seconds']:.1f}s in {len(edit['segments'])} stretches")
    return work


# =========================================================================== the server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:      # quiet: nothing about requests (or keys) is printed
        pass

    def do_GET(self) -> None:
        self.handle_request("GET")

    def do_POST(self) -> None:
        self.handle_request("POST")

    def do_DELETE(self) -> None:
        self.handle_request("DELETE")

    def handle_request(self, method: str) -> None:
        global last_seen
        last_seen = time.time()
        try:
            # Only this computer's own page may talk to the server: other websites cannot set this
            # header, and a foreign host name means the request was not meant for us.
            if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                raise Problem(403, "Not allowed")
            if method != "GET" and self.headers.get("X-Fovea") != "1":
                raise Problem(403, "Not allowed")
            url = urlparse(self.path)
            self.route(method, unquote(url.path), {k: v[0] for k, v in parse_qs(url.query).items()})
        except Problem as e:
            self.send_json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route(self, method: str, path: str, query: Dict[str, str]) -> None:
        if method == "GET" and path in STATIC:
            return self.send_file(HERE / STATIC[path], cache=False)
        if (method, path) == ("GET", "/api/ping"):
            return self.send_json({"fovea": True})
        if (method, path) == ("GET", "/api/state"):
            return self.send_json(self.app_state())
        if (method, path) == ("POST", "/api/settings"):
            body, current = self.read_json(), local_settings()
            if "gemini_key" in body:
                current["gemini_key"] = str(body["gemini_key"] or "").strip()
            if body.get("thinking") in THINKING_LEVELS:
                current["thinking"] = body["thinking"]
            if "idle_minutes" in body:
                try:
                    current["idle_minutes"] = min(240.0, max(1.0, float(body["idle_minutes"])))
                except (TypeError, ValueError):
                    raise Problem(400, "Give the idle time in minutes") from None
            LOCAL.mkdir(parents=True, exist_ok=True)
            write_json(LOCAL / "settings.json", current)
            return self.send_json(self.app_state())
        if (method, path) == ("POST", "/api/projects"):
            return self.upload(query.get("name", ""))

        m = re.fullmatch(r"/api/projects/([^/]+)(?:/(.+))?", path)
        if not m:
            raise Problem(404, "Not found")
        job, rest = project(m[1]), m[2] or ""
        pid = job.root.name
        if method == "GET" and rest == "":
            return self.send_json(describe(job))
        if method == "DELETE" and rest == "":
            if JOBS.get(pid, {}).get("status") == "running":
                raise Problem(409, "This project is still busy; wait for it to finish")
            shutil.rmtree(job.root)
            JOBS.pop(pid, None)
            return self.send_json(self.app_state())
        if method == "GET" and rest == "job":
            return self.send_json(JOBS.get(pid) or {"status": "none", "lines": [], "error": ""})
        if method == "GET" and rest == "video":
            # the file itself, so it plays at once; the prepared copy when the browser cannot play that
            return self.send_file(job.root / "prep" / "proxy.mp4" if "proxy" in query else job.source)
        if method == "GET" and (t := re.fullmatch(r"thumb/(V\d+)\.jpg", rest)):
            return self.send_file(self.thumb(job, t[1]))
        if method == "GET" and (o := re.fullmatch(r"outputs/([^/]+\.mp4)", rest)):
            return self.send_file(job.root / "outputs" / o[1], download="download" in query)
        if method == "POST" and rest == "state":
            body, state = self.read_json(), page_state(job)
            try:
                if "formats" in body:
                    state["formats"] = list(dict.fromkeys(f"{w:g}:{h:g}" for w, h in (
                        (float(x) for x in renderer.parse_format(str(f))[0].split("x")) for f in body["formats"])))
                if "skipped" in body:
                    state["skipped"] = sorted({str(s) for s in body["skipped"]})
                if "focus" in body:
                    state["focus"] = {str(fmt): {str(shot): max(0, min(100, round(float(x)))) for shot, x in moved.items()}
                                      for fmt, moved in body["focus"].items() if moved}
            except (ValueError, TypeError, AttributeError) as e:
                raise Problem(400, str(e) or "That could not be saved") from None
            write_json(job.root / "frontend.json", state)
            return self.send_json(describe(job))
        if method == "POST" and rest == "log":
            start(pid, "log", run_log(job, bool(self.read_json().get("refresh"))))
            return self.send_json(describe(job))
        if method == "POST" and rest == "render":
            name = self.read_json().get("edit")
            if name and not job.edit_json(str(name)).is_file():
                raise Problem(404, "No such edit")
            start(pid, "render", run_render(job, str(name) if name else None))
            return self.send_json(describe(job))
        if method == "POST" and rest == "edit":
            try:
                seconds = float(self.read_json().get("seconds"))
                if not 1 <= seconds <= 600:
                    raise ValueError
            except (TypeError, ValueError):
                raise Problem(400, "Give a length in seconds") from None
            start(pid, "edit", run_edit(job, seconds))
            return self.send_json(describe(job))
        if method == "POST" and (e := re.fullmatch(r"edits/([\w.]+)", rest)):
            return self.keep(job, e[1], self.read_json().get("keep"))
        raise Problem(404, "Not found")

    # ---------------------------------------------------------------- actions
    def app_state(self) -> Dict[str, Any]:
        folders = [d for d in DATA.iterdir() if (d / "job.json").is_file()] if DATA.is_dir() else []
        folders.sort(key=lambda d: -(d / "job.json").stat().st_mtime)
        settings = local_settings()
        return {"projects": [summary_of(Job(d)) for d in folders],
                "settings": {"has_key": bool(gemini_key()), "thinking": settings["thinking"],
                             "idle_minutes": settings["idle_minutes"],
                             "thinking_levels": list(THINKING_LEVELS)}}

    def upload(self, name: str) -> None:
        """A new project. It exists (and its video plays) as soon as the file has arrived; Fovea starts
        watching it at once when there is a Gemini key."""
        name = Path(name).name
        if not Path(name).stem or name.startswith("."):
            raise Problem(400, "The file needs a name")
        pid = Path(name).stem
        if JOBS.get(pid, {}).get("status") == "running":
            raise Problem(409, "A project with this name is still busy")
        arrived = DATA / ".uploads" / name
        arrived.parent.mkdir(parents=True, exist_ok=True)
        left = int(self.headers.get("Content-Length") or 0)
        with open(arrived, "wb") as f:
            while left > 0:
                block = self.rfile.read(min(left, 1 << 20))
                if not block:
                    break
                f.write(block)
                left -= len(block)
        try:
            if left or not arrived.stat().st_size:
                raise Problem(400, "The upload did not arrive completely")
            job = Job.open(arrived, DATA)
        finally:
            arrived.unlink(missing_ok=True)
        JOBS.pop(pid, None)
        try:
            write_json(job.video_info, media.probe(job.source))
        except Exception as e:  # noqa: BLE001 - the page says why when the video is watched
            print(f"Could not read {name}: {e}")
        if gemini_key():
            start(pid, "log", run_log(job, refresh=False))
        self.send_json({"id": pid})

    def keep(self, job: Job, name: str, keep: Any) -> None:
        """The user ticked units in or out of an edit: what they say goes."""
        stored = analysis.load(job)
        if not stored or not job.edit_json(name).is_file():
            raise Problem(404, "No such edit")
        edit = read_json(job.edit_json(name))
        wanted = {str(k) for k in keep or []}
        units = [a for a in stored["audio"] if a.id in wanted]
        edit["keep"] = [a.id for a in units]
        edit["segments"] = stretches([(a.start, a.end) for a in units])
        edit["seconds"] = round(sum(b - a for a, b in edit["segments"]), 3)
        write_json(job.edit_json(name), edit)
        self.send_json(describe(job))

    def thumb(self, job: Job, shot: str) -> Path:
        """A small picture from the middle of a shot, made once."""
        path = job.root / "frontend_thumbs" / f"{shot}.jpg"
        if not path.is_file():
            stored = analysis.load(job)
            unit = next((v for v in stored["video"] if v.id == shot), None) if stored else None
            if unit is None or shutil.which("ffmpeg") is None:
                raise Problem(404, "No picture")
            path.parent.mkdir(exist_ok=True)
            proxy = job.root / "prep" / "proxy.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{(unit.start + unit.end) / 2:.3f}", "-i",
                            str(proxy if proxy.is_file() else job.source), "-frames:v", "1", "-vf", "scale=240:-2", str(path)],
                           capture_output=True)
        return path

    # ---------------------------------------------------------------- plumbing
    def read_json(self) -> Dict[str, Any]:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise Problem(400, "That was not JSON") from None
        return body if isinstance(body, dict) else {}

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, download: bool = False, cache: bool = True) -> None:
        """A file, or the part of it the browser asks for (which is how a video is seeked)."""
        if not path.is_file():
            raise Problem(404, "Not found")
        size = path.stat().st_size
        first, last, status = 0, size - 1, 200
        asked = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if asked and (asked[1] or asked[2]):
            if asked[1]:
                first, last = int(asked[1]), min(int(asked[2]), size - 1) if asked[2] else size - 1
            else:
                first = max(0, size - int(asked[2]))
            if first > last:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(first)
            left = last - first + 1
            while left > 0:
                block = f.read(min(left, 1 << 18))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)


def sleep_when_idle(server: ThreadingHTTPServer) -> None:
    """Stop when the page has not asked for anything for a while, unless a job is running."""
    while True:
        time.sleep(5)
        busy = any(j["status"] == "running" for j in list(JOBS.values()))
        if busy:
            global last_seen
            last_seen = time.time()
        elif time.time() - last_seen > local_settings()["idle_minutes"] * 60:
            print("Fovea has been idle: going to sleep. Start it again to continue.")
            server.shutdown()
            return


def main() -> None:
    url = f"http://127.0.0.1:{PORT}/"
    show = "--no-browser" not in sys.argv
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        try:                                         # is that Fovea, already awake?
            with urllib.request.urlopen(f"{url}api/ping", timeout=3) as r:
                awake = json.load(r).get("fovea")
        except Exception:  # noqa: BLE001
            awake = False
        if not awake:
            raise SystemExit(f"Port {PORT} is used by another program. Set FOVEA_PORT to a free one.") from None
        if show:
            webbrowser.open(url)
        return
    threading.Thread(target=sleep_when_idle, args=(server,), daemon=True).start()
    print(f"Fovea is awake at {url} (it goes to sleep after {local_settings()['idle_minutes']:g} idle minutes)")
    if show:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
