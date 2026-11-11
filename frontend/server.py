"""Lightweight Fovea frontend server.

Serves the static UI on port 3000 (by default) and proxies JSON API calls (/api/v1/*)
to BACKEND_URL (/v1/*), attaching server-side credentials when needed:
  - Local mode:           no auth headers needed
  - API key mode:         sends X-API-Key: $BACKEND_API_KEY
  - Cloud Run IAM mode:   fetches a Google identity token for audience=BACKEND_URL

Works with FastAPI + Uvicorn when installed, and includes a built-in Python standard-library
ThreadingHTTPServer fallback so `python -m frontend` also runs with zero pip packages installed.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

STATIC_DIR = Path(__file__).resolve().parent / "static"


def get_backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _auth_headers() -> Dict[str, str]:
    """Build upstream auth headers for BACKEND_URL without exposing credentials to the browser."""
    backend_url = get_backend_url()
    api_key = os.environ.get("BACKEND_API_KEY") or os.environ.get("API_KEY") or ""
    auth_mode = os.environ.get("AUTH_MODE", "").lower()

    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
        return headers

    use_iam = auth_mode == "iam" or (
        auth_mode == "" and backend_url.startswith("https://") and ".run.app" in backend_url
    )
    if use_iam:
        try:
            from google.auth.transport.requests import Request as GoogleAuthRequest
            from google.oauth2 import id_token

            token = id_token.fetch_id_token(GoogleAuthRequest(), backend_url)
            headers["Authorization"] = f"Bearer {token}"
        except Exception:  # noqa: BLE001
            pass
    return headers


def _urlopen(req: urllib.request.Request, timeout: float = 25):
    """Open URL bypassing HTTP proxies for localhost/127.0.0.1."""
    host = urllib.parse.urlsplit(req.full_url).hostname or ""
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def check_upstream_health() -> Tuple[int, bytes]:
    backend_url = get_backend_url()
    req = urllib.request.Request(f"{backend_url}/healthz", headers=_auth_headers(), method="GET")
    try:
        with _urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            payload = {"ok": True, "backend_url": backend_url, "backend": data}
            return 200, json.dumps(payload).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "backend_url": backend_url, "error": str(exc)}
        return 502, json.dumps(payload).encode("utf-8")


def forward_v1_request(
    method: str, path: str, query_string: str, body: Optional[bytes], content_type: Optional[str]
) -> Tuple[int, str, bytes]:
    backend_url = get_backend_url()
    target_url = f"{backend_url}/v1/{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    headers = _auth_headers()
    if content_type:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
    try:
        with _urlopen(req, timeout=25) as resp:
            return (
                resp.status,
                resp.headers.get("Content-Type", "application/json"),
                resp.read(),
            )
    except urllib.error.HTTPError as http_err:
        return (
            http_err.code,
            http_err.headers.get("Content-Type", "application/json") if http_err.headers else "application/json",
            http_err.read(),
        )
    except Exception as exc:  # noqa: BLE001
        err_json = json.dumps({
            "detail": f"Could not connect to Fovea backend at {backend_url}. Is `python -m backend serve` running? ({exc})"
        }).encode("utf-8")
        return 502, "application/json", err_json


# ===========================================================================
# Optional FastAPI App (used when FastAPI + Uvicorn are installed)
# ===========================================================================
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Fovea Frontend", version="1.0.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/health")
    def frontend_health():
        status, payload = check_upstream_health()
        return Response(content=payload, status_code=status, media_type="application/json")

    @app.api_route("/api/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_v1(path: str, request: Request) -> Response:
        body = await request.body() if request.method in ("POST", "PUT") else None
        status, media_type, payload = forward_v1_request(
            method=request.method,
            path=path,
            query_string=str(request.url.query),
            body=body,
            content_type=request.headers.get("content-type"),
        )
        return Response(content=payload, status_code=status, media_type=media_type)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

except ImportError:
    app = None


# ===========================================================================
# Built-in Standard Library HTTP Server (Zero-Dependency Fallback)
# ===========================================================================
class _StdlibFrontendHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        pass

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method: str) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        route = parsed.path

        if route == "/api/health" and method == "GET":
            status, payload = check_upstream_health()
            self._send(status, "application/json", payload)
            return

        if route.startswith("/api/v1/"):
            subpath = route[len("/api/v1/"):]
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else None
            status, ctype, payload = forward_v1_request(
                method=method,
                path=subpath,
                query_string=parsed.query,
                body=body,
                content_type=self.headers.get("Content-Type"),
            )
            self._send(status, ctype, payload)
            return

        if method == "GET":
            if route == "/" or route == "/index.html":
                target = STATIC_DIR / "index.html"
            elif route.startswith("/static/"):
                rel = route[len("/static/"):]
                target = (STATIC_DIR / rel).resolve()
                if not str(target).startswith(str(STATIC_DIR.resolve())):
                    self._send(403, "text/plain", b"Forbidden")
                    return
            else:
                self._send(404, "text/plain", b"Not Found")
                return

            if target.is_file():
                mime, _ = mimetypes.guess_type(str(target))
                self._send(200, mime or "application/octet-stream", target.read_bytes())
            else:
                self._send(404, "text/plain", b"Not Found")
            return

        self._send(405, "text/plain", b"Method Not Allowed")

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")


def serve_stdlib(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _StdlibFrontendHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
