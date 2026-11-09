"""Where job FILES live.

    LocalStorage  (MODE=local)  a folder on disk: ./data/users/<user_id>/<job_id>/...
    GCSStorage    (MODE=gcp)    a bucket:   gs://<BUCKET>/users/<user_id>/<job_id>/...

Both expose the same methods, so the API and worker never care which one is in use.
Inside a job folder the layout is always the same (see fovea/utils/jobdir.py).

Upload/download links work the same way for the frontend in both modes:
the API returns ``{"url", "method", "headers"}`` and the browser calls that URL.
  * GCS: a Google-signed URL straight to the bucket
  * local: a link back to this API (/v1/files/...) protected by an HMAC signature
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import re
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

from .settings import get_settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


def job_prefix(user_id: str, job_id: str) -> str:
    """users/<user_id>/<job_id>: one folder per user, one sub-folder per video."""
    if not _SAFE_ID.match(user_id) or not _SAFE_ID.match(job_id):
        raise ValueError("user_id / job_id may only contain letters, digits and _ . @ -")
    return f"users/{user_id}/{job_id}"


def safe_filename(name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:80] or "video"
    return f"{stem}{ext if re.fullmatch(r'\.[a-z0-9]{1,5}', ext) else '.mp4'}"


# =========================================================================== local disk
class LocalStorage:
    def __init__(self, root: Path, public_url: str, url_minutes: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_url = public_url
        self.url_minutes = url_minutes
        secret_file = self.root / "_secret"
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32))
        self._secret = secret_file.read_text().strip().encode()

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError("path escapes the data folder")
        return p

    # -- the worker runs directly inside the job folder, so syncing is a no-op
    def job_workdir(self, prefix: str) -> Path:
        return self.path(prefix)

    def download(self, prefix: str, entries: List[str], local_root: Path) -> None:
        pass

    def upload(self, local_root: Path, entries: List[str], prefix: str) -> List[str]:
        return []

    # -- small objects
    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def size(self, key: str) -> Optional[int]:
        p = self.path(key)
        return p.stat().st_size if p.is_file() else None

    def read_json(self, key: str) -> Any:
        return json.loads(self.path(key).read_text())

    def write_json(self, key: str, data: Any) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, default=str))

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self.path(prefix), ignore_errors=True)

    # -- links (served by the API's /v1/files route)
    def _sign(self, method: str, key: str, expires: int, max_bytes: Optional[int]) -> str:
        msg = f"{method}\n{key}\n{expires}\n{max_bytes or ''}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _link(self, method: str, key: str, max_bytes: Optional[int] = None) -> str:
        expires = int(time.time()) + self.url_minutes * 60
        q = {"expires": expires, "sig": self._sign(method, key, expires, max_bytes)}
        if max_bytes:
            q["max_bytes"] = max_bytes
        return f"{self.public_url}/v1/files/{quote(key)}?{urlencode(q)}"

    def verify(self, method: str, key: str, expires: int, sig: str, max_bytes: Optional[int]) -> bool:
        if expires < time.time():
            return False
        return hmac.compare_digest(sig, self._sign(method, key, expires, max_bytes))

    def upload_url(self, key: str, max_bytes: Optional[int]) -> Dict[str, Any]:
        return {"url": self._link("PUT", key, max_bytes), "method": "PUT", "headers": {}}

    def download_url(self, key: str) -> str:
        return self._link("GET", key)


# =========================================================================== Cloud Storage
class GCSStorage:
    def __init__(self, bucket_name: str, project: str, scratch_dir: Path, url_minutes: int):
        from google.cloud import storage

        self.client = storage.Client(project=project or None)
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.scratch_dir = Path(scratch_dir)
        self.url_minutes = url_minutes
        self._creds = None

    # -- the worker copies files to local scratch disk, runs the step, copies results back
    def job_workdir(self, prefix: str) -> Path:
        return self.scratch_dir / prefix

    def download(self, prefix: str, entries: List[str], local_root: Path, workers: int = 8) -> None:
        blobs = []
        for entry in entries:
            if entry.endswith("/"):
                blobs += [b for b in self.client.list_blobs(self.bucket_name, prefix=f"{prefix}/{entry}")
                          if not b.name.endswith("/")]
            else:
                b = self.bucket.blob(f"{prefix}/{entry}")
                if b.exists():
                    blobs.append(b)

        def _get(b):
            dest = Path(local_root) / b.name[len(prefix) + 1:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            b.download_to_filename(str(dest))

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(_get, blobs))

    def upload(self, local_root: Path, entries: List[str], prefix: str, workers: int = 8) -> List[str]:
        local_root = Path(local_root)
        files: List[Path] = []
        for entry in entries:
            p = local_root / entry
            if entry.endswith("/"):
                files += [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
            elif p.is_file():
                files.append(p)

        def _put(f: Path) -> str:
            rel = f.relative_to(local_root).as_posix()
            ctype = "application/json" if f.suffix == ".json" else mimetypes.guess_type(f.name)[0]
            self.bucket.blob(f"{prefix}/{rel}").upload_from_filename(str(f), content_type=ctype)
            return rel

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(_put, files))

    # -- small objects
    def exists(self, key: str) -> bool:
        return self.bucket.blob(key).exists()

    def size(self, key: str) -> Optional[int]:
        b = self.bucket.get_blob(key)
        return b.size if b is not None else None

    def read_json(self, key: str) -> Any:
        return json.loads(self.bucket.blob(key).download_as_text())

    def write_json(self, key: str, data: Any) -> None:
        self.bucket.blob(key).upload_from_string(json.dumps(data, indent=2, default=str),
                                                 content_type="application/json")

    def delete(self, key: str) -> None:
        b = self.bucket.blob(key)
        if b.exists():
            b.delete()

    def delete_prefix(self, prefix: str) -> None:
        for b in self.client.list_blobs(self.bucket_name, prefix=prefix + "/"):
            b.delete()

    # -- signed links
    def _signing(self) -> Dict[str, Any]:
        """Cloud Run has no private key file; sign via the IAM API with the service
        account's token (needs roles/iam.serviceAccountTokenCreator on itself)."""
        import google.auth
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if isinstance(self._creds, service_account.Credentials):
            return {"credentials": self._creds}
        if not self._creds.valid:
            self._creds.refresh(Request())
        return {"service_account_email": self._creds.service_account_email, "access_token": self._creds.token}

    def upload_url(self, key: str, max_bytes: Optional[int]) -> Dict[str, Any]:
        headers = {"x-goog-content-length-range": f"0,{max_bytes}"} if max_bytes else {}
        url = self.bucket.blob(key).generate_signed_url(
            version="v4", method="PUT", expiration=timedelta(minutes=self.url_minutes),
            headers=headers or None, **self._signing())
        return {"url": url, "method": "PUT", "headers": headers}

    def download_url(self, key: str) -> str:
        return self.bucket.blob(key).generate_signed_url(
            version="v4", method="GET", expiration=timedelta(minutes=self.url_minutes), **self._signing())


@lru_cache(maxsize=1)
def get_storage():
    s = get_settings()
    if s.is_local:
        return LocalStorage(s.data_dir, s.public_url, s.signed_url_minutes)
    return GCSStorage(s.bucket, s.project, s.scratch_dir, s.signed_url_minutes)
