"""Progress callbacks for long steps (shown in the job record / UI).

A step calls ``progress(stage, done=..., total=..., unit=..., index=..., count=..., detail=...)``:

    progress("tracking", done=120, total=900, unit="frames", index=1, count=4)
    progress("speakers (TalkNet)", ..., detail="scoring 37 face tracks on the GPU")

``index``/``count`` place the stage within the step (stage 1 of 4). Callers may report from
several threads; ``Counter`` does the thread-safe counting. The backend throttles writes, so
reporting often is fine.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

Progress = Callable[..., None]


def noop(*_args, **_kwargs) -> None:
    pass


class Counter:
    """Thread-safe ``done`` counter for one stage that reports every ``add``."""

    def __init__(self, progress: Optional[Progress], stage: str, total: int, unit: str,
                 index: Optional[int] = None, count: Optional[int] = None):
        self.progress = progress or noop
        self.stage, self.total, self.unit, self.index, self.count = stage, int(total), unit, index, count
        self.done = 0
        self.detail: Optional[str] = None
        self._lock = threading.Lock()
        self._report()

    def add(self, n: int = 1) -> None:
        with self._lock:
            self.done += n
            self._report()

    def set_detail(self, detail: Optional[str]) -> None:
        """Free-text status shown next to the counter (e.g. what the stage is doing right now)."""
        with self._lock:
            self.detail = detail
            self._report()

    def _report(self) -> None:
        self.progress(self.stage, done=min(self.done, self.total), total=self.total, unit=self.unit,
                      index=self.index, count=self.count, detail=self.detail)
