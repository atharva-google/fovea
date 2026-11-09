"""Tiny thread-safe stopwatch for "where did the time go" logs.

    t = Timings()
    with t("yolo"):
        ...
    t.add("faces", 0.3)
    log(t.summary())     # "yolo 12.3s | faces 0.3s"
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, TypeVar

T = TypeVar("T")


class Timings:
    def __init__(self) -> None:
        self._t: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self._t[name] += seconds

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def timed_iter(self, name: str, it: Iterable[T]) -> Iterator[T]:
        """Yield from ``it``, adding the time spent WAITING for each item to ``name``."""
        it = iter(it)
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    return
                finally:
                    self.add(name, time.perf_counter() - t0)
                yield item
        finally:
            close = getattr(it, "close", None)
            if close:
                close()

    def as_dict(self) -> Dict[str, float]:
        with self._lock:
            return {k: round(v, 2) for k, v in self._t.items()}

    def summary(self) -> str:
        return " | ".join(f"{k} {v:.1f}s" for k, v in self.as_dict().items())
