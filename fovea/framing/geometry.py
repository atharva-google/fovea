"""Output formats and crop geometry for ANY aspect ratio.

A *target* string can be:
  * a ratio:          "9:16", "1:1", "4:5", "16:9", "2.35:1"
  * a preset name:    "vertical", "square", "portrait", "landscape", ...
  * an exact size:    "1080x1920", "720x720"

How cropping works
------------------
* Target narrower than the source (e.g. 9:16 from 16:9): crop uses the full
  source height and the camera pans horizontally (this is what the smart
  framing drives).
* Target wider than or equal to the source (e.g. 21:9 from 16:9): crop uses
  the full width, centred vertically (no horizontal panning is possible).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

PRESETS = {
    "vertical": "9:16",
    "shorts": "9:16",
    "reels": "9:16",
    "tiktok": "9:16",
    "square": "1:1",
    "portrait": "4:5",
    "classic": "4:3",
    "landscape": "16:9",
    "cinema": "21:9",
}


def _even(x: float) -> int:
    v = int(round(x))
    return v if v % 2 == 0 else v + 1


@dataclass(frozen=True)
class OutputTarget:
    label: str                  # used in the output filename, e.g. "9x16"
    aspect: float               # width / height
    size: Optional[Tuple[int, int]] = None  # exact output (w, h) if requested

    @classmethod
    def parse(cls, spec: str) -> "OutputTarget":
        raw = spec.strip().lower()
        raw = PRESETS.get(raw, raw)

        m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", raw)  # exact pixel size
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid size: {spec}")
            return cls(label=f"{w}x{h}", aspect=w / h, size=(_even(w), _even(h)))

        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", raw)  # ratio
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if a <= 0 or b <= 0:
                raise ValueError(f"Invalid ratio: {spec}")
            label = f"{m.group(1)}x{m.group(2)}".replace(".", "_")
            return cls(label=label, aspect=a / b)

        raise ValueError(
            f"Can't parse target '{spec}'. Use a ratio like 9:16, a size like 1080x1920, "
            f"or a preset: {', '.join(PRESETS)}"
        )


@dataclass(frozen=True)
class CropGeometry:
    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    out_w: int
    out_h: int

    @classmethod
    def build(cls, src_w: int, src_h: int, target: OutputTarget, short_side: int = 1080) -> "CropGeometry":
        ar = target.aspect
        if ar <= src_w / src_h:
            crop_h = src_h
            crop_w = min(_even(src_h * ar), src_w - (src_w % 2))
        else:
            crop_w = src_w
            crop_h = min(_even(src_w / ar), src_h - (src_h % 2))

        if target.size:
            out_w, out_h = target.size
        elif ar < 1:
            out_w, out_h = _even(short_side), _even(short_side / ar)
        else:
            out_w, out_h = _even(short_side * ar), _even(short_side)

        return cls(src_w, src_h, crop_w, crop_h, out_w, out_h)

    # Horizontal limits for the crop *centre* so the window stays inside the frame
    @property
    def min_cx(self) -> float:
        return self.crop_w / 2.0

    @property
    def max_cx(self) -> float:
        return self.src_w - self.crop_w / 2.0

    def window(self, cx: float) -> Tuple[int, int, int, int]:
        """Return the (x1, y1, x2, y2) crop window centred horizontally at ``cx``."""
        half = self.crop_w // 2
        x1 = max(0, int(round(cx)) - half)
        x2 = x1 + self.crop_w
        if x2 > self.src_w:
            x2 = self.src_w
            x1 = self.src_w - self.crop_w
        y1 = (self.src_h - self.crop_h) // 2
        return x1, y1, x2, y1 + self.crop_h
