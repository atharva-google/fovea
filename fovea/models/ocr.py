"""On-screen text detection with EasyOCR (a few sampled frames per shot)."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

OCR_COLUMNS = ["text", "shot_id", "start_s", "end_s", "duration_s", "confidence", "text_box_vertices"]


def empty_ocr_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OCR_COLUMNS)


def sample_frame_indices(start_f: int, end_f: int, samples: int = 1) -> List[int]:
    if samples == 1:
        return [(start_f + end_f) // 2]
    step = (end_f - start_f) / (samples + 1)
    return [int(start_f + step * (i + 1)) for i in range(samples)]


class TextReader:
    def __init__(self, model_dir: str, device: str, langs: List[str] = ("en",), min_conf: float = 0.35):
        import easyocr

        self.reader = easyocr.Reader(
            lang_list=list(langs), gpu=device != "cpu", model_storage_directory=str(model_dir),
            download_enabled=True, verbose=False,
        )
        self.min_conf = min_conf

    def read_shot(self, shot: Dict[str, Any], frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """OCR the sampled frames of one shot; returns de-duplicated text records."""
        s_start_s, s_end_s = float(shot["start_sec"]), float(shot["end_sec"])
        shot_texts: Dict[str, Dict[str, Any]] = {}

        for frame in frames:
            height, width = frame.shape[:2]
            for item in self.reader.readtext(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), paragraph=True):
                if len(item) == 3:
                    bbox, text, conf = item
                    conf = float(conf)
                else:
                    bbox, text = item
                    conf = 1.0

                clean = text.strip()
                if len(clean) < 2 or conf < self.min_conf:
                    continue

                # Normalised (0-1) polygon vertices
                vertices = [(round(float(pt[0]) / width, 4), round(float(pt[1]) / height, 4)) for pt in bbox]
                if clean not in shot_texts:
                    shot_texts[clean] = {
                        "text": clean, "shot_id": shot["shot_id"], "start_s": s_start_s, "end_s": s_end_s,
                        "duration_s": round(s_end_s - s_start_s, 3), "confidence": conf,
                        "text_box_vertices": vertices,
                    }
                else:
                    shot_texts[clean]["confidence"] = max(shot_texts[clean]["confidence"], conf)
        return list(shot_texts.values())


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return empty_ocr_df()
    return pd.DataFrame(records).sort_values(by="start_s").reset_index(drop=True)
