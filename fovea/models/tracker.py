"""Person tracking with face boxes.

Speed-ups compared to the original notebook:
  * frames are sent to YOLO in **batches** (one GPU call per N frames, FP16 on CUDA)
  * the detector only runs on every ``detect_every``-th frame; boxes for the frames
    in between are linearly interpolated per track
  * optional ``face_backend="pose"``: a YOLO-pose model gives person boxes AND head
    keypoints in one GPU pass, replacing the per-person CPU MediaPipe call

Output structure (``ShotTracks``)::

    {shot_id: {frame_idx: [ {track_id, bbox, conf, cname, face_bbox, face_conf} ]}}

Persons without a detectable face are dropped (same as the original notebook).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ShotTracks = Dict[int, Dict[int, List[Dict[str, Any]]]]
FrameTracks = Dict[int, List[Dict[str, Any]]]


def _make_bytetrack(frame_rate: int):
    """Build Ultralytics' ByteTrack directly so we can feed it batched detections."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    path = check_yaml("bytetrack.yaml")
    try:  # newer ultralytics
        from ultralytics.utils import YAML

        cfg = YAML.load(path)
    except ImportError:  # older ultralytics
        from ultralytics.utils import yaml_load

        cfg = yaml_load(path)
    return BYTETracker(args=IterableSimpleNamespace(**cfg), frame_rate=frame_rate)


class PersonTracker:
    def __init__(
        self,
        weights_path: str,
        device: str,
        fps: float,
        face_backend: str = "mediapipe",
        face_model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.5,
        face_min_conf: float = 0.5,
        detect_every: int = 1,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)  # detection model, or pose model for face_backend="pose"
        self.device = device
        self.half = device == "cuda"
        self.conf, self.iou = conf, iou
        self.face_backend = face_backend
        self.frame_rate = max(1, int(round(fps / max(1, detect_every))))
        self.tracker = None

        self.face_detector = None
        if face_backend == "mediapipe":
            from mediapipe.tasks.python import BaseOptions, vision

            self.face_detector = vision.FaceDetector.create_from_options(
                vision.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(face_model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=face_min_conf,
                )
            )

    def new_shot(self) -> None:
        """Reset tracker state so IDs don't leak across cuts."""
        self.tracker = _make_bytetrack(self.frame_rate)

    def process_batch(self, batch: Sequence[Tuple[int, np.ndarray]]) -> FrameTracks:
        """Detect + track a batch of consecutive (detected) frames of the current shot."""
        if not batch:
            return {}
        results = self.model.predict(
            source=[img for _, img in batch], conf=self.conf, iou=self.iou,
            half=self.half, device=self.device, verbose=False,
        )

        out: FrameTracks = {}
        for (frame_idx, img), res in zip(batch, results):
            det = res.boxes.cpu().numpy()
            current: List[Dict[str, Any]] = []
            if len(det):
                tracks = self.tracker.update(det, img)
                kpts = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else None
                for t in tracks:
                    # tracks rows: x1, y1, x2, y2, track_id, score, cls, det_index
                    box = [int(v) for v in t[:4]]
                    tid, score, cls_id, det_idx = int(t[4]), float(t[5]), int(t[6]), int(t[7])
                    cname = res.names[cls_id]
                    if cname == "person":
                        d = self._person(img, tid, box, score, kpts[det_idx] if kpts is not None else None)
                        if d is not None:
                            current.append(d)
                    else:
                        current.append({"track_id": tid, "bbox": box, "conf": score, "cname": cname})
            out[frame_idx] = current
        return out

    # ------------------------------------------------------------- faces
    def _person(self, img, tid, box, conf, kpts) -> Optional[Dict[str, Any]]:
        h, w = img.shape[:2]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w, box[2]), min(h, box[3])
        if x2 <= x1 or y2 <= y1:
            return None
        box = [x1, y1, x2, y2]

        if self.face_backend == "pose":
            face = _face_from_keypoints(kpts, box, w, h) if kpts is not None else None
        else:
            face = _face_from_mediapipe(img, box, self.face_detector)
        if face is None:
            return None
        face_bbox, face_conf = face
        return {"track_id": tid, "bbox": box, "conf": conf, "cname": "person",
                "face_bbox": face_bbox, "face_conf": face_conf}


def _face_from_mediapipe(frame, box, detector) -> Optional[Tuple[List[int], float]]:
    import mediapipe as mp

    px1, py1, px2, py2 = box
    crop = frame[py1:py2, px1:px2]
    h, w = crop.shape[:2]
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    if not res.detections:
        return None
    # Face closest to where a head usually is (top-center of the body box)
    tx, ty = w * 0.5, h * 0.2
    best = min(res.detections, key=lambda d: (
        (d.bounding_box.origin_x + d.bounding_box.width * 0.5 - tx) ** 2
        + (d.bounding_box.origin_y + d.bounding_box.height * 0.5 - ty) ** 2))
    bb = best.bounding_box
    fx1, fy1 = max(0, bb.origin_x), max(0, bb.origin_y)
    fx2, fy2 = min(w, fx1 + bb.width), min(h, fy1 + bb.height)
    return [px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2], float(best.categories[0].score)


def _face_from_keypoints(kpts: np.ndarray, box, frame_w: int, frame_h: int,
                         min_conf: float = 0.5) -> Optional[Tuple[List[int], float]]:
    """Approximate a face box from COCO head keypoints (nose, eyes, ears)."""
    head = kpts[:5]
    vis = head[head[:, 2] >= min_conf]
    if len(vis) < 2:
        return None
    cx, cy = float(vis[:, 0].mean()), float(vis[:, 1].mean())
    span = float(vis[:, 0].max() - vis[:, 0].min())
    shoulders = kpts[5:7]
    shoulder_w = float(abs(shoulders[0, 0] - shoulders[1, 0])) if (shoulders[:, 2] >= min_conf).all() else 0.0
    size = max(span * 1.8, shoulder_w * 0.5, (box[2] - box[0]) * 0.2, 8.0)
    x1, x2 = cx - size / 2, cx + size / 2
    y1, y2 = cy - size * 0.55, cy + size * 0.65
    return ([int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2))],
            float(vis[:, 2].mean()))


# ------------------------------------------------------------------ frame-skip fill
def _lerp_box(a, b, t):
    return [int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4)]


def fill_skipped_frames(frame_tracks: FrameTracks, start: int, end: int) -> FrameTracks:
    """Fill frames that the detector skipped by interpolating each track between
    its surrounding detected frames (or holding the last value at the shot end)."""
    detected = sorted(frame_tracks)
    if not detected:
        return {f: [] for f in range(start, end)}

    filled: FrameTracks = dict(frame_tracks)
    for f in range(start, detected[0]):
        filled[f] = []
    for d0, d1 in zip(detected, detected[1:] + [None]):
        stop = d1 if d1 is not None else end
        if stop - d0 <= 1:
            continue
        nxt = {d["track_id"]: d for d in frame_tracks[d1]} if d1 is not None else {}
        for f in range(d0 + 1, stop):
            t = (f - d0) / (d1 - d0) if d1 is not None else 0.0
            dets = []
            for a in frame_tracks[d0]:
                b = nxt.get(a["track_id"])
                if b is None:
                    dets.append(dict(a))  # hold
                    continue
                d = dict(a)
                d["bbox"] = _lerp_box(a["bbox"], b["bbox"], t)
                if a.get("face_bbox") is not None and b.get("face_bbox") is not None:
                    d["face_bbox"] = _lerp_box(a["face_bbox"], b["face_bbox"], t)
                dets.append(d)
            filled[f] = dets
    return filled
