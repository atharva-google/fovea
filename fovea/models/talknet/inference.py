"""Load the pretrained TalkNet checkpoint and score face tracks."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .model import talkNetModel


def load_talknet(checkpoint_path: str | Path, device: str) -> talkNetModel:
    model = talkNetModel()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    clean_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        # Skip training-only single-modality auxiliary heads
        if name.startswith("lossA.") or name.startswith("lossV."):
            continue
        # Map prefixed keys to top-level attributes in talkNetModel
        if name.startswith("model."):
            name = name[len("model."):]
        elif name.startswith("lossAV."):
            name = name[len("lossAV."):]
        clean_dict[name] = v

    # strict=True ensures all weights match exactly without silent failures
    model.load_state_dict(clean_dict, strict=True)
    model.eval()
    return model.to(device)


def score_tracks(
    model: torch.nn.Module, processed_tracks: List[Dict[str, Any]], device: str,
    max_batch_frames: int = 1200,
) -> List[Dict[str, Any]]:
    """Multi-window TalkNet scoring (same scheme as the official demo), batched on the GPU.

    Every track is cut into windows of 1-6 s, scored, and the per-window scores are averaged.
    Instead of one tiny forward pass per window, windows of the same length from ALL tracks are
    stacked into one batch (at most ``max_batch_frames`` video frames per pass). Each window
    is still its own sample (see ``_forward_windows``), so scores match the one-by-one version.
    The duplicated durations in ``duration_set`` are computed once and reused.
    """
    duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]

    # 1. list every window, grouped by (video frames, audio frames) so a group can be stacked
    groups: Dict[Tuple[int, int], List[Tuple[int, int, int, int, int, int, int]]] = defaultdict(list)
    for ti, trk in enumerate(processed_tracks):
        n_v, n_a = trk["video_features"].shape[0], trk["audio_features"].shape[0]
        length = n_v / 25.0
        for duration in sorted(set(duration_set)):
            for i in range(int(math.ceil(length / duration))):
                v0, v1 = i * duration * 25, min((i + 1) * duration * 25, n_v)
                a0, a1 = i * duration * 100, min((i + 1) * duration * 100, n_a)
                if v1 <= v0 or a1 <= a0:
                    continue
                groups[(v1 - v0, a1 - a0)].append((ti, duration, i, v0, v1, a0, a1))

    # 2. score each group in batches
    window_scores: Dict[Tuple[int, int, int], np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for (n_v, _), items in groups.items():
            per_batch = max(1, max_batch_frames // n_v)
            for k in range(0, len(items), per_batch):
                chunk = items[k:k + per_batch]
                v = np.stack([processed_tracks[ti]["video_features"][v0:v1] for ti, _, _, v0, v1, _, _ in chunk])
                a = np.stack([processed_tracks[ti]["audio_features"][a0:a1] for ti, _, _, _, _, a0, a1 in chunk])
                scores = _forward_windows(model, torch.FloatTensor(a).to(device), torch.FloatTensor(v).to(device))
                for (ti, duration, i, *_), row in zip(chunk, scores):
                    window_scores[(ti, duration, i)] = row

    # 3. per track: join windows per duration, average over durations, smooth
    scored = []
    for ti, trk in enumerate(processed_tracks):
        length = trk["video_features"].shape[0] / 25.0
        all_scores = []
        for duration in duration_set:
            parts = [window_scores[(ti, duration, i)] for i in range(int(math.ceil(length / duration)))
                     if (ti, duration, i) in window_scores]
            all_scores.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32))

        # Average over all duration windows
        min_len = min(len(s) for s in all_scores)
        mean_scores = np.mean(np.array([s[:min_len] for s in all_scores]), axis=0)

        # 5-frame smoothing window
        smoothed = [
            float(np.mean(mean_scores[max(i - 2, 0):min(i + 3, len(mean_scores))]))
            for i in range(len(mean_scores))
        ]

        scored.append({
            "shot_id": trk["shot_id"],
            "track_id": trk["track_id"],
            "frames": trk["frames"][:len(smoothed)],
            "scores": np.array(smoothed),
        })
    return scored


def _forward_windows(model: torch.nn.Module, input_a: torch.Tensor, input_v: torch.Tensor) -> np.ndarray:
    """Score a batch of equal-length windows: audio (B, 4T, 13), video (B, T, 112, 112) -> (B, T).

    ``model.forward_visual_frontend`` folds the batch into ONE long time axis before its 3D conv
    (fine for batch size 1, but neighbouring windows would bleed into each other). This version
    keeps every window as its own sample, so each row equals scoring that window alone.
    """
    B, T, W, H = input_v.shape
    x = (input_v / 255 - 0.4161) / 0.1688
    x = x.reshape(B, 1, T, W, H)                          # (batch, channel, time, W, H)
    x = model.visualFrontend.frontend3D(x)                # (B, 64, T, w, h)
    x = x.transpose(1, 2)
    x = x.reshape(B * T, x.shape[2], x.shape[3], x.shape[4])
    x = model.visualFrontend.resnet(x).reshape(B, T, 512)  # per-frame 2D ResNet
    x = x.transpose(1, 2)
    x = model.visualTCN(x)
    x = model.visualConv1D(x)
    embed_v = x.transpose(1, 2)

    embed_a = model.forward_audio_frontend(input_a)
    embed_a, embed_v = model.forward_cross_attention(embed_a, embed_v)
    out = model.forward_audio_visual_backend(embed_a, embed_v)  # (B*T, 256), window-major
    return model.lossAV(out, labels=None).reshape(B, -1)
