"""Load the pretrained TalkNet checkpoint and score face tracks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

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
    model: torch.nn.Module, processed_tracks: List[Dict[str, Any]], device: str
) -> List[Dict[str, Any]]:
    """Multi-window TalkNet scoring (same scheme as the official demo)."""
    duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    scored = []

    model.eval()
    with torch.no_grad():
        for trk in processed_tracks:
            video_feature = trk["video_features"]  # (T, 112, 112)
            audio_feature = trk["audio_features"]  # (4T, 13)
            length = video_feature.shape[0] / 25.0

            all_scores = []
            for duration in duration_set:
                batch_size = int(math.ceil(length / duration))
                scores = []
                for i in range(batch_size):
                    sub_a = audio_feature[i * duration * 100:(i + 1) * duration * 100, :]
                    sub_v = video_feature[i * duration * 25:(i + 1) * duration * 25, :, :]
                    if len(sub_a) == 0 or len(sub_v) == 0:
                        continue

                    input_a = torch.FloatTensor(sub_a).unsqueeze(0).to(device)
                    input_v = torch.FloatTensor(sub_v).unsqueeze(0).to(device)

                    embed_a = model.forward_audio_frontend(input_a)
                    embed_v = model.forward_visual_frontend(input_v)
                    embed_a, embed_v = model.forward_cross_attention(embed_a, embed_v)
                    out = model.forward_audio_visual_backend(embed_a, embed_v)
                    scores.extend(model.lossAV(out, labels=None))
                all_scores.append(scores)

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
