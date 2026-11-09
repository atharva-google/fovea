"""TalkNet active-speaker detection (model code adapted from TalkNet-ASD, MIT)."""

from .inference import load_talknet, score_tracks
from .model import talkNetModel

__all__ = ["load_talknet", "score_tracks", "talkNetModel"]
