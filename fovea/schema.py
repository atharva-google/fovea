"""The shape Gemini answers in for the base analysis. It never writes a timestamp: only unit ids."""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class CastMember(BaseModel):
    id: str = Field(description="P1, P2, ... in order of first appearance.")
    real_name: str = Field(description="Their real name, ONLY if this is a well-known person you recognise with "
                                       "confidence. Otherwise an empty string. Never guess.")
    ad_name: str = Field(description="The name the video itself gives them (spoken or on screen). Otherwise empty.")
    description: str = Field(description="How they LOOK, so they can be told apart: apparent age and gender, hair, "
                                         "build, clothing. Nothing about what they do in the story.")


class Kind(str, Enum):
    dialogue_onscreen = "dialogue_onscreen"     # spoken by someone whose face is SEEN saying it
    dialogue_offscreen = "dialogue_offscreen"   # spoken by someone in the scene who is not seen saying it
    voiceover = "voiceover"                     # a narrator
    lyrics = "lyrics"                           # sung words that are part of the music
    other = "other"                             # music, effects, a breath, silence


class AudioUnit(BaseModel):
    id: str = Field(description="The id as listed under the shot, e.g. A05 or A05b. Each appears exactly once.")
    heard: str = Field(description="What is REALLY heard. Speech written out and corrected from the rough "
                                   "transcript. Other sounds in round brackets: '(laughs)', '(music)', '(silence)'. "
                                   "Empty for a part marked (continued).")
    kind: Kind
    speaker: str = Field(description="The cast id of who says it. Empty when nothing is spoken, or the voice is not "
                                     "a cast member.")
    needs: List[str] = Field(description="EARLIER units (audio or video ids) this sound makes no sense without. Not "
                                         "the sound that merely comes before it. Most sounds need nothing.")
    focus_x: int = Field(description="Where the cropped picture should be centred WHILE this sound plays: the centre "
                                     "of ONE person or thing, left to right, as a percentage of the frame width "
                                     "(0 = left edge, 50 = middle, 100 = right edge).")


class VideoUnit(BaseModel):
    id: str = Field(description="Video unit id, e.g. V07. Every video unit appears exactly once.")
    shows: str = Field(description="What is SEEN, in one sentence: who, doing what. People who are not in the cast "
                                   "list (a crowd, passers-by) are mentioned here.")
    cast: List[str] = Field(description="Ids of the cast members recognisably visible at ANY point in this shot.")
    setting: str = Field(description="Where it is, as a short label. The same wording for every shot in the same place.")
    objects: List[str] = Field(description="Things that matter to the story or that someone would name: the product, "
                                           "props that are handled, vehicles, animals. Same wording in every shot.")
    text: List[str] = Field(description="Readable on-screen text that belongs to THIS shot, corrected. Not a logo or "
                                        "disclaimer that sits in the same place throughout the video.")
    needs: List[str] = Field(description="EARLIER units (video or audio ids) this shot makes no sense without. Not a "
                                         "shot that merely comes before it. Most shots need nothing.")
    audio: List[AudioUnit] = Field(description="Every audio unit listed under this shot, in the same order.")


class Analysis(BaseModel):
    summary: str = Field(description="One or two sentences: the story of the video.")
    cast: List[CastMember] = Field(description="Only people who play a part in the story: they speak, are spoken "
                                               "to, or do something the story depends on. Not people who are merely "
                                               "present.")
    video_units: List[VideoUnit]
