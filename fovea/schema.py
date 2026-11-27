"""The shapes Gemini answers in (the log, then the edit), and THE place where every field is explained:
the prompts (analysis.py, edit.py) only give the task and the rules that span units.

Fields are in the order Gemini should fill them in: what it sees and hears first, conclusions after.
"""

from __future__ import annotations

from enum import Enum
from typing import List, get_args

from pydantic import BaseModel, Field


class CastMember(BaseModel):
    id: str = Field(description="P1, P2, ... in order of first appearance.")
    real_name: str = Field(description="Their real name, ONLY if this is a well-known person you recognise with "
                                       "confidence. Otherwise ''. Never guess.")
    ad_name: str = Field(description="The name the video itself gives them (spoken or on screen). Otherwise ''.")
    description: str = Field(description="How they LOOK in these frames, so they can be told apart: apparent age and "
                                         "gender, hair, build, clothing. Only what is actually seen, never what a "
                                         "known person usually wears. Nothing about what they do.")


class Kind(str, Enum):
    dialogue_onscreen = "dialogue_onscreen"
    dialogue_offscreen = "dialogue_offscreen"
    voiceover = "voiceover"
    lyrics = "lyrics"
    other = "other"


class Music(str, Enum):
    none = "none"
    score = "score"
    song = "song"


class MusicPart(str, Enum):
    none = "none"
    intro = "intro"
    build = "build"
    peak = "peak"
    breakdown = "break"
    outro = "outro"


class AudioUnit(BaseModel):
    id: str = Field(description="The id exactly as listed, e.g. A05 or A05b.")
    time_range: str = Field(description="Its time range, copied exactly as listed, e.g. '1.80-4.36s'. Everything "
                                        "below is about THIS stretch only.")
    heard: str = Field(description="What is REALLY heard: speech written out in its own language and script, "
                                   "correcting the rough transcript; any other sound in round brackets, e.g. "
                                   "'(laughs)', '(music)', '(silence)'. '' for a unit marked (continued).")
    kind: Kind = Field(description="dialogue_onscreen: spoken by someone whose mouth is SEEN saying it in this "
                                   "stretch. dialogue_offscreen: spoken by someone in the scene who is not seen "
                                   "saying it (out of frame, back to the camera, or the picture shows something "
                                   "else). voiceover: a narrator. lyrics: sung words that are part of the music. "
                                   "other: music, effects, a breath, a laugh, silence.")
    speaker: str = Field(description="Cast id of who says it, decided for this unit on its own: by whose mouth "
                                     "moves, or by the voice when no mouth is seen. '' when nothing is spoken or "
                                     "the voice is not a cast member.")
    with_music: bool = Field(description="True ONLY when these words cannot be moved apart from the music: they "
                                          "are sung or lip-synced by someone seen, or deliberately timed to its "
                                          "beat. False for ordinary speech with music under it, and when nothing "
                                          "is said.")
    music: Music = Field(description="The music under this stretch. none: no music. score: instrumental music. "
                                     "song: a track with singing, also in the stretches between its sung lines.")
    music_part: MusicPart = Field(description="Which part of the track this stretch belongs to. intro: it starts "
                                              "up. build: tension or energy rises. peak: the high point (a chorus, "
                                              "a drop, the payoff). break: it thins out or pauses. outro: it winds "
                                              "down or ends. none when there is no music. Neighbouring units in "
                                              "the same part get the same value.")
    sfx: str = Field(description="A sound made by something that is SEEN in this stretch, in a few words: 'wrapper "
                                 "torn', 'door slams', 'bottle opens'. Not music, not a voice. '' if there is none.")
    focus_x: int = Field(description="Centre of the cropped picture during this stretch, as a percentage of the "
                                     "frame width (0 = left edge, 100 = right edge): the centre of ONE person or "
                                     "thing, never the space between two. In order of priority: 1) the action or "
                                     "movement that is the point of the shot, even while someone else speaks; "
                                     "2) the main subject; 3) in a wide shot, people the cropped picture showed "
                                     "in earlier shots; 4) the person speaking on screen. Follow a subject that "
                                     "moves; repeat the same number while nothing changes.")
    needs: List[str] = Field(description="Ids of EARLIER units this sound makes no sense without (a punchline needs "
                                         "its setup line). Not the sound that merely comes before it. Usually [].")


class VideoUnit(BaseModel):
    id: str = Field(description="The id exactly as listed, e.g. V07.")
    time_range: str = Field(description="Its time range, copied exactly as listed, e.g. '5.64-6.32s'. Everything "
                                        "below is about the frames inside it only.")
    setting: str = Field(description="Where it is, as a short label. The same wording for every shot in the same "
                                     "place.")
    cast: List[str] = Field(description="Ids of the cast members visible at any point in this shot, also partly or "
                                        "from behind when their clothes or hair identify them. [] for a shot of "
                                        "only objects or unidentifiable hands.")
    objects: List[str] = Field(description="Things in this shot that matter to the story or that someone would "
                                           "name: the product, props that are handled, vehicles, animals. A few "
                                           "words each, the same wording in every shot.")
    text: List[str] = Field(description="Readable text on screen or on objects that belongs to THIS shot, corrected. "
                                        "Not a logo or disclaimer that sits in the same place throughout. [] if none.")
    shows: str = Field(description="What is seen, in one factual sentence that agrees with the fields above: who "
                                   "physically does what (the person who speaks is not always the one who acts). "
                                   "People who are not in the cast are mentioned here.")
    needs: List[str] = Field(description="Ids of EARLIER units this shot makes no sense without (a reaction needs "
                                         "what it reacts to; a reveal needs its setup). Not a shot that merely "
                                         "comes before it or shows the same scene. Usually [].")
    audio: List[AudioUnit] = Field(description="Every audio unit listed under this shot, in the same order.")


class Analysis(BaseModel):
    summary: str = Field(description="One or two sentences: the story of the video.")
    cast: List[CastMember] = Field(description="ONLY the people who play a part in the story: they speak, are "
                                               "spoken to, or do something the story depends on. Someone who is "
                                               "merely present (sitting in the room, part of a crowd) is NOT cast, "
                                               "however clearly they are seen. The same person is ONE entry however "
                                               "often they appear.")
    video_units: List[VideoUnit] = Field(description="Every video unit, in the order listed.")


# =========================================================================== the edit
class Reason(BaseModel):
    id: str = Field(description="An audio unit id exactly as listed, e.g. A12.")
    why: str = Field(description="A few words: what the story loses without it.")


class Kept(BaseModel):
    id: str = Field(description="An audio unit id exactly as listed, e.g. A12.")
    seconds: float = Field(description="Its length, copied exactly as listed, e.g. 1.72. Add these up as you go.")


class Edit(BaseModel):
    story: str = Field(description="One sentence: what the short version must still tell, from setup to payoff.")
    must_keep: List[Reason] = Field(description="The few units the story cannot lose: the setup, the payoff, and "
                                                "where the product or brand is seen or named. Decide these before "
                                                "anything else.")
    keep: List[Kept] = Field(description="Every unit of the short version, in the order listed: everything in "
                                         "must_keep, plus what makes it play smoothly.")


def outline(model: type = Analysis, indent: int = 0) -> str:
    """The fields and their explanations as text, for a call that cannot be given the schema itself."""
    lines = []
    for name, f in model.model_fields.items():
        lines.append(f"{' ' * indent}{name}: {f.description}")
        inner = next((a for a in get_args(f.annotation) if isinstance(a, type) and issubclass(a, BaseModel)), None)
        if inner:
            lines.append(f"{' ' * indent}  each one:")
            lines.append(outline(inner, indent + 4))
    return "\n".join(lines)
