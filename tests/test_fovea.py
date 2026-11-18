"""Units, the base analysis (prompt, checking the answer, storage), the transcript check and the job
folder. No video model, Gemini or third-party package is needed; ffmpeg is used when installed.

    python -m unittest discover -s tests -v
"""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fovea import analysis, gemini, media, transcript
from fovea.framing import position_at
from fovea.job import Job
from fovea.units import build_audio_units, cut_units, slots

SHOTS = [(0.0, 4.0), (4.0, 8.3), (8.3, 11.0)]
WORDS = [{"word": w, "start_s": s, "end_s": e} for w, s, e in [
    ("I", 1.0, 1.3), ("hope", 1.35, 1.7), ("it's", 1.75, 2.0), ("shoes.", 2.05, 2.6),
    ("Oh", 4.9, 5.18),                                         # then a 0.32 s unspoken "aaa"
    ("A", 5.5, 5.7), ("sweater.", 5.75, 6.5),
    ("Thank", 7.9, 8.2), ("you.", 8.25, 8.9),                  # runs across the cut at 8.3
]]
# what a good answer says about each slot of the example: (kind, speaker, focus_x)
TALK, OTHER = "dialogue_onscreen", "other"
SLOTS = {
    "A01": (OTHER, "", 30), "A02": (TALK, "P1", 30), "A03": (OTHER, "", 30), "A04": (OTHER, "", 30),
    "A05": (OTHER, "", 40), "A06": (TALK, "P1", 40), "A07": (OTHER, "", 40),
    "A08": (TALK, "P2", 70),                                   # the other person answers
    "A09": (OTHER, "", 70), "A10": (OTHER, "", 76),            # who then leans to the right
    "A11a": (TALK, "P1", 40),
    "A11b": ("dialogue_offscreen", "P1", 55),                  # the rest of her line, over the sweater
    "A12": (OTHER, "", 55), "A13": (OTHER, "", 55), "A14": (OTHER, "", 55),
}


def tiles(units, total):
    edges = [(u.start, u.end) for u in units]
    return (abs(edges[0][0]) < 1e-6 and abs(edges[-1][1] - total) < 1e-6
            and all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(edges, edges[1:])) and all(e > s for s, e in edges))


def answer(audio, video):
    """A good answer for the example."""
    text = {a.id: a.text for a in audio}

    def sounds(shot):
        return [{"id": s["id"], "time_range": f"{s['start']:.2f}-{s['end']:.2f}s", "heard": "" if s["part"] else text[s["audio"]] or "(silence)",
                 "kind": SLOTS[s["id"]][0], "speaker": SLOTS[s["id"]][1],
                 "needs": ["A02"] if s["id"] == "A08" else [], "focus_x": SLOTS[s["id"]][2]}
                for s in slots(audio, video) if s["shot"] == shot]

    return {
        "summary": "She hopes for shoes and gets a sweater.",
        "cast": [{"id": "P1", "real_name": "", "ad_name": "Mira", "description": "woman, about 30, red kurta"},
                 {"id": "P2", "real_name": "", "ad_name": "", "description": "man, about 60, grey beard"}],
        "video_units": [
            {"id": "V01", "time_range": "0.00-4.00s", "shows": "a woman walks in with a gift", "cast": ["P1"], "setting": "living room",
             "objects": ["gift box"], "text": [], "needs": [], "audio": sounds("V01")},
            {"id": "V02", "time_range": "4.00-8.30s", "shows": "she opens it next to her father", "cast": ["P1", "P2"], "setting": "living room",
             "objects": ["gift box"], "text": [], "needs": ["V01"], "audio": sounds("V02")},
            {"id": "V03", "time_range": "8.30-11.00s", "shows": "a red sweater in the box", "cast": [], "setting": "living room",
             "objects": ["sweater"], "text": ["For you"], "needs": ["A08"], "audio": sounds("V03")},
        ],
    }


class UnitsTest(unittest.TestCase):
    def test_both_tracks_cover_the_whole_video(self):
        audio, video = cut_units(SHOTS, WORDS)
        self.assertTrue(tiles(audio, 11.0) and tiles(video, 11.0))
        self.assertEqual([v.id for v in video], ["V01", "V02", "V03"])
        self.assertEqual([a.text for a in audio if a.kind == "speech"],
                         ["I hope it's shoes.", "Oh", "A sweater.", "Thank you."])

    def test_a_short_audible_pause_is_its_own_unit_and_silence_comes_in_chunks_of_a_second(self):
        audio = build_audio_units(WORDS, SHOTS)
        oh = next(i for i, a in enumerate(audio) if a.text == "Oh")
        self.assertEqual((audio[oh + 1].kind, audio[oh + 2].text), ("gap", "A sweater."))
        self.assertLessEqual(max(a.dur for a in audio if a.kind == "gap"), 1.0)
        breath = [{"word": "Oh", "start": 5.0, "end": 5.28}, {"word": "no", "start": 5.5, "end": 5.9}]
        self.assertEqual([a.text for a in build_audio_units(breath, SHOTS) if a.kind == "speech"], ["Oh no"])
        self.assertEqual(len(build_audio_units([], [(0.0, 3.5)])), 4)

    def test_a_phrase_across_a_shot_cut_stays_one_unit_with_a_slot_in_each_shot(self):
        audio, video = cut_units(SHOTS, WORDS)
        phrase = next(a for a in audio if a.text == "Thank you.")
        self.assertEqual(phrase.over, ["V02", "V03"])
        parts = [s for s in slots(audio, video) if s["audio"] == phrase.id]
        self.assertEqual([(s["id"], s["shot"], s["start"], s["end"], s["part"]) for s in parts],
                         [("A11a", "V02", phrase.start, 8.3, 0), ("A11b", "V03", 8.3, phrase.end, 1)])
        self.assertEqual(list(SLOTS), [s["id"] for s in slots(audio, video)])

    def test_a_sliver_of_a_phrase_in_another_shot_is_not_a_slot_of_its_own(self):
        words = [{"word": "hello", "start": 3.0, "end": 3.5}, {"word": "there", "start": 3.55, "end": 4.1}]
        audio, video = cut_units(SHOTS, words)
        phrase = next(a for a in audio if a.kind == "speech")
        self.assertEqual([(s["id"], s["shot"], s["end"]) for s in slots(audio, video) if s["audio"] == phrase.id],
                         [(phrase.id, "V01", phrase.end)])

    def test_silence_is_split_at_shot_cuts_and_bad_timings_still_tile(self):
        self.assertTrue({4.0, 8.3} <= {a.start for a in build_audio_units([], SHOTS)})
        bad = [{"word": "one.", "start": 1.0, "end": 2.0}, {"word": "two.", "start": 1.5, "end": 2.5},
               {"word": "late", "start": 10.5, "end": 14.0}]
        self.assertTrue(tiles(build_audio_units(bad, SHOTS), 11.0))
        self.assertEqual(cut_units([], WORDS), ([], []))


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.audio, self.video = cut_units(SHOTS, WORDS)

    def points(self, raw, shot):
        framing = analysis.clean(raw, self.audio, self.video)[1]
        return [(p["t"], p["move"], p["x"]) for p in framing["shots"][shot]["points"]]

    def test_the_prompt_lists_each_shot_with_its_sounds_under_it(self):
        text = analysis.build_prompt(self.audio, self.video)
        self.assertNotIn("{table}", text)
        self.assertNotIn("```json", text)
        self.assertNotIn("{fields}", text)
        self.assertIn('V02 | 4.00-8.30s\n    A05 | 4.00-4.84s | (no words transcribed)\n'
                      '    A06 | 4.84-5.24s | rough transcript: "Oh"', text)
        self.assertIn('    A11a | 7.84-8.30s | rough transcript: "Thank you."\nV03 | 8.30-11.00s\n'
                      '    A11b | 8.30-8.96s | (continued)', text)
        self.assertIn("ONE THIRD of the frame's width", text)
        self.assertNotIn("merely present", text)                # fields are explained in the schema only

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "pydantic not installed")
    def test_without_a_schema_the_prompt_explains_the_fields_itself(self):
        text = analysis.build_prompt(self.audio, self.video, agentic=True)
        self.assertIn("```json", text)
        self.assertIn("merely present", text)
        self.assertLess(text.index("    cast: Ids of"), text.index("    shows: What is seen"))   # look, then describe
        self.assertIn("        focus_x: Centre of the cropped picture", text)

    def test_a_good_answer_is_split_into_log_and_framing(self):
        described, framing, issues = analysis.clean(answer(self.audio, self.video), self.audio, self.video)
        self.assertEqual(issues, [])
        self.assertEqual(described["cast"][0]["ad_name"], "Mira")
        self.assertEqual(sorted(described["video_units"][1]),
                         ["cast", "id", "needs", "objects", "setting", "shows", "text"])
        self.assertEqual(described["audio_units"][7],
                         {"id": "A08", "heard": "A sweater.", "kind": TALK, "speaker": "P2", "needs": ["A02"],
                          "parts": [{"id": "A08", "shot": "V02", "kind": TALK}]})
        thanks = described["audio_units"][10]                   # one line: seen being said, then heard over V03
        self.assertEqual((thanks["heard"], thanks["speaker"]), ("Thank you.", "P1"))
        self.assertEqual(thanks["parts"], [{"id": "A11a", "shot": "V02", "kind": TALK},
                                           {"id": "A11b", "shot": "V03", "kind": "dialogue_offscreen"}])
        self.assertEqual([s["id"] for s in framing["shots"]], ["V01", "V02", "V03"])

    def test_the_picture_jumps_with_the_speaker_and_glides_with_a_moving_subject(self):
        raw = answer(self.audio, self.video)
        self.assertEqual(self.points(raw, 0), [(0.0, "cut", 30)])                # nothing changes: one point
        self.assertEqual(self.points(raw, 1), [
            (4.0, "cut", 40),         # opens on her
            (5.44, "cut", 70),        # he starts to speak: jump, at the time his line starts
            (6.88, "glide", 70),      # stays on him until the middle of A09 ...
            (7.52, "glide", 76),      # ... and follows him, there by the middle of A10
            (7.84, "cut", 40)])       # she speaks: jump back
        self.assertEqual(self.points(raw, 2), [(8.3, "cut", 55)])
        framing = analysis.clean(raw, self.audio, self.video)[1]
        self.assertEqual([round(position_at(framing["shots"][1]["points"], t), 1) for t in (5.4, 5.5, 6.7, 7.2, 7.7)],
                         [40.0, 70.0, 70.0, 73.0, 76.0])

    def test_the_same_person_speaking_on_is_followed_not_jumped_to(self):
        raw = answer(self.audio, self.video)
        for s in raw["video_units"][1]["audio"]:
            if s["id"] in ("A06", "A07", "A08"):
                s.update(kind=TALK, speaker="P1", focus_x={"A06": 40, "A07": 44, "A08": 48}[s["id"]])
        self.assertEqual(self.points(raw, 1)[:4],
                         [(4.0, "cut", 40), (5.04, "glide", 40), (5.34, "glide", 44), (6.0, "glide", 48)])

    def test_a_unit_described_under_the_wrong_time_range_is_reported(self):
        raw = answer(self.audio, self.video)
        raw["video_units"][1]["time_range"] = "4.00 - 8.30"                     # same range, written loosely: fine
        raw["video_units"][2]["time_range"] = "4.00-8.30s"                      # V03 described as V02's stretch
        raw["video_units"][2]["audio"][1]["time_range"] = "9.64-10.32s"
        issues = analysis.clean(raw, self.audio, self.video)[2]
        self.assertEqual(len(issues), 2)
        self.assertIn("V03 was described as 4.00-8.30s but is 8.30-11.00s", issues[0])
        self.assertIn("A12 was described as 9.64-10.32s", issues[1])

    def test_a_missing_or_bad_focus_keeps_the_last_position_in_the_shot(self):
        raw = answer(self.audio, self.video)
        by_id = {s["id"]: s for v in raw["video_units"] for s in v["audio"]}
        del by_id["A01"]["focus_x"]
        by_id["A09"]["focus_x"] = "left"
        by_id["A12"]["focus_x"] = 140
        _, framing, issues = analysis.clean(raw, self.audio, self.video)
        self.assertEqual(framing["shots"][0]["points"][0]["x"], 50)             # nothing before it in the shot
        self.assertEqual(position_at(framing["shots"][1]["points"], 6.7), 70.0)
        self.assertEqual(position_at(framing["shots"][2]["points"], 9.3), 100.0)
        self.assertEqual(len([i for i in issues if "focus" in i]), 2)

    def test_the_picture_jumps_on_a_cut_and_moves_evenly_on_a_glide(self):
        points = [{"t": 4.0, "move": "cut", "x": 40}, {"t": 5.5, "move": "cut", "x": 70},
                  {"t": 7.0, "move": "glide", "x": 80}]
        self.assertEqual([position_at(points, t) for t in (4.0, 5.49, 5.5, 6.25, 7.0, 8.2)],
                         [40.0, 40.0, 70.0, 75.0, 80.0, 80.0])
        self.assertEqual(position_at(points, 0.0), 40.0)

    def test_ids_cast_speakers_kinds_and_needs_are_checked(self):
        raw = answer(self.audio, self.video)
        raw["video_units"][0].update(cast=["P1", "P9"], needs=["V03", "Z1"])
        by_id = {s["id"]: s for v in raw["video_units"] for s in v["audio"]}
        by_id["A02"].update(kind="shouting", speaker="P9")
        by_id["A01"]["kind"] = ""
        by_id["A08"]["needs"] = ["A11b", "A02"]                                  # a later unit, and a fine one
        raw["video_units"][1]["audio"].append({"id": "A77", "heard": "x", "kind": OTHER, "focus_x": 5})
        raw["video_units"][1]["audio"] += raw["video_units"][2]["audio"][1:]     # put under the wrong shot: fine
        raw["video_units"] = raw["video_units"][:2] + [{"id": "V99"}]
        described, framing, issues = analysis.clean(raw, self.audio, self.video)
        self.assertEqual((described["video_units"][0]["cast"], described["video_units"][0]["needs"]), (["P1"], []))
        self.assertEqual(described["video_units"][2]["shows"], "")                # V03 was missing
        self.assertEqual((described["audio_units"][1]["kind"], described["audio_units"][1]["speaker"]), (TALK, ""))
        self.assertEqual(described["audio_units"][0]["kind"], OTHER)
        self.assertEqual(described["audio_units"][7]["needs"], ["A02"])
        self.assertEqual(described["audio_units"][10]["parts"][1]["kind"], TALK)  # A11b missing: as its first part
        self.assertEqual(framing["shots"][2]["points"], [{"t": 8.3, "move": "cut", "x": 50, "slot": "A11b"},
                                                         {"t": 8.96, "move": "cut", "x": 55, "slot": "A12"}])
        for word in ("'P9'", "'Z1'", "V03, which is not earlier", "A11, which is not earlier", "'A77'", "'V99'",
                     "V03 missing", "A11b missing"):
            self.assertTrue(any(word in i for i in issues), word)

    def test_the_answer_is_stored_in_three_files_and_only_reused_for_the_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "my_ad.mp4")
            src.write_bytes(b"one video")
            job = Job.open(src, Path(d, "fovea_data"))
            self.assertIsNone(analysis.load(job))
            described, framing, _ = analysis.clean(answer(self.audio, self.video), self.audio, self.video)
            analysis.save(job, self.audio, self.video, described, framing, {"model": "m", "seconds": 1.0})
            self.assertEqual(sorted(p.name for p in job.root.rglob("*.json")),
                             ["framing.json", "job.json", "log.json", "units.json"])
            got = analysis.load(job)
            self.assertEqual((got["audio"], got["video"]), (self.audio, self.video))
            self.assertEqual(got["framing"]["shots"], framing["shots"])
            text = analysis.report(got)
            self.assertIn("A08     5.44-  6.56s  focus  70%         dialogue_onscreen  P2: A sweater.  needs A02", text)
            self.assertIn("A10     7.20-  7.84s  focus  73% >  76%  other", text)
            self.assertIn("A11b    8.30-  8.96s  focus  55%         dialogue_offscreen (continued)", text)

            src.write_bytes(b"a different video with the same name")
            job = Job.open(src, Path(d, "fovea_data"))                                 # the folder is emptied
            self.assertIsNone(analysis.load(job))
            self.assertEqual(job.source.read_bytes(), b"a different video with the same name")

    def test_without_a_stored_answer_or_a_key_it_stops_before_calling_gemini(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "ad.mp4")
            src.write_bytes(b"v")
            job = Job.open(src, Path(d, "data"))
            with self.assertRaises(analysis.KeyRequired):
                analysis.run(job, [], [], analysis.Settings(), api_key=None, log=lambda m: None)


class SmallPartsTest(unittest.TestCase):
    def test_json_is_found_in_a_plain_answer_and_in_a_block(self):
        self.assertEqual(gemini.extract_json('{"summary": "s"}'), {"summary": "s"})
        text = 'thinking...\n```json\n{"summary": "old"}\n```\nmore\n```json\n{"summary": "new"}\n```'
        self.assertEqual(gemini.extract_json(text), {"summary": "new"})
        self.assertIsNone(gemini.extract_json("no json here"))

    def test_a_stuck_transcript_is_spotted(self):
        ok = [{"word": w} for w in ["I", "hope", "it's", "shoes.", "no,", "no,", "no!"]]
        self.assertEqual(transcript.problems(ok), [])
        stuck = [{"word": "\u092a\u094d\u0930\u094b" + "\u0964" * 80}]
        self.assertEqual(len(transcript.problems(stuck)), 2)
        self.assertTrue(transcript.problems([{"word": "the"}] * 12))

    def test_transcript_csv_round_trip_keeps_unicode(self):
        words = [{"word": "\u0939\u093e\u0901!", "start_s": 1.52, "end_s": 1.7, "probability": 0.91, "segment": 0}]
        with tempfile.TemporaryDirectory() as d:
            transcript.write_csv(Path(d, "t.csv"), words)
            self.assertEqual(transcript.read_csv(Path(d, "t.csv")), words)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not installed")
class MediaTest(unittest.TestCase):
    def test_probe_proxy_and_sound(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "src.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(src)], check=True)
            info = media.probe(src)
            self.assertEqual((info["width"], info["height"], info["fps"], info["has_audio"]), (640, 360, 30.0, True))
            self.assertAlmostEqual(info["duration"], 2.0, delta=0.1)
            proxy = media.probe(media.make_proxy(src, Path(d, "proxy.mp4"), height=180, fps=25.0))
            self.assertEqual((proxy["width"], proxy["height"], proxy["fps"], proxy["has_audio"]), (320, 180, 25.0, True))
            wav = media.extract_audio(src, Path(d, "audio.wav"))
            self.assertGreater(wav.stat().st_size, 2 * 16000 * 2 * 0.9)          # ~2 s of 16 kHz mono 16-bit


if __name__ == "__main__":
    unittest.main()
