"""Units, the log and the edit (prompt, checking the answer, storage), rendering, the transcript check
and the job folder. No video model, Gemini or third-party package is needed; ffmpeg is used when installed.

    python -m unittest discover -s tests -v
"""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fovea import analysis, edit, gemini, media, render, transcript
from fovea.framing import crop_track, position_at
from fovea.job import Job
from fovea.units import VideoUnit, build_audio_units, cut_units, slots

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


class EditTest(unittest.TestCase):
    def setUp(self):
        self.audio, self.video = cut_units(SHOTS, WORDS)
        self.log = analysis.clean(answer(self.audio, self.video), self.audio, self.video)[0]

    def raw(self, *ids, must=("A08",)):
        dur = {a.id: round(a.dur, 2) for a in self.audio}
        return {"story": "She gets a sweater.", "must_keep": [{"id": m, "why": "the payoff"} for m in must],
                "keep": [{"id": i, "seconds": dur[i]} for i in ids]}

    def test_the_prompt_is_the_log_with_lengths_and_no_times(self):
        text = edit.build_prompt(self.audio, self.video, self.log, 6)
        self.assertNotIn("{log}", text)
        self.assertIn("Aim for about 6 seconds", text)
        self.assertIn("  P1 (Mira): woman, about 30, red kurta", text)
        self.assertIn("V02 | 4.30s | cast: P1 P2 | she opens it next to her father | needs: V01\n"
                      "    A05 | 0.84s | other | (silence)", text)
        self.assertIn("    A08 | 1.12s | dialogue_onscreen | P2: A sweater. | needs: A02", text)
        self.assertIn("    A11 | 1.12s | dialogue_onscreen | P1: Thank you.\n"
                      "V03 | 2.70s | cast: - | a red sweater in the box | text: For you | needs: A08\n"
                      "    A11 | (continued)", text)

    def test_kept_units_become_stretches_of_the_video_with_what_they_need(self):
        got, issues = edit.clean(self.raw("A07", "A08", "A11"), self.audio, self.log)
        self.assertEqual(got["keep"], ["A02", "A07", "A08", "A11"])               # A08 needs A02
        self.assertEqual(got["segments"], [[0.94, 2.66], [5.24, 6.56], [7.84, 8.96]])
        self.assertEqual(got["seconds"], 4.16)
        self.assertEqual(got["must_keep"], [{"id": "A08", "why": "the payoff"}])
        self.assertEqual(issues, ["A08 needs A02: added"])

    def test_a_shot_whose_setup_is_not_kept_is_reported(self):
        log = {**self.log, "audio_units": [{**u, "needs": []} for u in self.log["audio_units"]]}
        got, issues = edit.clean(self.raw("A08"), self.audio, log)                 # V02 needs V01: nothing of it kept
        self.assertEqual(got["keep"], ["A08"])
        self.assertEqual(issues, ["V02 needs V01, of which nothing is kept"])

    def test_wrong_ids_lengths_order_and_forgotten_essentials_are_fixed_and_reported(self):
        raw = self.raw("A11", "A02", must=("A08", "A99"))
        raw["keep"] += [{"id": "A11b", "seconds": 1.12}, {"id": "V02", "seconds": 4.3}, {"id": "A06", "seconds": 2.0},
                        {"id": "A07", "seconds": 0.2}]
        got, issues = edit.clean(raw, self.audio, self.log)
        self.assertEqual(got["keep"], ["A02", "A06", "A07", "A08", "A11"])
        self.assertEqual(got["segments"], [[0.94, 2.66], [4.84, 6.56], [7.84, 8.96]])
        for word in ("'V02'", "'A99'", "A06 was taken to be 2s but is 0.40s", "original order",
                     "A08 is in must_keep but was not kept"):
            self.assertTrue(any(word in i for i in issues), word)
        self.assertEqual(len(issues), 5)
        flash = edit.clean(self.raw("A07", must=()), self.audio, self.log)[1]
        self.assertTrue(any("flash" in i for i in flash))

    def test_an_edit_is_stored_per_length_and_made_again_when_the_log_is_new(self):
        asked = []

        def fake(prompt, *a, **k):
            asked.append(prompt)
            return self.raw("A02", "A08"), 1.0

        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "ad.mp4")
            src.write_bytes(b"v")
            job = Job.open(src, Path(d, "data"))
            args = (job, self.audio, self.video, self.log)
            with self.assertRaises(analysis.KeyRequired):
                edit.run(*args, 6, analysis.Settings(), None, log=lambda m: None)
            real, gemini.ask = gemini.ask, fake
            try:
                first = edit.run(*args, 6, analysis.Settings(), "key", log=lambda m: None)
                self.assertEqual(edit.run(*args, 6, analysis.Settings(), None, log=lambda m: None), first)
                edit.run(*args, 10, analysis.Settings(), "key", log=lambda m: None)
                edit.run(*args, 6, analysis.Settings(), "key", reuse=False, log=lambda m: None)
            finally:
                gemini.ask = real
            self.assertEqual(len(asked), 3)
            self.assertEqual(sorted(p.name for p in (job.root / "edit").iterdir()),
                             ["edit_synced_10s.json", "edit_synced_6s.json"])
            self.assertEqual((first["target"], first["seconds"], first["keep"]), (6, 2.84, ["A02", "A08"]))
            text = edit.report(first, self.audio, self.log)
            self.assertIn("EDIT (synced): 2.8s of 11.0s, aiming for about 6s", text)
            self.assertIn("  ! A08    5.44-  6.56s  P2: A sweater.   <- the payoff", text)
            self.assertIn("  + A02    0.94-  2.66s  P1: I hope it's shoes.", text)
            self.assertIn("    A03    2.66-  3.33s  (silence)", text)


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


class CropTrackTest(unittest.TestCase):
    """A 1000 px wide video, cropped 400 px wide: the left edge can be anywhere from 0 to 600."""

    def track(self, points, dead_zone=3.0, end=10.0):
        return crop_track([{"start": 0.0, "end": end, "points": points}], 1000, 400, dead_zone)

    def test_the_crop_is_centred_on_the_line_and_stays_inside_the_picture(self):
        self.assertEqual(self.track([{"t": 0.0, "move": "cut", "x": 50}]), [(0.0, 10.0, 300.0, 300.0)])
        self.assertEqual(self.track([{"t": 0.0, "move": "cut", "x": 5}])[0][2], 0.0)
        self.assertEqual(self.track([{"t": 0.0, "move": "cut", "x": 95}])[0][2], 600.0)

    def test_it_jumps_on_a_cut_and_moves_evenly_on_a_glide(self):
        points = [{"t": 0.0, "move": "cut", "x": 30}, {"t": 4.0, "move": "cut", "x": 70},
                  {"t": 6.0, "move": "glide", "x": 70}, {"t": 8.0, "move": "glide", "x": 50}]
        self.assertEqual(self.track(points), [(0.0, 4.0, 100.0, 100.0), (4.0, 6.0, 500.0, 500.0),
                                              (6.0, 8.0, 500.0, 300.0), (8.0, 10.0, 300.0, 300.0)])

    def test_small_moves_are_not_followed_but_a_slow_drift_is(self):
        twitch = [{"t": 0.0, "move": "cut", "x": 50}, {"t": 3.0, "move": "cut", "x": 52},
                  {"t": 5.0, "move": "glide", "x": 49}]
        self.assertEqual(self.track(twitch), [(0.0, 10.0, 300.0, 300.0)])
        drift = [{"t": 0.0, "move": "cut", "x": 40}] + [{"t": float(t), "move": "glide", "x": 40 + 2 * t}
                                                         for t in range(1, 6)]
        self.assertEqual(self.track(drift), [(0.0, 2.0, 200.0, 240.0), (2.0, 4.0, 240.0, 280.0),
                                             (4.0, 10.0, 280.0, 280.0)])

    def test_shots_follow_each_other_without_a_gap(self):
        shots = [{"start": 0.0, "end": 2.0, "points": [{"t": 0.0, "move": "cut", "x": 50}]},
                 {"start": 2.0, "end": 5.0, "points": [{"t": 2.0, "move": "cut", "x": 50}]},
                 {"start": 5.0, "end": 9.0, "points": [{"t": 5.0, "move": "cut", "x": 20}]}]
        self.assertEqual(crop_track(shots, 1000, 400), [(0.0, 5.0, 300.0, 300.0), (5.0, 9.0, 0.0, 0.0)])

    def test_formats(self):
        self.assertEqual(render.parse_format("9:16"), ("9x16", 0.5625))
        self.assertEqual(render.parse_format("4x5"), ("4x5", 0.8))
        for bad in ("vertical", "9:0", "9"):
            with self.assertRaises(ValueError):
                render.parse_format(bad)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not installed")
class RenderTest(unittest.TestCase):
    def test_every_format_is_cropped_along_the_line_and_switches_on_the_right_frame(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "ad.mp4")                        # red | green | blue, 2 s at 25 fps, with sound
            colours = [x for c, w in (("red", 214), ("green", 212), ("blue", 214))
                       for x in ("-f", "lavfi", "-i", f"color={c}:s={w}x360:r=25:d=2")]
            subprocess.run(["ffmpeg", "-y", "-v", "error", *colours, "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-filter_complex", "[0][1][2]hstack=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", str(src)], check=True)
            job = Job.open(src, Path(d, "data"))
            video = [VideoUnit("V01", 0.0, 1.0), VideoUnit("V02", 1.0, 2.0)]
            framing = {"shots": [{"id": "V01", "points": [{"t": 0.0, "move": "cut", "x": 20}]},
                                 {"id": "V02", "points": [{"t": 1.0, "move": "cut", "x": 80}]}]}
            notes = []
            files = render.render(job, video, framing, media.probe(src), ["9:16", "1:1", "16:9", "21:9"],
                                  log=notes.append)
            self.assertEqual([f.name for f in files], ["ad_9x16.mp4", "ad_1x1.mp4"])
            self.assertEqual(sum("skipped" in n for n in notes), 2)                # nothing to crop for the wide ones
            sizes = [media.probe(f) for f in files]
            self.assertEqual([(i["width"], i["height"], i["has_audio"]) for i in sizes], [(202, 360, True), (360, 360, True)])
            self.assertAlmostEqual(sizes[0]["duration"], 2.0, delta=0.1)

            def colour(frame):                                                      # the average colour of one frame
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(files[0]), "-vf",
                                      f"select=eq(n\\,{frame}),scale=1:1", "-frames:v", "1", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
                return "red" if raw[0] > raw[2] else "blue"

            self.assertEqual([colour(n) for n in (0, 24, 25, 49)], ["red", "red", "blue", "blue"])

    def test_an_edit_keeps_only_its_stretches_in_the_video_s_shape_and_in_every_format(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "ad.mp4")
            colours = [x for c, w in (("red", 214), ("green", 212), ("blue", 214))
                       for x in ("-f", "lavfi", "-i", f"color={c}:s={w}x360:r=25:d=2")]
            subprocess.run(["ffmpeg", "-y", "-v", "error", *colours, "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-filter_complex", "[0][1][2]hstack=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", str(src)], check=True)
            job = Job.open(src, Path(d, "data"))
            video = [VideoUnit("V01", 0.0, 1.0), VideoUnit("V02", 1.0, 2.0)]
            framing = {"shots": [{"id": "V01", "points": [{"t": 0.0, "move": "cut", "x": 20}]},
                                 {"id": "V02", "points": [{"t": 1.0, "move": "cut", "x": 80}]}]}
            files = render.render(job, video, framing, media.probe(src), ["9:16"], log=lambda m: None,
                                  segments=[(0.2, 0.6), (1.0, 1.6)], tag="_synced_1s")
            self.assertEqual([f.name for f in files], ["ad_synced_1s.mp4", "ad_synced_1s_9x16.mp4"])
            info = [media.probe(f) for f in files]
            self.assertEqual([(i["width"], i["has_audio"]) for i in info], [(640, True), (202, True)])

            def stream(path, kind, entry):
                out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-count_frames", "-show_entries",
                                      f"stream={entry}", "-of", "csv=p=0", str(path)], capture_output=True, check=True)
                return float(out.stdout.decode().strip().strip(","))

            def colour(frame):
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(files[1]), "-vf",
                                      f"select=eq(n\\,{frame}),scale=1:1", "-frames:v", "1", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
                return "red" if raw[0] > raw[2] else "blue"

            for f in files:
                self.assertEqual(stream(f, "v", "nb_read_frames"), 25)                 # 10 + 15 frames
                self.assertAlmostEqual(stream(f, "a", "duration"), 1.0, delta=0.06)
            self.assertEqual([colour(n) for n in (0, 9, 10, 24)], ["red", "red", "blue", "blue"])


if __name__ == "__main__":
    unittest.main()
