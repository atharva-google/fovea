"""Units, the base analysis (prompt, checking the answer, storage), the transcript check and the job
folder. No video model, Gemini or third-party package is needed; ffmpeg is used when installed.

    python -m unittest discover -s tests -v
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fovea import analysis, gemini, media, transcript
from fovea.framing import position_at
from fovea.job import Job
from fovea.units import build_audio_units, cut_units, focus_times

SHOTS = [(0.0, 4.0), (4.0, 8.3), (8.3, 11.0)]
WORDS = [
    {"word": "I", "start_s": 1.0, "end_s": 1.3}, {"word": "hope", "start_s": 1.35, "end_s": 1.7},
    {"word": "it's", "start_s": 1.75, "end_s": 2.0}, {"word": "shoes.", "start_s": 2.05, "end_s": 2.6},
    {"word": "Oh", "start_s": 5.0, "end_s": 5.28},                              # then a 0.22 s unspoken "aaa"
    {"word": "A", "start_s": 5.5, "end_s": 5.7}, {"word": "sweater.", "start_s": 5.75, "end_s": 6.5},
]


def tiles(units, total):
    edges = [(u.start, u.end) for u in units]
    return (abs(edges[0][0]) < 1e-6 and abs(edges[-1][1] - total) < 1e-6
            and all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(edges, edges[1:])) and all(e > s for s, e in edges))


def answer(audio, video):
    """A good answer for the example."""
    line = next(a for a in audio if a.text == "A sweater.")
    setup = next(a for a in audio if a.text.startswith("I hope"))
    return {
        "summary": "She hopes for shoes and gets a sweater.",
        "cast": [{"id": "P1", "real_name": "", "ad_name": "Mira", "description": "woman, about 30, red kurta"}],
        "video_units": [
            {"id": "V01", "shows": "a woman walks in with a gift", "cast": ["P1"], "setting": "living room",
             "objects": ["gift box"], "text": [], "needs": [],
             "focus": [{"at": "", "left": 300, "right": 500, "on": "P1's face"}]},
            {"id": "V02", "shows": "her face falls", "cast": ["P1"], "setting": "living room", "objects": ["gift box"],
             "text": [], "needs": ["V01"],
             "focus": [{"at": "", "left": 400, "right": 560, "on": "P1's face"},
                       {"at": line.id, "left": 640, "right": 800, "on": "the gift box"},
                       {"at": "7.00", "left": 700, "right": 860, "on": ""}]},
            {"id": "V03", "shows": "a red sweater in the box", "cast": [], "setting": "living room",
             "objects": ["sweater"], "text": ["For you"], "needs": [line.id],
             "focus": [{"at": "", "left": 450, "right": 650, "on": "the sweater"}]},
        ],
        "audio_units": [{"id": a.id, "heard": a.text or "(silence)", "kind": "dialogue_onscreen" if a.text else "other",
                         "speaker": "P1" if a.text else "", "needs": [setup.id] if a.id == line.id else []}
                        for a in audio],
    }


class UnitsTest(unittest.TestCase):
    def test_both_tracks_cover_the_whole_video(self):
        audio, video = cut_units(SHOTS, WORDS)
        self.assertTrue(tiles(audio, 11.0) and tiles(video, 11.0))
        self.assertEqual([v.id for v in video], ["V01", "V02", "V03"])
        self.assertEqual([a.text for a in audio if a.kind == "speech"], ["I hope it's shoes.", "Oh", "A sweater."])

    def test_a_short_audible_pause_is_its_own_unit(self):
        audio = build_audio_units(WORDS, SHOTS)
        oh = next(i for i, a in enumerate(audio) if a.text == "Oh")
        self.assertEqual((audio[oh + 1].kind, audio[oh + 2].text), ("gap", "A sweater."))

    def test_a_phrase_across_a_shot_cut_stays_one_unit_over_both_shots(self):
        words = [{"word": "hello", "start": 3.6, "end": 3.95}, {"word": "there", "start": 3.98, "end": 4.4}]
        audio, _ = cut_units(SHOTS, words)
        phrase = next(a for a in audio if a.kind == "speech")
        self.assertEqual((phrase.text, phrase.over), ("hello there", ["V01", "V02"]))

    def test_silence_is_split_at_shot_cuts_and_bad_timings_still_tile(self):
        self.assertTrue({4.0, 8.3} <= {a.start for a in build_audio_units([], SHOTS)})
        bad = [{"word": "one.", "start": 1.0, "end": 2.0}, {"word": "two.", "start": 1.5, "end": 2.5},
               {"word": "late", "start": 10.5, "end": 14.0}]
        self.assertTrue(tiles(build_audio_units(bad, SHOTS), 11.0))
        self.assertEqual(cut_units([], WORDS), ([], []))

    def test_focus_moments_are_the_start_every_second_and_a_late_last_frame(self):
        self.assertEqual(focus_times(0.0, 5.0), [0.0, 1.0, 2.0, 3.0, 4.0, 4.96])
        self.assertEqual(focus_times(4.0, 8.3), [4.0, 5.0, 6.0, 7.0, 8.0])        # 8.26 is too close to 8.0
        self.assertEqual(focus_times(2.0, 2.3), [2.0])
        self.assertEqual(focus_times(2.0, 2.8), [2.0, 2.76])


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.audio, self.video = cut_units(SHOTS, WORDS)

    def test_the_prompt_lists_every_unit_and_the_focus_moments(self):
        text = analysis.build_prompt(self.audio, self.video)
        self.assertNotIn("{table}", text)
        self.assertNotIn("```json", text)
        self.assertIn("V02 | 4.00-8.30s | marks: 4.00, 5.00, 6.00, 7.00, 8.00", text)
        self.assertIn("ONE THIRD of the frame's width", text)
        self.assertIn("merely present", text)
        self.assertIn('| over V02 | rough transcript: "A sweater."', text)
        self.assertIn("(no words transcribed)", text)
        self.assertIn("```json", analysis.build_prompt(self.audio, self.video, agentic=True))

    def test_a_good_answer_is_split_into_log_and_framing(self):
        described, framing, issues = analysis.clean(answer(self.audio, self.video), self.audio, self.video)
        self.assertEqual(issues, [])
        self.assertNotIn("focus", described["video_units"][0])
        self.assertEqual(described["cast"][0]["ad_name"], "Mira")
        line = next(a for a in self.audio if a.text == "A sweater.")
        v2 = framing["shots"][1]["points"]
        self.assertEqual([(p["t"], p["move"], p["x"], p["on"]) for p in v2],
                         [(4.0, "cut", 48.0, "P1's face"),              # opens on her face
                          (line.start, "cut", 72.0, "the gift box"),    # jumps when the line starts (Whisper's time)
                          (7.0, "glide", 78.0, "the gift box")])        # then follows it to the 7 s mark
        self.assertEqual((v2[1]["left"], v2[1]["right"]), (64.0, 80.0))
        self.assertEqual(len(framing["shots"][0]["points"]), 1)

    def test_focus_entries_that_cannot_be_placed_are_left_out_and_reported(self):
        raw = answer(self.audio, self.video)
        first_sound = self.audio[0].id                                  # plays over V01, not over V03
        raw["video_units"][0]["focus"] = []
        raw["video_units"][1]["focus"] = [{"at": "6.9", "left": 900, "right": 700, "on": "late"}]
        raw["video_units"][2]["focus"] = [{"at": "", "left": 1500, "right": -20, "on": "wide"},
                                          {"at": first_sound, "left": 1, "right": 2, "on": "x"},
                                          {"at": "soon", "left": 1, "right": 2, "on": "x"}, {"at": "", "left": "?"}]
        _, framing, issues = analysis.clean(raw, self.audio, self.video)
        middle = framing["shots"][0]["points"]
        self.assertEqual([(p["t"], p["x"]) for p in middle], [(0.0, 50.0)])
        late = framing["shots"][1]["points"]                             # nothing for the start: opens on it
        self.assertEqual([(p["t"], p["move"], p["left"], p["right"]) for p in late],
                         [(4.0, "cut", 70.0, 90.0), (7.0, "glide", 70.0, 90.0)])
        self.assertEqual([(p["left"], p["right"]) for p in framing["shots"][2]["points"]], [(0.0, 100.0)])
        self.assertEqual(len([i for i in issues if "focus" in i]), 4)

    def test_the_picture_jumps_on_a_cut_and_moves_evenly_on_a_glide(self):
        points = [{"t": 4.0, "move": "cut", "x": 40.0, "left": 30.0}, {"t": 5.5, "move": "cut", "x": 70.0, "left": 60.0},
                  {"t": 7.0, "move": "glide", "x": 80.0, "left": 70.0}]
        self.assertEqual([position_at(points, t) for t in (4.0, 5.49, 5.5, 6.25, 7.0, 8.2)],
                         [40.0, 40.0, 70.0, 75.0, 80.0, 80.0])
        self.assertEqual(position_at(points, 6.25, "left"), 65.0)
        self.assertEqual(position_at(points, 0.0), 40.0)

    def test_ids_cast_speakers_kinds_and_needs_are_checked(self):
        raw = answer(self.audio, self.video)
        raw["video_units"][0].update(cast=["P1", "P9"], needs=["V03", "Z1"])
        raw["video_units"] = raw["video_units"][:2] + [{"id": "V99"}]
        speech = next(i for i, a in enumerate(self.audio) if a.kind == "speech")
        raw["audio_units"][speech].update(kind="shouting", speaker="P9")
        raw["audio_units"][0]["kind"] = ""
        described, framing, issues = analysis.clean(raw, self.audio, self.video)
        self.assertEqual((described["video_units"][0]["cast"], described["video_units"][0]["needs"]), (["P1"], []))
        self.assertEqual(described["video_units"][2]["shows"], "")                    # V03 was missing
        self.assertEqual(len(framing["shots"]), 3)
        self.assertEqual((described["audio_units"][speech]["kind"], described["audio_units"][speech]["speaker"]),
                         ("dialogue_onscreen", "other"))
        self.assertEqual(described["audio_units"][0]["kind"], "other")
        self.assertGreaterEqual(len(issues), 6)

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
            self.assertIn("focus:   4.00s opens on   48% (40-56)  P1's face", text)
            self.assertIn("glides to  78% (70-86)  the gift box", text)
            self.assertIn("dialogue_onscreen", text)

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
