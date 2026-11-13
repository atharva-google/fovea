"""Tests of the story-map logic that needs no video, no Gemini and no third-party packages.

    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from fovea.story.report import format_cuts, format_story, make_cuts
from fovea.story.select import Piece, merge_pieces, select_units
from fovea.story import store
from fovea.story.store import fingerprint
from fovea.story.story_map import PROMPT_VERSION, clean_story_map, extract_json, story_prompt
from fovea.story.units import Unit, build_units

# The example from the brief: setup line, reaction + punchline, silent close-up of the sweater.
SHOTS = [(0.0, 4.0), (4.0, 8.0), (8.0, 11.0)]
WORDS = [
    {"start": 1.0, "end": 1.3, "word": "I"}, {"start": 1.35, "end": 1.7, "word": "hope"},
    {"start": 1.75, "end": 2.0, "word": "it's"}, {"start": 2.05, "end": 2.6, "word": "shoes."},
    {"start": 5.5, "end": 5.7, "word": "A"}, {"start": 5.75, "end": 6.5, "word": "sweater."},
]


def example():
    units = build_units(SHOTS, WORDS, lambda a, b, s: {"speaking": 0.9 if s < 2 else 0.0, "people": 1, "ocr": []})
    speech = [u for u in units if u.kind == "speech"]
    setup, punch = speech
    sweater = [u for u in units if u.shot == 2]
    raw = {
        "summary": "She hopes for shoes and gets a sweater.",
        "units": [{"id": u.id, "importance": 9 if u.kind == "speech" else (3 if u.shot == 2 else 1),
                   "kind": "dialogue" if u.kind == "speech" else ("visual" if u.shot == 2 else "filler"),
                   "shows": "the sweater" if u.shot == 2 else "her", "note": "n"} for u in units],
        "needs": [{"unit": punch.id, "requires": [setup.id], "why": "punchline needs setup"}],
        "cutaways": [{"speech": punch.id, "pictures": [sweater[0].id], "fit": 9, "why": "show the gift"}],
        "hook": {"units": [], "why": ""},
        "moments": [],
    }
    smap, issues = clean_story_map(raw, units)
    assert not issues, issues
    return units, smap, setup, punch, sweater


class UnitsTest(unittest.TestCase):
    def test_units_are_exact_ordered_and_stay_inside_their_shot(self):
        units, _, setup, punch, sweater = example()
        self.assertEqual([u.id for u in units], [f"U{i + 1:02d}" for i in range(len(units))])
        self.assertEqual(units, sorted(units, key=lambda u: u.start))
        for u in units:
            a, b = SHOTS[u.shot]
            self.assertGreaterEqual(u.start, a)
            self.assertLessEqual(u.end, b)
            self.assertGreater(u.dur, 0)
        self.assertEqual((setup.text, setup.shot), ("I hope it's shoes.", 0))
        self.assertEqual((punch.text, punch.shot), ("A sweater.", 1))
        self.assertLessEqual(setup.start, 1.0)   # the whole line is inside its unit
        self.assertGreaterEqual(setup.end, 2.6)
        self.assertTrue(sweater and all(u.kind == "picture" for u in sweater))
        self.assertTrue(all(u.dur <= 2.5 + 1e-6 for u in units if u.kind == "picture"))

    def test_units_never_overlap(self):
        units = example()[0]
        for a, b in zip(units, units[1:]):
            self.assertLessEqual(a.end, b.start + 1e-6)

    def test_video_without_speech_is_all_picture_units(self):
        units = build_units(SHOTS, [])
        self.assertTrue(units and all(u.kind == "picture" for u in units))
        self.assertAlmostEqual(sum(u.dur for u in units), 11.0, places=2)

    def test_no_shots_no_units(self):
        self.assertEqual(build_units([], WORDS), [])

    def test_unit_json_round_trip(self):
        for u in example()[0]:
            self.assertEqual(Unit.from_dict(json.loads(json.dumps(u.to_dict()))), u)


class StoryMapParsingTest(unittest.TestCase):
    def test_extract_json_takes_the_last_block(self):
        text = 'thinking...\n```json\n{"summary": "old"}\n```\nmore\n```json\n{"summary": "new"}\n```'
        self.assertEqual(extract_json(text), {"summary": "new"})
        self.assertIsNone(extract_json("no json here"))

    def test_clean_drops_unknown_ids_fills_missing_and_clamps(self):
        units = example()[0]
        raw = {"units": [{"id": units[0].id, "importance": 99, "kind": "visual"},
                         {"id": "U99", "importance": 5}, {"id": units[1].id, "importance": "high"}],
               "needs": [{"unit": units[0].id, "requires": ["U99", units[1].id]}, {"unit": "U99", "requires": []}],
               "cutaways": [{"speech": units[0].id, "pictures": ["U99"]}],
               "hook": {"units": ["U99", units[0].id]}}
        smap, issues = clean_story_map(raw, units)
        self.assertEqual([m["id"] for m in smap["units"]], [u.id for u in units])
        self.assertEqual(smap["units"][0]["importance"], 10.0)
        self.assertEqual(smap["units"][1]["importance"], 0.0)
        self.assertEqual(smap["needs"], [{"unit": units[0].id, "requires": [units[1].id], "why": ""}])
        self.assertEqual(smap["cutaways"], [])
        self.assertEqual(smap["hook"]["units"], [units[0].id])
        self.assertTrue(any("U99" in i for i in issues))
        self.assertEqual(len([i for i in issues if "missing" in i]), len(units) - 2)

    def test_prompt_lists_every_unit(self):
        units = example()[0]
        prompt = story_prompt(units)
        self.assertNotIn("{units}", prompt)
        for u in units:
            self.assertIn(f"{u.id} | ", prompt)


class SelectorTest(unittest.TestCase):
    def test_synced_keeps_setup_and_punchline_and_is_shorter(self):
        units, smap, setup, punch, _ = example()
        pieces, stats = select_units(units, smap, 5.0, "synced")
        self.assertTrue(stats["ok"], stats)
        self.assertIn(setup.id, stats["units"])
        self.assertIn(punch.id, stats["units"])
        self.assertLess(stats["total"], 11.0 * 0.95)
        for p in pieces:
            self.assertEqual(p.video_start, p.audio_start)   # synced: sound stays with its picture
        self.assertAlmostEqual(stats["total"], sum(p.length for p in pieces), places=2)

    def test_story_is_kept_even_when_it_is_over_the_target(self):
        units, smap, setup, punch, _ = example()
        pieces, stats = select_units(units, smap, 1.0, "synced")
        self.assertEqual(stats["units"], [setup.id, punch.id])
        self.assertGreater(stats["total"], 1.0)
        self.assertTrue(any("over the 1s target" in n for n in stats["notes"]), stats["notes"])
        self.assertTrue(stats["ok"])

    def test_a_low_scored_unit_that_another_needs_is_kept_with_it(self):
        units, smap, setup, punch, _ = example()
        for m in smap["units"]:
            if m["id"] == setup.id:
                m["importance"] = 0.0   # Gemini scored the setup as filler but said the punchline needs it
        _, stats = select_units(units, smap, 5.0, "synced")
        self.assertIn(setup.id, stats["units"])
        self.assertIn(punch.id, stats["units"])

    def test_cutaway_shows_the_sweater_over_the_line(self):
        units, smap, setup, punch, sweater = example()
        pieces, stats = select_units(units, smap, 5.0, "cutaway")
        over = [p for p in pieces if "<-" in p.units]
        self.assertEqual(len(over), 1, pieces)
        p = over[0]
        self.assertEqual(p.units, f"{punch.id}<-{sweater[0].id}")
        self.assertAlmostEqual(p.video_start, sweater[0].start, places=3)       # picture: the sweater
        self.assertGreaterEqual(p.audio_start, punch.start - 1e-6)              # sound: the line
        self.assertLessEqual(p.audio_start + p.length, punch.end + 1e-6)
        self.assertNotIn(sweater[0].id, stats["units"])                         # it takes no time of its own

    def test_never_cuts_away_to_someone_visibly_talking(self):
        units, smap, setup, punch, _ = example()
        smap["cutaways"] = [{"speech": punch.id, "pictures": [setup.id], "fit": 9, "why": ""}]
        pieces, _ = select_units(units, smap, 5.0, "cutaway")
        self.assertFalse([p for p in pieces if "<-" in p.units])

    def test_fills_toward_the_target_with_optional_units(self):
        units, smap, *_ = example()
        short = select_units(units, smap, 4.0, "synced", cut_penalty=0.0)[1]["total"]
        longer = select_units(units, smap, 8.0, "synced", cut_penalty=0.0)[1]["total"]
        self.assertGreater(longer, short)
        self.assertLessEqual(longer, 8.0 * 1.15 + 0.5)

    def test_hook_moves_to_the_front_only_when_it_stands_alone(self):
        units, smap, setup, punch, sweater = example()
        smap["hook"] = {"units": [punch.id], "why": ""}       # depends on the setup: must not move
        self.assertEqual(select_units(units, smap, 3.0, "synced")[1]["units"][0], setup.id)
        for m in smap["units"]:
            if m["id"] == sweater[0].id:
                m["importance"] = 9.0
        smap["hook"] = {"units": [sweater[0].id], "why": ""}
        self.assertEqual(select_units(units, smap, 6.0, "synced")[1]["units"][0], sweater[0].id)
        self.assertNotEqual(select_units(units, smap, 6.0, "synced", use_hook=False)[1]["units"][0], sweater[0].id)

    def test_keeping_everything_is_reported_as_not_usable(self):
        units, smap, *_ = example()
        for m in smap["units"]:
            m["importance"] = 10.0
        _, stats = select_units(units, smap, 5.0, "synced")
        self.assertFalse(stats["ok"])
        self.assertTrue(any("not meaningfully shorter" in n for n in stats["notes"]))

    def test_same_input_same_cut(self):
        units, smap, *_ = example()
        self.assertEqual(select_units(units, smap, 6.0, "cutaway"), select_units(units, smap, 6.0, "cutaway"))

    def test_merge_joins_continuous_pieces_and_bridges_short_pauses(self):
        merged = merge_pieces([Piece(0.0, 0.0, 1.0, "U01"), Piece(1.0, 1.0, 1.0, "U02"),      # continuous
                               Piece(2.3, 2.3, 1.0, "U03"),                                   # 0.3 s pause: bridged
                               Piece(4.0, 4.0, 1.0, "U04"),                                   # 0.7 s gap: a real cut
                               Piece(9.0, 5.0, 1.0, "U05<-U09")])                             # other picture: a cut
        self.assertEqual([(p.video_start, p.audio_start, p.length) for p in merged],
                         [(0.0, 0.0, 3.3), (4.0, 4.0, 1.0), (9.0, 5.0, 1.0)])
        self.assertEqual(merged[0].units, "U01 U02 U03")


class StoreTest(unittest.TestCase):
    def test_same_bytes_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d, "a.mp4"), Path(d, "renamed.mp4"), Path(d, "c.mp4")
            a.write_bytes(b"video bytes")
            b.write_bytes(b"video bytes")
            c.write_bytes(b"other bytes")
            self.assertEqual(fingerprint(a), fingerprint(b))
            self.assertNotEqual(fingerprint(a), fingerprint(c))

    def test_save_then_load_returns_the_same_data_in_the_videos_own_folder(self):
        units, smap, *_ = example()
        with tempfile.TemporaryDirectory() as d:
            prep = Path(d, "my_ad", "prep")
            self.assertIsNone(store.load(prep, "abc"))
            store.save(prep, "abc", units, {"story_map": smap, "raw_answer": "raw", "model": "m"}, "my_ad.mp4")
            self.assertEqual(sorted(p.name for p in prep.iterdir()), ["story_map.json", "units.json"])
            got_units, record = store.load(prep, "abc")
            self.assertEqual(got_units, units)
            self.assertEqual(record["story_map"], smap)
            self.assertEqual(record["raw_answer"], "raw")

    def test_a_different_file_with_the_same_name_does_not_inherit_the_story_map(self):
        units, smap, *_ = example()
        with tempfile.TemporaryDirectory() as d:
            store.save(d, "abc", units, {"story_map": smap})
            self.assertIsNone(store.load(d, "other-fingerprint"))

    def test_entry_from_an_older_prompt_is_not_reused(self):
        units, smap, *_ = example()
        with tempfile.TemporaryDirectory() as d:
            folder = store.save(d, "abc", units, {"story_map": smap})
            record = json.loads((folder / "story_map.json").read_text())
            record["prompt_version"] = PROMPT_VERSION - 1
            (folder / "story_map.json").write_text(json.dumps(record))
            self.assertIsNone(store.load(d, "abc"))

    def test_broken_entry_counts_as_missing(self):
        units, smap, *_ = example()
        with tempfile.TemporaryDirectory() as d:
            folder = store.save(d, "abc", units, {"story_map": smap})
            (folder / "units.json").write_text("{not json")
            self.assertIsNone(store.load(d, "abc"))


class ReportTest(unittest.TestCase):
    def test_cuts_for_every_duration_and_mode_and_too_long_targets_are_skipped(self):
        units, smap, *_ = example()
        cuts = make_cuts(units, smap, [5.0, 30.0], ["synced", "cutaway"])
        self.assertEqual(list(cuts), ["5s_synced", "5s_cutaway", "30s_synced", "30s_cutaway"])
        self.assertTrue(cuts["5s_synced"]["pieces"])
        self.assertEqual(cuts["30s_synced"]["pieces"], [])
        self.assertIn("only 11.0s long", cuts["30s_synced"]["stats"]["notes"][0])
        json.dumps(cuts)                                    # everything in it can be written to a file
        text = format_story(units, smap) + format_cuts(cuts)
        self.assertIn("SUMMARY: She hopes for shoes", text)
        self.assertIn("=== 5s_cutaway", text)
        self.assertIn("sound from", text)


if __name__ == "__main__":
    unittest.main()
