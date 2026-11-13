"""Spotting a transcript where Whisper got stuck (the case from a real Hindi ad).

    python -m unittest discover -s tests -v
"""

import unittest

from fovea.utils.transcript import last_word_end, transcript_problems


def rows(*words):
    """One segment per call; each word is (text, start, end)."""
    return [{"segment_transcript": " ".join(w[0] for w in words),
             "words_data": [{"word": w, "start": s, "end": e} for w, s, e in words]}]


class TranscriptCheckTest(unittest.TestCase):
    def test_a_normal_transcript_has_no_problems(self):
        ok = rows(("I", 1.0, 1.3), ("hope", 1.35, 1.7), ("it's", 1.75, 2.0), ("shoes.", 2.05, 2.6))
        self.assertEqual(transcript_problems(ok), [])
        self.assertEqual(transcript_problems(rows(("Wait...", 0.0, 0.5), ("no,", 0.6, 0.8), ("no,", 0.9, 1.1), ("no!", 1.2, 1.4))), [])
        self.assertEqual(transcript_problems([]), [])
        self.assertEqual(last_word_end(ok), 2.6)
        self.assertEqual(last_word_end([]), 0.0)

    def test_the_stuck_character_from_the_hindi_ad_is_caught(self):
        stuck = rows(("\u092a\u094d\u0930\u094b\u0921\u093c\u093e\u0915" + "\u0964" * 90, 13.6, 14.4))
        problems = transcript_problems(stuck)
        self.assertTrue(any("repeats 90 times" in p for p in problems), problems)
        self.assertTrue(any("characters long" in p for p in problems), problems)

    def test_one_word_repeated_over_and_over_is_caught(self):
        looped = rows(*[("the", i * 0.2, i * 0.2 + 0.1) for i in range(12)])
        self.assertTrue(any("'the' repeats" in p for p in transcript_problems(looped)))


if __name__ == "__main__":
    unittest.main()
