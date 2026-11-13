"""The ffmpeg side of rendering a cut, run against a real (generated) video. Needs ffmpeg, nothing else.

    python -m unittest discover -s tests -v
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fovea.story.edit import audio_graph, decode_args, encode_args, frame_plan, plan_seconds

W, H, FPS, SECONDS = 160, 90, 25, 6
FRAME = W * H * 3
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration,width,height",
               "-of", "json", str(path)]).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


class PlanTest(unittest.TestCase):
    def test_pieces_become_whole_frames_inside_the_source(self):
        pieces = [{"video_start": 0.92, "audio_start": 0.92, "length": 1.8},
                  {"video_start": 8.0, "audio_start": 5.42, "length": 1.2},
                  {"video_start": 10.9, "audio_start": 10.9, "length": 5.0}]     # runs past the end
        plan = frame_plan(pieces, 25.0, total_frames=275)
        self.assertEqual(plan, [(23, 45, 0.92), (200, 30, 5.42), (272, 3, 10.9)])
        self.assertAlmostEqual(plan_seconds(plan, 25.0), 78 / 25)

    def test_audio_graph_cuts_every_piece_to_its_picture_length(self):
        graph = audio_graph([(23, 45, 0.92), (200, 30, 5.42)], 25.0)
        self.assertIn("asplit=2[s0][s1]", graph)
        self.assertIn("[s0]atrim=start=0.920000:duration=1.800000", graph)
        self.assertIn("[s1]atrim=start=5.420000:duration=1.200000", graph)     # sound from elsewhere than the picture
        self.assertTrue(graph.endswith("[a0][a1]concat=n=2:v=0:a=1[aout]"))
        with self.assertRaises(ValueError):
            audio_graph([], 25.0)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
class FfmpegTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.src = Path(cls.tmp, "src.mp4")
        # every frame is different (moving test pattern + frame counter); keyframe only every 2 s, so
        # most seeks land between keyframes; a tone for sound
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={SECONDS}",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}", "-c:v", "libx264", "-g", str(FPS * 2),
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.src)])
        raw = run(["ffmpeg", "-v", "error", "-i", str(cls.src), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]).stdout
        cls.frames = [hashlib.md5(raw[i:i + FRAME]).hexdigest() for i in range(0, len(raw), FRAME)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_generated_video_is_what_the_tests_assume(self):
        self.assertEqual(len(self.frames), FPS * SECONDS)
        self.assertEqual(len(set(self.frames)), len(self.frames))       # all frames differ

    def test_decode_returns_exactly_the_asked_frames(self):
        for first, count in [(0, 5), (1, 3), (37, 20), (50, 1), (99, 30), (140, 10)]:
            raw = run(decode_args(self.src, first, count, FPS)).stdout
            got = [hashlib.md5(raw[i:i + FRAME]).hexdigest() for i in range(0, len(raw), FRAME)]
            self.assertEqual(got, self.frames[first:first + count], f"frames {first}..{first + count}")

    def test_decode_is_frame_exact_at_29_97_fps_too(self):
        src = Path(self.tmp, "ntsc.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=30000/1001:duration=8",
             "-c:v", "libx264", "-g", "90", "-pix_fmt", "yuv420p", str(src)])
        raw = run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]).stdout
        frames = [hashlib.md5(raw[i:i + FRAME]).hexdigest() for i in range(0, len(raw), FRAME)]
        fps = 30000 / 1001
        for first, count in [(1, 2), (29, 5), (100, 12), (233, 6)]:
            raw = run(decode_args(src, first, count, fps)).stdout
            got = [hashlib.md5(raw[i:i + FRAME]).hexdigest() for i in range(0, len(raw), FRAME)]
            self.assertEqual(got, frames[first:first + count], f"frames {first}..{first + count}")

    def test_a_cut_comes_out_with_the_planned_length_picture_and_sound(self):
        pieces = [{"video_start": 0.5, "audio_start": 0.5, "length": 1.0},
                  {"video_start": 4.0, "audio_start": 2.0, "length": 0.8},     # picture from 4 s, sound from 2 s
                  {"video_start": 3.0, "audio_start": 3.0, "length": 0.6}]
        plan = frame_plan(pieces, FPS, total_frames=len(self.frames))
        self.assertEqual([(f, c) for f, c, _ in plan], [(12, 25), (100, 20), (75, 15)])
        out = Path(self.tmp, "cut.mp4")
        enc = subprocess.Popen(encode_args(self.src, W, H, FPS, plan, out, ["-c:v", "libx264", "-preset", "veryfast"]),
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for first, count, _ in plan:
            enc.stdin.write(run(decode_args(self.src, first, count, FPS)).stdout)
        enc.stdin.close()
        err = enc.stderr.read().decode()
        self.assertEqual(enc.wait(), 0, err)

        info = probe(out)
        total = plan_seconds(plan, FPS)                                    # 60 frames = 2.4 s
        self.assertEqual(int(info["video"]["nb_frames"]), 60)
        self.assertEqual((info["video"]["width"], info["video"]["height"]), (W, H))
        self.assertAlmostEqual(float(info["video"]["duration"]), total, delta=0.05)
        self.assertAlmostEqual(float(info["audio"]["duration"]), total, delta=0.06)

    def test_a_cut_of_a_video_without_sound_has_no_audio_stream(self):
        plan = frame_plan([{"video_start": 1.0, "audio_start": 1.0, "length": 0.4}], FPS, len(self.frames))
        out = Path(self.tmp, "silent.mp4")
        enc = subprocess.Popen(encode_args(self.src, W, H, FPS, plan, out, ["-c:v", "libx264"], has_audio=False),
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        enc.stdin.write(run(decode_args(self.src, *plan[0][:2], FPS)).stdout)
        enc.stdin.close()
        self.assertEqual(enc.wait(), 0, enc.stderr.read().decode())
        self.assertEqual(list(probe(out)), ["video"])


if __name__ == "__main__":
    unittest.main()
