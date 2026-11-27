# Fovea

Turn a horizontal ad into other aspect ratios and shorter versions. Gemini does the understanding
(what happens, who is in it, what is said, where to look); code does what has to be exact (times,
cutting, cropping).

**Status:** the log (Gemini watches the video once), rendering in other shapes, and synced edits
(shorter versions). Not yet: cutaway edits, user prompts, a frontend.
The earlier, detector-based project is [fovea-free](https://github.com/atharva-google/fovea-free).

## Run it

Needs Python 3.10+, ffmpeg and a Gemini API key. No GPU.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
python -m fovea log my_ad.mp4
```

It prints the cast, every shot (with what the cropped picture must show, and when that changes) and
every piece of sound. Running it again reuses what is stored; `--refresh` asks Gemini again.
`--gemini-fps` sets how many frames per second Gemini looks at (default 4; its own default of 1 misses
shots shorter than a second). `--thinking minimal|low|medium|high` sets how hard Gemini thinks before it answers (default: the model
decides); use it with `--refresh` to compare levels on a video that is already analysed.

## Render

```bash
python -m fovea render my_ad.mp4                      # 9:16, 1:1, 4:3 and 3:4
python -m fovea render my_ad.mp4 --formats 9:16 4:5   # any WIDTH:HEIGHT
```

Each format is a crop of the full height that follows the focus line: it jumps where the line jumps
and glides where it glides. Moves smaller than `--dead-zone` (3% of the width) are not followed. All
formats come out of one ffmpeg run, in `fovea_data/<video name>/outputs/`. A video that has not been
analysed yet is analysed first; a format as wide as the video or wider is skipped.

## Edit

```bash
python -m fovea edit my_ad.mp4 --seconds 15           # a shorter version of about 15 s
```

Gemini reads the log (text only, it does not watch the video again) and picks the audio units to
keep; each sound plays with its own picture. The length is a guide: the edit may run longer when the
story needs it. Code adds what a kept unit depends on and reports what it changed. The result is
stored as `edit/edit_synced_15s.json` and rendered in the video's own shape and in every format
(`outputs/<video name>_synced_15s.mp4`, `..._synced_15s_9x16.mp4`, ...). `--refresh` asks for the
edit again; the log is kept.

## Music (experiment, branch `feat/music`)

Gemini is the only model. When it watches the video it also notes, for every audio unit, the music
under it (`none`, `score`, `song`), which part of the track that is (`intro`, `build`, `peak`,
`break`, `outro`), words that cannot be moved apart from the music (`with_music`), and sounds made by
something seen (`sfx`). The edit uses this:

- Gemini is told to cut where there is no music or where one part of the track ends and the next begins.
- A sung line, or words timed to the music, is kept whole (code adds the rest of the line).
- Every cut is judged and printed: `no music`, `between music parts`, or `inside the music`.
- The sound is blended over 60 ms at every cut. `--crossfade 0` makes hard cuts and names the files
  `..._hardcuts`, so both can be listened to side by side.

The beat itself is not analysed, so a cut between parts can still be off the beat.

## What happens

```
video ──┬── sound ── transcript (Whisper, word times) ──┐
        └── proxy ── shots (PySceneDetect) ─────────────┴── log (ONE Gemini call) ──┬── render
                                                                           └── edit (one Gemini call, text only) ── render
```

The two branches run at the same time. Code then cuts the video into **video units** (one per shot)
and **audio units** (a few spoken words, or the sound between them). Gemini watches the video once
and describes every unit by id. It copies each unit's time range to keep its place, but every time used
later comes from code.

Gemini is asked shot by shot, with the sounds of each shot listed under it. For every sound it also
says where the cropped picture should be centred while that sound plays. Because each position
belongs to a sound, code knows exactly when it applies: the picture jumps when a new person starts to
speak, and glides when a subject moves.

## Where files go

Everything a run creates stays in the project folder, under `fovea_data/` (ignored by git):

```
fovea_data/<video name>/
├── job.json                which file this folder belongs to
├── input/                  a copy of the video
├── prep/                   proxy.mp4, audio.wav, shots.csv, units.json, log.json
├── audio/transcript.csv    one row per word, with its times
├── framing/framing.json    per shot: where the picture is centred, over time
├── edit/                   edit_synced_15s.json: the units a shorter version keeps
└── outputs/                the rendered videos, one per format
```

`log.json` holds the cast and the description of every unit. `framing.json` is separate, so framing
can change later without touching the log. The Whisper model is downloaded to `weights/`.

## Code

```
fovea/
├── cli.py          the command line
├── pipeline.py     runs the steps
├── config.py       every setting
├── job.py          the folder of one video
├── media.py        ffmpeg: probe, proxy, sound
├── shots.py        PySceneDetect -> shots.csv
├── transcript.py   Whisper -> transcript.csv
├── units.py        video units and audio units
├── analysis.py     the log: prompt, checking the answer, storage
├── edit.py         the edit: prompt, checking the answer, storage
├── framing.py      where the picture is centred at a given time; the path of a crop
├── render.py       the video in other shapes (one ffmpeg run)
├── schema.py       the shapes Gemini answers in, and what every field means
└── gemini.py       one question to Gemini, about a video or in text only
tests/              python -m unittest discover -s tests
```
