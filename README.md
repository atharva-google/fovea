# Fovea

Turn a horizontal ad into other aspect ratios and shorter versions. Gemini does the understanding
(what happens, who is in it, what is said, where to look); code does what has to be exact (times,
cutting, cropping).

**Status: first step.** So far: shots, transcript, and the base analysis. No rendering yet.
The earlier, detector-based project is [fovea-free](https://github.com/atharva-google/fovea-free).

## Run it

Needs Python 3.10+, ffmpeg and a Gemini API key. No GPU.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
python -m fovea analyze my_ad.mp4
```

It prints the cast, every shot (with what the cropped picture must show, and when that changes) and
every piece of sound. Running it again reuses what is stored; `--refresh` asks Gemini again.

## What happens

```
video ──┬── sound ── transcript (Whisper, word times) ──┐
        └── proxy ── shots (PySceneDetect) ─────────────┴── base analysis (ONE Gemini call)
```

The two branches run at the same time. Code then cuts the video into **video units** (one per shot)
and **audio units** (a few spoken words, or the sound between them). Gemini watches the video once
and describes every unit by id; it never writes a timestamp.

## Where files go

Everything a run creates stays in the project folder, under `fovea_data/` (ignored by git):

```
fovea_data/<video name>/
├── job.json                which file this folder belongs to
├── input/                  a copy of the video
├── prep/                   proxy.mp4, audio.wav, shots.csv, units.json, log.json
├── audio/transcript.csv    one row per word, with its times
└── framing/framing.json    per shot: where the picture is centred, over time
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
├── analysis.py     the base analysis: prompt, checking the answer, storage
├── framing.py      where the picture is centred at a given time
├── schema.py       the shape Gemini answers in
└── gemini.py       one question to Gemini about one video
tests/              python -m unittest discover -s tests
```
