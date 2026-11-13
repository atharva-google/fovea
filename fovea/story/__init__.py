"""Cutdowns from a stored story map.

    units.py      chop the video into small exact pieces ("units"): phrases and stretches without speech
    story_map.py  ONE Gemini call that scores and links the units -> the story map
    store.py      units + story map saved under the video file's fingerprint (same file = same data)
    select.py     pick units for a target duration (code only: instant, repeatable, no Gemini)

Nothing here imports numpy / torch / google-genai at module level, so ``units``, ``select`` and
``store`` can be used (and tested) on a machine without the pipeline's dependencies.
"""
