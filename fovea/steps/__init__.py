"""The 4 steps of the Fovea pipeline. Each module has one ``run()`` function.

    prep    -> visual (per shard) -> render        (videos)
            \\-> audio (optional, in parallel)      (transcript)

The local runner (fovea/pipeline.py) and the backend worker
(backend/worker.py) both just call these ``run()`` functions.
"""

from . import audio, prep, render, visual

__all__ = ["prep", "visual", "audio", "render"]
