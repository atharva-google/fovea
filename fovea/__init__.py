"""Fovea - smart reframing of horizontal videos into vertical, square, or any aspect ratio."""

__version__ = "0.1.0"

from .config import PipelineConfig  # noqa: E402


def run(*args, **kwargs):
    """Lazy wrapper around :func:`fovea.pipeline.run` (keeps ``import fovea`` fast)."""
    from .pipeline import run as _run

    return _run(*args, **kwargs)


__all__ = ["run", "PipelineConfig", "__version__"]
