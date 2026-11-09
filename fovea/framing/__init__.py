"""Decide *where* the crop window goes: focal point search + camera motion planning."""

from .camera import plan_camera
from .focal import find_focal_points_viterbi
from .geometry import CropGeometry, OutputTarget
from .rules import find_focal_points_rules

__all__ = ["plan_camera", "find_focal_points_viterbi", "find_focal_points_rules", "CropGeometry", "OutputTarget"]
