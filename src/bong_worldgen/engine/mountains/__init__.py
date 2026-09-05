"""Mountain Spine + Distance Field + Ridged Multifractal 山脉生成。"""

from .generator import apply_mountains, apply_mountains_with_uplift
from .material_field import (
    MOUNTAIN_SURFACE_PALETTE,
    MountainMaterialField,
    mountain_surface_materials,
    sample_mountain_material_field,
)
from .peaks import PeakAnchor, apply_anisotropic_peaks, select_peak_anchors
from .ridged import sample_ridged_multifractal

__all__ = [
    "apply_mountains",
    "apply_mountains_with_uplift",
    "MOUNTAIN_SURFACE_PALETTE",
    "MountainMaterialField",
    "mountain_surface_materials",
    "sample_mountain_material_field",
    "PeakAnchor",
    "apply_anisotropic_peaks",
    "select_peak_anchors",
    "sample_ridged_multifractal",
]
