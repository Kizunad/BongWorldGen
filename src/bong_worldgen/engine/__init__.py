"""Pure procedural terrain engine.

The engine knows nothing about Minecraft blocks, zones, raster manifests, or
the Bong server. It only evaluates a typed terrain recipe into scalar fields.
"""

from .models import (
    DEFAULT_RIVERBED_MATERIALS,
    Basin,
    CaveNetwork,
    Heightfield,
    MountainRange,
    NoiseLayer,
    Point,
    Plateau,
    River,
    TerrainRecipe,
    UndergroundBlock,
)
from .pipeline import finish_heightfield, generate_heightfield, sample_surface

__all__ = [
    "Basin",
    "CaveNetwork",
    "DEFAULT_RIVERBED_MATERIALS",
    "Heightfield",
    "MountainRange",
    "NoiseLayer",
    "Point",
    "Plateau",
    "River",
    "TerrainRecipe",
    "UndergroundBlock",
    "generate_heightfield",
    "sample_surface",
    "finish_heightfield",
]
