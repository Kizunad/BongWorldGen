"""Pure procedural terrain engine.

The engine knows nothing about Minecraft blocks, zones, raster manifests, or
the Bong server. It only evaluates a typed terrain recipe into scalar fields.
"""

from .models import Basin, Heightfield, MountainRange, NoiseLayer, Point, River, TerrainRecipe
from .pipeline import generate_heightfield

__all__ = [
    "Basin",
    "Heightfield",
    "MountainRange",
    "NoiseLayer",
    "Point",
    "River",
    "TerrainRecipe",
    "generate_heightfield",
]
