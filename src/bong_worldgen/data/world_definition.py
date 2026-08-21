"""Complete typed world definition assembled from metadata and zones."""

from .world import WorldDefinition
from .world_metadata import SPAWN_ZONE, WORLD_BOUNDS, WORLD_NAME, WORLD_NOTES, WORLD_VERSION
from .zones import ALL_ZONES

WORLD = WorldDefinition(
    version=WORLD_VERSION,
    name=WORLD_NAME,
    spawn_zone=SPAWN_ZONE,
    bounds=WORLD_BOUNDS,
    notes=WORLD_NOTES,
    zones=ALL_ZONES,
)

__all__ = ['WORLD']
