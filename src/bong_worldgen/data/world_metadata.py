"""Generated world metadata. Edit this file for world-wide settings."""

from .world import WorldBounds

WORLD_VERSION = 1
WORLD_NAME = 'mofa_worldview_example'
SPAWN_ZONE = 'spawn'
WORLD_BOUNDS = WorldBounds(
    min_x=-10000.0, max_x=10000.0,
    min_z=-10400.0, max_z=10000.0,
)
WORLD_NOTES = ('Negative Z is north; positive Z is south.', 'This file is both a server zones example and a worldgen blueprint seed.', 'Unknown fields are intended for future worldgen/postprocess consumers.')
