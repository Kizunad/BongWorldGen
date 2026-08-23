"""Typed terrain data, kept separate from the engine and output adapters."""

from .recipes import DEFAULT_RECIPE
from .world import PoiDefinition, WorldBounds, WorldDefinition, ZoneDefinition
from .world_definition import WORLD
from .wilderness import WILDERNESS_BY_ID, WILDERNESS_PALETTE, WildernessType

__all__ = [
    "DEFAULT_RECIPE",
    "WORLD",
    "PoiDefinition",
    "WorldBounds",
    "WorldDefinition",
    "ZoneDefinition",
    "WildernessType",
    "WILDERNESS_BY_ID",
    "WILDERNESS_PALETTE",
]
