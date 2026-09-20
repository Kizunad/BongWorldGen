"""Authored terrain profiles compiled into ordinary engine recipes.

Recipes use local coordinates here. Compilation translates geometric anchors
to the zone center; noise always uses world coordinates in the engine.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from ..data.world import ZoneDefinition
from ..engine import MountainRange, NoiseLayer, Point, TerrainRecipe


PROFILE_RECIPES = MappingProxyType({
    "spawn_plain": TerrainRecipe(
        name="spawn_plain", base_height=70.0, sea_level=61.0,
        base_noise=(
            NoiseLayer(scale=460.0, amplitude=3.0, octaves=3),
            NoiseLayer(scale=95.0, amplitude=0.7, octaves=2, seed_offset=11),
        ),
    ),
    "broken_peaks": TerrainRecipe(
        name="broken_peaks", base_height=95.0, sea_level=61.0,
        base_noise=(
            NoiseLayer(kind="ridge", scale=160.0, amplitude=66.0, seed_offset=23),
            NoiseLayer(scale=64.0, amplitude=12.0, octaves=3, seed_offset=29),
        ),
        mountains=(MountainRange(
            path=(Point(-0.32, -0.32), Point(0.0, 0.0), Point(0.3, 0.22)),
            width=0.13, height=75.0,
            roughness=NoiseLayer(kind="ridge", scale=180.0, seed_offset=37),
        ),),
    ),
    "waste_plateau": TerrainRecipe(
        name="waste_plateau", base_height=119.0, sea_level=61.0,
        base_noise=(
            NoiseLayer(scale=720.0, amplitude=5.0, octaves=2, seed_offset=41),
            NoiseLayer(scale=110.0, amplitude=1.5, octaves=2, seed_offset=43),
        ),
    ),
})


def recipe_for_zone(zone: ZoneDefinition) -> TerrainRecipe:
    """Compile fractional local anchors to world units; reject missing profiles."""

    try:
        template = PROFILE_RECIPES[zone.terrain_profile]
    except KeyError as exc:
        raise ValueError(f"unimplemented terrain profile {zone.terrain_profile!r}") from exc
    scale = min(zone.size_x, zone.size_z)

    def point(value: Point) -> Point:
        return Point(zone.center_x + value.x * zone.size_x, zone.center_z + value.z * zone.size_z)

    return replace(
        template,
        mountains=tuple(replace(
            mountain, path=tuple(point(p) for p in mountain.path), width=mountain.width * scale,
        ) for mountain in template.mountains),
    )
