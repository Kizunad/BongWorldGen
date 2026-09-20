"""Authored terrain profiles compiled into ordinary engine recipes.

Recipes use local coordinates here. Compilation translates geometric anchors
to the zone center; noise always uses world coordinates in the engine.
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

from ..data.world import ZoneDefinition
from ..engine import Basin, MountainRange, NoiseLayer, Plateau, Point, TerrainRecipe


def _ring(radius: float) -> tuple[Point, ...]:
    return tuple(Point(radius * math.cos(i * math.tau / 24), radius * math.sin(i * math.tau / 24))
                 for i in range(25))


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
    "spring_marsh": TerrainRecipe(
        name="spring_marsh", base_height=64.0, sea_level=61.0,
        base_noise=(
            NoiseLayer(scale=135, amplitude=4.0, octaves=3, seed_offset=47),
            NoiseLayer(scale=34, amplitude=1.2, octaves=2, seed_offset=53),
        ),
        basins=(Basin(Point(0, 0), radius_x=0.29, radius_z=0.32, depth=6.0),),
    ),
    "rift_valley": TerrainRecipe(
        name="rift_valley", base_height=88.0, sea_level=61.0,
        base_noise=(NoiseLayer(scale=120, amplitude=5.0, octaves=3, seed_offset=59),),
        mountains=(MountainRange(
            path=(Point(0, -0.5), Point(0.035, 0), Point(-0.03, 0.5)),
            width=0.23, height=50, valley_depth=80,
            roughness_contrast=0.18,
            roughness=NoiseLayer(kind="ridge", scale=160, seed_offset=61),
        ),),
    ),
    "ash_dead_zone": TerrainRecipe(
        name="ash_dead_zone", base_height=73.0, sea_level=61.0,
        base_noise=(
            NoiseLayer(kind="ridge", scale=95, amplitude=17, seed_offset=67),
            NoiseLayer(kind="warp", scale=270, amplitude=7, octaves=3,
                       warp_strength=80, warp_scale=550, seed_offset=71),
        ),
        basins=(
            Basin(Point(-0.19, 0.12), radius_x=0.10, radius_z=0.13, depth=12),
            Basin(Point(0.18, -0.14), radius_x=0.12, radius_z=0.08, depth=10),
        ),
    ),
    "rift_mouth_barrens": TerrainRecipe(
        name="rift_mouth_barrens", base_height=82.0, sea_level=61.0,
        base_noise=(NoiseLayer(scale=35, amplitude=3, octaves=2, seed_offset=73),),
        basins=(Basin(Point(0, 0), radius_x=0.12, radius_z=0.18, depth=18),),
        mountains=(MountainRange(
            path=_ring(0.24), width=0.065, height=21, roughness_contrast=0.3,
            roughness=NoiseLayer(kind="ridge", scale=40, seed_offset=79),
        ),),
    ),
    "tribulation_scorch": TerrainRecipe(
        name="tribulation_scorch", base_height=88.0, sea_level=61.0,
        base_noise=(NoiseLayer(scale=85, amplitude=2.5, octaves=3, seed_offset=83),),
        basins=(Basin(Point(0, 0), radius_x=0.20, radius_z=0.20, depth=23),),
        mountains=(MountainRange(
            path=_ring(0.27), width=0.065, height=19, roughness_contrast=0.25,
            roughness=NoiseLayer(kind="ridge", scale=80, seed_offset=89),
        ),),
    ),
    "ancient_battlefield": TerrainRecipe(
        name="ancient_battlefield", base_height=78, sea_level=61,
        base_noise=(
            NoiseLayer(scale=230, amplitude=9, octaves=3, seed_offset=97),
            NoiseLayer(scale=48, amplitude=2, octaves=2, seed_offset=101),
        ),
        basins=(
            Basin(Point(-0.2, -0.14), radius_x=0.075, radius_z=0.09, depth=10),
            Basin(Point(0.17, 0.12), radius_x=0.09, radius_z=0.065, depth=8),
            Basin(Point(0.05, -0.27), radius_x=0.05, radius_z=0.06, depth=6),
        ),
        mountains=(MountainRange(
            path=(Point(-0.26, 0.26), Point(-0.08, 0.16)), width=0.06, height=9,
            roughness_contrast=0.2,
        ),),
    ),
    "jiu_zong_ruin": TerrainRecipe(
        name="jiu_zong_ruin", base_height=76, sea_level=61,
        base_noise=(NoiseLayer(scale=95, amplitude=6, octaves=3, seed_offset=103),),
        plateaus=(
            Plateau(Point(0, 0), radius_x=0.25, radius_z=0.23, height=96, edge_width=0.07),
            Plateau(Point(-0.28, -0.18), radius_x=0.10, radius_z=0.10, height=84, edge_width=0.035),
            Plateau(Point(0.27, 0.23), radius_x=0.12, radius_z=0.10, height=83, edge_width=0.035),
        ),
    ),
    "dan_zong_yi_yuan": TerrainRecipe(
        name="dan_zong_yi_yuan", base_height=78, sea_level=61,
        base_noise=(NoiseLayer(scale=240, amplitude=4, octaves=3, seed_offset=107),),
        plateaus=(
            Plateau(Point(0, -0.23), radius_x=0.33, radius_z=0.14, height=92, edge_width=0.045),
            Plateau(Point(0, 0), radius_x=0.34, radius_z=0.14, height=86, edge_width=0.045),
            Plateau(Point(0, 0.23), radius_x=0.33, radius_z=0.14, height=80, edge_width=0.045),
        ),
    ),
    "wangyintai": TerrainRecipe(
        name="wangyintai", base_height=78, sea_level=61,
        base_noise=(NoiseLayer(scale=180, amplitude=2, octaves=2, seed_offset=109),),
        plateaus=(
            Plateau(Point(0, 0), radius_x=0.35, radius_z=0.35, height=91, edge_width=0.06),
            Plateau(Point(0, 0), radius_x=0.18, radius_z=0.18, height=104, edge_width=0.055),
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
        x, z = value.x * zone.size_x, value.z * zone.size_z
        if zone.shape == "rotated_rift":
            angle = math.pi / 6.0
            x, z = x * math.cos(angle) - z * math.sin(angle), x * math.sin(angle) + z * math.cos(angle)
        return Point(zone.center_x + x, zone.center_z + z)

    return replace(
        template,
        basins=tuple(replace(
            basin, center=point(basin.center), radius_x=basin.radius_x * zone.size_x,
            radius_z=basin.radius_z * zone.size_z,
        ) for basin in template.basins),
        plateaus=tuple(replace(
            plateau, center=point(plateau.center), radius_x=plateau.radius_x * zone.size_x,
            radius_z=plateau.radius_z * zone.size_z, edge_width=plateau.edge_width * scale,
        ) for plateau in template.plateaus),
        mountains=tuple(replace(
            mountain, path=tuple(point(p) for p in mountain.path), width=mountain.width * scale,
        ) for mountain in template.mountains),
    )
