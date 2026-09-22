"""Authored terrain profiles compiled into ordinary engine recipes.

Recipes use local coordinates here. Compilation translates geometric anchors
to the zone center; noise always uses world coordinates in the engine.
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

from ..data.world import ZoneDefinition
from ..engine import Basin, FloatingIsland, MountainRange, NoiseLayer, Plateau, Point, TerrainRecipe
from .caves import caves_for_zone


def _arc(radius: float, start: float, end: float) -> tuple[Point, ...]:
    """An open ejecta segment; angles are degrees in the local X/Z plane."""
    return tuple(Point(radius * math.cos(math.radians(start + (end - start) * i / 12)),
                       radius * math.sin(math.radians(start + (end - start) * i / 12)))
                 for i in range(13))


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
        basins=(Basin(Point(-0.10, -0.21), radius_x=0.085, radius_z=0.13, depth=9),),
        mountains=(
            # A through-going fracture with a shorter diagonal collapse branch.
            MountainRange(path=(Point(-0.07, -0.44), Point(0.03, -0.20), Point(0, 0),
                                Point(0.04, 0.18), Point(-0.04, 0.44)),
                          width=0.16, height=0, valley_depth=19),
            MountainRange(path=(Point(0.02, 0.08), Point(-0.14, 0.22), Point(-0.25, 0.29)),
                          width=0.10, height=0, valley_depth=12),
            # Offset, unequal scarp remnants leave both ends open.
            MountainRange(path=(Point(0.22, -0.36), Point(0.22, -0.10), Point(0.26, 0),
                                Point(0.20, 0.28), Point(0.12, 0.40)),
                          width=0.055, height=25, roughness_contrast=0.3,
                          roughness=NoiseLayer(kind="ridge", scale=40, seed_offset=79)),
            MountainRange(path=(Point(-0.24, -0.32), Point(-0.20, -0.12), Point(-0.22, 0.06)),
                          width=0.065, height=15, roughness_contrast=0.3),
        ),
    ),
    "tribulation_scorch": TerrainRecipe(
        name="tribulation_scorch", base_height=88.0, sea_level=61.0,
        base_noise=(NoiseLayer(scale=85, amplitude=2.5, octaves=3, seed_offset=83),),
        basins=(Basin(Point(0, 0), radius_x=0.14, radius_z=0.17, depth=25),),
        mountains=(
            MountainRange(path=_arc(0.26, -30, 38), width=0.06, height=22,
                          roughness_contrast=0.25,
                          roughness=NoiseLayer(kind="ridge", scale=80, seed_offset=89)),
            MountainRange(path=_arc(0.23, 86, 145), width=0.055, height=17,
                          roughness_contrast=0.25),
            MountainRange(path=_arc(0.28, 188, 232), width=0.045, height=14,
                          roughness_contrast=0.25),
            # Branched outward scars cross the gaps in the broken impact rim.
            MountainRange(path=(Point(0.04, 0.07), Point(0.12, 0.20), Point(0.25, 0.35)),
                          width=0.10, height=0, valley_depth=12),
            MountainRange(path=(Point(0.12, 0.20), Point(0.07, 0.32), Point(0.10, 0.43)),
                          width=0.075, height=0, valley_depth=9),
            MountainRange(path=(Point(-0.07, 0), Point(-0.22, 0.10), Point(-0.39, 0.07)),
                          width=0.11, height=0, valley_depth=12),
            MountainRange(path=(Point(0, -0.08), Point(-0.06, -0.25), Point(-0.20, -0.39)),
                          width=0.11, height=0, valley_depth=13),
            MountainRange(path=(Point(-0.06, -0.25), Point(0.11, -0.30), Point(0.22, -0.40)),
                          width=0.08, height=0, valley_depth=10),
        ),
    ),
    "ancient_battlefield": TerrainRecipe(
        name="ancient_battlefield", base_height=78, sea_level=61,
        base_noise=(
            NoiseLayer(scale=230, amplitude=9, octaves=3, seed_offset=97),
            NoiseLayer(scale=48, amplitude=2, octaves=2, seed_offset=101),
        ),
        basins=(
            Basin(Point(-0.31, 0.20), radius_x=0.065, radius_z=0.09, depth=14),
            Basin(Point(0.24, 0.18), radius_x=0.085, radius_z=0.06, depth=16),
            Basin(Point(0.05, -0.27), radius_x=0.055, radius_z=0.065, depth=11),
            Basin(Point(-0.20, -0.1875), radius_x=0.055, radius_z=0.055, depth=10),
            Basin(Point(0.40, -0.3125), radius_x=0.065, radius_z=0.08, depth=15),
        ),
        mountains=(
            # Crossing strikes leave long narrow scars across the otherwise
            # low plain. Their shoulders are displaced soil, not mountain chains.
            MountainRange(path=(Point(-0.38, 0.12), Point(-0.22, 0.055), Point(-0.08, 0.02),
                                Point(0.05, -0.02), Point(0.18, -0.09), Point(0.35, -0.19)),
                          width=0.12, height=4, valley_depth=20, roughness_contrast=0.2),
            MountainRange(path=(Point(-0.27, -0.30), Point(-0.16, -0.17), Point(-0.02, 0.06),
                                Point(0.10, 0.18), Point(0.18, 0.35)),
                          width=0.10, height=3, valley_depth=16, roughness_contrast=0.2),
            # The authored burial ground (万骨冢) is on an elongated remnant mound.
            MountainRange(path=(Point(0.065, -0.18), Point(0.11, -0.11), Point(0.14, -0.07)),
                          width=0.045, height=16, roughness_contrast=0.2),
            MountainRange(path=(Point(-0.26, 0.26), Point(-0.08, 0.16)),
                          width=0.035, height=12, roughness_contrast=0.2),
        ),
        plateaus=(
            # Disconnected fragments around the authored broken formation.
            Plateau(Point(-0.255, -0.24), 0.025, 0.065, height=87, edge_width=0.01, shape="rectangle"),
            Plateau(Point(-0.19, -0.25), 0.055, 0.02, height=86, edge_width=0.008, shape="rectangle"),
            Plateau(Point(-0.145, -0.19), 0.02, 0.04, height=84, edge_width=0.008, shape="rectangle"),
        ),
    ),
    "jiu_zong_ruin": TerrainRecipe(
        name="jiu_zong_ruin", base_height=76, sea_level=61,
        base_noise=(NoiseLayer(scale=95, amplitude=6, octaves=3, seed_offset=103),),
        plateaus=(
            # Separate foundations around an open, broken courtyard.
            Plateau(Point(0, 0), 0.13, 0.12, height=96, edge_width=0.018, shape="rectangle"),
            Plateau(Point(-0.02, -0.26), 0.25, 0.035, height=90, edge_width=0.012, shape="rectangle"),
            Plateau(Point(-0.26, -0.045), 0.034, 0.18, height=89, edge_width=0.012, shape="rectangle"),
            Plateau(Point(-0.16, 0.22), 0.115, 0.035, height=86, edge_width=0.012, shape="rectangle"),
            Plateau(Point(0.26, -0.14), 0.033, 0.13, height=87, edge_width=0.012, shape="rectangle"),
            Plateau(Point(0.255, 0.155), 0.043, 0.075, height=84, edge_width=0.014, shape="rectangle"),
            Plateau(Point(-0.24, -0.255), 0.07, 0.06, height=91, edge_width=0.015, shape="rectangle"),
        ),
    ),
    "dan_zong_yi_yuan": TerrainRecipe(
        name="dan_zong_yi_yuan", base_height=78, sea_level=61,
        base_noise=(NoiseLayer(scale=240, amplitude=4, octaves=3, seed_offset=107),),
        plateaus=(
            # Three descending rows, each divided into five beds by lower aisles.
            *(Plateau(Point(0, z), 0.345, 0.095, height=h - 4, edge_width=0.012, shape="rectangle")
              for z, h in ((-0.23, 92), (0, 86), (0.23, 80))),
            *(Plateau(Point(x, z), 0.047, 0.078, height=h, edge_width=0.009, shape="rectangle")
              for z, h in ((-0.23, 92), (0, 86), (0.23, 80))
              for x in (-0.26, -0.13, 0, 0.13, 0.26)),
        ),
    ),
    "wangyintai": TerrainRecipe(
        name="wangyintai", base_height=78, sea_level=61,
        base_noise=(NoiseLayer(scale=180, amplitude=2, octaves=2, seed_offset=109),),
        plateaus=(
            Plateau(Point(0, 0), 0.27, 0.27, height=91, edge_width=0.035,
                    shape="rectangle", rotation=math.pi / 4),
            *(Plateau(Point(0, sign * z), 0.033, 0.042, height=h, edge_width=0.008, shape="rectangle")
              for sign in (-1, 1) for z, h in ((0.235, 99), (0.30, 95), (0.365, 85))),
            Plateau(Point(0, 0), 0.125, 0.125, height=104, edge_width=0.025,
                    shape="rectangle", rotation=math.pi / 4),
        ),
    ),
    "cave_network": TerrainRecipe(
        name="cave_network", base_height=84, sea_level=61,
        base_noise=(NoiseLayer(scale=170, amplitude=8, octaves=3, seed_offset=113),),
        basins=(Basin(Point(0, 0), radius_x=0.25, radius_z=0.25, depth=6),),
    ),
    "abyssal_maze": TerrainRecipe(
        name="abyssal_maze", base_height=90, sea_level=61,
        base_noise=(NoiseLayer(kind="ridge", scale=190, amplitude=12, seed_offset=127),),
        basins=(Basin(Point(0, 0), radius_x=0.23, radius_z=0.31, depth=14),),
    ),
    "sky_isle": TerrainRecipe(
        name="sky_isle", base_height=72, sea_level=61,
        base_noise=(NoiseLayer(scale=280, amplitude=4, octaves=3, seed_offset=131),),
        floating_islands=(
            FloatingIsland(Point(0, 0), radius_x=0.25, radius_z=0.25),
            FloatingIsland(Point(0.32, -0.20), radius_x=0.095, radius_z=0.095,
                           height=244, relief=18, thickness=28),
            FloatingIsland(Point(-0.32, 0.16), radius_x=0.085, radius_z=0.085,
                           height=255, relief=16, thickness=25),
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
        caves=caves_for_zone(zone, template.base_height),
        floating_islands=tuple(replace(
            island, center=point(island.center), radius_x=island.radius_x * zone.size_x,
            radius_z=island.radius_z * zone.size_z,
        ) for island in template.floating_islands),
        basins=tuple(replace(
            basin, center=point(basin.center), radius_x=basin.radius_x * zone.size_x,
            radius_z=basin.radius_z * zone.size_z,
        ) for basin in template.basins),
        plateaus=tuple(replace(
            plateau, center=point(plateau.center), radius_x=plateau.radius_x * zone.size_x,
            radius_z=plateau.radius_z * zone.size_z, edge_width=plateau.edge_width * scale,
            rotation=plateau.rotation + (math.pi / 6 if zone.shape == "rotated_rift" else 0),
        ) for plateau in template.plateaus),
        mountains=tuple(replace(
            mountain, path=tuple(point(p) for p in mountain.path), width=mountain.width * scale,
        ) for mountain in template.mountains),
    )
