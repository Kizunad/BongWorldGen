"""Compile underground region layouts into ordinary engine cave parameters."""

from __future__ import annotations

from dataclasses import replace
import math
import statistics

from ..data.world import ZoneDefinition
from ..engine import Basin, CaveNetwork, MountainRange, Point, TerrainRecipe


def caves_for_zone(zone: ZoneDefinition, surface_height: float) -> tuple[CaveNetwork, ...]:
    if zone.terrain_profile not in ("cave_network", "abyssal_maze"):
        return ()
    center = Point(zone.center_x, zone.center_z)
    scale = min(zone.size_x, zone.size_z)
    entrances = tuple(Point(poi.pos_xyz[0], poi.pos_xyz[2]) for poi in zone.pois
                      if poi.kind == "cave_mouth" or "bottomless" in poi.tags)
    rooms = tuple(poi for poi in zone.pois if poi.kind != "cave_mouth" and "bottomless" not in poi.tags)
    centers = tuple(Point(poi.pos_xyz[0], poi.pos_xyz[2]) for poi in rooms)
    # Rooms and authored entrances both connect to the main network. A shaft
    # alone can otherwise end in solid rock far from all horizontal passages.
    # Radial arms also serve regions with no POIs. Dimensions are world units.
    targets = (*centers, *entrances, Point(center.x - scale * 0.28, center.z - scale * 0.12),
               Point(center.x + scale * 0.27, center.z + scale * 0.15))
    paths = tuple((center, target) for target in targets if target != center)
    if zone.terrain_profile == "abyssal_maze":
        depths = (34.0, 72.0, 112.0)
    else:
        target_y = statistics.median(poi.pos_xyz[1] for poi in rooms) if rooms else 24.0
        depths = (min(125.0, max(24.0, surface_height - target_y - 4.0)),)
    return tuple(CaveNetwork(
        name=f"{zone.name}/level_{index + 1}", paths=paths,
        width=5.0 if len(depths) == 1 else 6.5, height=8,
        depth=depth, noise_strength=0.06, dead_end_strength=0.25,
        vertical_warp=1.5, domain_warp_strength=1.5,
        branch_count=4, branch_segments=3, branch_length=scale * 0.13,
        chamber_count=2, chamber_radius=min(18, scale * 0.035), chamber_height=8,
        chamber_centers=centers, entrance_count=0,
        entrance_points=entrances or (Point(center.x - scale * 0.28, center.z - scale * 0.12),),
        entrance_radius=4, roof_thickness=4,
        fill_vertical_gaps=True,
    ) for index, depth in enumerate(depths))


def with_cave_landforms(zone: ZoneDefinition, recipe: TerrainRecipe) -> TerrainRecipe:
    """Place surface collapse features at the compiled underground entrances."""

    if not recipe.caves:
        return recipe
    scale = min(zone.size_x, zone.size_z)
    entrances = recipe.caves[0].entrance_points
    basins, mountains = list(recipe.basins), list(recipe.mountains)
    if zone.terrain_profile == "cave_network":
        # A field of broad collapses covers the outer body as well as the
        # entrance. Circular zones use their true diameter on both axes.
        extent_x = scale if zone.shape == "circular" else zone.size_x
        extent_z = scale if zone.shape == "circular" else zone.size_z
        for x, z, rx, rz, depth in (
            (-0.31, -0.23, 0.11, 0.10, 18),
            (-0.10, -0.35, 0.12, 0.095, 20),
            (0.13, -0.36, 0.085, 0.085, 16),
            (0.37, -0.06, 0.085, 0.095, 18),
            (0.33, 0.20, 0.10, 0.10, 16),
            (0.07, 0.36, 0.11, 0.085, 18),
            (-0.16, 0.34, 0.10, 0.09, 17),
            (-0.36, 0.02, 0.095, 0.10, 18),
            (-0.14, -0.11, 0.09, 0.10, 12),
        ):
            basins.append(Basin(Point(zone.center_x + x * extent_x, zone.center_z + z * extent_z),
                                radius_x=rx * extent_x, radius_z=rz * extent_z, depth=depth))
        for entrance in entrances:
            basins.append(Basin(entrance, radius_x=0.075 * scale,
                                radius_z=0.095 * scale, depth=12))
            mountains.append(MountainRange(
                path=tuple(Point(entrance.x + x * scale, entrance.z + z * scale)
                           for x, z in ((-0.14, 0.04), (-0.055, -0.02), (0, 0), (0.11, 0.07))),
                width=0.065 * scale, height=2, valley_depth=5, roughness_contrast=0.2,
            ))
    elif zone.terrain_profile == "abyssal_maze":
        main = entrances[0]
        targets = entrances[1:] or (Point(main.x + scale * 0.23, main.z + scale * 0.46),)
        for target in targets:
            dx, dz = target.x - main.x, target.z - main.z
            length = math.hypot(dx, dz)
            if length == 0:
                continue
            nx, nz = -dz / length, dx / length
            # A bent collapse corridor joins the actual main door and shaft.
            # Unequal, offset shoulders form two broken escarpments, without
            # exposing the underground rooms as an open surface canyon.
            path = tuple(Point(main.x + t * dx + bend * scale * nx,
                               main.z + t * dz + bend * scale * nz)
                         for t, bend in ((-0.16, 0), (0, 0), (0.25, 0.045),
                                         (0.5, -0.04), (0.78, 0.03), (1, 0)))
            mountains.append(MountainRange(path=path, width=0.15 * scale,
                                             height=0, valley_depth=14))
            for side, start, end, offset, height, width in (
                (-1, 0, 4, 0.095, 32, 0.035),
                (1, 1, 6, 0.075, 18, 0.04),
            ):
                mountains.append(MountainRange(
                    path=tuple(Point(p.x + side * offset * scale * nx,
                                     p.z + side * offset * scale * nz)
                               for p in path[start:end]),
                    width=width * scale, height=height, roughness_contrast=0.2,
                ))
    return replace(recipe, basins=tuple(basins), mountains=tuple(mountains))
