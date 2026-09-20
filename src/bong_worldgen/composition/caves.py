"""Compile underground region layouts into ordinary engine cave parameters."""

from __future__ import annotations

import statistics

from ..data.world import ZoneDefinition
from ..engine import CaveNetwork, Point


def caves_for_zone(zone: ZoneDefinition, surface_height: float) -> tuple[CaveNetwork, ...]:
    if zone.terrain_profile not in ("cave_network", "abyssal_maze"):
        return ()
    center = Point(zone.center_x, zone.center_z)
    scale = min(zone.size_x, zone.size_z)
    entrances = tuple(Point(poi.pos_xyz[0], poi.pos_xyz[2]) for poi in zone.pois
                      if poi.kind == "cave_mouth" or "bottomless" in poi.tags)
    rooms = tuple(poi for poi in zone.pois if poi.kind != "cave_mouth" and "bottomless" not in poi.tags)
    centers = tuple(Point(poi.pos_xyz[0], poi.pos_xyz[2]) for poi in rooms)
    # Authored points anchor real rooms, and radial arms keep the network useful
    # even for a region with no POIs. All dimensions below are world units.
    targets = (*centers, Point(center.x - scale * 0.28, center.z - scale * 0.12),
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
    ) for index, depth in enumerate(depths))
