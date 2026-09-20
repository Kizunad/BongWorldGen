"""Place authored landmarks on the solid geometry actually exported."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..data.world import PoiDefinition, ZoneDefinition
from .terrain import ZoneTerrain


@dataclass(frozen=True)
class ResolvedPoi:
    zone: str
    poi: PoiDefinition
    pos_xyz: tuple[float, float, float]
    placement: str

    def manifest(self) -> dict[str, object]:
        return {
            "zone": self.zone, "kind": self.poi.kind, "name": self.poi.name,
            "pos_xyz": list(self.pos_xyz), "authored_pos_xyz": list(self.poi.pos_xyz),
            "placement": self.placement, "tags": list(self.poi.tags),
            "unlock": self.poi.unlock, "qi_affinity": self.poi.qi_affinity,
            "danger_bias": self.poi.danger_bias,
        }


def resolve_poi(composer: ZoneTerrain, zone: ZoneDefinition, poi: PoiDefinition) -> ResolvedPoi:
    """Keep X/Z; choose a supported feet Y with at least two blocks of headroom.

    Underground rooms use the valid cave floor closest to the authored height.
    Ground-tagged sky-region remains select the ground, not the island above.
    An entrance shaft may have its first solid surface below nominal height.
    """

    x, authored_y, z = poi.pos_xyz
    if not all(math.isfinite(value) for value in poi.pos_xyz):
        raise ValueError(f"non-finite position for POI {poi.name!r}")
    field = composer.generate(width=1, height=1, origin_x=math.floor(x), origin_z=math.floor(z))
    spans = [tuple(map(int, span)) for span in field.solid_spans[0, 0] if span[0] != 32767]
    underground = zone.terrain_profile in ("cave_network", "abyssal_maze")
    entrance = poi.kind == "cave_mouth" or "bottomless" in poi.tags
    if underground and not entrance:
        floors = [lower[1] + 1 for upper, lower in zip(spans, spans[1:])
                  if upper[0] - lower[1] - 1 >= 2]
        if not floors:
            raise ValueError(f"underground POI {poi.name!r} has no accessible cave floor")
        y = min(floors, key=lambda value: (abs(value - authored_y), value))
        placement = "cave_floor"
    elif "ground" in poi.tags and zone.terrain_profile == "sky_isle":
        y, placement = spans[-1][1] + 1, "ground"
    else:
        y = spans[0][1] + 1
        placement = "island_surface" if len(spans) > 1 and zone.terrain_profile == "sky_isle" else "surface"
        if entrance and y < round(float(field.height[0, 0])):
            placement = "entrance_floor"
        elif field.water_level[0, 0] >= y:
            placement = "underwater_surface"
    return ResolvedPoi(zone.name, poi, (x, float(y), z), placement)


def resolve_world_pois(composer: ZoneTerrain) -> tuple[ResolvedPoi, ...]:
    return tuple(resolve_poi(composer, zone, poi) for zone in composer.world.zones for poi in zone.pois)
