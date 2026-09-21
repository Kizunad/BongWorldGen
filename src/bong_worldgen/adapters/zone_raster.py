"""Join composed region ownership to generic engine fields at the output boundary."""

from dataclasses import replace

import numpy as np

from ..composition.layout import ZoneBlend
from ..engine import Heightfield
from .bong_raster import BongTile, ZONE_NONE_ID, to_bong_tile


def to_zone_tile(
    field: Heightfield,
    blend: ZoneBlend,
    *,
    sea_level: float,
    zone_palette: tuple[str, ...],
) -> BongTile:
    if len(zone_palette) > ZONE_NONE_ID:
        raise ValueError("zone_palette must contain at most 255 entries")
    if len(set(zone_palette)) != len(zone_palette):
        raise ValueError("zone_palette contains duplicate names")
    if blend.background.shape != field.height.shape:
        raise ValueError("zone blend and heightfield must share a shape")
    ids_by_name = {name: index for index, name in enumerate(zone_palette)}
    # -1, the composer's background index, selects the last lookup entry.
    try:
        lookup = np.asarray([ids_by_name[part.zone.name] for part in blend.contributions]
                            + [ZONE_NONE_ID], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"zone_palette is missing contributing zone {exc.args[0]!r}") from exc
    indices, weights = blend.dominant()
    tile = to_bong_tile(field, sea_level=sea_level, zone_id=lookup[indices], zone_palette=zone_palette)
    return replace(tile, boundary_weight=(1.0 - weights).astype(np.float32))
