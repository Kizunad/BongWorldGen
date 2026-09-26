"""Join composed region ownership to generic engine fields at the output boundary."""

from dataclasses import replace

import numpy as np

from ..composition.layout import ZoneBlend
from ..composition.terrain import ZoneTerrain
from ..composition.surface import scorch_mask
from ..engine import Heightfield
from .bong_raster import BongTile, ZONE_NONE_ID, to_bong_tile


def to_zone_tile(
    field: Heightfield,
    blend: ZoneBlend,
    *,
    sea_level: float,
    zone_palette: tuple[str, ...],
    surface_slope: np.ndarray | None = None,
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
    tile = to_bong_tile(field, sea_level=sea_level, zone_id=lookup[indices],
                        zone_palette=zone_palette, surface_slope=surface_slope)
    return replace(tile, boundary_weight=(1.0 - weights).astype(np.float32))


def generate_zone_tile(
    composer: ZoneTerrain,
    *,
    width: int,
    height: int,
    origin_x: float = 0,
    origin_z: float = 0,
    cell_size: float = 1,
) -> tuple[Heightfield, BongTile]:
    """Generate one tile with neighboring heights for crop-independent slope.

    A one-sample halo makes every requested column an interior sample, including
    single-column tiles. Slopes are measured in height units per world block;
    coarser grids still approximate them at their declared sampling distance.
    """

    if width < 1 or height < 1:
        raise ValueError("heightfield dimensions must be positive")
    padded = composer.generate(
        width=width + 2, height=height + 2,
        origin_x=origin_x - cell_size, origin_z=origin_z - cell_size, cell_size=cell_size,
    )
    elevations = padded.height.astype(np.float64)
    slope = np.hypot(
        (elevations[1:-1, 2:] - elevations[1:-1, :-2]) / (2 * cell_size),
        (elevations[2:, 1:-1] - elevations[:-2, 1:-1]) / (2 * cell_size),
    )
    field = replace(
        padded,
        **{name: np.ascontiguousarray(getattr(padded, name)[1:-1, 1:-1]) for name in (
            "height", "moisture", "water_level", "riverbed_id", "solid_spans", "cave_id",
        )},
        underground_blocks=tuple(
            replace(block, x=block.x - 1, z=block.z - 1) for block in padded.underground_blocks
            if 1 <= block.x <= width and 1 <= block.z <= height
        ),
    )
    x = origin_x + np.arange(width)[None, :] * cell_size
    z = origin_z + np.arange(height)[:, None] * cell_size
    blend = composer.index.query(x, z)
    tile = to_zone_tile(
        field, blend, sea_level=composer.background.sea_level,
        zone_palette=tuple(sorted(zone.name for zone in composer.index.zones)), surface_slope=slope,
    )
    scorched = scorch_mask(composer, x, z, blend) & (field.riverbed_id < 0)
    surfaces = tile.surface_id.copy()
    surfaces[scorched] = tile.surface_palette.index("blackstone")
    tile = replace(tile, surface_id=surfaces)
    return field, tile
