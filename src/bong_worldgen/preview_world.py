"""Export a complete standalone world for the existing Three.js console."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .adapters import write_bong_raster
from .adapters.bong_raster import BASE_SURFACE_PALETTE, ZONE_NONE_ID
from .adapters.zone_raster import to_zone_tile
from .composition import ZoneTerrain
from .composition.pois import resolve_world_pois
from .data.world import WorldDefinition, ZoneDefinition
from .data.world_definition import WORLD
from .data.recipes import DEFAULT_RECIPE
from .data.wilderness import wilderness_palette_manifest
from .engine import TerrainRecipe


SPAN_ENCODING = {
    "max_spans": 4,
    "bytes_per_column": 16,
    "sentinel": 32767,
    "count_file": "spans_count.bin",
    "spans_file": "spans.bin",
    "slot_layout": "i16_le(floor_y, ceiling_y) x max_spans, unused slots = sentinel",
}
OVERVIEW_STRIDE = 32


def _zone_manifest(zone: ZoneDefinition) -> dict[str, object]:
    """Project authored zone metadata into the console manifest shape."""

    # Keep this adapter local: the typed Python zone model remains independent
    # from the browser-facing manifest contract.
    return {
        "name": zone.name,
        "display_name": zone.display_name,
        "terrain_profile": zone.terrain_profile,
        "spirit_qi": zone.spirit_qi,
        "danger_level": zone.danger_level,
        "worldgen": {
            "center_xz": [zone.center_x, zone.center_z],
            "size_xz": [zone.size_x, zone.size_z],
            "shape": zone.shape,
            "boundary": {"mode": zone.boundary_mode, "width": zone.boundary_width},
            "source": "authored_zone_data",
        },
    }


def _tile_range(minimum: int, maximum: int, tile_size: int) -> range:
    first = math.floor(minimum / tile_size)
    last = math.floor(maximum / tile_size)
    return range(first, last + 1)


def export_preview_world(
    output_dir: Path,
    *,
    recipe: TerrainRecipe = DEFAULT_RECIPE,
    world: WorldDefinition = WORLD,
    seed: int = 812731,
    min_x: int = -4096,
    max_x: int = 4095,
    min_z: int = -4096,
    max_z: int = 4095,
    tile_size: int = 256,
) -> Path:
    """Generate every tile in a bounded world and return its manifest path."""

    if tile_size < 1 or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a positive power of two")
    if min_x > max_x or min_z > max_z:
        raise ValueError("world bounds must be ordered")
    composer = ZoneTerrain(world, background=recipe, seed=seed)
    rasters_dir = output_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    tile_entries: list[dict[str, object]] = []
    # Palette IDs must not depend on the order in which generated zone modules
    # happen to be assembled. Names are already unique by ZoneIndex validation.
    zone_palette = tuple(sorted(zone.name for zone in world.zones))
    overview_width = (max_x - min_x + 1 + OVERVIEW_STRIDE - 1) // OVERVIEW_STRIDE
    overview_height = (max_z - min_z + 1 + OVERVIEW_STRIDE - 1) // OVERVIEW_STRIDE
    overview_elevation = np.full((overview_height, overview_width), np.nan, dtype=np.float32)
    overview_surface = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_wilderness = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_zone = np.full((overview_height, overview_width), ZONE_NONE_ID, dtype=np.uint8)
    overview_x = min_x + np.arange(overview_width) * OVERVIEW_STRIDE
    overview_z = min_z + np.arange(overview_height) * OVERVIEW_STRIDE

    for tile_z in _tile_range(min_z, max_z, tile_size):
        for tile_x in _tile_range(min_x, max_x, tile_size):
            origin_x = tile_x * tile_size
            origin_z = tile_z * tile_size
            field = composer.generate(
                width=tile_size,
                height=tile_size,
                origin_x=origin_x,
                origin_z=origin_z,
            )
            blend = composer.index.query(
                origin_x + np.arange(tile_size)[None, :],
                origin_z + np.arange(tile_size)[:, None],
            )
            tile = to_zone_tile(
                field,
                blend,
                sea_level=recipe.sea_level,
                zone_palette=zone_palette,
            )
            # Sample the coordinates declared by overview.origin and cell_size.
            # Tile-local stride loops shift samples at unaligned world bounds
            # and overwrite them when tile_size is smaller than the stride.
            ox = np.flatnonzero((overview_x >= origin_x) & (overview_x < origin_x + tile_size))
            oz = np.flatnonzero((overview_z >= origin_z) & (overview_z < origin_z + tile_size))
            target = np.ix_(oz, ox)
            source = np.ix_(overview_z[oz] - origin_z, overview_x[ox] - origin_x)
            overview_elevation[target] = tile.height[source]
            overview_surface[target] = tile.surface_id[source]
            overview_wilderness[target] = tile.wilderness_id[source]
            overview_zone[target] = tile.zone_id[source]
            write_bong_raster(
                tile,
                rasters_dir,
                tile_x=tile_x,
                tile_z=tile_z,
                world_name=recipe.name,
                write_manifest=False,
            )
            tile_entries.append(
                {
                    "tile_x": tile_x,
                    "tile_z": tile_z,
                    "dir": f"tile_{tile_x}_{tile_z}",
                    "zones": [part.zone.name for part in blend.contributions] or ["procedural_world"],
                    "layers": [
                        "surface_id",
                        "subsurface_id",
                        "water_level",
                        "biome_id",
                        "feature_mask",
                        "boundary_weight",
                        "wilderness_id",
                        "riverbed_id",
                        "cave_id",
                        "zone_id",
                    ],
                    "spans": True,
                }
            )

    if not np.isfinite(overview_elevation).all():
        raise ValueError("overview sampling left uncovered cells")
    overview_elevation.tofile(rasters_dir / "overview_height.bin")
    overview_surface.tofile(rasters_dir / "overview_surface_id.bin")
    overview_wilderness.tofile(rasters_dir / "overview_wilderness_id.bin")
    overview_zone.tofile(rasters_dir / "overview_zone_id.bin")

    surface_palette = list(BASE_SURFACE_PALETTE)
    riverbed_palette = list(
        dict.fromkeys(material for river in recipe.rivers for material in river.bed_materials)
    )
    for material in riverbed_palette:
        if material not in surface_palette:
            surface_palette.append(material)

    manifest = {
        "version": 2,
        "backend": "raster",
        "generation": {"composer": "zone_terrain", "seed": seed, "pending_profiles": []},
        "world_name": recipe.name,
        "tile_size": tile_size,
        "spans_encoding": SPAN_ENCODING,
        "world_bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
        },
        "overview": {
            "width": overview_width,
            "height": overview_height,
            "origin_x": min_x,
            "origin_z": min_z,
            "cell_size": OVERVIEW_STRIDE,
            "height_file": "overview_height.bin",
            "surface_file": "overview_surface_id.bin",
            "wilderness_file": "overview_wilderness_id.bin",
            "zone_file": "overview_zone_id.bin",
            "note": "Display-only overview; full-resolution tile rasters remain authoritative.",
        },
        "surface_palette": surface_palette,
        "riverbed_palette": riverbed_palette,
        "cave_palette": [network.name for network in composer.feature_recipe.caves],
        "zone_palette": list(zone_palette),
        "zone_encoding": {"none": ZONE_NONE_ID, "dtype": "u8"},
        "biome_palette": ["minecraft:plains", "minecraft:river"],
        "wilderness_palette": wilderness_palette_manifest(),
        "tiles": tile_entries,
        "pois": [poi.manifest() for poi in resolve_world_pois(composer)],
        "poi_connections": [],
        "zones": [
            {
                "name": "procedural_world",
                "display_name": "程序山河",
                "terrain_profile": recipe.name,
                "spirit_qi": 0.35,
                "danger_level": 2,
                "worldgen": {"generator": "bong_worldgen", "seed": seed},
            },
            *(_zone_manifest(zone) for zone in world.zones),
        ],
        "semantic_layers": ["cave_id", "zone_id"],
        "vertical_layers": ["spans"],
        "notes": [
            "Generated by standalone BongWorldGen.",
            "The existing Three.js console reads this manifest directly.",
        ],
    }
    manifest_path = rasters_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["SPAN_ENCODING", "export_preview_world"]
