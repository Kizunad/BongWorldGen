"""Translate generic scalar fields into Bong raster layer arrays.

This is the only module that knows Bong's layer names and palette ids. The
procedural engine remains reusable for non-Minecraft consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..data.wilderness import classify_wilderness, wilderness_palette_manifest
from ..engine.models import Heightfield


BASE_SURFACE_PALETTE = ("stone", "coarse_dirt", "gravel", "grass_block")
RIVERBED_NONE_ID = 255
ZONE_NONE_ID = 255


@dataclass(frozen=True)
class BongTile:
    height: np.ndarray
    surface_id: np.ndarray
    subsurface_id: np.ndarray
    water_level: np.ndarray
    biome_id: np.ndarray
    feature_mask: np.ndarray
    boundary_weight: np.ndarray
    wilderness_id: np.ndarray
    riverbed_id: np.ndarray | None = None
    surface_palette: tuple[str, ...] = BASE_SURFACE_PALETTE
    riverbed_palette: tuple[str, ...] = ()
    solid_spans: np.ndarray | None = None
    cave_id: np.ndarray | None = None
    cave_palette: tuple[str, ...] = ()
    zone_id: np.ndarray | None = None
    zone_palette: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape = self.height.shape
        for name in (
            "surface_id",
            "subsurface_id",
            "water_level",
            "biome_id",
            "feature_mask",
            "boundary_weight",
            "wilderness_id",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Bong tile layer {name} shape mismatch")
        if self.riverbed_id is None:
            object.__setattr__(
                self,
                "riverbed_id",
                np.full(shape, RIVERBED_NONE_ID, dtype=np.uint8),
            )
        elif self.riverbed_id.shape != shape:
            raise ValueError("Bong tile layer riverbed_id shape mismatch")
        if self.cave_id is None:
            object.__setattr__(self, "cave_id", np.zeros(shape, dtype=np.uint8))
        elif self.cave_id.shape != shape:
            raise ValueError("Bong tile cave_id shape mismatch")
        if self.zone_id is None:
            object.__setattr__(self, "zone_id", np.full(shape, ZONE_NONE_ID, dtype=np.uint8))
        elif self.zone_id.shape != shape:
            raise ValueError("Bong tile zone_id shape mismatch")
        if not np.issubdtype(self.cave_id.dtype, np.integer):
            raise ValueError("Bong tile cave_id must contain integer palette ids")
        if np.any(self.cave_id > len(self.cave_palette)):
            raise ValueError("Bong tile cave_id contains an unknown palette id")
        if not np.issubdtype(self.zone_id.dtype, np.integer):
            raise ValueError("Bong tile zone_id must contain integer palette ids")
        if len(self.zone_palette) > ZONE_NONE_ID:
            raise ValueError("Bong tile zone_palette must contain at most 255 entries")
        valid_zone = (self.zone_id == ZONE_NONE_ID) | (
            (self.zone_id >= 0) & (self.zone_id < len(self.zone_palette))
        )
        if not np.all(valid_zone):
            raise ValueError("Bong tile zone_id contains an unknown palette id")
        if self.solid_spans is not None:
            expected = (*shape, 4, 2)
            if self.solid_spans.shape != expected:
                raise ValueError(f"Bong tile solid_spans must have shape {expected}")


def to_bong_tile(
    field: Heightfield,
    *,
    sea_level: float,
    grass_id: int = 3,
    stone_id: int = 0,
    gravel_id: int = 2,
    # This standalone adapter declares a two-entry palette below
    # (0=plains, 1=river); keep the default id aligned with that contract.
    river_biome_id: int = 1,
    land_biome_id: int = 0,
    zone_id: np.ndarray | None = None,
    zone_palette: tuple[str, ...] = (),
    surface_slope: np.ndarray | None = None,
) -> BongTile:
    """Apply a small, explicit palette policy to a generated heightfield."""

    wet = field.water_level >= 0.0
    high = field.height >= sea_level + 52.0
    surface_palette = list(BASE_SURFACE_PALETTE)
    for material in field.riverbed_palette:
        if material not in surface_palette:
            surface_palette.append(material)
    surface_id = np.where(wet, gravel_id, np.where(high, stone_id, grass_id)).astype(np.uint8)
    if field.riverbed_palette:
        for material_index, material in enumerate(field.riverbed_palette):
            surface_id[field.riverbed_id == material_index] = surface_palette.index(material)
    subsurface_id = np.where(high, stone_id, grass_id).astype(np.uint8)
    biome_id = np.where(wet, river_biome_id, land_biome_id).astype(np.uint8)
    feature_mask = np.clip(
        np.maximum(np.abs(field.moisture - 0.5) * 0.8, high.astype(np.float32) * 0.35),
        0.0,
        1.0,
    ).astype(np.float32)
    wilderness_id = classify_wilderness(
        field.height, field.water_level, sea_level=sea_level, surface_slope=surface_slope,
    )
    normalized_zone_id = None
    if zone_id is not None:
        raw_zone_id = np.asarray(zone_id)
        if not np.issubdtype(raw_zone_id.dtype, np.integer):
            raise ValueError("Bong tile zone_id must contain integer palette ids")
        if np.any(raw_zone_id < 0) or np.any(raw_zone_id > ZONE_NONE_ID):
            raise ValueError("Bong tile zone_id must fit in uint8")
        normalized_zone_id = np.ascontiguousarray(raw_zone_id, dtype=np.uint8)
    return BongTile(
        height=np.ascontiguousarray(field.height, dtype=np.float32),
        surface_id=surface_id,
        subsurface_id=subsurface_id,
        water_level=np.ascontiguousarray(field.water_level, dtype=np.float32),
        biome_id=biome_id,
        feature_mask=feature_mask,
        boundary_weight=np.zeros(field.height.shape, dtype=np.float32),
        wilderness_id=wilderness_id,
        riverbed_id=np.ascontiguousarray(
            np.where(field.riverbed_id >= 0, field.riverbed_id, RIVERBED_NONE_ID),
            dtype=np.uint8,
        ),
        surface_palette=tuple(surface_palette),
        riverbed_palette=field.riverbed_palette,
        solid_spans=field.solid_spans,
        cave_id=np.ascontiguousarray(field.cave_id, dtype=np.uint8),
        cave_palette=field.cave_palette,
        zone_id=normalized_zone_id,
        zone_palette=tuple(zone_palette),
    )


def write_bong_raster(
    tile: BongTile,
    output_dir: Path,
    *,
    tile_x: int = 0,
    tile_z: int = 0,
    world_name: str = "bong_worldgen_preview",
    write_manifest: bool = True,
) -> Path:
    """Write one loader-compatible v2 raster tile and its manifest.

    The first standalone adapter emits a single ground span per column. This
    deliberately keeps the vertical contract correct while leaving caves and
    floating islands to a later span modifier.
    """

    if tile.height.ndim != 2 or tile.height.shape[0] != tile.height.shape[1]:
        raise ValueError("Bong raster tiles must be square")
    tile_size = tile.height.shape[0]
    tile_dir = output_dir / f"tile_{tile_x}_{tile_z}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    surface = tile.surface_id.astype(np.uint8, copy=False)
    subsurface = tile.subsurface_id.astype(np.uint8, copy=False)
    biome = tile.biome_id.astype(np.uint8, copy=False)
    heights = np.rint(tile.height).astype(np.int16).reshape(-1)
    if tile.solid_spans is None:
        spans_count = np.ones(tile_size * tile_size, dtype=np.uint8)
        spans = np.full(tile_size * tile_size * 8, 32767, dtype="<i2")
        spans[0::8] = -64
        spans[1::8] = np.clip(heights, -64, 431)
    else:
        spans_array = np.asarray(tile.solid_spans, dtype="<i2")
        spans_count = np.count_nonzero(spans_array[..., 0] != 32767, axis=2).astype(np.uint8).reshape(-1)
        spans = spans_array.reshape(tile_size * tile_size * 8)

    binary_layers = {
        "spans_count.bin": spans_count,
        "spans.bin": spans,
        "surface_id.bin": surface,
        "subsurface_id.bin": subsurface,
        "water_level.bin": tile.water_level.astype("<f4", copy=False),
        "biome_id.bin": biome,
        "feature_mask.bin": tile.feature_mask.astype("<f4", copy=False),
        "boundary_weight.bin": tile.boundary_weight.astype("<f4", copy=False),
        "wilderness_id.bin": tile.wilderness_id.astype(np.uint8, copy=False),
        "riverbed_id.bin": tile.riverbed_id.astype(np.uint8, copy=False),
        "cave_id.bin": tile.cave_id.astype(np.uint8, copy=False),
        "zone_id.bin": tile.zone_id.astype(np.uint8, copy=False),
    }
    for filename, values in binary_layers.items():
        values.tofile(tile_dir / filename)

    manifest = {
        "version": 2,
        "backend": "raster",
        "world_name": world_name,
        "tile_size": tile_size,
        "world_bounds": {
            "min_x": tile_x * tile_size,
            "max_x": (tile_x + 1) * tile_size - 1,
            "min_z": tile_z * tile_size,
            "max_z": (tile_z + 1) * tile_size - 1,
        },
        "surface_palette": list(tile.surface_palette),
        "riverbed_palette": list(tile.riverbed_palette),
        "riverbed_encoding": {"none": RIVERBED_NONE_ID, "dtype": "u8"},
        "cave_palette": list(tile.cave_palette),
        "cave_encoding": {"none": 0, "dtype": "u8", "vertical_range": "spans.bin"},
        "zone_palette": list(tile.zone_palette),
        "zone_encoding": {"none": ZONE_NONE_ID, "dtype": "u8"},
        "biome_palette": ["minecraft:plains", "minecraft:river"],
        "wilderness_palette": wilderness_palette_manifest(),
        "tiles": [
            {
                "tile_x": tile_x,
                "tile_z": tile_z,
                "dir": tile_dir.name,
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
        ],
    }
    manifest_path = output_dir / "manifest.json"
    if write_manifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10),
            encoding="utf-8",
        )
    return manifest_path
