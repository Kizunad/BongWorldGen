"""Translate generic scalar fields into Bong raster layer arrays.

This is the only module that knows Bong's layer names and palette ids. The
procedural engine remains reusable for non-Minecraft consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..engine.models import Heightfield


@dataclass(frozen=True)
class BongTile:
    height: np.ndarray
    surface_id: np.ndarray
    subsurface_id: np.ndarray
    water_level: np.ndarray
    biome_id: np.ndarray
    feature_mask: np.ndarray
    boundary_weight: np.ndarray

    def __post_init__(self) -> None:
        shape = self.height.shape
        for name in (
            "surface_id",
            "subsurface_id",
            "water_level",
            "biome_id",
            "feature_mask",
            "boundary_weight",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Bong tile layer {name} shape mismatch")


def to_bong_tile(
    field: Heightfield,
    *,
    sea_level: float,
    grass_id: int = 3,
    stone_id: int = 0,
    gravel_id: int = 2,
    river_biome_id: int = 8,
    land_biome_id: int = 0,
) -> BongTile:
    """Apply a small, explicit palette policy to a generated heightfield."""

    wet = field.water_level >= 0.0
    high = field.height >= sea_level + 52.0
    surface_id = np.where(wet, gravel_id, np.where(high, stone_id, grass_id)).astype(np.uint8)
    subsurface_id = np.where(high, stone_id, grass_id).astype(np.uint8)
    biome_id = np.where(wet, river_biome_id, land_biome_id).astype(np.uint8)
    feature_mask = np.clip(
        np.maximum(np.abs(field.moisture - 0.5) * 0.8, high.astype(np.float32) * 0.35),
        0.0,
        1.0,
    ).astype(np.float32)
    return BongTile(
        height=np.ascontiguousarray(field.height, dtype=np.float32),
        surface_id=surface_id,
        subsurface_id=subsurface_id,
        water_level=np.ascontiguousarray(field.water_level, dtype=np.float32),
        biome_id=biome_id,
        feature_mask=feature_mask,
        boundary_weight=np.zeros(field.height.shape, dtype=np.float32),
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
    spans_count = np.ones(tile_size * tile_size, dtype=np.uint8)
    heights = np.rint(tile.height).astype(np.int16).reshape(-1)
    spans = np.full(tile_size * tile_size * 8, 32767, dtype="<i2")
    spans[0::8] = -64
    spans[1::8] = np.clip(heights, -64, 431)

    binary_layers = {
        "spans_count.bin": spans_count,
        "spans.bin": spans,
        "surface_id.bin": surface,
        "subsurface_id.bin": subsurface,
        "water_level.bin": tile.water_level.astype("<f4", copy=False),
        "biome_id.bin": biome,
        "feature_mask.bin": tile.feature_mask.astype("<f4", copy=False),
        "boundary_weight.bin": tile.boundary_weight.astype("<f4", copy=False),
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
        "surface_palette": ["stone", "coarse_dirt", "gravel", "grass_block"],
        "biome_palette": ["minecraft:plains", "minecraft:river"],
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
