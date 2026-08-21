"""Recipe interpreter for the standalone procedural terrain engine."""

from __future__ import annotations

import numpy as np

from .geometry import polyline_distance_and_progress
from .models import Heightfield, TerrainRecipe
from .noise import sample_noise


def _coordinate_grid(
    width: int, height: int, origin_x: float, origin_z: float, cell_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if width < 1 or height < 1:
        raise ValueError("heightfield dimensions must be positive")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    xs = origin_x + np.arange(width, dtype=np.float64) * cell_size
    zs = origin_z + np.arange(height, dtype=np.float64) * cell_size
    return np.meshgrid(xs, zs, indexing="xy")


def _apply_mountains(
    terrain: np.ndarray, x: np.ndarray, z: np.ndarray, recipe: TerrainRecipe, seed: int
) -> np.ndarray:
    output = terrain
    for index, mountain in enumerate(recipe.mountains):
        distance, _ = polyline_distance_and_progress(x, z, mountain.path)
        ridge_mask = np.exp(-((distance / mountain.width) ** 2))
        roughness = sample_noise(x, z, mountain.roughness, seed + index * 7919)
        contrast = mountain.roughness_contrast
        roughness = np.clip((1.0 - contrast) + contrast * roughness, 0.0, 1.0)
        uplift = ridge_mask * roughness * mountain.height
        if mountain.valley_depth:
            valley_mask = np.exp(-((distance / (mountain.width * 0.24)) ** 2))
            uplift -= valley_mask * mountain.valley_depth
        output = output + uplift
    return output


def _apply_basins(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
) -> np.ndarray:
    output = terrain
    for basin in recipe.basins:
        normalized = np.hypot(
            (x - basin.center.x) / basin.radius_x,
            (z - basin.center.z) / basin.radius_z,
        )
        bowl = np.exp(-(normalized**2.4))
        output = output - bowl * basin.depth
    return output


def _apply_rivers(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
) -> tuple[np.ndarray, np.ndarray]:
    output_height = terrain
    output_water = water
    for river in recipe.rivers:
        distance, progress = polyline_distance_and_progress(x, z, river.path)
        width = river.width * (1.0 + (river.widening - 1.0) * progress)
        channel = np.exp(-((distance / np.maximum(width, 1.0e-6)) ** 2))
        output_height = output_height - channel * river.depth
        river_water = output_height + 0.75
        output_water = np.where(channel > 0.16, np.maximum(output_water, river_water), output_water)
    return output_height, output_water


def generate_heightfield(
    recipe: TerrainRecipe,
    *,
    width: int,
    height: int,
    seed: int,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
    cell_size: float = 1.0,
) -> Heightfield:
    """Evaluate one recipe into contiguous float32 scalar fields."""

    x, z = _coordinate_grid(width, height, origin_x, origin_z, cell_size)
    terrain = np.full(x.shape, recipe.base_height, dtype=np.float64)
    for layer in recipe.base_noise:
        terrain += layer.amplitude * sample_noise(x, z, layer, seed)
    terrain = _apply_basins(terrain, x, z, recipe)
    terrain = _apply_mountains(terrain, x, z, recipe, seed)

    moisture = sample_noise(x, z, recipe.moisture_noise, seed + 100_003)
    moisture = np.clip((moisture + 1.0) * 0.5, 0.0, 1.0)
    water = np.where(terrain < recipe.sea_level, recipe.sea_level, -1.0)
    terrain, water = _apply_rivers(terrain, water, x, z, recipe)
    water = np.where(water >= 0.0, np.maximum(water, terrain), -1.0)

    return Heightfield(
        height=np.ascontiguousarray(terrain, dtype=np.float32),
        moisture=np.ascontiguousarray(moisture, dtype=np.float32),
        water_level=np.ascontiguousarray(water, dtype=np.float32),
    )
