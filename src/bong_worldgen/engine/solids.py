"""Generic detached solids, independent of region layout and output format."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .models import Heightfield, NoiseLayer, SPAN_MAX_SPANS, TerrainRecipe
from .noise import sample_noise


def add_floating_islands(
    field: Heightfield, x: np.ndarray, z: np.ndarray, recipe: TerrainRecipe, seed: int,
) -> Heightfield:
    if not recipe.floating_islands:
        return field
    spans = field.solid_spans.copy()
    heights = field.height.copy()
    water = field.water_level.copy()
    for index, island in enumerate(recipe.floating_islands):
        radius = np.hypot((x - island.center.x) / island.radius_x,
                          (z - island.center.z) / island.radius_z)
        dome = np.sqrt(np.clip(1 - radius**2, 0, 1))
        top = island.height + island.relief * dome + 2 * sample_noise(
            x, z, NoiseLayer(scale=120, octaves=2), seed + 151 * index,
        )
        bottom = top - island.thickness * dome
        active = (radius < 1) & (top - bottom >= 1)
        if not np.any(active):
            continue
        counts = np.count_nonzero(spans[..., 0] != 32767, axis=-1)
        if np.any(active & (counts >= SPAN_MAX_SPANS)):
            raise ValueError("floating island exceeds four solid spans")
        # A detached island is an additional top span. Intersecting islands or
        # ground are rejected rather than silently producing inverted spans.
        floor, ceiling = np.rint(bottom).astype(np.int16), np.rint(top).astype(np.int16)
        if np.any(active & (floor <= spans[..., 0, 1] + 1)):
            raise ValueError("floating islands must be detached above existing solids")
        shifted = spans.copy()
        shifted[..., 1:, :] = spans[..., :-1, :]
        shifted[..., 0, 0], shifted[..., 0, 1] = floor, ceiling
        spans = np.where(active[..., None, None], shifted, spans)
        heights = np.where(active, top, heights).astype(np.float32)
        # There is no single-plane representation of water under a detached
        # island. Existing ground water is omitted beneath that cap.
        water = np.where(active, -1, water).astype(np.float32)
    return replace(field, height=heights, water_level=water, solid_spans=spans)
