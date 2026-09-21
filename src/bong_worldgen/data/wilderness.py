"""Stable wilderness categories shared by the raster exporter and viewers.

The numeric ids are part of the generated-world contract.  Keep them stable:
the Rust decoration pass can use the same ids without depending on display
names or colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WildernessType:
    id: int
    key: str
    display_name: str
    color: str
    decoration_tags: tuple[str, ...]


GRASSLAND = 0
MOUNTAINS = 1
LAKE = 2
RIVER = 3

WILDERNESS_PALETTE: tuple[WildernessType, ...] = (
    WildernessType(0, "grassland", "草地", "#6f9f5e", ("grass", "shrub", "tree")),
    WildernessType(1, "mountains", "群山", "#a8adb8", ("stone", "pine", "ore")),
    WildernessType(2, "lake", "湖泊", "#4a80b8", ("reeds", "water_lily", "shore")),
    WildernessType(3, "river", "河流", "#4a70a8", ("reeds", "river_bank", "bridge")),
)

WILDERNESS_BY_ID = {item.id: item for item in WILDERNESS_PALETTE}


def wilderness_palette_manifest() -> list[dict[str, object]]:
    """Return JSON-compatible metadata for the manifest and Rust consumers."""

    return [
        {
            "id": item.id,
            "key": item.key,
            "display_name": item.display_name,
            "color": item.color,
            "decoration_tags": list(item.decoration_tags),
        }
        for item in WILDERNESS_PALETTE
    ]


def classify_wilderness(
    height: np.ndarray,
    water_level: np.ndarray,
    *,
    sea_level: float,
    surface_slope: np.ndarray | None = None,
) -> np.ndarray:
    """Classify each heightfield column into a stable wilderness category.

    This is intentionally a small, explainable classifier.  Water produced at
    sea level is a lake/low-water body; raised channel water is a river.  Land
    above the high relief threshold, or with a steep high-altitude slope, is
    mountains.  Everything else is grassland.

    Callers with world-coordinate neighbors supply surface_slope to keep tile
    edges consistent. Without it, standalone arrays use local grid gradients.
    """

    if height.shape != water_level.shape or height.ndim != 2:
        raise ValueError("height and water_level must be matching two-dimensional arrays")
    if not np.isfinite(height).all() or not np.isfinite(water_level).all():
        raise ValueError("height and water_level must contain finite values")

    result = np.full(height.shape, GRASSLAND, dtype=np.uint8)
    wet = water_level >= 0.0
    # River channels are raised above the sea-level water plane by the raster
    # pipeline.  This keeps river/lake classification independent of width.
    river = wet & (water_level > sea_level + 1.0e-3)
    lake = wet & ~river

    if surface_slope is not None:
        slope = np.asarray(surface_slope)
        if slope.shape != height.shape or not np.isfinite(slope).all() or np.any(slope < 0):
            raise ValueError("surface_slope must be finite, nonnegative and match height")
    elif height.shape[0] < 2 or height.shape[1] < 2:
        slope = np.zeros(height.shape, dtype=np.float64)
    else:
        dz, dx = np.gradient(height.astype(np.float64, copy=False))
        slope = np.hypot(dx, dz)
    mountain = ~wet & (
        (height >= sea_level + 52.0)
        | ((height >= sea_level + 16.0) & (slope >= 0.30))
    )

    result[mountain] = MOUNTAINS
    result[lake] = LAKE
    result[river] = RIVER
    return result


__all__ = [
    "GRASSLAND",
    "MOUNTAINS",
    "LAKE",
    "RIVER",
    "WildernessType",
    "WILDERNESS_BY_ID",
    "WILDERNESS_PALETTE",
    "classify_wilderness",
    "wilderness_palette_manifest",
]
