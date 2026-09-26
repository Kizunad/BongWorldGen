"""World-coordinate surface treatments derived from authored landforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..engine import NoiseLayer
from ..engine.noise import sample_noise
from .layout import ZoneBlend

if TYPE_CHECKING:
    from .terrain import ZoneTerrain


def scorch_mask(composer: ZoneTerrain, x: np.ndarray, z: np.ndarray,
                blend: ZoneBlend | None = None) -> np.ndarray:
    """Mark charred impact floors, with irregular edges and zone-weight fading."""

    x, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64))
    blend = composer.index.query(x, z) if blend is None else blend
    mask = np.zeros(x.shape, dtype=bool)
    for part in blend.contributions:
        if part.zone.terrain_profile != "tribulation_scorch":
            continue
        threshold = 0.65 + 0.06 * sample_noise(x, z, NoiseLayer(scale=17, octaves=2,
                                                             seed_offset=149), composer.seed)
        for basin in composer.recipes[part.zone.name].basins:
            radius = np.hypot((x - basin.center.x) / basin.radius_x,
                              (z - basin.center.z) / basin.radius_z)
            mask |= np.exp(-(radius ** 2.4)) * part.weight > threshold
    return mask
