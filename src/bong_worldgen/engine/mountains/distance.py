"""山脉脊柱的扭曲距离场。

实现采用“折线脊柱 + 世界坐标 domain warp + 有限宽度距离”的组合。
思路参考以下公开资料，没有复制其源码：

- https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
- https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..noise import sample_noise
from ..terrain_config import MountainRange


@dataclass(frozen=True)
class MountainDistanceField:
    """每个地表采样点相对山脉脊柱的几何信息。"""

    distance: np.ndarray
    progress: np.ndarray
    local_width: np.ndarray
    normalized_distance: np.ndarray


def mountain_distance_field(
    x: np.ndarray,
    z: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> MountainDistanceField:
    """计算 seed 驱动、分块稳定的山脉脊柱距离场。"""

    if x.shape != z.shape:
        raise ValueError("mountain coordinate fields must share a shape")

    warp_x = sample_noise(x, z, mountain.spine_warp_noise, seed + 31_003)
    warp_z = sample_noise(x, z, mountain.spine_warp_noise, seed + 31_037)
    warped_x = x + warp_x * mountain.spine_warp_strength
    warped_z = z + warp_z * mountain.spine_warp_strength
    distance, progress = polyline_distance_and_progress(
        warped_x,
        warped_z,
        mountain.path,
    )

    width_noise = sample_noise(x, z, mountain.width_noise, seed + 31_071)
    local_width = mountain.width * (
        1.0 + np.clip(width_noise, -1.0, 1.0) * mountain.width_variation
    )
    local_width = np.maximum(local_width, mountain.width * 0.08)
    normalized = np.clip(distance / local_width, 0.0, 1.0)
    return MountainDistanceField(distance, progress, local_width, normalized)


__all__ = ["MountainDistanceField", "mountain_distance_field"]
