"""峡谷的 domain-warped polyline distance field。"""

from __future__ import annotations

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..terrain_config import Canyon, NoiseLayer, Point
from ..noise import sample_noise
from ..randomness import stable_text_seed


def warped_polyline_distance_and_progress(
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
    canyon: Canyon,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在 domain-warp 后的坐标中计算到折线的距离和弧长进度。

    先以两个独立低频噪声把查询坐标横向扭曲，再复用连续折线 SDF；这比
    对每个像素硬画一条直槽更能产生自然的弯转峡谷。实现参考了 GitHub
    上的距离场与折线采样说明，以及 Godot/FastNoiseLite 的 domain-warp
    工作流，但没有复制外部代码：

    - https://github.com/Tomen/colonies/blob/0bb686f3e5b5c25cc6a9325b116cfd088e946a20/docs/world/01_physical_layer/distance-fields.md
    - https://github.com/alexanderpino/skills/blob/09f8696a3d2040d372bdd282fedddc9c895e8929/terrain-architect/reference-impl/meander.py
    - https://github.com/Auburn/FastNoiseLite
    """

    if canyon.domain_warp_strength <= 0.0:
        return polyline_distance_and_progress(x, z, path)
    name_seed = stable_text_seed(canyon.name)
    layer = NoiseLayer(
        kind="fbm",
        scale=canyon.domain_warp_scale,
        octaves=3,
        gain=0.55,
        seed_offset=name_seed & 0x7FFF,
    )
    offset_x = sample_noise(x, z, layer, seed + 71_001)
    offset_z = sample_noise(x, z, layer, seed + 71_037)
    warped_x = x + offset_x * canyon.domain_warp_strength
    warped_z = z + offset_z * canyon.domain_warp_strength
    return polyline_distance_and_progress(warped_x, warped_z, path)


__all__ = ["warped_polyline_distance_and_progress"]
