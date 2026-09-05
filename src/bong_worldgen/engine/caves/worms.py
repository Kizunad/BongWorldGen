"""Godot Voxel 风格的洞穴 worm 场。

算法参考 Godot Voxel 的程序化生成文档：
https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md

文档描述的核心是：对 2D 噪声平方后按阈值截取 worm 通道，用 Y 方向抛物线
调制截面，再用低频噪声制造收缩/死胡同，并用另一层噪声扰动洞穴高度。
噪声采样接口同时参考 FastNoiseLite 的公开 3D/domain-warp 设计：
https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain_config import NoiseLayer
from ..underground_config import CaveNetwork
from ..noise import sample_noise


@dataclass(frozen=True)
class WormField:
    """一条洞穴网络在 XZ 平面上的半径调制和垂直偏移。"""

    radius_scale: np.ndarray
    vertical_offset: np.ndarray
    corridor: np.ndarray
    dead_end: np.ndarray


def sample_worm_field(
    x: np.ndarray,
    z: np.ndarray,
    network: CaveNetwork,
    seed: int,
) -> WormField:
    """采样 2D worm 通道、死胡同调制和垂直扰动。"""

    raw = sample_noise(x, z, network.worm_noise, seed + 71_203)
    squared = np.square(raw)
    corridor = np.clip(
        (network.worm_threshold - squared) / network.worm_threshold,
        0.0,
        1.0,
    )
    # 通道外侧仍保留很窄的过渡，避免离散网格产生整齐的切断面；
    # dead_end_strength 再把低 corridor 区域压缩成自然的死胡同。
    radius_scale = 0.18 + 0.82 * corridor
    radius_scale *= 1.0 - 0.45 * min(network.dead_end_strength, 1.0) * (1.0 - corridor)
    # Godot Voxel 文档的第二个调制层：低频噪声被加到 worm 阈值，
    # 让通道在部分区域收缩并自然结束，而不是无限延伸。
    dead_end_layer = NoiseLayer(
        kind="fbm",
        scale=network.worm_noise.scale * 2.5,
        amplitude=1.0,
        octaves=2,
        lacunarity=network.worm_noise.lacunarity,
        gain=network.worm_noise.gain,
        seed_offset=network.worm_noise.seed_offset,
    )
    dead_end = np.clip(
        1.0 + network.dead_end_strength * sample_noise(
            x,
            z,
            dead_end_layer,
            seed + 89_431,
        ),
        0.0,
        1.0,
    )
    vertical_offset = network.vertical_warp * sample_noise(
        x,
        z,
        network.vertical_warp_noise,
        seed + 83_917,
    )
    return WormField(radius_scale, vertical_offset, corridor, dead_end)
