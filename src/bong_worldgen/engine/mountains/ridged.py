"""Ridged Multifractal 山脊噪声。

每个 octave 先把连续噪声折叠成零等值线附近的尖脊，再用上一层的脊线
强度限制下一层细节。这样高频褶皱会附着在较大的山脊上，而不是均匀铺满
整张高度图。

算法组织参考 FastNoiseLite 的 ``GenFractalRidged``，未复制其源码：
https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h
"""

from __future__ import annotations

import numpy as np

from ..noise import value_noise
from ..terrain_config import RidgedMultifractal


def sample_ridged_multifractal(
    x: np.ndarray,
    z: np.ndarray,
    config: RidgedMultifractal,
    seed: int,
) -> np.ndarray:
    """返回 ``[0, 1]`` 范围内、由世界坐标和 seed 决定的山脊场。"""

    if x.shape != z.shape:
        raise ValueError("ridged multifractal coordinate fields must share a shape")

    sample_x = np.asarray(x, dtype=np.float64)
    sample_z = np.asarray(z, dtype=np.float64)
    if config.warp_strength:
        warp_x = value_noise(
            sample_x,
            sample_z,
            scale=config.warp_scale,
            seed=seed + config.seed_offset + 7_003,
        )
        warp_z = value_noise(
            sample_x,
            sample_z,
            scale=config.warp_scale,
            seed=seed + config.seed_offset + 7_037,
        )
        sample_x = sample_x + warp_x * config.warp_strength
        sample_z = sample_z + warp_z * config.warp_strength

    output = np.zeros(x.shape, dtype=np.float64)
    feedback = np.ones(x.shape, dtype=np.float64)
    frequency = 1.0
    amplitude = 1.0
    normalization = 0.0

    for octave in range(config.octaves):
        source = value_noise(
            sample_x * frequency,
            sample_z * frequency,
            scale=config.scale,
            seed=seed + config.seed_offset + octave * 1_013,
        )
        ridge = np.clip(config.ridge_offset - np.abs(source), 0.0, 1.0)
        ridge = np.power(ridge, config.ridge_power)
        if octave:
            ridge *= feedback
        output += ridge * amplitude
        normalization += amplitude

        # 上一 octave 越接近山谷，下一 octave 越弱；细纹因此聚集在脊上。
        feedback = np.clip(ridge * config.ridge_gain, 0.0, 1.0)
        frequency *= config.lacunarity
        amplitude *= config.persistence

    return np.clip(output / max(normalization, 1.0e-12), 0.0, 1.0)


__all__ = ["sample_ridged_multifractal"]
