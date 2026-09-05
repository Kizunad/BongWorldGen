"""沿 Mountain Spine 生成主峰、次峰和鞍部。

Ridged Multifractal 负责峰鞍分布；峰宽平滑在世界坐标上完成，因此整图与
分块生成结果一致。Ridged 组织方式参考 FastNoiseLite 的公开实现，未复制源码：
https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain_config import MountainRange
from .ridged import sample_ridged_multifractal


_PEAK_FILTER_SAMPLES = 9


def _path_length(mountain: MountainRange) -> float:
    return max(
        sum(
            math.hypot(end.x - start.x, end.z - start.z)
            for start, end in zip(mountain.path, mountain.path[1:])
        ),
        1.0,
    )


def _sample_spine_ridge(
    coordinate: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    """采样沿脊高度场，并按配置宽度消除单体素尖峰。"""

    sample_z = np.zeros_like(coordinate)
    if mountain.spine_peak_width == 0.0:
        return sample_ridged_multifractal(
            coordinate,
            sample_z,
            mountain.spine_ridges,
            seed,
        )

    offsets = np.linspace(
        -mountain.spine_peak_width,
        mountain.spine_peak_width,
        _PEAK_FILTER_SAMPLES,
    )
    sigma = mountain.spine_peak_width * 0.5
    weights = np.exp(-0.5 * np.square(offsets / sigma))
    weights /= np.sum(weights)
    ridge = np.zeros_like(coordinate, dtype=np.float64)
    for offset, weight in zip(offsets, weights):
        ridge += weight * sample_ridged_multifractal(
            coordinate + offset,
            sample_z,
            mountain.spine_ridges,
            seed,
        )
    return ridge


def spine_height_reduction(
    progress: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    """返回每个脊柱位置相对配置峰高的落差。"""

    if mountain.spine_height_variation == 0.0:
        return np.zeros(progress.shape, dtype=np.float64)
    coordinate = progress * _path_length(mountain)
    ridge = _sample_spine_ridge(
        coordinate,
        mountain,
        seed + 31_151,
    )
    return mountain.spine_height_variation * (1.0 - ridge)


__all__ = ["spine_height_reduction"]
