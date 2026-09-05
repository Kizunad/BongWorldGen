"""洞穴密度场、坐标扭曲和垂直分带计算。

这些函数只计算连续场，不决定洞穴网络数量、不生成资源，也不写最终
spans。算法组织参考 FastNoiseLite domain warp 与 Godot Voxel 洞穴文档；
实现为本项目独立的 NumPy 代码：

https://github.com/Auburn/FastNoiseLite
https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain_config import NoiseLayer
from ..underground_config import CaveNetwork
from ..noise import sample_noise
from ..randomness import stable_text_seed
from .topology import CaveTopology


def smooth_density_union(left: np.ndarray, right: np.ndarray, radius: float) -> np.ndarray:
    """对两个密度场做平滑并集，避免洞道交汇处出现硬切接缝。"""

    if radius <= 0:
        return np.maximum(left, right)
    blend = np.clip(0.5 + 0.5 * (right - left) / radius, 0.0, 1.0)
    return np.maximum(left, right) + radius * blend * (1.0 - blend)


def domain_warp_coordinates(
    x: np.ndarray,
    z: np.ndarray,
    network_name: str,
    scale: float,
    strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用两个独立 seeded 噪声场扭曲洞穴 XZ 坐标。"""

    if strength <= 0.0:
        return x, z
    warp_layer = NoiseLayer(
        kind="fbm",
        scale=scale,
        octaves=2,
        gain=0.5,
        seed_offset=stable_text_seed(network_name) & 0x7FFF,
    )
    offset_x = sample_noise(x, z, warp_layer, seed + 101_021)
    offset_z = sample_noise(x, z, warp_layer, seed + 101_053)
    return x + offset_x * strength, z + offset_z * strength


def axis_slice(axis: np.ndarray, lower: float, upper: float) -> slice | None:
    """返回覆盖世界坐标区间的最小网格切片，并保留一格边界余量。"""

    if axis[-1] < lower or axis[0] > upper:
        return None
    start = max(0, int(np.searchsorted(axis, lower, side="left")) - 1)
    stop = min(axis.size, int(np.searchsorted(axis, upper, side="right")) + 1)
    return slice(start, stop) if start < stop else None


def network_bounds(
    network: CaveNetwork,
    topology: CaveTopology,
) -> tuple[float, float, float, float]:
    """计算洞网在 XZ 平面的保守影响范围。"""

    points = [point for path in topology.paths for point in path]
    points.extend(topology.chambers)
    points.extend(topology.entrances)
    margin = (
        max(network.width, network.chamber_radius, network.entrance_radius)
        + network.domain_warp_strength
        + network.smooth_union
        + 2.0
    )
    return (
        min(point.x for point in points) - margin,
        max(point.x for point in points) + margin,
        min(point.z for point in points) - margin,
        max(point.z for point in points) + margin,
    )


def vertical_tail_lift(
    noise: np.ndarray,
    *,
    depth: float,
    tail_fraction: float,
    rise_ratio: float,
    drop_ratio: float,
    upper_start: float | None = None,
    lower_start: float | None = None,
    upper_end: float | None = None,
    lower_end: float | None = None,
) -> np.ndarray:
    """按分位数生成 90/5/5 的洞穴中心垂直抬升场。"""

    if tail_fraction <= 0.0:
        return np.zeros_like(noise, dtype=np.float64)
    if upper_start is None:
        upper_start = 1.0 - 2.0 * math.sqrt(tail_fraction)
    if lower_start is None:
        lower_start = -upper_start
    if upper_end is None:
        upper_end = 1.0
    if lower_end is None:
        lower_end = -1.0
    upper_span = max(upper_end - upper_start, 1.0e-6)
    lower_span = max(lower_start - lower_end, 1.0e-6)
    upper_tail = np.clip((noise - upper_start) / upper_span, 0.0, 1.0)
    lower_tail = np.clip((lower_start - noise) / lower_span, 0.0, 1.0)
    return upper_tail * depth * rise_ratio - lower_tail * depth * drop_ratio


def stable_vertical_tail_thresholds(
    network: CaveNetwork,
    topology: CaveTopology,
    seed: int,
) -> tuple[float, float, float, float]:
    """在固定网络中心线上校准尾部阈值，避免结果依赖当前 tile。"""

    calibration_points: list[tuple[float, float]] = []
    for path in topology.paths:
        for start, end in zip(path, path[1:]):
            segment_length = math.hypot(end.x - start.x, end.z - start.z)
            sample_count = max(16, int(math.ceil(segment_length / 4.0)))
            calibration_points.extend(
                zip(
                    np.linspace(start.x, end.x, sample_count),
                    np.linspace(start.z, end.z, sample_count),
                )
            )
    if calibration_points:
        calibration_x = np.asarray([point[0] for point in calibration_points])
        calibration_z = np.asarray([point[1] for point in calibration_points])
    else:
        min_x, max_x, min_z, max_z = network_bounds(network, topology)
        calibration_x, calibration_z = np.meshgrid(
            np.linspace(min_x, max_x, 257, dtype=np.float64),
            np.linspace(min_z, max_z, 257, dtype=np.float64),
            indexing="xy",
        )
    warped_x, warped_z = domain_warp_coordinates(
        calibration_x,
        calibration_z,
        network.name,
        network.domain_warp_scale,
        network.domain_warp_strength,
        seed,
    )
    calibration_noise = sample_noise(
        warped_x,
        warped_z,
        network.vertical_warp_noise,
        seed + 92_117,
    )
    tail = network.vertical_tail_fraction
    return (
        float(np.quantile(calibration_noise, tail)),
        float(np.quantile(calibration_noise, 1.0 - tail)),
        float(np.quantile(calibration_noise, tail / 10.0)),
        float(np.quantile(calibration_noise, 1.0 - tail / 10.0)),
    )


__all__ = [
    "axis_slice",
    "domain_warp_coordinates",
    "network_bounds",
    "smooth_density_union",
    "stable_vertical_tail_thresholds",
    "vertical_tail_lift",
]
