"""冰斗和可变宽度 U 型冰川谷的高度场操作。"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..terrain_config import GlacialSystem, Point
from .topology import CirqueSeed


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def cirque_carve_depth(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cirque: CirqueSeed,
    system: GlacialSystem,
    sea_level: float,
) -> np.ndarray:
    """返回冰斗在每个地表列实际削去的连续高度。"""

    dx = x - cirque.center.x
    dz = z - cirque.center.z
    forward = dx * math.cos(cirque.outlet_angle) + dz * math.sin(cirque.outlet_angle)
    lateral = -dx * math.sin(cirque.outlet_angle) + dz * math.cos(cirque.outlet_angle)
    radius_x = system.cirque_radius * cirque.radius_scale * 1.20
    radius_z = system.cirque_radius * cirque.radius_scale
    radial = np.hypot(forward / radius_x, lateral / radius_z)
    bowl = np.clip(1.0 - radial**2, 0.0, 1.0) ** 1.7
    elevation_gate = _smoothstep(
        (terrain - (sea_level + system.cirque_min_elevation))
        / system.cirque_elevation_blend
    )
    return system.cirque_depth * bowl * elevation_gate


def carve_cirque(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cirque: CirqueSeed,
    system: GlacialSystem,
    sea_level: float,
) -> np.ndarray:
    """在高程足够高的位置 carve 一个带出口方向的碗状冰斗。

    冰斗的径向截面采用 accumulation basin 的可控近似；高程门控避免同一套
    寒带配置在低地生成不合理的冰斗。结构参考：
    https://github.com/oargudo/glaciers
    """

    return terrain - cirque_carve_depth(terrain, x, z, cirque, system, sea_level)


def glacier_valley_carve_depth(
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
    system: GlacialSystem,
) -> np.ndarray:
    """返回可变宽度 U 型谷在每个地表列的削去高度。"""

    distance, progress = polyline_distance_and_progress(x, z, path)
    width = system.valley_source_width * (1.0 + system.valley_width_growth * progress)
    normalized = np.clip(distance / np.maximum(width, 1.0e-6), 0.0, 1.0)
    floor_ratio = np.clip(
        system.valley_floor_width / np.maximum(width, 1.0e-6),
        0.05,
        0.85,
    )
    wall_t = np.clip(
        (1.0 - normalized) / np.maximum(1.0 - floor_ratio, 1.0e-6),
        0.0,
        1.0,
    )
    wall_profile = np.where(
        normalized <= floor_ratio,
        1.0,
        wall_t**system.valley_wall_power,
    )
    depth = system.valley_depth * (0.72 + 0.28 * progress)
    return np.where(distance <= width, depth * wall_profile, 0.0)


def carve_glacier_valley(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
    system: GlacialSystem,
) -> np.ndarray:
    """沿冰川中心线 carve 宽度随下游增长的 U 型截面。

    与地表河流的尖锐 V 型截面不同，这里中央保留较宽的平底，谷壁在边缘
    快速抬升。`valley_width_growth` 直接控制源头到出口的宽度变化。
    """

    return terrain - glacier_valley_carve_depth(x, z, path, system)


__all__ = [
    "carve_cirque",
    "carve_glacier_valley",
    "cirque_carve_depth",
    "glacier_valley_carve_depth",
]
