"""按主风向计算迎风积雪与背风削薄场。

这里使用三维地形表面法线的水平投影与风向点积：点积小于 ``-0.3`` 的坡面
视为迎风积雪面，点积大于 ``0.5`` 的坡面视为背风削薄面。实现参考
冰川积累/消融的分层思路：
https://github.com/oargudo/glaciers
以及
FastNoiseLite 文档中“由连续场驱动材质/特征”的组织方式，但没有复制其
源码；风场本身是解析几何计算，保证跨 tile 一致：
https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..terrain_config import GlacialSystem


@dataclass(frozen=True)
class WindSnowField:
    """一次采样得到的风雪查询场。

    ``response`` 为有符号厚度修正：正值表示积雪，负值表示风蚀。
    ``normal_alignment`` 是三维地形表面法线的水平投影与风向的点积，范围为
    ``[-1, 1]``。
    """

    normal_alignment: np.ndarray
    accumulation: np.ndarray
    erosion: np.ndarray
    response: np.ndarray


def _grid_spacing(values: np.ndarray, axis: int) -> float:
    if values.shape[axis] < 2:
        return 1.0
    difference = np.diff(values, axis=axis)
    spacing = float(np.median(np.abs(difference)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("wind-snow coordinates must be regularly ordered")
    return spacing


def _terrain_gradient(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dx = _grid_spacing(x, axis=1)
    dz = _grid_spacing(z, axis=0)
    gradient_x = (
        np.gradient(terrain, dx, axis=1)
        if terrain.shape[1] > 1
        else np.zeros_like(terrain, dtype=np.float64)
    )
    gradient_z = (
        np.gradient(terrain, dz, axis=0)
        if terrain.shape[0] > 1
        else np.zeros_like(terrain, dtype=np.float64)
    )
    return gradient_x, gradient_z


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def surface_wind_normal_alignment(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    direction_degrees: float,
) -> np.ndarray:
    """返回三维地形表面法线与水平风向的点积。

    ``-1`` 表示风吹向上坡面，``+1`` 表示风吹向下坡面；平地返回零。
    这一定义直接对应风向积雪的几何判定，不依赖当前 tile 的边界。
    """

    gradient_x, gradient_z = _terrain_gradient(terrain, x, z)
    slope = np.hypot(gradient_x, gradient_z)
    angle = math.radians(direction_degrees % 360.0)
    # 表面法线 n = (-dH/dx, 1, -dH/dz) / sqrt(1+|grad H|^2)。风向为
    # 水平向量，因此点积自然随坡度变陡而增大，平地不会被误判为迎风坡。
    alignment = (
        -(gradient_x * math.cos(angle) + gradient_z * math.sin(angle))
        / np.sqrt(1.0 + slope * slope)
    )
    return np.clip(alignment, -1.0, 1.0)


def _sample_system_wind_snow(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    system: GlacialSystem,
) -> tuple[np.ndarray, WindSnowField]:
    gradient_x, gradient_z = _terrain_gradient(terrain, x, z)
    slope = np.hypot(gradient_x, gradient_z)
    # 水平上坡方向就是地形向外“迎风”的法线；平地没有可靠坡向，响应归零。
    alignment = surface_wind_normal_alignment(
        terrain,
        x,
        z,
        system.prevailing_wind_angle_degrees,
    )

    distance = np.hypot(x - system.seed_center.x, z - system.seed_center.z)
    domain = _smoothstep(1.0 - distance / system.seed_extent)
    slope_gate = np.tanh(slope / system.mass_balance_slope_scale)
    # 用户定义的阈值：dot < -0.3 积雪，dot > 0.5 风蚀；中间区域平滑过渡。
    accumulation_threshold = system.wind_snow_accumulation_threshold
    erosion_threshold = system.wind_snow_erosion_threshold
    accumulation_gate = _smoothstep(
        (-alignment - accumulation_threshold) / (1.0 - accumulation_threshold)
    )
    erosion_gate = _smoothstep(
        (alignment - erosion_threshold) / (1.0 - erosion_threshold)
    )
    # accumulation/erosion 保留为归一化的方向压力，便于覆盖层按概率消费；
    # response 再应用配方强度，作为统一的有符号厚度修正。
    accumulation = domain * slope_gate * accumulation_gate
    erosion = domain * slope_gate * erosion_gate
    active_alignment = alignment * domain
    response = (
        system.windward_accumulation_strength * accumulation
        - system.leeward_drift_strength * erosion
    )
    field = WindSnowField(
        normal_alignment=np.ascontiguousarray(active_alignment, dtype=np.float64),
        accumulation=np.ascontiguousarray(accumulation, dtype=np.float64),
        erosion=np.ascontiguousarray(erosion, dtype=np.float64),
        response=np.ascontiguousarray(response, dtype=np.float64),
    )
    return domain, field


def sample_wind_snow_field(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    systems: tuple[GlacialSystem, ...],
) -> WindSnowField:
    """为多个冰川系统选择一个稳定的风雪响应场。

    系统重叠时由 ``domain`` 更大的系统拥有该位置，避免同一坡面叠加多次
    风雪厚度；没有启用的系统不产生响应。输入不被修改，输出均为连续数组。
    """

    terrain = np.asarray(terrain, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if terrain.ndim != 2 or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("wind-snow fields must share a two-dimensional shape")
    if 0 in terrain.shape:
        raise ValueError("wind-snow fields must not be empty")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("wind-snow fields must contain finite values")

    shape = terrain.shape
    owner = np.zeros(shape, dtype=np.float64)
    alignment = np.zeros(shape, dtype=np.float64)
    accumulation = np.zeros(shape, dtype=np.float64)
    erosion = np.zeros(shape, dtype=np.float64)
    response = np.zeros(shape, dtype=np.float64)
    for system in systems:
        if not system.wind_snow_enabled:
            continue
        domain, candidate = _sample_system_wind_snow(terrain, x, z, system)
        replace = domain > owner
        owner[replace] = domain[replace]
        alignment[replace] = candidate.normal_alignment[replace]
        accumulation[replace] = candidate.accumulation[replace]
        erosion[replace] = candidate.erosion[replace]
        response[replace] = candidate.response[replace]
    return WindSnowField(
        normal_alignment=np.ascontiguousarray(alignment),
        accumulation=np.ascontiguousarray(accumulation),
        erosion=np.ascontiguousarray(erosion),
        response=np.ascontiguousarray(response),
    )


__all__ = [
    "WindSnowField",
    "sample_wind_snow_field",
    "surface_wind_normal_alignment",
]
