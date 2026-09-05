"""寒带积雪与冰川质量平衡场。

积雪量由高程、寒带权重、降雪噪声和地形迎风程度共同决定；消融量由
低海拔、向阳坡和气候过渡共同决定。结果保持为连续标量场，本阶段不直接修改
冰川路径、地形高度或 Minecraft 方块。

算法结构参考公开冰川模拟中 accumulation / ablation / mass balance 的分层：
- https://github.com/oargudo/glaciers

本文件只参考公开算法思路，代码为本项目独立实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..noise import sample_noise
from ..terrain_config import GlacialSystem
from .wind_snow import sample_wind_snow_field


@dataclass(frozen=True)
class GlacialMassBalanceField:
    """相对积雪量和有符号冰川质量平衡。

    ``snow_accumulation`` 始终非负；``mass_balance`` 大于零表示净积累，
    小于零表示净消融。风雪字段记录同一方向场，供覆盖层和 Server 查询。
    """

    snow_accumulation: np.ndarray
    mass_balance: np.ndarray
    wind_snow_alignment: np.ndarray
    wind_snow_response: np.ndarray


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _axis_spacing(values: np.ndarray, axis: int) -> float:
    """读取规则世界坐标网格的间距；单行/单列输入使用单位间距。"""

    if values.shape[axis] < 2:
        return 1.0
    difference = np.diff(values, axis=axis)
    spacing = float(np.median(np.abs(difference)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("glacial mass-balance coordinates must be regularly ordered")
    return spacing


def _terrain_gradient(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回世界 X/Z 方向坡度，兼容只有一行或一列的诊断输入。"""

    dx = _axis_spacing(x, axis=1)
    dz = _axis_spacing(z, axis=0)
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


def _system_mass_balance(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cold_weight: np.ndarray,
    system: GlacialSystem,
    seed: int,
    sea_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单个冰川系统的域权重、积雪和质量平衡。"""

    distance = np.hypot(x - system.seed_center.x, z - system.seed_center.z)
    domain = _smoothstep(1.0 - distance / system.seed_extent)
    active_weight = domain * cold_weight

    snowline = sea_level + system.equilibrium_line_elevation
    elevation_position = (
        terrain - (snowline - system.equilibrium_transition * 0.5)
    ) / system.equilibrium_transition
    elevation_factor = _smoothstep(elevation_position)

    gradient_x, gradient_z = _terrain_gradient(terrain, x, z)
    slope = np.hypot(gradient_x, gradient_z)
    slope_response = np.tanh(slope / system.mass_balance_slope_scale)

    # 与可见覆盖层复用同一个有符号风雪场，确保法线、阈值和方向完全一致。
    wind_field = sample_wind_snow_field(terrain, x, z, (system,))
    orographic_response = np.tanh(
        -wind_field.normal_alignment / system.mass_balance_slope_scale
    )
    windward = np.maximum(orographic_response, 0.0)
    leeward = np.maximum(-orographic_response, 0.0)
    wind_multiplier = (
        1.0
        + system.windward_accumulation_strength * windward
        + system.leeward_drift_strength * leeward
    )

    snowfall_noise = sample_noise(x, z, system.snowfall_noise, seed)
    snowfall_variation = np.maximum(
        1.0 + system.snowfall_variability * snowfall_noise,
        0.0,
    )
    accumulation = (
        active_weight
        * elevation_factor
        * system.snowfall_rate
        * snowfall_variation
        * wind_multiplier
    )

    # 坡面朝向使用下坡方向表示：配置为 90 度时，朝 +Z 下倾的坡面获得
    # 最大日照。平地没有方向，因此其 solar_exposure 为零。
    downhill_x = np.divide(
        -gradient_x,
        slope,
        out=np.zeros_like(gradient_x),
        where=slope > 1.0e-9,
    )
    downhill_z = np.divide(
        -gradient_z,
        slope,
        out=np.zeros_like(gradient_z),
        where=slope > 1.0e-9,
    )
    sun_angle = math.radians(system.sun_facing_angle_degrees)
    sun_x = math.cos(sun_angle)
    sun_z = math.sin(sun_angle)
    solar_exposure = np.maximum(downhill_x * sun_x + downhill_z * sun_z, 0.0)
    solar_exposure *= slope_response

    low_elevation = 1.0 - elevation_factor
    ablation = domain * system.ablation_rate * (
        system.minimum_ablation
        + system.low_elevation_ablation_strength * low_elevation
        + system.solar_ablation_strength * solar_exposure
        + system.transition_ablation_strength * (1.0 - cold_weight)
    )
    return active_weight, accumulation, accumulation - ablation


def sample_glacial_mass_balance(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    systems: tuple[GlacialSystem, ...],
    seed: int,
    sea_level: float,
    *,
    cold_weight: np.ndarray | None = None,
) -> GlacialMassBalanceField:
    """生成积雪与质量平衡场，不修改输入地形。

    多个冰川系统重叠时，由当前位置 ``domain * cold_weight`` 最大的系统
    提供结果，避免相对降雪量因系统叠加而无意义翻倍。
    """

    if terrain.ndim != 2 or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("glacial mass-balance fields must share a two-dimensional shape")
    if 0 in terrain.shape:
        raise ValueError("glacial mass-balance fields must not be empty")
    if cold_weight is not None and cold_weight.shape != terrain.shape:
        raise ValueError("glacial mass-balance cold weight must match terrain")
    if not math.isfinite(sea_level):
        raise ValueError("glacial mass-balance sea level must be finite")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("glacial mass-balance inputs must be finite")

    cold = (
        np.ones(terrain.shape, dtype=np.float64)
        if cold_weight is None
        else np.clip(np.asarray(cold_weight, dtype=np.float64), 0.0, 1.0)
    )
    if not np.isfinite(cold).all():
        raise ValueError("glacial mass-balance cold weight must be finite")
    owner_weight = np.zeros(terrain.shape, dtype=np.float64)
    accumulation = np.zeros(terrain.shape, dtype=np.float64)
    mass_balance = np.zeros(terrain.shape, dtype=np.float64)
    wind_alignment = np.zeros(terrain.shape, dtype=np.float64)
    wind_response = np.zeros(terrain.shape, dtype=np.float64)

    for index, system in enumerate(systems):
        if not system.mass_balance_enabled:
            continue
        system_seed = seed + index * 1_301_071 + 29_003
        weight, candidate_accumulation, candidate_balance = _system_mass_balance(
            np.asarray(terrain, dtype=np.float64),
            x,
            z,
            cold,
            system,
            system_seed,
            sea_level,
        )
        replace = weight > owner_weight
        owner_weight[replace] = weight[replace]
        accumulation[replace] = candidate_accumulation[replace]
        mass_balance[replace] = candidate_balance[replace]
        # 质量平衡内部的响应字段保留给 Server/调试使用；覆盖层会在最终
        # 冰川地形上重新采样，以免冰川 carve 改变坡向后仍使用旧法线。
        system_wind = sample_wind_snow_field(
            np.asarray(terrain, dtype=np.float64), x, z, (system,)
        )
        wind_alignment[replace] = system_wind.normal_alignment[replace]
        wind_response[replace] = system_wind.response[replace]

    return GlacialMassBalanceField(
        snow_accumulation=np.ascontiguousarray(accumulation, dtype=np.float64),
        mass_balance=np.ascontiguousarray(mass_balance, dtype=np.float64),
        wind_snow_alignment=np.ascontiguousarray(wind_alignment, dtype=np.float64),
        wind_snow_response=np.ascontiguousarray(wind_response, dtype=np.float64),
    )


__all__ = ["GlacialMassBalanceField", "sample_glacial_mass_balance"]
