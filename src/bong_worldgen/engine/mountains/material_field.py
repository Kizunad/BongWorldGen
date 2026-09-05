"""群山内部的场驱动雪、冰与蓝冰材质。

材质场只消费已经完成的连续地形高度，不改写地形本身。山脉范围由
``Mountain Spine + Distance Field`` 提供，雪线和材质边界则由多尺度噪声、
坡度、暴露方向、气候与冰川权重共同决定。噪声接口的组织方式参考
FastNoiseLite（没有复制其实现）：
https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from ..noise import sample_noise
from ..constants import MOUNTAIN_ROCK_PALETTE
from ..terrain_config import MountainMaterialSettings, MountainRange
from .distance import mountain_distance_field


# 保持与现有 Anvil/raster 适配器的稳定 palette 顺序。
MOUNTAIN_SURFACE_PALETTE = (
    "minecraft:snow_block",
    "minecraft:powder_snow",
    "minecraft:ice",
    "minecraft:packed_ice",
    "minecraft:blue_ice",
    "minecraft:gravel",
    *MOUNTAIN_ROCK_PALETTE,
)


@dataclass(frozen=True)
class MountainMaterialField:
    """一次群山材质采样的可查询场。"""

    material_id: np.ndarray
    mountain_weight: np.ndarray
    snowline: np.ndarray
    snow_score: np.ndarray
    slope_angle: np.ndarray
    exposure: np.ndarray
    rock_exposure: np.ndarray


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _grid_spacing(x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """从规则世界坐标网格读取 X/Z 间距；单行/单列网格回退为一格。"""

    if x.shape[1] > 1:
        spacing_x = float(np.median(np.abs(np.diff(x[0, :]))))
    else:
        spacing_x = 1.0
    if z.shape[0] > 1:
        spacing_z = float(np.median(np.abs(np.diff(z[:, 0]))))
    else:
        spacing_z = 1.0
    if spacing_x <= 0.0 or spacing_z <= 0.0:
        raise ValueError("mountain material coordinate spacing must be positive")
    return spacing_x, spacing_z


def _terrain_slope_and_exposure(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    direction_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算坡度角和朝向给定暴露方向的归一化坡向。"""

    spacing_x, spacing_z = _grid_spacing(x, z)
    if terrain.shape[0] < 2 and terrain.shape[1] < 2:
        gradient_x = np.zeros_like(terrain, dtype=np.float64)
        gradient_z = np.zeros_like(terrain, dtype=np.float64)
    elif terrain.shape[0] < 2:
        gradient_x = np.gradient(terrain.astype(np.float64, copy=False), spacing_x, axis=1)
        gradient_z = np.zeros_like(terrain, dtype=np.float64)
    elif terrain.shape[1] < 2:
        gradient_z = np.gradient(terrain.astype(np.float64, copy=False), spacing_z, axis=0)
        gradient_x = np.zeros_like(terrain, dtype=np.float64)
    else:
        gradient_z, gradient_x = np.gradient(
            terrain.astype(np.float64, copy=False), spacing_z, spacing_x
        )
    slope = np.hypot(gradient_x, gradient_z)
    slope_angle = np.degrees(np.arctan(slope))
    direction = radians(direction_degrees % 360.0)
    facing = (gradient_x * cos(direction) + gradient_z * sin(direction)) / np.maximum(
        slope, 1.0e-12
    )
    # 只把朝向给定方向的坡面视为暴露面；背阴面返回零，方便直接作为
    # 雪分数的扣减项。这样不会把平坦地形的数值噪声误判成朝向。
    exposure = np.clip(facing, 0.0, 1.0)
    exposure[slope <= 1.0e-12] = 0.0
    return np.ascontiguousarray(slope_angle), np.ascontiguousarray(exposure)


def _mountain_fields(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回山脉权重；山外为零。"""

    weight = np.zeros(terrain.shape, dtype=np.float64)
    for index, mountain in enumerate(mountains):
        distance_field = mountain_distance_field(
            x,
            z,
            mountain,
            seed + index * 1_003_049,
        )
        # 仅在有限支持域内启用材质；edge_blend 让山脚材质也连续淡出，
        # 不会把普通平原误判成群山。
        support = _smoothstep(
            (1.0 - distance_field.normalized_distance) / max(mountain.edge_blend, 1.0e-6)
        )
        weight = np.maximum(weight, support)
    return np.ascontiguousarray(weight)


def sample_mountain_material_field(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
    settings: MountainMaterialSettings,
    *,
    climate_cold_weight: np.ndarray | None = None,
    glacier_weight: np.ndarray | None = None,
) -> MountainMaterialField:
    """采样群山材质场；山外 ``material_id`` 始终为零。"""

    terrain = np.asarray(terrain, dtype=np.float64)
    if terrain.ndim != 2 or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("mountain material fields must share a two-dimensional shape")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("mountain material fields must contain finite values")
    if climate_cold_weight is not None and climate_cold_weight.shape != terrain.shape:
        raise ValueError("climate cold weight must have the same shape as terrain")
    if glacier_weight is not None and glacier_weight.shape != terrain.shape:
        raise ValueError("glacier weight must have the same shape as terrain")

    zero = np.zeros(terrain.shape, dtype=np.float64)
    if not settings.enabled or not mountains:
        return MountainMaterialField(
            material_id=np.zeros(terrain.shape, dtype=np.uint8),
            mountain_weight=zero.astype(np.float32),
            snowline=np.zeros(terrain.shape, dtype=np.float32),
            snow_score=zero.astype(np.float32),
            slope_angle=zero.astype(np.float32),
            exposure=zero.astype(np.float32),
            rock_exposure=zero.astype(np.float32),
        )

    mountain_weight = _mountain_fields(terrain, x, z, mountains, seed)
    slope_angle, exposure = _terrain_slope_and_exposure(
        terrain,
        x,
        z,
        settings.exposure_direction_degrees,
    )
    large_noise = sample_noise(x, z, settings.snowline_noise, seed + settings.seed_offset + 11)
    detail_noise = sample_noise(
        x,
        z,
        settings.snowline_detail_noise,
        seed + settings.seed_offset + 23,
    )
    snowline = (
        settings.snowline_base_elevation
        + large_noise * settings.snowline_large_amplitude
        + detail_noise * settings.snowline_small_amplitude
    )
    above_snowline = 0.5 + 0.5 * np.tanh(
        (terrain - snowline) / settings.snowline_transition
    )

    slope_t = np.clip(
        (slope_angle - settings.snow_slope_full_angle)
        / (settings.snow_slope_cutoff_angle - settings.snow_slope_full_angle),
        0.0,
        1.0,
    )
    slope_adjustment = settings.snow_slope_bonus + (
        -settings.snow_slope_penalty - settings.snow_slope_bonus
    ) * _smoothstep(slope_t)
    cold_weight = (
        np.zeros(terrain.shape, dtype=np.float64)
        if climate_cold_weight is None
        else np.clip(climate_cold_weight.astype(np.float64, copy=False), 0.0, 1.0)
    )
    glacier = (
        np.zeros(terrain.shape, dtype=np.float64)
        if glacier_weight is None
        else np.clip(glacier_weight.astype(np.float64, copy=False), 0.0, 1.0)
    )
    snow_score = np.clip(
        above_snowline
        + settings.temperature_strength * cold_weight
        + settings.glacier_bonus * glacier
        + slope_adjustment
        - settings.exposure_strength * exposure,
        0.0,
        1.0,
    )
    # 坡度角转为归一化陡峭度，再套 smoothstep。这个场只表达“该处多容易
    # 裸岩”，不把悬崖高度改写成材质边界；阈值和噪声均来自配方。算法形式
    # 参考 FastNoiseLite 的连续噪声/领域采样用法（未复制其实现）：
    # https://github.com/Auburn/FastNoiseLite
    normalized_slope = np.clip(slope_angle / 90.0, 0.0, 1.0)
    rock_exposure = _smoothstep(
        (normalized_slope - settings.rock_exposure_start_slope)
        / max(settings.rock_exposure_full_slope - settings.rock_exposure_start_slope, 1.0e-12)
    )
    boundary_noise = sample_noise(
        x,
        z,
        settings.material_noise,
        seed + settings.seed_offset + 37,
    )
    classification = np.clip(
        snow_score + boundary_noise * settings.boundary_noise_strength,
        0.0,
        1.0,
    )
    rock_noise = (
        sample_noise(
            x,
            z,
            settings.rock_exposure_noise,
            seed + settings.seed_offset + 53,
        )
        + 1.0
    ) * 0.5
    material_id = np.zeros(terrain.shape, dtype=np.uint8)
    active = mountain_weight >= settings.minimum_mountain_weight
    # 低于粉雪门槛的区域保留原山石材质，避免寒带气候权重把整条山脚涂白。
    active &= classification >= settings.powder_threshold
    material_id[active & (classification < settings.snow_threshold)] = 2
    powder_or_snow = active & (classification >= settings.snow_threshold)
    material_id[powder_or_snow] = 1
    material_id[active & (classification >= settings.ice_threshold)] = 3
    material_id[active & (classification >= settings.packed_ice_threshold)] = 4
    material_id[active & (classification >= settings.blue_ice_threshold)] = 5
    # 裸岩是陡峭山脊和悬崖上的独立覆盖判定，优先于雪/冰，形成雪-岩-冰-
    # 岩-雪的混合剖面。低坡暴露场接近零，不会把普通山脚误刷成岩壁。
    rock_probability = rock_exposure * settings.rock_exposure_max_probability
    rock_mask = active & (rock_probability > rock_noise)
    normalized_rocks = tuple(
        material.strip().lower().removeprefix("minecraft:").replace("-", "_")
        for material in settings.rock_palette
    )
    rock_choices = np.asarray(
        [MOUNTAIN_SURFACE_PALETTE.index(f"minecraft:{material}") + 1 for material in normalized_rocks],
        dtype=np.uint8,
    )
    rock_variant_noise = (
        sample_noise(
            x,
            z,
            settings.rock_exposure_noise,
            seed + settings.seed_offset + 67,
        )
        + 1.0
    ) * 0.5
    rock_variant_index = np.minimum(
        (rock_variant_noise * len(rock_choices)).astype(np.int64),
        len(rock_choices) - 1,
    )
    material_id[rock_mask] = rock_choices[rock_variant_index[rock_mask]]

    return MountainMaterialField(
        material_id=np.ascontiguousarray(material_id),
        mountain_weight=np.ascontiguousarray(mountain_weight, dtype=np.float32),
        snowline=np.ascontiguousarray(snowline, dtype=np.float32),
        snow_score=np.ascontiguousarray(snow_score, dtype=np.float32),
        slope_angle=np.ascontiguousarray(slope_angle, dtype=np.float32),
        exposure=np.ascontiguousarray(exposure, dtype=np.float32),
        rock_exposure=np.ascontiguousarray(rock_exposure, dtype=np.float32),
    )


def mountain_surface_materials(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
    settings: MountainMaterialSettings,
    *,
    climate_cold_weight: np.ndarray | None = None,
    glacier_weight: np.ndarray | None = None,
) -> np.ndarray:
    """返回仅包含群山材质 ID 的二维层，山外为零。"""

    return sample_mountain_material_field(
        terrain,
        x,
        z,
        mountains,
        seed,
        settings,
        climate_cold_weight=climate_cold_weight,
        glacier_weight=glacier_weight,
    ).material_id


__all__ = [
    "MOUNTAIN_SURFACE_PALETTE",
    "MountainMaterialField",
    "mountain_surface_materials",
    "sample_mountain_material_field",
]
