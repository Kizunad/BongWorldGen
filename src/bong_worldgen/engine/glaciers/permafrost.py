"""寒带冻土地表材质。

冻土不是地下裂隙，也不是冰川谷的替代物；它是一层独立的地表语义。
这里用 seeded 的覆盖噪声和材质噪声生成连续斑块，让同一片冻土同时出现
粉雪、雪块、冰、砂砾和冻土，而不是整齐的一条等高线。

材质组合的思路参考公开的冰川/寒带特征放置项目，代码为本项目独立实现：
- https://github.com/oargudo/glaciers
- https://github.com/TheJanusStream/symbios-ground
"""

from __future__ import annotations

import numpy as np

from ..terrain_config import GlacialSystem, NoiseLayer
from ..noise import sample_noise


# 0 表示无冻土覆盖；其余 ID 是真实 Minecraft 方块材质。
PERMAFROST_PALETTE = (
    "minecraft:powder_snow",
    "minecraft:snow_block",
    "minecraft:ice",
    "minecraft:packed_ice",
    "minecraft:gravel",
    "minecraft:coarse_dirt",
    "minecraft:dirt",
    "minecraft:stone",
)


def _permafrost_material_choices(
    x: np.ndarray,
    z: np.ndarray,
    system: GlacialSystem,
    seed: int,
) -> np.ndarray:
    """用连续、扭曲的竞争场选择冻土材质。

    每种材质拥有一张独立的连续 FBM 场，配方权重通过 Gumbel-max 偏置
    参与竞争。两张更大尺度的位移场共同扭曲采样坐标，使材质形成弯曲、
    相互咬合的团块；算法只依赖世界坐标和 seed，分块生成不会产生接缝。
    """

    material_scale = max(system.permafrost_material_scale, 4.0)
    warp_scale = max(system.permafrost_patch_scale * 1.8, material_scale * 3.0)
    warp_strength = material_scale * 1.8
    warp_x = sample_noise(
        x,
        z,
        NoiseLayer(kind="value", scale=warp_scale, octaves=1, seed_offset=907),
        seed,
    )
    warp_z = sample_noise(
        x,
        z,
        NoiseLayer(kind="value", scale=warp_scale, octaves=1, seed_offset=1_411),
        seed,
    )
    warped_x = x + warp_x * warp_strength
    warped_z = z + warp_z * warp_strength

    weights = np.asarray(system.permafrost_material_weights, dtype=np.float64)
    probabilities = weights / weights.sum()
    scores = np.full((len(PERMAFROST_PALETTE), *x.shape), -np.inf, dtype=np.float64)
    for material_index, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        field = sample_noise(
            warped_x,
            warped_z,
            NoiseLayer(
                kind="fbm",
                scale=material_scale,
                octaves=3,
                lacunarity=2.05,
                gain=0.55,
                seed_offset=2_003 + material_index * 1_009,
            ),
            seed,
        )
        # Gumbel-max 把配方概率转换为可比较分数。输入场保持连续，因此
        # argmax 产生的是自然曲线边界，而不是世界坐标方格。
        uniform = np.clip((field + 1.0) * 0.5, 1.0e-6, 1.0 - 1.0e-6)
        scores[material_index] = np.log(probability) - np.log(-np.log(uniform))
    return np.argmax(scores, axis=0).astype(np.uint8)


def _validate_field_shapes(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cold_weight: np.ndarray | None,
    water_level: np.ndarray | None,
    existing_surface_material_id: np.ndarray | None,
) -> None:
    if terrain.ndim != 2 or x.shape != terrain.shape or z.shape != terrain.shape:
        raise ValueError("permafrost fields must share a two-dimensional shape")
    for name, value in (
        ("cold weight", cold_weight),
        ("water level", water_level),
        ("surface material ids", existing_surface_material_id),
    ):
        if value is not None and value.shape != terrain.shape:
            raise ValueError(f"permafrost {name} must have the same shape as terrain")


def permafrost_surface_materials(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    systems: tuple[GlacialSystem, ...],
    seed: int,
    sea_level: float,
    *,
    cold_weight: np.ndarray | None = None,
    water_level: np.ndarray | None = None,
    existing_surface_material_id: np.ndarray | None = None,
    existing_surface_material_palette: tuple[str, ...] = (),
) -> np.ndarray:
    """生成冻土材质 ID，不修改地形高度。

    每个冰川系统只在自己的 seeded 几何域内生成冻土。寒带权重控制覆盖
    强度，低频噪声决定冻土斑块，另一层噪声按配方概率选择真实方块。
    已有的冰川深部冰/蓝冰会被保护，避免冻土层把主冰川核心抹掉。
    """

    _validate_field_shapes(
        terrain,
        x,
        z,
        cold_weight,
        water_level,
        existing_surface_material_id,
    )
    output = np.zeros(terrain.shape, dtype=np.uint8)
    if not systems:
        return output
    cold = (
        np.ones(terrain.shape, dtype=np.float64)
        if cold_weight is None
        else np.clip(cold_weight.astype(np.float64, copy=False), 0.0, 1.0)
    )
    wet = (
        np.zeros(terrain.shape, dtype=bool)
        if water_level is None
        else water_level >= 0.0
    )

    protected_ids: set[int] = set()
    for material in ("minecraft:packed_ice", "minecraft:blue_ice"):
        if material in existing_surface_material_palette:
            protected_ids.add(existing_surface_material_palette.index(material) + 1)

    for index, system in enumerate(systems):
        if not system.permafrost_enabled:
            continue
        system_seed = seed + index * 1_301_071 + 17_003
        distance = np.hypot(x - system.seed_center.x, z - system.seed_center.z)
        domain = np.clip(1.0 - distance / max(system.seed_extent, 1.0e-6), 0.0, 1.0)
        gate = np.clip(
            (cold - system.permafrost_min_cold_weight)
            / max(
                system.permafrost_full_cold_weight - system.permafrost_min_cold_weight,
                1.0e-6,
            ),
            0.0,
            1.0,
        )
        elevation = np.clip(
            (terrain - sea_level - system.permafrost_elevation_start)
            / max(system.permafrost_elevation_range, 1.0e-6),
            0.0,
            1.0,
        )

        coverage_noise = (
            sample_noise(
                x,
                z,
                NoiseLayer(
                    kind="fbm",
                    scale=system.permafrost_patch_scale,
                    octaves=2,
                    gain=0.55,
                    seed_offset=701,
                ),
                system_seed,
            )
            + 1.0
        ) * 0.5
        # 冻土斑块随寒带权重渐入；低地只保留少量过渡斑块。
        coverage_probability = np.clip(
            system.permafrost_coverage * (0.32 + 0.68 * gate) * (0.55 + 0.45 * elevation),
            0.0,
            1.0,
        )
        active = (
            (domain > 0.0)
            & (gate > 0.0)
            & (terrain >= sea_level)
            & ~wet
            & (coverage_noise <= coverage_probability)
        )

        if protected_ids and existing_surface_material_id is not None:
            protected = np.isin(existing_surface_material_id, tuple(protected_ids))
            active &= ~protected
        if not np.any(active):
            continue

        choices = _permafrost_material_choices(x, z, system, system_seed)
        output[active] = (choices[active] + 1).astype(np.uint8)
    return output


__all__ = ["PERMAFROST_PALETTE", "permafrost_surface_materials"]
