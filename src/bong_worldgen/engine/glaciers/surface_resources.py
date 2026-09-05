"""寒带群山地表生成物。

这里的输出仍复用 ``UndergroundBlock`` 稀疏资源记录格式，因为 Server 已经
用它索引所有需要后续刷新真实产物的坐标。它并不表示方块在地下：枯木的
``y`` 是最终雪/冰覆盖层顶部上方一格，Anvil/BlueMap 会写入真实方块，Server
则通过 ``source=snow_mountain`` 区分它。

采样使用世界坐标和无状态 seed 哈希，保证整图导出、分块导出和运行时按块
生成得到完全一致的刷新点。噪声/seed 接口的组织方式参考 FastNoiseLite，
本模块没有复制其实现：

https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..randomness import mix64, unit_interval
from ..terrain_config import GlacialSystem
from ..underground_config import UndergroundBlock


_SYSTEM_SALT = 0x9E3779B97F4A7C15
_SPEC_SALT = 0xD6E8FEB86659FD93
_X_SALT = 0xBF58476D1CE4E5B9
_Z_SALT = 0x94D049BB133111EB


def _validate_field(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"snow mountain resource {name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"snow mountain resource {name} must contain finite values")
    return array


def _world_roll(
    seed: int,
    world_x: int,
    world_z: int,
    system_index: int,
    spec_index: int,
) -> float:
    """为一个世界列计算不依赖 tile 边界的稳定概率值。"""

    candidate_seed = mix64(
        seed
        + system_index * _SYSTEM_SALT
        + spec_index * _SPEC_SALT
        + world_x * _X_SALT
        + world_z * _Z_SALT
    )
    return unit_interval(candidate_seed, 0)


def _top_surface_y(
    terrain: np.ndarray,
    surface_cover_layers: np.ndarray,
) -> np.ndarray:
    """返回最终覆盖层顶部的方块坐标，不在地表和覆盖层之间制造空气。"""

    if surface_cover_layers.shape != (4, *terrain.shape):
        raise ValueError(
            "snow mountain resource surface_cover_layers must have shape "
            "(4, height, width)"
        )
    return np.rint(terrain).astype(np.int32) + np.sum(
        surface_cover_layers,
        axis=0,
        dtype=np.int32,
    )


def generate_snow_mountain_resource_blocks(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    systems: tuple[GlacialSystem, ...],
    seed: int,
    *,
    mountain_weight: np.ndarray,
    mountain_material_id: np.ndarray,
    surface_material_id: np.ndarray,
    surface_cover_layers: np.ndarray,
    water_level: np.ndarray,
    glacial_crevasse_id: np.ndarray,
    occupied_blocks: Iterable[UndergroundBlock] = (),
    climate_cold_weight: np.ndarray | None = None,
) -> tuple[UndergroundBlock, ...]:
    """按配置在寒带群山的最终表面生成稀疏雪山资源点。

    资源候选必须同时满足：群山权重、寒带权重、雪/冰材质或覆盖层、非水面、
    非冰川裂隙。每个世界列最多放置一个配置项，概率由对应配置控制；已有
    地下资源坐标不会被覆盖。返回值按世界坐标排序，便于 raster 输出稳定化。
    """

    terrain = np.asarray(terrain, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if terrain.ndim != 2 or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("snow mountain resource fields must share a two-dimensional shape")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("snow mountain resource fields must contain finite values")
    shape = terrain.shape
    mountain_weight = _validate_field("mountain_weight", mountain_weight, shape).astype(
        np.float64, copy=False
    )
    mountain_material_id = _validate_field("mountain_material_id", mountain_material_id, shape)
    surface_material_id = _validate_field("surface_material_id", surface_material_id, shape)
    water_level = _validate_field("water_level", water_level, shape).astype(np.float64, copy=False)
    glacial_crevasse_id = _validate_field("glacial_crevasse_id", glacial_crevasse_id, shape)
    if climate_cold_weight is None:
        cold_weight = np.ones(shape, dtype=np.float64)
    else:
        cold_weight = _validate_field("climate_cold_weight", climate_cold_weight, shape).astype(
            np.float64, copy=False
        )
    cover_layers = np.asarray(surface_cover_layers)
    top_y = _top_surface_y(terrain, cover_layers)
    world_x = np.rint(x).astype(np.int64)
    world_z = np.rint(z).astype(np.int64)
    snow_material = np.isin(mountain_material_id, (1, 2, 3, 4, 5)) | np.isin(
        surface_material_id, (1, 2, 3, 4, 5)
    )
    snow_material |= np.sum(cover_layers, axis=0) > 0
    base_eligible = snow_material & (water_level < 0.0) & (glacial_crevasse_id == 0)
    occupied = {(block.x, block.y, block.z) for block in occupied_blocks}
    result: dict[tuple[int, int, int], UndergroundBlock] = {}
    for system_index, system in enumerate(systems):
        for spec_index, spec in enumerate(system.snow_mountain_resources):
            eligible = base_eligible.copy()
            eligible &= mountain_weight >= spec.min_mountain_weight
            eligible &= cold_weight >= spec.min_cold_weight
            if not np.any(eligible) or spec.probability <= 0.0:
                continue
            for row, column in zip(*np.nonzero(eligible)):
                wx = int(world_x[row, column])
                wz = int(world_z[row, column])
                wy = int(top_y[row, column] + 1)
                position = (wx, wy, wz)
                if position in occupied or _world_roll(
                    seed + spec.seed_offset,
                    wx,
                    wz,
                    system_index,
                    spec_index,
                ) >= spec.probability:
                    continue
                result[position] = UndergroundBlock(
                    x=wx,
                    y=wy,
                    z=wz,
                    material=f"minecraft:{spec.material}",
                    resource_id=spec.resource_id,
                    source="snow_mountain",
                    rarity=spec.rarity,
                )
                occupied.add(position)
    return tuple(result[position] for position in sorted(result))


__all__ = ["generate_snow_mountain_resource_blocks"]
