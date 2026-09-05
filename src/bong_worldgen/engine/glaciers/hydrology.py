"""冰川融水河的确定性水文阶段。

这里采用一维冰川流线近似，而不是对每个 raster tile 单独运行 MFD。冰川路径
是由本项目的 seeded 拓扑生成，沿路径把冰体厚度和融水率转换为累计流量，再把
流量投影为连续水面。这样跨 tile 生成时拓扑不会因为边界改变。

算法结构参考：

* https://github.com/alexanderpino/skills/blob/main/terrain-architect/reference-impl/flow.py
  的填洼、汇流和排水面积概念；
* https://github.com/alexanderpino/skills/blob/main/terrain-architect/reference-impl/shallow_water.py
  的水量作为状态、而不是把颜色直接涂在地表；
* https://github.com/oargudo/glaciers 的冰厚度/冰面流动与融化分层思路。

本模块不复制外部代码，也不把普通河流、地下河或冰裂隙当成同一种水体。
完整的全局 MFD 可在未来拥有世界级 hydrology window 后替换此流线近似。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import polyline_distance_and_progress, polyline_stations, sample_regular_grid
from ..terrain_config import GlacialSystem, NoiseLayer
from ..noise import sample_noise
from .topology import GlacialFlowPlan


GLACIAL_WATER_PALETTE = ("glacial_meltwater",)


@dataclass(frozen=True)
class GlacialWaterField:
    """冰川融水的连续水面、来源 ID 和流量场。"""

    water_level: np.ndarray
    water_id: np.ndarray
    discharge: np.ndarray


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    """对路径剖面做短窗口平滑，不产生超出输入范围的尖峰。"""

    radius = min(4, max(1, profile.size // 32))
    kernel = np.full(2 * radius + 1, 1.0 / (2 * radius + 1), dtype=np.float64)
    padded = np.pad(profile, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _path_total_length(path) -> float:
    return max(
        sum(
            float(np.hypot(end.x - start.x, end.z - start.z))
            for start, end in zip(path, path[1:])
        ),
        1.0e-9,
    )


def _path_melt_profile(
    terrain: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    path,
    system: GlacialSystem,
    cold_weight: np.ndarray | None,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算一条冰川路径的进度、连续水面和累计流量。"""

    total_length = _path_total_length(path)
    station_count = max(32, min(256, int(total_length / max(cell_size * 4.0, 1.0)) + 1))
    station_t, station_x, station_z = polyline_stations(path, station_count)
    terrain_profile = _smooth_profile(
        sample_regular_grid(terrain, grid_x, grid_z, station_x, station_z)
    )
    width = system.valley_source_width * (1.0 + system.valley_width_growth * station_t)

    if cold_weight is None:
        local_cold = np.ones_like(station_t)
    else:
        local_cold = np.clip(
            sample_regular_grid(cold_weight, grid_x, grid_z, station_x, station_z),
            0.0,
            1.0,
        )

    # 冰体在源头较厚，向末端逐渐减薄；融水率由配置注入，避免引擎藏常量。
    thickness = system.meltwater_ice_thickness * (0.35 + 0.65 * (1.0 - station_t))
    source = width * thickness * system.meltwater_rate * local_cold
    station_spacing = total_length / max(station_count - 1, 1)
    discharge = np.cumsum(source * station_spacing)
    max_discharge = max(float(discharge[-1]), 1.0e-9)
    normalized_discharge = np.clip(discharge / max_discharge, 0.0, 1.0)
    depth = system.meltwater_min_depth + (
        system.meltwater_max_depth - system.meltwater_min_depth
    ) * np.power(normalized_discharge, system.meltwater_depth_exponent)

    # 水面沿下游不能抬升；遇到局部床面抬升时再抬到床面上方的最小水深，
    # 保证不产生悬空水。最终方块量化留给 Anvil 适配器。
    surface = _smooth_profile(terrain_profile + depth)
    surface = np.minimum.accumulate(surface)
    surface = np.maximum(surface, terrain_profile + system.meltwater_min_depth)
    return station_t, surface, discharge


def apply_glacial_meltwater(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    sea_level: float,
    cold_weight: np.ndarray | None = None,
    cell_size: float = 1.0,
) -> GlacialWaterField:
    """把冰川融水投影到地表水面，并保留独立的冰川水体 ID。

    现有水体取最大水面，冰川来源不会降低湖泊或普通河流水位。仅在冰川
    路径、寒带权重和地表高于海平面的列写入冰川水体。
    """

    if terrain.shape != water.shape or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("glacial hydrology fields must share a shape")
    if cold_weight is not None and cold_weight.shape != terrain.shape:
        raise ValueError("glacial cold weight must have the same shape as terrain")
    output_water = np.asarray(water, dtype=np.float64).copy()
    output_id = np.zeros(terrain.shape, dtype=np.uint8)
    output_discharge = np.zeros(terrain.shape, dtype=np.float64)
    if not plans:
        return GlacialWaterField(output_water, output_id, output_discharge)

    for plan in plans:
        system = plan.system
        if not system.meltwater_enabled:
            continue
        system_seed = plan.system_seed
        for path_index, path in enumerate(plan.paths):
            station_t, surface_profile, discharge_profile = _path_melt_profile(
                terrain,
                x,
                z,
                path,
                system,
                cold_weight,
                cell_size,
            )
            distance, progress = polyline_distance_and_progress(x, z, path)
            width = system.valley_source_width * (
                1.0 + system.valley_width_growth * progress
            )
            channel = distance <= width
            if cold_weight is None:
                active_climate = np.ones(terrain.shape, dtype=bool)
            else:
                active_climate = cold_weight >= system.meltwater_min_cold_weight
            surface = np.interp(progress, station_t, surface_profile)
            discharge = np.interp(progress, station_t, discharge_profile)
            # 低频噪声只负责让融水在宽谷中形成连续主槽，绝不改变水面高度。
            flow_noise = sample_noise(
                x,
                z,
                NoiseLayer(
                    kind="fbm",
                    scale=max(float(np.max(width)) * 3.0, 24.0),
                    octaves=2,
                    gain=0.55,
                    seed_offset=path_index * 97 + 17,
                ),
                system_seed,
            )
            corridor = np.clip(1.0 - distance / np.maximum(width, 1.0e-6), 0.0, 1.0)
            corridor *= np.clip(0.78 + 0.22 * flow_noise, 0.55, 1.0)
            valid = (
                channel
                & active_climate
                & (terrain >= sea_level)
                & (surface > terrain + system.meltwater_min_depth * 0.5)
            )
            output_water = np.where(valid, np.maximum(output_water, surface), output_water)
            output_id[valid] = 1
            output_discharge = np.where(
                valid,
                np.maximum(output_discharge, discharge * corridor),
                output_discharge,
            )
    return GlacialWaterField(
        water_level=np.ascontiguousarray(output_water, dtype=np.float64),
        water_id=np.ascontiguousarray(output_id, dtype=np.uint8),
        discharge=np.ascontiguousarray(output_discharge, dtype=np.float64),
    )


__all__ = ["GLACIAL_WATER_PALETTE", "GlacialWaterField", "apply_glacial_meltwater"]
