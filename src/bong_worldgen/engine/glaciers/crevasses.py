"""冰川谷场、蓝冰 veins 与冰面横向裂隙。

冰裂隙是冰川表面的独立语义层，不复用地下 ``fracture_id``。路径切线的
垂线决定裂隙主方向，seeded 低频噪声只负责弯曲和打散间距。噪声采样接口
参考 FastNoiseLite 的 seeded 3D/domain sampling 用法（未复制其实现）：

* https://github.com/Auburn/FastNoiseLite
* https://github.com/oargudo/glaciers
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..noise import sample_noise
from ..randomness import unit_interval
from ..terrain_config import GlacialSystem, NoiseLayer, Point
from .topology import GlacialFlowPlan


# 0 表示没有裂隙；其余值是最终地表可以直接使用的真实方块。
GLACIAL_CREVASSE_PALETTE = (
    "minecraft:air",
    "minecraft:packed_ice",
    "minecraft:blue_ice",
)


@dataclass(frozen=True)
class GlacierValleyField:
    """由所有冰川流路合并出的谷地覆盖与冰压场。"""

    mask: np.ndarray
    pressure: np.ndarray


def _validate_grid(x: np.ndarray, z: np.ndarray) -> None:
    if x.ndim != 2 or x.shape != z.shape or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("glacier valley coordinates must be finite two-dimensional fields")


def sample_glacier_valley_field(
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
) -> GlacierValleyField:
    """把流路投影成谷地 mask 与中心高压场。

    ``mask`` 是冰川覆盖的空间门控；``pressure`` 越接近冰川中心越高，供蓝冰
    概率和裂隙强度使用。所有宽度都来自对应 ``GlacialSystem``，不会硬编码
    固定的四条通道或固定地图区域。
    """

    _validate_grid(x, z)
    mask = np.zeros(x.shape, dtype=bool)
    pressure = np.zeros(x.shape, dtype=np.float64)
    for plan in plans:
        system = plan.system
        for path in plan.paths:
            distance, progress = polyline_distance_and_progress(x, z, path)
            width = system.valley_source_width * (
                1.0 + system.valley_width_growth * progress
            )
            width = np.maximum(width, 1.0e-6)
            lateral = np.clip(1.0 - distance / width, 0.0, 1.0)
            # 中心越靠近 floor，冰层越厚；路径下游因流路变宽而保持连续。
            core_radius = np.maximum(system.valley_floor_width * 1.35, 1.0)
            core = np.clip(1.0 - distance / core_radius, 0.0, 1.0)
            candidate_pressure = lateral * (0.35 + 0.65 * core)
            mask |= lateral > 0.0
            pressure = np.maximum(pressure, candidate_pressure)
    return GlacierValleyField(
        mask=np.ascontiguousarray(mask),
        pressure=np.ascontiguousarray(np.clip(pressure, 0.0, 1.0)),
    )


def _path_point_and_tangent(
    path: tuple[Point, ...], progress: float
) -> tuple[Point, tuple[float, float]]:
    """按归一化进度取路径点和局部切线。"""

    lengths = np.asarray(
        [math.hypot(end.x - start.x, end.z - start.z) for start, end in zip(path, path[1:])],
        dtype=np.float64,
    )
    total = max(float(lengths.sum()), 1.0e-9)
    target = np.clip(progress, 0.0, 1.0) * total
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    segment = min(int(np.searchsorted(cumulative[1:], target, side="right")), len(path) - 2)
    length = max(float(lengths[segment]), 1.0e-9)
    local = np.clip((target - cumulative[segment]) / length, 0.0, 1.0)
    start = path[segment]
    end = path[segment + 1]
    tx = (end.x - start.x) / length
    tz = (end.z - start.z) / length
    return Point(
        start.x + (end.x - start.x) * local,
        start.z + (end.z - start.z) * local,
    ), (tx, tz)


def _path_length(path: tuple[Point, ...]) -> float:
    return max(
        sum(math.hypot(end.x - start.x, end.z - start.z) for start, end in zip(path, path[1:])),
        1.0e-9,
    )


def sample_glacial_crevasses(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    seed: int,
    sea_level: float,
    *,
    climate_cold_weight: np.ndarray | None = None,
    surface_material_id: np.ndarray | None = None,
) -> np.ndarray:
    """生成横向冰裂隙 ID。

    裂隙沿流路切线的垂线展开：中央窄带是空气，边缘以 packed ice 为主，
    少量按配置变成 blue ice。它只在冰川谷和实际冰类材质上生成；没有传入
    ``surface_material_id`` 时，调用者可用它作为独立的几何测试场。
    """

    terrain = np.asarray(terrain, dtype=np.float64)
    if terrain.ndim != 2 or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("glacial crevasse fields must share a two-dimensional shape")
    _validate_grid(x, z)
    if climate_cold_weight is not None and climate_cold_weight.shape != terrain.shape:
        raise ValueError("glacial crevasse cold weight must have the same shape as terrain")
    if surface_material_id is not None and surface_material_id.shape != terrain.shape:
        raise ValueError("glacial crevasse surface material must have the same shape as terrain")
    output = np.zeros(terrain.shape, dtype=np.uint8)
    if not plans:
        return output

    for plan_index, plan in enumerate(plans):
        system = plan.system
        if not system.crevasses_enabled:
            continue
        random_seed = seed + plan.system_seed
        cold = (
            np.ones(terrain.shape, dtype=np.float64)
            if climate_cold_weight is None
            else np.clip(np.asarray(climate_cold_weight, dtype=np.float64), 0.0, 1.0)
        )
        for path_index, path in enumerate(plan.paths):
            total = _path_length(path)
            crack_count = max(1, int(math.ceil(total / system.crevasse_spacing)) - 1)
            for crack_index in range(crack_count):
                base = (
                    random_seed
                    + plan_index * 97_003
                    + path_index * 7_919
                    + crack_index * 1_003
                )
                progress = (crack_index + 1) / (crack_count + 1)
                progress += (
                    unit_interval(random_seed, int(base + 11)) - 0.5
                ) * system.crevasse_jitter * system.crevasse_spacing / total
                center, tangent = _path_point_and_tangent(path, progress)
                tx, tz = tangent
                nx, nz = -tz, tx
                width = system.valley_source_width * (
                    1.0 + system.valley_width_growth * np.clip(progress, 0.0, 1.0)
                )
                half_span = width * (
                    0.68 + 0.28 * unit_interval(random_seed, int(base + 13))
                )
                thickness = max(system.crevasse_width, 0.25)
                line_noise = sample_noise(
                    x,
                    z,
                    NoiseLayer(
                        kind="fbm",
                        scale=max(width * 1.8, 16.0),
                        octaves=2,
                        gain=0.55,
                        seed_offset=int(path_index * 101 + crack_index * 17 + 41),
                    ),
                    random_seed,
                )
                dx = x - center.x
                dz = z - center.z
                along = dx * tx + dz * tz
                across = dx * nx + dz * nz
                curved_along = (
                    along - line_noise * half_span * system.crevasse_meander * 0.28
                )
                in_span = np.abs(across) <= half_span
                valley_distance, _ = polyline_distance_and_progress(x, z, path)
                path_width = system.valley_source_width * (
                    1.0 + system.valley_width_growth * np.clip(progress, 0.0, 1.0)
                )
                active = (
                    in_span
                    & (valley_distance <= path_width)
                    & (terrain >= sea_level)
                    & (cold >= system.crevasse_min_cold_weight)
                )
                if surface_material_id is not None:
                    # surface_material_id: 3=ice, 4=packed ice, 5=blue ice。
                    active &= np.isin(surface_material_id, (3, 4, 5))
                local_pressure = np.clip(
                    1.0 - valley_distance / np.maximum(path_width, 1.0e-6),
                    0.0,
                    1.0,
                )
                probability = np.clip(
                    system.crevasse_probability * (0.55 + 0.45 * local_pressure) * cold,
                    0.0,
                    1.0,
                )
                roll = (
                    sample_noise(
                        x,
                        z,
                        NoiseLayer(
                            kind="value",
                            scale=max(width * 0.7, 8.0),
                            octaves=1,
                            seed_offset=83,
                        ),
                        int(base + 101),
                    )
                    + 1.0
                ) * 0.5
                band = (
                    active
                    & (np.abs(curved_along) <= thickness * 1.8)
                    & (roll <= probability)
                )
                if not np.any(band):
                    continue
                packed = band & (np.abs(curved_along) > thickness * 0.48)
                blue_roll = (
                    sample_noise(
                        x,
                        z,
                        NoiseLayer(
                            kind="value",
                            scale=max(width * 0.45, 6.0),
                            octaves=1,
                            seed_offset=127,
                        ),
                        int(base + 131),
                    )
                    + 1.0
                ) * 0.5
                blue = packed & (blue_roll <= system.crevasse_blue_ice_probability * local_pressure)
                output[band] = 2
                output[blue] = 3
                output[band & ~packed] = 1
    return np.ascontiguousarray(output)


__all__ = [
    "GLACIAL_CREVASSE_PALETTE",
    "GlacierValleyField",
    "sample_glacier_valley_field",
    "sample_glacial_crevasses",
]
