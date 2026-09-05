"""在连续高度场上运行确定性的 droplet hydraulic erosion。

该阶段只负责小尺度沟壑、支流和冲积沉积，不承担山脉或主河谷布局。
滴水更新顺序参考 Sebastien Lague 的公开算法说明，噪声/seed 组织方式
参考 FastNoiseLite；本文件是独立实现，没有复制外部代码：

* https://github.com/SebLague/Hydraulic-Erosion
* https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..randomness import unit_interval
from ..terrain_config import HydraulicErosionSettings


@dataclass(frozen=True)
class HydraulicErosionResult:
    """一次滴水 refinement 的高度、侵蚀量和沉积量。"""

    height: np.ndarray
    erosion: np.ndarray
    deposition: np.ndarray


def _validate_inputs(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: HydraulicErosionSettings,
) -> tuple[float, float]:
    if terrain.ndim != 2 or x.shape != terrain.shape or z.shape != terrain.shape:
        raise ValueError("hydraulic erosion fields must share a two-dimensional shape")
    if min(terrain.shape) < 3:
        raise ValueError("hydraulic erosion needs at least three rows and columns")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("hydraulic erosion inputs must contain finite values")
    cell_x = abs(float(x[0, 1] - x[0, 0]))
    cell_z = abs(float(z[1, 0] - z[0, 0]))
    if cell_x <= 0.0 or cell_z <= 0.0:
        raise ValueError("hydraulic erosion grid spacing must be positive")
    return cell_x, cell_z


def _sample(field: np.ndarray, px: float, pz: float) -> float:
    """双线性采样；水滴始终在内部一格活动，边缘只作安全钳制。"""

    height, width = field.shape
    gx = min(max(px, 0.0), width - 1.0)
    gz = min(max(pz, 0.0), height - 1.0)
    x0 = min(int(math.floor(gx)), width - 2)
    z0 = min(int(math.floor(gz)), height - 2)
    tx = gx - x0
    tz = gz - z0
    top = float(field[z0, x0]) * (1.0 - tx) + float(field[z0, x0 + 1]) * tx
    bottom = float(field[z0 + 1, x0]) * (1.0 - tx) + float(field[z0 + 1, x0 + 1]) * tx
    return top * (1.0 - tz) + bottom * tz


def _gradient(field: np.ndarray, px: float, pz: float, cell_x: float, cell_z: float) -> tuple[float, float]:
    """在水滴当前位置用中心差分求世界单位坡度。"""

    return (
        (_sample(field, px + 1.0, pz) - _sample(field, px - 1.0, pz)) / (2.0 * cell_x),
        (_sample(field, px, pz + 1.0) - _sample(field, px, pz - 1.0)) / (2.0 * cell_z),
    )


def _scatter(
    field: np.ndarray,
    amount_field: np.ndarray,
    px: float,
    pz: float,
    amount: float,
    *,
    field_sign: float = 1.0,
) -> None:
    """把一格侵蚀/沉积按双线性权重分摊到四个邻格。"""

    if amount == 0.0:
        return
    height, width = field.shape
    gx = min(max(px, 0.0), width - 1.001)
    gz = min(max(pz, 0.0), height - 1.001)
    x0 = min(int(math.floor(gx)), width - 2)
    z0 = min(int(math.floor(gz)), height - 2)
    tx = gx - x0
    tz = gz - z0
    weights = (
        (z0, x0, (1.0 - tx) * (1.0 - tz)),
        (z0, x0 + 1, tx * (1.0 - tz)),
        (z0 + 1, x0, (1.0 - tx) * tz),
        (z0 + 1, x0 + 1, tx * tz),
    )
    for row, column, weight in weights:
        delta = amount * weight
        field[row, column] += delta * field_sign
        amount_field[row, column] += delta


def apply_hydraulic_erosion(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: HydraulicErosionSettings,
    seed: int,
) -> HydraulicErosionResult:
    """对已有高度场执行滴水侵蚀，并返回可查询的差分场。

    起点、方向和所有状态都由 ``seed`` 决定，不使用进程级随机状态。水滴
    在边界内停止，避免把当前 tile 外的未知高度误当作坡面；宏观地貌仍由
    上游山脉/山谷阶段负责。
    """

    cell_x, cell_z = _validate_inputs(terrain, x, z, settings)
    original = np.asarray(terrain, dtype=np.float64)
    working = original.copy()
    erosion = np.zeros_like(working)
    deposition = np.zeros_like(working)
    if not settings.enabled or settings.droplet_count == 0:
        return HydraulicErosionResult(
            height=np.ascontiguousarray(working),
            erosion=np.ascontiguousarray(erosion),
            deposition=np.ascontiguousarray(deposition),
        )

    height, width = working.shape
    minimum_height = float(np.min(original))
    for droplet_index in range(settings.droplet_count):
        # 留出一格边界，让中心差分和双线性分摊永远有邻格可用。
        px = 1.0 + unit_interval(seed + settings.seed_offset, droplet_index * 2) * (width - 3.0)
        pz = 1.0 + unit_interval(seed + settings.seed_offset, droplet_index * 2 + 1) * (height - 3.0)
        direction_x = 0.0
        direction_z = 0.0
        speed = settings.initial_speed
        water = settings.initial_water
        sediment = 0.0

        for _ in range(settings.max_steps):
            gradient_x, gradient_z = _gradient(working, px, pz, cell_x, cell_z)
            direction_x = direction_x * settings.inertia - gradient_x * (1.0 - settings.inertia)
            direction_z = direction_z * settings.inertia - gradient_z * (1.0 - settings.inertia)
            direction_length = math.hypot(direction_x, direction_z)
            if direction_length <= 1.0e-12:
                break
            direction_x /= direction_length
            direction_z /= direction_length
            next_x = px + direction_x
            next_z = pz + direction_z
            if not (0.5 <= next_x < width - 1.5 and 0.5 <= next_z < height - 1.5):
                break

            current_height = _sample(working, px, pz)
            next_height = _sample(working, next_x, next_z)
            height_delta = next_height - current_height
            slope = max(-height_delta / max(math.hypot(cell_x, cell_z), 1.0e-12), settings.minimum_slope)
            capacity = max(
                slope * speed * water * settings.sediment_capacity,
                settings.minimum_slope,
            )

            if height_delta > 0.0:
                # 水滴撞上上坡时先沉积，避免它凭空穿过山脊。
                deposit = min(sediment, height_delta * settings.deposition_rate + sediment * settings.deposition_rate)
                sediment -= deposit
                _scatter(working, deposition, px, pz, deposit)
            elif sediment > capacity:
                deposit = (sediment - capacity) * settings.deposition_rate
                sediment -= deposit
                _scatter(working, deposition, px, pz, deposit)
            else:
                erode = min(
                    (capacity - sediment) * settings.erosion_rate,
                    settings.max_erosion_per_step,
                )
                # 不允许 refinement 把最低原始地势再挖穿；沉积仍可在此处抬高。
                erode = max(0.0, min(erode, current_height - minimum_height))
                sediment += erode
                _scatter(working, erosion, px, pz, erode, field_sign=-1.0)

            speed = math.sqrt(max(0.0, speed * speed - height_delta * settings.gravity))
            water *= 1.0 - settings.evaporation
            if water <= 1.0e-5:
                break
            px = next_x
            pz = next_z

    return HydraulicErosionResult(
        height=np.ascontiguousarray(working),
        erosion=np.ascontiguousarray(erosion),
        deposition=np.ascontiguousarray(deposition),
    )


__all__ = ["HydraulicErosionResult", "apply_hydraulic_erosion"]
