"""寒带冻融侵蚀和碎石坡识别。

这里使用轻量的热力侵蚀近似：当相邻方块的高度差超过 talus 临界坡度时，
把一部分岩屑从高处转移到低处。它不是完整冰川动力学模拟，但会稳定地产生
冰川谷壁、崩塌带和谷底碎石堆，并且适合当前的二维高度场管线。

算法结构参考：
- https://github.com/TheJanusStream/symbios-ground
- https://github.com/oargudo/glaciers

本文件只使用公开项目描述的算法思路，不复制外部源码。
"""

from __future__ import annotations

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..terrain_config import GlacialSystem, NoiseLayer
from ..noise import sample_noise
from .topology import GlacialFlowPlan


def glacial_exposure_mask(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plan: GlacialFlowPlan,
    sea_level: float,
    *,
    margin: float = 1.0,
) -> np.ndarray:
    """返回当前冰川系统允许发生冻融的地表范围。"""

    if margin <= 0:
        raise ValueError("glacial exposure margin must be positive")
    system = plan.system
    exposure = np.zeros(x.shape, dtype=bool)
    for cirque in plan.cirques:
        radius = system.cirque_radius * cirque.radius_scale * 1.35 * margin
        exposure |= np.hypot(x - cirque.center.x, z - cirque.center.z) <= radius
    for path in plan.paths:
        distance, _ = polyline_distance_and_progress(x, z, path)
        maximum_width = system.valley_source_width * (1.0 + system.valley_width_growth)
        exposure |= distance <= maximum_width * 1.55 * margin
    elevation_gate = terrain >= sea_level + system.cirque_min_elevation * 0.30
    return exposure & elevation_gate


def _slices(
    shape: tuple[int, int], dz: int, dx: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """返回 source/destination 切片，避免 np.roll 在边缘产生环绕地形。"""

    height, width = shape
    source_z_start = max(0, -dz)
    source_z_stop = min(height, height - dz)
    source_x_start = max(0, -dx)
    source_x_stop = min(width, width - dx)
    source = (slice(source_z_start, source_z_stop), slice(source_x_start, source_x_stop))
    destination = (
        slice(source_z_start + dz, source_z_stop + dz),
        slice(source_x_start + dx, source_x_stop + dx),
    )
    return source, destination


def apply_freeze_thaw(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    exposure: np.ndarray,
    system: GlacialSystem,
    seed: int,
) -> np.ndarray:
    """执行确定性的冻融/热力侵蚀并返回新的高度场。

    每轮只把超过临界坡度的部分向邻格转移，转移量受 seeded 噪声调制，
    因而会形成不规则的崩塌和碎石堆，而不是沿网格方向出现规则阶梯。
    """

    if exposure.shape != terrain.shape:
        raise ValueError("glacial exposure mask shape must match terrain")
    output = np.asarray(terrain, dtype=np.float64).copy()
    if system.freeze_thaw_iterations == 0 or system.freeze_thaw_strength == 0:
        return output
    noise = sample_noise(
        x,
        z,
        NoiseLayer(
            kind="fbm",
            scale=system.surface_patch_radius * 1.8,
            octaves=2,
            gain=0.55,
            seed_offset=701,
        ),
        seed,
    )
    local_strength = system.freeze_thaw_strength * (0.65 + 0.70 * ((noise + 1.0) * 0.5))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _ in range(system.freeze_thaw_iterations):
        for dz, dx in directions:
            source, destination = _slices(output.shape, dz, dx)
            source_height = output[source]
            destination_height = output[destination]
            difference = source_height - destination_height
            excess = np.maximum(difference - system.talus_slope_threshold, 0.0)
            transfer = np.minimum(excess * local_strength[source], difference * 0.35)
            transfer *= exposure[source]
            delta = np.zeros_like(output)
            delta[source] -= transfer
            delta[destination] += transfer
            output += delta
    return output


def talus_surface_mask(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    sea_level: float,
) -> np.ndarray:
    """识别冻融后应显示为碎石坡的低侧沉积带。"""

    result = np.zeros(terrain.shape, dtype=bool)
    gradient_z, gradient_x = np.gradient(terrain)
    slope = np.hypot(gradient_x, gradient_z)
    for plan in plans:
        system = plan.system
        exposure = glacial_exposure_mask(
            terrain,
            x,
            z,
            plan,
            sea_level,
            margin=1.12,
        )
        source_steep = exposure & (slope >= system.talus_slope_threshold * 0.85)
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            source, destination = _slices(terrain.shape, dz, dx)
            downhill = terrain[source] - terrain[destination]
            destination_mask = downhill >= system.talus_slope_threshold * 0.45
            marked = source_steep[source] & destination_mask
            result[destination] |= marked
    return result


__all__ = ["apply_freeze_thaw", "glacial_exposure_mask", "talus_surface_mask"]
