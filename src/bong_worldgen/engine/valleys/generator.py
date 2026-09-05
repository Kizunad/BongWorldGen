"""固定世界域上的山谷规划与分块稳定采样。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import sample_regular_grid
from ..noise import sample_noise
from ..randomness import stable_text_seed
from ..terrain_config import ValleySystem
from .hydrology import route_flow_d8
from .divide import watershed_divide_strength
from .stream_power import stream_power_erosion


@dataclass(frozen=True)
class ValleyErosionPlan:
    """一个水文域生成的不可变侵蚀数据。"""

    system: ValleySystem
    grid_x: np.ndarray
    grid_z: np.ndarray
    erosion_depth: np.ndarray
    flow_accumulation: np.ndarray
    stream_power: np.ndarray
    watershed_divide: np.ndarray
    uplift: np.ndarray | None = None


@dataclass(frozen=True)
class ValleyField:
    """采样到当前 tile 的谷深、汇水量和 Stream Power。"""

    depth: np.ndarray
    flow_accumulation: np.ndarray
    stream_power: np.ndarray
    watershed_divide: np.ndarray


def plan_valley_erosion(
    system: ValleySystem,
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    seed: int,
    sea_level: float,
    uplift: np.ndarray | None = None,
) -> ValleyErosionPlan:
    """在完整规划域内计算降雨、汇流与 Stream Power 下切。

    ``uplift`` 是山脉阶段产生的构造抬升场。它不替代 DEM：D8 仍从抬升后
    的真实地形路由水流，但 Stream Power 会额外读取这张场，表达持续抬升
    为河流提供更多侵蚀势能的关系。
    """

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("valley planning fields must share a shape")
    if uplift is None:
        uplift = np.zeros_like(terrain, dtype=np.float64)
    else:
        uplift = np.asarray(uplift, dtype=np.float64)
        if uplift.shape != terrain.shape:
            raise ValueError("valley planning uplift must share the terrain shape")
        if not np.isfinite(uplift).all() or np.any(uplift < 0.0):
            raise ValueError("valley planning uplift must be finite and non-negative")
    if min(terrain.shape) < 3:
        raise ValueError("valley planning grid needs at least three samples per axis")
    cell_size_x = float(x[0, 1] - x[0, 0])
    cell_size_z = float(z[1, 0] - z[0, 0])
    if cell_size_x <= 0 or cell_size_z <= 0:
        raise ValueError("valley planning coordinates must increase")
    axis_x = np.asarray(x[0], dtype=np.float64)
    axis_z = np.asarray(z[:, 0], dtype=np.float64)
    if (
        not np.allclose(x, axis_x[None, :])
        or not np.allclose(z, axis_z[:, None])
        or not np.allclose(np.diff(axis_x), cell_size_x)
        or not np.allclose(np.diff(axis_z), cell_size_z)
    ):
        raise ValueError("valley planning coordinates must form a regular grid")

    system_seed = seed + stable_text_seed(system.name)
    rainfall_noise = sample_noise(x, z, system.rainfall_noise, system_seed + 43_003)
    rainfall = np.clip(
        1.0 + rainfall_noise * system.rainfall_variation,
        1.0 - system.rainfall_variation,
        1.0 + system.rainfall_variation,
    )
    # 有限轮次的反馈：山脉先抬升形成坡面，河流切割后用新地形重新
    # 路由。累计深度受 maximum_depth 限制，避免迭代次数改变地貌尺度上限。
    working_terrain = np.asarray(terrain, dtype=np.float64).copy()
    cumulative_depth = np.zeros_like(working_terrain)
    flow = None
    stream_power = np.zeros_like(working_terrain)
    for _ in range(system.coupling_iterations):
        flow = route_flow_d8(
            working_terrain,
            rainfall,
            cell_size_x=cell_size_x,
            cell_size_z=cell_size_z,
            minimum_slope=system.minimum_slope,
        )
        erosion_depth, current_stream_power = stream_power_erosion(
            working_terrain,
            flow,
            system,
            sea_level=sea_level,
            cell_size_x=cell_size_x,
            cell_size_z=cell_size_z,
            uplift=uplift,
        )
        # 单轮函数给出当前水文状态下的目标下切深度；取最大值而不是
        # 重复相加，保证 ``maximum_depth`` 是整个耦合过程的上限。
        cumulative_depth = np.maximum(cumulative_depth, erosion_depth)
        working_terrain = np.asarray(terrain, dtype=np.float64) - cumulative_depth
        # 输出字段保留耦合过程中出现过的最大侵蚀势能，和累计谷深保持同一
        # 物理解释；下一轮仍使用上一次的累计切割结果重新路由。
        stream_power = np.maximum(stream_power, current_stream_power)
    if flow is None:
        raise RuntimeError("valley coupling did not produce a flow field")
    divide = watershed_divide_strength(
        flow,
        system,
        cell_size_x=cell_size_x,
        cell_size_z=cell_size_z,
        uplift=uplift,
        stream_power=stream_power,
        erosion_depth=cumulative_depth,
    )
    return ValleyErosionPlan(
        system=system,
        grid_x=np.ascontiguousarray(x),
        grid_z=np.ascontiguousarray(z),
        erosion_depth=np.ascontiguousarray(cumulative_depth),
        flow_accumulation=flow.accumulation,
        stream_power=stream_power,
        watershed_divide=divide,
        uplift=np.ascontiguousarray(uplift),
    )


def sample_valley_fields(
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[ValleyErosionPlan, ...],
) -> ValleyField:
    """按世界坐标采样并合并规划场；规划域外严格为零。"""

    if x.shape != z.shape:
        raise ValueError("valley output coordinates must share a shape")
    depth = np.zeros(x.shape, dtype=np.float64)
    accumulation = np.zeros(x.shape, dtype=np.float64)
    stream_power = np.zeros(x.shape, dtype=np.float64)
    divide = np.zeros(x.shape, dtype=np.float64)
    for plan in plans:
        system = plan.system
        inside = (
            (x >= system.domain_center.x - system.domain_extent_x)
            & (x <= system.domain_center.x + system.domain_extent_x)
            & (z >= system.domain_center.z - system.domain_extent_z)
            & (z <= system.domain_center.z + system.domain_extent_z)
        )
        sampled_depth = sample_regular_grid(plan.erosion_depth, plan.grid_x, plan.grid_z, x, z)
        sampled_accumulation = sample_regular_grid(
            plan.flow_accumulation, plan.grid_x, plan.grid_z, x, z
        )
        sampled_stream_power = sample_regular_grid(
            plan.stream_power, plan.grid_x, plan.grid_z, x, z
        )
        sampled_divide = sample_regular_grid(
            plan.watershed_divide, plan.grid_x, plan.grid_z, x, z
        )
        depth = np.where(inside, np.maximum(depth, sampled_depth), depth)
        accumulation = np.where(
            inside, np.maximum(accumulation, sampled_accumulation), accumulation
        )
        stream_power = np.where(
            inside, np.maximum(stream_power, sampled_stream_power), stream_power
        )
        divide = np.where(inside, np.maximum(divide, sampled_divide), divide)
    return ValleyField(depth, accumulation, stream_power, divide)


def apply_valley_erosion(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[ValleyErosionPlan, ...],
) -> np.ndarray:
    """把采样后的谷深从地形扣除。"""

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("valley output fields must share a shape")
    field = sample_valley_fields(x, z, plans)
    return np.asarray(terrain, dtype=np.float64) - field.depth


def apply_watershed_divide(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[ValleyErosionPlan, ...],
) -> np.ndarray:
    """把已规划的分水岭抬升叠加到 Stream Power 侵蚀后的地形。"""

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("watershed divide output fields must share a shape")
    field = sample_valley_fields(x, z, plans)
    return np.asarray(terrain, dtype=np.float64) + field.watershed_divide


__all__ = [
    "ValleyErosionPlan",
    "ValleyField",
    "apply_valley_erosion",
    "apply_watershed_divide",
    "plan_valley_erosion",
    "sample_valley_fields",
]
