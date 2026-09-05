"""由积雪与地形共同决定冰斗和冰川流路。

本模块把冰川拓扑视为一个独立的规划阶段：先在完整冰川域上寻找正质量
平衡区的积雪极大值，再沿地形下坡方向追踪流线。路径惯性负责保持冰流
连续，seed 扰动只在允许的转向范围内增加弯曲；路径不会在边界反弹，
也不会无故穿越更高的山脊。较晚生成的支流可以接入已经存在的低处
主干。

积累区、冰面流动和流线汇合的结构参考：
- https://github.com/oargudo/glaciers

本文件只参考公开算法思路，代码为本项目独立实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..geometry import sample_regular_grid
from ..randomness import stable_text_seed, unit_interval
from ..terrain_config import GlacialSystem, Point


@dataclass(frozen=True)
class CirqueSeed:
    """从真实积累极大值选出的冰斗源头。"""

    center: Point
    outlet_angle: float
    radius_scale: float
    accumulation: float
    mass_balance: float
    elevation: float


@dataclass(frozen=True)
class GlacialFlowPlan:
    """一个冰川系统经过物理场规划后的不可变拓扑。"""

    system: GlacialSystem
    system_seed: int
    cirques: tuple[CirqueSeed, ...]
    paths: tuple[tuple[Point, ...], ...]


def _validate_fields(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    snow_accumulation: np.ndarray,
    mass_balance: np.ndarray,
) -> None:
    fields = (terrain, x, z, snow_accumulation, mass_balance)
    if terrain.ndim != 2 or 0 in terrain.shape:
        raise ValueError("glacial flow fields must be non-empty and two-dimensional")
    if any(field.shape != terrain.shape for field in fields[1:]):
        raise ValueError("glacial flow fields must share a shape")
    if terrain.shape[0] < 2 or terrain.shape[1] < 2:
        raise ValueError("glacial flow planning grid needs at least two rows and columns")
    if any(not np.isfinite(field).all() for field in fields):
        raise ValueError("glacial flow fields must be finite")

    axis_x = np.asarray(x[0], dtype=np.float64)
    axis_z = np.asarray(z[:, 0], dtype=np.float64)
    if (
        np.any(np.diff(axis_x) <= 0.0)
        or np.any(np.diff(axis_z) <= 0.0)
        or not np.allclose(x, axis_x[None, :])
        or not np.allclose(z, axis_z[:, None])
    ):
        raise ValueError("glacial flow coordinates must form an increasing regular grid")
    if not np.allclose(np.diff(axis_x), axis_x[1] - axis_x[0]) or not np.allclose(
        np.diff(axis_z), axis_z[1] - axis_z[0]
    ):
        raise ValueError("glacial flow coordinates must form an increasing regular grid")


def _sample(values: np.ndarray, x: np.ndarray, z: np.ndarray, point: Point) -> float:
    return float(sample_regular_grid(values, x, z, point.x, point.z))


def _normalize(dx: float, dz: float) -> tuple[float, float] | None:
    length = math.hypot(dx, dz)
    if length <= 1.0e-12:
        return None
    return dx / length, dz / length


def _downhill_direction(
    gradient_x: np.ndarray,
    gradient_z: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    point: Point,
) -> tuple[float, float] | None:
    return _normalize(
        -_sample(gradient_x, x, z, point),
        -_sample(gradient_z, x, z, point),
    )


def _source_candidates(
    system: GlacialSystem,
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    snow_accumulation: np.ndarray,
    mass_balance: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """按积雪和正质量平衡联合得分选择彼此分离的源头。"""

    distance = np.hypot(x - system.seed_center.x, z - system.seed_center.z)
    positive_balance = np.maximum(mass_balance, 0.0)
    score = np.maximum(snow_accumulation, 0.0) * positive_balance
    padded = np.pad(score, 1, mode="constant", constant_values=-np.inf)
    local_maximum = np.ones(score.shape, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbor = padded[
                row_offset : row_offset + score.shape[0],
                column_offset : column_offset + score.shape[1],
            ]
            local_maximum &= score >= neighbor
    source_limit = system.seed_extent - system.flow_min_length
    valid = (distance < source_limit) & (score > 0.0) & local_maximum
    if not np.any(valid) or system.cirque_count == 0:
        return ()

    flat_indices = np.flatnonzero(valid)
    order = flat_indices[np.argsort(-score.ravel()[flat_indices], kind="stable")]
    selected: list[tuple[int, int]] = []
    for flat_index in order:
        row, column = np.unravel_index(int(flat_index), terrain.shape)
        candidate_x = float(x[row, column])
        candidate_z = float(z[row, column])
        if any(
            math.hypot(
                candidate_x - float(x[other_row, other_column]),
                candidate_z - float(z[other_row, other_column]),
            )
            < system.flow_source_separation
            for other_row, other_column in selected
        ):
            continue
        selected.append((row, column))
        if len(selected) >= system.cirque_count:
            break
    return tuple(selected)


def _candidate_directions(
    desired_x: float,
    desired_z: float,
    maximum_turn: float,
    sample_count: int,
) -> tuple[tuple[float, float], ...]:
    """按与期望方向的夹角从小到大返回候选方向。"""

    desired_angle = math.atan2(desired_z, desired_x)
    offsets = np.linspace(-maximum_turn, maximum_turn, sample_count)
    offsets = sorted((float(offset) for offset in offsets), key=lambda value: abs(value))
    return tuple(
        (math.cos(desired_angle + offset), math.sin(desired_angle + offset))
        for offset in offsets
    )


def _nearest_trunk(
    point: Point,
    current_elevation: float,
    paths: list[tuple[Point, ...]],
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    merge_distance: float,
    uphill_tolerance: float,
    sample_spacing: float,
) -> tuple[Point, ...] | None:
    if merge_distance <= 0.0:
        return None
    best_distance = math.inf
    best_suffix: tuple[Point, ...] | None = None
    for path in paths:
        for index, trunk_point in enumerate(path):
            distance = math.hypot(point.x - trunk_point.x, point.z - trunk_point.z)
            if distance >= best_distance or distance > merge_distance:
                continue
            trunk_elevation = _sample(terrain, x, z, trunk_point)
            if trunk_elevation > current_elevation + uphill_tolerance:
                continue
            sample_count = max(2, math.ceil(distance / sample_spacing) + 1)
            connection_x = np.linspace(point.x, trunk_point.x, sample_count)
            connection_z = np.linspace(point.z, trunk_point.z, sample_count)
            connection_profile = sample_regular_grid(
                terrain,
                x,
                z,
                connection_x,
                connection_z,
            )
            if np.any(np.diff(connection_profile) > uphill_tolerance):
                continue
            best_distance = distance
            best_suffix = path[index:]
    return best_suffix


def _trace_path(
    system: GlacialSystem,
    source: Point,
    source_direction: tuple[float, float],
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mass_balance: np.ndarray,
    gradient_x: np.ndarray,
    gradient_z: np.ndarray,
    existing_paths: list[tuple[Point, ...]],
    seed: int,
    path_index: int,
) -> tuple[Point, ...]:
    step_length = system.valley_length / system.valley_segments
    previous_x, previous_z = source_direction
    current = source
    current_elevation = _sample(terrain, x, z, source)
    points = [source]
    travelled = 0.0
    name_seed = stable_text_seed(system.name)

    for segment in range(system.valley_segments):
        downhill = _downhill_direction(gradient_x, gradient_z, x, z, current)
        if downhill is None:
            downhill = (previous_x, previous_z)
        blended = _normalize(
            previous_x * system.flow_inertia + downhill[0] * (1.0 - system.flow_inertia),
            previous_z * system.flow_inertia + downhill[1] * (1.0 - system.flow_inertia),
        )
        if blended is None:
            break

        random_key = name_seed + path_index * 97_003 + segment * 193
        meander = (unit_interval(seed, random_key) * 2.0 - 1.0) * (
            math.pi * system.valley_turn * system.flow_meander_strength
        )
        desired_angle = math.atan2(blended[1], blended[0]) + meander
        desired_x = math.cos(desired_angle)
        desired_z = math.sin(desired_angle)
        maximum_turn = math.pi * system.valley_turn

        next_point: Point | None = None
        next_elevation = math.inf
        for direction_x, direction_z in _candidate_directions(
            desired_x,
            desired_z,
            maximum_turn,
            system.flow_direction_samples,
        ):
            candidate = Point(
                current.x + direction_x * step_length,
                current.z + direction_z * step_length,
            )
            if (
                math.hypot(
                    candidate.x - system.seed_center.x,
                    candidate.z - system.seed_center.z,
                )
                >= system.seed_extent
            ):
                continue
            candidate_elevation = _sample(terrain, x, z, candidate)
            if candidate_elevation > current_elevation + system.flow_uphill_tolerance:
                continue
            if any(
                math.hypot(candidate.x - previous.x, candidate.z - previous.z)
                < step_length * 0.45
                for previous in points[:-2]
            ):
                continue
            next_point = candidate
            next_elevation = candidate_elevation
            break

        if next_point is None:
            break
        points.append(next_point)
        travelled += step_length
        previous_direction = _normalize(
            next_point.x - current.x,
            next_point.z - current.z,
        )
        if previous_direction is not None:
            previous_x, previous_z = previous_direction
        current = next_point
        current_elevation = next_elevation

        trunk_suffix = _nearest_trunk(
            current,
            current_elevation,
            existing_paths,
            terrain,
            x,
            z,
            system.flow_merge_distance,
            system.flow_uphill_tolerance,
            min(float(x[0, 1] - x[0, 0]), float(z[1, 0] - z[0, 0])),
        )
        if trunk_suffix is not None:
            if current != trunk_suffix[0]:
                points.append(trunk_suffix[0])
            points.extend(trunk_suffix[1:])
            break
        if (
            travelled >= system.flow_min_length
            and _sample(mass_balance, x, z, current) <= system.flow_termination_balance
        ):
            break

    return tuple(points)


def plan_glacial_flow(
    system: GlacialSystem,
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    snow_accumulation: np.ndarray,
    mass_balance: np.ndarray,
    seed: int,
) -> GlacialFlowPlan:
    """从规划网格生成冰斗和地形导向的冰川流路。"""

    _validate_fields(terrain, x, z, snow_accumulation, mass_balance)
    if not system.mass_balance_enabled:
        return GlacialFlowPlan(system, seed, (), ())

    dx = float(x[0, 1] - x[0, 0])
    dz = float(z[1, 0] - z[0, 0])
    gradient_x = np.gradient(terrain, dx, axis=1)
    gradient_z = np.gradient(terrain, dz, axis=0)
    candidates = _source_candidates(
        system,
        terrain,
        x,
        z,
        snow_accumulation,
        mass_balance,
    )

    preliminary: list[CirqueSeed] = []
    source_directions: list[tuple[float, float]] = []
    name_seed = stable_text_seed(system.name)
    for index, (row, column) in enumerate(candidates):
        center = Point(float(x[row, column]), float(z[row, column]))
        downhill = _downhill_direction(gradient_x, gradient_z, x, z, center)
        if downhill is None:
            angle = unit_interval(seed, name_seed + index * 13_007 + 3) * math.tau
            downhill = (math.cos(angle), math.sin(angle))
        outlet_angle = math.atan2(downhill[1], downhill[0])
        preliminary.append(
            CirqueSeed(
                center=center,
                outlet_angle=outlet_angle,
                radius_scale=0.78
                + 0.46 * unit_interval(seed, name_seed + index * 13_007 + 4),
                accumulation=float(snow_accumulation[row, column]),
                mass_balance=float(mass_balance[row, column]),
                elevation=float(terrain[row, column]),
            )
        )
        source_directions.append(downhill)

    paths: list[tuple[Point, ...]] = []
    final_cirques = list(preliminary)
    for index, cirque in enumerate(preliminary[: system.valley_count]):
        path = _trace_path(
            system,
            cirque.center,
            source_directions[index],
            terrain,
            x,
            z,
            mass_balance,
            gradient_x,
            gradient_z,
            paths,
            seed,
            index,
        )
        if len(path) < 2:
            continue
        first, second = path[0], path[1]
        final_cirques[index] = CirqueSeed(
            center=cirque.center,
            outlet_angle=math.atan2(second.z - first.z, second.x - first.x),
            radius_scale=cirque.radius_scale,
            accumulation=cirque.accumulation,
            mass_balance=cirque.mass_balance,
            elevation=cirque.elevation,
        )
        paths.append(path)

    return GlacialFlowPlan(system, seed, tuple(final_cirques), tuple(paths))


__all__ = ["CirqueSeed", "GlacialFlowPlan", "plan_glacial_flow"]
