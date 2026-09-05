"""确定性的洞穴拓扑图生成。

Godot Voxel 的文档重点讲的是 worm 体积场，而不是固定的手工折线图。
默认不使用配方路径：先从世界 seed 生成随机游走主洞，再派生支洞和洞室节点。
显式路径仍可作为兼容和测试用的覆盖入口：

* 不使用全局随机状态，任意区块都能独立重建同一张图；
* 支洞是图的边，洞室是图的节点；
* 最终几何仍交给 ``generator.py`` 的 SDF/smooth-union 阶段。

参考：
https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..terrain_config import Point
from ..underground_config import CaveNetwork
from ..randomness import unit_interval as _unit


def _lerp(start: Point, end: Point, amount: float) -> Point:
    return Point(
        start.x + (end.x - start.x) * amount,
        start.z + (end.z - start.z) * amount,
    )


def _smooth_path(points: list[Point]) -> tuple[Point, ...]:
    """把随机游走折点做一次邻域平滑，避免出现折尺一样的尖角。"""

    if len(points) < 3:
        return tuple(points)
    smoothed = [points[0]]
    for previous, current, following in zip(points, points[1:], points[2:]):
        smoothed.append(
            Point(
                previous.x * 0.2 + current.x * 0.6 + following.x * 0.2,
                previous.z * 0.2 + current.z * 0.6 + following.z * 0.2,
            )
        )
    smoothed.append(points[-1])
    return tuple(smoothed)


def _generate_seed_paths(network: CaveNetwork, seed: int) -> list[tuple[Point, ...]]:
    """根据 seed 生成有持续转向的随机游走洞道，不依赖固定锚点。"""

    paths: list[tuple[Point, ...]] = []
    extent = network.seed_extent
    for path_index in range(network.seed_path_count):
        base = path_index * 10_003
        start_radius = extent * (0.12 + 0.22 * _unit(seed, base + 1))
        start_angle = _unit(seed, base + 2) * math.tau
        start = Point(
            network.seed_center.x + math.cos(start_angle) * start_radius,
            network.seed_center.z + math.sin(start_angle) * start_radius,
        )
        direction = _unit(seed, base + 3) * math.tau
        points = [start]
        for segment in range(network.seed_path_segments):
            # 低频持续转向：每段只改变当前方向的一部分，形成弯转 worm，
            # 而不是每段重新抽取一条互不相关的直线。
            turn = (
                _unit(seed, base + 100 + segment * 2) * 2.0 - 1.0
            ) * math.pi * network.seed_path_turn * 0.45
            direction += turn
            length = network.seed_path_length / network.seed_path_segments * (
                0.72 + 0.56 * _unit(seed, base + 101 + segment * 2)
            )
            previous = points[-1]
            next_point = Point(
                previous.x + math.cos(direction) * length,
                previous.z + math.sin(direction) * length,
            )
            # 用反射而不是硬截断，保持边界处的方向连续。
            local_x = next_point.x - network.seed_center.x
            local_z = next_point.z - network.seed_center.z
            if abs(local_x) > extent:
                direction = math.pi - direction
                next_point = Point(
                    network.seed_center.x
                    + math.copysign(extent, local_x)
                    - (local_x - math.copysign(extent, local_x)) * 0.25,
                    next_point.z,
                )
            if abs(local_z) > extent:
                direction = -direction
                next_point = Point(
                    next_point.x,
                    network.seed_center.z
                    + math.copysign(extent, local_z)
                    - (local_z - math.copysign(extent, local_z)) * 0.25,
                )
            points.append(next_point)
        paths.append(_smooth_path(points))
    return paths


@dataclass(frozen=True)
class CaveTopology:
    """一张洞穴图：折线边和洞室节点。"""

    paths: tuple[tuple[Point, ...], ...]
    chambers: tuple[Point, ...]
    entrances: tuple[Point, ...]


def generate_cave_topology(network: CaveNetwork, seed: int) -> CaveTopology:
    """从 seed 确定性地产生随机主洞、支洞和洞室节点。"""

    base_paths = list(network.paths) or _generate_seed_paths(network, seed)
    paths = list(base_paths)
    anchors = [point for path in base_paths for point in path]
    branch_endpoints: list[Point] = []

    for branch_index in range(network.branch_count):
        source_path = base_paths[branch_index % len(base_paths)]
        segment_index = min(
            int(_unit(seed, branch_index * 7) * (len(source_path) - 1)),
            len(source_path) - 2,
        )
        start = source_path[segment_index]
        end = source_path[segment_index + 1]
        amount = 0.2 + 0.6 * _unit(seed, branch_index * 7 + 1)
        origin = _lerp(start, end, amount)
        angle = math.atan2(end.z - start.z, end.x - start.x)
        if branch_index % 2:
            angle += math.pi
        angle += (_unit(seed, branch_index * 7 + 2) * 2.0 - 1.0) * math.pi * network.branch_turn

        points = [origin]
        for segment in range(network.branch_segments):
            turn = (
                _unit(seed, branch_index * 101 + segment * 3 + 3) * 2.0 - 1.0
            ) * math.pi * network.branch_turn
            angle += turn
            length = network.branch_length / network.branch_segments * (
                0.7 + 0.6 * _unit(seed, branch_index * 101 + segment * 3 + 4)
            )
            previous = points[-1]
            points.append(
                Point(
                    previous.x + math.cos(angle) * length,
                    previous.z + math.sin(angle) * length,
                )
            )
        paths.append(tuple(points))
        branch_endpoints.append(points[-1])

    chamber_candidates = anchors + branch_endpoints
    chambers: list[Point] = []
    for index in range(network.chamber_count):
        if not chamber_candidates:
            break
        candidate = chamber_candidates[
            int(_unit(seed, 10_000 + index * 5) * len(chamber_candidates))
            % len(chamber_candidates)
        ]
        chambers.append(candidate)

    # 入口优先落在随机生成的基础洞道上，而不是支洞末端；这样入口始终
    # 属于网络的主影响范围，不会因为某条支洞越过 tile 边界而消失。
    entrance_candidates = anchors
    entrances: list[Point] = []
    for index in range(network.entrance_count):
        if not entrance_candidates:
            break
        candidate = entrance_candidates[
            int(_unit(seed, 20_000 + index * 5) * len(entrance_candidates))
            % len(entrance_candidates)
        ]
        entrances.append(candidate)

    return CaveTopology(tuple(paths), tuple(chambers), tuple(entrances))


__all__ = ["CaveTopology", "generate_cave_topology"]
