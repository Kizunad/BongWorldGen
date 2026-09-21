"""确定性的洞穴拓扑图生成。

Godot Voxel 的文档重点讲的是 worm 体积场，而不是固定的手工折线图。
这里保留配方路径作为可控锚点，再从世界 seed 派生支洞和洞室节点：

* 不使用全局随机状态，任意区块都能独立重建同一张图；
* 支洞是图的边，洞室是图的节点；
* 最终几何仍交给 ``generator.py`` 的 SDF/smooth-union 阶段。

参考：
https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from ..models import CaveNetwork, Point


_MASK_64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """返回跨进程稳定的 64 位混合值，不依赖 Python hash 随机化。"""

    value &= _MASK_64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK_64
    value ^= value >> 31
    return value & _MASK_64


def _unit(seed: int, index: int) -> float:
    """从 seed 和索引得到 [0, 1) 内的确定性浮点数。"""

    mixed = _mix64(seed + index * 0x9E3779B97F4A7C15)
    return float(mixed >> 11) / float(1 << 53)


def _lerp(start: Point, end: Point, amount: float) -> Point:
    return Point(
        start.x + (end.x - start.x) * amount,
        start.z + (end.z - start.z) * amount,
    )


@dataclass(frozen=True)
class CaveTopology:
    """一张洞穴图：折线边和洞室节点。"""

    paths: tuple[tuple[Point, ...], ...]
    chambers: tuple[Point, ...]
    entrances: tuple[Point, ...]


@lru_cache(maxsize=128)
def generate_cave_topology(network: CaveNetwork, seed: int) -> CaveTopology:
    """从手工锚点确定性地产生主洞、支洞和洞室节点。"""

    paths = list(network.paths)
    anchors = [point for path in network.paths for point in path]
    branch_endpoints: list[Point] = []

    for branch_index in range(network.branch_count):
        source_path = network.paths[branch_index % len(network.paths)]
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
    chambers: list[Point] = list(network.chamber_centers)
    for index in range(network.chamber_count):
        if not chamber_candidates:
            break
        candidate = chamber_candidates[
            int(_unit(seed, 10_000 + index * 5) * len(chamber_candidates))
            % len(chamber_candidates)
        ]
        chambers.append(candidate)

    entrance_candidates = anchors + branch_endpoints
    entrances: list[Point] = list(network.entrance_points)
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
