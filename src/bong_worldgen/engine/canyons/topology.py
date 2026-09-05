"""确定性的大峡谷中心线生成。

峡谷中心线只由 seed 和 XZ 配置决定，不读取高度场。这样任何地表高度都
可以成为峡谷起点，避免“只有山顶才会生成峡谷”的隐式过滤。
"""

from __future__ import annotations

import math

from ..terrain_config import Canyon, Point
from ..randomness import stable_text_seed, unit_interval as _unit


def _smooth_path(points: list[Point]) -> tuple[Point, ...]:
    """对随机游走折点做一次平滑，避免峡谷边线呈折尺状。"""

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


def _reflect(point: Point, center: Point, extent: float, direction: float) -> tuple[Point, float]:
    """在生成域边缘反射路径，保持连续方向而不是硬截断。"""

    local_x = point.x - center.x
    local_z = point.z - center.z
    if abs(local_x) > extent:
        direction = math.pi - direction
        local_x = math.copysign(extent, local_x) - (abs(local_x) - extent)
    if abs(local_z) > extent:
        direction = -direction
        local_z = math.copysign(extent, local_z) - (abs(local_z) - extent)
    return Point(center.x + local_x, center.z + local_z), direction


def _generated_paths(canyon: Canyon, seed: int) -> tuple[tuple[Point, ...], ...]:
    """从任意 XZ 起点生成持续转向的长峡谷中心线。"""

    paths: list[tuple[Point, ...]] = []
    name_seed = stable_text_seed(canyon.name)
    for path_index in range(canyon.path_count):
        base = name_seed + path_index * 10_007
        start_radius = canyon.seed_extent * (0.08 + 0.72 * _unit(seed, base + 1))
        start_angle = _unit(seed, base + 2) * math.tau
        start = Point(
            canyon.seed_center.x + math.cos(start_angle) * start_radius,
            canyon.seed_center.z + math.sin(start_angle) * start_radius,
        )
        direction = _unit(seed, base + 3) * math.tau
        points = [start]
        for segment in range(canyon.path_segments):
            turn = (
                _unit(seed, base + 100 + segment * 2) * 2.0 - 1.0
            ) * math.pi * canyon.path_turn * 0.34
            direction += turn
            segment_length = canyon.path_length / canyon.path_segments * (
                0.74 + 0.52 * _unit(seed, base + 101 + segment * 2)
            )
            previous = points[-1]
            next_point = Point(
                previous.x + math.cos(direction) * segment_length,
                previous.z + math.sin(direction) * segment_length,
            )
            next_point, direction = _reflect(
                next_point,
                canyon.seed_center,
                canyon.seed_extent,
                direction,
            )
            points.append(next_point)
        paths.append(_smooth_path(points))
    return tuple(paths)


def generate_canyon_paths(canyon: Canyon, seed: int) -> tuple[tuple[Point, ...], ...]:
    """返回显式路径或由 seed 生成的路径。"""

    return canyon.paths or _generated_paths(canyon, seed)


__all__ = ["generate_canyon_paths"]
