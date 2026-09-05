"""适用于自然聚落的乡村道路路径生成。

道路不是建筑之间的直线，而是在地形代价场中逐步寻找一条便宜路线：

* A* 负责绕开水体、陡坡和悬崖；
* 低频、seed 绑定的随机势场让路线产生连续的自然偏转；
* Chaikin corner-cutting 消除网格寻路留下的折角。

参考资料（仅借鉴公开算法思想，未复制代码）：
* Red Blob Games, A* pathfinding: https://www.redblobgames.com/pathfinding/a-star/
* Parish 与 Müller, Procedural Modeling of Cities:
  https://dl.acm.org/doi/10.1145/383259.383292
* Yatoom/city-generator（道路生长的程序化聚落案例）：
  https://github.com/Yatoom/city-generator
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .bridges import BridgeSpan, plan_bridge
from .randomness import stable_text_seed, unit_interval
from .road_terrain import RoadTerrain
from .terrain_config import Point, TownSettings


def _clamped_cell(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    world_x: float,
    world_z: float,
) -> tuple[int, int]:
    """将世界坐标映射到当前 tile，越界时钳制到边缘。"""

    step_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    step_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    column = int(np.clip(np.rint((world_x - x_axis[0]) / max(step_x, 1.0e-9)), 0, x_axis.size - 1))
    row = int(np.clip(np.rint((world_z - z_axis[0]) / max(step_z, 1.0e-9)), 0, z_axis.size - 1))
    return row, column


def _road_noise(world_x: float, world_z: float, seed: int) -> float:
    """采样连续的低频随机势场，而不是给每个方块单独掷骰子。"""

    lattice_size = 64.0
    lattice_x = math.floor(world_x / lattice_size)
    lattice_z = math.floor(world_z / lattice_size)
    tx = world_x / lattice_size - lattice_x
    tz = world_z / lattice_size - lattice_z

    def corner(offset_x: int, offset_z: int) -> float:
        coordinate_seed = (
            seed
            + (lattice_x + offset_x) * 73856093
            + (lattice_z + offset_z) * 19349663
        )
        return unit_interval(coordinate_seed, 917)

    top = corner(0, 0) * (1.0 - tx) + corner(1, 0) * tx
    bottom = corner(0, 1) * (1.0 - tx) + corner(1, 1) * tx
    return top * (1.0 - tz) + bottom * tz


def _chaikin_path(path: list[Point], passes: int) -> list[Point]:
    """对折线路径做 corner-cutting，同时保持端点不动。"""

    result = path
    for _ in range(passes):
        if len(result) < 3:
            break
        smoothed = [result[0]]
        for first, second in zip(result, result[1:]):
            smoothed.extend(
                (
                    Point(
                        0.75 * first.x + 0.25 * second.x,
                        0.75 * first.z + 0.25 * second.z,
                    ),
                    Point(
                        0.25 * first.x + 0.75 * second.x,
                        0.25 * first.z + 0.75 * second.z,
                    ),
                )
            )
        smoothed.append(result[-1])
        result = smoothed
    return result


@dataclass(frozen=True)
class RoadPlan:
    """中心线与经过验证的桥段；不可达时返回空计划。"""

    path: tuple[Point, ...] = ()
    bridges: tuple[BridgeSpan, ...] = ()


def _bridge_neighbors(
    current: tuple[int, int],
    rows: np.ndarray,
    columns: np.ndarray,
    grid: RoadTerrain,
    settings: TownSettings,
):
    """向四个方向寻找可施工的双岸连接；水中不设普通寻路节点。"""

    if settings.bridge_max_span <= 0:
        return
    row, column = current
    rr, cc = int(rows[row]), int(columns[column])
    start = Point(float(grid.x_axis[cc]), float(grid.z_axis[rr]))
    radius_x = int(math.ceil(settings.bridge_approach_length / grid.dx)) + grid.width
    radius_z = int(math.ceil(settings.bridge_approach_length / grid.dz)) + grid.width
    nearby = grid.water[max(0, rr-radius_z):rr+radius_z+1,
                        max(0, cc-radius_x):cc+radius_x+1]
    if not np.any(nearby >= 0.0):
        return
    limit = settings.bridge_max_span + 2 * settings.bridge_approach_length
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, column + dc
        while 0 <= nr < len(rows) and 0 <= nc < len(columns):
            end = Point(float(grid.x_axis[columns[nc]]), float(grid.z_axis[rows[nr]]))
            distance = abs(end.x - start.x) + abs(end.z - start.z)
            if distance > limit:
                break
            if grid.dry(end):
                er, ec = int(rows[nr]), int(columns[nc])
                pad_x = int(math.ceil((grid.width // 2 + 1) / grid.dx))
                pad_z = int(math.ceil((grid.width // 2 + 1) / grid.dz))
                corridor = grid.water[
                    max(0, min(rr, er)-pad_z):max(rr, er)+pad_z+1,
                    max(0, min(cc, ec)-pad_x):max(cc, ec)+pad_x+1,
                ]
                bridge = (plan_bridge(start, end, grid, settings)
                          if np.any(corridor >= 0.0) else None)
                if bridge is not None:
                    yield (nr, nc), bridge, distance * settings.bridge_cost_factor
                    # 同方向先接到最近的可施工岸点，后续延伸仍由陆路负责。
                    break
            nr, nc = nr + dr, nc + dc


def _astar_path(
    grid: RoadTerrain,
    start: tuple[int, int],
    target: tuple[int, int],
    rows: np.ndarray,
    columns: np.ndarray,
    settings: TownSettings,
    seed: int,
) -> RoadPlan:
    """八方向陆路与双岸桥段共用 A*；禁止不可达时回退成非法直线。"""

    def point(node: tuple[int, int]) -> Point:
        return Point(float(grid.x_axis[columns[node[1]]]), float(grid.z_axis[rows[node[0]]]))

    end = point(target)
    def heuristic(node: tuple[int, int]) -> float:
        p = point(node)
        return math.hypot(p.x - end.x, p.z - end.z)

    if not grid.dry(point(start)) or not grid.dry(end):
        return RoadPlan()
    queue = [(heuristic(start), 0.0, start)]
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[tuple[int, int], BridgeSpan | None]] = {}
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        _, current_distance, current = heapq.heappop(queue)
        if current_distance != distances.get(current):
            continue
        if current == target:
            break
        first = point(current)
        neighbors = []
        for dr, dc in directions:
            neighbor = current[0] + dr, current[1] + dc
            if not (0 <= neighbor[0] < len(rows) and 0 <= neighbor[1] < len(columns)):
                continue
            second = point(neighbor)
            slope = grid.land_slope(first, second)
            # road_max_slope 是偏好坡度；超过每格一格高差则不能当普通步道。
            if slope is None or slope > 1.0:
                continue
            distance = math.hypot(second.x - first.x, second.z - first.z)
            cost = distance * (
                1.0 + settings.road_wander * _road_noise(second.x, second.z, seed)
                + settings.road_slope_penalty * slope * slope
                + settings.road_cliff_penalty * max(0.0, slope - settings.road_max_slope)
            )
            neighbors.append((neighbor, None, cost))
        neighbors.extend(_bridge_neighbors(current, rows, columns, grid, settings))
        for neighbor, bridge, cost in neighbors:
            candidate = current_distance + cost
            if candidate >= distances.get(neighbor, math.inf):
                continue
            distances[neighbor] = candidate
            previous[neighbor] = current, bridge
            heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
    if target not in distances:
        return RoadPlan()
    nodes, bridges = [target], []
    while nodes[-1] != start:
        predecessor, bridge = previous[nodes[-1]]
        nodes.append(predecessor)
        if bridge is not None:
            bridges.append(bridge)
    return RoadPlan(tuple(point(node) for node in reversed(nodes)), tuple(reversed(bridges)))


def _smooth_land(plan: RoadPlan, grid: RoadTerrain, passes: int) -> RoadPlan:
    """仅平滑连续陆路；桥头与直桥锁定，整段复核后才接受平滑结果。"""

    bridge_edges = {(bridge.start, bridge.end) for bridge in plan.bridges}
    result: list[Point] = []
    run: list[Point] = []

    def flush() -> None:
        if not run:
            return
        smoothed = _chaikin_path(run, passes)
        slopes = [grid.land_slope(a, b) for a, b in zip(smoothed, smoothed[1:])]
        chosen = smoothed if all(s is not None and s <= 1.0 for s in slopes) else run
        result.extend(chosen if not result else chosen[1:])

    for first, second in zip(plan.path, plan.path[1:]):
        if not run:
            run.append(first)
        if (first, second) in bridge_edges:
            flush()
            result.append(second)
            run = [second]
        else:
            run.append(second)
    flush()
    return RoadPlan(tuple(result) if result else plan.path, plan.bridges)


def plan_rural_road(
    start: Point,
    end: Point,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    seed: int,
) -> RoadPlan:
    """先验证整条路线，再铺路；随机导引点不可达时重新尝试直接寻路。"""

    if terrain.shape != water.shape or terrain.ndim != 2:
        raise ValueError("rural road terrain and water fields must share a 2D shape")
    if (x_axis.ndim != 1 or z_axis.ndim != 1
            or terrain.shape != (z_axis.size, x_axis.size)
            or x_axis.size == 0 or z_axis.size == 0):
        raise ValueError("rural road axes must match the terrain field")
    if (not np.all(np.isfinite(x_axis)) or not np.all(np.isfinite(z_axis))
            or np.any(np.diff(x_axis) <= 0) or np.any(np.diff(z_axis) <= 0)
            or (x_axis.size > 2 and not np.allclose(np.diff(x_axis), np.diff(x_axis)[0]))
            or (z_axis.size > 2 and not np.allclose(np.diff(z_axis), np.diff(z_axis)[0]))):
        raise ValueError("rural road expects regular increasing axes")
    grid = RoadTerrain(terrain, water, x_axis, z_axis, settings.road_width)
    # 与铺设使用同一方块中心，避免粗网格吸附后直接替换端点切穿岸线。
    start_cell = grid.cell(start.x, start.z)
    end_cell = grid.cell(end.x, end.z)
    if start_cell is None or end_cell is None:
        return RoadPlan()
    start = Point(float(x_axis[start_cell[1]]), float(z_axis[start_cell[0]]))
    end = Point(float(x_axis[end_cell[1]]), float(z_axis[end_cell[0]]))
    if not grid.dry(start) or not grid.dry(end):
        return RoadPlan()
    direct_distance = math.hypot(end.x - start.x, end.z - start.z)
    guides = [start]
    if direct_distance >= 80.0 and settings.road_wander > 0.0:
        normal_x, normal_z = -(end.z-start.z)/direct_distance, (end.x-start.x)/direct_distance
        offset_limit = min(direct_distance * settings.road_wander * 0.22, 80.0)
        guide_seed = seed + stable_text_seed("rural_road_guides")
        for index, progress in enumerate((0.34, 0.67)):
            offset = (unit_interval(guide_seed, index) * 2.0 - 1.0) * offset_limit
            candidate = Point(start.x + (end.x-start.x)*progress + normal_x*offset,
                              start.z + (end.z-start.z)*progress + normal_z*offset)
            if grid.dry(candidate):
                guides.append(candidate)
    guides.append(end)
    guide_cells = [grid.cell(p.x, p.z) for p in guides]
    margin_x = int(math.ceil(settings.road_search_margin / grid.dx))
    margin_z = int(math.ceil(settings.road_search_margin / grid.dz))
    rows = np.unique(np.concatenate((
        np.arange(max(0, min(r for r, c in guide_cells)-margin_z),
                  min(len(z_axis), max(r for r, c in guide_cells)+margin_z+1),
                  settings.road_path_step),
        [r for r, c in guide_cells],
    ))).astype(np.int64)
    columns = np.unique(np.concatenate((
        np.arange(max(0, min(c for r, c in guide_cells)-margin_x),
                  min(len(x_axis), max(c for r, c in guide_cells)+margin_x+1),
                  settings.road_path_step),
        [c for r, c in guide_cells],
    ))).astype(np.int64)

    def route(cells: list[tuple[int, int]]) -> RoadPlan:
        path: list[Point] = []
        bridges: list[BridgeSpan] = []
        for index, (a, b) in enumerate(zip(cells, cells[1:])):
            plan = _astar_path(
                grid,
                (int(np.searchsorted(rows, a[0])), int(np.searchsorted(columns, a[1]))),
                (int(np.searchsorted(rows, b[0])), int(np.searchsorted(columns, b[1]))),
                rows, columns, settings, seed + stable_text_seed("rural_road_path") + index,
            )
            if not plan.path:
                return RoadPlan()
            path.extend(plan.path if not path else plan.path[1:])
            bridges.extend(plan.bridges)
        return RoadPlan(tuple(path), tuple(bridges))

    plan = route(guide_cells)
    if not plan.path and len(guide_cells) > 2:
        plan = route([start_cell, end_cell])
    return _smooth_land(plan, grid, settings.road_smoothing_passes)


def rural_road_path(
    start: Point, end: Point, terrain: np.ndarray, water: np.ndarray,
    x_axis: np.ndarray, z_axis: np.ndarray, settings: TownSettings, seed: int,
) -> list[Point]:
    """返回含桥头节点的中心线；实际铺设必须消费完整的 RoadPlan。"""

    return list(plan_rural_road(start, end, terrain, water, x_axis, z_axis, settings, seed).path)


__all__ = ["RoadPlan", "plan_rural_road", "rural_road_path"]
