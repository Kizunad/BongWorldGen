"""跨水道路的桥段规划与方块化，不修改河床或水位。

桥段作为 A* 的高代价、岸到岸连接边参与寻路，不是事后把水替换成路。
图边建模参考：https://www.redblobgames.com/pathfinding/a-star/implementation.html
Minecraft 聚落道路参考：https://github.com/Jandhi/Tome
这里只借鉴图建模思想；跨度、接坡和方块结构是本项目实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterator

import numpy as np

from .road_terrain import RoadTerrain
from .terrain_config import Point, TownSettings


@dataclass(frozen=True)
class BridgeSpan:
    start: Point
    end: Point
    width: int
    deck: tuple[int, ...]
    wet_start: int
    wet_end: int

    def position(self, station: int, lateral: int) -> tuple[int, int]:
        dx = int(np.sign(self.end.x - self.start.x))
        dz = int(np.sign(self.end.z - self.start.z))
        return (int(self.start.x) + dx * station - dz * lateral,
                int(self.start.z) + dz * station + dx * lateral)


def plan_bridge(
    start: Point,
    end: Point,
    grid: RoadTerrain,
    settings: TownSettings,
) -> BridgeSpan | None:
    """验证一条直桥：双岸、净空、跨度、接坡及桥墩深度必须同时满足。

    首版采用沿 X/Z 的直桥；陆路可以自由转弯，水上不做弯桥。
    ``deck`` 的 Y 是桥面方块底面，不是玩家脚部高度。
    """

    if settings.bridge_max_span <= 0:
        return None
    if (start.x != end.x and start.z != end.z) or start == end:
        return None
    if any(value != round(value) for value in (start.x, start.z, end.x, end.z)):
        return None
    length = int(abs(end.x - start.x) + abs(end.z - start.z))
    if length > settings.bridge_max_span + 2 * settings.bridge_approach_length:
        return None
    provisional = BridgeSpan(start, end, grid.width, (), 0, 0)
    # 护栏在可行走路面外各占一格；连这两列也必须得到完整检查。
    # 写成显式左右边界，避免 Python 对负数整除导致奇数宽度偏一格。
    lateral = range(-(grid.width // 2) - 1, grid.width - grid.width // 2 + 1)
    grounds: list[list[int]] = []
    levels: list[float] = []
    for station in range(length + 1):
        cells = [grid.cell(*provisional.position(station, offset)) for offset in lateral]
        if any(cell is None for cell in cells):
            return None
        if any(not np.isfinite(grid.terrain[cell]) for cell in cells):
            return None
        grounds.append([int(round(float(grid.terrain[cell]))) for cell in cells])
        levels.append(max(float(grid.water[cell]) for cell in cells))
    wet = np.flatnonzero(np.asarray(levels) >= 0.0)
    if wet.size == 0:
        return None
    first, last = int(wet[0]), int(wet[-1])
    if (first < 2 or length - last < 2 or last - first + 1 > settings.bridge_max_span
            or first > settings.bridge_approach_length
            or length - last > settings.bridge_approach_length):
        return None
    # 不把多个独立水体或中间岛屿拼成一条长高架。
    if wet.size != last - first + 1:
        return None
    deck_y = max(math.ceil(max(levels)) + settings.bridge_clearance,
                 max(grounds[first - 1]), max(grounds[last + 1]))
    heights = np.full(length + 1, float(deck_y))
    heights[:first] = np.linspace(max(grounds[0]), deck_y, first)
    heights[last + 1:] = np.linspace(deck_y, max(grounds[-1]), length - last)
    if np.max(np.abs(np.diff(heights))) > min(settings.road_max_slope, 1.0) + 1.0e-9:
        return None
    deck = np.rint(heights).astype(np.int64)
    for station, ground_row in enumerate(grounds):
        # 不开挖岸坡、不跨过深渊；选择另一处桥位比造悬空断头路可靠。
        if (max(ground_row) > deck[station]
                or deck[station] - min(ground_row) > settings.bridge_max_support_depth):
            return None
    if max(grounds[0]) - min(grounds[0]) > 1 or max(grounds[-1]) - min(grounds[-1]) > 1:
        return None
    return BridgeSpan(start, end, grid.width, tuple(int(y) for y in deck), first, last)


def bridge_blocks(
    bridge: BridgeSpan,
    grid: RoadTerrain,
    settings: TownSettings,
) -> Iterator[tuple[int, int, int, str, str]]:
    """木桥面、两侧护栏、干岸桥台及水中边墩；不封死中央水道。"""

    left = -(bridge.width // 2) - 1
    right = bridge.width - bridge.width // 2
    pier_stations = set(range(bridge.wet_start, bridge.wet_end + 1, 6))
    for station, y in enumerate(bridge.deck):
        for offset in range(left, right + 1):
            x, z = bridge.position(station, offset)
            yield x, y, z, settings.bridge_deck_material, "bridge_deck"
            cell = grid.cell(x, z)
            ground = int(round(float(grid.terrain[cell])))
            dry = grid.water[cell] < 0.0
            if dry or (station in pier_stations and offset in (left, right)):
                for support_y in range(ground + 1, y):
                    yield x, support_y, z, settings.bridge_support_material, "bridge_support"
            # 接坡的出入口不横放护栏；护栏只位于桥面的侧边。
            if offset in (left, right) and 0 < station < len(bridge.deck) - 1:
                material = settings.bridge_rail_material
                if material.endswith("_fence"):
                    connections = ("east=true,west=true" if bridge.start.z == bridge.end.z
                                   else "north=true,south=true")
                    material = f"minecraft:{material.removeprefix('minecraft:')}[{connections}]"
                yield x, y + 1, z, material, "bridge_rail"


__all__ = ["BridgeSpan", "plan_bridge", "bridge_blocks"]
