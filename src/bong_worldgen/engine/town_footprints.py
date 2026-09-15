"""城镇建筑和完整防御边界的占地计算与干地检查。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Footprint = tuple[int, int, int, int]


def dry_footprint(
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    bounds: Footprint,
) -> bool:
    """检查整个闭区间；不能用中心点代替模板边缘的水域和窗口检查。"""

    min_x, max_x, min_z, max_z = bounds
    if not (
        x_axis[0] <= min_x <= max_x <= x_axis[-1]
        and z_axis[0] <= min_z <= max_z <= z_axis[-1]
    ):
        return False
    step_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    step_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    left, right = (int(round((value - x_axis[0]) / step_x)) for value in (min_x, max_x))
    top, bottom = (int(round((value - z_axis[0]) / step_z)) for value in (min_z, max_z))
    levels = water[top : bottom + 1, left : right + 1]
    return bool(np.all(np.isfinite(levels) & (levels < 0.0)))


@dataclass(frozen=True)
class WallFootprint:
    """布局和发射共用模板尺寸，包含完整墙段、城门深度及角塔外伸。"""

    wall_span: int
    wall_thickness: int
    gate_span: int = 0
    gate_depth: int = 0
    tower_width: int = 0
    tower_length: int = 0
    margin: int = 0

    def wall_bounds(self, buildings: Footprint) -> Footprint:
        min_x, max_x, min_z, max_z = buildings
        min_x, max_x = min_x - self.margin, max_x + self.margin
        min_z, max_z = min_z - self.margin, max_z + self.margin
        long_axis_x = max_x - min_x >= max_z - min_z
        required_span = self.gate_span + 2 * self.wall_span

        def align(low: int, high: int, minimum: int) -> tuple[int, int]:
            span = max(high - low + 1, minimum, self.wall_span)
            count = math.ceil(span / self.wall_span)
            aligned_low = math.floor(low / self.wall_span) * self.wall_span
            count = max(count, math.ceil((high - aligned_low + 1) / self.wall_span))
            return aligned_low, aligned_low + count * self.wall_span - 1

        min_x, max_x = align(min_x, max_x, required_span if long_axis_x else 0)
        min_z, max_z = align(min_z, max_z, 0 if long_axis_x else required_span)
        return min_x, max_x, min_z, max_z

    def footprints(self, buildings: Footprint) -> tuple[Footprint, ...]:
        min_x, max_x, min_z, max_z = self.wall_bounds(buildings)
        long_axis_x = max_x - min_x >= max_z - min_z
        negative = self.wall_thickness // 2
        positive = (self.wall_thickness - 1) // 2
        result = [
            (min_x, max_x, side - negative, side + positive)
            for side in (min_z, max_z)
        ] + [
            (side - negative, side + positive, min_z, max_z)
            for side in (min_x, max_x)
        ]

        def centered(x: int, z: int, width: int, length: int) -> Footprint:
            left, top = x - width // 2, z - length // 2
            return left, left + width - 1, top, top + length - 1

        # 门体纵深只影响门口，不能把整条城墙都扩成同样宽的禁入带。
        if self.gate_span and self.gate_depth:
            if long_axis_x:
                result.extend(
                    centered((min_x + max_x) // 2, side, self.gate_span, self.gate_depth)
                    for side in (min_z, max_z)
                )
            else:
                result.extend(
                    centered(side, (min_z + max_z) // 2, self.gate_depth, self.gate_span)
                    for side in (min_x, max_x)
                )
        if self.tower_width and self.tower_length:
            result.extend(
                centered(x, z, self.tower_width, self.tower_length)
                for x in (min_x, max_x)
                for z in (min_z, max_z)
            )
        return tuple(result)
