"""道路使用的地形采样与走廊检查，寻路和平滑共用同一套约束。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .terrain_config import Point


@dataclass
class RoadTerrain:
    terrain: np.ndarray
    water: np.ndarray
    x_axis: np.ndarray
    z_axis: np.ndarray
    width: int

    def __post_init__(self) -> None:
        # 先扩大水体/边界禁入范围，防止中心线在岸上、道路边缘却落在水中。
        blocked = (self.water >= 0.0) | ~np.isfinite(self.terrain) | ~np.isfinite(self.water)
        dx = float(self.x_axis[1] - self.x_axis[0]) if self.x_axis.size > 1 else 1.0
        dz = float(self.z_axis[1] - self.z_axis[0]) if self.z_axis.size > 1 else 1.0
        self.dx, self.dz = dx, dz
        rx = int(math.ceil((self.width // 2) / dx))
        rz = int(math.ceil((self.width // 2) / dz))
        padded = np.pad(blocked, ((rz, rz), (rx, rx)), constant_values=True)
        self.blocked = np.zeros_like(blocked)
        for oz in range(2 * rz + 1):
            for ox in range(2 * rx + 1):
                self.blocked |= padded[oz:oz + blocked.shape[0], ox:ox + blocked.shape[1]]

    def cell(self, x: float, z: float) -> tuple[int, int] | None:
        if not (self.x_axis[0] <= x <= self.x_axis[-1]
                and self.z_axis[0] <= z <= self.z_axis[-1]):
            return None
        return (int(round((z - self.z_axis[0]) / self.dz)),
                int(round((x - self.x_axis[0]) / self.dx)))

    def dry(self, point: Point) -> bool:
        cell = self.cell(point.x, point.z)
        return cell is not None and not self.blocked[cell]

    def land_slope(self, first: Point, second: Point) -> float | None:
        """整段逐格检查；粗网格两端干燥不代表中间也干燥。"""

        distance = math.hypot(second.x - first.x, second.z - first.z)
        count = max(1, int(math.ceil(distance / (min(self.dx, self.dz) * 0.5))))
        previous = None
        max_slope = 0.0
        for i in range(count + 1):
            t = i / count
            cell = self.cell(first.x + (second.x - first.x) * t,
                             first.z + (second.z - first.z) * t)
            if cell is None or self.blocked[cell]:
                return None
            if previous is not None and cell != previous:
                # 对角跳格的两侧也要可通行，不能擦着水/障碍的角穿过。
                if self.blocked[cell[0], previous[1]] or self.blocked[previous[0], cell[1]]:
                    return None
                step = math.hypot((cell[0] - previous[0]) * self.dz,
                                  (cell[1] - previous[1]) * self.dx)
                max_slope = max(
                    max_slope,
                    abs(float(self.terrain[cell] - self.terrain[previous])) / step,
                )
            previous = cell
        return max_slope


__all__ = ["RoadTerrain"]
