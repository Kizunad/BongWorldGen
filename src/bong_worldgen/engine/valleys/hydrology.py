"""Priority-Flood、D8 流向与汇水面积。

这里把洼地填充后的 DEM 只用于求拓扑，不会把填洼高度写回真实地形。
Priority-Flood 的边缘出水口合同参考 RichDEM 与 pyflwdir：

- https://github.com/r-barnes/richdem/blob/master/include/richdem/depressions/Barnes2014.hpp
- https://github.com/Deltares/pyflwdir/blob/main/pyflwdir/dem.py

本文件为 NumPy/Python 独立实现，没有复制外部源码。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from collections import deque

import numpy as np


_D8_OFFSETS = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class FlowField:
    """经过填洼和 D8 路由后的水文数据场。"""

    conditioned_height: np.ndarray
    receiver: np.ndarray
    accumulation: np.ndarray
    slope: np.ndarray


def _validate_input(terrain: np.ndarray, rainfall: np.ndarray) -> None:
    if terrain.ndim != 2 or min(terrain.shape) < 3:
        raise ValueError("flow routing terrain needs at least three rows and columns")
    if rainfall.shape != terrain.shape:
        raise ValueError("flow routing rainfall must share the terrain shape")
    if not np.isfinite(terrain).all() or not np.isfinite(rainfall).all():
        raise ValueError("flow routing fields must be finite")
    if np.any(rainfall <= 0.0):
        raise ValueError("flow routing rainfall must be positive")


def _priority_flood(
    terrain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从边缘向内填洼，返回填洼 DEM、平坦区父节点和访问顺序。"""

    height, width = terrain.shape
    conditioned = np.asarray(terrain, dtype=np.float64).copy()
    visited = np.zeros(terrain.shape, dtype=bool)
    parent = np.full(terrain.size, -1, dtype=np.int64)
    queue: list[tuple[float, int]] = []

    boundary = np.zeros(terrain.shape, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    for flat_index in np.flatnonzero(boundary):
        flat_index = int(flat_index)
        visited.flat[flat_index] = True
        parent[flat_index] = flat_index
        heapq.heappush(queue, (float(conditioned.flat[flat_index]), flat_index))

    order = np.empty(terrain.size, dtype=np.int64)
    order_size = 0
    while queue:
        current_height, flat_index = heapq.heappop(queue)
        order[order_size] = flat_index
        order_size += 1
        row, column = divmod(flat_index, width)
        for row_offset, column_offset, _ in _D8_OFFSETS:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                continue
            neighbor_index = neighbor_row * width + neighbor_column
            if visited.flat[neighbor_index]:
                continue
            visited.flat[neighbor_index] = True
            parent[neighbor_index] = flat_index
            filled_height = max(float(terrain.flat[neighbor_index]), current_height)
            conditioned.flat[neighbor_index] = filled_height
            heapq.heappush(queue, (filled_height, neighbor_index))

    if order_size != terrain.size:
        raise RuntimeError("priority flood did not visit every terrain cell")
    return conditioned, parent.reshape(terrain.shape), order


def _neighbor_slices(
    shape: tuple[int, int], row_offset: int, column_offset: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    height, width = shape
    source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
    source_columns = slice(max(0, -column_offset), min(width, width - column_offset))
    target_rows = slice(max(0, row_offset), min(height, height + row_offset))
    target_columns = slice(max(0, column_offset), min(width, width + column_offset))
    return (source_rows, source_columns), (target_rows, target_columns)


def _d8_receivers(
    conditioned: np.ndarray,
    parent: np.ndarray,
    cell_size_x: float,
    cell_size_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """优先选择最陡 D8 下坡；填平区沿 Priority-Flood 父树排出。"""

    flat_indices = np.arange(conditioned.size, dtype=np.int64).reshape(conditioned.shape)
    receiver = parent.copy()
    best_slope = np.full(conditioned.shape, -np.inf, dtype=np.float64)

    for row_offset, column_offset, _ in _D8_OFFSETS:
        source, target = _neighbor_slices(conditioned.shape, row_offset, column_offset)
        distance = math.hypot(column_offset * cell_size_x, row_offset * cell_size_z)
        slope = (conditioned[source] - conditioned[target]) / distance
        better = (slope > 0.0) & (slope > best_slope[source])
        receiver[source] = np.where(better, flat_indices[target], receiver[source])
        best_slope[source] = np.where(better, slope, best_slope[source])

    # 规划域边缘就是明确出水口，不能因为邻近内陆更低又倒流回域内。
    receiver[0, :] = flat_indices[0, :]
    receiver[-1, :] = flat_indices[-1, :]
    receiver[:, 0] = flat_indices[:, 0]
    receiver[:, -1] = flat_indices[:, -1]

    receiver_rows, receiver_columns = np.divmod(receiver, conditioned.shape[1])
    rows, columns = np.indices(conditioned.shape)
    receiver_distance = np.hypot(
        (receiver_columns - columns) * cell_size_x,
        (receiver_rows - rows) * cell_size_z,
    )
    return receiver, receiver_distance


def route_flow_d8(
    terrain: np.ndarray,
    rainfall: np.ndarray,
    *,
    cell_size_x: float,
    cell_size_z: float,
    minimum_slope: float,
) -> FlowField:
    """计算 D8 下游节点、汇水面积和接收节点方向的局部坡度。"""

    terrain = np.asarray(terrain, dtype=np.float64)
    rainfall = np.asarray(rainfall, dtype=np.float64)
    _validate_input(terrain, rainfall)
    if cell_size_x <= 0 or cell_size_z <= 0 or minimum_slope <= 0:
        raise ValueError("flow routing cell sizes and minimum_slope must be positive")

    conditioned, parent, _ = _priority_flood(terrain)
    receiver, receiver_distance = _d8_receivers(
        conditioned,
        parent,
        cell_size_x,
        cell_size_z,
    )
    flat_receiver = receiver.ravel()
    flat_conditioned = conditioned.ravel()
    flat_distance = receiver_distance.ravel()
    own_index = np.arange(terrain.size, dtype=np.int64)
    drains = flat_receiver != own_index

    accumulation = rainfall.ravel().copy() * cell_size_x * cell_size_z
    # D8 接收关系是有向无环图：严格下坡边降低高度，平坦区边沿 PF 父树
    # 排出。按入度做拓扑累加，比依赖 heap 的同高 tie-break 更可靠。
    downstream_count = np.bincount(flat_receiver[drains], minlength=terrain.size)
    ready = deque(int(index) for index in np.flatnonzero(downstream_count == 0))
    processed = 0
    while ready:
        flat_index = ready.popleft()
        processed += 1
        destination = int(flat_receiver[flat_index])
        if destination == flat_index:
            continue
        accumulation[destination] += accumulation[flat_index]
        downstream_count[destination] -= 1
        if downstream_count[destination] == 0:
            ready.append(destination)
    if processed != terrain.size:
        raise RuntimeError("D8 flow graph contains a cycle")

    slope = np.zeros(terrain.size, dtype=np.float64)
    raw_drop = flat_conditioned - flat_conditioned[flat_receiver]
    slope[drains] = np.maximum(
        raw_drop[drains] / np.maximum(flat_distance[drains], 1.0e-12),
        minimum_slope,
    )
    return FlowField(
        conditioned_height=np.ascontiguousarray(conditioned),
        receiver=np.ascontiguousarray(receiver),
        accumulation=np.ascontiguousarray(accumulation.reshape(terrain.shape)),
        slope=np.ascontiguousarray(slope.reshape(terrain.shape)),
    )


__all__ = ["FlowField", "route_flow_d8"]
