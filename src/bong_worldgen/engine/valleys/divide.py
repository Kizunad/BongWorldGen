"""从 D8 流域边界提取并修饰山脊。

Watershed Divide 不是另一条随机山脉，而是同一张水文图上相邻流域的
分界线。本模块先追踪每个网格单元最终流向的边缘出口，再把出口标签不同
的相邻单元标成分水岭，最后以世界坐标宽度形成连续的窄脊。算法组织参考
以下公开 DEM 水文实现，未复制其源码：

- https://github.com/Deltares/pyflwdir
- https://github.com/mdbartos/pysheds
- https://github.com/r-barnes/richdem/blob/master/include/richdem/depressions/Barnes2014.hpp
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain_config import ValleySystem
from .hydrology import FlowField


_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _outlet_labels(receiver: np.ndarray) -> np.ndarray:
    """为每个 D8 单元计算最终出口标签。

    ``route_flow_d8`` 保证 receiver 图没有环；这里使用路径压缩而不是递归，
    因此大规划域不会受到 Python 递归深度限制影响。
    """

    receiver = np.asarray(receiver, dtype=np.int64)
    if receiver.ndim != 2:
        raise ValueError("watershed receiver must be two-dimensional")
    size = receiver.size
    flat_receiver = receiver.ravel()
    indices = np.arange(size, dtype=np.int64)
    if np.any(flat_receiver < 0) or np.any(flat_receiver >= size):
        raise ValueError("watershed receiver contains an invalid cell index")

    labels = np.full(size, -1, dtype=np.int64)
    outlets = flat_receiver == indices
    labels[outlets] = indices[outlets]
    for _ in range(size):
        unresolved = labels < 0
        if not np.any(unresolved):
            return labels.reshape(receiver.shape)
        next_indices = flat_receiver[unresolved]
        known = labels[next_indices] >= 0
        if np.any(known):
            unresolved_indices = np.flatnonzero(unresolved)
            labels[unresolved_indices[known]] = labels[next_indices[known]]
        if not np.any(labels[unresolved] >= 0):
            break
    raise ValueError("watershed receiver contains a cycle without an outlet")


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """标出相邻流域标签不同的内部单元。"""

    boundary = np.zeros(labels.shape, dtype=bool)
    height, width = labels.shape
    for row_offset, column_offset in _NEIGHBOR_OFFSETS:
        source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
        source_columns = slice(max(0, -column_offset), min(width, width - column_offset))
        target_rows = slice(max(0, row_offset), min(height, height + row_offset))
        target_columns = slice(max(0, column_offset), min(width, width + column_offset))
        boundary[source_rows, source_columns] |= (
            labels[source_rows, source_columns]
            != labels[target_rows, target_columns]
        )

    # 规划域边缘是人为设置的出水口，不是山脊；排除边缘避免外围形成一圈假墙。
    boundary[0, :] = False
    boundary[-1, :] = False
    boundary[:, 0] = False
    boundary[:, -1] = False
    return boundary


def _boundary_corridor(
    boundary: np.ndarray,
    *,
    cell_size_x: float,
    cell_size_z: float,
    width: float,
) -> np.ndarray:
    """把一格宽的拓扑分界线扩为连续的世界坐标窄脊。"""

    if width <= 0.0:
        return boundary.astype(np.float64)
    radius_x = math.ceil(width / cell_size_x)
    radius_z = math.ceil(width / cell_size_z)
    corridor = np.zeros(boundary.shape, dtype=np.float64)
    height, grid_width = boundary.shape
    for row_offset in range(-radius_z, radius_z + 1):
        for column_offset in range(-radius_x, radius_x + 1):
            # 规划域小于分水岭宽度时，超出整个数组的偏移没有重叠区域；
            # 显式跳过它们，避免空切片与非空切片发生广播错误。
            if abs(row_offset) >= height or abs(column_offset) >= grid_width:
                continue
            distance = math.hypot(
                row_offset * cell_size_z,
                column_offset * cell_size_x,
            )
            if distance > width:
                continue
            source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            source_columns = slice(
                max(0, -column_offset), min(grid_width, grid_width - column_offset)
            )
            target_rows = slice(max(0, row_offset), min(height, height + row_offset))
            target_columns = slice(
                max(0, column_offset), min(grid_width, grid_width + column_offset)
            )
            falloff = math.exp(-0.5 * (distance / max(width * 0.58, 1.0e-12)) ** 2)
            corridor[target_rows, target_columns] = np.maximum(
                corridor[target_rows, target_columns],
                boundary[source_rows, source_columns] * falloff,
            )
    # 分水岭边界不包含规划域边缘；扩散后的窄脊也必须保持这一合同，
    # 否则每个水文域外围都会出现一圈与地貌无关的抬升。
    corridor[0, :] = 0.0
    corridor[-1, :] = 0.0
    corridor[:, 0] = 0.0
    corridor[:, -1] = 0.0
    return corridor


def _crestness(height: np.ndarray) -> np.ndarray:
    """计算单元相对八邻域的局部冠部权重。"""

    result = np.zeros(height.shape, dtype=np.float64)
    count = np.zeros(height.shape, dtype=np.float64)
    for row_offset, column_offset in _NEIGHBOR_OFFSETS:
        source_rows = slice(max(0, -row_offset), min(height.shape[0], height.shape[0] - row_offset))
        source_columns = slice(
            max(0, -column_offset), min(height.shape[1], height.shape[1] - column_offset)
        )
        target_rows = slice(max(0, row_offset), min(height.shape[0], height.shape[0] + row_offset))
        target_columns = slice(
            max(0, column_offset), min(height.shape[1], height.shape[1] + column_offset)
        )
        result[source_rows, source_columns] += (
            height[source_rows, source_columns] >= height[target_rows, target_columns]
        )
        count[source_rows, source_columns] += 1.0
    return result / np.maximum(count, 1.0)


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def watershed_divide_strength(
    flow: FlowField,
    system: ValleySystem,
    *,
    cell_size_x: float,
    cell_size_z: float,
    uplift: np.ndarray | None = None,
    stream_power: np.ndarray | None = None,
    erosion_depth: np.ndarray | None = None,
) -> np.ndarray:
    """返回可叠加到地形上的分水岭抬升场，单位与高度场相同。

    分水岭强度由三部分共同决定：拓扑边界给出位置，局部冠部权重避免在
    河谷底部画脊，Uplift 提供构造背景；Stream Power 越强的地方越接近河槽，
    会抑制修饰，避免把新山脊重新盖到刚切出的谷地里。
    """

    terrain = np.asarray(flow.conditioned_height, dtype=np.float64)
    if cell_size_x <= 0.0 or cell_size_z <= 0.0:
        raise ValueError("watershed divide cell sizes must be positive")
    labels = _outlet_labels(flow.receiver)
    boundary = _boundary_mask(labels)
    corridor = _boundary_corridor(
        boundary,
        cell_size_x=cell_size_x,
        cell_size_z=cell_size_z,
        width=system.watershed_divide_width,
    )
    crest_weight = 0.35 + 0.65 * _crestness(terrain)

    if uplift is None:
        uplift_field = np.zeros(terrain.shape, dtype=np.float64)
    else:
        uplift_field = np.asarray(uplift, dtype=np.float64)
        if uplift_field.shape != terrain.shape:
            raise ValueError("watershed divide uplift must share the flow shape")
        if not np.isfinite(uplift_field).all() or np.any(uplift_field < 0.0):
            raise ValueError("watershed divide uplift must be finite and non-negative")
    uplift_weight = 0.55 + 0.45 * _normalize(uplift_field)

    if stream_power is None:
        stream_power_field = np.zeros(terrain.shape, dtype=np.float64)
    else:
        stream_power_field = np.asarray(stream_power, dtype=np.float64)
        if stream_power_field.shape != terrain.shape:
            raise ValueError("watershed divide stream power must share the flow shape")
        if not np.isfinite(stream_power_field).all() or np.any(stream_power_field < 0.0):
            raise ValueError("watershed divide stream power must be finite and non-negative")
    stream_weight = 1.0 - 0.72 * _normalize(stream_power_field)

    score = np.clip(corridor * crest_weight * uplift_weight * stream_weight, 0.0, 1.0)
    divide = system.watershed_divide_strength * score
    if erosion_depth is not None:
        valley_depth = np.asarray(erosion_depth, dtype=np.float64)
        if valley_depth.shape != terrain.shape:
            raise ValueError("watershed divide erosion depth must share the flow shape")
        if not np.isfinite(valley_depth).all() or np.any(valley_depth < 0.0):
            raise ValueError("watershed divide erosion depth must be finite and non-negative")
        # 分水岭不是重新填谷：在已被 Stream Power 切开的单元最多回填配置比例，
        # 确保谷地的负形态仍然可见，只有未切蚀的高地承接完整窄脊抬升。
        divide = np.minimum(
            divide,
            valley_depth * system.watershed_divide_valley_fill,
        )
    return np.ascontiguousarray(divide, dtype=np.float64)


__all__ = ["watershed_divide_strength"]
