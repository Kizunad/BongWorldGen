"""可控地貌特征共用的向量化几何工具。"""

from __future__ import annotations

import numpy as np

from .terrain_config import Point


def polyline_stations(
    path: tuple[Point, ...], station_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿折线按弧长均匀采样，返回进度、X 坐标和 Z 坐标。"""

    if len(path) < 2:
        raise ValueError("polyline needs at least two points")
    if station_count < 2:
        raise ValueError("polyline stations need at least two samples")
    lengths = np.asarray(
        [
            np.hypot(path[index + 1].x - point.x, path[index + 1].z - point.z)
            for index, point in enumerate(path[:-1])
        ],
        dtype=np.float64,
    )
    total = max(float(lengths.sum()), 1.0e-9)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, total, station_count)
    segment_ids = np.searchsorted(cumulative[1:], distances, side="right")
    segment_ids = np.minimum(segment_ids, len(path) - 2)
    segment_start = cumulative[segment_ids]
    segment_length = np.maximum(lengths[segment_ids], 1.0e-9)
    local_t = np.clip((distances - segment_start) / segment_length, 0.0, 1.0)
    start = path[:-1]
    end = path[1:]
    start_x = np.asarray([point.x for point in start], dtype=np.float64)
    start_z = np.asarray([point.z for point in start], dtype=np.float64)
    end_x = np.asarray([point.x for point in end], dtype=np.float64)
    end_z = np.asarray([point.z for point in end], dtype=np.float64)
    station_x = start_x[segment_ids] + (end_x[segment_ids] - start_x[segment_ids]) * local_t
    station_z = start_z[segment_ids] + (end_z[segment_ids] - start_z[segment_ids]) * local_t
    return distances / total, station_x, station_z


def polyline_distance_and_progress(
    x: np.ndarray, z: np.ndarray, path: tuple[Point, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """返回网格到折线的最近距离，以及折线上的归一化进度。"""

    best_distance = np.full_like(x, np.inf, dtype=np.float64)
    best_progress = np.zeros_like(x, dtype=np.float64)
    total_length = sum(
        float(np.hypot(path[index + 1].x - point.x, path[index + 1].z - point.z))
        for index, point in enumerate(path[:-1])
    )
    total_length = max(total_length, 1.0e-9)
    travelled = 0.0

    for start, end in zip(path, path[1:]):
        vx = end.x - start.x
        vz = end.z - start.z
        length_sq = max(vx * vx + vz * vz, 1.0e-12)
        raw_t = ((x - start.x) * vx + (z - start.z) * vz) / length_sq
        t = np.clip(raw_t, 0.0, 1.0)
        nearest_x = start.x + t * vx
        nearest_z = start.z + t * vz
        distance = np.hypot(x - nearest_x, z - nearest_z)
        segment_length = float(np.sqrt(length_sq))
        progress = (travelled + t * segment_length) / total_length
        better = distance < best_distance
        best_distance = np.where(better, distance, best_distance)
        best_progress = np.where(better, progress, best_progress)
        travelled += segment_length

    return best_distance, best_progress


def sample_regular_grid(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    sample_x: np.ndarray,
    sample_z: np.ndarray,
) -> np.ndarray:
    """在规则世界网格上做双线性采样，越界坐标钳制到边界。

    河流、峡谷和其它沿路径采样的地貌都使用同一份规则网格合同，避免
    每个生成器各自实现一套边界处理逻辑。
    """

    height, width = values.shape
    cell_x = float(grid_x[0, 1] - grid_x[0, 0]) if width > 1 else 1.0
    cell_z = float(grid_z[1, 0] - grid_z[0, 0]) if height > 1 else 1.0
    origin_x = float(grid_x[0, 0])
    origin_z = float(grid_z[0, 0])
    gx = np.clip((sample_x - origin_x) / cell_x, 0.0, width - 1.0)
    gz = np.clip((sample_z - origin_z) / cell_z, 0.0, height - 1.0)
    x0 = np.floor(gx).astype(np.int64)
    z0 = np.floor(gz).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    z1 = np.minimum(z0 + 1, height - 1)
    tx = gx - x0
    tz = gz - z0
    top = values[z0, x0] * (1.0 - tx) + values[z0, x1] * tx
    bottom = values[z1, x0] * (1.0 - tx) + values[z1, x1] * tx
    return top * (1.0 - tz) + bottom * tz


__all__ = ["polyline_distance_and_progress", "polyline_stations", "sample_regular_grid"]
