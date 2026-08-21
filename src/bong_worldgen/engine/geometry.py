"""Vectorized geometry helpers for controllable terrain features."""

from __future__ import annotations

import numpy as np

from .models import Point


def polyline_distance_and_progress(
    x: np.ndarray, z: np.ndarray, path: tuple[Point, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest distance and normalized progress along a polyline."""

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
