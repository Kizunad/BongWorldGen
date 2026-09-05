"""把大峡谷中心线烧录进连续高度场。"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import sample_regular_grid
from ..terrain_config import Canyon, Point
from ..noise import sample_noise
from .distance import warped_polyline_distance_and_progress
from .topology import generate_canyon_paths


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _path_stations(path: tuple[Point, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(
        [math.hypot(end.x - start.x, end.z - start.z) for start, end in zip(path, path[1:])],
        dtype=np.float64,
    )
    total = max(float(lengths.sum()), 1.0e-9)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    station_s = np.linspace(0.0, total, max(32, min(512, int(total / 8.0) + 1)))
    segment_ids = np.minimum(np.searchsorted(cumulative[1:], station_s, side="right"), len(path) - 2)
    start_s = cumulative[segment_ids]
    local_t = np.clip(station_s - start_s, 0.0, lengths[segment_ids]) / np.maximum(
        lengths[segment_ids], 1.0e-9
    )
    starts = path[:-1]
    ends = path[1:]
    sx = np.asarray([point.x for point in starts], dtype=np.float64)
    sz = np.asarray([point.z for point in starts], dtype=np.float64)
    ex = np.asarray([point.x for point in ends], dtype=np.float64)
    ez = np.asarray([point.z for point in ends], dtype=np.float64)
    station_x = sx[segment_ids] + (ex[segment_ids] - sx[segment_ids]) * local_t
    station_z = sz[segment_ids] + (ez[segment_ids] - sz[segment_ids]) * local_t
    return station_s / total, station_x, station_z


def _carve_path(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    canyon: Canyon,
    path: tuple[Point, ...],
    seed: int,
    path_index: int,
) -> np.ndarray:
    distance, progress = warped_polyline_distance_and_progress(x, z, path, canyon, seed)
    width_noise = sample_noise(
        x,
        z,
        canyon.wall_noise,
        seed + 1009 + path_index * 37_001,
    )
    width = canyon.width * (1.0 + canyon.width_variation * 0.5 * width_noise)
    floor_width = np.minimum(canyon.floor_width, width * 0.85)
    influence = width + canyon.bank_width
    influence_mask = distance <= influence

    floor_t, floor_x, floor_z = _path_stations(path)
    floor_profile = sample_regular_grid(terrain, x, z, floor_x, floor_z)
    floor_profile = np.convolve(
        np.pad(floor_profile, (2, 2), mode="edge"),
        np.full(5, 0.2, dtype=np.float64),
        mode="valid",
    )
    floor_elevation = np.interp(progress, floor_t, floor_profile) - canyon.depth

    floor_ratio = np.clip(floor_width / np.maximum(width, 1.0e-6), 0.08, 0.9)
    wall_t = np.clip((distance / np.maximum(width, 1.0e-6) - floor_ratio) / np.maximum(1.0 - floor_ratio, 1.0e-6), 0.0, 1.0)
    wall_profile = 1.0 - _smoothstep(wall_t)
    wall_profile = np.where(distance <= floor_width, 1.0, wall_profile)
    wall_profile = np.power(wall_profile, canyon.wall_power)

    depth_noise = sample_noise(
        x,
        z,
        canyon.wall_noise,
        seed + 2_017 + path_index * 37_001,
    )
    local_depth = canyon.depth * (1.0 + canyon.depth_variation * 0.5 * depth_noise)
    target_core = floor_elevation + local_depth * (1.0 - wall_profile)

    # 参考大峡谷分层设计：只在峡谷核心墙面量化少量台阶，保留噪声扰动，
    # 让高墙出现地层平台而不是一整面光滑斜坡。
    if canyon.terrace_count > 0 and canyon.terrace_strength > 0.0:
        relative = np.clip((target_core - floor_elevation) / np.maximum(local_depth, 1.0e-6), 0.0, 1.0)
        step = 1.0 / canyon.terrace_count
        terraced = np.floor(relative / step + 0.5) * step
        relative = relative * (1.0 - canyon.terrace_strength) + terraced * canyon.terrace_strength
        target_core = floor_elevation + relative * local_depth

    # 外缘只做 feathered carve；整个阶段是 carve-only，绝不会抬高原地形。
    feather = 1.0 - _smoothstep((distance - width) / max(canyon.bank_width, 1.0e-6))
    target_outer = terrain - local_depth * wall_profile * feather
    target = np.where(distance <= width, np.minimum(target_core, terrain), target_outer)
    return np.where(influence_mask, np.minimum(terrain, target), terrain)


def apply_canyons(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    canyons: tuple[Canyon, ...],
    seed: int,
) -> np.ndarray:
    """按配方顺序 carve 大峡谷，路径选择不受地表高度限制。"""

    output = np.asarray(terrain, dtype=np.float64)
    for canyon_index, canyon in enumerate(canyons):
        canyon_seed = seed + canyon_index * 91_337
        for path_index, path in enumerate(generate_canyon_paths(canyon, canyon_seed)):
            output = _carve_path(output, x, z, canyon, path, canyon_seed, path_index)
    return output


__all__ = ["apply_canyons"]
