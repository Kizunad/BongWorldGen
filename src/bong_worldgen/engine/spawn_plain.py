"""从自然地形中选择并二次整形出生平原。

出生平原不是一块与世界脱节的硬编码平台。这里先对完整宏观地形的候选窗口
计算局部坡度、局部起伏和海拔安全度，再把评分最高的连续区域做轻度整平。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .noise import sample_noise
from .randomness import stable_text_seed
from .terrain_config import SpawnPlainSettings


@dataclass(frozen=True)
class SpawnPlainSelection:
    """一次 seed 下选中的出生平原中心和目标高程。"""

    center_x: float
    center_z: float
    target_height: float
    score: float


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    """用 NumPy 一维卷积组合出可复现的二维局部统计。"""

    if radius <= 0:
        return values.astype(np.float64, copy=False)
    kernel = np.full(2 * radius + 1, 1.0 / (2 * radius + 1), dtype=np.float64)
    horizontal = np.apply_along_axis(
        lambda row: np.convolve(
            np.pad(row, (radius, radius), mode="edge"), kernel, mode="valid"
        ),
        1,
        values,
    )
    return np.apply_along_axis(
        lambda column: np.convolve(
            np.pad(column, (radius, radius), mode="edge"), kernel, mode="valid"
        ),
        0,
        horizontal,
    )


def _candidate_grid(settings: SpawnPlainSettings) -> tuple[np.ndarray, np.ndarray]:
    count_x = max(
        3, int(np.ceil(settings.search_extent_x * 2.0 / settings.search_resolution)) + 1
    )
    count_z = max(
        3, int(np.ceil(settings.search_extent_z * 2.0 / settings.search_resolution)) + 1
    )
    xs = np.linspace(
        settings.center.x - settings.search_extent_x,
        settings.center.x + settings.search_extent_x,
        count_x,
        dtype=np.float64,
    )
    zs = np.linspace(
        settings.center.z - settings.search_extent_z,
        settings.center.z + settings.search_extent_z,
        count_z,
        dtype=np.float64,
    )
    return np.meshgrid(xs, zs, indexing="xy")


def select_spawn_plain_anchor(
    settings: SpawnPlainSettings,
    seed: int,
    terrain_sampler: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    sea_level: float,
) -> SpawnPlainSelection:
    """在自然地形采样窗口内选择最适合的出生平原中心。"""

    if not settings.enabled:
        return SpawnPlainSelection(settings.center.x, settings.center.z, sea_level, 0.0)

    x, z = _candidate_grid(settings)
    terrain = np.asarray(terrain_sampler(x, z), dtype=np.float64)
    if terrain.shape != x.shape or not np.isfinite(terrain).all():
        raise ValueError("spawn plain terrain sampler must return a finite matching field")

    spacing = settings.search_resolution
    gradient_z, gradient_x = np.gradient(terrain, spacing, spacing)
    local_slope = np.hypot(gradient_x, gradient_z)
    radius_cells = max(1, int(round(settings.candidate_window_radius / spacing)))
    mean_slope = _box_blur(local_slope, radius_cells)
    mean_height = _box_blur(terrain, radius_cells)
    mean_square = _box_blur(terrain * terrain, radius_cells)
    local_relief = np.sqrt(np.maximum(mean_square - mean_height * mean_height, 0.0)) * 2.0

    candidate_margin_x = settings.plain_radius_x + settings.edge_blend
    candidate_margin_z = settings.plain_radius_z + settings.edge_blend
    inside = (
        (np.abs(x - settings.center.x) <= settings.search_extent_x - candidate_margin_x)
        & (np.abs(z - settings.center.z) <= settings.search_extent_z - candidate_margin_z)
    )
    land = mean_height >= sea_level + settings.minimum_elevation_above_sea
    slope_score = 1.0 - np.clip(mean_slope / settings.maximum_mean_slope, 0.0, 1.0)
    relief_score = 1.0 - np.clip(local_relief / settings.maximum_local_relief, 0.0, 1.0)
    elevation_score = np.clip(
        (mean_height - sea_level) / max(settings.maximum_local_relief * 4.0, 1.0),
        0.0,
        1.0,
    )
    score = 0.55 * slope_score + 0.35 * relief_score + 0.10 * elevation_score
    score = np.where(inside & land, score, -np.inf)

    if not np.isfinite(score).any():
        index = (score.shape[0] // 2, score.shape[1] // 2)
        score_value = 0.0
    else:
        # argmax 按行优先选择，保证相同分数时不依赖进程随机状态。
        index = np.unravel_index(int(np.argmax(score)), score.shape)
        score_value = float(score[index])
    return SpawnPlainSelection(
        center_x=float(x[index]),
        center_z=float(z[index]),
        target_height=float(mean_height[index]),
        score=score_value,
    )


def apply_spawn_plain(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: SpawnPlainSettings | None,
    selection: SpawnPlainSelection,
    seed: int,
) -> np.ndarray:
    """把选中的自然平原做带边缘渐变的轻度二次整形。"""

    if settings is None or not settings.enabled:
        return np.asarray(terrain, dtype=np.float64).copy()
    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("spawn plain fields must share a shape")

    dx = (x - selection.center_x) / settings.plain_radius_x
    dz = (z - selection.center_z) / settings.plain_radius_z
    radial = np.hypot(dx, dz)
    edge_start = max(
        0.0,
        1.0 - settings.edge_blend / min(settings.plain_radius_x, settings.plain_radius_z),
    )
    weight = 1.0 - _smoothstep(
        (radial - edge_start) / max(1.0 - edge_start, 1.0e-9)
    )
    weight = np.clip(weight, 0.0, 1.0)

    noise_seed = seed + stable_text_seed("spawn_plain")
    variation = sample_noise(x, z, settings.micro_noise, noise_seed) * settings.natural_variation
    target = selection.target_height + variation
    blend = weight * settings.smoothing_strength
    output = np.asarray(terrain, dtype=np.float64).copy()
    return output + (target - output) * blend


__all__ = ["SpawnPlainSelection", "apply_spawn_plain", "select_spawn_plain_anchor"]
