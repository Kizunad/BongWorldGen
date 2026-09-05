"""沿主山脊选择峰点，并生成各向异性的峰值包络。

峰点不是独立放置的 Gaussian 山包：候选点来自同一条 Mountain Spine，
评分同时读取 Ridged Multifractal、折线曲率和脊柱折点，再按弧长做 Poisson
式最小间距筛选。峰核沿山脊较宽、横跨山脊较窄，因此峰值会成为山脊的一部分。

噪声采样的分形组织参考 FastNoiseLite 公开实现，未复制其源码：
https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain_config import MountainPeakSettings, MountainRange, Point
from .ridged import sample_ridged_multifractal


@dataclass(frozen=True)
class PeakAnchor:
    """主脊上的一个可查询峰点。``progress`` 是沿脊弧长的归一化位置。"""

    progress: float
    score: float
    x: float
    z: float


def _path_geometry(
    path: tuple[Point, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points_x = np.asarray([point.x for point in path], dtype=np.float64)
    points_z = np.asarray([point.z for point in path], dtype=np.float64)
    segment_lengths = np.hypot(np.diff(points_x), np.diff(points_z))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = max(float(cumulative[-1]), 1.0e-9)
    return points_x, points_z, cumulative, total


def _point_at_progress(
    progress: float,
    points_x: np.ndarray,
    points_z: np.ndarray,
    cumulative: np.ndarray,
    total: float,
) -> tuple[float, float]:
    distance = np.clip(progress, 0.0, 1.0) * total
    segment = int(np.searchsorted(cumulative[1:], distance, side="right"))
    segment = min(segment, len(points_x) - 2)
    length = max(float(cumulative[segment + 1] - cumulative[segment]), 1.0e-9)
    local = (distance - cumulative[segment]) / length
    x = points_x[segment] + (points_x[segment + 1] - points_x[segment]) * local
    z = points_z[segment] + (points_z[segment + 1] - points_z[segment]) * local
    return float(x), float(z)


def _curvature_at_progress(
    progress: float,
    points_x: np.ndarray,
    points_z: np.ndarray,
    cumulative: np.ndarray,
    total: float,
) -> float:
    """用相邻切向量夹角估算主脊曲率，直线为零、回折趋近一。"""

    delta = min(0.12, max(20.0 / total, 0.004))
    before = _point_at_progress(
        max(progress - delta, 0.0), points_x, points_z, cumulative, total
    )
    center = _point_at_progress(progress, points_x, points_z, cumulative, total)
    after = _point_at_progress(
        min(progress + delta, 1.0), points_x, points_z, cumulative, total
    )
    first = np.asarray(center, dtype=np.float64) - before
    second = np.asarray(after, dtype=np.float64) - center
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1.0e-9 or second_norm <= 1.0e-9:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(0.5 * (1.0 - cosine), 0.0, 1.0))


def _candidate_progresses(
    settings: MountainPeakSettings,
    total: float,
    cumulative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回候选进度和折点/段中点的额外优先级。"""

    candidate_spacing = settings.candidate_spacing
    regular_distances = np.arange(0.0, total + candidate_spacing * 0.5, candidate_spacing)
    progresses: list[float] = []
    junction_bonus: list[float] = []
    for distance in regular_distances:
        progress = float(np.clip(distance / total, 0.0, 1.0))
        if 0.0 < progress < 1.0:
            progresses.append(progress)
            junction_bonus.append(0.0)

    # 折点是潜在的 ridge junction；段中点确保短折线段也有候选峰。
    vertex_progresses = cumulative[1:-1] / total
    for progress in vertex_progresses:
        progresses.append(float(progress))
        junction_bonus.append(1.0)
    segment_midpoints = 0.5 * (cumulative[:-1] + cumulative[1:]) / total
    for progress in segment_midpoints:
        if 0.0 < progress < 1.0:
            progresses.append(float(progress))
            junction_bonus.append(0.35)

    if not progresses:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # 同一位置可能同时是常规采样点、折点和段中点，只保留优先级最高的记录。
    merged: dict[int, tuple[float, float]] = {}
    for progress, bonus in zip(progresses, junction_bonus):
        key = int(round(progress * 1_000_000_000.0))
        existing = merged.get(key)
        if existing is None or bonus > existing[1]:
            merged[key] = (progress, bonus)
    values = sorted(merged.values())
    return (
        np.asarray([value[0] for value in values], dtype=np.float64),
        np.asarray([value[1] for value in values], dtype=np.float64),
    )


def select_peak_anchors(
    mountain: MountainRange,
    seed: int,
) -> tuple[PeakAnchor, ...]:
    """从主脊候选中按 seed 选择彼此分离的峰点。"""

    settings = mountain.peaks
    if not settings.enabled or settings.count == 0:
        return ()
    _, _, cumulative, total = _path_geometry(mountain.path)
    progresses, junction_bonus = _candidate_progresses(settings, total, cumulative)
    if progresses.size == 0:
        return ()
    points_x, points_z, _, _ = _path_geometry(mountain.path)
    positions = np.asarray(
        [_point_at_progress(value, points_x, points_z, cumulative, total) for value in progresses],
        dtype=np.float64,
    )
    ridge_scores = sample_ridged_multifractal(
        positions[:, 0],
        positions[:, 1],
        mountain.spine_ridges,
        seed + settings.seed_offset + 67_013,
    )
    curvatures = np.asarray(
        [
            _curvature_at_progress(value, points_x, points_z, cumulative, total)
            for value in progresses
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [settings.ridge_weight, settings.curvature_weight, settings.junction_weight],
        dtype=np.float64,
    )
    weights /= np.sum(weights)
    scores = np.clip(
        weights[0] * ridge_scores
        + weights[1] * curvatures
        + weights[2] * junction_bonus,
        0.0,
        1.0,
    )

    # 排序的微小随机扰动只用于相同分数的候选，主排序仍由地貌信号决定。
    rng_seed = (int(seed) + settings.seed_offset + 67_097) & ((1 << 63) - 1)
    tie_break = np.random.default_rng(rng_seed).random(progresses.size) * 1.0e-9
    order = np.argsort(-(scores + tie_break), kind="stable")
    selected: list[int] = []
    for index in order:
        if all(
            abs(float(progresses[index] - progresses[other])) * total
            >= settings.minimum_spacing
            for other in selected
        ):
            selected.append(int(index))
            if len(selected) >= settings.count:
                break
    selected.sort(key=lambda index: float(progresses[index]))
    return tuple(
        PeakAnchor(
            progress=float(progresses[index]),
            score=float(scores[index]),
            x=float(positions[index, 0]),
            z=float(positions[index, 1]),
        )
        for index in selected
    )


def _smooth_max(
    first: np.ndarray,
    second: np.ndarray,
    smoothness: np.ndarray,
) -> np.ndarray:
    """连续的 soft maximum，smoothness 为零时退化为普通最大值。"""

    amount = np.maximum(np.asarray(smoothness, dtype=np.float64), 0.0)
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return 0.5 * (
        np.asarray(first, dtype=np.float64)
        + np.asarray(second, dtype=np.float64)
        + np.sqrt(np.square(difference) + np.square(amount))
        - amount
    )


def apply_anisotropic_peaks(
    ridge_height: np.ndarray,
    progress: np.ndarray,
    distance: np.ndarray,
    local_width: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    """把脊柱峰值包络以 smooth-max 方式合成到现有山脊高度。

    ``distance`` 是横跨主脊的距离，``progress`` 是沿主脊的距离；二者使用
    不同尺度进入指数核，故峰沿脊延展、横向收窄。峰值只在山体有限支持域
    内生效，不会在山脚外制造孤立的圆包。
    """

    settings = mountain.peaks
    source = np.asarray(ridge_height, dtype=np.float64)
    if not settings.enabled or settings.count == 0 or settings.amplitude == 0.0:
        return source
    if not (source.shape == progress.shape == distance.shape == local_width.shape):
        raise ValueError("mountain peak fields must share a shape")
    if not all(np.isfinite(field).all() for field in (source, progress, distance, local_width)):
        raise ValueError("mountain peak fields must contain finite values")

    anchors = select_peak_anchors(mountain, seed)
    if not anchors:
        return source
    _, _, _, path_length = _path_geometry(mountain.path)
    normalized_distance = distance / np.maximum(local_width, 1.0e-9)
    inside = normalized_distance <= 1.0
    peak_delta = np.zeros_like(source, dtype=np.float64)
    peak_support = np.zeros_like(source, dtype=np.float64)
    for anchor in anchors:
        along = (progress - anchor.progress) * path_length
        across_width = settings.across_width * np.maximum(
            local_width / max(mountain.width, 1.0e-9),
            0.08,
        )
        kernel = np.exp(
            -0.5
            * (
                np.square(along / settings.along_width)
                + np.square(distance / np.maximum(across_width, 1.0e-9))
            )
        )
        kernel = np.where(inside, kernel, 0.0)
        # 峰值是已有脊高上的局部增量，而不是“参考峰高乘椭圆核”。
        # 后者会在山脊旁边独立抬出一块椭圆体，产生明显的切断轮廓。
        anchor_delta = settings.amplitude * (0.55 + 0.45 * anchor.score)
        peak_delta = np.maximum(peak_delta, anchor_delta * kernel)
        peak_support = np.maximum(peak_support, kernel)

    smoothness = settings.smoothness * peak_support
    return np.ascontiguousarray(_smooth_max(source, source + peak_delta, smoothness))


__all__ = ["PeakAnchor", "apply_anisotropic_peaks", "select_peak_anchors"]
