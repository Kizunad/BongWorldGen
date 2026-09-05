"""把 Mountain Spine 距离场转换为山体高度。

与无限延伸的 Gaussian uplift 不同，这里的山体在 ``distance == width``
处严格回到山脚；Ridged Multifractal 分别负责脊柱峰鞍、两侧支脊和上部
岩石褶皱，因此不会把所有频率混成一层均匀粗糙度。
"""

from __future__ import annotations

import numpy as np

from ..terrain_config import MountainRange
from .details import carve_flank_relief, rock_fold_delta
from .distance import mountain_distance_field
from .elevation import build_absolute_target, build_relative_target
from .peaks import apply_anisotropic_peaks
from .profile import build_mountain_profile
from .spine import spine_height_reduction


def _apply_mountain_stack(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算山脉抬升后的地形，并保留正向构造抬升场。

    ``uplift`` 是山峰/山脊对基准地形施加的净正向高程变化；谷地阶段会
    读取它来决定哪些坡面具有更强的构造驱动。山脉内部的 valley_depth
    属于局部几何修饰，不会被误报为负的“抬升”。
    """

    initial = np.asarray(terrain, dtype=np.float64)
    output = initial.copy()
    for index, mountain in enumerate(mountains):
        output = _apply_one_mountain(
            output,
            x,
            z,
            mountain,
            seed + index * 1_003_049,
        )
    uplift = np.maximum(output - initial, 0.0)
    return output, np.ascontiguousarray(uplift, dtype=np.float64)


def _apply_one_mountain(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    field = mountain_distance_field(x, z, mountain, seed)
    profile = build_mountain_profile(
        field.normalized_distance,
        field.local_width,
        mountain,
    )
    profile = carve_flank_relief(
        profile,
        field.normalized_distance,
        x,
        z,
        mountain,
        seed,
    )
    fold_delta = rock_fold_delta(
        profile,
        field.normalized_distance,
        x,
        z,
        mountain,
        seed,
    )
    reduction = spine_height_reduction(field.progress, mountain, seed)
    ridge_height: np.ndarray | None = None
    if mountain.peaks.enabled and mountain.peaks.count and mountain.peaks.amplitude:
        # 先得到原始脊高，再让峰值包络依附这条脊；峰值模块只改变脊高，
        # 山脚收口和绝对/相对高度标定仍由 elevation 模块负责。
        local_relief = (
            mountain.height - reduction
            if mountain.base_elevation is None
            else mountain.summit_elevation - mountain.base_elevation - reduction
        )
        ridge_height = np.maximum(profile * np.maximum(local_relief, 0.0) + fold_delta, 0.0)
        ridge_height = apply_anisotropic_peaks(
            ridge_height,
            field.progress,
            field.distance,
            field.local_width,
            mountain,
            seed,
        )

    if mountain.base_elevation is None:
        return build_relative_target(
            terrain,
            profile,
            reduction,
            fold_delta,
            mountain,
            ridge_height=ridge_height,
        )

    return build_absolute_target(
        terrain,
        profile,
        field.normalized_distance,
        reduction,
        fold_delta,
        mountain,
        ridge_height=ridge_height,
    )


def apply_mountains(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
) -> np.ndarray:
    """依次叠加所有 Mountain Spine；每条脊柱使用独立 seed 域。"""

    output, _ = apply_mountains_with_uplift(terrain, x, z, mountains, seed)
    return output


def apply_mountains_with_uplift(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountains: tuple[MountainRange, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(抬升后地形, 正向构造抬升场)``。

    该接口让后续 Stream Power 阶段消费真实的 Uplift 场，而不是重新从
    山脉配置猜测高程。输出只依赖世界坐标和 seed，分块采样与整图一致。
    """

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("mountain fields must share a shape")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("mountain fields must contain finite values")
    return _apply_mountain_stack(terrain, x, z, mountains, seed)


__all__ = ["apply_mountains", "apply_mountains_with_uplift"]
