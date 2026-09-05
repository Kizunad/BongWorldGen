"""把无量纲山体剖面标定为相对或绝对高程。"""

from __future__ import annotations

import numpy as np

from ..terrain_config import MountainRange
from .profile import build_edge_support


def build_relative_target(
    terrain: np.ndarray,
    profile: np.ndarray,
    reduction: np.ndarray,
    fold_delta: np.ndarray,
    mountain: MountainRange,
    ridge_height: np.ndarray | None = None,
) -> np.ndarray:
    """计算只提供相对抬升量的山脉目标。

    ``ridge_height`` 由峰值阶段提供时，使用已经完成 smooth-max 的脊高，
    避免在高度标定层再次把峰值当成独立地貌叠加。
    """

    local_relief = np.maximum(mountain.height - reduction, 0.0)
    if ridge_height is None:
        uplift = np.maximum(profile * local_relief + fold_delta, 0.0)
    else:
        uplift = np.maximum(np.asarray(ridge_height, dtype=np.float64), 0.0)
    if mountain.valley_depth:
        uplift -= np.power(profile, 2.0) * mountain.valley_depth
    return terrain + uplift


def build_absolute_target(
    terrain: np.ndarray,
    profile: np.ndarray,
    normalized_distance: np.ndarray,
    reduction: np.ndarray,
    fold_delta: np.ndarray,
    mountain: MountainRange,
    ridge_height: np.ndarray | None = None,
) -> np.ndarray:
    """计算绝对峰高目标，并只在边缘混回原地形。

    ``profile`` 只参与一次 base 到 summit 的高程插值；边缘混合使用独立的
    ``edge_support``。旧公式在这里再次乘 profile，导致峰顶附近有效高度
    近似按 profile 的平方衰减，正是单格尖顶的主要来源。
    """

    local_summit = mountain.summit_elevation - reduction
    if ridge_height is None:
        nominal_target = mountain.base_elevation + profile * (
            local_summit - mountain.base_elevation
        )
    else:
        # 峰值的幅度是“脊线上方的附加高度”，因此 summit_elevation 作为
        # 基准脊高保留，peak settings.amplitude 决定峰顶可额外抬升多少。
        nominal_target = mountain.base_elevation + np.maximum(
            np.asarray(ridge_height, dtype=np.float64),
            0.0,
        )
    edge_support = build_edge_support(normalized_distance, mountain.edge_blend)
    target = terrain + edge_support * np.maximum(nominal_target - terrain, 0.0)
    if ridge_height is None:
        target = np.minimum(target + fold_delta, local_summit)
    if mountain.valley_depth:
        target -= np.power(profile, 2.0) * mountain.valley_depth
    return np.maximum(terrain, target)


__all__ = ["build_absolute_target", "build_relative_target"]
