"""山脉横截面和有限支持的连续剖面。

本模块只负责把距离场转换为无量纲形状，不负责高度标定、噪声采样或地形
写回。这样可以分别测试“冠部形状”和“绝对海拔目标”，避免同一个 profile
在多个阶段重复应用。
"""

from __future__ import annotations

import numpy as np

from ..terrain_config import MountainRange


def _smoothstep(value: np.ndarray) -> np.ndarray:
    """把 [0, 1] 输入转换为两端斜率为零的平滑权重。"""

    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def build_mountain_profile(
    normalized_distance: np.ndarray,
    local_width: np.ndarray,
    mountain: MountainRange,
) -> np.ndarray:
    """生成从冠部到山脚的单次高度剖面。

    主坡使用 ``(1-d)^p`` 保留陡峭山壁；冠部在配置宽度内平滑混入
    ``1-d^p``，使中心附近的导数趋近于零。它不会把一段区域强行设为常数，
    因此不会产生人工平台，但相邻体素能共享连续的高程过渡。
    """

    distance = np.clip(np.asarray(normalized_distance, dtype=np.float64), 0.0, 1.0)
    flank = np.power(1.0 - distance, mountain.slope_power)
    if mountain.crest_width == 0.0:
        return flank

    # 局部宽度会受 width noise 影响，冠部比例也必须随之变化，保证分块
    # 生成时同一世界坐标得到同一剖面，而不会在 tile 边缘改变冠部宽度。
    crest_ratio = np.clip(mountain.crest_width / np.maximum(local_width, 1.0e-12), 0.0, 1.0)
    crest_progress = distance / np.maximum(crest_ratio, 1.0e-12)
    crest_weight = 1.0 - _smoothstep(crest_progress)
    rounded_crest = 1.0 - np.power(distance, mountain.slope_power)
    return np.clip(
        rounded_crest * crest_weight + flank * (1.0 - crest_weight),
        0.0,
        1.0,
    )


def build_edge_support(
    normalized_distance: np.ndarray,
    edge_blend: float,
) -> np.ndarray:
    """只在山脚边缘把目标高度连续混回现有地形。"""

    distance = np.clip(np.asarray(normalized_distance, dtype=np.float64), 0.0, 1.0)
    # d <= 1-edge_blend 时完全采用山体目标；最后一段才逐渐回到地形。
    return _smoothstep((1.0 - distance) / edge_blend)


__all__ = ["build_edge_support", "build_mountain_profile"]
