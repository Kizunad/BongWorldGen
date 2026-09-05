"""山脉两侧支脊、凹谷和岩石褶皱细节。"""

from __future__ import annotations

import numpy as np

from ..terrain_config import MountainRange
from .ridged import sample_ridged_multifractal


def carve_flank_relief(
    profile: np.ndarray,
    normalized_distance: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    """在两侧山坡切出 seed 驱动的支脊与凹谷。

    粗糙场在主脊和山脚的影响都归零：主脊仍由一维高度场决定峰与鞍部，
    山脚仍严格收口到原地形；中坡的 ridge noise 高值保留为支脊，低值
    切成山谷。这样不会重新产生恒高中心线或山脚断壁。
    """

    if mountain.flank_carving == 0.0:
        return profile
    roughness = sample_ridged_multifractal(
        x,
        z,
        mountain.flank_ridges,
        seed + 31_109,
    )
    # 4d(1-d) 在半坡为 1，在主脊和山脚均为 0，保证两端合同不变。
    flank_weight = 4.0 * normalized_distance * (1.0 - normalized_distance)
    retained = 1.0 - mountain.flank_carving * (1.0 - roughness) * flank_weight
    return profile * np.clip(retained, 0.0, 1.0)


def rock_fold_delta(
    profile: np.ndarray,
    normalized_distance: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mountain: MountainRange,
    seed: int,
) -> np.ndarray:
    """只在上部山坡叠加细尺度岩石褶皱，并保持峰顶与山脚标高不变。"""

    if mountain.rock_fold_height == 0.0:
        return np.zeros(profile.shape, dtype=np.float64)
    folds = sample_ridged_multifractal(x, z, mountain.rock_folds, seed + 31_193)
    centered_folds = (folds - 0.5) * 2.0
    highland_gate = np.clip((profile - 0.24) / 0.56, 0.0, 1.0)
    highland_gate = highland_gate * highland_gate * (3.0 - 2.0 * highland_gate)
    flank_gate = 4.0 * normalized_distance * (1.0 - normalized_distance)
    return centered_folds * highland_gate * flank_gate * mountain.rock_fold_height


__all__ = ["carve_flank_relief", "rock_fold_delta"]
