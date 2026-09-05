"""由 seed 驱动的连续崎岖地形场。

这里不把世界切成预设的平原、山地和悬崖区域，而是把多尺度噪声、域扭曲、
山脊场和陡峭过渡叠加成一个连续高度场。地貌分类在生成完成后进行，不反向
规定地形必须长成什么样。

噪声和域扭曲的组织方式参考 FastNoiseLite 的公开用法，没有复制其源码：
https://github.com/Auburn/FastNoiseLite
相关的三维 SDF/悬挑表达仍由 ``solid_spans`` 负责；本模块只生成地表高度，
因此不会伪造超过高度场表达能力的倒挂几何。
"""

from __future__ import annotations

import numpy as np

from .noise import sample_noise
from .terrain_config import NaturalRelief


def _unit_interval(values: np.ndarray) -> np.ndarray:
    """把约定的 [-1, 1] 噪声映射到 [0, 1]。"""

    return np.clip((values + 1.0) * 0.5, 0.0, 1.0)


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """返回连续的平滑门控，避免地貌边缘出现人工直线。"""

    if edge1 <= edge0:
        raise ValueError("smoothstep edges must be ordered")
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_natural_relief(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    config: NaturalRelief,
    seed: int,
) -> np.ndarray:
    """把多尺度随机起伏叠加到已有连续高度场。

    生成顺序是：低频宏观起伏 → 域扭曲山脊 → 局部细节 → 随机陡壁。
    陡壁使用 ``tanh`` 的快速连续过渡，不设置坡度上限；最终方块化时自然
    可以得到多格落差。所有坐标采样均来自世界坐标和 seed，因此分块生成
    不会因为当前 tile 的边界改变地貌。
    """

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("relief fields must share a shape")
    if not np.isfinite(terrain).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("relief inputs must contain finite values")

    warp_layer = config.warp_noise
    warp_x = sample_noise(x, z, warp_layer, seed + 17_003)
    warp_z = sample_noise(x, z, warp_layer, seed + 17_037)
    warped_x = x + warp_x * config.warp_strength
    warped_z = z + warp_z * config.warp_strength

    macro = sample_noise(warped_x, warped_z, config.macro_noise, seed + 17_071)
    macro_unit = _unit_interval(macro)

    # 山脊不是沿固定方向画线，而是由域扭曲后的 ridged 场自然形成。
    ridge_source = sample_noise(warped_x, warped_z, config.ridge_noise, seed + 17_109)
    ridge_unit = _unit_interval(1.0 - np.abs(ridge_source))
    ridge_shape = np.power(np.clip(ridge_unit, 0.0, 1.0), config.ridge_power)
    ridge_gate = _smoothstep(macro_unit, 0.20, 0.72)

    detail = sample_noise(
        warped_x,
        warped_z,
        config.detail_noise,
        seed + 17_149,
    )
    detail_gate = 0.28 + 0.72 * ridge_gate

    output = np.asarray(terrain, dtype=np.float64).copy()
    output += macro * config.macro_amplitude
    output += ridge_shape * ridge_gate * config.ridge_amplitude
    output += detail * detail_gate * config.detail_amplitude

    # 在随机山脊边缘制造陡峭但不规则的岩壁。它不是分类区域，也没有固定
    # 数量；阈值和形状都由 seed 噪声决定，只有局部高起伏处更容易出现。
    cliff_source = sample_noise(
        warped_x,
        warped_z,
        config.cliff_noise,
        seed + 17_191,
    )
    cliff_gate = _smoothstep(ridge_unit, 0.42, 0.78) * _smoothstep(
        macro_unit, 0.38, 0.82
    )
    cliff_edge = np.tanh(
        (cliff_source - config.cliff_threshold) * config.cliff_sharpness
    )
    output += cliff_edge * cliff_gate * config.cliff_amplitude
    return output


__all__ = ["apply_natural_relief"]
