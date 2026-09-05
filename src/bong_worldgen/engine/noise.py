"""小型、低依赖、确定性的噪声后端。

实现保持自包含：先用带平滑插值的哈希值网格，再构造 ridge 和 fractal
变体。
这样可以让引擎跨机器复现，同时不依赖 fork/ 下没有授权的参考仓库。

下面的 3D 函数参考 FastNoiseLite 的公开 API 和使用模式（seeded 3D noise
以及 domain warping），没有复制其源代码：
https://github.com/Auburn/FastNoiseLite
"""

from __future__ import annotations

import numpy as np

from .terrain_config import NoiseLayer


_MANTISSA_MASK = np.uint64((1 << 53) - 1)


def _hash_grid(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """为整数网格坐标返回 [-1, 1] 范围内的确定性值。"""

    x = np.asarray(ix, dtype=np.uint64)
    z = np.asarray(iz, dtype=np.uint64)
    seed_value = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        value = x * np.uint64(0x9E3779B185EBCA87)
        value += z * np.uint64(0xC2B2AE3D27D4EB4F)
        value += seed_value * np.uint64(0x165667B19E3779F9)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    normalized = (value & _MANTISSA_MASK).astype(np.float64) / float(1 << 53)
    return normalized * 2.0 - 1.0


def _hash_grid_3d(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """为三维整数晶格返回 [-1, 1] 范围内的确定性值。"""

    x = np.asarray(ix, dtype=np.uint64)
    y = np.asarray(iy, dtype=np.uint64)
    z = np.asarray(iz, dtype=np.uint64)
    seed_value = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        value = x * np.uint64(0x9E3779B185EBCA87)
        value += y * np.uint64(0xC2B2AE3D27D4EB4F)
        value += z * np.uint64(0x165667B19E3779F9)
        value += seed_value * np.uint64(0xD6E8FEB86659FD93)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    normalized = (value & _MANTISSA_MASK).astype(np.float64) / float(1 << 53)
    return normalized * 2.0 - 1.0


def value_noise_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scale: float,
    seed: int,
) -> np.ndarray:
    """在可广播数组上采样平滑的确定性三维 value noise。"""

    if scale <= 0:
        raise ValueError("noise scale must be positive")
    sx = np.asarray(x, dtype=np.float64) / scale
    sy = np.asarray(y, dtype=np.float64) / scale
    sz = np.asarray(z, dtype=np.float64) / scale
    ix = np.floor(sx).astype(np.int64)
    iy = np.floor(sy).astype(np.int64)
    iz = np.floor(sz).astype(np.int64)
    fx = sx - ix
    fy = sy - iy
    fz = sz - iz
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)

    n000 = _hash_grid_3d(ix, iy, iz, seed)
    n100 = _hash_grid_3d(ix + 1, iy, iz, seed)
    n010 = _hash_grid_3d(ix, iy + 1, iz, seed)
    n110 = _hash_grid_3d(ix + 1, iy + 1, iz, seed)
    n001 = _hash_grid_3d(ix, iy, iz + 1, seed)
    n101 = _hash_grid_3d(ix + 1, iy, iz + 1, seed)
    n011 = _hash_grid_3d(ix, iy + 1, iz + 1, seed)
    n111 = _hash_grid_3d(ix + 1, iy + 1, iz + 1, seed)
    x00 = n000 + (n100 - n000) * fx
    x10 = n010 + (n110 - n010) * fx
    x01 = n001 + (n101 - n001) * fx
    x11 = n011 + (n111 - n011) * fx
    y0 = x00 + (x10 - x00) * fy
    y1 = x01 + (x11 - x01) * fy
    return y0 + (y1 - y0) * fz


def sample_noise_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: NoiseLayer,
    seed: int,
) -> np.ndarray:
    """计算配置好的三维噪声层，并支持 domain warp。"""

    if layer.kind == "value":
        return value_noise_3d(x, y, z, scale=layer.scale, seed=seed + layer.seed_offset)
    if layer.kind == "ridge":
        base = value_noise_3d(x, y, z, scale=layer.scale, seed=seed + layer.seed_offset)
        return 1.0 - np.abs(base)
    if layer.kind == "warp":
        offset_x = value_noise_3d(
            x, y, z, scale=layer.warp_scale, seed=seed + layer.seed_offset + 17
        )
        offset_y = value_noise_3d(
            x, y, z, scale=layer.warp_scale, seed=seed + layer.seed_offset + 23
        )
        offset_z = value_noise_3d(
            x, y, z, scale=layer.warp_scale, seed=seed + layer.seed_offset + 31
        )
        warped = NoiseLayer(
            kind="fbm",
            scale=layer.scale,
            amplitude=layer.amplitude,
            octaves=layer.octaves,
            lacunarity=layer.lacunarity,
            gain=layer.gain,
            seed_offset=layer.seed_offset,
        )
        return sample_noise_3d(
            x + offset_x * layer.warp_strength,
            y + offset_y * layer.warp_strength,
            z + offset_z * layer.warp_strength,
            warped,
            seed,
        )
    if layer.kind != "fbm":
        raise ValueError(f"unknown noise kind {layer.kind!r}")

    output = np.zeros_like(np.broadcast_arrays(x, y, z)[0], dtype=np.float64)
    amplitude = 1.0
    normalization = 0.0
    frequency = 1.0
    for octave in range(layer.octaves):
        output += amplitude * value_noise_3d(
            np.asarray(x) * frequency,
            np.asarray(y) * frequency,
            np.asarray(z) * frequency,
            scale=layer.scale,
            seed=seed + layer.seed_offset + octave * 1013,
        )
        normalization += amplitude
        amplitude *= layer.gain
        frequency *= layer.lacunarity
    return output / max(normalization, 1.0e-12)


def value_noise(x: np.ndarray, z: np.ndarray, *, scale: float, seed: int) -> np.ndarray:
    """在世界坐标上采样平滑的确定性 value noise。"""

    if scale <= 0:
        raise ValueError("noise scale must be positive")
    sx = np.asarray(x, dtype=np.float64) / scale
    sz = np.asarray(z, dtype=np.float64) / scale
    ix = np.floor(sx).astype(np.int64)
    iz = np.floor(sz).astype(np.int64)
    fx = sx - ix
    fz = sz - iz
    fx = fx * fx * (3.0 - 2.0 * fx)
    fz = fz * fz * (3.0 - 2.0 * fz)

    n00 = _hash_grid(ix, iz, seed)
    n10 = _hash_grid(ix + 1, iz, seed)
    n01 = _hash_grid(ix, iz + 1, seed)
    n11 = _hash_grid(ix + 1, iz + 1, seed)
    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fz


def sample_noise(x: np.ndarray, z: np.ndarray, layer: NoiseLayer, seed: int) -> np.ndarray:
    """计算一个配置好的噪声层，结果保持在约定的 [-1, 1] 范围。"""

    if layer.kind == "value":
        return value_noise(x, z, scale=layer.scale, seed=seed + layer.seed_offset)
    if layer.kind == "ridge":
        base = value_noise(x, z, scale=layer.scale, seed=seed + layer.seed_offset)
        return 1.0 - np.abs(base)
    if layer.kind == "warp":
        offset_x = value_noise(
            x,
            z,
            scale=layer.warp_scale,
            seed=seed + layer.seed_offset + 17,
        )
        offset_z = value_noise(
            x,
            z,
            scale=layer.warp_scale,
            seed=seed + layer.seed_offset + 31,
        )
        warped_x = x + offset_x * layer.warp_strength
        warped_z = z + offset_z * layer.warp_strength
        base_layer = NoiseLayer(
            kind="fbm",
            scale=layer.scale,
            amplitude=layer.amplitude,
            octaves=layer.octaves,
            lacunarity=layer.lacunarity,
            gain=layer.gain,
            seed_offset=layer.seed_offset,
        )
        return sample_noise(warped_x, warped_z, base_layer, seed)
    if layer.kind != "fbm":
        raise ValueError(f"unknown noise kind {layer.kind!r}")

    output = np.zeros_like(np.asarray(x, dtype=np.float64))
    amplitude = 1.0
    normalization = 0.0
    frequency = 1.0
    for octave in range(layer.octaves):
        output += amplitude * value_noise(
            x * frequency,
            z * frequency,
            scale=layer.scale,
            seed=seed + layer.seed_offset + octave * 1013,
        )
        normalization += amplitude
        amplitude *= layer.gain
        frequency *= layer.lacunarity
    return output / max(normalization, 1.0e-12)
