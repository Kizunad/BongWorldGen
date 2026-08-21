"""Small, dependency-light deterministic noise backend.

The implementation is intentionally self-contained. It uses a hashed value
grid with smooth interpolation, then builds ridged and fractal variants on top.
That keeps the engine reproducible across machines without depending on the
unlicensed reference repository under fork/.
"""

from __future__ import annotations

import numpy as np

from .models import NoiseLayer


_MANTISSA_MASK = np.uint64((1 << 53) - 1)


def _hash_grid(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Return deterministic values in [-1, 1] for integer grid coordinates."""

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


def value_noise(x: np.ndarray, z: np.ndarray, *, scale: float, seed: int) -> np.ndarray:
    """Sample smooth value noise at world coordinates."""

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
    """Evaluate one configured noise layer in the canonical [-1, 1] range."""

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
