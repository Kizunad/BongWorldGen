from __future__ import annotations

import numpy as np
import pytest

from bong_worldgen.engine import NaturalRelief, NoiseLayer
from bong_worldgen.engine.relief import apply_natural_relief


def _grid(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float64)
    return np.meshgrid(axis, axis, indexing="xy")


def test_natural_relief_is_seeded_and_has_real_variation() -> None:
    x, z = _grid()
    base = np.full(x.shape, 68.0, dtype=np.float64)
    config = NaturalRelief()

    first = apply_natural_relief(base, x, z, config, seed=812731)
    repeat = apply_natural_relief(base, x, z, config, seed=812731)
    other = apply_natural_relief(base, x, z, config, seed=812732)

    assert np.array_equal(first, repeat), "同一 seed 必须得到完全相同的连续高度场"
    assert not np.array_equal(first, other), "不同 seed 必须改变崎岖场"
    assert float(np.ptp(first)) > 5.0, "默认崎岖场不能退化成近似平面"
    assert np.isfinite(first).all()


def test_natural_relief_allows_local_multi_block_drops_without_clamping() -> None:
    x, z = _grid()
    base = np.zeros(x.shape, dtype=np.float64)
    config = NaturalRelief(
        macro_amplitude=12.0,
        ridge_amplitude=60.0,
        detail_amplitude=8.0,
        cliff_amplitude=60.0,
        cliff_sharpness=12.0,
        cliff_noise=NoiseLayer(kind="fbm", scale=32.0, octaves=3, gain=0.56),
    )
    result = apply_natural_relief(base, x, z, config, seed=17)
    adjacent = np.concatenate(
        (
            np.abs(np.diff(result, axis=0)).ravel(),
            np.abs(np.diff(result, axis=1)).ravel(),
        )
    )

    assert float(np.max(adjacent)) > 1.0, "陡壁应允许相邻列出现多格高度落差"
    with pytest.raises(ValueError, match="ridge power"):
        NaturalRelief(ridge_power=0.0)
