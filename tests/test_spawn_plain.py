from __future__ import annotations

import numpy as np

from bong_worldgen.engine.spawn_plain import (
    SpawnPlainSelection,
    apply_spawn_plain,
    select_spawn_plain_anchor,
)
from bong_worldgen.engine.terrain_config import NoiseLayer, Point, SpawnPlainSettings


def _settings(**overrides: object) -> SpawnPlainSettings:
    values: dict[str, object] = {
        "center": Point(0.0, 0.0),
        "search_extent_x": 64.0,
        "search_extent_z": 64.0,
        "search_resolution": 8.0,
        "candidate_window_radius": 12.0,
        "plain_radius_x": 24.0,
        "plain_radius_z": 24.0,
        "edge_blend": 6.0,
        "maximum_mean_slope": 0.20,
        "maximum_local_relief": 8.0,
        "minimum_elevation_above_sea": 3.0,
        "smoothing_strength": 0.72,
        "natural_variation": 1.8,
        "micro_noise": NoiseLayer(scale=20.0, octaves=2),
    }
    values.update(overrides)
    return SpawnPlainSettings(**values)


def test_spawn_plain_selects_a_flat_natural_patch_instead_of_fixed_center() -> None:
    settings = _settings()
    axis = np.arange(-64.0, 65.0, 8.0)
    x, z = np.meshgrid(axis, axis, indexing="xy")

    terrain = 70.0 + 0.45 * x + 0.30 * z
    patch = (np.abs(x - 24.0) <= 16.0) & (np.abs(z + 16.0) <= 16.0)
    terrain = np.where(patch, 72.0, terrain)

    selection = select_spawn_plain_anchor(
        settings,
        seed=17,
        terrain_sampler=lambda sample_x, sample_z: np.where(
            (np.abs(sample_x - 24.0) <= 16.0) & (np.abs(sample_z + 16.0) <= 16.0),
            72.0,
            70.0 + 0.45 * sample_x + 0.30 * sample_z,
        ),
        sea_level=61.0,
    )

    assert abs(selection.center_x - 24.0) <= 8.0
    assert abs(selection.center_z + 16.0) <= 8.0
    assert selection.target_height == 72.0
    assert selection.score > 0.0
    assert terrain.shape == x.shape  # 明确测试场与采样网格保持一致


def test_spawn_plain_selection_is_stable_for_same_seed_and_sampler() -> None:
    settings = _settings()

    def sampler(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 74.0 + 0.02 * np.sin(x / 9.0) + 0.03 * np.cos(z / 11.0)

    first = select_spawn_plain_anchor(settings, 812731, sampler, sea_level=61.0)
    second = select_spawn_plain_anchor(settings, 812731, sampler, sea_level=61.0)

    assert first == second


def test_spawn_plain_disabled_is_a_no_op() -> None:
    settings = _settings(enabled=False)
    axis = np.arange(-32.0, 33.0, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 70.0 + x * 0.7 + z * 0.2
    selection = SpawnPlainSelection(0.0, 0.0, 70.0, 1.0)

    output = apply_spawn_plain(terrain, x, z, settings, selection, seed=9)

    assert np.array_equal(output, terrain)
    assert output is not terrain


def test_spawn_plain_falls_back_to_requested_center_without_land_candidate() -> None:
    settings = _settings(center=Point(12.0, -8.0))
    selection = select_spawn_plain_anchor(
        settings,
        seed=1,
        terrain_sampler=lambda x, z: np.full(x.shape, 40.0, dtype=np.float64),
        sea_level=61.0,
    )

    assert (selection.center_x, selection.center_z) == (12.0, -8.0)
    assert selection.target_height == 40.0
    assert selection.score == 0.0


def test_spawn_plain_blends_back_to_natural_terrain_and_keeps_seeded_micro_relief() -> None:
    settings = _settings(
        plain_radius_x=24.0,
        plain_radius_z=24.0,
        edge_blend=6.0,
        smoothing_strength=1.0,
        natural_variation=1.8,
    )
    axis = np.arange(-40.0, 41.0, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 92.0 + 0.6 * x + 0.2 * z
    selection = SpawnPlainSelection(0.0, 0.0, 80.0, 1.0)

    output = apply_spawn_plain(terrain, x, z, settings, selection, seed=73)
    outside = np.hypot(x / 24.0, z / 24.0) >= 1.0
    inside = np.hypot(x / 24.0, z / 24.0) < 0.4

    assert np.array_equal(output[outside], terrain[outside])
    assert np.max(np.abs(output[inside] - terrain[inside])) > 8.0
    assert np.ptp(output[inside] - np.mean(output[inside])) > 0.1
