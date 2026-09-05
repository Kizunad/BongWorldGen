from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.engine.mountains import (
    MOUNTAIN_SURFACE_PALETTE,
    apply_anisotropic_peaks,
    apply_mountains,
    apply_mountains_with_uplift,
    sample_mountain_material_field,
    sample_ridged_multifractal,
    select_peak_anchors,
)
from bong_worldgen.engine.mountains.distance import mountain_distance_field
from bong_worldgen.engine.pipeline import _coordinate_grid, _generate_pre_glacial_terrain
from bong_worldgen.engine.pipeline import generate_heightfield
from bong_worldgen.engine.world_config import TerrainRecipe
from bong_worldgen.engine.terrain_config import (
    MountainRange,
    MountainMaterialSettings,
    MountainPeakSettings,
    NoiseLayer,
    Point,
    RidgedMultifractal,
)


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )


def _mountain(**overrides: object) -> MountainRange:
    values: dict[str, object] = {
        "path": (Point(0.0, 64.0), Point(255.0, 64.0)),
        "width": 36.0,
        "height": 0.0,
        "base_elevation": 72.0,
        "summit_elevation": 360.0,
        "slope_power": 1.45,
        "spine_height_variation": 70.0,
        "spine_ridges": RidgedMultifractal(
            scale=38.0,
            octaves=4,
            warp_scale=90.0,
            warp_strength=7.0,
        ),
        "spine_warp_strength": 18.0,
        "spine_warp_noise": NoiseLayer(kind="fbm", scale=74.0, octaves=3),
        "width_variation": 0.22,
        "width_noise": NoiseLayer(kind="fbm", scale=52.0, octaves=3),
        "flank_ridges": RidgedMultifractal(
            scale=22.0,
            octaves=4,
            warp_scale=70.0,
            warp_strength=5.0,
        ),
        "flank_carving": 0.58,
        "rock_folds": RidgedMultifractal(
            scale=8.0,
            octaves=3,
            warp_scale=28.0,
            warp_strength=2.0,
        ),
        "rock_fold_height": 5.0,
    }
    values.update(overrides)
    return MountainRange(**values)


def test_mountain_spine_is_seeded_and_reproducible() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    mountain = _mountain()

    first = apply_mountains(terrain, x, z, (mountain,), seed=37)
    repeated = apply_mountains(terrain, x, z, (mountain,), seed=37)
    changed = apply_mountains(terrain, x, z, (mountain,), seed=38)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_mountain_uplift_field_matches_positive_terrain_delta() -> None:
    x, z = _grid(128, 65)
    terrain = np.full(x.shape, 70.0)
    mountain = _mountain()

    uplifted, uplift = apply_mountains_with_uplift(terrain, x, z, (mountain,), seed=37)

    assert np.array_equal(uplifted, apply_mountains(terrain, x, z, (mountain,), seed=37))
    assert np.all(uplift >= 0.0)
    assert np.array_equal(uplift, np.maximum(uplifted - terrain, 0.0))
    assert float(uplift.max()) > 0.0


def test_mountain_spine_matches_between_whole_field_and_tiles() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    mountain = _mountain()
    whole = apply_mountains(terrain, x, z, (mountain,), seed=73)

    left = apply_mountains(terrain[:, :128], x[:, :128], z[:, :128], (mountain,), seed=73)
    right = apply_mountains(terrain[:, 128:], x[:, 128:], z[:, 128:], (mountain,), seed=73)

    assert np.array_equal(whole, np.concatenate((left, right), axis=1))


def test_mountain_has_strict_finite_support() -> None:
    x, z = _grid(256, 129)
    terrain = 64.0 + x * 0.01 + z * 0.02
    mountain = _mountain(
        width=28.0,
        spine_warp_strength=0.0,
        width_variation=0.0,
    )
    field = mountain_distance_field(x, z, mountain, seed=11)
    output = apply_mountains(terrain, x, z, (mountain,), seed=11)
    outside = field.distance >= mountain.width

    assert np.count_nonzero(outside) > 0
    assert np.array_equal(output[outside], terrain[outside])


def test_absolute_mountain_profile_is_applied_once() -> None:
    x = np.array([[0.0]])
    z = np.array([[20.0]])
    terrain = np.array([[70.0]])
    mountain = MountainRange(
        path=(Point(-10.0, 0.0), Point(10.0, 0.0)),
        width=40.0,
        height=0.0,
        base_elevation=72.0,
        summit_elevation=360.0,
        slope_power=1.0,
        edge_blend=0.1,
        flank_carving=0.0,
    )

    output = apply_mountains(terrain, x, z, (mountain,), seed=11)

    # 距脊柱一半宽度时 profile=0.5，目标应直接处于 base 与 summit 中点。
    # 旧公式会再次乘 profile，把 216 错压到 143。
    assert output[0, 0] == pytest.approx(216.0)


def test_absolute_mountain_blends_only_at_finite_support_edge() -> None:
    x = np.zeros((1, 3), dtype=np.float64)
    z = np.array([[39.0, 40.0, 41.0]])
    terrain = np.full(x.shape, 20.0)
    mountain = MountainRange(
        path=(Point(-10.0, 0.0), Point(10.0, 0.0)),
        width=40.0,
        height=0.0,
        base_elevation=72.0,
        summit_elevation=360.0,
        slope_power=1.0,
        edge_blend=0.1,
        flank_carving=0.0,
    )

    output = apply_mountains(terrain, x, z, (mountain,), seed=11)

    assert output[0, 0] < 30.0
    assert output[0, 1] == terrain[0, 1]
    assert output[0, 2] == terrain[0, 2]


def test_spine_height_contains_seeded_peaks_and_saddles_without_plateau() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    mountain = _mountain(
        spine_warp_strength=0.0,
        width_variation=0.0,
        flank_carving=0.0,
        rock_fold_height=0.0,
    )
    output = apply_mountains(terrain, x, z, (mountain,), seed=37)
    spine = output[64]
    direction = np.sign(np.diff(spine))
    turning_points = np.count_nonzero(direction[:-1] * direction[1:] < 0.0)

    assert float(np.ptp(spine)) >= 40.0
    assert turning_points >= 4
    assert np.count_nonzero(output >= float(output.max()) - 0.25) < 8


def test_peak_anchors_are_seeded_and_poisson_spaced_on_the_spine() -> None:
    mountain = _mountain(
        peaks=MountainPeakSettings(
            enabled=True,
            count=6,
            candidate_spacing=24.0,
            minimum_spacing=64.0,
            amplitude=18.0,
            along_width=140.0,
            across_width=24.0,
        )
    )

    first = select_peak_anchors(mountain, seed=37)
    repeated = select_peak_anchors(mountain, seed=37)
    changed = select_peak_anchors(mountain, seed=38)

    assert first == repeated
    assert first
    assert len(first) <= mountain.peaks.count
    assert all(0.0 < anchor.progress < 1.0 for anchor in first)
    assert all(
        (second.progress - previous.progress) * 255.0 >= mountain.peaks.minimum_spacing
        for previous, second in zip(first, first[1:])
    )
    assert first != changed


def test_peak_kernel_is_wide_along_spine_and_narrow_across_it() -> None:
    mountain = _mountain(
        peaks=MountainPeakSettings(
            enabled=True,
            count=1,
            candidate_spacing=48.0,
            minimum_spacing=48.0,
            amplitude=30.0,
            along_width=300.0,
            across_width=18.0,
            smoothness=0.0,
        )
    )
    anchor = select_peak_anchors(mountain, seed=37)[0]
    progress = np.array(
        [[anchor.progress, anchor.progress + 100.0 / 255.0, anchor.progress]],
        dtype=np.float64,
    )
    distance = np.array([[0.0, 0.0, 18.0]], dtype=np.float64)
    ridge_height = np.zeros(progress.shape, dtype=np.float64)
    local_width = np.full(progress.shape, mountain.width, dtype=np.float64)

    result = apply_anisotropic_peaks(
        ridge_height,
        progress,
        distance,
        local_width,
        mountain,
        seed=37,
    )

    assert result[0, 0] > result[0, 1] > 0.0
    assert result[0, 1] > result[0, 2]


def test_peaks_raise_the_spine_but_have_no_effect_outside_mountain_support() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    plain = _mountain(
        spine_warp_strength=0.0,
        width_variation=0.0,
        flank_carving=0.0,
        rock_fold_height=0.0,
        peaks=MountainPeakSettings(),
    )
    peaked = replace(
        plain,
        peaks=MountainPeakSettings(
            enabled=True,
            count=5,
            candidate_spacing=40.0,
            minimum_spacing=72.0,
            amplitude=28.0,
            along_width=180.0,
            across_width=24.0,
        ),
    )

    plain_output = apply_mountains(terrain, x, z, (plain,), seed=37)
    peaked_output = apply_mountains(terrain, x, z, (peaked,), seed=37)
    field = mountain_distance_field(x, z, peaked, seed=37)
    outside = field.distance >= peaked.width

    assert float(np.max(peaked_output - plain_output)) > 1.0
    assert np.array_equal(peaked_output[outside], plain_output[outside])


def test_flank_noise_carves_valleys_without_changing_spine_or_foot() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    smooth_mountain = _mountain(
        spine_warp_strength=0.0,
        width_variation=0.0,
        flank_carving=0.0,
        rock_fold_height=0.0,
    )
    carved_mountain = replace(smooth_mountain, flank_carving=0.78)
    smooth = apply_mountains(terrain, x, z, (smooth_mountain,), seed=101)
    carved = apply_mountains(terrain, x, z, (carved_mountain,), seed=101)

    assert np.array_equal(carved[64], smooth[64])
    assert np.array_equal(carved[0], terrain[0])
    assert np.all(carved <= smooth)
    assert float(np.max(smooth - carved)) >= 20.0


def test_mountain_cross_section_produces_multi_block_voxel_slopes() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    mountain = _mountain(
        spine_warp_strength=0.0,
        width_variation=0.0,
        flank_carving=0.0,
        rock_fold_height=0.0,
    )
    output = apply_mountains(terrain, x, z, (mountain,), seed=37)
    voxel_height = np.rint(output)
    delta_x = np.abs(np.diff(voxel_height, axis=1))
    delta_z = np.abs(np.diff(voxel_height, axis=0))

    assert max(float(delta_x.max()), float(delta_z.max())) >= 4.0
    assert np.count_nonzero(delta_x >= 2.0) + np.count_nonzero(delta_z >= 2.0) >= 1_000


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"slope_power": 0.0}, "slope_power"),
        ({"crest_width": -1.0}, "crest_width"),
        ({"crest_width": 36.0}, "crest_width"),
        ({"edge_blend": 0.0}, "edge_blend"),
        ({"spine_height_variation": -1.0}, "variations"),
        ({"spine_peak_width": -1.0}, "variations"),
        ({"spine_warp_strength": -1.0}, "variations"),
        ({"width_variation": 1.0}, "width_variation"),
        ({"spine_height_variation": 288.0}, "absolute relief"),
        ({"flank_carving": 1.1}, "flank_carving"),
        ({"rock_fold_height": -1.0}, "rock_fold_height"),
    ),
)
def test_mountain_spine_rejects_invalid_parameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _mountain(**overrides)


def test_default_mountain_window_has_hundreds_of_blocks_of_relief() -> None:
    x, z = _coordinate_grid(512, 512, -7200.0, -5600.0, 4.0)
    terrain = _generate_pre_glacial_terrain(DEFAULT_RECIPE, x, z, seed=812731)

    assert float(np.ptp(terrain)) >= 300.0
    assert float(terrain.max()) >= 400.0


def test_default_main_peak_has_connected_voxel_crest() -> None:
    # 默认预览最高峰附近按真实 1 方块分辨率采样，直接锁住 BlueMap 中可见的
    # 峰冠行为：最高层至少有两个相邻方块，同时不能扩成大面积平顶。
    x, z = _coordinate_grid(80, 80, -5232.0, -4528.0, 1.0)
    terrain = _generate_pre_glacial_terrain(DEFAULT_RECIPE, x, z, seed=812731)
    voxel_height = np.rint(terrain)
    peak = voxel_height == float(voxel_height.max())
    horizontally_connected = np.any(peak[:, :-1] & peak[:, 1:])
    vertically_connected = np.any(peak[:-1, :] & peak[1:, :])

    assert horizontally_connected or vertically_connected
    assert 2 <= int(np.count_nonzero(peak)) <= 24


def test_ridged_multifractal_is_bounded_seeded_and_tile_stable() -> None:
    x, z = _grid(192, 96)
    config = RidgedMultifractal(
        scale=34.0,
        octaves=5,
        warp_scale=110.0,
        warp_strength=13.0,
    )
    whole = sample_ridged_multifractal(x, z, config, seed=29)
    repeated = sample_ridged_multifractal(x, z, config, seed=29)
    changed = sample_ridged_multifractal(x, z, config, seed=30)
    tiled = np.concatenate(
        (
            sample_ridged_multifractal(x[:, :91], z[:, :91], config, seed=29),
            sample_ridged_multifractal(x[:, 91:], z[:, 91:], config, seed=29),
        ),
        axis=1,
    )

    assert np.array_equal(whole, repeated)
    assert np.array_equal(whole, tiled)
    assert not np.array_equal(whole, changed)
    assert float(whole.min()) >= 0.0
    assert float(whole.max()) <= 1.0
    assert float(np.ptp(whole)) >= 0.25


def test_mountain_material_field_is_seeded_mixed_and_limited_to_spine() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 72.0, dtype=np.float64)
    mountain = _mountain(width=36.0, spine_warp_strength=0.0, width_variation=0.0)
    # 由脊线向两侧连续升高，覆盖雪线和多种坡度情形。
    terrain += np.maximum(0.0, 1.0 - np.abs(z - 64.0) / 36.0) * 288.0
    settings = MountainMaterialSettings(
        snowline_base_elevation=170.0,
        snowline_large_amplitude=20.0,
        snowline_small_amplitude=7.0,
    )
    cold = np.ones_like(terrain, dtype=np.float32)
    first = sample_mountain_material_field(
        terrain, x, z, (mountain,), seed=41, settings=settings, climate_cold_weight=cold
    )
    repeated = sample_mountain_material_field(
        terrain, x, z, (mountain,), seed=41, settings=settings, climate_cold_weight=cold
    )
    changed = sample_mountain_material_field(
        terrain, x, z, (mountain,), seed=42, settings=settings, climate_cold_weight=cold
    )

    outside = np.abs(z - 64.0) >= mountain.width
    assert np.all(first.material_id[outside] == 0), "山脉支持域外不得写入寒带材质"
    assert np.array_equal(first.material_id, repeated.material_id)
    assert not np.array_equal(first.material_id, changed.material_id)
    assert set(np.unique(first.material_id)) >= {0, 1, 2, 3, 4, 5}
    assert first.material_id.dtype == np.uint8
    assert first.mountain_weight.dtype == np.float32
    assert first.snowline.dtype == np.float32
    assert first.slope_angle.dtype == np.float32
    assert first.rock_exposure.dtype == np.float32
    assert MOUNTAIN_SURFACE_PALETTE[4] == "minecraft:blue_ice"


def test_mountain_material_field_uses_climate_and_glacier_as_continuous_inputs() -> None:
    x, z = _grid(129, 129)
    terrain = np.full(x.shape, 210.0, dtype=np.float64)
    mountain = _mountain(
        path=(Point(0.0, 64.0), Point(128.0, 64.0)),
        width=30.0,
        spine_warp_strength=0.0,
        width_variation=0.0,
    )
    settings = MountainMaterialSettings(
        snowline_base_elevation=200.0,
        snowline_large_amplitude=0.0,
        snowline_small_amplitude=0.0,
        boundary_noise_strength=0.0,
        snow_slope_bonus=0.0,
        snow_slope_penalty=0.0,
        exposure_strength=0.0,
    )
    warm = sample_mountain_material_field(
        terrain,
        x,
        z,
        (mountain,),
        seed=7,
        settings=settings,
        climate_cold_weight=np.zeros_like(terrain),
        glacier_weight=np.zeros_like(terrain),
    )
    cold_glacier = sample_mountain_material_field(
        terrain,
        x,
        z,
        (mountain,),
        seed=7,
        settings=settings,
        climate_cold_weight=np.ones_like(terrain),
        glacier_weight=np.ones_like(terrain),
    )

    assert float(cold_glacier.snow_score.max()) > float(warm.snow_score.max())
    assert np.count_nonzero(cold_glacier.material_id) >= np.count_nonzero(warm.material_id)


def test_mountain_material_field_adds_seeded_rock_exposure_on_steep_flanks() -> None:
    x, z = _grid(128, 96)
    mountain = _mountain(
        path=(Point(0.0, 48.0), Point(127.0, 48.0)),
        width=24.0,
        spine_warp_strength=0.0,
        width_variation=0.0,
    )
    # 同一条山脉同时包含平缓台肩和陡坡，验证岩石由坡度场而非固定海拔决定。
    terrain = 92.0 + np.abs(z - 48.0) * 5.0
    settings = MountainMaterialSettings(
        snowline_base_elevation=90.0,
        snowline_large_amplitude=0.0,
        snowline_small_amplitude=0.0,
        boundary_noise_strength=0.0,
        rock_exposure_max_probability=0.58,
    )
    field = sample_mountain_material_field(
        terrain,
        x,
        z,
        (mountain,),
        seed=912,
        settings=settings,
        climate_cold_weight=np.ones_like(terrain),
    )

    assert np.all((field.rock_exposure >= 0.0) & (field.rock_exposure <= 1.0))
    assert np.any(field.rock_exposure > 0.8), "陡坡应进入高裸岩暴露区"
    assert np.any(field.rock_exposure == 0.0), "平缓坡/脊顶不应被判定为高裸岩暴露"
    rock_ids = field.material_id[field.material_id >= 7]
    assert rock_ids.size > 0, "陡坡裸岩应实际写入山脉材质 ID"
    assert np.all(rock_ids <= len(MOUNTAIN_SURFACE_PALETTE))
    assert np.array_equal(
        field.material_id,
        sample_mountain_material_field(
            terrain,
            x,
            z,
            (mountain,),
            seed=912,
            settings=settings,
            climate_cold_weight=np.ones_like(terrain),
        ).material_id,
    )


def test_mountain_material_field_honors_per_biome_rock_palette() -> None:
    x, z = _grid(96, 64)
    mountain = _mountain(
        path=(Point(0.0, 32.0), Point(95.0, 32.0)),
        width=20.0,
        spine_warp_strength=0.0,
        width_variation=0.0,
    )
    settings = MountainMaterialSettings(
        snowline_base_elevation=0.0,
        snowline_large_amplitude=0.0,
        snowline_small_amplitude=0.0,
        rock_palette=("minecraft:deepslate",),
    )
    field = sample_mountain_material_field(
        100.0 + np.abs(z - 32.0) * 2.0,
        x,
        z,
        (mountain,),
        seed=77,
        settings=settings,
        climate_cold_weight=np.ones(x.shape),
    )

    deepslate_id = MOUNTAIN_SURFACE_PALETTE.index("minecraft:deepslate") + 1
    rock_ids = field.material_id[field.material_id >= 7]
    assert rock_ids.size > 0
    assert set(np.unique(rock_ids)) == {deepslate_id}


def test_generate_heightfield_exposes_mountain_material_query_layers() -> None:
    mountain = _mountain(
        path=(Point(0.0, 32.0), Point(63.0, 32.0)),
        width=22.0,
        spine_warp_strength=0.0,
        width_variation=0.0,
    )
    recipe = TerrainRecipe(
        name="mountain_material_contract",
        base_height=72.0,
        mountains=(mountain,),
        mountain_materials=MountainMaterialSettings(
            snowline_base_elevation=120.0,
            boundary_noise_strength=0.0,
        ),
    )
    field = generate_heightfield(recipe, width=64, height=64, seed=99)

    assert field.mountain_material_palette == MOUNTAIN_SURFACE_PALETTE
    assert field.mountain_material_id.shape == field.height.shape
    assert field.mountain_weight.shape == field.height.shape
    assert field.mountain_snowline.shape == field.height.shape
    assert field.mountain_material_score.shape == field.height.shape
    assert field.mountain_slope_angle.shape == field.height.shape
    assert field.mountain_exposure.shape == field.height.shape
    assert field.mountain_rock_exposure.shape == field.height.shape
    assert np.count_nonzero(field.mountain_material_id) > 0


def test_rock_folds_only_change_upper_flanks() -> None:
    x, z = _grid(256, 129)
    terrain = np.full(x.shape, 70.0)
    smooth_mountain = _mountain(rock_fold_height=0.0)
    folded_mountain = replace(smooth_mountain, rock_fold_height=12.0)
    smooth = apply_mountains(terrain, x, z, (smooth_mountain,), seed=113)
    folded = apply_mountains(terrain, x, z, (folded_mountain,), seed=113)
    field = mountain_distance_field(x, z, smooth_mountain, seed=113)
    changed = np.abs(folded - smooth) > 1.0e-9

    assert np.count_nonzero(changed) > 500
    assert not np.any(changed[field.normalized_distance >= 1.0])
    assert not np.any(changed[field.normalized_distance == 0.0])
    outside = field.normalized_distance >= 1.0
    assert np.array_equal(folded[outside], terrain[outside])
    assert float(folded.max()) <= folded_mountain.summit_elevation


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"scale": 0.0}, "scales"),
        ({"octaves": 0}, "octaves"),
        ({"lacunarity": 1.0}, "lacunarity"),
        ({"persistence": 0.0}, "persistence"),
        ({"ridge_offset": 0.0}, "ridge parameters"),
        ({"ridge_gain": -1.0}, "ridge parameters"),
        ({"ridge_power": 0.0}, "ridge parameters"),
        ({"warp_strength": -1.0}, "warp strength"),
    ),
)
def test_ridged_multifractal_rejects_invalid_parameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RidgedMultifractal(**overrides)
