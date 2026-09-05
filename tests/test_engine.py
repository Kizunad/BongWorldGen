from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.data.wilderness import GRASSLAND, LAKE, MOUNTAINS, RIVER, classify_wilderness
from bong_worldgen.preview_world import export_preview_world
from bong_worldgen.engine import (
    Basin,
    Canyon,
    CaveNetwork,
    ClimateBand,
    ClimateTransition,
    ClimateWorldBounds,
    GlobalClimatePlan,
    GlacialSystem,
    MountainRange,
    NoiseLayer,
    Point,
    River,
    SolidOreSpec,
    TerrainRecipe,
    UndergroundBlock,
    UndergroundRiverNetwork,
    generate_heightfield,
    sample_climate,
)
from bong_worldgen.engine.caves import generate_cave_topology, sample_worm_field
from bong_worldgen.engine.constants import CAVE_RARITY_MULTIPLIERS, SPAN_MIN_Y
from bong_worldgen.engine.distribution import iter_world_network_instances
from bong_worldgen.engine.noise import sample_noise_3d
from bong_worldgen.engine.underground_rivers import generate_underground_rivers
from bong_worldgen.engine.underground_rivers.fractures import zero_isoline_band
from bong_worldgen.engine.canyons import apply_canyons
from bong_worldgen.engine.canyons.distance import warped_polyline_distance_and_progress
from bong_worldgen.engine.geometry import polyline_stations, sample_regular_grid
from bong_worldgen.engine.mountains import MOUNTAIN_SURFACE_PALETTE
from bong_worldgen.engine.randomness import mix64, stable_text_seed, unit_interval
from bong_worldgen.engine.surface import build_visible_surface_layer
from bong_worldgen.engine.glaciers import (
    GLACIAL_COVER_PALETTE,
    GLACIAL_LANDFORM_PALETTE,
    GLACIAL_SURFACE_PALETTE,
    GLACIAL_WATER_PALETTE,
    PERMAFROST_PALETTE,
    GlacialFlowPlan,
    apply_surface_protrusions,
    apply_glacial_meltwater,
    glacial_landform_ids,
    glacial_surface_materials,
    glacial_surface_layers,
    permafrost_surface_materials,
    plan_glacial_flow,
    sample_glacial_mass_balance,
)
from bong_worldgen.engine.glaciers.erosion import apply_freeze_thaw


def _positive_glacial_plan(
    system: GlacialSystem,
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> GlacialFlowPlan:
    """为下游算法测试提供不受气候门控影响的正积累规划场。"""

    distance = np.hypot(x - system.seed_center.x, z - system.seed_center.z)
    spread = max(system.flow_source_separation, system.cirque_radius, 1.0)
    accumulation = 0.2 + np.exp(-((distance / spread) ** 2))
    balance = accumulation.copy()
    return plan_glacial_flow(system, terrain, x, z, accumulation, balance, seed)


def test_default_global_climate_plan_has_ordered_north_to_south_bands() -> None:
    plan = GlobalClimatePlan()

    assert [band.kind for band in plan.bands] == ["cold", "temperate", "tropical"]
    assert [band.surface_family for band in plan.bands] == [
        "snow",
        "temperate_grassland",
        "desert",
    ]
    assert [(item.from_kind, item.to_kind) for item in plan.transitions] == [
        ("cold", "temperate"),
        ("temperate", "tropical"),
    ]
    assert plan.axis == "world_z"


def test_global_climate_plan_rejects_non_adjacent_band_layout() -> None:
    with pytest.raises(ValueError, match="ordered cold, temperate, tropical"):
        GlobalClimatePlan(
            bands=(
                ClimateBand("temperate", "温带", "temperate_grassland", -1.0, -0.34),
                ClimateBand("cold", "寒带", "snow", -0.34, 0.34),
                ClimateBand("tropical", "热带", "desert", 0.34, 1.0),
            )
        )

    with pytest.raises(ValueError, match="two different bands"):
        ClimateTransition("temperate", "temperate")

    with pytest.raises(ValueError, match="contiguous"):
        GlobalClimatePlan(
            bands=(
                ClimateBand("cold", "寒带", "snow", -1.0, -0.40),
                ClimateBand("temperate", "温带", "temperate_grassland", -0.34, 0.34),
                ClimateBand("tropical", "热带", "desert", 0.34, 1.0),
            )
        )

    with pytest.raises(ValueError, match="width"):
        ClimateTransition("cold", "temperate", width=float("nan"))


def test_sample_climate_uses_explicit_world_bounds_and_transition_weights() -> None:
    plan = GlobalClimatePlan()
    bounds = ClimateWorldBounds(north_z=-100.0, south_z=100.0)
    z = np.asarray([[-100.0, -50.0, -34.0, 0.0, 34.0, 50.0, 100.0]])

    field = sample_climate(z, plan, bounds)

    assert field.climate_id.tolist() == [[1, 1, 2, 2, 3, 3, 3]]
    assert field.transition_id[0, 2] == 1
    assert field.transition_id[0, 4] == 2
    assert 0.0 < field.transition_weight[0, 2] <= 1.0
    assert 0.0 < field.transition_weight[0, 4] <= 1.0
    assert field.cold_weight[0, 0] == pytest.approx(1.0)
    assert field.cold_weight[0, 4] == pytest.approx(0.0)


def test_custom_recipe_without_world_bounds_does_not_fake_global_climate() -> None:
    field = generate_heightfield(TerrainRecipe(name="unmapped_climate"), width=8, height=8, seed=7)

    assert field.climate_palette == ()
    assert np.count_nonzero(field.climate_id) == 0


def test_default_recipe_emits_climate_layers_from_world_metadata() -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=16, height=16, seed=7)

    assert field.climate_palette == ("cold", "temperate", "tropical")
    assert field.climate_surface_palette == ("snow", "temperate_grassland", "desert")
    assert field.climate_id.dtype == np.uint8
    assert field.climate_transition_weight.dtype == np.float32
    assert set(np.unique(field.climate_id)).issubset({1, 2, 3})


def test_climate_metadata_does_not_change_generation_before_algorithm_exists() -> None:
    default = GlobalClimatePlan()
    custom = replace(
        default,
        transitions=(
            ClimateTransition("cold", "temperate", width=0.20),
            ClimateTransition("temperate", "tropical", width=0.08),
        ),
    )
    first = generate_heightfield(
        TerrainRecipe(name="climate_default", climate=default), width=24, height=20, seed=7
    )
    second = generate_heightfield(
        TerrainRecipe(name="climate_custom", climate=custom), width=24, height=20, seed=7
    )

    assert np.array_equal(first.height, second.height)
    assert np.array_equal(first.water_level, second.water_level)


def test_glacial_flow_uses_positive_accumulation_peaks_and_descends() -> None:
    axis = np.arange(81, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 220.0 - z * 0.75 + 3.0 * np.sin(x / 11.0)
    accumulation = np.zeros_like(terrain)
    for center_x in (20.0, 40.0, 60.0):
        accumulation += np.exp(-((x - center_x) ** 2 + (z - 18.0) ** 2) / 18.0)
    balance = np.where(z < 58.0, accumulation - 0.02, -0.25)
    system = GlacialSystem(
        name="test_glacier",
        seed_center=Point(40.0, 40.0),
        seed_extent=56.0,
        cirque_count=3,
        valley_count=3,
        valley_segments=12,
        valley_length=60.0,
        flow_source_separation=14.0,
        flow_inertia=0.62,
        flow_meander_strength=0.30,
        flow_merge_distance=0.0,
        flow_min_length=20.0,
        flow_termination_balance=-0.10,
    )
    first = plan_glacial_flow(system, terrain, x, z, accumulation, balance, seed=21)
    repeated = plan_glacial_flow(system, terrain, x, z, accumulation, balance, seed=21)
    other_seed = plan_glacial_flow(system, terrain, x, z, accumulation, balance, seed=22)

    assert first == repeated
    assert len(first.cirques) == 3
    assert len(first.paths) == 3
    assert all(cirque.mass_balance > 0.0 for cirque in first.cirques)
    assert sorted(round(cirque.center.x) for cirque in first.cirques) == [20, 40, 60]
    assert first.paths != other_seed.paths, "seed 应改变惯性流线内的局部弯曲"
    for path in first.paths:
        elevations = sample_regular_grid(
            terrain,
            x,
            z,
            np.asarray([point.x for point in path]),
            np.asarray([point.z for point in path]),
        )
        assert np.all(np.diff(elevations) <= 1.0e-8), "冰川流路不能无故爬坡"
        assert all(
            np.hypot(point.x - system.seed_center.x, point.z - system.seed_center.z)
            < system.seed_extent
            for point in path
        ), "流路到达系统边界后必须终止，不能反弹"


def test_glacial_flow_requires_positive_mass_balance() -> None:
    axis = np.arange(24, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 120.0 - z
    system = GlacialSystem(
        name="no_accumulation",
        seed_center=Point(12.0, 12.0),
        seed_extent=16.0,
        flow_source_separation=4.0,
        flow_min_length=0.0,
    )
    plan = plan_glacial_flow(
        system,
        terrain,
        x,
        z,
        np.ones_like(terrain),
        -np.ones_like(terrain),
        seed=9,
    )

    assert plan.cirques == ()
    assert plan.paths == ()


def test_glacial_flow_merges_tributary_into_existing_lower_trunk() -> None:
    axis = np.arange(81, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 220.0 - z * 0.8 + np.abs(x - 40.0) * 0.25
    accumulation = np.exp(-((x - 24.0) ** 2 + (z - 12.0) ** 2) / 10.0)
    accumulation += np.exp(-((x - 56.0) ** 2 + (z - 12.0) ** 2) / 10.0)
    balance = np.where(z < 70.0, accumulation + 0.05, -0.20)
    system = GlacialSystem(
        name="tributary_merge",
        seed_center=Point(40.0, 40.0),
        seed_extent=56.0,
        cirque_count=2,
        valley_count=2,
        valley_segments=18,
        valley_length=90.0,
        valley_turn=0.50,
        flow_source_separation=20.0,
        flow_inertia=0.25,
        flow_meander_strength=0.0,
        flow_direction_samples=11,
        flow_merge_distance=7.0,
        flow_min_length=20.0,
        flow_termination_balance=-0.10,
    )

    plan = plan_glacial_flow(system, terrain, x, z, accumulation, balance, seed=4)

    assert len(plan.paths) == 2
    shared_points = set(plan.paths[0]) & set(plan.paths[1])
    assert len(shared_points) >= 2, (
        "支流汇入后必须复用主干后缀，而不是平行重复开槽"
    )
    first_shared = next(
        index for index, point in enumerate(plan.paths[1]) if point in shared_points
    )
    assert tuple(plan.paths[1][first_shared:]) == tuple(
        plan.paths[0][plan.paths[0].index(plan.paths[1][first_shared]) :]
    )


def test_glacial_flow_rejects_invalid_fields_and_disabled_system_is_empty() -> None:
    axis = np.arange(8, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 100.0 - z
    accumulation = np.ones_like(terrain)
    balance = np.ones_like(terrain)
    system = GlacialSystem(
        name="flow_contract",
        seed_center=Point(4.0, 4.0),
        seed_extent=6.0,
        flow_source_separation=2.0,
    )

    with pytest.raises(ValueError, match="share a shape"):
        plan_glacial_flow(system, terrain, x[:, :-1], z, accumulation, balance, seed=1)
    invalid_terrain = terrain.copy()
    invalid_terrain[2, 2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        plan_glacial_flow(
            system,
            invalid_terrain,
            x,
            z,
            accumulation,
            balance,
            seed=1,
        )
    irregular_x = x.copy()
    irregular_x[:, 4:] += 0.5
    with pytest.raises(ValueError, match="regular grid"):
        plan_glacial_flow(
            system,
            terrain,
            irregular_x,
            z,
            accumulation,
            balance,
            seed=1,
        )

    disabled = replace(system, mass_balance_enabled=False)
    plan = plan_glacial_flow(
        disabled,
        terrain,
        x,
        z,
        accumulation,
        balance,
        seed=1,
    )
    assert plan.cirques == () and plan.paths == ()


def test_glacial_mass_balance_accumulates_more_snow_on_windward_slopes() -> None:
    axis = np.linspace(-32.0, 32.0, 65, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 180.0 + x * 0.5
    common = dict(
        name="windward_snow",
        seed_center=Point(0.0, 0.0),
        seed_extent=400.0,
        cirque_count=0,
        valley_count=0,
        equilibrium_line_elevation=20.0,
        equilibrium_transition=10.0,
        snowfall_variability=0.0,
        windward_accumulation_strength=0.8,
        leeward_drift_strength=0.1,
        ablation_rate=0.0,
    )
    windward = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (GlacialSystem(**common, prevailing_wind_angle_degrees=0.0),),
        seed=17,
        sea_level=62.0,
    )
    leeward = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (GlacialSystem(**common, prevailing_wind_angle_degrees=180.0),),
        seed=17,
        sea_level=62.0,
    )

    core = np.s_[8:-8, 8:-8]
    assert float(np.mean(windward.snow_accumulation[core])) > float(
        np.mean(leeward.snow_accumulation[core])
    ) * 1.25
    assert np.allclose(windward.mass_balance, windward.snow_accumulation)


def test_glacial_mass_balance_separates_accumulation_and_ablation_zones() -> None:
    axis = np.linspace(-40.0, 40.0, 81, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 62.0 + (z + 40.0) * 2.0
    system = GlacialSystem(
        name="equilibrium_line",
        seed_center=Point(0.0, 0.0),
        seed_extent=500.0,
        cirque_count=0,
        valley_count=0,
        equilibrium_line_elevation=80.0,
        equilibrium_transition=30.0,
        snowfall_variability=0.0,
        windward_accumulation_strength=0.0,
        leeward_drift_strength=0.0,
        ablation_rate=0.9,
        minimum_ablation=0.08,
        low_elevation_ablation_strength=0.9,
        solar_ablation_strength=0.0,
        transition_ablation_strength=0.0,
    )
    field = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (system,),
        seed=23,
        sea_level=62.0,
        cold_weight=np.ones_like(terrain),
    )

    assert float(np.mean(field.mass_balance[8:20, 20:-20])) < 0.0
    assert float(np.mean(field.mass_balance[-20:-8, 20:-20])) > 0.0
    assert np.all(field.snow_accumulation >= 0.0)


def test_glacial_mass_balance_increases_ablation_on_sun_facing_slopes() -> None:
    axis = np.linspace(-40.0, 40.0, 81, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = 180.0 + np.abs(z) * 0.5
    system = GlacialSystem(
        name="solar_ablation",
        seed_center=Point(0.0, 0.0),
        seed_extent=500.0,
        cirque_count=0,
        valley_count=0,
        prevailing_wind_angle_degrees=0.0,
        sun_facing_angle_degrees=90.0,
        equilibrium_line_elevation=20.0,
        equilibrium_transition=10.0,
        snowfall_variability=0.0,
        windward_accumulation_strength=0.0,
        leeward_drift_strength=0.0,
        minimum_ablation=0.0,
        low_elevation_ablation_strength=0.0,
        solar_ablation_strength=0.8,
        transition_ablation_strength=0.0,
    )
    field = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (system,),
        seed=29,
        sea_level=62.0,
    )

    sun_facing = field.mass_balance[8:30, 20:-20]
    shaded = field.mass_balance[-30:-8, 20:-20]
    assert float(np.mean(sun_facing)) < float(np.mean(shaded))


def test_glacial_mass_balance_is_zero_outside_cold_climate_or_when_disabled() -> None:
    axis = np.arange(16, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    terrain = np.full(x.shape, 180.0)
    system = GlacialSystem(
        name="climate_gate",
        seed_center=Point(8.0, 8.0),
        seed_extent=100.0,
        cirque_count=0,
        valley_count=0,
    )
    no_cold = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (system,),
        seed=37,
        sea_level=62.0,
        cold_weight=np.zeros_like(terrain),
    )
    disabled = sample_glacial_mass_balance(
        terrain,
        x,
        z,
        (replace(system, mass_balance_enabled=False),),
        seed=37,
        sea_level=62.0,
    )

    assert not np.any(no_cold.snow_accumulation)
    assert not np.any(no_cold.mass_balance)
    assert not np.any(disabled.snow_accumulation)
    assert not np.any(disabled.mass_balance)


def test_glacial_mass_balance_validates_shapes_and_finite_inputs() -> None:
    terrain = np.ones((4, 4), dtype=np.float64)
    x, z = np.meshgrid(np.arange(4, dtype=np.float64), np.arange(4, dtype=np.float64))
    system = GlacialSystem(name="input_contract", cirque_count=0, valley_count=0)

    with pytest.raises(ValueError, match="share a two-dimensional shape"):
        sample_glacial_mass_balance(
            terrain,
            x[:, :-1],
            z,
            (system,),
            seed=1,
            sea_level=0.0,
        )
    with pytest.raises(ValueError, match="cold weight must match"):
        sample_glacial_mass_balance(
            terrain,
            x,
            z,
            (system,),
            seed=1,
            sea_level=0.0,
            cold_weight=np.ones((3, 4)),
        )
    invalid = terrain.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="inputs must be finite"):
        sample_glacial_mass_balance(
            invalid,
            x,
            z,
            (system,),
            seed=1,
            sea_level=0.0,
        )
    invalid_cold = np.ones_like(terrain)
    invalid_cold[0, 0] = np.nan
    with pytest.raises(ValueError, match="cold weight must be finite"):
        sample_glacial_mass_balance(
            terrain,
            x,
            z,
            (system,),
            seed=1,
            sea_level=0.0,
            cold_weight=invalid_cold,
        )
    with pytest.raises(ValueError, match="sea level must be finite"):
        sample_glacial_mass_balance(
            terrain,
            x,
            z,
            (system,),
            seed=1,
            sea_level=float("nan"),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        sample_glacial_mass_balance(
            np.empty((0, 4)),
            np.empty((0, 4)),
            np.empty((0, 4)),
            (system,),
            seed=1,
            sea_level=0.0,
        )


def test_glacial_mass_balance_accepts_single_cell_fields() -> None:
    field = sample_glacial_mass_balance(
        np.asarray([[180.0]]),
        np.asarray([[4.0]]),
        np.asarray([[7.0]]),
        (
            GlacialSystem(
                name="single_cell",
                seed_center=Point(4.0, 7.0),
                cirque_count=0,
                valley_count=0,
            ),
        ),
        seed=11,
        sea_level=62.0,
    )

    assert field.snow_accumulation.shape == (1, 1)
    assert field.mass_balance.shape == (1, 1)
    assert np.isfinite(field.snow_accumulation).all()
    assert np.isfinite(field.mass_balance).all()


def test_glacial_mass_balance_pipeline_is_chunk_stable_at_shared_border() -> None:
    recipe = TerrainRecipe(
        name="mass_balance_chunk_contract",
        base_height=150.0,
        base_noise=(NoiseLayer(kind="fbm", scale=48.0, amplitude=18.0, octaves=3),),
        glaciers=(
            GlacialSystem(
                name="chunk_stable_snow",
                seed_center=Point(32.0, 16.0),
                seed_extent=500.0,
                cirque_count=0,
                valley_count=0,
                drumlin_count=0,
                moraine_height=0.0,
                freeze_thaw_iterations=0,
                equilibrium_line_elevation=20.0,
            ),
        ),
    )
    whole = generate_heightfield(recipe, width=64, height=32, seed=31)
    left = generate_heightfield(recipe, width=32, height=32, seed=31)
    right = generate_heightfield(recipe, width=32, height=32, seed=31, origin_x=32.0)

    stitched_snow = np.concatenate(
        (left.snow_accumulation, right.snow_accumulation), axis=1
    )
    stitched_balance = np.concatenate(
        (left.glacial_mass_balance, right.glacial_mass_balance), axis=1
    )
    assert np.array_equal(whole.snow_accumulation, stitched_snow)
    assert np.array_equal(whole.glacial_mass_balance, stitched_balance)


def test_glacial_flow_plan_keeps_carving_and_semantics_stable_across_tiles() -> None:
    system = GlacialSystem(
        name="flow_chunk_contract",
        seed_center=Point(32.0, 16.0),
        seed_extent=48.0,
        cirque_count=2,
        cirque_radius=8.0,
        cirque_min_elevation=1.0,
        valley_count=2,
        valley_segments=12,
        valley_length=48.0,
        valley_source_width=4.0,
        valley_floor_width=1.5,
        valley_depth=12.0,
        moraine_height=0.0,
        drumlin_count=0,
        freeze_thaw_iterations=0,
        surface_protrusion_probability=0.0,
        equilibrium_line_elevation=1.0,
        equilibrium_transition=12.0,
        flow_planning_resolution=4.0,
        flow_source_separation=10.0,
        flow_inertia=0.55,
        flow_merge_distance=5.0,
    )
    recipe = TerrainRecipe(
        name="flow_chunk_contract",
        base_height=140.0,
        base_noise=(
            NoiseLayer(kind="fbm", scale=36.0, amplitude=12.0, octaves=3),
        ),
        glaciers=(system,),
    )

    whole = generate_heightfield(recipe, width=64, height=32, seed=37)
    left = generate_heightfield(recipe, width=32, height=32, seed=37)
    right = generate_heightfield(recipe, width=32, height=32, seed=37, origin_x=32.0)

    assert np.array_equal(whole.height, np.concatenate((left.height, right.height), axis=1))
    assert np.array_equal(
        whole.glacial_landform_id,
        np.concatenate((left.glacial_landform_id, right.glacial_landform_id), axis=1),
    )
    assert np.count_nonzero(whole.glacial_landform_id) > 0


def test_glacial_landform_ids_cover_each_algorithm_and_follow_climate_gate() -> None:
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full(x.shape, 120.0, dtype=np.float64)
    system = GlacialSystem(
        name="semantic_glacier",
        seed_center=Point(96.0, 96.0),
        seed_extent=32.0,
        cirque_count=2,
        cirque_radius=18.0,
        cirque_min_elevation=1.0,
        valley_count=2,
        valley_segments=6,
        valley_length=70.0,
        valley_source_width=5.0,
        valley_floor_width=2.0,
        valley_depth=18.0,
        valley_width_growth=3.0,
        moraine_height=8.0,
        moraine_width=8.0,
        moraine_length=14.0,
        drumlin_count=8,
        drumlin_length=18.0,
        drumlin_width=5.0,
        drumlin_height=6.0,
        drumlin_max_elevation=100.0,
        flow_source_separation=14.0,
        flow_min_length=0.0,
    )
    plan = _positive_glacial_plan(system, terrain, x, z, seed=13)

    first = glacial_landform_ids(terrain, x, z, (plan,), 62.0)
    repeated = glacial_landform_ids(terrain, x, z, (plan,), 62.0)
    blocked = glacial_landform_ids(
        terrain,
        x,
        z,
        (plan,),
        62.0,
        climate_cold_weight=np.zeros(x.shape, dtype=np.float64),
    )

    assert GLACIAL_LANDFORM_PALETTE == (
        "cirque",
        "u_valley",
        "terminal_moraine",
        "drumlin_field",
    )
    assert set(np.unique(first)) == {0, 1, 2, 3, 4}
    assert np.array_equal(first, repeated), "相同 seed 必须输出完全相同的地貌 ID"
    assert not np.any(blocked), "寒带权重为零时不能泄漏冰川地貌语义"


def test_glacial_landform_ids_validate_field_shapes() -> None:
    terrain = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="must share a shape"):
        glacial_landform_ids(
            terrain,
            np.zeros((4, 3), dtype=np.float64),
            terrain,
            (),
            62.0,
        )


def test_glacial_system_carves_cirque_and_variable_width_u_valley() -> None:
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(
        name="visible_glacier",
        seed_center=Point(96.0, 96.0),
        seed_extent=32.0,
        cirque_count=1,
        cirque_radius=18.0,
        cirque_min_elevation=1.0,
        valley_count=1,
        valley_segments=6,
        valley_length=80.0,
        valley_source_width=5.0,
        valley_floor_width=2.0,
        valley_depth=18.0,
        valley_width_growth=3.0,
        moraine_height=0.0,
        drumlin_count=0,
        equilibrium_line_elevation=1.0,
        equilibrium_transition=12.0,
        flow_planning_resolution=4.0,
        flow_source_separation=8.0,
    )
    recipe = TerrainRecipe(
        name="glacial_carve",
        base_height=120.0,
        glaciers=(system,),
    )
    field = generate_heightfield(recipe, width=192, height=192, seed=13)

    assert float(np.min(field.height)) < 105.0, "冰斗或冰川谷应实际 carve 地表"
    # 在规则水平路径上直接截取两个 station，验证 U 谷宽度会随进度增长。
    from bong_worldgen.engine.glaciers.carving import carve_glacier_valley

    flat = np.full((192, 192), 120.0, dtype=np.float64)
    test_path = (Point(20.0, 96.0), Point(170.0, 96.0))
    carved = carve_glacier_valley(flat, x, z, test_path, system)
    early = carved[96 - 4 : 96 + 5, 40]
    late = carved[96 - 8 : 96 + 9, 140]
    assert float(early[4]) < float(early[0])
    assert float(early[4]) < float(early[-1])
    assert np.count_nonzero(early < 119.9) < np.count_nonzero(late < 119.9)


def test_glacial_meltwater_is_seeded_and_accumulates_along_ice_path() -> None:
    system = GlacialSystem(
        name="meltwater_test",
        seed_center=Point(96.0, 96.0),
        seed_extent=32.0,
        cirque_count=1,
        cirque_radius=18.0,
        cirque_min_elevation=1.0,
        valley_count=1,
        valley_segments=6,
        valley_length=80.0,
        valley_source_width=5.0,
        valley_floor_width=2.0,
        valley_depth=18.0,
        valley_width_growth=3.0,
        moraine_height=0.0,
        drumlin_count=0,
        meltwater_ice_thickness=14.0,
        meltwater_rate=0.01,
        meltwater_min_depth=0.2,
        meltwater_max_depth=2.0,
        flow_source_separation=4.0,
        flow_min_length=0.0,
    )
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full((192, 192), 120.0, dtype=np.float64)
    dry = np.full_like(terrain, -1.0)
    plan = _positive_glacial_plan(system, terrain, x, z, seed=17)
    first = apply_glacial_meltwater(
        terrain, dry, x, z, (plan,), sea_level=62.0
    )
    second = apply_glacial_meltwater(
        terrain, dry, x, z, (plan,), sea_level=62.0
    )

    assert GLACIAL_WATER_PALETTE == ("glacial_meltwater",)
    assert np.array_equal(first.water_id, second.water_id), "同 seed 的融水来源必须稳定"
    assert np.array_equal(first.discharge, second.discharge), "同 seed 的流量场必须稳定"
    assert np.count_nonzero(first.water_id) > 0, "冰川路径上应出现融水水体"
    assert float(np.max(first.discharge)) > 0.0, "融水水体必须携带正流量"
    path = plan.paths[0]
    _, station_x, station_z = polyline_stations(path, 32)
    sample_discharge = sample_regular_grid(first.discharge, x, z, station_x, station_z)
    assert np.all(np.diff(sample_discharge) >= -1.0e-6), "沿冰川下游流量不能减少"


def test_glacial_meltwater_does_not_lower_existing_water_or_cross_climate_gate() -> None:
    system = GlacialSystem(
        name="meltwater_gate_test",
        seed_center=Point(48.0, 48.0),
        seed_extent=20.0,
        cirque_count=1,
        valley_count=1,
        valley_segments=4,
        valley_length=50.0,
        cirque_min_elevation=1.0,
        moraine_height=0.0,
        drumlin_count=0,
        meltwater_rate=0.01,
        flow_source_separation=4.0,
        flow_min_length=0.0,
    )
    x, z = np.meshgrid(
        np.arange(96, dtype=np.float64),
        np.arange(96, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full((96, 96), 100.0, dtype=np.float64)
    existing = np.full_like(terrain, -1.0)
    existing[48, 48] = 130.0
    no_cold = np.zeros_like(terrain, dtype=np.float32)
    plan = _positive_glacial_plan(system, terrain, x, z, seed=4)
    result = apply_glacial_meltwater(
        terrain,
        existing,
        x,
        z,
        (plan,),
        sea_level=62.0,
        cold_weight=no_cold,
    )

    assert np.all(result.water_level >= existing), "融水阶段不能降低已有水面"
    assert not np.any(result.water_id), "寒带权重为零时不能误标冰川融水"


def test_glacial_surface_materials_emit_distinct_snow_ice_and_blue_ice_ids() -> None:
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(
        name="glacial_materials",
        seed_center=Point(96.0, 96.0),
        seed_extent=70.0,
        cirque_count=2,
        cirque_radius=25.0,
        cirque_min_elevation=1.0,
        valley_count=2,
        valley_segments=8,
        valley_length=120.0,
        valley_source_width=8.0,
        valley_floor_width=3.0,
        valley_width_growth=2.0,
        surface_patch_radius=40.0,
        flow_source_separation=20.0,
        flow_min_length=0.0,
    )
    terrain = np.full(x.shape, 120.0)
    plan = _positive_glacial_plan(system, terrain, x, z, seed=13)
    ids = glacial_surface_materials(terrain, x, z, (plan,), seed=13, sea_level=61.0)

    assert GLACIAL_SURFACE_PALETTE == (
        "minecraft:snow_block",
        "minecraft:powder_snow",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
        "minecraft:gravel",
    )
    unique_ids = set(np.unique(ids))
    assert 0 in unique_ids
    assert len(unique_ids - {0}) >= 4, "局部概率团块应至少混合四种寒带材质"


def test_glacial_cover_layers_unlock_from_powder_snow_to_blue_ice() -> None:
    assert GLACIAL_COVER_PALETTE == (
        "minecraft:powder_snow",
        "minecraft:snow_block",
        "minecraft:ice",
        "minecraft:blue_ice",
    )
    size = 32
    x, z = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full((size, size), 150.0, dtype=np.float64)
    material = np.ones((size, size), dtype=np.uint8)
    system = GlacialSystem(name="cover", cirque_count=0, valley_count=0)
    transition = glacial_surface_layers(
        terrain,
        x,
        z,
        (system,),
        seed=41,
        sea_level=61.0,
        climate_cold_weight=np.full((size, size), 0.10, dtype=np.float32),
        surface_material_id=material,
    )
    cold = glacial_surface_layers(
        terrain,
        x,
        z,
        (system,),
        seed=41,
        sea_level=61.0,
        climate_cold_weight=np.ones((size, size), dtype=np.float32),
        surface_material_id=material,
    )

    assert np.count_nonzero(transition[0]) > 0, "过渡带应有松雪覆盖"
    assert np.count_nonzero(transition[1:]) == 0, "低寒带权重不应提前生成冰层"
    assert np.all(np.sum(cold, axis=0) >= 4), "寒带高地应至少具备四层覆盖"
    assert np.count_nonzero(cold[3]) > 0, "完整寒带应出现蓝冰层"


def test_glacial_cover_layers_do_not_create_snow_without_a_glacial_system() -> None:
    terrain = np.full((8, 8), 150.0, dtype=np.float64)
    x, z = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64))
    layers = glacial_surface_layers(
        terrain,
        x,
        z,
        (),
        seed=1,
        sea_level=61.0,
        surface_material_id=np.zeros(terrain.shape, dtype=np.uint8),
    )
    assert not np.any(layers), "没有配置寒带系统时不能凭空生成覆盖层"


def test_glacial_cover_layers_follow_mountain_material_field() -> None:
    """群山材质启用后，寒带权重不能在山外刷出整片覆盖雪层。"""

    size = 16
    terrain = np.full((size, size), 150.0, dtype=np.float64)
    x, z = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(name="mountain_gate", cirque_count=0, valley_count=0)
    cold = np.ones((size, size), dtype=np.float32)
    no_mountain = np.zeros((size, size), dtype=np.uint8)
    mountain = np.zeros((size, size), dtype=np.uint8)
    mountain[4:12, 4:12] = 1

    outside = glacial_surface_layers(
        terrain,
        x,
        z,
        (system,),
        seed=73,
        sea_level=61.0,
        climate_cold_weight=cold,
        surface_material_id=np.ones((size, size), dtype=np.uint8),
        mountain_material_id=no_mountain,
    )
    inside = glacial_surface_layers(
        terrain,
        x,
        z,
        (system,),
        seed=73,
        sea_level=61.0,
        climate_cold_weight=cold,
        surface_material_id=np.ones((size, size), dtype=np.uint8),
        mountain_material_id=mountain,
    )

    assert not np.any(outside), "山外不能仅因 cold_weight>0 生成覆盖层"
    assert np.count_nonzero(inside) > 0, "实际群山材质应仍能生成覆盖层"

    temperate_mountain = glacial_surface_layers(
        terrain,
        x,
        z,
        (system,),
        seed=73,
        sea_level=61.0,
        climate_cold_weight=np.zeros((size, size), dtype=np.float32),
        surface_material_id=np.ones((size, size), dtype=np.uint8),
        mountain_material_id=mountain,
    )
    assert np.count_nonzero(temperate_mountain) > 0, (
        "群山材质场在温带侧仍应保留最低覆盖层，不能被气候边界整行截断"
    )


def test_glacial_cover_layers_leave_rock_exposure_uncovered() -> None:
    size = 16
    terrain = np.full((size, size), 180.0, dtype=np.float64)
    x, z = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    mountain = np.ones((size, size), dtype=np.uint8)
    mountain[6:10, 6:10] = MOUNTAIN_SURFACE_PALETTE.index("minecraft:stone") + 1
    cover = glacial_surface_layers(
        terrain,
        x,
        z,
        (GlacialSystem(name="rock_gate", cirque_count=0, valley_count=0),),
        seed=73,
        sea_level=61.0,
        climate_cold_weight=np.ones((size, size), dtype=np.float32),
        surface_material_id=np.ones((size, size), dtype=np.uint8),
        mountain_material_id=mountain,
    )

    assert not np.any(cover[:, 6:10, 6:10]), "裸岩基底不能再次被雪冰覆盖"
    assert np.any(cover[:, :6, :]), "非裸岩群山仍应保留覆盖层"


def test_generated_mountain_rock_is_visible_and_not_replaced_by_permafrost() -> None:
    recipe = replace(
        DEFAULT_RECIPE,
        mountains=(
            MountainRange(
                path=(Point(0.0, 32.0), Point(63.0, 32.0)),
                width=28.0,
                height=300.0,
                base_elevation=72.0,
                summit_elevation=380.0,
                slope_power=1.1,
                spine_warp_strength=0.0,
                width_variation=0.0,
            ),
        ),
    )
    field = generate_heightfield(recipe, width=64, height=64, seed=1847)
    rock_start = 1 + len(GLACIAL_SURFACE_PALETTE)
    rock = field.mountain_material_id >= rock_start
    assert np.any(rock), "陡峭山脊应存在裸岩材质列"
    assert np.all(field.permafrost_id[rock] == 0), "冻土不能覆盖裸岩列"
    visible_names = np.asarray(field.surface_visible_palette)
    visible_rock = np.isin(field.surface_visible_id, [
        int(np.flatnonzero(visible_names == name)[0]) + 1
        for name in (
            "minecraft:stone",
            "minecraft:andesite",
            "minecraft:calcite",
            "minecraft:tuff",
            "minecraft:deepslate",
        )
        if np.any(visible_names == name)
    ])
    assert np.any(visible_rock & rock), "裸岩必须进入最终可见地表层"


def test_visible_surface_layer_uses_one_stable_precedence_order() -> None:
    terrain = np.full((2, 2), 140.0, dtype=np.float64)
    water = np.full((2, 2), -1.0, dtype=np.float64)
    water[1, 1] = 141.0
    riverbed = np.full((2, 2), -1, dtype=np.int16)
    riverbed[0, 1] = 0
    surface_material = np.zeros((2, 2), dtype=np.uint8)
    surface_material[0, 1] = 1
    permafrost = np.zeros((2, 2), dtype=np.uint8)
    cover = np.zeros((4, 2, 2), dtype=np.uint8)
    cover[:, 0, 0] = 1
    cover[:, 1, 0] = 1

    visible, palette = build_visible_surface_layer(
        terrain,
        water,
        sea_level=61.0,
        riverbed_id=riverbed,
        riverbed_palette=("mud",),
        surface_material_id=surface_material,
        surface_material_palette=("minecraft:snow_block",),
        permafrost_id=permafrost,
        permafrost_palette=(),
        surface_cover_layers=cover,
        surface_cover_palette=(
            "minecraft:powder_snow",
            "minecraft:snow_block",
            "minecraft:ice",
            "minecraft:blue_ice",
        ),
        surface_columns=np.asarray([[True, True], [False, True]]),
    )
    names = np.full(visible.shape, "", dtype=object)
    non_empty = visible > 0
    names[non_empty] = np.asarray(palette)[visible[non_empty] - 1]

    assert names[0, 0] == "minecraft:powder_snow", "覆盖层顶部必须优先于基底"
    assert visible[1, 0] == 0, "洞口列的最终可见地表应为空气"
    assert names[0, 1] == "minecraft:mud", "河床材质应优先于寒带基底材质"
    assert names[1, 1] == "minecraft:gravel", "水面列不能被覆盖层或冰雪材质替换"


def test_generated_heightfield_exposes_stable_visible_surface_palette() -> None:
    field = generate_heightfield(
        DEFAULT_RECIPE,
        width=16,
        height=16,
        seed=812731,
        origin_x=0.0,
        origin_z=-4096.0,
        cell_size=4.0,
    )

    assert field.surface_visible_palette
    assert field.surface_visible_id.shape == field.height.shape
    assert np.all(field.surface_visible_id <= len(field.surface_visible_palette))


def test_permafrost_is_seeded_mixed_and_limited_to_cold_dry_system_domain() -> None:
    size = 128
    x, z = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full((size, size), 130.0, dtype=np.float64)
    cold = np.ones((size, size), dtype=np.float32)
    water = np.full((size, size), -1.0, dtype=np.float32)
    water[52:60, :] = 132.0
    protected = np.zeros((size, size), dtype=np.uint8)
    protected[68:76, 68:76] = 1
    system = GlacialSystem(
        name="permafrost_contract",
        seed_center=Point(64.0, 64.0),
        seed_extent=52.0,
        cirque_count=0,
        valley_count=0,
        permafrost_min_cold_weight=0.10,
        permafrost_full_cold_weight=0.20,
        permafrost_coverage=1.0,
        permafrost_elevation_start=0.0,
        permafrost_elevation_range=1.0,
        permafrost_patch_scale=18.0,
        permafrost_material_scale=10.0,
    )
    kwargs = {
        "terrain": terrain,
        "x": x,
        "z": z,
        "systems": (system,),
        "sea_level": 61.0,
        "cold_weight": cold,
        "water_level": water,
        "existing_surface_material_id": protected,
        "existing_surface_material_palette": ("minecraft:packed_ice",),
    }

    first = permafrost_surface_materials(seed=19, **kwargs)
    repeated = permafrost_surface_materials(seed=19, **kwargs)
    changed = permafrost_surface_materials(seed=20, **kwargs)
    domain = np.hypot(x - 64.0, z - 64.0) < 52.0

    assert PERMAFROST_PALETTE == (
        "minecraft:powder_snow",
        "minecraft:snow_block",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:gravel",
        "minecraft:coarse_dirt",
        "minecraft:dirt",
        "minecraft:stone",
    )
    assert np.array_equal(first, repeated), "同一 seed 必须生成完全相同的冻土"
    assert not np.array_equal(first, changed), "不同 seed 应改变冻土斑块和材质"
    assert not np.any(first[~domain]), "冻土不能越过冰川系统的生成域"
    assert not np.any(first[water >= 0.0]), "水体列不能被冻土材质覆盖"
    assert not np.any(first[protected == 1]), "已有冰川核心材质必须保留"
    assert len(set(np.unique(first)) - {0}) >= 5, "中等密度冻土应混合多种方块材质"

    no_cold = permafrost_surface_materials(
        seed=19,
        **{**kwargs, "cold_weight": np.zeros_like(cold)},
    )
    assert not np.any(no_cold), "非寒带区域不能生成冻土"


def test_permafrost_material_fields_are_chunk_stable_without_grid_aligned_boundaries() -> None:
    size = 128
    origin_x = -96
    origin_z = -160
    x, z = np.meshgrid(
        origin_x + np.arange(size, dtype=np.float64),
        origin_z + np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.full((size, size), 150.0, dtype=np.float64)
    cold = np.ones((size, size), dtype=np.float32)
    water = np.full((size, size), -1.0, dtype=np.float32)
    system = GlacialSystem(
        name="continuous_permafrost",
        seed_center=Point(-32.0, -96.0),
        seed_extent=180.0,
        cirque_count=0,
        valley_count=0,
        permafrost_min_cold_weight=0.0,
        permafrost_full_cold_weight=0.2,
        permafrost_coverage=1.0,
        permafrost_elevation_start=0.0,
        permafrost_elevation_range=1.0,
        permafrost_patch_scale=64.0,
        permafrost_material_scale=18.0,
    )

    whole = permafrost_surface_materials(
        terrain,
        x,
        z,
        (system,),
        seed=91,
        sea_level=61.0,
        cold_weight=cold,
        water_level=water,
    )
    halves = []
    for column_slice in (slice(0, 64), slice(64, 128)):
        halves.append(
            permafrost_surface_materials(
                terrain[:, column_slice],
                x[:, column_slice],
                z[:, column_slice],
                (system,),
                seed=91,
                sea_level=61.0,
                cold_weight=cold[:, column_slice],
                water_level=water[:, column_slice],
            )
        )

    assert np.array_equal(whole, np.concatenate(halves, axis=1)), (
        "冻土材质必须只依赖世界坐标和 seed，不能在 tile 边界产生接缝"
    )
    horizontal_changes = np.argwhere(whole[:, 1:] != whole[:, :-1])[:, 1] + origin_x + 1
    vertical_changes = np.argwhere(whole[1:, :] != whole[:-1, :])[:, 0] + origin_z + 1
    assert horizontal_changes.size > 0 and vertical_changes.size > 0
    assert np.count_nonzero(horizontal_changes % 18) > 0
    assert np.count_nonzero(vertical_changes % 18) > 0
    assert len(set(np.unique(whole)) - {0}) >= 5


@pytest.mark.parametrize(
    "full_weight",
    (0.10, 1.01),
)
def test_permafrost_full_cold_weight_must_follow_start_threshold(full_weight: float) -> None:
    with pytest.raises(ValueError, match="glacial system parameters are out of range"):
        GlacialSystem(
            name="invalid_permafrost_gate",
            permafrost_min_cold_weight=0.10,
            permafrost_full_cold_weight=full_weight,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("prevailing_wind_angle_degrees", float("nan")),
        ("sun_facing_angle_degrees", float("inf")),
        ("equilibrium_line_elevation", -1.0),
        ("equilibrium_transition", 0.0),
        ("snowfall_rate", -0.1),
        ("snowfall_rate", float("inf")),
        ("snowfall_variability", 1.01),
        ("windward_accumulation_strength", -0.1),
        ("leeward_drift_strength", -0.1),
        ("mass_balance_slope_scale", 0.0),
        ("ablation_rate", -0.1),
        ("minimum_ablation", -0.1),
        ("low_elevation_ablation_strength", -0.1),
        ("solar_ablation_strength", -0.1),
        ("transition_ablation_strength", -0.1),
        ("flow_planning_resolution", 0.0),
        ("flow_source_separation", 0.0),
        ("flow_inertia", 1.0),
        ("flow_meander_strength", 1.01),
        ("flow_direction_samples", 2),
        ("flow_direction_samples", 4),
        ("flow_merge_distance", -0.1),
        ("flow_uphill_tolerance", -0.1),
        ("flow_min_length", -0.1),
        ("flow_termination_balance", float("inf")),
    ),
)
def test_glacial_mass_balance_configuration_rejects_invalid_values(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match="glacial system parameters are out of range"):
        GlacialSystem(name="invalid_mass_balance", **{field_name: invalid_value})


def test_glacial_material_fade_is_probabilistic_not_a_hard_transition_line() -> None:
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.full(4, 96.0, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(
        name="climate_fade",
        seed_center=Point(96.0, 96.0),
        seed_extent=70.0,
        cirque_count=2,
        cirque_radius=25.0,
        cirque_min_elevation=1.0,
        valley_count=2,
        valley_segments=8,
        valley_length=120.0,
        valley_source_width=8.0,
        valley_floor_width=3.0,
        valley_width_growth=2.0,
        surface_patch_radius=40.0,
        flow_source_separation=20.0,
        flow_min_length=0.0,
    )
    planning_x, planning_z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    plan = _positive_glacial_plan(
        system,
        np.full(planning_x.shape, 120.0),
        planning_x,
        planning_z,
        seed=13,
    )
    weights = np.repeat(
        np.asarray([[0.0], [0.1], [0.2], [1.0]], dtype=np.float32),
        x.shape[1],
        axis=1,
    )
    ids = glacial_surface_materials(
        np.full(x.shape, 120.0),
        x,
        z,
        (plan,),
        seed=13,
        sea_level=61.0,
        climate_cold_weight=weights,
    )

    counts = np.count_nonzero(ids, axis=1)
    assert counts[0] == 0
    assert 0 < counts[1] <= counts[2] == counts[3]
    assert counts[1] < ids.shape[1], "过渡带同一行必须同时存在保留与淡出区域"
    assert counts[1] <= counts[2] <= counts[3]


def test_glacial_material_fade_keeps_low_weight_transition_visible() -> None:
    x, z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.full(4, 96.0, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(
        name="climate_fade_curve",
        seed_center=Point(96.0, 96.0),
        seed_extent=70.0,
        cirque_count=2,
        cirque_radius=25.0,
        cirque_min_elevation=1.0,
        valley_count=2,
        valley_segments=8,
        valley_length=120.0,
        valley_source_width=8.0,
        valley_floor_width=3.0,
        valley_width_growth=2.0,
        climate_fade_full_weight=0.20,
        flow_source_separation=20.0,
        flow_min_length=0.0,
    )
    planning_x, planning_z = np.meshgrid(
        np.arange(192, dtype=np.float64),
        np.arange(192, dtype=np.float64),
        indexing="xy",
    )
    plan = _positive_glacial_plan(
        system,
        np.full(planning_x.shape, 120.0),
        planning_x,
        planning_z,
        seed=13,
    )
    weights = np.repeat(
        np.asarray([[0.0], [0.1], [0.2], [1.0]], dtype=np.float32),
        x.shape[1],
        axis=1,
    )
    ids = glacial_surface_materials(
        np.full(x.shape, 120.0),
        x,
        z,
        (plan,),
        seed=13,
        sea_level=61.0,
        climate_cold_weight=weights,
    )

    counts = np.count_nonzero(ids, axis=1)
    assert counts[0] == 0
    assert counts[1] > 0, "低权重过渡带仍应保留可见冰雪材质"
    assert counts[1] < counts[2] == counts[3]


def test_glacial_surface_protrusions_raise_at_most_one_block_with_same_material() -> None:
    x, z = np.meshgrid(
        np.arange(96, dtype=np.float64),
        np.arange(96, dtype=np.float64),
        indexing="xy",
    )
    system = GlacialSystem(
        name="surface_protrusions",
        cirque_count=0,
        valley_count=0,
        surface_patch_radius=24.0,
        surface_protrusion_probability=0.25,
    )
    terrain = np.full(x.shape, 120.0, dtype=np.float64)
    materials = np.full(x.shape, 3, dtype=np.uint8)
    raised_terrain, raised = apply_surface_protrusions(
        terrain, x, z, materials, (system,), seed=33, sea_level=61.0
    )

    delta = raised_terrain - terrain
    assert np.any(raised), "概率凸起应在寒带材质表面产生少量方块"
    assert np.all((delta == 0.0) | (delta == 1.0))
    assert np.array_equal(materials, np.full(x.shape, 3, dtype=np.uint8))


def test_freeze_thaw_transfers_cliff_material_without_changing_total_height() -> None:
    x, z = np.meshgrid(
        np.arange(32, dtype=np.float64),
        np.arange(32, dtype=np.float64),
        indexing="xy",
    )
    terrain = np.zeros((32, 32), dtype=np.float64)
    terrain[:, 12:20] = 40.0
    system = GlacialSystem(
        name="freeze_thaw",
        cirque_count=0,
        valley_count=0,
        freeze_thaw_iterations=4,
        freeze_thaw_strength=0.25,
        talus_slope_threshold=2.0,
    )
    exposure = np.ones(terrain.shape, dtype=bool)
    result = apply_freeze_thaw(terrain, x, z, exposure, system, seed=99)

    assert result[:, 12].mean() < terrain[:, 12].mean(), "冻融应削低陡峭高侧"
    assert result[:, 11].mean() > terrain[:, 11].mean(), "冻融应把岩屑沉积到低侧"
    assert np.isclose(float(result.sum()), float(terrain.sum())), "冻融转移不应凭空增减地形"


def test_glacial_deposits_raise_terminal_moraine_and_lowland_drumlins() -> None:
    path = (Point(20.0, 40.0), Point(108.0, 40.0))
    system = GlacialSystem(
        name="deposits",
        seed_center=Point(64.0, 64.0),
        seed_extent=48.0,
        cirque_count=0,
        valley_count=1,
        moraine_height=12.0,
        moraine_width=10.0,
        moraine_length=8.0,
        drumlin_count=8,
        drumlin_height=7.0,
        drumlin_max_elevation=20.0,
    )
    x, z = np.meshgrid(
        np.arange(128, dtype=np.float64),
        np.arange(128, dtype=np.float64),
        indexing="xy",
    )
    base = np.full((128, 128), 70.0, dtype=np.float64)
    from bong_worldgen.engine.glaciers.deposits import deposit_drumlins, deposit_terminal_moraine

    moraine = deposit_terminal_moraine(base, x, z, path, system)
    drumlins = deposit_drumlins(base, x, z, system, seed=8, sea_level=60.0)
    assert float(np.max(moraine - base)) > 10.0
    assert float(np.max(drumlins - base)) > 0.0


def test_shared_randomness_is_stable_and_bounded() -> None:
    values = [unit_interval(812731, index) for index in range(32)]

    assert values == [unit_interval(812731, index) for index in range(32)]
    assert all(0.0 <= value < 1.0 for value in values)
    assert mix64(812731) == mix64(812731)
    assert mix64(812731) != mix64(812732)


def test_text_seed_is_stable_and_order_sensitive() -> None:
    assert stable_text_seed("cold_glacier") == stable_text_seed("cold_glacier")
    assert stable_text_seed("cold_glacier") != stable_text_seed("glacier_cold")
    assert stable_text_seed("") == 0


def test_shared_grid_sampler_interpolates_and_clamps_edges() -> None:
    values = np.asarray([[0.0, 10.0], [20.0, 30.0]])
    grid_x, grid_z = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="xy")
    sample_x = np.asarray([-1.0, 0.5, 2.0])
    sample_z = np.asarray([0.5, 0.5, 0.5])

    result = sample_regular_grid(values, grid_x, grid_z, sample_x, sample_z)

    assert np.allclose(result, [10.0, 15.0, 20.0])


def test_same_seed_is_byte_deterministic() -> None:
    first = generate_heightfield(DEFAULT_RECIPE, width=48, height=40, seed=812731)
    second = generate_heightfield(DEFAULT_RECIPE, width=48, height=40, seed=812731)
    assert np.array_equal(first.height, second.height)
    assert np.array_equal(first.moisture, second.moisture)
    assert np.array_equal(first.water_level, second.water_level)


def test_seed_changes_noise_without_changing_shape() -> None:
    first = generate_heightfield(DEFAULT_RECIPE, width=32, height=32, seed=1)
    second = generate_heightfield(DEFAULT_RECIPE, width=32, height=32, seed=2)
    assert first.height.shape == second.height.shape
    assert not np.array_equal(first.height, second.height)


def test_domain_warp_noise_is_finite_and_seeded() -> None:
    recipe = TerrainRecipe(
        name="warp",
        base_noise=(
            NoiseLayer(kind="warp", scale=64.0, warp_scale=180.0, warp_strength=40.0),
        ),
    )
    field = generate_heightfield(recipe, width=24, height=24, seed=5)
    assert np.isfinite(field.height).all()
    assert not np.array_equal(
        field.height,
        generate_heightfield(recipe, width=24, height=24, seed=6).height,
    )


def test_3d_noise_is_seeded_and_varies_along_vertical_axis() -> None:
    x = np.full((4, 4), 12.0)
    z = np.full((4, 4), -8.0)
    layer = NoiseLayer(kind="fbm", scale=24.0, octaves=2)
    low = sample_noise_3d(x, np.full((4, 4), 50.0), z, layer, seed=7)
    high = sample_noise_3d(x, np.full((4, 4), 62.0), z, layer, seed=7)
    repeat = sample_noise_3d(x, np.full((4, 4), 50.0), z, layer, seed=7)

    assert np.array_equal(low, repeat)
    assert not np.array_equal(low, high)
    assert np.isfinite(low).all() and np.isfinite(high).all()


def test_godot_style_worm_field_is_seeded_bounded_and_varied() -> None:
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(32, dtype=np.float64),
        indexing="xy",
    )
    network = CaveNetwork(
        name="worm",
        paths=((Point(0, 16), Point(47, 16)),),
        worm_threshold=0.55,
        dead_end_strength=1.15,
        vertical_warp=3.0,
    )

    first = sample_worm_field(x, z, network, seed=19)
    repeat = sample_worm_field(x, z, network, seed=19)
    other = sample_worm_field(x, z, network, seed=20)

    assert np.array_equal(first.corridor, repeat.corridor)
    assert not np.array_equal(first.corridor, other.corridor)
    assert np.all((first.corridor >= 0.0) & (first.corridor <= 1.0))
    assert np.all((first.dead_end >= 0.0) & (first.dead_end <= 1.0))
    assert np.all(first.radius_scale > 0.0)
    assert float(np.ptp(first.radius_scale)) > 0.0
    assert float(np.ptp(first.vertical_offset)) > 0.0
    assert float(np.ptp(first.dead_end)) > 0.0


def test_cave_domain_warp_changes_the_sdf_without_breaking_determinism() -> None:
    common = dict(
        name="warped",
        paths=((Point(0, 16), Point(63, 16)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=3.0,
        height=6,
        depth=10.0,
    )
    straight_recipe = TerrainRecipe(
        name="straight_cave",
        caves=(CaveNetwork(**common, domain_warp_strength=0.0),),
    )
    warped_recipe = TerrainRecipe(
        name="warped_cave",
        caves=(CaveNetwork(**common, domain_warp_strength=8.0),),
    )

    straight = generate_heightfield(straight_recipe, width=64, height=32, seed=10)
    warped = generate_heightfield(warped_recipe, width=64, height=32, seed=10)
    repeat = generate_heightfield(warped_recipe, width=64, height=32, seed=10)

    assert not np.array_equal(straight.cave_id, warped.cave_id)
    assert np.array_equal(warped.cave_id, repeat.cave_id)


def test_polyline_river_carves_and_widens() -> None:
    recipe = TerrainRecipe(
        name="river",
        base_height=80.0,
        rivers=(River((Point(0, 0), Point(30, 0)), width=2.0, depth=8.0, widening=4.0),),
    )
    field = generate_heightfield(recipe, width=40, height=8, seed=4, origin_x=0, origin_z=-3)
    center = field.height[:, 3]
    assert np.min(center) < 74.0
    assert np.count_nonzero(field.water_level >= 0) > 0


def test_solid_ores_replace_stone_outside_cavities() -> None:
    spec = SolidOreSpec(
        material="coal_ore",
        rarity="多",
        cluster_count=4,
        min_depth=4.0,
        max_depth=20.0,
        vein_length=7,
    )
    recipe = TerrainRecipe(name="solid_ore_only", base_height=80.0, solid_ores=(spec,))
    first = generate_heightfield(recipe, width=128, height=128, seed=4)
    repeat = generate_heightfield(recipe, width=128, height=128, seed=4)

    assert first.underground_blocks == repeat.underground_blocks
    assert first.underground_blocks
    assert all(block.source == "solid_ore" for block in first.underground_blocks)
    assert all(block.resource_id == "solid:coal_ore" for block in first.underground_blocks)
    surface = np.rint(first.height).astype(np.int32)
    assert all(
        SPAN_MIN_Y < block.y < int(surface[block.z, block.x])
        for block in first.underground_blocks
    )
    positions = {(block.x, block.y, block.z) for block in first.underground_blocks}
    assert any(
        (x + dx, y + dy, z + dz) in positions
        for x, y, z in positions
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    )


def test_solid_ores_skip_generated_cave_voids() -> None:
    cave = CaveNetwork(
        name="ore_exclusion_cave",
        paths=((Point(0, 64), Point(127, 64)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=3.0,
        height=6,
        depth=20.0,
    )
    ore = SolidOreSpec(
        material="iron_ore",
        rarity="多",
        cluster_count=12,
        min_depth=10.0,
        max_depth=28.0,
        vein_length=6,
    )
    field = generate_heightfield(
        TerrainRecipe(name="ore_exclusion", base_height=80.0, caves=(cave,), solid_ores=(ore,)),
        width=128,
        height=128,
        seed=4,
    )
    solid_blocks = [block for block in field.underground_blocks if block.source == "solid_ore"]

    assert solid_blocks
    assert all(
        any(
            int(span[0]) <= block.y <= int(span[1])
            for span in field.solid_spans[block.z, block.x]
            if int(span[0]) != 32767
        )
        for block in solid_blocks
    ), "普通矿物必须位于最终实心 spans，而不能落入洞穴空腔"


def test_river_bed_materials_are_seeded_and_exposed() -> None:
    recipe = TerrainRecipe(
        name="riverbed_materials",
        base_height=80.0,
        rivers=(
            River(
                (Point(0, 4), Point(63, 4)),
                width=3.0,
                depth=8.0,
                bed_materials=("minecraft:mud", "sand", "clay"),
            ),
        ),
    )
    first = generate_heightfield(recipe, width=64, height=12, seed=23, origin_z=-2)
    second = generate_heightfield(recipe, width=64, height=12, seed=23, origin_z=-2)

    assert first.riverbed_palette == ("mud", "sand", "clay")
    assert np.array_equal(first.riverbed_id, second.riverbed_id)
    wet_bed = first.riverbed_id >= 0
    assert np.count_nonzero(wet_bed) > 0
    assert np.all(np.isin(first.riverbed_id[wet_bed], np.arange(3)))


def test_standalone_biome_ids_match_declared_two_entry_palette() -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=48, height=48, seed=812731)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)

    assert set(np.unique(tile.biome_id)).issubset({0, 1})


def test_riverbed_buffer_covers_one_extra_grid_cell() -> None:
    recipe = TerrainRecipe(
        name="riverbed_buffer",
        base_height=80.0,
        rivers=(
            River(
                (Point(0, 4), Point(31, 4)),
                width=2.0,
                depth=4.0,
                bed_materials=("sand",),
            ),
        ),
    )
    field = generate_heightfield(recipe, width=32, height=12, seed=4, origin_z=0)

    # The channel radius is two cells, but the riverbed palette intentionally
    # extends one cell farther to avoid an uncovered border after voxelization.
    assert field.riverbed_id[0, 4] == -1
    assert field.riverbed_id[1, 4] >= 0
    assert field.riverbed_id[2, 4] >= 0
    assert field.riverbed_id[3, 4] >= 0


def test_shallow_cave_network_emits_a_connected_air_gap() -> None:
    recipe = TerrainRecipe(
        name="shallow_caves",
        base_height=80.0,
        caves=(
            CaveNetwork(
                name="main",
                paths=((Point(0, 6), Point(31, 6)),),
                width=2.0,
                height=4,
                depth=9.0,
            ),
        ),
    )
    field = generate_heightfield(recipe, width=32, height=16, seed=4, origin_z=0)

    column = field.solid_spans[6, 16]
    spans = [tuple(int(value) for value in span) for span in column if span[0] != 32767]
    assert len(spans) == 2
    assert spans[0][1] == 80
    assert spans[1][0] == -64
    assert spans[1][1] < spans[0][0] - 1
    assert field.cave_palette == ("main",)
    assert np.all(field.cave_id[field.cave_id > 0] == 1)


def test_cave_topology_is_seeded_and_not_fixed_to_anchor_paths() -> None:
    network = CaveNetwork(
        name="topology",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=6,
        branch_segments=4,
        chamber_count=3,
        entrance_count=2,
    )

    first = generate_cave_topology(network, seed=41)
    repeat = generate_cave_topology(network, seed=41)
    other = generate_cave_topology(network, seed=42)

    assert first == repeat
    assert len(first.paths) == 1 + network.branch_count
    assert len(first.chambers) == network.chamber_count
    assert len(first.entrances) == network.entrance_count
    assert first.paths != other.paths


def test_cave_topology_generates_random_paths_when_recipe_has_no_paths() -> None:
    network = CaveNetwork(
        name="seeded_topology",
        paths=(),
        seed_center=Point(-100.0, 80.0),
        seed_extent=240.0,
        seed_path_count=4,
        seed_path_segments=10,
        seed_path_length=320.0,
        seed_path_turn=0.85,
        branch_count=2,
    )

    first = generate_cave_topology(network, seed=41)
    repeat = generate_cave_topology(network, seed=41)
    other = generate_cave_topology(network, seed=42)

    assert first == repeat
    assert first != other
    assert len(first.paths) == network.seed_path_count + network.branch_count
    assert all(len(path) == network.seed_path_segments + 1 for path in first.paths[: network.seed_path_count])

    # 随机游走不是由一条常量直线复制出来的，至少一条路径存在连续转向。
    assert any(
        any(
            abs(
                (b.x - a.x) * (c.z - b.z)
                - (b.z - a.z) * (c.x - b.x)
            )
            > 1.0e-6
            for a, b, c in zip(first_path, first_path[1:], first_path[2:])
        )
        for first_path in first.paths[: network.seed_path_count]
    )


def test_default_underground_networks_are_seed_distributed_across_world_cells() -> None:
    network = CaveNetwork(
        name="distributed",
        seed_center=Point(0.0, 0.0),
        seed_extent=96.0,
        seed_path_count=2,
        seed_path_segments=6,
        seed_path_length=180.0,
        branch_count=1,
    )
    x_axis = np.arange(-2400.0, 2401.0)
    z_axis = np.arange(-2400.0, 2401.0)
    first = list(iter_world_network_instances(network, x_axis, z_axis, 41, 0))
    repeat = list(iter_world_network_instances(network, x_axis, z_axis, 41, 0))
    other = list(iter_world_network_instances(network, x_axis, z_axis, 42, 0))

    assert first == repeat
    centers = {(round(instance.seed_center.x), round(instance.seed_center.z)) for instance, _ in first}
    assert len(centers) >= 4, "默认地下网络应在多个世界网格中生成，而不是绑定单一区域"
    assert first != other, "世界 seed 改变后，地下网络实例位置和子 seed 必须变化"


def test_cave_entrance_reaches_surface_without_opening_everywhere() -> None:
    network = CaveNetwork(
        name="entrance",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=1,
        width=2.0,
        height=4,
        depth=9.0,
    )
    recipe = TerrainRecipe(name="entrance", base_height=80.0, caves=(network,))
    topology = generate_cave_topology(network, seed=4)
    entrance = topology.entrances[0]
    entrance_x = round(entrance.x)
    entrance_z = round(entrance.z)

    field = generate_heightfield(recipe, width=32, height=20, seed=4)
    entrance_spans = field.solid_spans[entrance_z, entrance_x]
    assert field.cave_id[entrance_z, entrance_x] == 1
    assert int(entrance_spans[0, 1]) < int(round(field.height[entrance_z, entrance_x]))
    assert np.count_nonzero(field.cave_id) < field.cave_id.size


def test_cave_generation_is_identical_when_split_into_tiles() -> None:
    network = CaveNetwork(
        name="tile_stable",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=4,
        branch_segments=3,
        chamber_count=2,
        entrance_count=1,
    )
    recipe = TerrainRecipe(name="tile_stable", base_height=80.0, caves=(network,))
    full = generate_heightfield(recipe, width=32, height=20, seed=9, origin_x=0, origin_z=0)
    left = generate_heightfield(recipe, width=16, height=20, seed=9, origin_x=0, origin_z=0)
    right = generate_heightfield(recipe, width=16, height=20, seed=9, origin_x=16, origin_z=0)

    assert np.array_equal(full.height, np.concatenate((left.height, right.height), axis=1))
    assert np.array_equal(full.water_level, np.concatenate((left.water_level, right.water_level), axis=1))
    assert np.array_equal(full.cave_id, np.concatenate((left.cave_id, right.cave_id), axis=1))
    assert np.array_equal(
        full.solid_spans,
        np.concatenate((left.solid_spans, right.solid_spans), axis=1),
    )


def test_shallow_cave_depth_is_in_requested_band() -> None:
    depth = CaveNetwork(name="depth", paths=((Point(0, 0), Point(1, 0)),)).depth
    assert 50.0 <= depth <= 60.0
    assert DEFAULT_RECIPE.caves[0].depth == 56.0


def test_cave_vertical_tails_reach_surface_and_sink_below_main_level() -> None:
    network = CaveNetwork(
        name="vertical_tails",
        paths=((Point(0, 64), Point(255, 64)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=4.0,
        height=6,
        depth=56.0,
        noise_strength=0.0,
        dead_end_strength=0.0,
        domain_warp_strength=0.0,
        vertical_warp_noise=NoiseLayer(kind="fbm", scale=80.0, octaves=2, gain=0.55),
    )
    field = generate_heightfield(
        TerrainRecipe(name="vertical_tails", base_height=80.0, caves=(network,)),
        width=256,
        height=128,
        seed=5,
    )
    surface = np.rint(field.height).astype(np.int16)
    midpoints: list[float] = []
    surface_open_columns = 0
    for z_index, x_index in zip(*np.where(field.cave_id > 0)):
        spans = [
            tuple(int(value) for value in span)
            for span in field.solid_spans[z_index, x_index]
            if span[0] != 32767
        ]
        if spans and spans[0][1] < surface[z_index, x_index]:
            surface_open_columns += 1
        if len(spans) >= 2:
            midpoints.append(
                (spans[0][0] + spans[1][1]) * 0.5 - surface[z_index, x_index]
            )

    assert surface_open_columns > 0, "上抬尾部必须至少形成一处通向地表的洞口"
    assert min(midpoints) < -60.0, "下沉尾部必须低于主洞约 56 格的中心深度"


def test_cave_emits_real_minecraft_placeholder_blocks() -> None:
    network = CaveNetwork(
        name="placeholder",
        paths=((Point(0, 6), Point(31, 6)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=2.0,
        height=6,
        depth=56.0,
        placeholder_count=16,
    )
    field = generate_heightfield(
        TerrainRecipe(name="placeholder", base_height=80.0, caves=(network,)),
        width=32,
        height=16,
        seed=4,
    )

    materials = {block.material for block in field.underground_blocks}
    assert materials & {"minecraft:coal_ore", "minecraft:iron_ore", "minecraft:copper_ore"}
    assert "minecraft:glow_lichen" in materials
    assert all(block.y > -64 for block in field.underground_blocks)


def test_cave_resources_have_rarity_classes_and_seeded_veins() -> None:
    network = CaveNetwork(
        name="resource_rarity",
        paths=((Point(0, 6), Point(63, 6)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=2.0,
        height=6,
        depth=56.0,
        placeholder_count=16,
    )
    field = generate_heightfield(
        TerrainRecipe(name="resource_rarity", base_height=80.0, caves=(network,)),
        width=64,
        height=16,
        seed=4,
    )

    assert CAVE_RARITY_MULTIPLIERS == {"少": 0.5, "中": 1.0, "多": 2.0}
    assert network.placeholder_density == "中"
    assert network.placeholder_rarities == ("多", "中", "少", "多")
    by_material: dict[str, set[tuple[int, int, int]]] = {}
    for block in field.underground_blocks:
        by_material.setdefault(block.material, set()).add((block.x, block.y, block.z))

    assert {
        "minecraft:coal_ore",
        "minecraft:iron_ore",
        "minecraft:copper_ore",
        "minecraft:glow_lichen",
    } <= by_material.keys()
    assert all(block.source == "cave" for block in field.underground_blocks)
    assert all(block.resource_id.startswith("cave:") for block in field.underground_blocks)
    for material, positions in by_material.items():
        assert any(
            (x + dx, y + dy, z + dz) in positions
            for x, y, z in positions
            for dx, dy, dz in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            )
        ), f"{material} 应形成至少两格相邻的矿脉/植物簇"
    assert len(by_material["minecraft:coal_ore"]) > len(by_material["minecraft:copper_ore"])


def test_underground_hydrology_emits_seeded_river_and_lake_blocks() -> None:
    network = UndergroundRiverNetwork(
        name="independent_underground_water",
        seed_center=Point(16, 8),
        seed_extent=20,
        path_count=1,
        path_segments=8,
        path_length=30,
        branch_count=0,
        lake_count=1,
        lake_radius=8.0,
        height=4,
        water_depth=2,
        lake_depth=2,
    )
    recipe = TerrainRecipe(
        name="underground_water",
        base_height=80.0,
        underground_rivers=(network,),
    )
    first = generate_heightfield(recipe, width=32, height=16, seed=3)
    second = generate_heightfield(recipe, width=32, height=16, seed=3)

    assert first.underground_water_blocks == second.underground_water_blocks
    kinds = {block.kind for block in first.underground_water_blocks}
    assert {"river", "lake"} <= kinds
    assert all(block.flowing == (block.kind == "river") for block in first.underground_water_blocks)
    river_positions = {
        (block.x, block.y, block.z)
        for block in first.underground_water_blocks
        if block.kind == "river"
    }
    assert any(
        (x + dx, y + dy, z + dz) in river_positions
        for x, y, z in river_positions
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    surface_by_position = {
        (x, z): float(first.height[z, x])
        for z in range(first.height.shape[0])
        for x in range(first.height.shape[1])
    }
    assert all(
        block.y < surface_by_position[(block.x, block.z)]
        for block in first.underground_water_blocks
        if (block.x, block.z) in surface_by_position
    )


def test_underground_river_emits_seeded_ore_veins_and_wet_plants() -> None:
    network = UndergroundRiverNetwork(
        name="river_resources",
        seed_center=Point(16, 8),
        seed_extent=20,
        path_count=1,
        branch_count=0,
        lake_count=1,
        lake_radius=8.0,
        height=5,
        source_depth=12.0,
        outlet_depth=18.0,
        width=3.0,
        water_depth=2,
        resource_density="中",
        ore_cluster_count=20,
        plant_cluster_count=20,
    )
    recipe = TerrainRecipe(
        name="river_resources",
        base_height=80.0,
        underground_rivers=(network,),
    )
    first = generate_heightfield(recipe, width=32, height=16, seed=23)
    repeat = generate_heightfield(recipe, width=32, height=16, seed=23)

    assert first.underground_blocks == repeat.underground_blocks
    by_material: dict[str, set[tuple[int, int, int]]] = {}
    for block in first.underground_blocks:
        by_material.setdefault(block.material, set()).add((block.x, block.y, block.z))
    assert {
        "minecraft:coal_ore",
        "minecraft:iron_ore",
        "minecraft:copper_ore",
    } <= by_material.keys()
    assert "minecraft:glow_lichen" in by_material
    assert all(block.source == "underground_river" for block in first.underground_blocks)
    assert all(
        block.resource_id.startswith("underground_river:")
        for block in first.underground_blocks
    )
    assert {block.rarity for block in first.underground_blocks} <= {"少", "中", "多"}
    water_positions = {
        (block.x, block.y, block.z) for block in first.underground_water_blocks
    }
    assert first.underground_blocks
    assert all(
        (x, y, z) not in water_positions
        for blocks in by_material.values()
        for x, y, z in blocks
    )
    assert any(
        (x + dx, y + dy, z + dz) in by_material["minecraft:coal_ore"]
        for x, y, z in by_material["minecraft:coal_ore"]
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    ), "煤矿应形成相连矿脉"


def test_independent_underground_river_does_not_require_cave_network() -> None:
    river = UndergroundRiverNetwork(
        name="river_only",
        seed_center=Point(16, 8),
        seed_extent=20,
        path_count=1,
        path_segments=8,
        path_length=30,
        branch_count=0,
        lake_count=0,
        height=4,
        water_depth=2,
    )
    field = generate_heightfield(
        TerrainRecipe(
            name="river_only",
            base_height=80.0,
            underground_rivers=(river,),
        ),
        width=32,
        height=16,
        seed=4,
    )

    assert np.count_nonzero(field.cave_id) == 0
    assert field.underground_water_blocks
    assert all(block.kind == "river" for block in field.underground_water_blocks)
    assert any(
        span[0] != 32767
        for column in field.solid_spans.reshape(-1, 4, 2)
        for span in column[1:]
    ), "独立地下河必须切出自己的地下空腔，而不是只放水块"


def test_underground_river_does_not_change_regular_cave_ids() -> None:
    cave = CaveNetwork(
        name="regular_cave",
        paths=((Point(0, 8), Point(31, 8)),),
        seed_center=Point(16, 8),
        seed_extent=40.0,
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=2.5,
        height=5,
        depth=20.0,
        domain_warp_strength=0.0,
    )
    river = UndergroundRiverNetwork(
        name="separate_river",
        seed_center=Point(16, 8),
        seed_extent=40.0,
        path_count=1,
        path_segments=8,
        path_length=30.0,
        branch_count=2,
        lake_count=1,
        source_depth=18.0,
        outlet_depth=30.0,
        height=4,
        water_depth=2,
    )
    with_river = generate_heightfield(
        TerrainRecipe(
            name="cave_and_river",
            base_height=80.0,
            caves=(cave,),
            underground_rivers=(river,),
        ),
        width=32,
        height=16,
        seed=17,
    )
    without_river = generate_heightfield(
        replace(
            TerrainRecipe(
                name="cave_and_river",
                base_height=80.0,
                caves=(cave,),
                underground_rivers=(river,),
            ),
            underground_rivers=(),
        ),
        width=32,
        height=16,
        seed=17,
    )

    assert np.count_nonzero(with_river.cave_id) > 0
    assert np.array_equal(
        with_river.cave_id,
        without_river.cave_id,
    ), "独立地下河不能改变普通洞穴的 cave_id 分布"


def test_underground_river_requires_downstream_depth() -> None:
    with pytest.raises(ValueError, match="outlet_depth"):
        UndergroundRiverNetwork(
            name="uphill_water",
            source_depth=60.0,
            outlet_depth=40.0,
        )


def test_zero_isoline_band_is_bounded_centered_and_rejects_invalid_width() -> None:
    field = np.array([-1.0, -0.08, 0.0, 0.08, 1.0], dtype=np.float64)
    band = zero_isoline_band(field, 0.16)

    assert band.shape == field.shape
    assert np.all((band >= 0.0) & (band <= 1.0))
    assert band[2] == pytest.approx(1.0)
    assert band[1] == pytest.approx(band[3])
    assert band[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="zero-isoline width"):
        zero_isoline_band(field, 0.0)


def test_underground_river_zero_isoline_changes_routes_deterministically() -> None:
    network = UndergroundRiverNetwork(
        name="zero_isoline_routes",
        seed_center=Point(24, 24),
        seed_extent=28,
        inlet_count=4,
        outlet_count=1,
        lake_count=0,
        height=4,
        river_chance=1.0,
        fracture_bias=0.9,
        fracture_isoline_width=0.12,
    )
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(48, dtype=np.float64),
        indexing="xy",
    )
    offsets = np.arange(-70, 0, dtype=np.int16)
    terrain = np.full((48, 48), 80.0, dtype=np.float64)
    first = generate_underground_rivers(terrain, x, z, offsets, network, seed=91)
    repeat = generate_underground_rivers(terrain, x, z, offsets, network, seed=91)
    other = generate_underground_rivers(terrain, x, z, offsets, network, seed=92)

    assert first.paths == repeat.paths
    assert first.paths != other.paths
    assert first.paths


def test_underground_river_network_has_seeded_inlets_outlets_and_paths() -> None:
    network = UndergroundRiverNetwork(
        name="geodesic_network",
        seed_center=Point(24, 24),
        seed_extent=28,
        inlet_count=4,
        outlet_count=1,
        path_count=1,
        branch_count=0,
        network_iterations=2,
        lake_count=0,
        height=4,
        river_chance=1.0,
    )
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(48, dtype=np.float64),
        indexing="xy",
    )
    offsets = np.arange(-70, 0, dtype=np.int16)
    terrain = np.full((48, 48), 80.0, dtype=np.float64)
    first = generate_underground_rivers(terrain, x, z, offsets, network, seed=17)
    repeat = generate_underground_rivers(terrain, x, z, offsets, network, seed=17)
    other = generate_underground_rivers(terrain, x, z, offsets, network, seed=18)

    assert first.inlets == repeat.inlets
    assert first.outlets == repeat.outlets
    assert first.paths == repeat.paths
    assert first.inlets != other.inlets or first.outlets != other.outlets
    assert len(first.inlets) == 4
    assert len(first.outlets) == 1
    assert first.paths
    assert not first.unreachable_inlets
    assert first.fracture_void is not None
    assert np.count_nonzero(first.fracture_void) > 0
    assert not np.array_equal(first.void, first.fracture_void)
    outside = generate_underground_rivers(
        terrain,
        x + 1_000.0,
        z + 1_000.0,
        offsets,
        network,
        seed=17,
    )
    assert outside.fracture_void is not None
    assert np.count_nonzero(outside.fracture_void) == 0
    for path in first.paths:
        assert min(
            np.hypot(path[0].x - inlet.x, path[0].z - inlet.z) for inlet in first.inlets
        ) <= 2.0
        assert min(
            np.hypot(path[-1].x - outlet.x, path[-1].z - outlet.z) for outlet in first.outlets
        ) <= 2.0


def test_fracture_is_a_narrow_vertical_seam_not_a_cave_tunnel() -> None:
    network = UndergroundRiverNetwork(
        name="tectonic_seam",
        seed_center=Point(24, 24),
        seed_extent=28,
        inlet_count=3,
        outlet_count=1,
        lake_count=0,
        height=4,
        fracture_width=1.2,
        fracture_height=28,
    )
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(48, dtype=np.float64),
        indexing="xy",
    )
    offsets = np.arange(-70, 0, dtype=np.int16)
    terrain = np.full((48, 48), 80.0, dtype=np.float64)
    result = generate_underground_rivers(terrain, x, z, offsets, network, seed=123)

    assert result.fracture_void is not None
    vertical_occupancy = np.count_nonzero(result.fracture_void, axis=0)
    occupied = vertical_occupancy[vertical_occupancy > 0]
    assert occupied.size > 0
    # 裂隙中段应明显高于普通洞穴的 4 格高度；端点允许自然渐缩。
    assert np.percentile(occupied, 50) >= 12
    assert int(occupied.max()) <= network.fracture_height
    # 横向截面仍然保持一条窄缝，不能整体铺成矿洞。
    assert np.count_nonzero(result.fracture_void) < np.count_nonzero(result.void) * 2


def test_underground_river_iterations_create_shared_conduits_and_lake_nodes() -> None:
    common = dict(
        name="merged_network",
        seed_center=Point(24, 24),
        seed_extent=28,
        inlet_count=5,
        outlet_count=1,
        lake_count=2,
        lake_radius=5.0,
        height=4,
        river_chance=1.0,
    )
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(48, dtype=np.float64),
        indexing="xy",
    )
    offsets = np.arange(-70, 0, dtype=np.int16)
    terrain = np.full((48, 48), 80.0, dtype=np.float64)
    one_round = generate_underground_rivers(
        terrain,
        x,
        z,
        offsets,
        UndergroundRiverNetwork(**common, network_iterations=1),
        seed=29,
    )
    multi_round = generate_underground_rivers(
        terrain,
        x,
        z,
        offsets,
        UndergroundRiverNetwork(**common, network_iterations=3),
        seed=29,
    )

    assert len(multi_round.junctions) > 0
    assert len(multi_round.junctions) >= len(one_round.junctions)
    lake_positions = {
        (block.x, block.z)
        for block in multi_round.water_blocks
        if block.kind == "lake"
    }
    assert lake_positions
    assert any(
        min(np.hypot(x - junction.x, z - junction.z) for junction in multi_round.junctions) <= 8.0
        for x, z in lake_positions
    )


def test_river_bed_materials_reject_empty_or_duplicate_names() -> None:
    with pytest.raises(ValueError, match="at least one"):
        River((Point(0, 0), Point(1, 0)), width=1.0, depth=1.0, bed_materials=())
    with pytest.raises(ValueError, match="duplicates"):
        River(
            (Point(0, 0), Point(1, 0)),
            width=1.0,
            depth=1.0,
            bed_materials=("sand", "minecraft:sand"),
        )


def test_river_waterline_stays_below_the_lowest_bank() -> None:
    recipe = TerrainRecipe(
        name="contained_river",
        base_height=80.0,
        base_noise=(NoiseLayer(kind="fbm", scale=12.0, amplitude=8.0, octaves=2),),
        rivers=(River((Point(0, 4), Point(47, 4)), width=2.0, depth=8.0),),
    )
    field = generate_heightfield(recipe, width=48, height=12, seed=9, origin_x=0, origin_z=-2)
    surface = np.rint(field.height).astype(np.int16)
    water_top = np.ceil(field.water_level).astype(np.int16)
    wet = field.water_level >= 0.0
    padded_surface = np.pad(surface, 1, mode="edge")
    padded_wet = np.pad(wet, 1, mode="edge")
    for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour_surface = padded_surface[
            1 + dz : 1 + dz + surface.shape[0], 1 + dx : 1 + dx + surface.shape[1]
        ]
        neighbour_wet = padded_wet[
            1 + dz : 1 + dz + surface.shape[0], 1 + dx : 1 + dx + surface.shape[1]
        ]
        # For every dry bank adjacent to water, the water top cannot be higher
        # than that bank's top block.
        assert np.all(water_top[wet & ~neighbour_wet] <= neighbour_surface[wet & ~neighbour_wet])
    assert np.max((water_top - surface)[wet]) >= 2
    center_water = field.water_level[6]
    center_wet = center_water >= 0.0
    assert np.all(np.diff(center_water[center_wet]) <= 0.0)


def test_river_water_surface_is_flat_across_each_cross_section() -> None:
    recipe = TerrainRecipe(
        name="flat_cross_section",
        base_height=80.0,
        rivers=(River((Point(0, 0), Point(47, 0)), width=3.0, depth=8.0),),
    )
    field = generate_heightfield(recipe, width=48, height=11, seed=1, origin_z=-5)

    for x_index in (10, 20, 30):
        cross_section_water = field.water_level[:, x_index]
        wet = cross_section_water >= 0.0
        assert np.count_nonzero(wet) >= 3
        assert float(np.ptp(cross_section_water[wet])) == 0.0
        cross_section_height = field.height[:, x_index]
        assert float(cross_section_height[5]) < float(cross_section_height[3])
        assert float(cross_section_height[5]) < float(cross_section_height[7])


def test_mountain_and_basin_are_visible_in_field() -> None:
    recipe = TerrainRecipe(
        name="relief",
        base_height=70.0,
        basins=(Basin(Point(10, 10), 12, 12, 8),),
        mountains=(MountainRange((Point(-10, 0), Point(30, 0)), width=5, height=50),),
    )
    field = generate_heightfield(recipe, width=40, height=30, seed=3, origin_x=-10, origin_z=-10)
    assert float(np.max(field.height)) - float(np.min(field.height)) > 20.0


def test_absolute_mountain_profile_reaches_configured_summit() -> None:
    """绝对高度山脉必须把峰顶标定到配方值，而不是只增加噪声振幅。"""

    recipe = TerrainRecipe(
        name="absolute_mountain",
        base_height=70.0,
        mountains=(
            MountainRange(
                (Point(0.0, 64.0), Point(127.0, 64.0)),
                width=16.0,
                height=0.0,
                base_elevation=72.0,
                summit_elevation=360.0,
            ),
        ),
    )
    field = generate_heightfield(recipe, width=128, height=128, seed=11)

    assert float(np.max(field.height)) == pytest.approx(360.0)
    # 有限支持山体在宽度之外必须保留原始 Y=70；base_elevation 只用于
    # 山体内部的高度标定，不能把整张地图抬到 Y=72。
    assert float(field.height[0, 0]) == pytest.approx(70.0)
    assert float(np.max(field.height)) - float(np.min(field.height)) == pytest.approx(290.0)


def test_mountain_flank_carving_is_bounded() -> None:
    with pytest.raises(ValueError, match="flank_carving"):
        MountainRange((Point(0, 0), Point(10, 0)), width=5, height=20, flank_carving=1.1)


def test_warped_canyon_distance_is_seeded_and_not_a_straight_slot() -> None:
    x, z = np.meshgrid(
        np.arange(96, dtype=np.float64),
        np.arange(96, dtype=np.float64),
        indexing="xy",
    )
    path = (Point(4.0, 48.0), Point(92.0, 48.0))
    straight = Canyon(name="straight", paths=(path,), domain_warp_strength=0.0)
    warped = Canyon(
        name="warped",
        paths=(path,),
        domain_warp_scale=28.0,
        domain_warp_strength=18.0,
    )
    first_distance, first_progress = warped_polyline_distance_and_progress(
        x, z, path, warped, seed=73
    )
    repeat_distance, repeat_progress = warped_polyline_distance_and_progress(
        x, z, path, warped, seed=73
    )
    straight_distance, _ = warped_polyline_distance_and_progress(
        x, z, path, straight, seed=73
    )

    assert np.array_equal(first_distance, repeat_distance)
    assert np.array_equal(first_progress, repeat_progress)
    assert not np.array_equal(first_distance, straight_distance)
    assert float(np.ptp(first_distance[:, 48])) > 0.0


def test_large_canyon_is_carve_only_and_has_deep_flat_floor() -> None:
    canyon = Canyon(
        name="grand",
        paths=((Point(4.0, 32.0), Point(59.0, 32.0)),),
        width=16.0,
        floor_width=4.0,
        depth=36.0,
        bank_width=4.0,
        domain_warp_strength=0.0,
    )
    field = generate_heightfield(
        TerrainRecipe(name="grand_canyon", base_height=100.0, canyons=(canyon,)),
        width=64,
        height=64,
        seed=19,
    )
    center = field.height[32, 32]
    rim = field.height[32, 4]
    outside = field.height[4, 32]

    assert float(center) <= 64.0, "峡谷中心应至少向下切出配置深度"
    assert float(rim) <= 100.0, "峡谷只能 carve，不能抬高原地形"
    assert float(outside) == pytest.approx(100.0), "影响带外的平原应保持不变"
    assert float(field.height[32, 28]) == pytest.approx(float(center)), (
        "底部平坦区应比两岸宽度更稳定"
    )


def test_canyon_path_and_carve_do_not_depend_on_surface_height() -> None:
    canyon = Canyon(
        name="height_free_start",
        paths=((Point(0.0, 20.0), Point(63.0, 20.0)),),
        width=10.0,
        floor_width=3.0,
        depth=24.0,
        bank_width=3.0,
        domain_warp_strength=0.0,
    )
    x, z = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="xy",
    )
    low = np.full((64, 64), 40.0, dtype=np.float64)
    high = np.full((64, 64), 220.0, dtype=np.float64)
    low_carved = apply_canyons(low, x, z, (canyon,), seed=31)
    high_carved = apply_canyons(high, x, z, (canyon,), seed=31)
    low_delta = low - low_carved
    high_delta = high - high_carved

    assert np.allclose(low_delta, high_delta), "峡谷路径不能根据地表高度决定是否生成"
    assert np.count_nonzero(low_delta) > 0


def test_bong_adapter_uses_expected_dtypes_and_shapes() -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=16, height=12, seed=7)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)
    assert tile.height.dtype == np.float32
    assert tile.surface_id.dtype == np.uint8
    assert tile.biome_id.dtype == np.uint8
    assert tile.feature_mask.dtype == np.float32
    assert tile.boundary_weight.dtype == np.float32
    assert tile.wilderness_id.dtype == np.uint8
    assert tile.glacial_landform_id.dtype == np.uint8
    assert tile.glacial_landform_palette == GLACIAL_LANDFORM_PALETTE
    assert tile.glacial_water_id.dtype == np.uint8
    assert tile.glacial_discharge.dtype == np.float32
    assert tile.valley_depth.dtype == np.float32
    assert tile.valley_flow_accumulation.dtype == np.float32
    assert tile.valley_stream_power.dtype == np.float32
    assert tile.snow_accumulation.dtype == np.float32
    assert tile.glacial_mass_balance.dtype == np.float32
    assert tile.surface_material_id.dtype == np.uint8
    assert tile.surface_material_palette == GLACIAL_SURFACE_PALETTE
    assert tile.mountain_rock_exposure.dtype == np.float32
    assert tile.permafrost_id.dtype == np.uint8
    assert tile.permafrost_palette == PERMAFROST_PALETTE
    assert tile.surface_cover_layers.shape == (4, 12, 16)
    assert tile.surface_cover_palette == (
        "minecraft:powder_snow",
        "minecraft:snow_block",
        "minecraft:ice",
        "minecraft:blue_ice",
    )
    assert tile.height.flags.c_contiguous


def test_bong_raster_writer_emits_v2_core_files(tmp_path) -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=16, height=16, seed=7)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)
    manifest_path = write_bong_raster(tile, tmp_path, tile_x=-1, tile_z=2)
    manifest = manifest_path.read_text(encoding="utf-8")
    tile_dir = tmp_path / "tile_-1_2"
    assert '"version": 2' in manifest
    assert (tile_dir / "spans_count.bin").stat().st_size == 16 * 16
    assert (tile_dir / "spans.bin").stat().st_size == 16 * 16 * 16
    assert (tile_dir / "wilderness_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "riverbed_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "cave_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "fracture_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "glacial_landform_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "glacial_water_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "glacial_discharge.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "uplift.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "valley_depth.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "valley_flow_accumulation.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "valley_stream_power.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "snow_accumulation.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "glacial_mass_balance.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "surface_material_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "mountain_material_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "mountain_weight.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "mountain_snowline.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "mountain_material_score.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "mountain_slope_angle.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "mountain_exposure.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "permafrost_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "surface_cover_layers.bin").stat().st_size == 4 * 16 * 16
    assert (tile_dir / "underground_water.bin").stat().st_size % 16 == 0
    assert (tile_dir / "underground_resources.bin").stat().st_size % 16 == 0
    assert '"wilderness_palette"' in manifest
    assert '"riverbed_palette"' in manifest
    assert '"cave_palette"' in manifest
    assert '"fracture_palette"' in manifest
    assert '"fracture_encoding"' in manifest
    assert '"surface_material_palette"' in manifest
    assert '"mountain_material_palette"' in manifest
    assert '"mountain_material_encoding"' in manifest
    assert '"permafrost_palette"' in manifest
    assert '"permafrost_encoding"' in manifest
    assert '"surface_cover_palette"' in manifest
    assert '"surface_cover_encoding"' in manifest
    assert '"glacial_landform_encoding"' in manifest
    assert '"glacial_water_encoding"' in manifest
    assert '"valley_hydrology_encoding"' in manifest
    assert '"glacial_mass_balance_encoding"' in manifest
    assert '"underground_water_encoding"' in manifest
    assert '"underground_resources_encoding"' in manifest
    parsed = json.loads(manifest)
    assert parsed["permafrost_encoding"] == {
        "file": "permafrost_id.bin",
        "none": 0,
        "dtype": "u8",
    }
    assert parsed["glacial_landform_palette"] == list(GLACIAL_LANDFORM_PALETTE)
    assert "glacial_landform_id" in parsed["tiles"][0]["layers"]
    assert "snow_accumulation" in parsed["tiles"][0]["layers"]
    assert "glacial_mass_balance" in parsed["tiles"][0]["layers"]
    assert "valley_depth" in parsed["tiles"][0]["layers"]
    assert "valley_flow_accumulation" in parsed["tiles"][0]["layers"]
    assert "valley_stream_power" in parsed["tiles"][0]["layers"]
    assert parsed["glacial_mass_balance_encoding"]["mass_balance_sign"] == {
        "positive": "accumulation",
        "negative": "ablation",
    }
    assert "permafrost_id" in parsed["tiles"][0]["layers"]


def test_cli_npz_keeps_glacial_semantic_layers(tmp_path) -> None:
    from bong_worldgen.cli import main

    output = tmp_path / "terrain.npz"
    assert (
        main(["--width", "8", "--height", "8", "--seed", "7", "--output", str(output)])
        == 0
    )

    with np.load(output) as generated:
        assert generated["glacial_landform_id"].shape == (8, 8)
        assert (
            tuple(generated["glacial_landform_palette"].tolist())
            == GLACIAL_LANDFORM_PALETTE
        )
        assert generated["glacial_water_id"].shape == (8, 8)
        assert generated["uplift"].shape == (8, 8)
        assert generated["snow_accumulation"].shape == (8, 8)
        assert generated["glacial_mass_balance"].shape == (8, 8)
        assert generated["surface_cover_layers"].shape == (4, 8, 8)
        assert generated["permafrost_id"].shape == (8, 8)


def test_bong_raster_keeps_logical_resource_ids_separate_from_materials(tmp_path) -> None:
    field = replace(
        generate_heightfield(TerrainRecipe(name="resource_ids"), width=16, height=16, seed=7),
        underground_blocks=(
            UndergroundBlock(
                x=3,
                y=35,
                z=3,
                material="minecraft:iron_ore",
                resource_id="cave:iron_ore",
                source="cave",
                rarity="中",
            ),
            UndergroundBlock(
                x=4,
                y=35,
                z=3,
                material="minecraft:iron_ore",
                resource_id="underground_river:iron_ore",
                source="underground_river",
                rarity="多",
            ),
        ),
    )
    manifest_path = write_bong_raster(
        to_bong_tile(field, sea_level=61.0),
        tmp_path,
        tile_x=0,
        tile_z=0,
    )
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["underground_resource_palette"]) == {
        "cave:iron_ore",
        "underground_river:iron_ore",
    }
    catalog = {
        entry["resource_id"]: entry for entry in manifest["underground_resource_catalog"]
    }
    assert catalog["cave:iron_ore"]["material"] == "minecraft:iron_ore"
    assert catalog["underground_river:iron_ore"]["rarity"] == "多"
    assert (tmp_path / "tile_0_0" / "underground_resources.bin").stat().st_size == 32


def test_preview_world_manifest_is_self_describing(tmp_path) -> None:
    manifest_path = export_preview_world(
        tmp_path,
        recipe=TerrainRecipe(name="test_world"),
        seed=7,
        min_x=-16,
        max_x=15,
        min_z=-16,
        max_z=15,
        tile_size=16,
    )
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["spans_encoding"]["bytes_per_column"] == 16
    assert manifest["overview"]["cell_size"] == 32
    assert manifest["overview"]["surface_file"] == "overview_surface_id.bin"
    assert manifest["overview"]["permafrost_file"] == "overview_permafrost_id.bin"
    assert (
        manifest["overview"]["glacial_landform_file"]
        == "overview_glacial_landform_id.bin"
    )
    assert manifest["overview"]["uplift_file"] == "overview_uplift.bin"
    assert (tmp_path / "rasters" / "overview_surface_id.bin").stat().st_size == 1
    assert (tmp_path / "rasters" / "overview_permafrost_id.bin").stat().st_size == 1
    assert (tmp_path / "rasters" / "overview_glacial_landform_id.bin").stat().st_size == 1
    assert (tmp_path / "rasters" / "overview_snow_accumulation.bin").stat().st_size == 4
    assert (tmp_path / "rasters" / "overview_glacial_mass_balance.bin").stat().st_size == 4
    assert (tmp_path / "rasters" / "overview_uplift.bin").stat().st_size == 4
    assert manifest["climate_palette"] == ["cold", "temperate", "tropical"]
    assert manifest["climate_encoding"]["none"] == 0
    assert manifest["climate_transition_encoding"]["weight_dtype"] == "f32"
    assert manifest["climate_surface_encoding"]["none"] == 0
    assert "climate_id" in manifest["semantic_layers"]
    assert "climate_transition_weight" in manifest["semantic_layers"]
    assert manifest["wilderness_palette"][1]["key"] == "mountains"
    assert manifest["underground_water_encoding"]["record_size"] == 16
    assert manifest["underground_resources_encoding"]["record_size"] == 16
    assert "underground_water" in manifest["semantic_layers"]
    assert "underground_resources" in manifest["semantic_layers"]
    assert "cave_id" in manifest["semantic_layers"]
    assert "fracture_id" in manifest["semantic_layers"]
    assert "glacial_water_id" in manifest["semantic_layers"]
    assert "glacial_landform_id" in manifest["semantic_layers"]
    assert "glacial_discharge" in manifest["semantic_layers"]
    assert "snow_accumulation" in manifest["semantic_layers"]
    assert "glacial_mass_balance" in manifest["semantic_layers"]
    assert "valley_depth" in manifest["semantic_layers"]
    assert "valley_flow_accumulation" in manifest["semantic_layers"]
    assert "valley_stream_power" in manifest["semantic_layers"]
    assert "uplift" in manifest["semantic_layers"]
    assert manifest["uplift_encoding"]["file"] == "uplift.bin"
    assert manifest["overview"]["snow_accumulation_file"] == (
        "overview_snow_accumulation.bin"
    )
    assert manifest["overview"]["glacial_mass_balance_file"] == (
        "overview_glacial_mass_balance.bin"
    )
    assert "surface_cover_layers" in manifest["tiles"][0]["layers"]
    assert "uplift" in manifest["tiles"][0]["layers"]
    assert manifest["surface_cover_encoding"]["shape"] == [4, "height", "width"]
    assert "hanging_valley_id" not in manifest["semantic_layers"]
    assert "waterfall_id" not in manifest["semantic_layers"]
    assert "waterfall_blocks" not in manifest["semantic_layers"]
    assert "permafrost_id" in manifest["semantic_layers"]
    assert manifest["permafrost_encoding"] == {
        "file": "permafrost_id.bin",
        "none": 0,
        "dtype": "u8",
    }
    assert "hanging_valley_encoding" not in manifest
    assert "waterfall_encoding" not in manifest
    assert "waterfall_blocks_encoding" not in manifest
    assert not (tmp_path / "rasters" / "tile_0_0" / "waterfall_blocks.bin").exists()
    assert "cave_id" in manifest["tiles"][0]["layers"]
    assert "fracture_id" in manifest["tiles"][0]["layers"]
    assert len(manifest["tiles"]) == 4
    assert all(tile["spans"] for tile in manifest["tiles"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 8},
        {"width": 8, "height": 0},
    ],
)
def test_invalid_dimensions_fail_fast(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        generate_heightfield(DEFAULT_RECIPE, seed=1, **kwargs)


def test_wilderness_classifier_exposes_stable_categories() -> None:
    height = np.full((5, 5), 70.0, dtype=np.float32)
    water = np.full((5, 5), -1.0, dtype=np.float32)
    water[0, 0] = 61.0
    water[1, 1] = 66.0
    height[3, 3] = 130.0

    wilderness = classify_wilderness(height, water, sea_level=61.0)

    assert int(wilderness[2, 2]) == GRASSLAND
    assert int(wilderness[0, 0]) == LAKE
    assert int(wilderness[1, 1]) == RIVER
    assert int(wilderness[3, 3]) == MOUNTAINS


def test_wilderness_classifier_accepts_single_column_fields() -> None:
    wilderness = classify_wilderness(
        np.asarray([[70.0]], dtype=np.float32),
        np.asarray([[-1.0]], dtype=np.float32),
        sea_level=61.0,
    )
    assert wilderness.tolist() == [[GRASSLAND]]
