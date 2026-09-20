from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneIndex, ZoneTerrain
from bong_worldgen.composition.layout import boundary_alpha
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SPAWN
from bong_worldgen.engine import Point, River, TerrainRecipe


@pytest.mark.parametrize("mode", ("soft", "semi_hard", "hard"))
def test_boundary_is_monotone_bounded_and_continuous_with_zero_endpoint_slopes(mode):
    zone = replace(SPAWN, boundary_mode=mode, boundary_width=100)
    distances = np.linspace(-101, 101, 20201)
    weights = boundary_alpha(zone, distances)
    assert weights[0] == 0 and weights[-1] == 1
    assert weights[len(weights) // 2] == pytest.approx(0.5)
    assert np.all(np.diff(weights) >= -1e-12)
    assert np.max(np.diff(weights)) < 0.001
    active = np.flatnonzero((weights > 1e-12) & (weights < 1 - 1e-12))
    assert weights[active[0]] < 1e-7
    assert 1 - weights[active[-1]] < 1e-7


def test_boundary_mode_and_width_control_actual_transition_extent():
    inside = np.array([20.0])
    values = [boundary_alpha(replace(SPAWN, boundary_mode=mode, boundary_width=100), inside)[0]
              for mode in ("soft", "semi_hard", "hard")]
    assert 0.5 < values[0] < values[1] < values[2] < 1
    assert boundary_alpha(replace(SPAWN, boundary_width=200), inside)[0] < values[0]
    step = boundary_alpha(replace(SPAWN, boundary_width=0), np.array([-0.001, 0, 0.001]))
    assert step.tolist() == [0, 1, 1]


def test_overlapping_blends_conserve_weight_without_source_order_dependence():
    small = replace(SPAWN, name="small", size_x=200, size_z=200, center_x=100)
    grid = np.linspace(-900, 900, 1000)
    first = ZoneIndex((SPAWN, small)).query(grid, 0)
    second = ZoneIndex((small, SPAWN)).query(grid, 0)
    total = first.background.copy()
    for part in first.contributions:
        assert np.all(part.weight >= 0)
        np.testing.assert_array_equal(part.weight, second.weight_for(part.zone.name))
        total += part.weight
    np.testing.assert_allclose(total, 1, atol=1e-14)


@pytest.mark.parametrize("mode", ("soft", "semi_hard", "hard"))
def test_boundary_heights_have_no_cutoff_wall(mode):
    zone = replace(SPAWN, size_x=600, size_z=600, boundary_mode=mode, boundary_width=128)
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), background=TerrainRecipe(name="flat", base_height=120))
    field = composer.generate(width=601, height=1, origin_x=0)
    assert 65 < field.height[0, 0] < 75
    assert field.height[0, -1] == 120
    assert np.abs(np.diff(field.height[0])).max() < 1.6


def test_river_crossing_zone_and_tile_edges_matches_whole_world_even_at_single_column():
    zone = replace(SPAWN, center_x=-30, center_z=0, size_x=100, size_z=100, boundary_width=48)
    background = TerrainRecipe(
        name="river_crossing", base_height=110,
        rivers=(River(path=(Point(-500, 0), Point(500, 0)), width=4, depth=5),),
    )
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), background=background, seed=9)
    full = composer.generate(width=128, height=16, origin_x=-64, origin_z=-8)
    for origin, width in ((-64, 17), (-47, 46), (-1, 1), (0, 64)):
        part = composer.generate(width=width, height=16, origin_x=origin, origin_z=-8)
        for layer in ("height", "moisture", "water_level", "riverbed_id", "solid_spans", "cave_id"):
            np.testing.assert_array_equal(
                getattr(part, layer), getattr(full, layer)[:, origin + 64:origin + 64 + width],
            )
    wet = full.water_level[8] >= 0
    assert wet.any()
    assert np.all(np.diff(full.water_level[8, wet]) <= 0)
    # Querying at a coarser stride must still sample the same physical river.
    coarse = composer.generate(width=32, height=4, origin_x=-64, origin_z=-8, cell_size=4)
    for layer in ("height", "water_level", "riverbed_id", "solid_spans"):
        np.testing.assert_array_equal(getattr(coarse, layer), getattr(full, layer)[::4, ::4])
