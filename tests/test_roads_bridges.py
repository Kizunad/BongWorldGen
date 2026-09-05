"""跨水寻路和真实桥梁方块的回归测试。"""

from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.engine.bridges import bridge_blocks, plan_bridge
from bong_worldgen.engine.road_terrain import RoadTerrain
from bong_worldgen.engine.rural_roads import RoadPlan, _smooth_land, plan_rural_road
from bong_worldgen.engine.terrain_config import Point, TownSettings
from bong_worldgen.engine.town import _emit_road


def river_scene(size=48, step=1):
    axis = np.arange(size, dtype=np.float64) * step
    terrain = np.full((size, size), 80.0)
    water = np.full_like(terrain, -1.0)
    wet = (axis >= 20) & (axis <= 25)
    terrain[:, wet] = 74.0
    water[:, wet] = 80.0
    return terrain, water, axis


def settings(**kwargs):
    return replace(TownSettings(
        road_wander=0.0, road_path_step=3, road_search_margin=12,
        road_width=3, bridge_max_span=12,
    ), **kwargs)


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_cross_river_plan_has_dry_anchors_level_deck_and_accessible_ramps(width):
    terrain, water, axis = river_scene()
    config = settings(road_width=width)
    plan = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis, config, 5)
    assert plan.path[0] == Point(8, 24)
    assert plan.path[-1] == Point(40, 24)
    assert len(plan.bridges) == 1
    bridge = plan.bridges[0]
    assert set(bridge.deck[bridge.wet_start:bridge.wet_end + 1]) == {81}
    assert max(abs(a - b) for a, b in zip(bridge.deck, bridge.deck[1:])) <= 1
    assert bridge.deck[0] == bridge.deck[-1] == 80
    assert bridge.width == width
    assert plan == plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis, config, 5)


def test_no_bridge_or_overlong_crossing_is_unreachable_not_a_fallback_line():
    terrain, water, axis = river_scene()
    for config in (settings(bridge_max_span=0), settings(bridge_max_span=4)):
        plan = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis, config, 9)
        assert plan == RoadPlan()


def test_one_block_water_between_coarse_nodes_cannot_be_skipped():
    terrain, water, axis = river_scene()
    terrain[:] = 80
    water[:] = -1
    water[:, 22] = 80
    terrain[:, 22] = 76
    blocked = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis,
                             settings(bridge_max_span=0, road_path_step=5), 2)
    assert blocked == RoadPlan()
    bridged = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis,
                             settings(road_path_step=5), 2)
    assert len(bridged.bridges) == 1


def test_span_limit_is_in_world_blocks_not_search_cells():
    terrain, water, axis = river_scene(size=24, step=2)
    plan = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis,
                          settings(bridge_max_span=3), 9)
    assert plan == RoadPlan()


def test_bridge_emission_preserves_river_and_uses_full_width_supported_ends():
    terrain, water, axis = river_scene()
    original_terrain, original_water = terrain.copy(), water.copy()
    config = settings(road_materials=("dirt", "gravel", "coarse_dirt"),
                      road_material_weights=(0.5, 0.3, 0.2))
    blocks = {}
    path = _emit_road(blocks, Point(8, 24), Point(40, 24), terrain, water,
                      axis, axis, config, 6, width=5)
    assert path
    decks = [b for b in blocks.values() if b.kind == "bridge_deck"]
    rails = [b for b in blocks.values() if b.kind == "bridge_rail"]
    supports = [b for b in blocks.values() if b.kind == "bridge_support"]
    assert decks and rails and supports
    assert {b.material for b in decks} == {"oak_planks"}
    assert all(b.material.startswith("minecraft:oak_fence[") for b in rails)
    # 正中湿区桥面有 5 格行走宽度，另外 2 格是两侧护栏位置。
    deck_row = [b for b in decks if b.x == 22]
    assert len(deck_row) == 7
    assert len({b.y for b in deck_row}) == 1
    center_z = sorted(b.z for b in deck_row)[3]
    assert not any(b.kind == "bridge_support" and b.z == center_z and 20 <= b.x <= 25
                   for b in blocks.values())
    for b in decks:
        if water[b.z, b.x] >= 0:
            assert b.y > water[b.z, b.x]
    assert np.array_equal(terrain, original_terrain)
    assert np.array_equal(water, original_water)


def test_bridge_rejects_steep_bank_deep_ravine_diagonal_and_missing_bank():
    terrain, water, axis = river_scene()
    config = settings()
    grid = RoadTerrain(terrain, water, axis, axis, 3)
    assert plan_bridge(Point(16, 24), Point(30, 24), grid, config) is not None
    assert plan_bridge(Point(16, 24), Point(30, 30), grid, config) is None
    assert plan_bridge(Point(16, 24), Point(23, 24), grid, config) is None
    assert plan_bridge(Point(16, 0), Point(30, 0), grid, config) is None
    terrain[:, 20:26] = 10
    assert plan_bridge(Point(16, 24), Point(30, 24), grid, config) is None
    terrain[:, 20:26] = 74
    terrain[:, 26:] = 95
    assert plan_bridge(Point(16, 24), Point(30, 24), grid, config) is None


def test_water_level_uses_highest_surface_not_bed_or_center_sample():
    terrain, water, axis = river_scene()
    water[22, 23] = 81.4
    grid = RoadTerrain(terrain, water, axis, axis, 3)
    bridge = plan_bridge(Point(12, 24), Point(35, 24), grid, settings())
    assert bridge is not None
    assert set(bridge.deck[bridge.wet_start:bridge.wet_end + 1]) == {83}


def test_smoothed_route_cannot_cut_through_a_thin_obstacle():
    axis = np.arange(32, dtype=float)
    terrain = np.zeros((32, 32))
    water = np.full_like(terrain, -1)
    water[6:24, 10:16] = 0
    grid = RoadTerrain(terrain, water, axis, axis, 1)
    original = RoadPlan((Point(5, 20), Point(5, 4), Point(20, 4), Point(20, 20)))
    smoothed = _smooth_land(original, grid, 2)
    assert all(grid.land_slope(a, b) is not None for a, b in zip(smoothed.path, smoothed.path[1:]))


def test_path_does_not_hop_over_a_thin_cliff():
    axis = np.arange(32, dtype=float)
    terrain = np.zeros((32, 32))
    terrain[:, 15] = 8
    water = np.full_like(terrain, -1)
    plan = plan_rural_road(Point(5, 16), Point(26, 16), terrain, water, axis, axis,
                          settings(road_path_step=5), 3)
    assert plan == RoadPlan()


def test_bridge_to_negative_coordinates_has_same_geometry():
    terrain, water, axis = river_scene()
    grid = RoadTerrain(terrain, water, axis - 100, axis - 100, 3)
    bridge = plan_bridge(Point(-84, -76), Point(-70, -76), grid, settings())
    assert bridge is not None
    assert all(-100 <= x <= -53 and -100 <= z <= -53
               for x, y, z, material, kind in bridge_blocks(bridge, grid, settings()))


def test_bridge_blocks_reach_anvil_without_filling_the_water_channel():
    from bong_worldgen.adapters.anvil_nbt import WATER, _section_blocks, register_blockstate
    from bong_worldgen.adapters.minecraft_world import _chunk_structure_blocks
    from bong_worldgen.engine.generated_world import Heightfield

    terrain, water, axis = river_scene()
    blocks = {}
    _emit_road(blocks, Point(8, 24), Point(40, 24), terrain, water, axis, axis, settings(), 6)
    field = Heightfield(height=terrain, moisture=np.zeros_like(terrain), water_level=water,
                        settlement_blocks=tuple(blocks.values()))
    overlay = _chunk_structure_blocks(field, world_min_x=16, world_min_z=16)
    section = _section_blocks(
        5, terrain[16:32, 16:32].astype(np.int16), water[16:32, 16:32].astype(np.int16),
        np.zeros((16, 16), dtype=np.uint8), structure_blocks=overlay,
    )
    decks = [b for b in blocks.values() if b.kind == "bridge_deck" and b.x == 22]
    middle = sorted(decks, key=lambda b: b.z)[len(decks) // 2]
    assert section[1, middle.z - 16, 6] == register_blockstate("minecraft:oak_planks")
    assert section[0, middle.z - 16, 6] == WATER


def test_short_dry_detour_is_preferred_to_an_expensive_bridge():
    terrain, water, axis = river_scene()
    water[:20] = -1
    water[29:] = -1
    terrain[water < 0] = 80
    plan = plan_rural_road(Point(8, 24), Point(40, 24), terrain, water, axis, axis,
                          settings(bridge_cost_factor=20.0), 1)
    assert plan.path and not plan.bridges


def test_north_south_and_reversed_bridge_layouts():
    terrain, water, axis = river_scene()
    grid = RoadTerrain(terrain.T.copy(), water.T.copy(), axis, axis, 2)
    bridge = plan_bridge(Point(24, 30), Point(24, 16), grid, settings(road_width=2))
    assert bridge is not None
    emitted = list(bridge_blocks(bridge, grid, settings(road_width=2)))
    assert any("north=true,south=true" in material for x, y, z, material, kind in emitted)
    assert len([b for b in emitted if b[2] == 22 and b[4] == "bridge_deck"]) == 4


@pytest.mark.parametrize("kwargs", [
    {"bridge_max_span": -1}, {"bridge_clearance": 0},
    {"bridge_approach_length": 1}, {"bridge_max_support_depth": 0},
    {"bridge_cost_factor": 0.5},
])
def test_bridge_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)
