from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bong_worldgen.engine import (
    Point,
    TownSettings,
    TerrainRecipe,
    generate_heightfield,
    generate_town,
    generate_town_result,
    SettlementInterestPoint,
)
from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.engine.settlement_points import SettlementSpawnArea
from bong_worldgen.engine.town import _split_settlement_schematics
from bong_worldgen.engine.town_growth import (
    generate_town_layout,
    wall_utilization,
)
from bong_worldgen.engine.rural_roads import _clamped_cell, rural_road_path
from bong_worldgen.engine.structures import SchematicBlock, SchematicStructure, load_litematic
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.engine.structures import load_structure_directory


def _settings() -> TownSettings:
    return TownSettings(
        center=Point(48.0, 48.0),
        radius=28.0,
        core_radius=20.0,
        core_house_count=5,
        outer_house_count=0,
        outer_min_radius=24.0,
        house_min_size=5,
        house_max_size=7,
        house_height=3,
        road_width=3,
        building_gap_min=1,
        building_gap_max=2,
        maximum_slope=0.24,
        maximum_relief=1.0,
    )


def test_town_emits_seeded_stone_brick_structures() -> None:
    recipe = TerrainRecipe(
        name="settlement_test",
        base_height=80.0,
        town=_settings(),
    )
    first = generate_heightfield(recipe, width=128, height=128, seed=41)
    repeat = generate_heightfield(recipe, width=128, height=128, seed=41)
    other = generate_heightfield(recipe, width=128, height=128, seed=42)

    assert first.settlement_blocks
    assert first.settlement_blocks == repeat.settlement_blocks
    assert first.settlement_blocks != other.settlement_blocks
    assert {block.material for block in first.settlement_blocks} == {"stone_bricks"}
    assert {block.kind for block in first.settlement_blocks} >= {
        "road",
        "house_foundation",
        "house_wall",
        "house_roof",
    }
    assert all(block.y >= 80 for block in first.settlement_blocks)


def test_town_is_empty_when_disabled() -> None:
    settings = TownSettings(enabled=False)
    recipe = TerrainRecipe(name="settlement_disabled", base_height=80.0, town=settings)
    field = generate_heightfield(recipe, width=32, height=32, seed=7)

    assert field.settlement_blocks == ()
    assert np.all(field.water_level == -1.0)


def test_rural_road_is_seeded_and_bends_on_flat_ground() -> None:
    axis = np.arange(256, dtype=np.float64)
    terrain = np.zeros((256, 256), dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        road_wander=0.45,
        road_path_step=3,
        road_search_margin=72.0,
        road_smoothing_passes=1,
    )
    first = rural_road_path(
        Point(20.0, 20.0), Point(220.0, 220.0), terrain, water, axis, axis, settings, 41
    )
    repeat = rural_road_path(
        Point(20.0, 20.0), Point(220.0, 220.0), terrain, water, axis, axis, settings, 41
    )
    other = rural_road_path(
        Point(20.0, 20.0), Point(220.0, 220.0), terrain, water, axis, axis, settings, 42
    )

    assert first == repeat
    assert first != other
    assert first[0] == Point(20.0, 20.0)
    assert first[-1] == Point(220.0, 220.0)
    deviation = max(
        abs((point.x - 20.0) * 200.0 - (point.z - 20.0) * 200.0)
        / np.sqrt(80000.0)
        for point in first
    )
    assert deviation > 1.0


def test_rural_road_routes_around_water_barrier() -> None:
    axis = np.arange(96, dtype=np.float64)
    terrain = np.zeros((96, 96), dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    water[42:54, 20:76] = 0.0
    path = rural_road_path(
        Point(10.0, 48.0),
        Point(86.0, 48.0),
        terrain,
        water,
        axis,
        axis,
        TownSettings(
            road_wander=0.0, road_path_step=2, road_search_margin=20.0
        ),
        73,
    )

    assert path[0] == Point(10.0, 48.0)
    assert path[-1] == Point(86.0, 48.0)
    # 以生成器的最近格采样规则判断；道路可以贴着岸线通过，但不能进入水面。
    assert all(water[_clamped_cell(axis, axis, point.x, point.z)] < 0.0 for point in path)


def test_schematic_houses_are_loaded_and_placed_with_blockstates() -> None:
    axis = np.arange(160, dtype=np.float64)
    terrain = np.full((160, 160), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        center=Point(80.0, 80.0),
        radius=46.0,
        core_radius=28.0,
        core_house_count=3,
        outer_house_count=0,
        outer_min_radius=34.0,
        maximum_relief=1.0,
        house_schematic_directory="assets/structures/houses",
    )

    blocks = generate_town(
        terrain,
        water,
        np.broadcast_to(axis, (160, 160)),
        np.broadcast_to(axis[:, None], (160, 160)),
        settings,
        Point(80.0, 80.0),
        812731,
    )

    schematic_blocks = [block for block in blocks if block.kind == "house_schematic"]
    assert schematic_blocks
    assert any("[" in block.material for block in schematic_blocks)
    assert any(block.material.startswith("minecraft:") for block in schematic_blocks)


def test_spawn_core_replaces_plaza_and_wall_is_emitted() -> None:
    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        center=Point(0.0, 0.0),
        radius=110.0,
        core_radius=64.0,
        core_house_count=6,
        outer_house_count=4,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        attractor_count=2,
        main_road_count=1,
        growth_candidate_count=24,
        house_schematic_directory="assets/structures/houses",
        tree_count=0,
    )
    grid_x = np.broadcast_to(axis, (axis.size, axis.size))
    grid_z = np.broadcast_to(axis[:, None], (axis.size, axis.size))
    result = generate_town_result(
        terrain, water, grid_x, grid_z, settings, Point(0.0, 0.0), 812731
    )

    kinds = {block.kind for block in result.blocks}
    assert "spawn_core" in kinds
    assert "settlement_wall" in kinds
    assert "settlement_gate" in kinds
    assert "plaza" not in kinds
    # 核心 Spawn 地标不消耗住宅配额，全部区域仍必须带有明确分区。
    assert len(result.spawn_areas) <= settings.core_house_count + settings.outer_house_count + 1
    assert {area.district for area in result.spawn_areas} <= {"core", "outer"}


def test_wall_gate_towers_and_outer_road_share_the_defensive_boundary() -> None:
    """城墙使用完整墙段，城门/角塔与门外土路遵守同一地表基准。"""

    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        center=Point(0.0, 0.0),
        radius=110.0,
        core_radius=64.0,
        core_house_count=6,
        outer_house_count=4,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        attractor_count=2,
        main_road_count=1,
        growth_candidate_count=24,
        house_schematic_directory="assets/structures/houses",
        tree_count=0,
    )
    result = generate_town_result(
        terrain,
        water,
        np.broadcast_to(axis, terrain.shape),
        np.broadcast_to(axis[:, None], terrain.shape),
        settings,
        Point(0.0, 0.0),
        812731,
    )

    walls = [block for block in result.blocks if block.kind == "settlement_wall"]
    gates = [block for block in result.blocks if block.kind == "settlement_gate"]
    towers = [block for block in result.blocks if block.kind == "settlement_tower"]
    gate_roads = [block for block in result.blocks if block.kind == "gate_road"]

    assert walls and gates and towers and gate_roads
    # 普通墙仍在地表上一格；城门和四角塔按配置下沉一格与铺路齐平。
    assert min(block.y for block in walls) == 81
    assert min(block.y for block in gates) == 80
    assert min(block.y for block in towers) == 80
    assert {block.material for block in gate_roads} == {settings.gate_road_material}
    gate_floor = [block for block in gates if block.y == 80]
    assert min(
        abs(road.x - gate.x) + abs(road.z - gate.z)
        for road in gate_roads
        for gate in gate_floor
    ) <= 1


def test_outer_alleys_only_connect_to_gate_road_network(monkeypatch) -> None:
    """核心区不生成巷道；外围巷道必须从城门道路网络分支。"""

    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        center=Point(0.0, 0.0),
        radius=110.0,
        core_radius=64.0,
        core_house_count=6,
        outer_house_count=4,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        attractor_count=2,
        main_road_count=1,
        growth_candidate_count=24,
        house_schematic_directory="assets/structures/houses",
        tree_count=0,
        gate_road_min_width=2,
        gate_road_max_width=5,
    )
    grid_x = np.broadcast_to(axis, terrain.shape)
    grid_z = np.broadcast_to(axis[:, None], terrain.shape)
    calls: list[tuple[Point, Point, str, int | None]] = []

    def capture_road(
        _blocks: object,
        start: Point,
        end: Point,
        *_args: object,
        kind: str = "road",
        width: int | None = None,
        **_kwargs: object,
    ) -> tuple[Point, ...]:
        calls.append((start, end, kind, width))
        return (start, end)

    monkeypatch.setattr("bong_worldgen.engine.town._emit_road", capture_road)
    generate_town_result(
        terrain, water, grid_x, grid_z, settings, Point(0.0, 0.0), 812731
    )

    houses, cores, _, _ = _split_settlement_schematics(
        load_structure_directory(Path("assets/structures/houses"))
    )
    layout = generate_town_layout(
        terrain,
        water,
        axis,
        axis,
        settings,
        Point(0.0, 0.0),
        812731,
        houses,
        (),
        cores,
    )
    outer_centers = {
        building.center for building in layout.buildings if building.district == "outer"
    }
    gate_roads = [call for call in calls if call[2] == "gate_road"]
    alleys = [call for call in calls if call[2] == "alley"]

    assert gate_roads
    assert all(
        settings.gate_road_min_width <= width <= settings.gate_road_max_width
        for _start, _end, _kind, width in gate_roads
        if width is not None
    )
    assert all(start in outer_centers for start, _end, _kind, _width in alleys)

    road_network: set[Point] = set()
    for start, end, kind, _width in calls:
        if kind == "gate_road":
            road_network.update((start, end))
        elif kind == "alley":
            assert end in road_network
            road_network.update((start, end))


def test_core_wall_does_not_include_scattered_outer_buildings(monkeypatch) -> None:
    """外围部落即使更分散，也不能扩大核心区城墙。"""

    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        radius=110.0,
        core_radius=64.0,
        core_house_count=4,
        outer_house_count=4,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        growth_candidate_count=24,
        house_schematic_directory="assets/structures/houses",
        tree_count=0,
    )
    captured: dict[str, tuple[object, ...]] = {}

    def capture_wall(_: object, buildings: tuple[object, ...], *_args: object) -> None:
        captured["buildings"] = buildings

    monkeypatch.setattr("bong_worldgen.engine.town._emit_settlement_wall", capture_wall)
    result = generate_town_result(
        terrain,
        water,
        np.broadcast_to(axis, terrain.shape),
        np.broadcast_to(axis[:, None], terrain.shape),
        settings,
        Point(0.0, 0.0),
        812731,
    )

    wall_buildings = captured["buildings"]
    assert wall_buildings
    assert all(getattr(building, "district") == "core" for building in wall_buildings)
    assert len(wall_buildings) < len(result.spawn_areas)


def test_default_town_keeps_the_wall_inside_the_dense_core() -> None:
    """默认城镇的围墙只包住核心，并保持足够高的核心利用率。"""

    axis = np.arange(-600, 601, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = DEFAULT_RECIPE.town
    houses, cores, _, _ = _split_settlement_schematics(
        load_structure_directory(Path("assets/structures/houses"))
    )
    for seed in (1, 812731):
        layout = generate_town_layout(
            terrain,
            water,
            axis,
            axis,
            settings,
            settings.center,
            seed,
            houses,
            (),
            cores,
        )
        outer_buildings = [item for item in layout.buildings if item.district == "outer"]
        assert layout.core_buildings
        assert all(item.district == "core" for item in layout.core_buildings)
        assert outer_buildings
        assert all(
            np.hypot(item.center.x - settings.center.x, item.center.z - settings.center.z)
            >= settings.outer_min_radius
            for item in outer_buildings
        )
        assert wall_utilization(layout.core_buildings) >= (
            settings.core_wall_minimum_utilization
        )


def test_schematic_rotation_moves_directional_connection_properties() -> None:
    structure = SchematicStructure(
        name="directional-test",
        width=2,
        height=1,
        length=1,
        blocks=(
            SchematicBlock(
                0,
                0,
                0,
                "minecraft:oak_fence[north=true,east=false]",
            ),
        ),
    )

    rotated = structure.rotated(1)

    assert rotated.width == 1
    assert rotated.length == 2
    assert rotated.blocks[0].x == 0
    assert rotated.blocks[0].z == 0
    assert rotated.blocks[0].blockstate == "minecraft:oak_fence[east=true,south=false]"


def test_growth_layout_emits_trees_and_mixed_rural_roads() -> None:
    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        radius=110.0,
        core_radius=64.0,
        core_house_count=8,
        outer_house_count=5,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        attractor_count=3,
        main_road_count=2,
        growth_candidate_count=24,
        road_materials=("dirt", "gravel", "coarse_dirt"),
        road_material_weights=(0.5, 0.3, 0.2),
        house_schematic_directory="assets/structures/houses",
        tree_schematic_directory="assets/structures/trees",
        tree_count=4,
    )
    grid_x = np.broadcast_to(axis, (axis.size, axis.size))
    grid_z = np.broadcast_to(axis[:, None], (axis.size, axis.size))
    first = generate_town(terrain, water, grid_x, grid_z, settings, Point(0.0, 0.0), 812731)
    repeat = generate_town(terrain, water, grid_x, grid_z, settings, Point(0.0, 0.0), 812731)
    assert first == repeat
    assert any(block.kind == "tree_schematic" for block in first)
    assert any(block.kind == "alley" for block in first)
    road_materials = {block.material for block in first if block.kind in {"road", "alley"}}
    assert road_materials <= {"dirt", "gravel", "coarse_dirt"}
    assert len(road_materials) >= 2


def test_settlement_spawn_areas_include_structure_footprint_and_height() -> None:
    axis = np.arange(-320, 321, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 80.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    settings = TownSettings(
        radius=110.0,
        core_radius=64.0,
        core_house_count=4,
        outer_house_count=4,
        outer_min_radius=76.0,
        outer_cluster_spread=20.0,
        growth_candidate_count=24,
        npc_spawn_radius=9,
        house_schematic_directory="assets/structures/houses",
    )
    grid_x = np.broadcast_to(axis, (axis.size, axis.size))
    grid_z = np.broadcast_to(axis[:, None], (axis.size, axis.size))
    result = generate_town_result(
        terrain, water, grid_x, grid_z, settings, Point(0.0, 0.0), 812731
    )

    assert result.spawn_areas
    assert len(result.spawn_areas) <= settings.core_house_count + settings.outer_house_count + 1
    core_area = next(area for area in result.spawn_areas if area.category == "spawn")
    core_npcs = [
        point for point in result.interest_points
        if point.area_id == core_area.area_id and point.role == "npc_spawn"
    ]
    assert len(core_npcs) == 1
    assert core_npcs[0].kind == "spawn_interior_spawn"
    for area in result.spawn_areas:
        assert area.min_x <= area.footprint_min_x <= area.footprint_max_x <= area.max_x
        assert area.min_z <= area.footprint_min_z <= area.footprint_max_z <= area.max_z
        assert area.min_y == area.ground_y
        assert area.min_y <= area.max_y
        # 普通房屋地基在地表上方；井等模板允许保留地表以下的水/井底，
        # 但完整 footprint 必须覆盖周围地形的站立层。
        assert area.footprint_min_y <= area.footprint_max_y
        assert area.defense_radius == settings.npc_spawn_radius
        assert area.district in {"core", "outer"}


def test_settlement_spawn_areas_are_exported_for_server(tmp_path: Path) -> None:
    area = SettlementSpawnArea(
        area_id="town-core-test-0-0",
        structure_name="31232",
        category="house",
        district="core",
        center_x=4,
        center_z=4,
        min_x=-4,
        max_x=12,
        min_z=-4,
        max_z=12,
        min_y=80,
        max_y=92,
        ground_y=80,
        footprint_min_x=0,
        footprint_max_x=8,
        footprint_min_z=0,
        footprint_max_z=8,
        footprint_min_y=80,
        footprint_max_y=92,
        defense_radius=4,
    )
    recipe = TerrainRecipe(name="settlement_area_export", base_height=80.0)
    field = generate_heightfield(recipe, width=16, height=16, seed=11)
    from bong_worldgen.engine.generated_world import Heightfield

    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        settlement_spawn_areas=(area,),
    )
    tile = to_bong_tile(field, sea_level=recipe.sea_level)
    write_bong_raster(tile, tmp_path)
    payload = (tmp_path / "tile_0_0" / "settlement_spawn_areas.json").read_text(
        encoding="utf-8"
    )
    assert "town-core-test-0-0" in payload
    assert '"min_y": 80' in payload
    assert '"district": "core"' in payload


def test_settlement_interest_points_are_exported_per_tile_and_preview(tmp_path: Path) -> None:
    point = SettlementInterestPoint(
        point_id="town-core-31232-0-0-interior-spawn",
        kind="house_interior_spawn",
        role="npc_spawn",
        district="core",
        x=4,
        y=81,
        z=4,
        area_id="town-core-31232-0-0",
        structure_name="31232",
        tags=("house", "interior", "npc_spawn"),
    )
    recipe = TerrainRecipe(name="settlement_point_export", base_height=80.0)
    field = generate_heightfield(recipe, width=16, height=16, seed=11)
    from bong_worldgen.engine.generated_world import Heightfield

    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        settlement_interest_points=(point,),
    )
    tile = to_bong_tile(field, sea_level=recipe.sea_level)
    write_bong_raster(tile, tmp_path)
    tile_payload = (tmp_path / "tile_0_0" / "settlement_interest_points.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert point.point_id in tile_payload
    assert manifest["settlement_interest_point_encoding"]["file"] == "settlement_interest_points.json"


def test_litematic_dimensions_preserve_air_borders() -> None:
    structure = load_litematic(Path("assets/structures/houses/31403.litematic"))
    assert (structure.width, structure.height, structure.length) == (18, 25, 23)
    assert structure.blocks
