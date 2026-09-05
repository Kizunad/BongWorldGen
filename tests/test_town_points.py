from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bong_worldgen.engine.structures import (
    SchematicBlock,
    SchematicStructure,
    load_schematic,
    schematic_interior_spawn,
)
from bong_worldgen.engine.structures import interiors
from bong_worldgen.engine.structures.catalog import STRUCTURE_NOTES, StructureNote
from bong_worldgen.engine.terrain_config import Point, TownSettings
from bong_worldgen.engine.town_growth import GrowthBuilding
from bong_worldgen.engine.town_points import building_interest_points, building_spawn_area


def _building(structure: SchematicStructure, rotation: int = 0) -> GrowthBuilding:
    rotated = structure.rotated(rotation)
    return GrowthBuilding(
        center=Point(-83, 147),
        size_x=rotated.width,
        size_z=rotated.length,
        ground=93,
        schematic=structure,
        rotation=rotation,
        district="core",
    )


@pytest.mark.parametrize("name", ["19212", "19237"])
@pytest.mark.parametrize("rotation", range(4))
def test_actual_spawn_templates_have_covered_npc_points(name: str, rotation: int) -> None:
    structure = load_schematic(Path(f"assets/structures/houses/{name}.schem"))
    building = _building(structure, rotation)
    settings = TownSettings()
    area = building_spawn_area(building, settings)
    center, npc = building_interest_points(building, area, settings)

    assert center.role == "settlement_core"
    assert npc.kind == "spawn_interior_spawn"
    assert npc.role == "npc_spawn"
    assert npc.tags == ("spawn", "interior", "npc_spawn")
    assert npc.area_id == area.area_id
    assert npc.structure_name == name
    assert area.footprint_min_x < npc.x < area.footprint_max_x
    assert area.footprint_min_z < npc.z < area.footprint_max_z
    assert area.ground_y < npc.y < area.footprint_max_y

    # 反向映射回旋转后的真实模板，验证点位确实在地板上、两格净空、非树冠下。
    local = (
        npc.x - area.footprint_min_x,
        npc.y - area.footprint_min_y,
        npc.z - area.footprint_min_z,
    )
    x, y, z = local
    blocks = {(b.x, b.y, b.z): b.material for b in structure.rotated(rotation).blocks}
    assert blocks[x, y - 1, z] not in {"minecraft:water", "minecraft:lava"}
    assert local not in blocks
    assert (x, y + 1, z) not in blocks
    assert any(
        bx == x and bz == z and by > y + 1 and not material.endswith("_leaves")
        for (bx, by, bz), material in blocks.items()
    )
    assert building_interest_points(building, area, settings) == (center, npc)


@pytest.mark.parametrize("roof", [
    None, "minecraft:oak_leaves", "minecraft:oak_log[axis=x]", "minecraft:water",
])
def test_open_courtyard_or_tree_canopy_does_not_produce_an_indoor_spawn(roof: str | None) -> None:
    blocks = [SchematicBlock(2, 0, 2, "minecraft:stone")]
    if roof:
        blocks.append(SchematicBlock(2, 4, 2, roof))
    building = _building(SchematicStructure("19212", 5, 6, 5, tuple(blocks)))
    settings = TownSettings()
    area = building_spawn_area(building, settings)
    assert len(building_interest_points(building, area, settings)) == 1


@pytest.mark.parametrize(("floor", "expected"), [
    ("minecraft:grass", False),
    ("minecraft:white_carpet", False),
    ("minecraft:water[level=0]", False),
    ("minecraft:oak_fence", False),
    ("minecraft:stone_slab[type=bottom,waterlogged=false]", False),
    ("minecraft:stone_slab[type=top,waterlogged=false]", True),
    ("minecraft:stone_slab[type=double,waterlogged=false]", True),
    ("minecraft:oak_planks", True),
])
def test_spawn_requires_a_support_surface_at_the_exported_integer_height(
    floor: str, expected: bool,
) -> None:
    structure = SchematicStructure("19212", 5, 5, 5, (
        SchematicBlock(2, 0, 2, floor),
        SchematicBlock(2, 3, 2, "minecraft:stone"),
    ))
    building = _building(structure)
    settings = TownSettings()
    area = building_spawn_area(building, settings)
    points = building_interest_points(building, area, settings)
    assert len(points) == (2 if expected else 1)
    if expected:
        assert points[1].y == area.footprint_min_y + 1


@pytest.mark.parametrize("obstruction", ["minecraft:stone", "minecraft:water[level=0]"])
def test_spawn_rejects_obstructed_headroom(obstruction: str) -> None:
    structure = SchematicStructure("19212", 5, 5, 5, (
        SchematicBlock(2, 0, 2, "minecraft:stone"),
        SchematicBlock(2, 2, 2, obstruction),
        SchematicBlock(2, 3, 2, "minecraft:stone"),
    ))
    building = _building(structure)
    settings = TownSettings()
    area = building_spawn_area(building, settings)
    assert len(building_interest_points(building, area, settings)) == 1


def test_house_kind_and_outer_district_policy_are_preserved() -> None:
    settings = TownSettings()
    building = GrowthBuilding(Point(4, 8), 5, 5, 70, None, 0, "core")
    area = building_spawn_area(building, settings)
    _, npc = building_interest_points(building, area, settings)
    assert npc.kind == "house_interior_spawn"
    assert npc.y == area.footprint_min_y + 1

    outer = replace(building, district="outer")
    outer_area = building_spawn_area(outer, settings)
    [center] = building_interest_points(outer, outer_area, settings)
    assert center.kind == "outer_building"


def test_template_is_analyzed_once_and_its_point_follows_every_placement(monkeypatch) -> None:
    # 两个同分候选用于防止“旋转后重新选点”悄悄换到另一个房间。
    structure = SchematicStructure("19212", 7, 5, 5, tuple(
        SchematicBlock(x, y, 2, "minecraft:stone")
        for x in (2, 4) for y in (0, 3)
    ))
    expected_local_points = ((2, 1, 2), (2, 1, 2), (4, 1, 2), (2, 1, 4))
    scan = interiors._scan_interior_feet
    calls = []

    def counted_scan(template, first_y):
        calls.append(template)
        return scan(template, first_y)

    schematic_interior_spawn.cache_clear()
    monkeypatch.setattr(interiors, "_scan_interior_feet", counted_scan)
    for center, ground, offset in ((Point(-83, 147), 93, 1), (Point(149, -83), 140, 2)):
        settings = TownSettings(structure_ground_offset=offset)
        for rotation, expected in enumerate(expected_local_points):
            building = replace(_building(structure, rotation), center=center, ground=ground)
            area = building_spawn_area(building, settings)
            _, npc = building_interest_points(building, area, settings)
            assert (npc.x, npc.y, npc.z) == (
                area.footprint_min_x + expected[0],
                area.footprint_min_y + expected[1],
                area.footprint_min_z + expected[2],
            )
    assert calls == [structure]


def test_changed_template_contents_invalidate_the_cached_point() -> None:
    structure = SchematicStructure("19212", 5, 5, 5, (
        SchematicBlock(2, 0, 2, "minecraft:stone"),
        SchematicBlock(2, 3, 2, "minecraft:stone"),
    ))
    assert schematic_interior_spawn(structure) == (2, 1, 2)
    obstructed = replace(structure, blocks=structure.blocks + (
        SchematicBlock(2, 1, 2, "minecraft:stone"),
    ))
    assert schematic_interior_spawn(obstructed) is None
    assert schematic_interior_spawn(structure) == (2, 1, 2)


def test_local_point_respects_template_anchor_before_world_height_mapping(monkeypatch) -> None:
    name = "test_raised_spawn"
    monkeypatch.setitem(STRUCTURE_NOTES, name, StructureNote(
        "测试核心", "围栏为首层标记", "spawn", anchor_marker="fence",
    ))
    structure = SchematicStructure(name, 5, 8, 5, (
        SchematicBlock(0, 3, 0, "minecraft:oak_fence"),
        SchematicBlock(2, 3, 2, "minecraft:stone"),
        SchematicBlock(2, 6, 2, "minecraft:stone"),
    ))
    assert schematic_interior_spawn(structure) == (2, 4, 2)
    settings = TownSettings(structure_ground_offset=2)
    building = _building(structure, 1)
    area = building_spawn_area(building, settings)
    _, npc = building_interest_points(building, area, settings)
    assert area.footprint_min_y == building.ground + 2 - 3
    assert npc.y == building.ground + 2 + 1
