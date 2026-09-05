from __future__ import annotations

from pathlib import Path

import numpy as np

import bong_worldgen.engine.standalone_structures as standalone_structures
from bong_worldgen.engine import Point, StandaloneStructureSettings, generate_standalone_structures
from bong_worldgen.engine.town import _split_settlement_schematics
from bong_worldgen.engine.structures import (
    SchematicBlock,
    SchematicStructure,
    load_schematic,
    structure_block_kind,
    structure_note,
)


def _standalone_template() -> SchematicStructure:
    """用最小真实 footprint 覆盖独立结构层的选址逻辑。"""

    return SchematicStructure(
        name="31382",
        width=60,
        height=30,
        length=60,
        blocks=(SchematicBlock(0, 0, 0, "minecraft:deepslate"),),
    )


def _tango_towers() -> tuple[SchematicStructure, SchematicStructure]:
    """构造小型双塔，覆盖成组候选与原子写入逻辑。"""

    blocks = (SchematicBlock(0, 0, 0, "minecraft:stone_bricks"),)
    return (
        SchematicStructure("20493", 10, 12, 10, blocks),
        SchematicStructure("20503", 10, 12, 10, blocks),
    )


def _world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-600, 601, dtype=np.float64)
    terrain = np.full((axis.size, axis.size), 96.0, dtype=np.float64)
    water = np.full_like(terrain, -1.0)
    return (
        terrain,
        water,
        np.broadcast_to(axis, terrain.shape),
        np.broadcast_to(axis[:, None], terrain.shape),
    )


def _settings() -> StandaloneStructureSettings:
    return StandaloneStructureSettings(
        structure_directory="assets/structures/standalone",
        grid_spacing=1024.0,
        maximum_slope=0.35,
        maximum_relief=7.0,
        exclusion_center=Point(-20_000.0, -20_000.0),
        exclusion_radius=10_000.0,
        npc_spawn_radius=14,
    )


def test_standalone_catalog_keeps_large_templates_out_of_settlement_pool() -> None:
    standalone = tuple(
        load_schematic(Path("assets/structures/standalone") / f"{name}.schem")
        for name in ("31382", "31279")
    )

    assert [(structure.width, structure.height, structure.length) for structure in standalone] == [
        (60, 30, 60),
        (60, 36, 60),
    ]
    assert all(structure_note(structure.name).category == "standalone" for structure in standalone)
    assert _split_settlement_schematics(standalone) == ((), (), (), ())
    assert structure_block_kind("31382@rot90", "minecraft:deepslate") == "standalone_structure"


def test_standalone_structure_generation_is_stable_and_exports_npc_area(monkeypatch) -> None:
    structure = _standalone_template()
    monkeypatch.setattr(
        "bong_worldgen.engine.standalone_structures.load_structure_directory",
        lambda _: (structure,),
    )
    terrain, water, x, z = _world()

    first = generate_standalone_structures(terrain, water, x, z, _settings(), 73)
    repeat = generate_standalone_structures(terrain, water, x, z, _settings(), 73)

    assert first == repeat
    assert first.blocks
    assert first.spawn_areas
    assert {block.kind for block in first.blocks} == {"standalone_structure"}
    for area in first.spawn_areas:
        assert area.category == "standalone"
        assert area.structure_name.startswith("31382")
        assert area.footprint_max_x - area.footprint_min_x + 1 == 60
        assert area.footprint_max_z - area.footprint_min_z + 1 == 60
        assert area.defense_radius == 14


def test_standalone_structure_rejects_water_relief_and_dynamic_spawn_exclusion(monkeypatch) -> None:
    structure = _standalone_template()
    monkeypatch.setattr(
        "bong_worldgen.engine.standalone_structures.load_structure_directory",
        lambda _: (structure,),
    )
    terrain, water, x, z = _world()
    settings = _settings()

    assert generate_standalone_structures(terrain, water, x, z, settings, 73).blocks
    assert not generate_standalone_structures(
        terrain,
        np.zeros_like(water),
        x,
        z,
        settings,
        73,
    ).blocks
    assert not generate_standalone_structures(
        terrain + x,
        water,
        x,
        z,
        settings,
        73,
    ).blocks
    assert not generate_standalone_structures(
        terrain,
        water,
        x,
        z,
        settings,
        73,
        exclusion_center=Point(0.0, 0.0),
    ).blocks


def test_grouped_standalone_structures_emit_all_members_or_none(monkeypatch) -> None:
    """成组地标不能因其中一座不适宜而输出半组建筑。"""

    towers = _tango_towers()
    monkeypatch.setattr(
        "bong_worldgen.engine.standalone_structures.load_structure_directory",
        lambda _: towers,
    )
    terrain, water, x, z = _world()

    generated = generate_standalone_structures(terrain, water, x, z, _settings(), 73)
    assert generated.blocks
    assert {area.structure_name.split("@", 1)[0] for area in generated.spawn_areas} == {
        "20493",
        "20503",
    }

    original_suitable_location = standalone_structures._suitable_location

    def reject_second_tower(*args, **kwargs):
        structure = args[5]
        if structure.name.split("@", 1)[0] == "20503":
            return None
        return original_suitable_location(*args, **kwargs)

    monkeypatch.setattr(
        "bong_worldgen.engine.standalone_structures._suitable_location",
        reject_second_tower,
    )
    rejected = generate_standalone_structures(terrain, water, x, z, _settings(), 73)
    assert rejected.blocks == ()
    assert rejected.spawn_areas == ()
