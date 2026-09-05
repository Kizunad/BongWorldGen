from __future__ import annotations

from pathlib import Path

from bong_worldgen.engine.structures import (
    load_litematic,
    load_schematic,
    structure_anchor_y,
    structure_block_kind,
    structure_note,
)
from bong_worldgen.engine.town import _split_settlement_schematics


def test_structure_catalog_contains_added_templates() -> None:
    for name in ("31062", "31085", "31087", "31101", "31122", "18804", "19237", "31406", "31293"):
        structure = load_schematic(Path("assets/structures/houses") / f"{name}.schem")
        assert structure_note(structure.name) is not None


def test_well_anchor_uses_marker_layer_and_marks_water() -> None:
    stone_well = load_schematic(Path("assets/structures/houses/31280.schem"))
    desert_well = load_schematic(Path("assets/structures/houses/31101.schem"))

    assert structure_anchor_y(stone_well.name, stone_well.blocks) == 10
    assert structure_anchor_y(desert_well.name, desert_well.blocks) == 6
    assert structure_block_kind("31280", "minecraft:water") == "well_water"
    assert structure_block_kind("31101", "minecraft:water") == "well_water"


def test_worldedit_v3_wrapper_and_new_resource_pools_are_loadable() -> None:
    farmhouse = load_schematic(Path("assets/structures/houses/31244.schem"))
    tree = load_schematic(Path("assets/structures/trees/31211.schem"))
    dry_tree = load_schematic(Path("assets/structures/trees/31052.schem"))
    pillar = load_schematic(Path("assets/structures/decorations/31212.schem"))
    statue = load_schematic(Path("assets/structures/decorations/31056.schem"))
    campsite = load_schematic(Path("assets/structures/decorations/31075.schem"))
    archived_school = load_litematic(Path("assets/structures/excluded/29642.litematic"))

    assert (farmhouse.width, farmhouse.height, farmhouse.length) == (56, 42, 61)
    assert (tree.width, tree.height, tree.length) == (53, 47, 51)
    assert (dry_tree.width, dry_tree.height, dry_tree.length) == (65, 77, 65)
    assert (pillar.width, pillar.height, pillar.length) == (5, 26, 5)
    assert (statue.width, statue.height, statue.length) == (31, 52, 33)
    assert (campsite.width, campsite.height, campsite.length) == (17, 7, 17)
    assert (archived_school.width, archived_school.height, archived_school.length) == (174, 176, 214)
    assert structure_note(farmhouse.name).category == "house"
    assert structure_note(tree.name).category == "tree"
    assert structure_note(dry_tree.name).category == "tree"
    assert structure_note(pillar.name).category == "prop"
    assert structure_note(statue.name).category == "prop"
    assert structure_note(campsite.name).category == "camp"
    assert structure_note(archived_school.name).category == "excluded"


def test_negative_y_litematic_keeps_ground_at_structure_bottom() -> None:
    """负 Y 区域的基层不能被错误翻到建筑顶部。"""

    structure = load_litematic(Path("assets/structures/houses/22031.litematic"))
    ground = [
        block
        for block in structure.blocks
        if block.blockstate.split("[", 1)[0]
        in {"minecraft:grass_block", "minecraft:dirt_path"}
    ]

    assert ground
    assert min(block.y for block in ground) == 0
    assert max(block.y for block in ground) < structure.height - 1
    assert sum(block.y == 0 for block in ground) > 1000


def test_city_gate_is_classified_for_wall_placement() -> None:
    gate = load_schematic(Path("assets/structures/houses/31293.schem"))
    tower = load_schematic(Path("assets/structures/houses/31122.schem"))

    assert (gate.width, gate.height, gate.length) == (23, 27, 31)
    assert structure_note(gate.name).category == "gate"
    assert structure_block_kind(gate.name, "minecraft:stone_bricks") == "settlement_gate"
    assert structure_block_kind(tower.name, "minecraft:stone_bricks") == "settlement_tower"


def test_antorus_is_standalone_and_market_stall_joins_settlement_pool() -> None:
    """大型地标不得混入部落；市场摊位应使用部落设施语义。"""

    antorus = load_litematic(Path("assets/structures/standalone/22434.litematic"))
    stall = load_litematic(Path("assets/structures/houses/22072.litematic"))
    houses, cores, walls, gates = _split_settlement_schematics((stall, antorus))

    assert (antorus.width, antorus.height, antorus.length) == (108, 193, 80)
    assert structure_note(antorus.name).category == "standalone"
    assert structure_note(stall.name).category == "stall"
    assert structure_block_kind(stall.name, "minecraft:oak_planks") == "settlement_stall"
    assert houses == (stall,)
    assert cores == walls == gates == ()
