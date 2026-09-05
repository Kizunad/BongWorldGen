from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from bong_worldgen.adapters.anvil_nbt import (
    AIR,
    BLUE_ICE,
    CHEST,
    DATA_VERSION,
    GRASS_BLOCK,
    ICE,
    OAK_LOG,
    POWDER_SNOW,
    read_root_compound,
    SNOW_BLOCK,
    STONE_BRICKS,
    STONE,
    _section_blocks,
)
from bong_worldgen.adapters.minecraft_world import (
    _block_heights,
    chunk_index_in_region,
    export_minecraft_world,
    region_for_chunk,
)
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config
from bong_worldgen.engine.generated_world import Heightfield
from bong_worldgen.engine.settlement_points import SettlementInterestPoint
from bong_worldgen.engine.town import SettlementBlock
from bong_worldgen.engine.underground_config import UndergroundBlock, UndergroundWaterBlock


def _field(size: int = 16) -> Heightfield:
    height = np.full((size, size), 70.0, dtype=np.float32)
    water = np.full((size, size), -1.0, dtype=np.float32)
    moisture = np.full((size, size), 0.5, dtype=np.float32)
    height[0, 0] = 160.0
    height[0, 1] = 59.0
    water[0, 1] = 61.0
    return Heightfield(height=height, moisture=moisture, water_level=water)


def _read_region_chunk(path: Path, chunk_x: int, chunk_z: int) -> dict[str, object]:
    data = path.read_bytes()
    index = chunk_index_in_region(chunk_x, chunk_z)
    entry = data[index * 4 : index * 4 + 4]
    offset = int.from_bytes(entry[:3], "big") * 4096
    sectors = entry[3]
    assert offset >= 8192
    assert sectors >= 1
    length = struct.unpack(">I", data[offset : offset + 4])[0]
    assert data[offset + 4] == 2
    compressed = data[offset + 5 : offset + 4 + length]
    return read_root_compound(zlib.decompress(compressed))


def test_negative_chunk_and_region_coordinate_contract() -> None:
    assert region_for_chunk(-1, -1) == (-1, -1)
    assert region_for_chunk(-32, -32) == (-1, -1)
    assert region_for_chunk(-33, -33) == (-2, -2)
    assert chunk_index_in_region(-1, -1) == 1023


def test_export_writes_readable_chunk_water_snow_and_level_dat(tmp_path: Path) -> None:
    result = export_minecraft_world(
        _field(),
        tmp_path,
        origin_x=-16,
        origin_z=-16,
        sea_level=61.0,
        seed=7,
        world_name="test-world",
    )

    assert result.chunks_written == 1
    assert result.regions_written == 1
    chunk = _read_region_chunk(tmp_path / "region" / "r.-1.-1.mca", -1, -1)
    assert chunk["DataVersion"] == DATA_VERSION
    assert chunk["xPos"] == -1
    assert chunk["zPos"] == -1
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert "minecraft:water" in palette_names
    assert "minecraft:snow_block" in palette_names
    assert "minecraft:grass_block" in palette_names
    grass_states = [
        entry
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
        if entry["Name"] == "minecraft:grass_block"
    ]
    assert grass_states == [
        {"Name": "minecraft:grass_block", "Properties": {"snowy": "false"}}
    ]

    level = read_root_compound(gzip.decompress((tmp_path / "level.dat").read_bytes()))
    assert level["Data"]["DataVersion"] == DATA_VERSION
    assert level["Data"]["LevelName"] == "test-world"
    assert level["Data"]["RandomSeed"] == 7


def test_section_overlay_accepts_town_stone_bricks() -> None:
    surface = np.full((16, 16), 70, dtype=np.int16)
    water = np.full((16, 16), -1, dtype=np.int16)
    blocks = np.full((16, 16), GRASS_BLOCK, dtype=np.uint8)
    structure = np.asarray([(2, 3, 71, STONE_BRICKS)], dtype=np.int32)

    section = _section_blocks(
        4,
        surface,
        water,
        blocks,
        structure_blocks=structure,
    )

    assert section[7, 3, 2] == STONE_BRICKS


def test_export_writes_town_blocks_to_anvil(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        settlement_blocks=(SettlementBlock(2, 71, 3, "stone_bricks", "house_wall"),),
    )
    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="settlement-world",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert "minecraft:stone_bricks" in palette_names


def test_preview_villagers_use_spawn_feet_and_survive_entity_region_roundtrip(tmp_path: Path) -> None:
    point = SettlementInterestPoint(
        point_id="town-core-house-test-interior-spawn",
        kind="house_interior_spawn",
        role="npc_spawn",
        district="core",
        x=-1,
        y=71,
        z=-1,
    )
    field = replace(
        _field(),
        settlement_interest_points=(
            point,
            point,  # 重复引用只放一个实体。
            replace(point, point_id="outside-preview", x=0),
            replace(point, point_id="building-center", role="residential", x=-2),
        ),
    )
    options = dict(origin_x=-16, origin_z=-16, sea_level=61.0, seed=7, world_name="npc-preview")
    normal = export_minecraft_world(field, tmp_path / "normal", **options)
    assert normal.preview_villagers_written == 0
    assert not (tmp_path / "normal" / "entities").exists()

    preview = tmp_path / "preview"
    result = export_minecraft_world(field, preview, preview_npc_spawns=True, **options)
    assert result.preview_villagers_written == 1
    path = preview / "entities" / "r.-1.-1.mca"
    chunk = _read_region_chunk(path, -1, -1)
    assert chunk["DataVersion"] == 3465
    assert chunk["Position"] == [-1, -1]
    [villager] = chunk["Entities"]
    assert villager["id"] == "minecraft:villager"
    assert villager["Pos"] == [-0.5, 71.0, -0.5]
    assert villager["NoAI"] == villager["NoGravity"] == villager["PersistenceRequired"] == 1
    assert villager["Age"] == 0
    assert villager["VillagerData"]["type"] == "minecraft:plains"
    assert point.point_id in villager["Tags"]
    assert len(villager["UUID"]) == 4

    first = path.read_bytes()
    export_minecraft_world(field, preview, preview_npc_spawns=True, **options)
    assert path.read_bytes() == first
    export_minecraft_world(
        replace(field, settlement_interest_points=()), preview, preview_npc_spawns=True, **options
    )
    assert _read_region_chunk(path, -1, -1)["Entities"] == []


def test_export_preserves_schematic_blockstate_properties(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        settlement_blocks=(
            SettlementBlock(
                2,
                192,
                3,
                "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]",
                "house_schematic",
            ),
        ),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="schematic-blockstate-world",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    stair_states = [
        entry
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
        if entry["Name"] == "minecraft:oak_stairs"
    ]

    assert stair_states == [
        {
            "Name": "minecraft:oak_stairs",
            "Properties": {
                "facing": "east",
                "half": "bottom",
                "shape": "straight",
                "waterlogged": "false",
            },
        }
    ]
    assert 12 in {section["Y"] for section in chunk["sections"]}, (
        "高于地形所在 section 的房屋方块必须让导出器创建新的 section"
    )


def test_export_supports_bong_extended_world_height(tmp_path: Path) -> None:
    """自定义 496 格世界的高山必须写出对应的高位 section。"""

    field = Heightfield(
        height=np.full((16, 16), 424.0, dtype=np.float32),
        moisture=np.full((16, 16), 0.5, dtype=np.float32),
        water_level=np.full((16, 16), -1.0, dtype=np.float32),
    )
    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="extended-height-world",
    )

    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    section_ys = {section["Y"] for section in chunk["sections"]}
    assert 26 in section_ys, "Y=424 应写入 416..431 对应的高位 section"


def test_export_maps_glacial_surface_materials_to_real_minecraft_blocks(tmp_path: Path) -> None:
    size = 16
    material_ids = np.tile(np.arange(6, dtype=np.uint8), (size, 3))[:size, :size]
    field = Heightfield(
        height=np.full((size, size), 140.0, dtype=np.float32),
        moisture=np.full((size, size), 0.5, dtype=np.float32),
        water_level=np.full((size, size), -1.0, dtype=np.float32),
        surface_material_id=material_ids,
        surface_material_palette=(
            "minecraft:snow_block",
            "minecraft:powder_snow",
            "minecraft:ice",
            "minecraft:packed_ice",
            "minecraft:blue_ice",
        ),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="glacial-materials",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert {
        "minecraft:snow_block",
        "minecraft:powder_snow",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
    } <= palette_names


def test_export_maps_permafrost_ids_to_real_minecraft_blocks(tmp_path: Path) -> None:
    size = 16
    palette = (
        "minecraft:powder_snow",
        "minecraft:snow_block",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:gravel",
        "minecraft:coarse_dirt",
        "minecraft:dirt",
        "minecraft:stone",
    )
    ids = np.resize(np.arange(1, len(palette) + 1, dtype=np.uint8), (size, size))
    field = Heightfield(
        height=np.full((size, size), 90.0, dtype=np.float32),
        moisture=np.full((size, size), 0.5, dtype=np.float32),
        water_level=np.full((size, size), -1.0, dtype=np.float32),
        permafrost_id=ids,
        permafrost_palette=palette,
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="permafrost-materials",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert set(palette) <= palette_names


def test_export_uses_final_visible_surface_when_no_cover_layers(tmp_path: Path) -> None:
    size = 16
    visible = np.ones((size, size), dtype=np.uint8)
    field = Heightfield(
        height=np.full((size, size), 140.0, dtype=np.float32),
        moisture=np.full((size, size), 0.5, dtype=np.float32),
        water_level=np.full((size, size), -1.0, dtype=np.float32),
        surface_visible_id=visible,
        surface_visible_palette=("minecraft:blue_ice",),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="visible-surface",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert "minecraft:blue_ice" in palette_names


def test_export_does_not_reintroduce_fixed_height_snow_with_visible_layer(tmp_path: Path) -> None:
    """最终可见层存在时，高程不能在适配器中重新制造固定雪线。"""

    size = 16
    field = Heightfield(
        height=np.full((size, size), 160.0, dtype=np.float32),
        moisture=np.full((size, size), 0.5, dtype=np.float32),
        water_level=np.full((size, size), -1.0, dtype=np.float32),
        surface_visible_id=np.ones((size, size), dtype=np.uint8),
        surface_visible_palette=("minecraft:grass_block",),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="visible-highland",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert "minecraft:grass_block" in palette_names
    assert "minecraft:snow_block" not in palette_names


def test_section_stacks_glacial_cover_above_original_surface() -> None:
    surface_y = np.full((16, 16), 70, dtype=np.int16)
    water_y = np.full((16, 16), -1, dtype=np.int16)
    surface_blocks = np.full((16, 16), STONE, dtype=np.uint8)
    cover = np.zeros((4, 16, 16), dtype=np.uint8)
    cover[:, 3, 4] = (1, 1, 1, 1)

    blocks = _section_blocks(
        4,
        surface_y,
        water_y,
        surface_blocks,
        surface_cover_layers=cover,
    )

    # section 4 starts at Y=64; base terrain is Y=70, so the four additions
    # must occupy 71..74 while retaining stone at Y=70.
    assert blocks[6, 3, 4] == STONE
    assert blocks[7:11, 3, 4].tolist() == [BLUE_ICE, ICE, SNOW_BLOCK, POWDER_SNOW]
    assert blocks[11, 3, 4] == AIR


def test_export_marks_raised_river_water_as_flowing_state(tmp_path: Path) -> None:
    size = 16
    height = np.full((size, size), 70.0, dtype=np.float32)
    moisture = np.full((size, size), 0.5, dtype=np.float32)
    water = np.full((size, size), -1.0, dtype=np.float32)
    water[7, :] = 74.0
    field = Heightfield(height=height, moisture=moisture, water_level=water)

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="flowing-river",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    water_states = [
        entry
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
        if entry["Name"] == "minecraft:water"
    ]
    assert {entry.get("Properties", {}).get("level") for entry in water_states} >= {"1"}


def test_lake_shore_does_not_raise_water_above_same_height_bank() -> None:
    height = np.asarray(
        [
            [60.8, 59.2, 58.0],
            [60.6, 59.0, 58.0],
            [60.4, 59.1, 58.0],
        ],
        dtype=np.float32,
    )
    water_level = np.where(height < 61.0, 61.0, -1.0).astype(np.float32)
    field = Heightfield(
        height=height,
        moisture=np.full(height.shape, 0.5, dtype=np.float32),
        water_level=water_level,
    )

    surface, water_y, water_flow = _block_heights(field, sea_level=61.0)

    # 连续高度低于湖面但离散后与湖面同高的单元是岸线，不应被抬成 Y=62 的水。
    assert surface[0, 0] == 61
    assert water_y[0, 0] == -1
    assert water_flow[0, 0] == -1
    # 真正低于水面的列仍在统一 Y=61 放置静水源方块。
    assert np.all(water_y[1:, 1:] == 61)
    assert np.all(water_flow[1:, 1:] == 0)


def test_block_heights_keep_the_highest_valid_world_block() -> None:
    """地表可写到 Y=431，不能提前在 Y=430 形成平切。"""

    field = Heightfield(
        height=np.full((16, 16), 431.4, dtype=np.float32),
        moisture=np.full((16, 16), 0.5, dtype=np.float32),
        water_level=np.full((16, 16), -1.0, dtype=np.float32),
    )

    surface, water, flow = _block_heights(field, sea_level=61.0)

    assert np.all(surface == 431), "合法的最高地表方块应保留在 Y=431"
    assert np.all(water == -1)
    assert np.all(flow == -1)


def test_anvil_section_carves_cave_spans_and_overlays_structure_blocks() -> None:
    surface = np.full((16, 16), 80, dtype=np.int16)
    water = np.full((16, 16), -1, dtype=np.int16)
    surface_blocks = np.full((16, 16), 1, dtype=np.uint8)
    spans = np.full((16, 16, 4, 2), 32767, dtype=np.int16)
    spans[:, :, 0] = (74, 80)
    spans[:, :, 1] = (-64, 69)
    structure = np.asarray([[3, 3, 71, OAK_LOG], [4, 3, 71, CHEST]], dtype=np.int32)

    blocks = _section_blocks(4, surface, water, surface_blocks, solid_spans=spans, structure_blocks=structure)

    assert blocks[6, 3, 3] == AIR
    assert blocks[7, 3, 3] == OAK_LOG
    assert blocks[7, 3, 4] == CHEST


def test_anvil_keeps_cave_entrance_open_when_top_span_is_below_surface() -> None:
    surface = np.full((1, 1), 80, dtype=np.int16)
    water = np.full((1, 1), -1, dtype=np.int16)
    surface_blocks = np.full((1, 1), 1, dtype=np.uint8)
    spans = np.full((1, 1, 4, 2), 32767, dtype=np.int16)
    spans[0, 0, 0] = (70, 79)

    blocks = _section_blocks(
        5,
        surface,
        water,
        surface_blocks,
        solid_spans=spans,
    )

    assert blocks[0, 0, 0] == AIR, "入口的 surface_y 应保持空气，不能重新盖回地表方块"

    lower_section = _section_blocks(
        4,
        surface,
        water,
        surface_blocks,
        solid_spans=spans,
    )
    assert lower_section[15, 0, 0] == 1, "入口下方仍应保留顶层实心 span"


def test_export_maps_configured_riverbed_materials_to_minecraft_blocks(tmp_path: Path) -> None:
    size = 16
    height = np.full((size, size), 70.0, dtype=np.float32)
    moisture = np.full((size, size), 0.5, dtype=np.float32)
    water = np.full((size, size), -1.0, dtype=np.float32)
    water[7, :] = 74.0
    riverbed = np.full((size, size), -1, dtype=np.int16)
    riverbed[7, :] = np.arange(size, dtype=np.int16) % 5
    field = Heightfield(
        height=height,
        moisture=moisture,
        water_level=water,
        riverbed_id=riverbed,
        riverbed_palette=("dirt", "mud", "gravel", "sand", "clay"),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="riverbed-materials",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert {
        "minecraft:dirt",
        "minecraft:mud",
        "minecraft:gravel",
        "minecraft:sand",
        "minecraft:clay",
    } <= palette_names


def test_export_maps_real_cave_placeholder_blocks(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        underground_blocks=(
            UndergroundBlock(x=3, y=65, z=3, material="minecraft:coal_ore"),
            UndergroundBlock(x=4, y=65, z=3, material="minecraft:glow_lichen"),
        ),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="cave-placeholders",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert {"minecraft:coal_ore", "minecraft:glow_lichen"} <= palette_names
    lichen_states = [
        entry
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
        if entry["Name"] == "minecraft:glow_lichen"
    ]
    assert lichen_states
    assert lichen_states[0]["Properties"]["up"] == "true"
    assert lichen_states[0]["Properties"]["waterlogged"] == "false"


def test_export_maps_underground_river_plants(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        underground_blocks=(
            UndergroundBlock(x=3, y=65, z=3, material="minecraft:moss_block"),
            UndergroundBlock(x=4, y=65, z=3, material="minecraft:spore_blossom"),
        ),
    )
    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="river-plants",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert {"minecraft:moss_block", "minecraft:spore_blossom"} <= palette_names


def test_export_maps_underground_river_and_lake_water_states(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        underground_water_blocks=(
            UndergroundWaterBlock(x=3, y=65, z=3, kind="river", flowing=True),
            UndergroundWaterBlock(x=4, y=65, z=3, kind="lake", flowing=False),
        ),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=0,
        origin_z=0,
        sea_level=61.0,
        seed=7,
        world_name="underground-water",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.0.0.mca", 0, 0)
    water_states = [
        entry
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
        if entry["Name"] == "minecraft:water"
    ]
    assert water_states
    assert {entry.get("Properties", {}).get("level") for entry in water_states} >= {None, "1"}


def test_export_maps_cave_placeholder_blocks_at_negative_world_coordinates(tmp_path: Path) -> None:
    field = _field()
    field = Heightfield(
        height=field.height,
        moisture=field.moisture,
        water_level=field.water_level,
        underground_blocks=(
            UndergroundBlock(x=-13, y=65, z=-13, material="minecraft:iron_ore"),
        ),
    )

    export_minecraft_world(
        field,
        tmp_path,
        origin_x=-16,
        origin_z=-16,
        sea_level=61.0,
        seed=7,
        world_name="negative-cave-placeholders",
    )
    chunk = _read_region_chunk(tmp_path / "region" / "r.-1.-1.mca", -1, -1)
    palette_names = {
        entry["Name"]
        for section in chunk["sections"]
        for entry in section["block_states"]["palette"]
    }
    assert "minecraft:iron_ore" in palette_names


def test_export_rejects_unknown_riverbed_material(tmp_path: Path) -> None:
    height = np.full((16, 16), 70.0, dtype=np.float32)
    moisture = np.full((16, 16), 0.5, dtype=np.float32)
    water = np.full((16, 16), -1.0, dtype=np.float32)
    water[7, :] = 74.0
    riverbed = np.full((16, 16), -1, dtype=np.int16)
    riverbed[7, :] = 0
    field = Heightfield(
        height=height,
        moisture=moisture,
        water_level=water,
        riverbed_id=riverbed,
        riverbed_palette=("obsidian",),
    )

    with pytest.raises(ValueError, match="no Minecraft block mapping"):
        export_minecraft_world(
            field,
            tmp_path,
            origin_x=0,
            origin_z=0,
            sea_level=61.0,
            seed=7,
            world_name="unknown-riverbed-material",
        )


def test_region_output_is_seed_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "origin_x": 0,
        "origin_z": 0,
        "sea_level": 61.0,
        "seed": 11,
        "world_name": "same",
    }
    export_minecraft_world(_field(), first, **kwargs)
    export_minecraft_world(_field(), second, **kwargs)
    assert (first / "region" / "r.0.0.mca").read_bytes() == (
        second / "region" / "r.0.0.mca"
    ).read_bytes()
    assert (first / "level.dat").read_bytes() == (second / "level.dat").read_bytes()


@pytest.mark.parametrize(
    ("size", "origin_x", "message"),
    ((15, 0, "multiples of 16"), (16, 1, "aligned")),
)
def test_export_rejects_unaligned_inputs(
    tmp_path: Path, size: int, origin_x: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        export_minecraft_world(
            _field(size),
            tmp_path,
            origin_x=origin_x,
            origin_z=0,
            sea_level=61.0,
            seed=1,
            world_name="bad",
        )


def test_bluemap_config_pins_world_bounds_and_download_consent(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config = BlueMapConfig(
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        web_dir=tmp_path / "web",
        world_dir=tmp_path / "world",
        min_x=-16,
        max_x=15,
        min_z=-32,
        max_z=31,
        start_x=0,
        start_z=0,
        port=8123,
        accept_download=True,
    )
    write_bluemap_config(config)

    assert "accept-download: true" in (config_dir / "core.conf").read_text()
    map_config = (config_dir / "maps" / "bong.conf").read_text()
    assert "min-x: -16" in map_config
    assert "max-z: 31" in map_config
    assert "ignore-missing-light-data: true" in map_config
    assert "port: 8123" in (config_dir / "webserver.conf").read_text()
    assert (
        'start-location: "bong:0:220:0:420:0.1:0.19:0:0:perspective"'
        in (config_dir / "webapp.conf").read_text()
    )

    large_config = BlueMapConfig(
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        web_dir=tmp_path / "web",
        world_dir=tmp_path / "world",
        min_x=-512,
        max_x=511,
        min_z=-4160,
        max_z=-3137,
        start_x=0,
        start_z=-3648,
    )
    write_bluemap_config(large_config)
    assert (
        'start-location: "bong:0:220:-3648:1381.05:0.1:0.19:0:0:perspective"'
        in (config_dir / "webapp.conf").read_text()
    )


def test_bluemap_config_installs_grass_preview_resource_pack(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    write_bluemap_config(
        BlueMapConfig(
            config_dir=config_dir,
            data_dir=tmp_path / "data",
            web_dir=tmp_path / "web",
            world_dir=tmp_path / "world",
            min_x=0,
            max_x=15,
            min_z=0,
            max_z=15,
            start_x=8,
            start_z=8,
        )
    )

    pack_dir = config_dir / "packs" / "99_bong_worldgen_preview"
    assert (pack_dir / "pack.mcmeta").is_file()
    colors = json.loads(
        (pack_dir / "assets" / "minecraft" / "blockColors.json").read_text()
    )
    assert colors["minecraft:grass_block"] == "#6f9f5e"
    texture = pack_dir / "assets" / "minecraft" / "textures" / "block" / "grass_block_top.png"
    assert texture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
