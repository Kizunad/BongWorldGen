from __future__ import annotations

import gzip
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from bong_worldgen.adapters.anvil_nbt import (
    AIR,
    CHEST,
    DATA_VERSION,
    OAK_LOG,
    read_root_compound,
    _section_blocks,
)
from bong_worldgen.adapters.minecraft_world import (
    chunk_index_in_region,
    export_minecraft_world,
    region_for_chunk,
)
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config
from bong_worldgen.engine.models import Heightfield


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

    level = read_root_compound(gzip.decompress((tmp_path / "level.dat").read_bytes()))
    assert level["Data"]["DataVersion"] == DATA_VERSION
    assert level["Data"]["LevelName"] == "test-world"
    assert level["Data"]["RandomSeed"] == 7


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
        'start-location: "bong:0:80:0:420:0.1:0.19:0:0:perspective"'
        in (config_dir / "webapp.conf").read_text()
    )
