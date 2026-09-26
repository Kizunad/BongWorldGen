from dataclasses import replace
import struct
import zlib

import numpy as np
import pytest

from bong_worldgen.adapters import export_minecraft_world, generate_zone_tile
from bong_worldgen.adapters.anvil_nbt import read_root_compound
from bong_worldgen.adapters.minecraft_world import chunk_index_in_region
from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.composition.surface import scorch_mask
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SPAWN, ZONE_BY_NAME
from bong_worldgen.engine import Point, River, TerrainRecipe
from bong_worldgen.preview_world import export_preview_world


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name", ("drift_scorch_001", "blood_valley_east_scorch", "north_waste_east_scorch"))
def test_impact_floors_are_charred_but_raised_rims_and_outer_ground_are_not(name, seed):
    zone = ZONE_BY_NAME[name]
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), seed=seed)
    for x, z, charred in ((0, 0, True), (-0.27, -0.20, True), (0.20, -0.28, True),
                           (-0.28, 0.26, True), (0.34, 0.10, True),
                           (0, 0.23, False), (0.26, 0, False), (0.48, 0, False)):
        _, tile = generate_zone_tile(composer, width=1, height=1,
                                     origin_x=zone.center_x + x * zone.size_x,
                                     origin_z=zone.center_z + z * zone.size_z)
        assert (tile.surface_palette[tile.surface_id[0, 0]] == "blackstone") == charred


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_charred_patch_edge_is_identical_across_fractional_negative_tiles(seed):
    composer = ZoneTerrain(seed=seed)
    # Cross the central bowl's ragged western edge at negative world X.
    options = dict(height=24, origin_x=-4115.5, origin_z=3973.25, cell_size=1.25)
    _, whole = generate_zone_tile(composer, width=80, **options)
    parts = [generate_zone_tile(composer, width=width,
             **{**options, "origin_x": options["origin_x"] + offset * 1.25})[1]
             for offset, width in ((0, 19), (19, 1), (20, 60))]
    np.testing.assert_array_equal(whole.surface_id, np.concatenate([p.surface_id for p in parts], axis=1))
    assert np.any(whole.surface_id == whole.surface_palette.index("blackstone"))
    assert np.any(whole.surface_id != whole.surface_palette.index("blackstone"))


def test_charred_surface_respects_overlaid_zones_and_authored_riverbeds():
    zone = replace(ZONE_BY_NAME["drift_scorch_001"], center_x=0, center_z=0)
    overlay = replace(SPAWN, size_x=100, size_z=100, boundary_width=20)
    composer = ZoneTerrain(replace(WORLD, zones=(zone, overlay)))
    assert not scorch_mask(composer, np.array([0]), np.array([0]))[0]
    background = TerrainRecipe(name="river", rivers=(River(path=(Point(-150, 0), Point(150, 0)),
                                 width=4, depth=4, bed_materials=("clay",)),))
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), background=background)
    _, tile = generate_zone_tile(composer, width=1, height=17, origin_x=0, origin_z=-8)
    assert tile.surface_palette[tile.surface_id[8, 0]] == "clay"
    assert tile.surface_palette[tile.surface_id[0, 0]] == "blackstone"


def test_preview_and_anvil_write_blackstone_at_the_actual_impact_surface(tmp_path):
    zone = ZONE_BY_NAME["drift_scorch_001"]
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)))
    origin_x, origin_z = -4000, 4000
    field, tile = generate_zone_tile(composer, width=16, height=16,
                                     origin_x=origin_x, origin_z=origin_z)
    output = tmp_path / "anvil"
    export_minecraft_world(field, output, origin_x=origin_x, origin_z=origin_z,
                           sea_level=61, seed=composer.seed, world_name="scorch", surface_tile=tile)
    cx, cz = origin_x // 16, origin_z // 16
    region = (output / "region" / f"r.{cx >> 5}.{cz >> 5}.mca").read_bytes()
    index = chunk_index_in_region(cx, cz)
    offset = int.from_bytes(region[index * 4:index * 4 + 3], "big") * 4096
    length = struct.unpack_from(">I", region, offset)[0]
    chunk = read_root_compound(zlib.decompress(region[offset + 5:offset + 4 + length]))
    top = int(np.rint(field.height[0, 0]))
    section = next(s for s in chunk["sections"] if s["Y"] == top // 16)
    states = section["block_states"]
    bits = max(4, (len(states["palette"]) - 1).bit_length())
    voxel_index = (top % 16) * 256
    word = states["data"][voxel_index // (64 // bits)] & ((1 << 64) - 1)
    block = (word >> ((voxel_index % (64 // bits)) * bits)) & ((1 << bits) - 1)
    assert states["palette"][block]["Name"] == "minecraft:blackstone"

    export_preview_world(tmp_path / "raster", world=composer.world,
                         min_x=origin_x, max_x=origin_x + 15,
                         min_z=origin_z, max_z=origin_z + 15, tile_size=16)
    stored = np.fromfile(tmp_path / "raster" / "rasters" / f"tile_{cx}_{cz}" / "surface_id.bin", dtype="u1")
    np.testing.assert_array_equal(stored.reshape(16, 16), tile.surface_id)


def test_mismatched_surface_tile_is_rejected_before_export(tmp_path):
    field, tile = generate_zone_tile(ZoneTerrain(), width=16, height=16, origin_x=-4000, origin_z=4000)
    with pytest.raises(ValueError, match="match the exported heightfield"):
        export_minecraft_world(field, tmp_path / "invalid", origin_x=-4000, origin_z=4000,
                               sea_level=61, seed=7, world_name="invalid",
                               surface_tile=replace(tile, height=tile.height + 1))
    assert not (tmp_path / "invalid").exists()
