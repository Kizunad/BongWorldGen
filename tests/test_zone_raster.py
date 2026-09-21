from dataclasses import replace
import json

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile
from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import QINGYUN_PEAKS, SPAWN
from bong_worldgen.engine import TerrainRecipe, generate_heightfield
from bong_worldgen.preview_world import export_preview_world


def test_preview_raster_writes_dominant_zone_ids_and_palette(tmp_path):
    first = replace(SPAWN, name="first", center_x=0, center_z=16, size_x=32, size_z=32)
    second = replace(QINGYUN_PEAKS, name="second", center_x=32, center_z=16,
                     size_x=32, size_z=32)
    world = replace(WORLD, zones=(first, second))
    manifest_path = export_preview_world(
        tmp_path, world=world, min_x=0, max_x=63, min_z=0, max_z=31, tile_size=32,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["zone_palette"] == ["first", "second"]
    assert manifest["zone_encoding"] == {"none": 255, "dtype": "u8"}
    assert "zone_id" in manifest["semantic_layers"]
    assert "zone_id" in manifest["tiles"][0]["layers"]

    left = np.fromfile(tmp_path / "rasters/tile_0_0/zone_id.bin", dtype=np.uint8).reshape(32, 32)
    right = np.fromfile(tmp_path / "rasters/tile_1_0/zone_id.bin", dtype=np.uint8).reshape(32, 32)
    assert set(np.unique(left)) <= {0, 1, 255}
    assert set(np.unique(right)) <= {0, 1, 255}
    assert np.any(left == 0) and np.any(right == 1)
    boundary = np.fromfile(
        tmp_path / "rasters/tile_0_0/boundary_weight.bin", dtype="<f4"
    ).reshape(32, 32)
    assert np.any(boundary > 0)

    composer = ZoneTerrain(world)
    assert composer.index.zone_at(8, 16).name == "first"
    assert composer.index.zone_at(40, 16).name == "second"


def test_zone_palette_and_ids_are_independent_of_world_zone_order(tmp_path):
    first = replace(SPAWN, name="first", center_x=0, center_z=16, size_x=32, size_z=32)
    second = replace(QINGYUN_PEAKS, name="second", center_x=32, center_z=16,
                     size_x=32, size_z=32)
    forward = export_preview_world(
        tmp_path / "forward", world=replace(WORLD, zones=(first, second)),
        min_x=0, max_x=63, min_z=0, max_z=31, tile_size=32,
    )
    reverse = export_preview_world(
        tmp_path / "reverse", world=replace(WORLD, zones=(second, first)),
        min_x=0, max_x=63, min_z=0, max_z=31, tile_size=32,
    )
    forward_manifest = json.loads(forward.read_text(encoding="utf-8"))
    reverse_manifest = json.loads(reverse.read_text(encoding="utf-8"))
    assert forward_manifest["zone_palette"] == reverse_manifest["zone_palette"] == ["first", "second"]
    for tile_name in ("tile_0_0", "tile_1_0"):
        first_ids = np.fromfile(
            tmp_path / "forward" / "rasters" / tile_name / "zone_id.bin", dtype=np.uint8,
        )
        second_ids = np.fromfile(
            tmp_path / "reverse" / "rasters" / tile_name / "zone_id.bin", dtype=np.uint8,
        )
        np.testing.assert_array_equal(first_ids, second_ids)


def test_zone_id_adapter_rejects_negative_and_oversized_contract_values():
    field = generate_heightfield(TerrainRecipe(name="zone_contract"), width=1, height=1, seed=1)
    with pytest.raises(ValueError, match="zone_id"):
        to_bong_tile(field, sea_level=62, zone_id=np.array([[-1]], dtype=np.int16),
                     zone_palette=("zone",))
    with pytest.raises(ValueError, match="at most 255"):
        to_bong_tile(field, sea_level=62, zone_id=np.zeros((1, 1), dtype=np.uint8),
                     zone_palette=tuple(f"zone-{index}" for index in range(256)))
