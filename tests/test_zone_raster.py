from dataclasses import replace
import json

import numpy as np

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import QINGYUN_PEAKS, SPAWN
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
