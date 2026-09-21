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


def test_zone_raster_matches_point_queries_and_background_across_tile_sizes(tmp_path):
    first = replace(SPAWN, name="first", center_x=-4, center_z=16,
                    size_x=24, size_z=24, boundary_width=8, pois=())
    second = replace(QINGYUN_PEAKS, name="second", shape="circular", center_x=12,
                     center_z=16, size_x=16, size_z=16, boundary_width=8, pois=())
    world = replace(WORLD, zones=(first, second))
    composer = ZoneTerrain(world)
    expected = np.full((32, 64), 255, dtype=np.uint8)
    for z in range(32):
        for x in range(-32, 32):
            zone = composer.index.zone_at(x, z)
            if zone is not None:
                expected[z, x + 32] = ("first", "second").index(zone.name)
    assert set(np.unique(expected)) == {0, 1, 255}
    # An exact half-weight tie at the outer contour belongs to background.
    assert expected[16, -16 + 32] == 255

    for tile_size in (32, 64):
        output = tmp_path / str(tile_size)
        manifest_path = export_preview_world(
            output, world=world, recipe=TerrainRecipe(name="background"),
            min_x=-32, max_x=31, min_z=0, max_z=31, tile_size=tile_size,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["zone_palette"] == ["first", "second"]
        parts = [np.fromfile(output / "rasters" / f"tile_{x}_0/zone_id.bin",
                             dtype=np.uint8).reshape(tile_size, tile_size)
                 for x in (-1, 0)]
        actual = np.concatenate(parts, axis=1)[:32, tile_size - 32:tile_size + 32]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tile_size", (16, 64))
def test_overview_samples_declared_coordinates_and_matches_tile_zone_ids(tmp_path, tile_size):
    first = replace(SPAWN, name="first", center_x=-13, center_z=11,
                    size_x=48, size_z=48, boundary_width=8, pois=())
    second = replace(QINGYUN_PEAKS, name="second", center_x=51, center_z=43,
                     size_x=40, size_z=40, boundary_width=8, pois=())
    background = TerrainRecipe(name="background")
    world = replace(WORLD, zones=(first, second))
    manifest_path = export_preview_world(
        tmp_path, world=world, recipe=background, min_x=-45, max_x=51,
        min_z=-21, max_z=43, tile_size=tile_size,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overview = manifest["overview"]
    assert (overview["origin_x"], overview["origin_z"], overview["cell_size"]) == (-45, -21, 32)
    assert (overview["width"], overview["height"]) == (4, 3)
    raster_dir = manifest_path.parent
    zone_ids = np.fromfile(raster_dir / overview["zone_file"], dtype=np.uint8).reshape(3, 4)
    assert set(np.unique(zone_ids)) == {0, 1, 255}
    for oz in range(3):
        for ox in range(4):
            x, z = -45 + ox * 32, -21 + oz * 32
            tile_x, local_x = divmod(x, tile_size)
            tile_z, local_z = divmod(z, tile_size)
            layer = np.fromfile(raster_dir / f"tile_{tile_x}_{tile_z}/zone_id.bin",
                                dtype=np.uint8).reshape(tile_size, tile_size)
            assert zone_ids[oz, ox] == layer[local_z, local_x]

    composer = ZoneTerrain(world, background=background)
    field = composer.generate(width=4, height=3, origin_x=-45, origin_z=-21, cell_size=32)
    expected = to_bong_tile(field, sea_level=background.sea_level)
    for key, values, dtype in (("height_file", field.height, "<f4"),
                               ("surface_file", expected.surface_id, "u1"),
                               ("wilderness_file", expected.wilderness_id, "u1")):
        actual = np.fromfile(raster_dir / overview[key], dtype=dtype).reshape(3, 4)
        np.testing.assert_array_equal(actual, values)
