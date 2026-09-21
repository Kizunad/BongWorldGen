from dataclasses import replace
import json

import numpy as np
import pytest

from bong_worldgen.adapters import generate_zone_tile, to_bong_tile
from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SOUTH_ASH_DEAD_ZONE
from bong_worldgen.engine import Heightfield
from bong_worldgen.preview_world import export_preview_world


@pytest.mark.parametrize("cell_size", (1, 4))
def test_zone_semantic_layers_match_whole_window_at_internal_edges_and_single_columns(cell_size):
    composer = ZoneTerrain()
    origin_x, origin_z = -1232, 7968
    field, full = generate_zone_tile(composer, width=64, height=64, origin_x=origin_x,
                                     origin_z=origin_z, cell_size=cell_size)
    assert set(np.unique(full.wilderness_id)) == {0, 1}
    original = composer.generate(width=64, height=64, origin_x=origin_x,
                                 origin_z=origin_z, cell_size=cell_size)
    np.testing.assert_array_equal(field.height, original.height)
    np.testing.assert_array_equal(field.solid_spans, original.solid_spans)
    for x, z, width, height in ((0, 0, 32, 64), (32, 0, 32, 64),
                                (31, 22, 1, 1), (20, 31, 16, 1)):
        _, part = generate_zone_tile(composer, width=width, height=height,
                                     origin_x=origin_x + x * cell_size,
                                     origin_z=origin_z + z * cell_size, cell_size=cell_size)
        for layer in ("height", "surface_id", "wilderness_id", "water_level",
                      "solid_spans", "zone_id", "boundary_weight", "cave_id"):
            np.testing.assert_array_equal(getattr(part, layer),
                                          getattr(full, layer)[z:z + height, x:x + width])


def test_preview_wilderness_and_overview_are_independent_of_tile_size(tmp_path):
    world = replace(WORLD, zones=(replace(SOUTH_ASH_DEAD_ZONE, pois=()),))
    layers = []
    for tile_size in (32, 64):
        manifest_path = export_preview_world(
            tmp_path / str(tile_size), world=world, min_x=-1248, max_x=-1185,
            min_z=7968, max_z=8031, tile_size=tile_size,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tile_cache = {}
        assembled = np.empty((64, 64), dtype=np.uint8)
        for z in range(64):
            for x in range(64):
                tx, lx = divmod(-1248 + x, tile_size)
                tz, lz = divmod(7968 + z, tile_size)
                if (tx, tz) not in tile_cache:
                    tile_cache[tx, tz] = np.fromfile(
                        manifest_path.parent / f"tile_{tx}_{tz}/wilderness_id.bin", dtype=np.uint8,
                    ).reshape(tile_size, tile_size)
                assembled[z, x] = tile_cache[tx, tz][lz, lx]
        coarse = np.fromfile(manifest_path.parent / manifest["overview"]["wilderness_file"],
                             dtype=np.uint8).reshape(2, 2)
        np.testing.assert_array_equal(coarse, assembled[::32, ::32])
        layers.append(assembled)
    np.testing.assert_array_equal(*layers)


def test_supplied_world_slope_classifies_single_columns_and_rejects_invalid_values():
    field = Heightfield(height=np.array([[85.0]], dtype=np.float32),
                        water_level=np.array([[-1.0]], dtype=np.float32),
                        moisture=np.array([[0.5]], dtype=np.float32))
    assert to_bong_tile(field, sea_level=61, surface_slope=np.array([[0.4]])).wilderness_id == 1
    assert to_bong_tile(field, sea_level=61, surface_slope=np.array([[0.2]])).wilderness_id == 0
    for slope in (np.array([[np.nan]]), np.array([[-1.0]]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="surface_slope"):
            to_bong_tile(field, sea_level=61, surface_slope=slope)
