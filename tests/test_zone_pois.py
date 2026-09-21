from dataclasses import replace
import json
import math

import numpy as np
import pytest

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.composition.pois import resolve_poi, resolve_world_pois
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SPAWN, ZONE_BY_NAME
from bong_worldgen.engine import CaveNetwork, Point, TerrainRecipe
from bong_worldgen.preview_world import export_preview_world


def test_every_authored_poi_stands_on_real_solid_with_headroom_without_mutating_input():
    composer = ZoneTerrain()
    before = tuple(poi.pos_xyz for zone in WORLD.zones for poi in zone.pois)
    resolved = resolve_world_pois(composer)
    assert len(resolved) == len(before)
    for result in resolved:
        x, y, z = result.pos_xyz
        field = composer.generate(width=1, height=1, origin_x=math.floor(x), origin_z=math.floor(z))
        spans = [span for span in field.solid_spans[0, 0] if span[0] != 32767]
        assert any(floor <= y - 1 <= ceiling for floor, ceiling in spans), result
        assert not any(floor <= y + offset <= ceiling for floor, ceiling in spans for offset in (0, 1)), result
        assert (x, z) == (result.poi.pos_xyz[0], result.poi.pos_xyz[2])
    assert before == tuple(poi.pos_xyz for zone in WORLD.zones for poi in zone.pois)
    modes = {result.placement for result in resolved}
    assert {"cave_floor", "surface", "island_surface", "ground", "underwater_surface"} <= modes
    sky = [result for result in resolved if result.zone == "celestial_isles"]
    assert sky[0].pos_xyz[1] > 250 and sky[2].pos_xyz[1] < 100


def test_fractional_negative_poi_uses_exported_voxel_column():
    composer = ZoneTerrain()
    poi = replace(SPAWN.pois[0], pos_xyz=(-0.25, 999, -0.75))
    result = resolve_poi(composer, SPAWN, poi)
    field = composer.generate(width=16, height=16, origin_x=-16, origin_z=-16)
    assert result.pos_xyz[1] == field.solid_spans[-1, -1, 0, 1] + 1


def test_exported_poi_height_matches_raster_and_preserves_original_position(tmp_path):
    poi = replace(SPAWN.pois[0], pos_xyz=(4, 999, 7))
    world = replace(WORLD, zones=(replace(SPAWN, pois=(poi,)),))
    path = export_preview_world(tmp_path, world=world, min_x=0, max_x=31, min_z=0, max_z=31, tile_size=32)
    manifest = json.loads(path.read_text())
    result = manifest["pois"][0]
    spans = np.fromfile(tmp_path / "rasters/tile_0_0/spans.bin", dtype="<i2").reshape(32, 32, 4, 2)
    assert result["pos_xyz"][1] == int(spans[7, 4, 0, 1]) + 1
    assert result["authored_pos_xyz"] == [4, 999, 7]
    assert result["placement"] == "surface"


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_ground_tag_below_sky_island_selects_ground_above_a_cave(seed):
    zone = replace(ZONE_BY_NAME["celestial_isles"], center_x=-1, center_z=-1,
                   size_x=512, size_z=512, pois=())
    poi = replace(SPAWN.pois[0], name="ground ruin", pos_xyz=(-0.25, 72, -0.25), tags=("ground",))
    cave = CaveNetwork(name="under_island", paths=((Point(-1, -65), Point(-1, 63)),),
                       width=8, height=8, depth=30, branch_count=0, chamber_count=0,
                       entrance_count=0, noise_strength=0, dead_end_strength=0,
                       vertical_warp=0, domain_warp_strength=0, smooth_union=0)
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), seed=seed,
                           background=TerrainRecipe(name="ground_with_cave", caves=(cave,)))
    field = composer.generate(width=1, height=1, origin_x=-1, origin_z=-1)
    column = field.solid_spans[0, 0]
    assert np.count_nonzero(column[:, 0] != 32767) == 3  # Island, surface roof, cave base.
    result = resolve_poi(composer, zone, poi)
    assert result.pos_xyz == (-0.25, float(column[1, 1] + 1), -0.25)
    assert result.placement == "ground"
    assert result.pos_xyz[1] > column[2, 1] + 20
    assert result.pos_xyz[1] + 1 < column[0, 0]
