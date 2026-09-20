from dataclasses import replace
import json

import numpy as np

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SPAWN, QINGYUN_PEAKS, NORTH_WASTES
from bong_worldgen.engine import TerrainRecipe, generate_heightfield
from bong_worldgen.preview_world import export_preview_world


def test_world_generation_uses_zone_profiles_with_real_solid_spans():
    composer = ZoneTerrain(seed=812731)
    fields = [composer.generate(
        width=64, height=64, origin_x=zone.center_x - 256,
        origin_z=zone.center_z - 256, cell_size=8,
    ) for zone in (SPAWN, QINGYUN_PEAKS, NORTH_WASTES)]
    plain, peaks, plateau = [field.height for field in fields]
    assert peaks.std() > plain.std() * 5
    assert plateau.mean() > plain.mean() + 40
    for field in fields:
        np.testing.assert_array_equal(field.solid_spans[..., 0, 1], np.rint(field.height))
        assert np.isfinite(field.water_level).all()


def test_uncovered_world_still_uses_background_recipe_exactly():
    options = dict(width=32, height=32, origin_x=-9000, origin_z=8000)
    actual = ZoneTerrain(seed=19).generate(**options)
    expected = generate_heightfield(DEFAULT_RECIPE, seed=19, **options)
    for layer in ("height", "moisture", "water_level", "solid_spans", "cave_id"):
        np.testing.assert_array_equal(getattr(actual, layer), getattr(expected, layer))


def test_preview_exports_generated_zone_heights_and_actual_membership(tmp_path):
    small_world = replace(WORLD, zones=(SPAWN,))
    manifest_path = export_preview_world(
        tmp_path, world=small_world, recipe=TerrainRecipe(name="background", base_height=180),
        min_x=0, max_x=31, min_z=0, max_z=31, tile_size=32,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generation"]["composer"] == "zone_terrain"
    assert manifest["tiles"][0]["zones"] == ["spawn"]
    spans = np.fromfile(tmp_path / "rasters/tile_0_0/spans.bin", dtype="<i2").reshape(32, 32, 4, 2)
    assert 66 < spans[..., 0, 1].mean() < 74
    assert manifest["generation"]["pending_profiles"] == []
