from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import ZONE_BY_NAME
from bong_worldgen.engine import TerrainRecipe


def _local_composer(name):
    return ZoneTerrain(replace(WORLD, zones=(ZONE_BY_NAME[name],)),
                       background=TerrainRecipe(name="background"))


@pytest.mark.parametrize("name", ("youan_depths", "baolongwang_cavern_deep", "wuxing_abyss"))
def test_authored_underground_rooms_have_real_air_above_a_solid_floor(name):
    zone = ZONE_BY_NAME[name]
    composer = _local_composer(name)
    rooms = [poi for poi in zone.pois if poi.kind != "cave_mouth" and "bottomless" not in poi.tags]
    for poi in rooms:
        field = composer.generate(width=1, height=1, origin_x=poi.pos_xyz[0], origin_z=poi.pos_xyz[2])
        spans = field.solid_spans[0, 0]
        valid = spans[spans[:, 0] != 32767]
        assert len(valid) >= 2
        assert np.max(valid[:-1, 0] - valid[1:, 1] - 1) >= 4
        assert field.cave_id[0, 0] > 0
        assert np.all(valid[:, 0] >= -64)
        if name == "wuxing_abyss":
            assert len(valid) == 4  # Three real layers, not three metadata labels.


def test_cave_entrance_opens_surface_and_connects_to_air_gap():
    zone = ZONE_BY_NAME["youan_depths"]
    field = _local_composer(zone.name).generate(width=9, height=9, origin_x=zone.center_x - 4,
                                              origin_z=zone.center_z - 4)
    assert np.any(field.solid_spans[..., 0, 1] < np.rint(field.height) - 10)
    assert np.any(field.solid_spans[..., 0, 1] == np.rint(field.height))


def test_multilevel_cave_spans_and_ids_are_identical_when_generated_as_tiles():
    composer = _local_composer("wuxing_abyss")
    full = composer.generate(width=40, height=16, origin_x=5180, origin_z=1472)
    parts = [composer.generate(width=20, height=16, origin_x=x, origin_z=1472) for x in (5180, 5200)]
    for layer in ("height", "solid_spans", "cave_id"):
        np.testing.assert_array_equal(getattr(full, layer),
                                      np.concatenate([getattr(part, layer) for part in parts], axis=1))
    assert np.any(full.cave_id)


@pytest.mark.parametrize("name,origin_x,origin_z", (
    ("youan_depths", 1872, 2864), ("wuxing_abyss", 5120, 1264),
))
def test_full_preview_cave_walls_and_shafts_fit_the_four_span_contract(name, origin_x, origin_z):
    # This includes tangential wall columns missed by room-center checks.
    # Previously the 256-block BlueMap window raised a span overflow here.
    field = ZoneTerrain().generate(width=64, height=64, origin_x=origin_x,
                                   origin_z=origin_z, cell_size=4)
    spans = field.solid_spans
    counts = np.count_nonzero(spans[..., 0] != 32767, axis=-1)
    assert np.max(counts) <= 4
    assert np.any(counts >= (4 if name == "wuxing_abyss" else 2))
    assert np.all(spans[..., 0, 1] <= np.rint(field.height))
