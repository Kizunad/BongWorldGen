from dataclasses import replace

import numpy as np

from bong_worldgen.adapters.anvil_nbt import AIR, STONE, _section_blocks
from bong_worldgen.composition import PROFILE_RECIPES, ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import CELESTIAL_ISLES
from bong_worldgen.engine import TerrainRecipe


def _composer():
    return ZoneTerrain(replace(WORLD, zones=(CELESTIAL_ISLES,)),
                       background=TerrainRecipe(name="ground"))


def test_every_authored_profile_has_an_implementation():
    assert len(PROFILE_RECIPES) == 15
    assert {zone.terrain_profile for zone in WORLD.zones} == set(PROFILE_RECIPES)


def test_sky_isle_is_detached_above_ground_and_survives_anvil_block_conversion():
    field = _composer().generate(width=16, height=16, origin_x=-4400, origin_z=1200)
    spans = field.solid_spans
    assert np.all(spans[..., 0, 0] > spans[..., 1, 1] + 150)
    assert np.all(spans[..., 1, 0] == -64)
    assert field.height.min() > 285 and field.height.max() < 318
    air_section = _section_blocks(10, np.rint(field.height).astype(np.int16),
                                  np.full((16, 16), -1), np.full((16, 16), STONE),
                                  solid_spans=spans)
    assert np.all(air_section == AIR)  # y=160..175 is real air below the island.
    rock_section = _section_blocks(17, np.rint(field.height).astype(np.int16),
                                   np.full((16, 16), -1), np.full((16, 16), STONE),
                                   solid_spans=spans)
    assert np.any(rock_section == STONE)


def test_island_rim_and_ground_remnant_are_not_solid_cliff_walls():
    field = _composer().generate(width=900, height=1, origin_x=-4850, origin_z=1200)
    counts = np.count_nonzero(field.solid_spans[..., 0] != 32767, axis=-1)[0]
    assert counts[0] == counts[-1] == 1
    assert counts[450] == 2
    active = counts == 2
    thickness = field.solid_spans[0, active, 0, 1] - field.solid_spans[0, active, 0, 0]
    assert thickness.min() < 5 and thickness.max() >= 35
    ground = _composer().generate(width=1, height=1, origin_x=-4800, origin_z=800)
    assert ground.height[0, 0] < 100
    assert ground.solid_spans[0, 0, 1, 0] == 32767


def test_island_spans_are_identical_across_tile_edge():
    composer = _composer()
    field = composer.generate(width=32, height=8, origin_x=-4016, origin_z=1200)
    left = composer.generate(width=16, height=8, origin_x=-4016, origin_z=1200)
    right = composer.generate(width=16, height=8, origin_x=-4000, origin_z=1200)
    for layer in ("height", "solid_spans", "water_level"):
        np.testing.assert_array_equal(getattr(field, layer),
                                      np.concatenate([getattr(left, layer), getattr(right, layer)], axis=1))
