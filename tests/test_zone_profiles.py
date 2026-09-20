from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import ZONE_BY_NAME


def _composer(name, seed=812731):
    zone = ZONE_BY_NAME[name]
    return zone, ZoneTerrain(replace(WORLD, zones=(zone,)), seed=seed)


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_spring_marsh_has_shallow_water_and_dry_banks(seed):
    zone, composer = _composer("lingquan_marsh", seed)
    field = composer.generate(width=100, height=100, origin_x=zone.center_x - 400,
                              origin_z=zone.center_z - 400, cell_size=8)
    wet = field.water_level >= 0
    assert 0.15 < wet.mean() < 0.85
    assert np.median(field.water_level[wet] - field.height[wet]) < 6
    assert field.height[50, 50] < 61


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_rotated_rift_has_deep_floor_between_raised_shoulders(seed):
    zone, composer = _composer("blood_valley", seed)
    x = np.array([-120, 0, 120]) * np.cos(np.pi / 6)
    z = np.array([-120, 0, 120]) * np.sin(np.pi / 6)
    heights, _ = composer.sample_surface(x + zone.center_x, z + zone.center_z)
    assert heights[0] - heights[1] > 30
    assert heights[2] - heights[1] > 30


@pytest.mark.parametrize("zone_name", ("drift_scorch_001", "blood_valley_east_scorch", "rift_mouth_blood_001"))
def test_scorch_and_rift_mouth_form_craters_with_raised_rims(zone_name):
    zone, composer = _composer(zone_name)
    offsets = np.array([0, 0.26 * zone.size_x, 0.40 * zone.size_x])
    heights, _ = composer.sample_surface(zone.center_x + offsets, np.full(3, zone.center_z))
    assert heights[1] - heights[0] > 25
    assert heights[1] - heights[2] > 8


def test_ash_land_has_eroded_relief_and_is_different_from_background():
    zone, composer = _composer("south_ash_dead_zone")
    field = composer.generate(width=100, height=100, origin_x=zone.center_x - 400,
                              origin_z=zone.center_z - 400, cell_size=8)
    assert field.height.std() > 3.0
    assert np.ptp(field.height) > 20
    assert (field.water_level < 0).mean() > 0.9


def test_battlefield_has_low_relief_and_multiple_depressions():
    zone, composer = _composer("zhanhun_plain")
    field = composer.generate(width=100, height=100, origin_x=zone.center_x - 500,
                              origin_z=zone.center_z - 500, cell_size=10)
    assert 2 < field.height.std() < 10
    assert np.quantile(field.height, 0.05) < 73
    assert np.quantile(field.height, 0.95) > 79


@pytest.mark.parametrize("name", [name for name in ZONE_BY_NAME if name.startswith("jiuzong_")])
def test_sect_ruins_have_a_level_raised_central_platform(name):
    zone, composer = _composer(name)
    field = composer.generate(width=32, height=32, origin_x=zone.center_x - 16,
                              origin_z=zone.center_z - 16)
    assert np.ptp(field.height) < 0.01
    outer, _ = composer.sample_surface(np.array([zone.center_x + zone.size_x * 0.35]),
                                      np.array([zone.center_z]))
    assert field.height.mean() - outer[0] > 12


def test_garden_has_three_descending_cultivation_terraces():
    zone, composer = _composer("dan_zong_yi_yuan")
    heights, _ = composer.sample_surface(np.full(3, zone.center_x),
                                        zone.center_z + np.array([-0.23, 0, 0.23]) * zone.size_z)
    np.testing.assert_allclose(heights, [92, 86, 80])


def test_wangyintai_has_two_level_concentric_platforms():
    zone, composer = _composer("wangyintai")
    heights, _ = composer.sample_surface(zone.center_x + np.array([0, 70, 230, 270, 420]),
                                        np.full(5, zone.center_z))
    np.testing.assert_allclose(heights[:4], [104, 104, 91, 91])
    assert heights[4] < 81
