"""Spatial contracts: similar elevation distributions must not hide identical shapes."""

from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import ZONE_BY_NAME


def _surface(name, seed, points):
    zone = ZONE_BY_NAME[name]
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), seed=seed)
    points = np.asarray(points)
    return composer.sample_surface(zone.center_x + points[..., 0] * zone.size_x,
                                   zone.center_z + points[..., 1] * zone.size_z)[0]


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name", ("drift_scorch_001", "blood_valley_east_scorch"))
def test_scorch_has_outward_branching_scars_beyond_a_broken_impact_rim(name, seed):
    # Three rays and a fork well outside the central bowl. Adjacent terrain
    # must be higher on BOTH sides, so a bowl or a closed ring cannot pass.
    probes = np.array([[-0.31, 0.086], [0.205, 0.298], [-0.116, -0.306],
                       [0.075, 0.337]])
    normals = np.array([[0, 0.055], [0.045, -0.04], [-0.045, 0.04], [0.05, 0]])
    floor = _surface(name, seed, probes)
    banks = np.minimum(_surface(name, seed, probes + normals),
                       _surface(name, seed, probes - normals))
    assert np.all(banks - floor > 5), banks - floor

    # Raised eastern ejecta, open south side: no continuous enclosing rim.
    east, south = _surface(name, seed, [(0.26, 0), (0, -0.27)])
    assert east - south > 16


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name", ("rift_mouth_blood_001", "rift_mouth_north_001"))
def test_rift_mouth_has_a_long_open_fracture_and_unequal_scarps(name, seed):
    # Four cross-sections span most of the footprint, including outside the
    # old circular crater. The fracture remains lower than both shoulders.
    floor_points = np.array([[-0.02, -0.32], [0.0165, -0.11],
                             [0.006, 0.29], [-0.028, 0.40]])
    floor = _surface(name, seed, floor_points)
    west = _surface(name, seed, floor_points + [-0.10, 0])
    east = _surface(name, seed, floor_points + [0.10, 0])
    assert np.all(np.minimum(west, east) - floor > 10)
    west_scarp, east_scarp = _surface(name, seed, [(-0.26, 0.22), (0.215, 0.22)])
    assert east_scarp - west_scarp > 15


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name", [n for n in ZONE_BY_NAME if n.startswith("jiuzong_")])
def test_sect_ruin_keeps_rectilinear_foundations_separated_by_open_courtyards(name, seed):
    corners = _surface(name, seed, [(-0.09, -0.08), (0.09, -0.08),
                                   (-0.09, 0.08), (0.09, 0.08)])
    courts = _surface(name, seed, [(-0.18, 0), (0.18, 0), (0, -0.18), (0, 0.18)])
    np.testing.assert_array_equal(corners, 96)
    assert np.all(corners - courts > 12)
    wall, breach = _surface(name, seed, [(0.26, -0.1), (0.255, 0.035)])
    assert wall - breach > 6  # Eastern wall is actually broken, not a closed enclosure.


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_garden_terraces_are_parallel_rows_of_five_separate_flat_beds(seed):
    xs = np.linspace(-0.34, 0.34, 341)
    for z, h in ((-0.23, 92), (0, 86), (0.23, 80)):
        for offset in (-0.04, 0.04):
            heights = _surface("dan_zong_yi_yuan", seed,
                               np.column_stack((xs, np.full_like(xs, z + offset))))
            bed = heights > h - 0.1
            starts = np.flatnonzero(np.diff(np.r_[False, bed, False].astype(int)) == 1)
            assert len(starts) == 5
            assert np.min(heights) <= h - 3.9
        beds = _surface("dan_zong_yi_yuan", seed, [(x, z) for x in (-0.26, -0.13, 0, 0.13, 0.26)])
        np.testing.assert_array_equal(beds, h)


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_observation_platform_has_diamond_corners_and_an_axial_stair(seed):
    heights = _surface("wangyintai", seed, [(0.28, 0), (-0.28, 0), (0.2, 0.2), (-0.2, 0.2)])
    np.testing.assert_array_equal(heights[:2], 91)
    assert np.all(heights[2:] < 82)  # Same radius, but outside the oblique straight edge.
    stair = _surface("wangyintai", seed, [(0, z) for z in (0, 0.235, 0.30, 0.365)])
    np.testing.assert_allclose(stair, [104, 99, 95, 85])


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_battlefield_contains_crossing_scars_craters_and_a_burial_mound(seed):
    name = "zhanhun_plain"
    # Sections on both strike directions, away from their crossing.
    probes = np.array([[-0.29, 0.0834], [0.245, -0.1282], [0.29, -0.1547],
                       [-0.1015, -0.074], [0.143, 0.271]])
    normals = np.array([[0.02, 0.05], [0.02, 0.05], [0.02, 0.05],
                        [0.05, -0.04], [0.05, -0.03]])
    floor = _surface(name, seed, probes)
    banks = np.minimum(_surface(name, seed, probes + normals),
                       _surface(name, seed, probes - normals))
    assert np.all(banks - floor > 6)

    centers = np.array([[-0.31, 0.20], [0.24, 0.18], [0.05, -0.27]])
    angles = np.arange(8) * np.pi / 4
    ring = centers[:, None, :] + 0.11 * np.column_stack((np.cos(angles), np.sin(angles)))
    assert np.all(np.median(_surface(name, seed, ring), axis=1) - _surface(name, seed, centers) > 8)

    mound, west, east = _surface(name, seed, [(0.1, -0.125), (0.04, -0.125), (0.16, -0.125)])
    assert mound - max(west, east) > 10
    remnants = _surface(name, seed, [(-0.255, -0.24), (-0.19, -0.25), (-0.145, -0.19)])
    broken_core = _surface(name, seed, [(-0.20, -0.1875)])[0]
    assert np.all(remnants - broken_core > 15)


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name", ("youan_depths", "baolongwang_cavern_deep"))
def test_cave_network_has_separate_sinkholes_with_intact_ground_between_them(name, seed):
    centers = np.array([[-0.23, 0.16], [0.25, -0.22], [0.18, 0.21]])
    angles = np.arange(8) * np.pi / 4
    rims = centers[:, None, :] + 0.13 * np.column_stack((np.cos(angles), np.sin(angles)))
    floors = _surface(name, seed, centers)
    assert np.all(np.median(_surface(name, seed, rims), axis=1) - floors > 10)
    # The outer depressions remain isolated, separated by elevated saddles.
    saddles = _surface(name, seed, (centers[:2] + centers[2]) / 2)
    assert np.all(saddles - np.maximum(floors[:2], floors[2]) > 8)


@pytest.mark.parametrize("seed", (7, 812731, 2026))
@pytest.mark.parametrize("name,entrance", (
    ("youan_depths", (2000, 3000)),
    ("baolongwang_cavern_deep", (1610, -5210)),  # Existing fallback entrance.
))
def test_cave_surface_collapse_marks_the_real_entrance(name, entrance, seed):
    zone = ZONE_BY_NAME[name]
    composer = ZoneTerrain(replace(WORLD, zones=(zone,)), seed=seed)
    angles = np.arange(8) * np.pi / 4
    ring = np.asarray(entrance) + min(zone.size_x, zone.size_z) * 0.13 * np.column_stack(
        (np.cos(angles), np.sin(angles)))
    rim = composer.sample_surface(ring[:, 0], ring[:, 1])[0]
    floor = composer.sample_surface(*entrance)[0]
    assert np.median(rim) - floor > 10
    # The depression leads into a real open shaft; adjacent ground retains rock.
    field = composer.generate(width=13, height=13, origin_x=entrance[0] - 6,
                              origin_z=entrance[1] - 6)
    assert field.solid_spans[6, 6, 0, 1] < field.height[6, 6] - 20
    assert field.solid_spans[0, 0, 0, 1] == np.rint(field.height[0, 0])


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_abyss_has_a_bent_trough_with_offset_unequal_escarpments(seed):
    composer = ZoneTerrain(seed=seed)
    # World-space sections between the main door and the well, beyond the
    # original bowl. A broad centered depression cannot produce these shoulders.
    points = np.array([[5263.56, 1502.59], [5377.90, 1664.18]])
    normal = np.array([-2, 1]) / np.sqrt(5)
    floor = composer.sample_surface(points[:, 0], points[:, 1])[0]
    lower_points = points + normal * 112.5
    upper_points = points - normal * 142.5
    lower = composer.sample_surface(lower_points[:, 0], lower_points[:, 1])[0]
    upper = composer.sample_surface(upper_points[:, 0], upper_points[:, 1])[0]
    assert np.all(np.minimum(lower, upper) - floor > 20)
    assert np.all(upper - lower > 8)
    # The taller shoulder breaks before the well; the opposite one persists.
    end = np.array([5541.38, 2033.06])
    broken, surviving = end - normal * 142.5, end + normal * 112.5
    assert composer.sample_surface(*surviving)[0] - composer.sample_surface(*broken)[0] > 25


@pytest.mark.parametrize("profile", ("cave_network", "abyssal_maze"))
def test_surface_collapse_follows_relocated_entrance_data(profile):
    zone = ZONE_BY_NAME["youan_depths" if profile == "cave_network" else "wuxing_abyss"]
    entrance = zone.pois[0]
    # Move the existing entrance to intact ground, leaving the zone center fixed.
    moved = replace(entrance, pos_xyz=(zone.center_x - 0.3 * zone.size_x,
                                      entrance.pos_xyz[1], zone.center_z - 0.25 * zone.size_z))
    moved_zone = replace(zone, pois=(moved, *zone.pois[1:]))
    original = ZoneTerrain(replace(WORLD, zones=(zone,)))
    updated = ZoneTerrain(replace(WORLD, zones=(moved_zone,)))
    x, _, z = moved.pos_xyz
    before = original.sample_surface(x, z)[0]
    after = updated.sample_surface(x, z)[0]
    assert before - after > 10
    column = updated.generate(width=1, height=1, origin_x=x, origin_z=z)
    assert column.solid_spans[0, 0, 0, 1] < after - 20
