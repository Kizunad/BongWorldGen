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
