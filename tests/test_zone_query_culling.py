"""Spatial culling must preserve every weight, including rotated soft edges."""

from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneIndex
from bong_worldgen.composition.layout import SHAPES, boundary_alpha, boundary_distance
from bong_worldgen.data.zones import ALL_ZONES, SPAWN


def assert_unculled_weights(index, x, z):
    x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
    actual = index.query(x, z)
    remaining = np.ones_like(x)
    names = []
    for zone in index.zones:
        alpha = boundary_alpha(zone, boundary_distance(zone, x, z))
        weight = remaining * alpha
        if np.any(weight):
            names.append(zone.name)
        np.testing.assert_array_equal(actual.weight_for(zone.name), weight)
        remaining *= 1 - alpha
    assert [part.zone.name for part in actual.contributions] == names
    np.testing.assert_array_equal(actual.background, remaining)


@pytest.mark.parametrize("mode,width", (("soft", 128), ("semi_hard", 128), ("hard", 128), ("hard", 0)))
def test_every_shape_and_transition_matches_unculled_boundary_distances(mode, width):
    for shape in sorted(SHAPES):
        zone = replace(SPAWN, shape=shape, center_x=-127.25, center_z=31.5,
                       size_x=9, size_z=160, boundary_mode=mode, boundary_width=width)
        index = ZoneIndex((zone,))
        assert_unculled_weights(index, np.linspace(-290, 40, 173)[None, :],
                                np.linspace(-150, 200, 181)[:, None])
        # Both scalar and array endpoints of the rotated major axis, including
        # the exact boundary and its two adjacent floating-point coordinates.
        x, z = zone.center_x - 40, zone.center_z + 80 * np.cos(np.pi / 6)
        for value in (np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf)):
            assert_unculled_weights(index, value, z)
        assert_unculled_weights(index, np.array([]), 0)


def test_all_authored_overlaps_match_unculled_weights_and_dominance():
    index = ZoneIndex(ALL_ZONES)
    assert_unculled_weights(index, np.linspace(-7000, 8500, 313)[None, :],
                            np.linspace(-8500, 10500, 317)[:, None])
