from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.composition import ZoneIndex, recipe_for_zone
from bong_worldgen.composition.layout import SHAPES, boundary_distance
from bong_worldgen.data.zones import SPAWN, QINGYUN_PEAKS, NORTH_WASTES, ALL_ZONES
from bong_worldgen.engine import generate_heightfield


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_footprints_contain_center_and_exclude_distant_points(shape):
    zone = replace(SPAWN, shape=shape, center_x=-700, center_z=900, size_x=100, size_z=160)
    index = ZoneIndex((zone,))
    assert index.zone_at(-700, 900) == zone
    assert index.zone_at(-500, 900) is None
    assert index.zone_at(-700, 650) is None


def test_ellipse_sizes_are_full_diameters_in_world_coordinates():
    zone = replace(SPAWN, center_x=-700, center_z=900, size_x=100, size_z=160)
    distance = boundary_distance(zone, np.array([-750, -700, -650]), np.array([900, 980, 900]))
    np.testing.assert_allclose(distance, 0)
    assert ZoneIndex((zone,)).zone_at(-660, 970) is None


def test_nested_zone_wins_independently_of_source_order_and_weights_sum_to_one():
    small = replace(SPAWN, name="small", size_x=100, size_z=100)
    xs = np.array([-2000, -100, 0, 100, 2000])
    first = ZoneIndex((SPAWN, small)).query(xs, 0)
    second = ZoneIndex((small, SPAWN)).query(xs, 0)
    assert ZoneIndex((SPAWN, small)).zone_at(0, 0) == small
    assert first.weight_for("small").tolist() == [0, 0, 1, 0, 0]
    np.testing.assert_array_equal(first.background, second.background)
    total = first.background.copy()
    for part in first.contributions:
        np.testing.assert_array_equal(part.weight, second.weight_for(part.zone.name))
        total += part.weight
    np.testing.assert_array_equal(total, 1)


def test_all_authored_zone_shapes_can_be_queried():
    index = ZoneIndex(ALL_ZONES)
    for zone in ALL_ZONES:
        assert index.zone_at(zone.center_x, zone.center_z) is not None


def test_invalid_layout_and_missing_profiles_fail_explicitly():
    for change in ({"shape": "typo"}, {"boundary_mode": "typo"}, {"size_x": 0}, {"boundary_width": -1}):
        with pytest.raises(ValueError):
            ZoneIndex((replace(SPAWN, **change),))
    with pytest.raises(ValueError, match="duplicate"):
        ZoneIndex((SPAWN, SPAWN))
    with pytest.raises(ValueError, match="profile"):
        recipe_for_zone(replace(SPAWN, terrain_profile="typo"))


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_peaks_are_rougher_than_spawn_and_plateau_is_high_and_flat(seed):
    fields = []
    for zone in (SPAWN, QINGYUN_PEAKS, NORTH_WASTES):
        fields.append(generate_heightfield(
            recipe_for_zone(zone), width=64, height=64, seed=seed, cell_size=8,
            origin_x=zone.center_x - 256, origin_z=zone.center_z - 256,
        ).height)
    plain, peaks, plateau = fields
    assert peaks.std() > plain.std() * 8
    assert peaks.std() > plateau.std() * 5
    assert plateau.mean() > plain.mean() + 40
    assert np.ptp(plateau) < 15
