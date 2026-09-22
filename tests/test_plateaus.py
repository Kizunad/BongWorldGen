from dataclasses import replace

import numpy as np
import pytest

from bong_worldgen.engine import Plateau, Point, TerrainRecipe, generate_heightfield, sample_surface


def test_rectangle_keeps_flat_corners_that_are_outside_an_ellipse():
    shelf = Plateau(Point(-20, 30), 10, 6, 100, 1, shape="rectangle")
    recipe = TerrainRecipe(name="shelf", base_height=70, plateaus=(shelf,))
    # These corners fit the rectangle but lie outside the original ellipse.
    x, z = np.array([-28, -12, -28, -12]), np.array([26, 26, 34, 34])
    rectangle, _ = sample_surface(recipe, x, z, 7)
    ellipse, _ = sample_surface(replace(recipe, plateaus=(replace(shelf, shape="ellipse"),)), x, z, 7)
    np.testing.assert_array_equal(rectangle, 100)
    np.testing.assert_array_equal(ellipse, 70)


@pytest.mark.parametrize("shape", ("ellipse", "rectangle"))
def test_rotated_shelf_has_an_oblique_long_axis_and_exact_crop_invariance(shape):
    shelf = Plateau(Point(-20, -30), 30, 5, 100, 1, shape=shape, rotation=np.pi / 4)
    recipe = TerrainRecipe(name="oblique_shelf", base_height=70, plateaus=(shelf,))
    h, _ = sample_surface(recipe, np.array([-10, -30]), np.array([-20, -20]), 7)
    np.testing.assert_array_equal(h, [100, 70])
    options = dict(height=47, origin_z=-52.25, cell_size=1.25, seed=7)
    whole = generate_heightfield(recipe, width=41, origin_x=-43.75, **options)
    parts = [generate_heightfield(recipe, width=n, origin_x=x, **options)
             for n, x in ((19, -43.75), (22, -20.0))]
    for layer in ("height", "solid_spans"):
        np.testing.assert_array_equal(getattr(whole, layer),
                                      np.concatenate([getattr(p, layer) for p in parts], axis=1))


@pytest.mark.parametrize("shape", ("ellipse", "rectangle"))
def test_shelf_transition_is_continuous_and_monotone_at_both_ends(shape):
    shelf = Plateau(Point(0, 0), 10, 10, 100, 2, shape=shape)
    recipe = TerrainRecipe(name="smooth_edge", base_height=70, plateaus=(shelf,))
    h, _ = sample_surface(recipe, np.linspace(7.99, 10.01, 2021), 0, 7)
    assert h[0] == 100 and h[-1] == 70
    assert np.all(np.diff(h) <= 0)
    assert np.max(np.abs(np.diff(h))) < 0.03
    assert abs(h[11] - h[9]) < 1e-6
    assert abs(h[-10] - h[-12]) < 1e-6


@pytest.mark.parametrize("change", ({"shape": "hexagon"}, {"rotation": np.nan}, {"rotation": np.inf}))
def test_shelf_rejects_unknown_shapes_and_nonfinite_rotations(change):
    with pytest.raises(ValueError, match="plateau"):
        Plateau(Point(0, 0), 10, 10, 100, 2, **change)
