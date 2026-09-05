from __future__ import annotations

import numpy as np
import pytest

from bong_worldgen.engine import MountainRange, Point, ValleySystem
from bong_worldgen.engine.pipeline import generate_heightfield
from bong_worldgen.engine.world_config import TerrainRecipe
from bong_worldgen.engine.valleys import (
    apply_valley_erosion,
    apply_watershed_divide,
    plan_valley_erosion,
    route_flow_d8,
    stream_power_erosion,
)
from bong_worldgen.engine.valleys.divide import watershed_divide_strength


def _grid(size: int = 9) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float64)
    return np.meshgrid(axis, axis, indexing="xy")


def _system(**overrides: object) -> ValleySystem:
    values: dict[str, object] = {
        "name": "test_valleys",
        "domain_center": Point(4.0, 4.0),
        "domain_extent_x": 4.0,
        "domain_extent_z": 4.0,
        "planning_resolution": 1.0,
        "minimum_drainage_area": 2.0,
        "reference_drainage_area": 32.0,
        "maximum_depth": 12.0,
        "minimum_width": 1.0,
        "maximum_width": 3.0,
    }
    values.update(overrides)
    return ValleySystem(**values)


def test_priority_flood_raises_closed_pit_without_changing_outlet() -> None:
    terrain = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 3.0, 8.0, 10.0],
            [10.0, 8.0, 1.0, 8.0, 10.0],
            [10.0, 8.0, 3.0, 8.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    flow = route_flow_d8(
        terrain,
        np.ones_like(terrain),
        cell_size_x=1.0,
        cell_size_z=1.0,
        minimum_slope=0.01,
    )

    assert flow.conditioned_height[2, 2] == pytest.approx(10.0)
    assert flow.conditioned_height[0, 0] == pytest.approx(10.0)
    assert flow.accumulation.shape == terrain.shape
    assert np.isfinite(flow.accumulation).all()


def test_flat_priority_flood_routes_all_rainfall_to_boundary() -> None:
    terrain = np.full((7, 7), 100.0)
    flow = route_flow_d8(
        terrain,
        np.ones_like(terrain),
        cell_size_x=1.0,
        cell_size_z=1.0,
        minimum_slope=0.01,
    )
    receiver = flow.receiver.ravel()
    outlets = receiver == np.arange(receiver.size)

    assert np.all(flow.conditioned_height == 100.0)
    assert float(np.sum(flow.accumulation.ravel()[outlets])) == pytest.approx(49.0)
    assert np.all(flow.accumulation > 0.0)


def test_d8_accumulation_increases_downhill_and_has_no_interior_sink() -> None:
    x, z = _grid(13)
    terrain = 200.0 - z * 3.0 - x * 0.4
    flow = route_flow_d8(
        terrain,
        np.ones_like(terrain),
        cell_size_x=1.0,
        cell_size_z=1.0,
        minimum_slope=0.01,
    )

    assert float(flow.accumulation[11, 11]) > float(flow.accumulation[1, 1])
    interior = np.ones(terrain.shape, dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    rows, columns = np.indices(terrain.shape)
    receiver_rows, receiver_columns = np.divmod(flow.receiver, terrain.shape[1])
    moved = (receiver_rows != rows) | (receiver_columns != columns)
    assert np.all(moved[interior])
    assert np.all(flow.accumulation > 0.0)
    outlets = ~moved
    assert float(np.sum(flow.accumulation[outlets])) == pytest.approx(float(terrain.size))


def test_stream_power_creates_bounded_widened_valley() -> None:
    x, z = _grid(17)
    terrain = 160.0 - z * 2.0 - x * 0.1
    flow = route_flow_d8(
        terrain,
        np.ones_like(terrain),
        cell_size_x=1.0,
        cell_size_z=1.0,
        minimum_slope=0.01,
    )
    system = _system(
        minimum_drainage_area=2.0,
        reference_drainage_area=80.0,
        maximum_depth=18.0,
        maximum_width=4.0,
    )
    depth, stream_power = stream_power_erosion(
        terrain,
        flow,
        system,
        sea_level=0.0,
        cell_size_x=1.0,
        cell_size_z=1.0,
    )

    assert float(np.max(depth)) > 0.0
    assert float(np.max(depth)) <= system.maximum_depth + 1.0e-9
    assert np.count_nonzero(depth > 0.0) > np.count_nonzero(stream_power > 0.0)
    assert np.all(stream_power[stream_power > 0.0] > 0.0)


def test_valley_plan_is_seeded_and_tile_stable() -> None:
    x, z = _grid(33)
    terrain = 180.0 - z * 2.0 + np.sin(x * 0.4) * 2.0
    system = _system(
        domain_center=Point(16.0, 16.0),
        domain_extent_x=16.0,
        domain_extent_z=16.0,
        maximum_width=5.0,
    )
    plan = plan_valley_erosion(system, terrain, x, z, seed=41, sea_level=0.0)
    repeated = plan_valley_erosion(system, terrain, x, z, seed=41, sea_level=0.0)
    changed = plan_valley_erosion(system, terrain, x, z, seed=42, sea_level=0.0)
    whole = apply_valley_erosion(terrain, x, z, (plan,))
    left = apply_valley_erosion(terrain[:, :17], x[:, :17], z[:, :17], (plan,))
    right = apply_valley_erosion(terrain[:, 17:], x[:, 17:], z[:, 17:], (plan,))

    assert np.array_equal(plan.erosion_depth, repeated.erosion_depth)
    assert not np.array_equal(plan.erosion_depth, changed.erosion_depth)
    assert np.array_equal(whole, np.concatenate((left, right), axis=1))
    assert plan.flow_accumulation.shape == terrain.shape


def test_uplift_strength_changes_stream_power_without_changing_rainfall_topology() -> None:
    x, z = _grid(17)
    terrain = 160.0 - z * 2.0 - x * 0.1
    uplift = np.zeros_like(terrain)
    uplift[:, 5:12] = 128.0
    base = _system(uplift_strength=0.0, coupling_iterations=1)
    coupled = _system(uplift_strength=1.0, uplift_reference=128.0, coupling_iterations=1)

    base_plan = plan_valley_erosion(base, terrain, x, z, seed=41, sea_level=0.0, uplift=uplift)
    coupled_plan = plan_valley_erosion(
        coupled,
        terrain,
        x,
        z,
        seed=41,
        sea_level=0.0,
        uplift=uplift,
    )

    assert np.array_equal(base_plan.flow_accumulation, coupled_plan.flow_accumulation)
    assert np.all(coupled_plan.erosion_depth >= base_plan.erosion_depth)
    assert np.any(coupled_plan.erosion_depth > base_plan.erosion_depth)
    assert np.array_equal(coupled_plan.uplift, uplift)


def test_valley_erosion_does_not_change_domain_outside() -> None:
    x, z = _grid(9)
    terrain = np.full(x.shape, 100.0)
    system = _system(
        domain_center=Point(4.0, 4.0),
        domain_extent_x=2.0,
        domain_extent_z=2.0,
        maximum_width=2.0,
    )
    plan_x, plan_z = np.meshgrid(
        np.linspace(2.0, 6.0, 5),
        np.linspace(2.0, 6.0, 5),
        indexing="xy",
    )
    plan = plan_valley_erosion(
        system,
        100.0 - plan_z * 2.0,
        plan_x,
        plan_z,
        seed=7,
        sea_level=0.0,
    )
    output = apply_valley_erosion(terrain, x, z, (plan,))
    outside = (x < 2.0) | (x > 6.0) | (z < 2.0) | (z > 6.0)

    assert np.array_equal(output[outside], terrain[outside])
    assert np.any(output[~outside] < terrain[~outside])


def test_heightfield_exposes_valley_physical_fields() -> None:
    system = _system(
        domain_center=Point(8.0, 8.0),
        domain_extent_x=8.0,
        domain_extent_z=8.0,
        minimum_drainage_area=2.0,
        reference_drainage_area=32.0,
        maximum_width=3.0,
    )
    with_valleys = generate_heightfield(
        TerrainRecipe(name="valley_fields", base_height=100.0, valleys=(system,)),
        width=17,
        height=17,
        seed=9,
    )
    without_valleys = generate_heightfield(
        TerrainRecipe(name="no_valley_fields", base_height=100.0),
        width=17,
        height=17,
        seed=9,
    )

    assert float(np.max(with_valleys.valley_depth)) > 0.0
    assert float(np.max(with_valleys.valley_flow_accumulation)) > 0.0
    assert float(np.max(with_valleys.valley_stream_power)) > 0.0
    assert float(np.max(with_valleys.watershed_divide)) > 0.0
    assert np.any(with_valleys.height < without_valleys.height)
    assert not np.any(without_valleys.valley_depth)
    assert not np.any(without_valleys.valley_flow_accumulation)
    assert not np.any(without_valleys.valley_stream_power)
    assert not np.any(without_valleys.watershed_divide)


def test_watershed_divide_is_seeded_bounded_and_preserves_stream_power_valleys() -> None:
    x, z = _grid(25)
    terrain = 180.0 - z * 2.0 + np.sin(x * 0.4) * 2.0
    system = _system(
        domain_center=Point(12.0, 12.0),
        domain_extent_x=12.0,
        domain_extent_z=12.0,
        maximum_width=5.0,
        watershed_divide_strength=18.0,
        watershed_divide_width=3.0,
    )
    first = plan_valley_erosion(system, terrain, x, z, seed=41, sea_level=0.0)
    repeated = plan_valley_erosion(system, terrain, x, z, seed=41, sea_level=0.0)
    changed = plan_valley_erosion(system, terrain, x, z, seed=42, sea_level=0.0)

    assert np.array_equal(first.watershed_divide, repeated.watershed_divide)
    assert not np.array_equal(first.watershed_divide, changed.watershed_divide)
    assert float(first.watershed_divide.min()) >= 0.0
    assert float(first.watershed_divide.max()) <= system.watershed_divide_strength
    output = apply_watershed_divide(terrain - first.erosion_depth, x, z, (first,))
    assert np.any(output < terrain)
    assert np.allclose(
        output - (terrain - first.erosion_depth),
        first.watershed_divide,
    )


def test_watershed_divide_rejects_invalid_derived_fields() -> None:
    x, z = _grid(5)
    terrain = 100.0 - z
    flow = route_flow_d8(
        terrain,
        np.ones_like(terrain),
        cell_size_x=1.0,
        cell_size_z=1.0,
        minimum_slope=0.01,
    )
    system = _system(domain_center=Point(2.0, 2.0), domain_extent_x=2.0, domain_extent_z=2.0)
    with pytest.raises(ValueError, match="uplift"):
        watershed_divide_strength(
            flow,
            system,
            cell_size_x=1.0,
            cell_size_z=1.0,
            uplift=np.zeros((4, 4)),
        )
    with pytest.raises(ValueError, match="stream power"):
        watershed_divide_strength(
            flow,
            system,
            cell_size_x=1.0,
            cell_size_z=1.0,
            stream_power=-np.ones_like(terrain),
        )


def test_heightfield_exposes_uplift_field_for_valley_coupling() -> None:
    recipe = TerrainRecipe(
        name="uplift_fields",
        base_height=70.0,
        mountains=(
            MountainRange(
                (Point(0.0, 8.0), Point(16.0, 8.0)),
                width=5.0,
                height=32.0,
            ),
        ),
        valleys=(_system(domain_center=Point(8.0, 8.0), domain_extent_x=8.0, domain_extent_z=8.0),),
    )
    field = generate_heightfield(recipe, width=17, height=17, seed=9)

    assert field.uplift.shape == field.height.shape
    assert np.all(field.uplift >= 0.0)
    assert float(np.max(field.uplift)) > 0.0
    assert field.watershed_divide.shape == field.height.shape
    assert np.all(field.watershed_divide >= 0.0)
    assert float(np.max(field.watershed_divide)) > 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"domain_extent_x": 0.0}, "extents"),
        ({"planning_resolution": 0.0}, "planning_resolution"),
        ({"rainfall_variation": 1.0}, "rainfall_variation"),
        ({"minimum_drainage_area": 0.0}, "minimum_drainage_area"),
        ({"reference_drainage_area": 1.0}, "reference_drainage_area"),
        ({"minimum_slope": 0.0}, "slope references"),
        ({"area_exponent": 0.0}, "exponents"),
        ({"erosion_strength": -1.0}, "non-negative"),
        ({"maximum_width": 0.5}, "widths"),
        ({"cross_section_power": 0.0}, "exponents"),
        ({"uplift_strength": -1.0}, "uplift coupling"),
        ({"uplift_reference": 0.0}, "uplift coupling"),
        ({"watershed_divide_strength": -1.0}, "watershed divide"),
        ({"watershed_divide_width": -1.0}, "watershed divide"),
        ({"watershed_divide_valley_fill": 1.1}, "valley fill"),
        ({"coupling_iterations": 0}, "coupling_iterations"),
    ),
)
def test_valley_system_rejects_invalid_parameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _system(**overrides)


def test_valley_system_rejects_non_positive_rainfall() -> None:
    x, z = _grid(5)
    terrain = np.ones(x.shape)
    with pytest.raises(ValueError, match="rainfall"):
        route_flow_d8(
            terrain,
            np.zeros_like(terrain),
            cell_size_x=1.0,
            cell_size_z=1.0,
            minimum_slope=0.01,
        )


def test_valley_uplift_contract_rejects_invalid_fields() -> None:
    x, z = _grid(5)
    terrain = 100.0 - z
    system = _system(domain_center=Point(2.0, 2.0), domain_extent_x=2.0, domain_extent_z=2.0)
    with pytest.raises(ValueError, match="uplift"):
        plan_valley_erosion(
            system,
            terrain,
            x,
            z,
            seed=4,
            sea_level=0.0,
            uplift=np.zeros((4, 4)),
        )
    with pytest.raises(ValueError, match="uplift"):
        plan_valley_erosion(
            system,
            terrain,
            x,
            z,
            seed=4,
            sea_level=0.0,
            uplift=-np.ones_like(terrain),
        )
