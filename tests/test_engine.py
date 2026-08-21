from __future__ import annotations

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.preview_world import export_preview_world
from bong_worldgen.engine import (
    Basin,
    MountainRange,
    NoiseLayer,
    Point,
    River,
    TerrainRecipe,
    generate_heightfield,
)


def test_same_seed_is_byte_deterministic() -> None:
    first = generate_heightfield(DEFAULT_RECIPE, width=48, height=40, seed=812731)
    second = generate_heightfield(DEFAULT_RECIPE, width=48, height=40, seed=812731)
    assert np.array_equal(first.height, second.height)
    assert np.array_equal(first.moisture, second.moisture)
    assert np.array_equal(first.water_level, second.water_level)


def test_seed_changes_noise_without_changing_shape() -> None:
    first = generate_heightfield(DEFAULT_RECIPE, width=32, height=32, seed=1)
    second = generate_heightfield(DEFAULT_RECIPE, width=32, height=32, seed=2)
    assert first.height.shape == second.height.shape
    assert not np.array_equal(first.height, second.height)


def test_domain_warp_noise_is_finite_and_seeded() -> None:
    recipe = TerrainRecipe(
        name="warp",
        base_noise=(
            NoiseLayer(kind="warp", scale=64.0, warp_scale=180.0, warp_strength=40.0),
        ),
    )
    field = generate_heightfield(recipe, width=24, height=24, seed=5)
    assert np.isfinite(field.height).all()
    assert not np.array_equal(
        field.height,
        generate_heightfield(recipe, width=24, height=24, seed=6).height,
    )


def test_polyline_river_carves_and_widens() -> None:
    recipe = TerrainRecipe(
        name="river",
        base_height=80.0,
        rivers=(River((Point(0, 0), Point(30, 0)), width=2.0, depth=8.0, widening=4.0),),
    )
    field = generate_heightfield(recipe, width=40, height=8, seed=4, origin_x=0, origin_z=-3)
    center = field.height[:, 3]
    assert np.min(center) < 74.0
    assert np.count_nonzero(field.water_level >= 0) > 0


def test_mountain_and_basin_are_visible_in_field() -> None:
    recipe = TerrainRecipe(
        name="relief",
        base_height=70.0,
        basins=(Basin(Point(10, 10), 12, 12, 8),),
        mountains=(MountainRange((Point(-10, 0), Point(30, 0)), width=5, height=50),),
    )
    field = generate_heightfield(recipe, width=40, height=30, seed=3, origin_x=-10, origin_z=-10)
    assert float(np.max(field.height)) - float(np.min(field.height)) > 20.0


def test_mountain_roughness_contrast_is_bounded() -> None:
    with pytest.raises(ValueError, match="roughness contrast"):
        MountainRange((Point(0, 0), Point(10, 0)), width=5, height=20, roughness_contrast=1.1)


def test_bong_adapter_uses_expected_dtypes_and_shapes() -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=16, height=12, seed=7)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)
    assert tile.height.dtype == np.float32
    assert tile.surface_id.dtype == np.uint8
    assert tile.biome_id.dtype == np.uint8
    assert tile.feature_mask.dtype == np.float32
    assert tile.boundary_weight.dtype == np.float32
    assert tile.height.flags.c_contiguous


def test_bong_raster_writer_emits_v2_core_files(tmp_path) -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=16, height=16, seed=7)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)
    manifest_path = write_bong_raster(tile, tmp_path, tile_x=-1, tile_z=2)
    manifest = manifest_path.read_text(encoding="utf-8")
    tile_dir = tmp_path / "tile_-1_2"
    assert '"version": 2' in manifest
    assert (tile_dir / "spans_count.bin").stat().st_size == 16 * 16
    assert (tile_dir / "spans.bin").stat().st_size == 16 * 16 * 16


def test_preview_world_manifest_is_console_compatible(tmp_path) -> None:
    manifest_path = export_preview_world(
        tmp_path,
        recipe=TerrainRecipe(name="test_world"),
        seed=7,
        min_x=-16,
        max_x=15,
        min_z=-16,
        max_z=15,
        tile_size=16,
    )
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["spans_encoding"]["bytes_per_column"] == 16
    assert len(manifest["tiles"]) == 4
    assert all(tile["spans"] for tile in manifest["tiles"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 8},
        {"width": 8, "height": 0},
    ],
)
def test_invalid_dimensions_fail_fast(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        generate_heightfield(DEFAULT_RECIPE, seed=1, **kwargs)
