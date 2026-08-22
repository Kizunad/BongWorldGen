from __future__ import annotations

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.data.wilderness import GRASSLAND, LAKE, MOUNTAINS, RIVER, classify_wilderness
from bong_worldgen.preview_world import export_preview_world
from bong_worldgen.engine import (
    Basin,
    CaveNetwork,
    MountainRange,
    NoiseLayer,
    Point,
    River,
    TerrainRecipe,
    generate_heightfield,
)
from bong_worldgen.engine.caves import generate_cave_topology, sample_worm_field
from bong_worldgen.engine.noise import sample_noise_3d


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


def test_3d_noise_is_seeded_and_varies_along_vertical_axis() -> None:
    x = np.full((4, 4), 12.0)
    z = np.full((4, 4), -8.0)
    layer = NoiseLayer(kind="fbm", scale=24.0, octaves=2)
    low = sample_noise_3d(x, np.full((4, 4), 50.0), z, layer, seed=7)
    high = sample_noise_3d(x, np.full((4, 4), 62.0), z, layer, seed=7)
    repeat = sample_noise_3d(x, np.full((4, 4), 50.0), z, layer, seed=7)

    assert np.array_equal(low, repeat)
    assert not np.array_equal(low, high)
    assert np.isfinite(low).all() and np.isfinite(high).all()


def test_godot_style_worm_field_is_seeded_bounded_and_varied() -> None:
    x, z = np.meshgrid(
        np.arange(48, dtype=np.float64),
        np.arange(32, dtype=np.float64),
        indexing="xy",
    )
    network = CaveNetwork(
        name="worm",
        paths=((Point(0, 16), Point(47, 16)),),
        worm_threshold=0.55,
        dead_end_strength=1.15,
        vertical_warp=3.0,
    )

    first = sample_worm_field(x, z, network, seed=19)
    repeat = sample_worm_field(x, z, network, seed=19)
    other = sample_worm_field(x, z, network, seed=20)

    assert np.array_equal(first.corridor, repeat.corridor)
    assert not np.array_equal(first.corridor, other.corridor)
    assert np.all((first.corridor >= 0.0) & (first.corridor <= 1.0))
    assert np.all((first.dead_end >= 0.0) & (first.dead_end <= 1.0))
    assert np.all(first.radius_scale > 0.0)
    assert float(np.ptp(first.radius_scale)) > 0.0
    assert float(np.ptp(first.vertical_offset)) > 0.0
    assert float(np.ptp(first.dead_end)) > 0.0


def test_cave_domain_warp_changes_the_sdf_without_breaking_determinism() -> None:
    common = dict(
        name="warped",
        paths=((Point(0, 16), Point(63, 16)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=0,
        width=3.0,
        height=6,
        depth=10.0,
    )
    straight_recipe = TerrainRecipe(
        name="straight_cave",
        caves=(CaveNetwork(**common, domain_warp_strength=0.0),),
    )
    warped_recipe = TerrainRecipe(
        name="warped_cave",
        caves=(CaveNetwork(**common, domain_warp_strength=8.0),),
    )

    straight = generate_heightfield(straight_recipe, width=64, height=32, seed=10)
    warped = generate_heightfield(warped_recipe, width=64, height=32, seed=10)
    repeat = generate_heightfield(warped_recipe, width=64, height=32, seed=10)

    assert not np.array_equal(straight.cave_id, warped.cave_id)
    assert np.array_equal(warped.cave_id, repeat.cave_id)


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


def test_river_bed_materials_are_seeded_and_exposed() -> None:
    recipe = TerrainRecipe(
        name="riverbed_materials",
        base_height=80.0,
        rivers=(
            River(
                (Point(0, 4), Point(63, 4)),
                width=3.0,
                depth=8.0,
                bed_materials=("minecraft:mud", "sand", "clay"),
            ),
        ),
    )
    first = generate_heightfield(recipe, width=64, height=12, seed=23, origin_z=-2)
    second = generate_heightfield(recipe, width=64, height=12, seed=23, origin_z=-2)

    assert first.riverbed_palette == ("mud", "sand", "clay")
    assert np.array_equal(first.riverbed_id, second.riverbed_id)
    wet_bed = first.riverbed_id >= 0
    assert np.count_nonzero(wet_bed) > 0
    assert np.all(np.isin(first.riverbed_id[wet_bed], np.arange(3)))


def test_standalone_biome_ids_match_declared_two_entry_palette() -> None:
    field = generate_heightfield(DEFAULT_RECIPE, width=48, height=48, seed=812731)
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)

    assert set(np.unique(tile.biome_id)).issubset({0, 1})


def test_riverbed_buffer_covers_one_extra_grid_cell() -> None:
    recipe = TerrainRecipe(
        name="riverbed_buffer",
        base_height=80.0,
        rivers=(
            River(
                (Point(0, 4), Point(31, 4)),
                width=2.0,
                depth=4.0,
                bed_materials=("sand",),
            ),
        ),
    )
    field = generate_heightfield(recipe, width=32, height=12, seed=4, origin_z=0)

    # The channel radius is two cells, but the riverbed palette intentionally
    # extends one cell farther to avoid an uncovered border after voxelization.
    assert field.riverbed_id[0, 4] == -1
    assert field.riverbed_id[1, 4] >= 0
    assert field.riverbed_id[2, 4] >= 0
    assert field.riverbed_id[3, 4] >= 0


def test_shallow_cave_network_emits_a_connected_air_gap() -> None:
    recipe = TerrainRecipe(
        name="shallow_caves",
        base_height=80.0,
        caves=(
            CaveNetwork(
                name="main",
                paths=((Point(0, 6), Point(31, 6)),),
                width=2.0,
                height=4,
                depth=9.0,
            ),
        ),
    )
    field = generate_heightfield(recipe, width=32, height=16, seed=4, origin_z=0)

    column = field.solid_spans[6, 16]
    spans = [tuple(int(value) for value in span) for span in column if span[0] != 32767]
    assert len(spans) == 2
    assert spans[0][1] == 80
    assert spans[1][0] == -64
    assert spans[1][1] < spans[0][0] - 1
    assert field.cave_palette == ("main",)
    assert np.all(field.cave_id[field.cave_id > 0] == 1)


def test_cave_topology_is_seeded_and_not_fixed_to_anchor_paths() -> None:
    network = CaveNetwork(
        name="topology",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=6,
        branch_segments=4,
        chamber_count=3,
        entrance_count=2,
    )

    first = generate_cave_topology(network, seed=41)
    repeat = generate_cave_topology(network, seed=41)
    other = generate_cave_topology(network, seed=42)

    assert first == repeat
    assert len(first.paths) == 1 + network.branch_count
    assert len(first.chambers) == network.chamber_count
    assert len(first.entrances) == network.entrance_count
    assert first.paths != other.paths


def test_cave_entrance_reaches_surface_without_opening_everywhere() -> None:
    network = CaveNetwork(
        name="entrance",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=0,
        chamber_count=0,
        entrance_count=1,
        width=2.0,
        height=4,
        depth=9.0,
    )
    recipe = TerrainRecipe(name="entrance", base_height=80.0, caves=(network,))
    topology = generate_cave_topology(network, seed=4)
    entrance = topology.entrances[0]
    entrance_x = round(entrance.x)
    entrance_z = round(entrance.z)

    field = generate_heightfield(recipe, width=32, height=20, seed=4)
    entrance_spans = field.solid_spans[entrance_z, entrance_x]
    assert field.cave_id[entrance_z, entrance_x] == 1
    assert int(entrance_spans[0, 1]) < int(round(field.height[entrance_z, entrance_x]))
    assert np.count_nonzero(field.cave_id) < field.cave_id.size


def test_cave_generation_is_identical_when_split_into_tiles() -> None:
    network = CaveNetwork(
        name="tile_stable",
        paths=((Point(0, 8), Point(31, 8)),),
        branch_count=4,
        branch_segments=3,
        chamber_count=2,
        entrance_count=1,
    )
    recipe = TerrainRecipe(name="tile_stable", base_height=80.0, caves=(network,))
    full = generate_heightfield(recipe, width=32, height=20, seed=9, origin_x=0, origin_z=0)
    left = generate_heightfield(recipe, width=16, height=20, seed=9, origin_x=0, origin_z=0)
    right = generate_heightfield(recipe, width=16, height=20, seed=9, origin_x=16, origin_z=0)

    assert np.array_equal(full.height, np.concatenate((left.height, right.height), axis=1))
    assert np.array_equal(full.water_level, np.concatenate((left.water_level, right.water_level), axis=1))
    assert np.array_equal(full.cave_id, np.concatenate((left.cave_id, right.cave_id), axis=1))
    assert np.array_equal(
        full.solid_spans,
        np.concatenate((left.solid_spans, right.solid_spans), axis=1),
    )


def test_river_bed_materials_reject_empty_or_duplicate_names() -> None:
    with pytest.raises(ValueError, match="at least one"):
        River((Point(0, 0), Point(1, 0)), width=1.0, depth=1.0, bed_materials=())
    with pytest.raises(ValueError, match="duplicates"):
        River(
            (Point(0, 0), Point(1, 0)),
            width=1.0,
            depth=1.0,
            bed_materials=("sand", "minecraft:sand"),
        )


def test_river_waterline_stays_below_the_lowest_bank() -> None:
    recipe = TerrainRecipe(
        name="contained_river",
        base_height=80.0,
        base_noise=(NoiseLayer(kind="fbm", scale=12.0, amplitude=8.0, octaves=2),),
        rivers=(River((Point(0, 4), Point(47, 4)), width=2.0, depth=8.0),),
    )
    field = generate_heightfield(recipe, width=48, height=12, seed=9, origin_x=0, origin_z=-2)
    surface = np.rint(field.height).astype(np.int16)
    water_top = np.ceil(field.water_level).astype(np.int16)
    wet = field.water_level >= 0.0
    padded_surface = np.pad(surface, 1, mode="edge")
    padded_wet = np.pad(wet, 1, mode="edge")
    for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour_surface = padded_surface[
            1 + dz : 1 + dz + surface.shape[0], 1 + dx : 1 + dx + surface.shape[1]
        ]
        neighbour_wet = padded_wet[
            1 + dz : 1 + dz + surface.shape[0], 1 + dx : 1 + dx + surface.shape[1]
        ]
        # For every dry bank adjacent to water, the water top cannot be higher
        # than that bank's top block.
        assert np.all(water_top[wet & ~neighbour_wet] <= neighbour_surface[wet & ~neighbour_wet])
    assert np.max((water_top - surface)[wet]) >= 2
    center_water = field.water_level[6]
    center_wet = center_water >= 0.0
    assert np.all(np.diff(center_water[center_wet]) <= 0.0)


def test_river_water_surface_is_flat_across_each_cross_section() -> None:
    recipe = TerrainRecipe(
        name="flat_cross_section",
        base_height=80.0,
        rivers=(River((Point(0, 0), Point(47, 0)), width=3.0, depth=8.0),),
    )
    field = generate_heightfield(recipe, width=48, height=11, seed=1, origin_z=-5)

    for x_index in (10, 20, 30):
        cross_section_water = field.water_level[:, x_index]
        wet = cross_section_water >= 0.0
        assert np.count_nonzero(wet) >= 3
        assert float(np.ptp(cross_section_water[wet])) == 0.0
        cross_section_height = field.height[:, x_index]
        assert float(cross_section_height[5]) < float(cross_section_height[3])
        assert float(cross_section_height[5]) < float(cross_section_height[7])


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
    assert tile.wilderness_id.dtype == np.uint8
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
    assert (tile_dir / "wilderness_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "riverbed_id.bin").stat().st_size == 16 * 16
    assert (tile_dir / "cave_id.bin").stat().st_size == 16 * 16
    assert '"wilderness_palette"' in manifest
    assert '"riverbed_palette"' in manifest
    assert '"cave_palette"' in manifest


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
    assert manifest["overview"]["cell_size"] == 32
    assert manifest["overview"]["surface_file"] == "overview_surface_id.bin"
    assert (tmp_path / "rasters" / "overview_surface_id.bin").stat().st_size == 1
    assert manifest["wilderness_palette"][1]["key"] == "mountains"
    assert "cave_id" in manifest["semantic_layers"]
    assert "cave_id" in manifest["tiles"][0]["layers"]
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


def test_wilderness_classifier_exposes_stable_categories() -> None:
    height = np.full((5, 5), 70.0, dtype=np.float32)
    water = np.full((5, 5), -1.0, dtype=np.float32)
    water[0, 0] = 61.0
    water[1, 1] = 66.0
    height[3, 3] = 130.0

    wilderness = classify_wilderness(height, water, sea_level=61.0)

    assert int(wilderness[2, 2]) == GRASSLAND
    assert int(wilderness[0, 0]) == LAKE
    assert int(wilderness[1, 1]) == RIVER
    assert int(wilderness[3, 3]) == MOUNTAINS


def test_wilderness_classifier_accepts_single_column_fields() -> None:
    wilderness = classify_wilderness(
        np.asarray([[70.0]], dtype=np.float32),
        np.asarray([[-1.0]], dtype=np.float32),
        sea_level=61.0,
    )
    assert wilderness.tolist() == [[GRASSLAND]]
