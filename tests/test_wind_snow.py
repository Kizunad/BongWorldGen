"""风向积雪场和覆盖层的行为契约。"""

from dataclasses import replace
import json

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.engine import (
    GlacialSystem,
    NoiseLayer,
    Point,
    TerrainRecipe,
    generate_heightfield,
    sample_wind_snow_field,
)
from bong_worldgen.engine.glaciers import glacial_surface_layers


def _slope(size: int = 65) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-32.0, 32.0, size, dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    return 180.0 + x * 0.9, x, z


def _system(**overrides: object) -> GlacialSystem:
    base = GlacialSystem(
        name="wind_snow_contract",
        seed_center=Point(0.0, 0.0),
        seed_extent=400.0,
        cirque_count=0,
        valley_count=0,
        prevailing_wind_angle_degrees=0.0,
        windward_accumulation_strength=0.55,
        leeward_drift_strength=0.24,
        wind_snow_max_extra_layers=1,
    )
    return replace(base, **overrides)


def test_wind_snow_uses_surface_normal_dot_wind_direction() -> None:
    terrain, x, z = _slope()
    windward = sample_wind_snow_field(
        terrain,
        x,
        z,
        (_system(prevailing_wind_angle_degrees=0.0),),
    )
    leeward = sample_wind_snow_field(
        terrain,
        x,
        z,
        (_system(prevailing_wind_angle_degrees=180.0),),
    )

    core = np.s_[8:-8, 8:-8]
    assert float(np.mean(windward.normal_alignment[core])) < -0.3
    assert float(np.mean(leeward.normal_alignment[core])) > 0.3
    assert float(np.mean(windward.accumulation[core])) > 0.0
    assert np.allclose(windward.erosion, 0.0)
    assert float(np.mean(leeward.erosion[core])) > 0.0
    assert np.allclose(leeward.accumulation, 0.0)
    assert float(np.mean(windward.response[core])) > float(
        np.mean(leeward.response[core])
    )


def test_wind_snow_is_zero_on_flat_ground_or_when_disabled() -> None:
    terrain, x, z = _slope()
    flat = sample_wind_snow_field(
        np.full_like(terrain, 180.0),
        x,
        z,
        (_system(),),
    )
    disabled = sample_wind_snow_field(
        terrain,
        x,
        z,
        (_system(wind_snow_enabled=False),),
    )
    for field in (flat, disabled):
        assert not np.any(field.accumulation)
        assert not np.any(field.erosion)
        assert not np.any(field.response)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("wind_snow_accumulation_threshold", -0.01),
        ("wind_snow_accumulation_threshold", 1.0),
        ("wind_snow_erosion_threshold", -0.01),
        ("wind_snow_erosion_threshold", 1.0),
        ("wind_snow_max_extra_layers", -1),
        ("wind_snow_max_extra_layers", 256),
        ("wind_snow_erosion_retention_scale", -0.01),
        ("wind_snow_erosion_retention_scale", 1.01),
    ],
)
def test_wind_snow_configuration_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="glacial system parameters are out of range"):
        GlacialSystem(name="invalid_wind_snow", **{field_name: value})


def test_wind_snow_field_is_stable_when_a_world_is_split_into_tiles() -> None:
    terrain, x, z = _slope()
    system = _system()
    whole = sample_wind_snow_field(terrain, x, z, (system,))
    left = sample_wind_snow_field(terrain[:, :33], x[:, :33], z[:, :33], (system,))
    right = sample_wind_snow_field(terrain[:, 32:], x[:, 32:], z[:, 32:], (system,))

    # 共享边界的有限差分需要 halo；比较每个 tile 的非边界列，验证内部
    # 采样只依赖世界坐标而不是 tile 原点。
    assert np.array_equal(whole.response[:, :32], left.response[:, :32])
    assert np.array_equal(whole.response[:, 33:], right.response[:, 1:])
    assert np.array_equal(whole.normal_alignment[:, :32], left.normal_alignment[:, :32])
    assert np.array_equal(whole.normal_alignment[:, 33:], right.normal_alignment[:, 1:])


def test_wind_snow_changes_visible_cover_without_changing_eligibility() -> None:
    terrain, x, z = _slope()
    system_windward = _system(prevailing_wind_angle_degrees=0.0)
    system_leeward = _system(prevailing_wind_angle_degrees=180.0)
    material = np.ones(terrain.shape, dtype=np.uint8)
    cold = np.ones(terrain.shape, dtype=np.float32)
    windward_field = sample_wind_snow_field(terrain, x, z, (system_windward,))
    leeward_field = sample_wind_snow_field(terrain, x, z, (system_leeward,))
    windward = glacial_surface_layers(
        terrain,
        x,
        z,
        (system_windward,),
        seed=41,
        sea_level=61.0,
        surface_material_id=material,
        climate_cold_weight=cold,
        wind_snow_field=windward_field,
    )
    leeward = glacial_surface_layers(
        terrain,
        x,
        z,
        (system_leeward,),
        seed=41,
        sea_level=61.0,
        surface_material_id=material,
        climate_cold_weight=cold,
        wind_snow_field=leeward_field,
    )

    assert np.any(windward)
    assert np.any(leeward)
    assert int(np.sum(windward)) > int(np.sum(leeward)), (
        "迎风积雪应增加可见覆盖层，背风风蚀应削薄覆盖层"
    )


def test_wind_snow_layers_are_exported_for_server_queries(tmp_path) -> None:
    recipe = TerrainRecipe(
        name="wind_snow_raster",
        base_height=120.0,
        base_noise=(NoiseLayer(kind="fbm", scale=18.0, amplitude=14.0, octaves=3),),
        glaciers=(_system(),),
    )
    field = generate_heightfield(recipe, width=16, height=16, seed=23)
    tile = to_bong_tile(field, sea_level=recipe.sea_level)
    manifest_path = write_bong_raster(tile, tmp_path)
    tile_dir = tmp_path / "tile_0_0"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert tile.wind_snow_alignment.shape == (16, 16)
    assert tile.wind_snow_response.shape == (16, 16)
    assert (tile_dir / "wind_snow_alignment.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "wind_snow_response.bin").stat().st_size == 16 * 16 * 4
    assert manifest["wind_snow_encoding"]["response"] == (
        "positive accumulation, negative wind erosion"
    )
    assert "wind_snow_response" in manifest["tiles"][0]["layers"]
