"""末端 droplet hydraulic erosion 的行为契约。"""

from dataclasses import replace
import json

import numpy as np
import pytest

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.engine import (
    HydraulicErosionSettings,
    NoiseLayer,
    TerrainRecipe,
    apply_hydraulic_erosion,
    generate_heightfield,
)


def _slope(size: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, z = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="xy",
    )
    terrain = 140.0 - z * 1.1 + np.sin(x * 0.35) * 3.0
    return terrain, x, z


def _settings(**overrides: object) -> HydraulicErosionSettings:
    base = HydraulicErosionSettings(
        enabled=True,
        droplet_count=140,
        max_steps=28,
        inertia=0.28,
        erosion_rate=0.18,
        deposition_rate=0.14,
        max_erosion_per_step=0.8,
    )
    return replace(base, **overrides)


def test_hydraulic_erosion_is_seeded_and_produces_refinement_fields() -> None:
    terrain, x, z = _slope()

    first = apply_hydraulic_erosion(terrain, x, z, _settings(), seed=31)
    repeated = apply_hydraulic_erosion(terrain, x, z, _settings(), seed=31)
    changed = apply_hydraulic_erosion(terrain, x, z, _settings(), seed=32)

    assert np.array_equal(first.height, repeated.height)
    assert np.array_equal(first.erosion, repeated.erosion)
    assert not np.array_equal(first.height, changed.height)
    assert np.any(first.erosion > 0.0), "下坡滴水必须产生可查询的小尺度下切"
    assert np.any(first.deposition > 0.0), "携沙量超过容量后必须形成冲积沉积"
    assert np.all(first.erosion >= 0.0) and np.all(first.deposition >= 0.0)
    assert np.isfinite(first.height).all()


def test_disabled_hydraulic_erosion_is_exact_noop() -> None:
    terrain, x, z = _slope()

    result = apply_hydraulic_erosion(
        terrain,
        x,
        z,
        _settings(enabled=False),
        seed=31,
    )

    assert np.array_equal(result.height, terrain)
    assert not np.any(result.erosion)
    assert not np.any(result.deposition)
    assert not np.shares_memory(result.height, terrain)


def test_pipeline_exposes_hydraulic_fields_and_preserves_macro_relief() -> None:
    settings = _settings(droplet_count=180)
    refined_recipe = TerrainRecipe(
        name="hydraulic_pipeline",
        base_height=150.0,
        base_noise=(NoiseLayer(kind="fbm", scale=18.0, amplitude=14.0, octaves=3),),
        hydraulic_erosion=settings,
    )
    plain_recipe = replace(refined_recipe, hydraulic_erosion=None)

    refined = generate_heightfield(refined_recipe, width=48, height=48, seed=19)
    plain = generate_heightfield(plain_recipe, width=48, height=48, seed=19)

    assert refined.hydraulic_erosion.dtype == np.float32
    assert refined.hydraulic_deposition.dtype == np.float32
    assert np.any(refined.hydraulic_erosion > 0.0)
    assert np.any(refined.hydraulic_deposition > 0.0)
    assert float(np.max(np.abs(refined.height - plain.height))) <= (
        settings.max_steps * settings.max_erosion_per_step
    )
    assert not np.any(plain.hydraulic_erosion)
    assert not np.any(plain.hydraulic_deposition)


def test_raster_exports_hydraulic_layers(tmp_path) -> None:
    recipe = TerrainRecipe(
        name="hydraulic_raster",
        base_height=120.0,
        base_noise=(NoiseLayer(kind="fbm", scale=18.0, amplitude=14.0, octaves=3),),
        hydraulic_erosion=_settings(droplet_count=80),
    )
    field = generate_heightfield(recipe, width=16, height=16, seed=23)
    manifest_path = write_bong_raster(to_bong_tile(field, sea_level=recipe.sea_level), tmp_path)
    tile_dir = tmp_path / "tile_0_0"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert (tile_dir / "hydraulic_erosion.bin").stat().st_size == 16 * 16 * 4
    assert (tile_dir / "hydraulic_deposition.bin").stat().st_size == 16 * 16 * 4
    assert "hydraulic_erosion" in manifest["tiles"][0]["layers"]
    assert manifest["hydraulic_erosion_encoding"]["dtype"] == "f32"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"droplet_count": -1}, "counts"),
        ({"max_steps": 0}, "steps"),
        ({"inertia": 1.1}, "inertia"),
        ({"evaporation": 0.0}, "evaporation"),
        ({"erosion_rate": -0.1}, "rates"),
        ({"deposition_rate": 1.1}, "rates"),
        ({"minimum_slope": -0.1}, "slope"),
    ],
)
def test_hydraulic_settings_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings(**overrides)


def test_hydraulic_erosion_rejects_invalid_grids() -> None:
    terrain = np.ones((2, 2), dtype=np.float64)
    x, z = np.meshgrid(np.arange(2), np.arange(2), indexing="xy")

    with pytest.raises(ValueError, match="three rows"):
        apply_hydraulic_erosion(terrain, x, z, _settings(), seed=1)
    with pytest.raises(ValueError, match="share"):
        apply_hydraulic_erosion(
            np.ones((4, 4)),
            np.ones((4, 3)),
            np.ones((4, 4)),
            _settings(),
            seed=1,
        )
