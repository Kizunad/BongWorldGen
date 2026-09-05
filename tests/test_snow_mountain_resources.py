from __future__ import annotations

import json

import numpy as np

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.adapters.anvil_nbt import BLOCK_NAMES, DEAD_BUSH
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.engine import (
    GlacialSystem,
    Heightfield,
    SnowMountainResourceSpec,
    UndergroundBlock,
)
from bong_worldgen.engine.glaciers.surface_resources import (
    generate_snow_mountain_resource_blocks,
)


def _fields(size: int = 12) -> dict[str, np.ndarray]:
    coordinates = np.arange(size, dtype=np.float64)
    z, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    shape = (size, size)
    return {
        "terrain": np.full(shape, 100.0, dtype=np.float64),
        "x": x,
        "z": z,
        "mountain_weight": np.ones(shape, dtype=np.float64),
        "mountain_material_id": np.ones(shape, dtype=np.uint8),
        "surface_material_id": np.ones(shape, dtype=np.uint8),
        "surface_cover_layers": np.zeros((4, *shape), dtype=np.uint8),
        "water_level": np.full(shape, -1.0, dtype=np.float64),
        "glacial_crevasse_id": np.zeros(shape, dtype=np.uint8),
        "climate_cold_weight": np.ones(shape, dtype=np.float64),
    }


def _system(
    *,
    probability: float = 1.0,
    seed_offset: int = 0,
) -> GlacialSystem:
    return GlacialSystem(
        name="resource-test-glacier",
        cirque_count=0,
        valley_count=0,
        snow_mountain_resources=(
            SnowMountainResourceSpec(
                probability=probability,
                seed_offset=seed_offset,
            ),
        ),
    )


def _generate(fields: dict[str, np.ndarray], *, seed: int, system: GlacialSystem) -> tuple:
    return generate_snow_mountain_resource_blocks(
        **fields,
        systems=(system,),
        seed=seed,
    )


def test_default_config_exposes_named_snow_mountain_resource() -> None:
    spec = DEFAULT_RECIPE.glaciers[0].snow_mountain_resources[0]

    assert spec.name == "雪山生成物"
    assert spec.material == "dead_bush"
    assert spec.resource_id == "snow_mountain:dead_bush"
    assert spec.rarity == "少"
    assert spec.probability == 0.018


def test_resource_generation_is_world_coordinate_deterministic() -> None:
    fields = _fields()
    system = _system(probability=0.5)

    whole = _generate(fields, seed=812731, system=system)
    left_fields = {
        name: value[:, :6] if value.ndim == 2 else value[:, :, :6]
        for name, value in fields.items()
    }
    right_fields = {
        name: value[:, 6:] if value.ndim == 2 else value[:, :, 6:]
        for name, value in fields.items()
    }
    left = _generate(left_fields, seed=812731, system=system)
    right = _generate(right_fields, seed=812731, system=system)

    assert whole == _generate(fields, seed=812731, system=system)
    assert whole == tuple(sorted(left + right, key=lambda block: (block.x, block.y, block.z)))


def test_different_seed_or_configured_offset_changes_probability_stream() -> None:
    fields = _fields()
    first = _generate(fields, seed=812731, system=_system(probability=0.5, seed_offset=0))
    second = _generate(fields, seed=812732, system=_system(probability=0.5, seed_offset=0))
    offset = _generate(fields, seed=812731, system=_system(probability=0.5, seed_offset=17))

    assert first != second
    assert first != offset


def test_only_cold_mountain_surface_is_eligible() -> None:
    fields = _fields(4)
    system = _system()

    fields["climate_cold_weight"][:] = 0.0
    assert _generate(fields, seed=1, system=system) == ()

    fields = _fields(4)
    fields["mountain_weight"][:] = 0.0
    assert _generate(fields, seed=1, system=system) == ()

    fields = _fields(4)
    fields["water_level"][1, 1] = 2.0
    fields["glacial_crevasse_id"][2, 2] = 1
    blocks = _generate(fields, seed=1, system=system)
    positions = {(block.x, block.z) for block in blocks}
    assert (1, 1) not in positions
    assert (2, 2) not in positions
    assert len(blocks) == 14


def test_probability_bounds_and_surface_cover_top_are_respected() -> None:
    fields = _fields(3)
    fields["terrain"][:] = 100.4
    fields["surface_cover_layers"][0] = 1
    fields["surface_cover_layers"][1] = 2
    fields["surface_cover_layers"][2] = 1
    fields["surface_cover_layers"][3] = 3

    assert _generate(fields, seed=1, system=_system(probability=0.0)) == ()
    blocks = _generate(fields, seed=1, system=_system(probability=1.0))
    assert len(blocks) == 9
    assert {block.y for block in blocks} == {108}
    assert {block.material for block in blocks} == {"minecraft:dead_bush"}
    assert {block.resource_id for block in blocks} == {"snow_mountain:dead_bush"}
    assert {block.source for block in blocks} == {"snow_mountain"}
    assert {block.rarity for block in blocks} == {"少"}


def test_dead_bush_has_real_minecraft_1_20_1_anvil_mapping() -> None:
    assert BLOCK_NAMES[DEAD_BUSH] == "minecraft:dead_bush"


def test_raster_manifest_preserves_snow_mountain_resource_identity(tmp_path) -> None:
    height = np.full((4, 4), 100.0, dtype=np.float32)
    field = Heightfield(
        height=height,
        moisture=np.zeros_like(height),
        water_level=np.full_like(height, -1.0),
        underground_blocks=(
            UndergroundBlock(
                x=1,
                y=101,
                z=2,
                material="minecraft:dead_bush",
                resource_id="snow_mountain:dead_bush",
                source="snow_mountain",
                rarity="少",
            ),
        ),
    )

    manifest_path = write_bong_raster(
        to_bong_tile(field, sea_level=61.0),
        tmp_path,
        tile_x=0,
        tile_z=0,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["underground_resource_palette"] == ["snow_mountain:dead_bush"]
    assert manifest["underground_resource_catalog"] == [
        {
            "resource_id": "snow_mountain:dead_bush",
            "material": "minecraft:dead_bush",
            "source": "snow_mountain",
            "rarity": "少",
        }
    ]
