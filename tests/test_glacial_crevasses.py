from __future__ import annotations

import json

import numpy as np

from bong_worldgen.adapters import to_bong_tile, write_bong_raster
from bong_worldgen.engine import GlacialFlowPlan, GlacialSystem, Heightfield, Point
from bong_worldgen.engine.glaciers import (
    GLACIAL_CREVASSE_PALETTE,
    glacial_surface_layers,
    sample_glacial_crevasses,
    sample_glacier_valley_field,
)


def _plan(*, blue_probability: float = 0.18) -> GlacialFlowPlan:
    system = GlacialSystem(
        name="crevasse_test",
        seed_center=Point(64.0, 64.0),
        valley_source_width=14.0,
        valley_floor_width=5.0,
        valley_width_growth=0.8,
        valley_length=110.0,
        valley_segments=3,
        valley_count=1,
        cirque_count=0,
        flow_min_length=0.0,
        crevasse_spacing=28.0,
        crevasse_width=1.0,
        crevasse_probability=1.0,
        crevasse_jitter=0.0,
        crevasse_meander=0.30,
        crevasse_blue_ice_probability=blue_probability,
    )
    path = (Point(12.0, 64.0), Point(64.0, 64.0), Point(116.0, 64.0))
    return GlacialFlowPlan(system, 991, (), (path,))


def test_glacier_valley_field_is_path_gated_and_has_center_pressure() -> None:
    x, z = np.meshgrid(np.arange(128.0), np.arange(128.0), indexing="xy")
    field = sample_glacier_valley_field(x, z, (_plan(),))

    assert field.mask[64, 64]
    assert not field.mask[12, 12]
    assert field.pressure[64, 64] > field.pressure[68, 64]
    assert np.all((field.pressure >= 0.0) & (field.pressure <= 1.0))


def test_blue_ice_cover_is_sparse_inside_valley_and_absent_outside() -> None:
    x, z = np.meshgrid(np.arange(128.0), np.arange(128.0), indexing="xy")
    terrain = np.full(x.shape, 180.0)
    surface_material = np.full(x.shape, 3, dtype=np.uint8)
    plan = _plan(blue_probability=0.0)
    # 使用默认 veins 概率的独立覆盖调用，避免蓝冰成为整层。
    layers = glacial_surface_layers(
        terrain,
        x,
        z,
        (plan.system,),
        seed=41,
        sea_level=61.0,
        surface_material_id=surface_material,
        plans=(plan,),
    )
    valley = sample_glacier_valley_field(x, z, (plan,)).mask
    blue = layers[3] > 0
    assert np.any(blue)
    assert np.all(~blue | valley)
    assert np.count_nonzero(blue) < np.count_nonzero(valley) * 0.5


def test_crevasses_are_seeded_transverse_real_block_ids() -> None:
    x, z = np.meshgrid(np.arange(128.0), np.arange(128.0), indexing="xy")
    terrain = np.full(x.shape, 180.0)
    surface_material = np.full(x.shape, 3, dtype=np.uint8)
    ids = sample_glacial_crevasses(
        terrain,
        x,
        z,
        (_plan(blue_probability=1.0),),
        seed=73,
        sea_level=61.0,
        surface_material_id=surface_material,
    )

    assert GLACIAL_CREVASSE_PALETTE == (
        "minecraft:air",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
    )
    assert np.any(ids == 1), "裂隙中央应使用真实空气方块"
    assert np.any(ids == 2), "裂隙边缘应保留 packed ice"
    assert np.any(ids == 3), "高压裂隙边缘应允许 blue ice"
    assert not np.any(ids[0:20])
    assert np.array_equal(
        ids,
        sample_glacial_crevasses(
            terrain,
            x,
            z,
            (_plan(blue_probability=1.0),),
            seed=73,
            sea_level=61.0,
            surface_material_id=surface_material,
        ),
    )


def test_crevasse_ids_survive_tile_and_raster_export(tmp_path) -> None:
    x, z = np.meshgrid(np.arange(64.0), np.arange(64.0), indexing="xy")
    terrain = np.full(x.shape, 180.0)
    material = np.full(x.shape, 3, dtype=np.uint8)
    plan = _plan(blue_probability=1.0)
    crevasses = sample_glacial_crevasses(
        terrain,
        x,
        z,
        (plan,),
        seed=13,
        sea_level=61.0,
        surface_material_id=material,
    )
    tile_field = Heightfield(
        height=terrain,
        moisture=np.full(x.shape, 0.5),
        water_level=np.full(x.shape, -1.0),
        glacial_crevasse_id=crevasses,
        glacial_crevasse_palette=GLACIAL_CREVASSE_PALETTE,
    )
    tile = to_bong_tile(tile_field, sea_level=61.0)
    assert np.array_equal(tile.glacial_crevasse_id, crevasses)
    manifest = write_bong_raster(tile, tmp_path, write_manifest=True)
    payload = json.loads(manifest.read_text())
    assert (tmp_path / "tile_0_0" / "glacial_crevasse_id.bin").exists()
    assert payload["glacial_crevasse_encoding"]["file"] == "glacial_crevasse_id.bin"
    assert "glacial_crevasse_id" in payload["tiles"][0]["layers"]
