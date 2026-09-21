import json

import numpy as np
import pytest

from bong_worldgen.cli import main
from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.preview_world import export_preview_world


@pytest.mark.parametrize("origin_x,origin_z,min_spans", (
    (-768.25, -0.75, 1), (5180, 1472, 4), (-4400, 1200, 2),
))
def test_cli_preserves_world_coordinates_ownership_and_actual_solids(
    tmp_path, origin_x, origin_z, min_spans,
):
    path = tmp_path / "sample.npz"
    assert main(["--width", "24", "--height", "16", "--cell-size", "2.5",
                 "--origin-x", str(origin_x), "--origin-z", str(origin_z),
                 "--seed", "812731", "--output", str(path)]) == 0
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["origin"], [origin_x, origin_z])
        assert data["cell_size"] == 2.5 and data["seed"] == 812731
        assert data["zone_none_id"] == 255
        assert np.max(np.count_nonzero(data["solid_spans"][..., 0] != 32767, axis=-1)) >= min_spans
        composer = ZoneTerrain()
        field = composer.generate(width=24, height=16, origin_x=origin_x,
                                  origin_z=origin_z, cell_size=2.5)
        np.testing.assert_array_equal(data["solid_spans"], field.solid_spans)
        np.testing.assert_array_equal(data["height"], field.height)
        palette = data["zone_palette"].tolist()
        for oz, ox in ((0, 0), (0, 23), (15, 23)):
            zone = composer.index.zone_at(origin_x + ox * 2.5, origin_z + oz * 2.5)
            assert data["zone_id"][oz, ox] == (255 if zone is None else palette.index(zone.name))
        if origin_x < -700 and origin_z < 0:
            assert data["zone_id"][0, 0] == 255
            assert data["zone_id"][0, -1] == palette.index("spawn")
            assert np.any(data["boundary_weight"] > 0)


def test_cli_and_raster_export_identical_zone_layers_and_spans(tmp_path):
    path = tmp_path / "sample.npz"
    main(["--width", "32", "--height", "32", "--cell-size", "1",
          "--origin-x", "736", "--origin-z", "0", "--seed", "7", "--output", str(path)])
    manifest_path = export_preview_world(
        tmp_path / "world", seed=7, min_x=736, max_x=767, min_z=0, max_z=31, tile_size=32,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tile_dir = manifest_path.parent / "tile_23_0"
    with np.load(path, allow_pickle=False) as data:
        assert data["zone_palette"].tolist() == manifest["zone_palette"]
        assert set(np.unique(data["zone_id"])) == {255, manifest["zone_palette"].index("spawn")}
        for name, dtype in (("zone_id", "u1"), ("boundary_weight", "<f4"),
                             ("water_level", "<f4"), ("surface_id", "u1"),
                             ("wilderness_id", "u1"), ("cave_id", "u1")):
            raster = np.fromfile(tile_dir / f"{name}.bin", dtype=dtype).reshape(32, 32)
            np.testing.assert_array_equal(data[name], raster)
        spans = np.fromfile(tile_dir / "spans.bin", dtype="<i2").reshape(32, 32, 4, 2)
        np.testing.assert_array_equal(data["solid_spans"], spans)
