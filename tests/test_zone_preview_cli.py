from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import generate_preview_world as cli  # noqa: E402

from bong_worldgen.composition import ZoneIndex, ZoneTerrain
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.data.zones import SPAWN


def test_named_preview_includes_rotated_blend_and_neighbor_ownership(tmp_path, monkeypatch):
    zone = replace(SPAWN, name="rift", shape="rotated_rift", center_x=-37.25,
                   center_z=15.5, size_x=20, size_z=50, boundary_width=12, pois=())
    neighbor = replace(SPAWN, name="neighbor", center_x=-55, center_z=25,
                       size_x=12, size_z=12, boundary_width=6, pois=())
    world = replace(WORLD, zones=(zone, neighbor))
    monkeypatch.setattr(cli, "WORLD", world)
    assert cli.main(["--zone", "rift", "--padding", "2", "--tile-size", "16",
                     "--output", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "rasters/manifest.json").read_text())
    bounds = manifest["world_bounds"]
    # Every nonzero sample of the rotated footprint and its wide transition
    # must lie inside the generated rectangle, including fractional negatives.
    x, z = np.meshgrid(np.arange(-100, 30), np.arange(-50, 90))
    active = ZoneIndex((zone,)).query(x, z).weight_for("rift") > 0
    assert x[active].min() >= bounds["min_x"] and x[active].max() <= bounds["max_x"]
    assert z[active].min() >= bounds["min_z"] and z[active].max() <= bounds["max_z"]
    assert manifest["zone_palette"] == ["neighbor", "rift"]
    column_x, column_z = -55, 25
    entry = next(t for t in manifest["tiles"] if (t["tile_x"], t["tile_z"]) == (-4, 1))
    ids = np.fromfile(tmp_path / "rasters" / entry["dir"] / "zone_id.bin", dtype="u1").reshape(16, 16)
    assert manifest["zone_palette"][ids[column_z % 16, column_x % 16]] == "neighbor"
    assert ZoneTerrain(world).index.zone_at(column_x, column_z).name == "neighbor"


def test_explicit_preview_bounds_match_manifest_coordinates(tmp_path):
    cli.main(["--min-x", "-9", "--max-x", "22", "--min-z", "17", "--max-z", "31",
              "--tile-size", "16", "--output", str(tmp_path)])
    manifest = json.loads((tmp_path / "rasters/manifest.json").read_text())
    assert manifest["world_bounds"] == {"min_x": -9, "max_x": 22, "min_z": 17, "max_z": 31}
    assert manifest["overview"]["origin_x"] == -9 and manifest["overview"]["origin_z"] == 17
    assert {(t["tile_x"], t["tile_z"]) for t in manifest["tiles"]} == {(-1, 1), (0, 1), (1, 1)}


@pytest.mark.parametrize("args", (
    ["--zone", "missing"], ["--min-x", "1"], ["--zone", "spawn", "--min-x", "0"],
    ["--padding", "2"], ["--zone", "spawn", "--padding", "-1"],
    ["--min-x", "1", "--max-x", "0", "--min-z", "0", "--max-z", "0"],
))
def test_invalid_preview_selection_does_not_create_outputs(tmp_path, args):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
