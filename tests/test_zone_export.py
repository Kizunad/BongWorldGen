import ast
from pathlib import Path
import zlib

import numpy as np

from bong_worldgen.adapters import export_minecraft_world
from bong_worldgen.adapters.minecraft_world import _existing_chunks
from bong_worldgen.composition import ZoneTerrain


def test_appending_world_patches_keeps_existing_chunks_and_is_reproducible(tmp_path):
    composer = ZoneTerrain()
    common = dict(origin_z=0, sea_level=61, seed=812731, world_name="patches", append=True)
    first = composer.generate(width=16, height=16)
    second = composer.generate(width=16, height=16, origin_x=16)
    export_minecraft_world(first, tmp_path, origin_x=0, **common)
    path = tmp_path / "region/r.0.0.mca"
    initial = _existing_chunks(path, 0, 0)
    export_minecraft_world(second, tmp_path, origin_x=16, **common)
    merged = _existing_chunks(path, 0, 0)
    assert set(merged) == {(0, 0), (1, 0)}
    assert merged[(0, 0)] == initial[(0, 0)]
    assert all(zlib.decompress(payload) for payload in merged.values())
    before = path.read_bytes()
    export_minecraft_world(first, tmp_path, origin_x=0, **common)
    assert path.read_bytes() == before


def test_engine_dependency_boundary_excludes_world_layout_and_adapters():
    root = Path(__file__).parents[1] / "src/bong_worldgen/engine"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 2, path
                assert not any(part in (node.module or "").split(".")
                               for part in ("composition", "data", "adapters")), path
            elif isinstance(node, ast.Import):
                assert not any("bong_worldgen" in alias.name and any(
                    part in alias.name.split(".") for part in ("composition", "data", "adapters")
                ) for alias in node.names), path
