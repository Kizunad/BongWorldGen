from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_importer():
    path = Path(__file__).parents[1] / "tools" / "import_legacy_world.py"
    spec = importlib.util.spec_from_file_location("legacy_importer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_import_splits_world_into_zone_modules(tmp_path) -> None:
    importer = _load_importer()
    raw = {
        "version": 1,
        "world": {
            "name": "test_world",
            "spawn_zone": "spawn",
            "bounds_xz": {"min": [-10, -20], "max": [30, 40]},
            "notes": ["kept"],
        },
        "zones": [
            {
                "name": "spawn",
                "display_name": "初醒原",
                "center_xz": [0, 0],
                "size_xz": [20, 20],
                "spirit_qi": 0.35,
                "danger_level": 1,
                "worldgen": {
                    "terrain_profile": "spawn_plain",
                    "shape": "ellipse",
                    "boundary": {"mode": "soft", "width": 96},
                },
                "pois": [{"kind": "shrine", "name": "测试 shrine", "pos_xyz": [1, 2, 3]}],
            }
        ],
    }
    destination = tmp_path / "imported_world.py"
    written = importer.write_package(raw, destination)
    assert len(written) == 5
    assert (tmp_path / "world_metadata.py").exists()
    assert (tmp_path / "world_definition.py").exists()
    assert (tmp_path / "zones" / "spawn.py").exists()
    assert "world_definition" in destination.read_text(encoding="utf-8")
    zone_source = (tmp_path / "zones" / "spawn.py").read_text(encoding="utf-8")
    assert "terrain_profile='spawn_plain'" in zone_source
    assert "PoiDefinition(kind='shrine'" in zone_source


def test_checked_in_world_has_one_module_per_zone() -> None:
    from bong_worldgen.data import WORLD
    from bong_worldgen.data.imported_world import WORLD as COMPAT_WORLD
    from bong_worldgen.data.world import PoiDefinition
    from bong_worldgen.data.zones import ALL_ZONES

    assert COMPAT_WORLD is WORLD
    assert len(ALL_ZONES) == len(WORLD.zones) == 27
    assert {zone.name for zone in ALL_ZONES} == {zone.name for zone in WORLD.zones}
    assert all(
        isinstance(poi, PoiDefinition)
        for zone in WORLD.zones
        for poi in zone.pois
    )
