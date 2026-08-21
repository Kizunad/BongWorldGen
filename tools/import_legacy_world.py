#!/usr/bin/env python3
"""Split Bong's legacy world JSON into small, typed Python source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
from typing import Any


_POI_FIELDS = ("kind", "name", "pos_xyz", "tags", "unlock", "qi_affinity", "danger_bias")


def _as_poi(raw: dict[str, Any]) -> dict[str, Any]:
    position = raw["pos_xyz"]
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"POI {raw.get('name', '<unnamed>')!r} must have a 3D position")
    return {
        "kind": str(raw["kind"]),
        "name": str(raw.get("name", raw["kind"])),
        "pos_xyz": tuple(float(value) for value in position),
        "tags": tuple(str(value) for value in raw.get("tags", ())),
        "unlock": str(raw.get("unlock", "")),
        "qi_affinity": float(raw.get("qi_affinity", 0.0)),
        "danger_bias": int(raw.get("danger_bias", 0)),
    }


def _as_zone(raw: dict[str, Any]) -> dict[str, Any]:
    worldgen = raw["worldgen"]
    center = raw["center_xz"]
    size = raw["size_xz"]
    if not isinstance(worldgen, dict):
        raise ValueError(f"zone {raw['name']!r} has invalid worldgen data")
    boundary = worldgen["boundary"]
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError(f"zone {raw['name']!r} must have a 2D center")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"zone {raw['name']!r} must have a 2D size")
    if not isinstance(boundary, dict):
        raise ValueError(f"zone {raw['name']!r} has invalid boundary data")
    return {
        "name": str(raw["name"]),
        "display_name": str(raw.get("display_name", raw["name"])),
        "center_x": float(center[0]),
        "center_z": float(center[1]),
        "size_x": float(size[0]),
        "size_z": float(size[1]),
        "terrain_profile": str(worldgen["terrain_profile"]),
        "shape": str(worldgen.get("shape", "unknown")),
        "boundary_mode": str(boundary["mode"]),
        "boundary_width": int(boundary["width"]),
        "spirit_qi": float(raw.get("spirit_qi", 0.0)),
        "danger_level": int(raw.get("danger_level", 0)),
        "pois": tuple(_as_poi(poi) for poi in raw.get("pois", ())),
    }


def _world_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    world = raw["world"]
    bounds = world["bounds_xz"]
    minimum = bounds["min"]
    maximum = bounds["max"]
    if not isinstance(world, dict) or not isinstance(bounds, dict):
        raise ValueError("world must contain a bounds_xz object")
    if not isinstance(minimum, list) or len(minimum) != 2:
        raise ValueError("world.bounds_xz.min must contain x and z")
    if not isinstance(maximum, list) or len(maximum) != 2:
        raise ValueError("world.bounds_xz.max must contain x and z")
    metadata = {
        "version": int(raw.get("version", 1)),
        "name": str(world["name"]),
        "spawn_zone": str(world["spawn_zone"]),
        "min_x": float(minimum[0]),
        "max_x": float(maximum[0]),
        "min_z": float(minimum[1]),
        "max_z": float(maximum[1]),
        "notes": tuple(str(note) for note in world.get("notes", ())),
    }
    zones = tuple(_as_zone(zone) for zone in raw["zones"])
    return metadata, zones


def _module_name(zone_name: str) -> str:
    return zone_name.lower().replace("-", "_")


def _constant_name(zone_name: str) -> str:
    return _module_name(zone_name).upper()


def _render_poi(poi: dict[str, Any]) -> str:
    fields = ", ".join(f"{field}={pformat(poi[field], width=100)}" for field in _POI_FIELDS)
    return f"        PoiDefinition({fields}),"


def _render_zone(zone: dict[str, Any]) -> str:
    lines = [
        '"""Generated zone data. Edit this file for zone-local changes."""',
        "",
        "from ..world import PoiDefinition, ZoneDefinition",
        "",
        "ZONE = ZoneDefinition(",
    ]
    for field in (
        "name",
        "display_name",
        "center_x",
        "center_z",
        "size_x",
        "size_z",
        "terrain_profile",
        "shape",
        "boundary_mode",
        "boundary_width",
        "spirit_qi",
        "danger_level",
    ):
        lines.append(f"    {field}={pformat(zone[field], width=100)},")
    lines.append("    pois=(")
    lines.extend(_render_poi(poi) for poi in zone["pois"])
    lines.extend(("    ),", ")"))
    return "\n".join(lines) + "\n"


def _render_metadata(metadata: dict[str, Any]) -> str:
    return (
        '"""Generated world metadata. Edit this file for world-wide settings."""\n\n'
        "from .world import WorldBounds\n\n"
        f"WORLD_VERSION = {metadata['version']!r}\n"
        f"WORLD_NAME = {metadata['name']!r}\n"
        f"SPAWN_ZONE = {metadata['spawn_zone']!r}\n"
        "WORLD_BOUNDS = WorldBounds(\n"
        f"    min_x={metadata['min_x']!r}, max_x={metadata['max_x']!r},\n"
        f"    min_z={metadata['min_z']!r}, max_z={metadata['max_z']!r},\n"
        ")\n"
        f"WORLD_NOTES = {metadata['notes']!r}\n"
    )


def _render_zone_index(zones: tuple[dict[str, Any], ...]) -> str:
    imports = [
        f"from .{_module_name(zone['name'])} import ZONE as {_constant_name(zone['name'])}"
        for zone in zones
    ]
    constants = ",\n    ".join(_constant_name(zone["name"]) for zone in zones)
    return (
        '"""All authored zones in deterministic source order."""\n\n'
        + "\n".join(imports)
        + "\n\nALL_ZONES = (\n    "
        + constants
        + ",\n)\n\nZONE_BY_NAME = {zone.name: zone for zone in ALL_ZONES}\n"
    )


def _render_world_definition() -> str:
    return (
        '"""Complete typed world definition assembled from metadata and zones."""\n\n'
        "from .world import WorldDefinition\n"
        "from .world_metadata import SPAWN_ZONE, WORLD_BOUNDS, WORLD_NAME, WORLD_NOTES, WORLD_VERSION\n"
        "from .zones import ALL_ZONES\n\n"
        "WORLD = WorldDefinition(\n"
        "    version=WORLD_VERSION,\n"
        "    name=WORLD_NAME,\n"
        "    spawn_zone=SPAWN_ZONE,\n"
        "    bounds=WORLD_BOUNDS,\n"
        "    notes=WORLD_NOTES,\n"
        "    zones=ALL_ZONES,\n"
        ")\n\n"
        "__all__ = ['WORLD']\n"
    )


def _render_compatibility() -> str:
    return (
        '"""Compatibility export for callers migrating from the old monolith."""\n\n'
        "from .world_definition import WORLD\n\n"
        "__all__ = ['WORLD']\n"
    )


def write_package(raw: dict[str, Any], destination: Path) -> list[Path]:
    """Write metadata, one module per zone, an index, and a compatibility shim."""

    metadata, zones = _world_payload(raw)
    data_dir = destination.parent
    zones_dir = data_dir / "zones"
    zones_dir.mkdir(parents=True, exist_ok=True)
    world_definition = data_dir / "world_definition.py"
    written = [
        data_dir / "world_metadata.py",
        zones_dir / "__init__.py",
        world_definition,
        destination,
    ]
    written[0].write_text(_render_metadata(metadata), encoding="utf-8")
    written[1].write_text(_render_zone_index(zones), encoding="utf-8")
    world_definition.write_text(_render_world_definition(), encoding="utf-8")
    destination.write_text(_render_compatibility(), encoding="utf-8")
    for zone in zones:
        path = zones_dir / f"{_module_name(zone['name'])}.py"
        path.write_text(_render_zone(zone), encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    written = write_package(raw, args.destination)
    print(f"wrote {len(written)} Python data files under {args.destination.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
