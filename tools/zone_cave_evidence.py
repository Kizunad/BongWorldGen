#!/usr/bin/env python3
"""Check two-block air connectivity from cave entrances to authored POIs.

Each continuous vertical air interval is a graph node. Neighboring columns
connect only where their valid feet heights overlap, exactly matching a
six-neighbor voxel search with two blocks of clearance. Surface air is excluded.
This proves geometric connectivity, not walkability, climbability or game locks.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from zipfile import BadZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.composition.pois import resolve_poi
from bong_worldgen.engine.caves.topology import generate_cave_topology
from bong_worldgen.engine.geometry import polyline_distance_and_progress


def air_intervals(surface: np.ndarray, spans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive feet ranges, with the head also at/below the original surface."""

    lower = np.full(spans.shape[:-1], 32767, dtype=np.int16)
    upper = np.full(spans.shape[:-1], -32768, dtype=np.int16)
    cursor = np.rint(surface).astype(np.int32) + 1
    for slot in range(spans.shape[-2]):
        floor, ceiling = spans[..., slot, 0], spans[..., slot, 1]
        valid = (floor != 32767) & (cursor - ceiling - 1 >= 2)
        lower[..., slot] = np.where(valid, ceiling + 1, 32767)
        upper[..., slot] = np.where(valid, cursor - 2, -32768)
        cursor = np.where(floor != 32767, floor, cursor)
    return lower, upper


def connected_intervals(lower: np.ndarray, upper: np.ndarray,
                        source: tuple[int, int, int]) -> np.ndarray:
    """Flood connected air intervals; source is (row, column, feet Y)."""

    row, column, y = source
    seen = np.zeros(lower.shape, dtype=bool)
    queue = deque()
    for slot in range(lower.shape[-1]):
        if lower[row, column, slot] <= y <= upper[row, column, slot]:
            point = (row, column, slot)
            seen[point] = True
            queue.append(point)
    while queue:
        row, column, slot = queue.popleft()
        low, high = lower[row, column, slot], upper[row, column, slot]
        for r, c in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1)):
            if not (0 <= r < lower.shape[0] and 0 <= c < lower.shape[1]):
                continue
            for s in range(lower.shape[-1]):
                point = (r, c, s)
                if not seen[point] and max(low, lower[point]) <= min(high, upper[point]):
                    seen[point] = True
                    queue.append(point)
    return seen


def generation_fingerprint(composer: ZoneTerrain, tile_size: int) -> str:
    """Invalidate checkpoints when inputs, generator sources or NumPy change."""

    digest = hashlib.sha256(repr((composer.seed, tile_size, composer.world,
        composer.background, composer.feature_recipe, tuple(composer.recipes.items()),
        sys.version, np.__version__)).encode())
    for path in [Path(__file__), *sorted((ROOT / "src/bong_worldgen").rglob("*.py"))]:
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tile_intervals(composer: ZoneTerrain, x: int, z: int, tile_size: int,
                   cache: Path | None) -> tuple[np.ndarray, np.ndarray, bool]:
    path = cache / f"{x}_{z}.npz" if cache is not None else None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                lower, upper = data["lower"], data["upper"]
                valid = (lower.shape == upper.shape == (tile_size, tile_size, 4)
                         and lower.dtype == upper.dtype == np.dtype("int16")
                         and np.array_equal(data["origin"], (x, z)))
                if valid:
                    return lower, upper, True
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass  # An interrupted or damaged checkpoint must be regenerated.
    field = composer.generate(width=tile_size, height=tile_size, origin_x=x, origin_z=z)
    lower, upper = air_intervals(field.height, field.solid_spans)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".npz.partial")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, lower=lower, upper=upper, origin=[x, z])
        partial.replace(path)
    return lower, upper, False


def check_zone(composer: ZoneTerrain, zone, output: Path, tile_size: int,
               cache_key: str | None = None) -> dict:
    start = time.perf_counter()
    networks = composer.recipes[zone.name].caves
    paths, points = [], []
    margin = 0.0
    for network in networks:
        index = composer.feature_recipe.caves.index(network)
        topology = generate_cave_topology(network, composer.seed + index * 9973)
        paths.extend(topology.paths)
        points.extend((*topology.chambers, *topology.entrances))
        margin = max(margin, max(network.width, network.chamber_radius, network.entrance_radius)
                     + network.domain_warp_strength + network.smooth_union + 2)
    vertices = points + [point for path in paths for point in path]
    min_x = math.floor((min(p.x for p in vertices) - margin) / tile_size) * tile_size
    min_z = math.floor((min(p.z for p in vertices) - margin) / tile_size) * tile_size
    max_x = math.ceil((max(p.x for p in vertices) + margin + 1) / tile_size) * tile_size
    max_z = math.ceil((max(p.z for p in vertices) + margin + 1) / tile_size) * tile_size
    xs, zs = np.meshgrid(np.arange(min_x, max_x, tile_size), np.arange(min_z, max_z, tile_size))
    cx, cz = xs + (tile_size - 1) / 2, zs + (tile_size - 1) / 2
    distance = np.full(cx.shape, np.inf)
    for path in paths:
        distance = np.minimum(distance, polyline_distance_and_progress(cx, cz, path)[0])
    for point in points:
        distance = np.minimum(distance, np.hypot(cx - point.x, cz - point.z))
    active = distance <= margin + math.sqrt(2) * tile_size / 2
    lower = np.full((max_z - min_z, max_x - min_x, 4), 32767, dtype=np.int16)
    upper = np.full_like(lower, -32768)
    tiles = list(zip(xs[active].tolist(), zs[active].tolist()))
    cache = output / "tile-cache" / cache_key / zone.name if cache_key else None
    reused_tiles = 0
    for index, (x, z) in enumerate(tiles):
        try:
            lo, hi, reused = tile_intervals(composer, x, z, tile_size, cache)
        except ValueError as error:
            error.add_note(f"seed={composer.seed}, zone={zone.name}, tile origin=({x}, {z}), "
                           f"size={tile_size}, tile {index + 1}/{len(tiles)}")
            raise
        reused_tiles += int(reused)
        rows, columns = slice(z - min_z, z - min_z + tile_size), slice(x - min_x, x - min_x + tile_size)
        lower[rows, columns], upper[rows, columns] = lo, hi
        if (index + 1) % 16 == 0 or index + 1 == len(tiles):
            print(f"{zone.name}: read {index + 1}/{len(tiles)} tiles ({reused_tiles} reused)", flush=True)

    entrance = next((poi for poi in zone.pois if poi.kind == "cave_mouth"), None)
    if entrance is not None:
        sx, _, sz = entrance.pos_xyz
    else:
        point = networks[0].entrance_points[0]
        sx, sz = point.x, point.z
    sx, sz = math.floor(sx), math.floor(sz)
    surface = composer.generate(width=1, height=1, origin_x=sx, origin_z=sz).height[0, 0]
    sy = int(np.rint(surface)) - 1
    connected = connected_intervals(lower, upper, (sz - min_z, sx - min_x, sy))
    rows = []
    for poi in zone.pois:
        result = resolve_poi(composer, zone, poi)
        x, y, z = result.pos_xyz
        row, column = math.floor(z) - min_z, math.floor(x) - min_x
        valid = (lower[row, column] <= y) & (y <= upper[row, column])
        reachable = bool(np.any(connected[row, column] & valid))
        rows.append({"name": poi.name, "pos_xyz": list(result.pos_xyz), "reachable": reachable})
    np.savez_compressed(output / f"{zone.name}.npz", lower=lower, upper=upper,
                        connected=connected, origin=[min_x, min_z], tiles=tiles)
    return {"zone": zone.name, "source": [sx, sy, sz], "tiles": len(tiles),
            "sampled_columns": len(tiles) * tile_size**2,
            "cache_key": cache_key, "reused_tiles": reused_tiles,
            "generated_tiles": len(tiles) - reused_tiles, "seconds": time.perf_counter() - start,
            "connected_intervals": int(connected.sum()), "pois": rows,
            "all_pois_reachable": all(row["reachable"] for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--zone", action="append", help="only these zone names (repeatable)")
    parser.add_argument("--resume", action="store_true",
                        help="save and reuse complete tiles with matching generator and world inputs")
    parser.add_argument("--output", type=Path, default=Path("generated/zone-cave-evidence"))
    args = parser.parse_args()
    if args.tile_size < 1:
        parser.error("--tile-size must be positive")
    composer = ZoneTerrain(seed=args.seed)
    zones = [zone for zone in composer.index.zones if composer.recipes[zone.name].caves]
    if args.zone:
        unknown = set(args.zone) - {zone.name for zone in zones}
        if unknown:
            parser.error(f"unknown cave zones: {sorted(unknown)}")
        zones = [zone for zone in zones if zone.name in args.zone]
    args.output.mkdir(parents=True, exist_ok=True)
    cache_key = generation_fingerprint(composer, args.tile_size) if args.resume else None
    report = {"seed": args.seed, "tile_size": args.tile_size,
              "scope": "All authored and generated cave paths; six-neighbor air with two-block clearance; "
                       "surface air excluded. Does not prove walking, climbing, or game unlock rules.",
              "zones": []}
    for zone in zones:
        row = check_zone(composer, zone, args.output, args.tile_size, cache_key)
        report["zones"].append(row)
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{zone.name}: {sum(p['reachable'] for p in row['pois'])}/{len(row['pois'])} POIs reachable",
              flush=True)
    return 0 if all(row["all_pois_reachable"] for row in report["zones"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
