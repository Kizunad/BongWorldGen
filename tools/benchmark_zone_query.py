#!/usr/bin/env python3
"""Compare culled zone weights against the original all-footprints query."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bong_worldgen.composition import ZoneIndex
from bong_worldgen.composition import layout
from bong_worldgen.data.world_definition import WORLD


def unculled(index, x, z):
    remaining = np.ones_like(x)
    parts = []
    for zone in index.zones:
        alpha = layout.boundary_alpha(zone, layout.boundary_distance(zone, x, z))
        weight = remaining * alpha
        if np.any(weight):
            parts.append(layout.ZoneContribution(zone, weight))
        remaining *= 1 - alpha
    return layout.ZoneBlend(tuple(parts), remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("generated/zone-query-benchmark/result.json"))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    index = ZoneIndex(WORLD.zones)
    bounds = WORLD.bounds
    windows = [
        ("world", *np.meshgrid(np.linspace(bounds.min_x, bounds.max_x, 401),
                              np.linspace(bounds.min_z, bounds.max_z, 409))),
        ("spawn_tile", *np.meshgrid(np.arange(-128, 128, dtype=float), np.arange(-128, 128, dtype=float))),
        ("background_tile", *np.meshgrid(np.arange(-10000, -9744, dtype=float),
                                        np.arange(12000, 12256, dtype=float))),
    ]
    rows = []
    original = layout.boundary_distance
    counts = [0, 0]

    def counted(zone, x, z):
        counts[0] += 1
        counts[1] += x.size
        return original(zone, x, z)

    layout.boundary_distance = counted
    try:
        for name, x, z in windows:
            durations = {"unculled": [], "culled": []}
            for _ in range(args.repeat):
                counts[:] = [0, 0]
                start = time.perf_counter()
                reference = unculled(index, x, z)
                durations["unculled"].append(time.perf_counter() - start)
                before = counts.copy()
                counts[:] = [0, 0]
                start = time.perf_counter()
                actual = index.query(x, z)
                durations["culled"].append(time.perf_counter() - start)
                after = counts.copy()
                assert [p.zone.name for p in reference.contributions] == [p.zone.name for p in actual.contributions]
                assert reference.background.tobytes() == actual.background.tobytes()
                for left, right in zip(reference.contributions, actual.contributions):
                    assert left.weight.tobytes() == right.weight.tobytes(), left.zone.name
            digest = hashlib.sha256(actual.background.tobytes())
            for part in actual.contributions:
                digest.update(part.zone.name.encode())
                digest.update(part.weight.tobytes())
            rows.append({"window": name, "shape": list(x.shape), "byte_identical": True,
                         "unculled_seconds": statistics.median(durations["unculled"]),
                         "culled_seconds": statistics.median(durations["culled"]),
                         "unculled_samples": before[1], "culled_samples": after[1],
                         "unculled_calls": before[0], "culled_calls": after[0],
                         "weights_sha256": digest.hexdigest()})
    finally:
        layout.boundary_distance = original
    report = {"repeat": args.repeat, "windows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
