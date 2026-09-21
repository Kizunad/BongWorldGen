#!/usr/bin/env python3
"""Compare sparse zone sampling against the dense weighted-sum reference."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.engine import sample_surface


def dense_surface(composer, x, z):
    """Original all-columns evaluation of the world-coordinate blend."""
    blend = composer.index.query(x, z)
    height, moisture = sample_surface(composer.background, x, z, composer.seed)
    height *= blend.background
    moisture *= blend.background
    for part in blend.contributions:
        local_height, local_moisture = sample_surface(composer.recipes[part.zone.name], x, z, composer.seed)
        height += part.weight * local_height
        moisture += part.weight * local_moisture
    return height, moisture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=401)
    parser.add_argument("--height", type=int, default=409)
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("generated/zone-sampling-benchmark/benchmark.json"))
    args = parser.parse_args()
    if min(args.width, args.height, args.repeat) < 1:
        parser.error("dimensions and repeat count must be positive")
    composer = ZoneTerrain(seed=args.seed)
    bounds = composer.world.bounds
    x, z = np.meshgrid(np.linspace(bounds.min_x, bounds.max_x, args.width),
                       np.linspace(bounds.min_z, bounds.max_z, args.height))
    durations = {"dense": [], "sparse": []}
    for _ in range(args.repeat):
        start = time.perf_counter()
        reference = dense_surface(composer, x, z)
        durations["dense"].append(time.perf_counter() - start)
        start = time.perf_counter()
        actual = composer.sample_surface(x, z)
        durations["sparse"].append(time.perf_counter() - start)
        for name, expected, observed in zip(("height", "moisture"), reference, actual):
            if observed.tobytes() != expected.tobytes():
                raise AssertionError(f"{name} is not byte-identical to dense reference")
    blend = composer.index.query(x, z)
    dense_seconds, sparse_seconds = (statistics.median(durations[key]) for key in ("dense", "sparse"))
    result = {
        "seed": args.seed, "shape": [args.height, args.width], "repeat": args.repeat,
        "byte_identical": True, "dense_seconds": dense_seconds, "sparse_seconds": sparse_seconds,
        "speedup": dense_seconds / sparse_seconds,
        "dense_recipe_samples": x.size * (1 + len(blend.contributions)),
        "nonzero_recipe_samples": int(np.count_nonzero(blend.background)
                                      + sum(np.count_nonzero(p.weight) for p in blend.contributions)),
        "height_sha256": hashlib.sha256(actual[0].tobytes()).hexdigest(),
        "moisture_sha256": hashlib.sha256(actual[1].tobytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
