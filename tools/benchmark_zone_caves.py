#!/usr/bin/env python3
"""Measure real cave windows and compare all heightfield layers by SHA-256."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.engine.caves import generator


def snapshot(field) -> dict:
    result = {}
    for member in fields(field):
        value = getattr(field, member.name)
        if isinstance(value, np.ndarray):
            result[member.name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                   "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        else:
            result[member.name] = repr(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--output", type=Path, default=Path("generated/zone-cave-benchmark/result.json"))
    parser.add_argument("--compare", type=Path, help="require identical layers to an earlier report")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    composer = ZoneTerrain(seed=args.seed)
    # Prepare rivers outside the timer; the benchmark concerns cave generation.
    composer.river_profiles
    rows = []
    original = generator.sample_noise_3d
    counts = [0, 0]

    def counted(x, *positional, **keywords):
        counts[0] += 1
        counts[1] += x.size
        return original(x, *positional, **keywords)

    generator.sample_noise_3d = counted
    try:
        for name, x, z in (("youan_depths", 1984, 2984), ("wuxing_abyss", 5224, 1376),
                           ("baolongwang_cavern_deep", 1720, -5184)):
            times, snapshots, noise = [], [], []
            for _ in range(args.repeat):
                counts[:] = [0, 0]
                start = time.perf_counter()
                field = composer.generate(width=64, height=64, origin_x=x, origin_z=z)
                times.append(time.perf_counter() - start)
                snapshots.append(snapshot(field))
                noise.append(tuple(counts))
            assert all(value == snapshots[0] for value in snapshots), name
            assert len(set(noise)) == 1, name
            rows.append({"zone": name, "origin": [x, z], "shape": [64, 64],
                         "seconds": times, "median_seconds": statistics.median(times),
                         "noise_calls": noise[0][0], "noise_samples": noise[0][1],
                         "layers": snapshots[0]})
            print(f"{name}: {rows[-1]['median_seconds']:.3f}s, {noise[0][0]} noise calls", flush=True)
    finally:
        generator.sample_noise_3d = original
    report = {"seed": args.seed, "repeat": args.repeat, "windows": rows}
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        assert baseline["seed"] == report["seed"], "seed differs"
        before = [(row["zone"], row["origin"], row["shape"], row["layers"]) for row in baseline["windows"]]
        after = [(row["zone"], row["origin"], row["shape"], row["layers"]) for row in rows]
        assert before == after, "heightfield layers differ from baseline"
        report["identical_to"] = str(args.compare)
        print("All layer shapes, dtypes and SHA-256 hashes match the baseline.", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
