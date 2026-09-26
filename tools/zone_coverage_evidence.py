#!/usr/bin/env python3
"""Compare the spatial footprint of landforms on saved, matching zone samples."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.engine import sample_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, default=Path("generated/zone-evidence"))
    parser.add_argument("--output", type=Path, default=Path("generated/zone-coverage/report.json"))
    args = parser.parse_args()
    previous = json.loads((args.before / "report.json").read_text())
    current = json.loads((args.after / "report.json").read_text())
    if previous["seed"] != current["seed"]:
        parser.error("before and after require the same seed")
    composer = ZoneTerrain(seed=current["seed"])
    rows = []
    for zone in composer.world.zones:
        if zone.terrain_profile not in ("cave_network", "abyssal_maze", "tribulation_scorch"):
            continue
        with np.load(args.before / f"{zone.name}.npz") as old, np.load(args.after / f"{zone.name}.npz") as new:
            for key in ("origin", "cell_size", "weight"):
                np.testing.assert_array_equal(old[key], new[key])
            if old["height"].shape != new["height"].shape:
                parser.error(f"different sampling grids for {zone.name}")
            h, w = new["height"].shape
            x, z = np.meshgrid(new["origin"][0] + np.arange(w) * new["cell_size"],
                               new["origin"][1] + np.arange(h) * new["cell_size"])
            size_x = min(zone.size_x, zone.size_z) if zone.shape == "circular" else zone.size_x
            size_z = size_x if zone.shape == "circular" else zone.size_z
            lx, lz = (x - zone.center_x) / size_x, (z - zone.center_z) / size_z
            core = new["weight"] > 0.99
            outer = core & (np.hypot(lx, lz) > 0.25)
            recipe = replace(composer.recipes[zone.name], basins=(), mountains=(), plateaus=())
            plain, _ = sample_surface(recipe, x, z, composer.seed)
            row = {"zone": zone.name, "profile": zone.terrain_profile, "core_samples": int(core.sum()),
                   "outer_samples": int(outer.sum()), "baseline_recipe": repr(recipe)}
            for label, data in (("before", old), ("after", new)):
                affected = np.abs(data["height"] - plain) > 6
                row[label] = {"core_fraction": float(affected[core].mean()),
                              "outer_fraction": float(affected[outer].mean()),
                              "height_sha256": hashlib.sha256(data["height"].tobytes()).hexdigest()}
            rows.append(row)
            print(f"{zone.name}: core {row['before']['core_fraction']:.1%} -> {row['after']['core_fraction']:.1%}; "
                  f"outer {row['before']['outer_fraction']:.1%} -> {row['after']['outer_fraction']:.1%}")
    result = {"seed": composer.seed, "before_directory": str(args.before), "after_directory": str(args.after),
              "definition": "Absolute relief >6 blocks relative to the SAME base height/noise without basins, mountains or plateaus; "
                            "core weight >0.99; outer fractional radius >0.25. Base height/noise must be unchanged between versions.",
              "zones": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
