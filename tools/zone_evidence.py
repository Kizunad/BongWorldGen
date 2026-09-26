#!/usr/bin/env python3
"""Reproduce zone metrics, actual solids, POI placement and crop invariance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.composition.pois import resolve_world_pois
from bong_worldgen.adapters import generate_zone_tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/zone-evidence"))
    parser.add_argument("--seed", type=int, default=812731)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    composer = ZoneTerrain(seed=args.seed)
    rows = []
    for zone in composer.world.zones:
        # Include the complete authored footprint and its transition band.
        extent = max(zone.size_x, zone.size_z) + 2 * zone.boundary_width
        stride = max(1, int(np.ceil(extent / 96)))
        size = 97
        origin_x, origin_z = zone.center_x - stride * 48, zone.center_z - stride * 48
        field, tile = generate_zone_tile(composer, width=size, height=size, origin_x=origin_x,
                                         origin_z=origin_z, cell_size=stride)
        x, z = np.meshgrid(origin_x + np.arange(size) * stride, origin_z + np.arange(size) * stride)
        weights = composer.index.query(x, z).weight_for(zone.name)
        core = weights >= 0.9
        if not np.any(core):
            raise ValueError(f"zone {zone.name} has no core in evidence sample")
        row = {
            "zone": zone.name, "profile": zone.terrain_profile,
            "mean_height": float(field.height[core].mean()),
            "std_height": float(field.height[core].std()),
            "min_height": float(field.height[core].min()),
            "max_height": float(field.height[core].max()),
            "wet_fraction": float((field.water_level[core] >= 0).mean()),
            "max_solid_spans": int(np.count_nonzero(field.solid_spans[..., 0] != 32767, axis=-1).max()),
            "height_sha256": hashlib.sha256(field.height.tobytes()).hexdigest(),
            "blackstone_fraction": float((tile.surface_id[core] == tile.surface_palette.index("blackstone")).mean()),
        }
        # Check the same coordinates when generated in two separate crops.
        for offset, width in ((0, 48), (48, 49)):
            part, part_tile = generate_zone_tile(composer, width=width, height=size,
                origin_x=origin_x + offset * stride, origin_z=origin_z, cell_size=stride)
            for layer in ("height", "water_level", "riverbed_id", "solid_spans", "cave_id"):
                np.testing.assert_array_equal(getattr(part, layer),
                                              getattr(field, layer)[:, offset:offset + width])
            np.testing.assert_array_equal(part_tile.surface_id, tile.surface_id[:, offset:offset + width])
        row["split_equal"] = True
        np.savez_compressed(args.output / f"{zone.name}.npz", height=field.height,
                            water=field.water_level, spans=field.solid_spans, weight=weights,
                            origin=np.array([origin_x, origin_z]), cell_size=stride,
                            surface_id=tile.surface_id, surface_palette=np.asarray(tile.surface_palette))
        rows.append(row)
        print(f"{zone.name}: mean={row['mean_height']:.2f}, std={row['std_height']:.2f}, split=exact", flush=True)
    pois = [poi.manifest() for poi in resolve_world_pois(composer)]
    report = {"seed": args.seed, "zones": rows, "pois": pois, "zone_count": len(rows),
              "profile_count": len({row['profile'] for row in rows}), "poi_count": len(pois),
              "sampling": "97 x 97, integer stride; metrics use zone weight >= 0.9"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
