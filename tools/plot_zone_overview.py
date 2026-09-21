#!/usr/bin/env python3
"""Compare exported zone_id tiles with their overview samples, without a web server."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated/zone-overview.png"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    bounds, overview = manifest["world_bounds"], manifest["overview"]
    min_x, min_z = bounds["min_x"], bounds["min_z"]
    width, height = bounds["max_x"] - min_x + 1, bounds["max_z"] - min_z + 1
    detailed = np.full((height, width), 255, dtype=np.uint8)
    size = manifest["tile_size"]
    for tile in manifest["tiles"]:
        tx, tz = tile["tile_x"] * size, tile["tile_z"] * size
        x0, z0 = max(tx, min_x), max(tz, min_z)
        x1, z1 = min(tx + size, min_x + width), min(tz + size, min_z + height)
        if x0 >= x1 or z0 >= z1:
            continue
        layer = np.fromfile(root / tile["dir"] / "zone_id.bin", dtype=np.uint8).reshape(size, size)
        detailed[z0 - min_z:z1 - min_z, x0 - min_x:x1 - min_x] = layer[z0 - tz:z1 - tz, x0 - tx:x1 - tx]
    coarse = np.fromfile(root / overview["zone_file"], dtype=np.uint8).reshape(
        overview["height"], overview["width"],
    )
    xs = overview["origin_x"] + np.arange(overview["width"]) * overview["cell_size"]
    zs = overview["origin_z"] + np.arange(overview["height"]) * overview["cell_size"]
    np.testing.assert_array_equal(coarse, detailed[np.ix_(zs - min_z, xs - min_x)])
    ids = np.unique(detailed)
    colors = [plt.get_cmap("tab20")(i % 20) if zone != 255 else (0.22, 0.24, 0.27, 1)
              for i, zone in enumerate(ids)]
    names = [manifest["zone_palette"][int(zone)] if zone != 255 else "background" for zone in ids]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, data, step, title in zip(axes, (detailed, coarse), (1, overview["cell_size"]),
                                    ("Detailed zone_id", "Overview zone_id (nearest samples)")):
        ax.imshow(np.searchsorted(ids, data), cmap=cmap, vmin=0, vmax=max(1, len(ids) - 1),
                  origin="lower", interpolation="nearest",
                  extent=(min_x, min_x + data.shape[1] * step, min_z, min_z + data.shape[0] * step))
        ax.set(title=title, xlabel="World X", ylabel="World Z",
               xlim=(min_x, min_x + width), ylim=(min_z, min_z + height))
    fig.legend(handles=[Patch(color=color, label=name) for color, name in zip(colors, names)],
               loc="outside lower center", ncol=min(3, len(ids)))
    fig.suptitle(f"All {coarse.size} overview samples match the detailed raster")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    print(f"Verified {coarse.size} samples; wrote {args.output}")


if __name__ == "__main__":
    main()
