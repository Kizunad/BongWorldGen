#!/usr/bin/env python3
"""Plot zone_evidence outputs (requires matplotlib, no regeneration)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("generated/zone-evidence"))
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    profiles = {}
    for row in report["zones"]:
        profiles.setdefault(row["profile"], row)
    fig, axes = plt.subplots(5, 3, figsize=(16, 21), constrained_layout=True)
    light = LightSource(azdeg=315, altdeg=50)
    for ax, (profile, row) in zip(axes.flat, profiles.items()):
        data = np.load(args.directory / f"{row['zone']}.npz")
        values = data["height"]
        rgb = light.shade(values, cmap=plt.get_cmap("terrain"), vmin=45, vmax=310,
                          dx=float(data["cell_size"]), dy=float(data["cell_size"]), vert_exag=2)
        rgb[data["water"] >= 0, :3] = [0.16, 0.40, 0.68]
        ax.imshow(rgb, origin="upper")
        ax.contour(data["weight"], levels=[0.5], colors="white", linewidths=0.6)
        ax.set_title(f"{profile}\nmean {row['mean_height']:.1f} | std {row['std_height']:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Zone terrain evidence | seed {report['seed']} | common elevation scale 45-310", fontsize=18)
    fig.savefig(args.directory / "profiles.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    island = np.load(args.directory / "celestial_isles.npz")
    spans = island["spans"][48]
    xs = island["origin"][0] + np.arange(spans.shape[0]) * island["cell_size"]
    for slot in range(4):
        valid = spans[:, slot, 0] != 32767
        ax.fill_between(xs, spans[:, slot, 0], spans[:, slot, 1], where=valid,
                        color="#547342", step="mid")
    ax.set(xlabel="World X (Z = 1200)", ylabel="World Y", ylim=(-64, 320),
           title="Celestial isles: actual solid spans and air below")
    fig.savefig(args.directory / "island-section.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
