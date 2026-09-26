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


def draw_profile(ax, directory, row, light, *, show_boundaries=False):
    with np.load(directory / f"{row['zone']}.npz") as data:
        rgb = light.shade(data["height"], cmap=plt.get_cmap("terrain"), vmin=45, vmax=310,
                          dx=float(data["cell_size"]), dy=float(data["cell_size"]), vert_exag=2)
        rgb[data["water"] >= 0, :3] = [0.16, 0.40, 0.68]
        ax.imshow(rgb, origin="upper")
        if show_boundaries:
            ax.contour(data["weight"], levels=[0.5], colors="white", linewidths=0.6)
    ax.set_title(f"{row['profile']}\nmean {row['mean_height']:.1f} | std {row['std_height']:.1f}")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("generated/zone-evidence"))
    parser.add_argument("--compare", type=Path, help="earlier zone_evidence output on the same grid and seed")
    parser.add_argument("--show-boundaries", action="store_true",
                        help="diagnostic overlay of the 0.5 zone weight contour; not terrain geometry")
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    profiles = {}
    for row in report["zones"]:
        profiles.setdefault(row["profile"], row)
    fig, axes = plt.subplots(5, 3, figsize=(16, 21), constrained_layout=True)
    light = LightSource(azdeg=315, altdeg=50)
    for ax, (profile, row) in zip(axes.flat, profiles.items()):
        draw_profile(ax, args.directory, row, light, show_boundaries=args.show_boundaries)
    fig.suptitle(f"Zone terrain evidence | seed {report['seed']} | common elevation scale 45-310", fontsize=18)
    fig.savefig(args.directory / "profiles.png", dpi=130)
    plt.close(fig)

    if args.compare:
        before = json.loads((args.compare / "report.json").read_text())
        if before["seed"] != report["seed"]:
            parser.error("comparison requires the same seed")
        old_rows = {row["zone"]: row for row in before["zones"]}
        changed = []
        for profile, row in profiles.items():
            old_row = old_rows.get(row["zone"])
            if old_row is None or old_row["profile"] != profile:
                parser.error(f"comparison missing matching zone: {row['zone']}")
            with np.load(args.compare / f"{row['zone']}.npz") as old, \
                    np.load(args.directory / f"{row['zone']}.npz") as new:
                if (old["height"].shape != new["height"].shape or
                        any(not np.array_equal(old[key], new[key]) for key in ("origin", "cell_size", "weight"))):
                    parser.error(f"comparison grids or boundaries differ: {row['zone']}")
                # Include ash as an unchanged control alongside the revised shapes.
                if profile == "ash_dead_zone" or any(
                        not np.array_equal(old[key], new[key]) for key in ("height", "water", "spans")):
                    changed.append((old_row, row))
        if changed:
            fig, axes = plt.subplots(len(changed), 2, figsize=(12, 3.4 * len(changed)),
                                     constrained_layout=True, squeeze=False)
            for (left, right), (old_row, row) in zip(axes, changed):
                draw_profile(left, args.compare, old_row, light, show_boundaries=args.show_boundaries)
                draw_profile(right, args.directory, row, light, show_boundaries=args.show_boundaries)
                left.set_title("Before | " + left.get_title())
                right.set_title("After | " + right.get_title())
            fig.suptitle(f"Morphology comparison | seed {report['seed']} | common elevation scale 45-310",
                         fontsize=16)
            fig.savefig(args.directory / "profiles-before-after.png", dpi=130)
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
