#!/usr/bin/env python3
"""Generate one complete world for Bong's existing Three.js preview console."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from bong_worldgen.composition import ZoneIndex
from bong_worldgen.data.world_definition import WORLD
from bong_worldgen.preview_world import export_preview_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/console-world"))
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--zone", help="export this zone's footprint and blend band at world coordinates")
    parser.add_argument("--padding", type=int, default=0, help="extra blocks around --zone bounds")
    for name in ("min-x", "max-x", "min-z", "max-z"):
        parser.add_argument(f"--{name}", type=int, help="explicit inclusive world bounds; supply all four")
    args = parser.parse_args(argv)
    names = ("min_x", "max_x", "min_z", "max_z")
    explicit = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.padding < 0 or (args.padding and not args.zone):
        parser.error("--padding must be nonnegative and requires --zone")
    if args.zone and explicit:
        parser.error("--zone and explicit world bounds are mutually exclusive")
    if explicit and len(explicit) != 4:
        parser.error("explicit bounds require --min-x, --max-x, --min-z and --max-z")
    if args.zone:
        try:
            min_x, max_x, min_z, max_z = ZoneIndex(WORLD.zones).bounds_for(args.zone)
        except ValueError as error:
            parser.error(str(error))
        explicit = {
            "min_x": math.floor(min_x) - args.padding, "max_x": math.ceil(max_x) + args.padding,
            "min_z": math.floor(min_z) - args.padding, "max_z": math.ceil(max_z) + args.padding,
        }
    if explicit and (explicit["min_x"] > explicit["max_x"] or explicit["min_z"] > explicit["max_z"]):
        parser.error("world bounds must be ordered")
    path = export_preview_world(args.output, world=WORLD, seed=args.seed,
                                tile_size=args.tile_size, **explicit)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
