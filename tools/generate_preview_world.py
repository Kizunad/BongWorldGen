#!/usr/bin/env python3
"""Generate one complete world for Bong's existing Three.js preview console."""

from __future__ import annotations

import argparse
from pathlib import Path

from bong_worldgen.preview_world import export_preview_world


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("generated/console-world"))
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    path = export_preview_world(args.output, seed=args.seed, tile_size=args.tile_size)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
