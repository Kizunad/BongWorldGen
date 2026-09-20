"""Command line entry point for a standalone procedural generation pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .adapters import to_bong_tile
from .data.recipes import DEFAULT_RECIPE
from .composition import ZoneTerrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic BongWorldGen heightfield")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--origin-x", type=float, default=-128.0)
    parser.add_argument("--origin-z", type=float, default=-128.0)
    parser.add_argument("--cell-size", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=Path("generated/terrain.npz"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    field = ZoneTerrain(seed=args.seed).generate(
        width=args.width,
        height=args.height,
        origin_x=args.origin_x,
        origin_z=args.origin_z,
        cell_size=args.cell_size,
    )
    tile = to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        height=tile.height,
        surface_id=tile.surface_id,
        subsurface_id=tile.subsurface_id,
        water_level=tile.water_level,
        biome_id=tile.biome_id,
        feature_mask=tile.feature_mask,
        wilderness_id=tile.wilderness_id,
        riverbed_id=tile.riverbed_id,
        riverbed_palette=np.asarray(tile.riverbed_palette),
        cave_id=tile.cave_id,
        cave_palette=np.asarray(tile.cave_palette),
        moisture=field.moisture,
        seed=np.asarray(args.seed, dtype=np.int64),
        recipe=np.asarray(DEFAULT_RECIPE.name),
    )
    print(f"wrote {args.output} ({args.width}x{args.height}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
