"""Command line entry point for a standalone procedural generation pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .adapters import to_bong_tile
from .data.recipes import DEFAULT_RECIPE
from .engine import generate_heightfield


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
    field = generate_heightfield(
        DEFAULT_RECIPE,
        width=args.width,
        height=args.height,
        seed=args.seed,
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
        uplift=tile.uplift,
        glacial_landform_id=tile.glacial_landform_id,
        glacial_landform_palette=np.asarray(tile.glacial_landform_palette),
        glacial_water_id=tile.glacial_water_id,
        glacial_water_palette=np.asarray(tile.glacial_water_palette),
        glacial_crevasse_id=tile.glacial_crevasse_id,
        glacial_crevasse_palette=np.asarray(tile.glacial_crevasse_palette),
        glacial_discharge=tile.glacial_discharge,
        valley_depth=tile.valley_depth,
        valley_flow_accumulation=tile.valley_flow_accumulation,
        valley_stream_power=tile.valley_stream_power,
        watershed_divide=tile.watershed_divide,
        hydraulic_erosion=tile.hydraulic_erosion,
        hydraulic_deposition=tile.hydraulic_deposition,
        snow_accumulation=tile.snow_accumulation,
        glacial_mass_balance=tile.glacial_mass_balance,
        wind_snow_alignment=tile.wind_snow_alignment,
        wind_snow_response=tile.wind_snow_response,
        biome_id=tile.biome_id,
        feature_mask=tile.feature_mask,
        wilderness_id=tile.wilderness_id,
        riverbed_id=tile.riverbed_id,
        riverbed_palette=np.asarray(tile.riverbed_palette),
        cave_id=tile.cave_id,
        cave_palette=np.asarray(tile.cave_palette),
        fracture_id=tile.fracture_id,
        fracture_palette=np.asarray(tile.fracture_palette),
        surface_material_id=tile.surface_material_id,
        surface_material_palette=np.asarray(tile.surface_material_palette),
        mountain_material_id=tile.mountain_material_id,
        mountain_material_palette=np.asarray(tile.mountain_material_palette),
        mountain_weight=tile.mountain_weight,
        mountain_snowline=tile.mountain_snowline,
        mountain_material_score=tile.mountain_material_score,
        mountain_slope_angle=tile.mountain_slope_angle,
        mountain_exposure=tile.mountain_exposure,
        mountain_rock_exposure=tile.mountain_rock_exposure,
        surface_visible_id=tile.surface_visible_id,
        surface_visible_palette=np.asarray(tile.surface_visible_palette),
        permafrost_id=tile.permafrost_id,
        permafrost_palette=np.asarray(tile.permafrost_palette),
        surface_cover_layers=tile.surface_cover_layers,
        surface_cover_palette=np.asarray(tile.surface_cover_palette),
        climate_id=tile.climate_id,
        climate_palette=np.asarray(tile.climate_palette),
        climate_transition_id=tile.climate_transition_id,
        climate_transition_palette=np.asarray(tile.climate_transition_palette),
        climate_transition_weight=tile.climate_transition_weight,
        climate_surface_id=tile.climate_surface_id,
        climate_surface_palette=np.asarray(tile.climate_surface_palette),
        moisture=field.moisture,
        seed=np.asarray(args.seed, dtype=np.int64),
        recipe=np.asarray(DEFAULT_RECIPE.name),
    )
    print(f"wrote {args.output} ({args.width}x{args.height}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
