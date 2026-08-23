"""Human-readable procedural terrain recipes.

These are Python values by design. They are versioned, importable, and easy to
review without mixing configuration with the generator implementation.
"""

from __future__ import annotations

from ..engine.models import (
    Basin,
    CaveNetwork,
    MountainRange,
    NoiseLayer,
    Point,
    River,
    TerrainRecipe,
)


DEFAULT_RECIPE = TerrainRecipe(
    name="bong_himalayan_baseline",
    base_height=68.0,
    base_noise=(
        NoiseLayer(kind="fbm", scale=1900.0, amplitude=10.0, octaves=4, gain=0.55),
        NoiseLayer(kind="fbm", scale=420.0, amplitude=5.0, octaves=3, gain=0.52, seed_offset=17),
        NoiseLayer(
            kind="warp",
            scale=260.0,
            amplitude=1.0,
            octaves=3,
            gain=0.54,
            seed_offset=23,
            warp_scale=1700.0,
            warp_strength=160.0,
        ),
    ),
    basins=(
        Basin(Point(-1600.0, 1200.0), radius_x=1800.0, radius_z=1200.0, depth=12.0),
        Basin(Point(2600.0, 2600.0), radius_x=1000.0, radius_z=760.0, depth=8.0),
    ),
    mountains=(
        MountainRange(
            path=(
                Point(-8500.0, -6100.0),
                Point(-4700.0, -4200.0),
                Point(-1200.0, -2500.0),
                Point(2300.0, -900.0),
                Point(6400.0, 700.0),
            ),
            # Slightly wider than the first preview: keep the crest height but
            # reduce the apparent edge sharpness by roughly three percent.
            width=927.0,
            height=135.0,
            roughness_contrast=0.53,
            valley_depth=17.5,
            roughness=NoiseLayer(
                kind="ridge", scale=380.0, amplitude=1.0, octaves=1, gain=0.5, seed_offset=41
            ),
        ),
        MountainRange(
            path=(
                Point(-7200.0, 6400.0),
                Point(-4400.0, 5200.0),
                Point(-1700.0, 4700.0),
            ),
            width=742.0,
            height=95.0,
            roughness_contrast=0.53,
            valley_depth=11.7,
            roughness=NoiseLayer(
                kind="ridge", scale=300.0, amplitude=1.0, octaves=1, gain=0.5, seed_offset=73
            ),
        ),
    ),
    rivers=(
        River(
            path=(
                Point(-1200.0, -2500.0),
                Point(-700.0, -1400.0),
                Point(150.0, -300.0),
                Point(1000.0, 1000.0),
                Point(1600.0, 3200.0),
            ),
            width=22.0,
            depth=7.0,
            widening=3.0,
            bed_materials=("mud", "gravel", "sand", "clay"),
        ),
        River(
            path=(Point(-4500.0, -4200.0), Point(-3500.0, -2600.0), Point(-1900.0, -1200.0)),
            width=10.0,
            depth=4.0,
            widening=2.0,
            bed_materials=("dirt", "gravel", "sand"),
        ),
    ),
    caves=(
        CaveNetwork(
            name="shallow_mine_network",
            paths=(
                (
                    Point(-920.0, -2180.0),
                    Point(-820.0, -2050.0),
                    Point(-700.0, -2140.0),
                ),
                (
                    Point(-820.0, -2050.0),
                    Point(-650.0, -1950.0),
                ),
                (
                    Point(-820.0, -2050.0),
                    Point(-860.0, -1900.0),
                ),
            ),
            width=2.5,
            height=4,
            depth=10.0,
        ),
    ),
    sea_level=61.0,
)

__all__ = ["DEFAULT_RECIPE"]
