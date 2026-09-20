"""Compose authored world regions before engine water and cave generation."""

from __future__ import annotations

import numpy as np

from ..data.recipes import DEFAULT_RECIPE
from ..data.world import WorldDefinition
from ..data.world_definition import WORLD
from ..engine import Heightfield, TerrainRecipe, finish_heightfield, sample_surface
from .layout import ZoneIndex
from .profiles import PROFILE_RECIPES, recipe_for_zone


# Explicit staging list: only these known profiles may temporarily use the
# background while the remaining implementation steps land. Typos fail early.
PENDING_PROFILES = frozenset((
    "cave_network", "abyssal_maze", "sky_isle",
))


class ZoneTerrain:
    """A reusable, seeded world sampler shared by raster, CLI and BlueMap."""

    def __init__(
        self,
        world: WorldDefinition = WORLD,
        *,
        background: TerrainRecipe = DEFAULT_RECIPE,
        seed: int = 812731,
    ) -> None:
        self.world, self.background, self.seed = world, background, seed
        ZoneIndex(world.zones)  # Validate even temporarily unimplemented footprints.
        unknown = {zone.terrain_profile for zone in world.zones} - PROFILE_RECIPES.keys() - PENDING_PROFILES
        if unknown:
            raise ValueError(f"unknown terrain profiles: {sorted(unknown)}")
        self.pending_profiles = sorted({zone.terrain_profile for zone in world.zones} & PENDING_PROFILES)
        active = tuple(zone for zone in world.zones if zone.terrain_profile in PROFILE_RECIPES)
        self.index = ZoneIndex(active)
        self.recipes = {zone.name: recipe_for_zone(zone) for zone in self.index.zones}

    def sample_surface(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blend = self.index.query(x, z)
        terrain, moisture = sample_surface(self.background, x, z, self.seed)
        terrain *= blend.background
        moisture *= blend.background
        for part in blend.contributions:
            local_height, local_moisture = sample_surface(self.recipes[part.zone.name], x, z, self.seed)
            terrain += part.weight * local_height
            moisture += part.weight * local_moisture
        return terrain, moisture

    def generate(
        self,
        *,
        width: int,
        height: int,
        origin_x: float = 0,
        origin_z: float = 0,
        cell_size: float = 1,
    ) -> Heightfield:
        if width < 1 or height < 1:
            raise ValueError("heightfield dimensions must be positive")
        if cell_size <= 0 or not np.isfinite((origin_x, origin_z, cell_size)).all():
            raise ValueError("coordinates must be finite and cell_size positive")
        x, z = np.meshgrid(
            origin_x + np.arange(width, dtype=np.float64) * cell_size,
            origin_z + np.arange(height, dtype=np.float64) * cell_size,
        )
        terrain, moisture = self.sample_surface(x, z)
        return finish_heightfield(
            self.background, terrain, moisture, x, z, self.seed,
            surface_sampler=lambda sx, sz: self.sample_surface(sx, sz)[0],
        )
