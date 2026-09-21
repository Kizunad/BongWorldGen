"""Compose authored world regions before engine water and cave generation."""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property

import numpy as np

from ..data.recipes import DEFAULT_RECIPE
from ..data.world import WorldDefinition
from ..data.world_definition import WORLD
from ..engine import Heightfield, TerrainRecipe, finish_heightfield, prepare_river_profiles, sample_surface
from .layout import ZoneIndex
from .profiles import PROFILE_RECIPES, recipe_for_zone


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
        self.index = ZoneIndex(world.zones)
        unknown = {zone.terrain_profile for zone in world.zones} - PROFILE_RECIPES.keys()
        if unknown:
            raise ValueError(f"unknown terrain profiles: {sorted(unknown)}")
        self.recipes = {zone.name: recipe_for_zone(zone) for zone in self.index.zones}
        self.feature_recipe = replace(
            background,
            caves=background.caves + tuple(cave for recipe in self.recipes.values() for cave in recipe.caves),
            floating_islands=background.floating_islands + tuple(
                island for recipe in self.recipes.values() for island in recipe.floating_islands
            ),
        )

    def sample_surface(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64))
        blend = self.index.query(x, z)
        terrain = np.zeros(x.shape, dtype=np.float64)
        moisture = np.zeros(x.shape, dtype=np.float64)
        recipes = [(self.background, blend.background)] + [
            (self.recipes[part.zone.name], part.weight) for part in blend.contributions
        ]
        for recipe, weight in recipes:
            active = weight > 0
            if not np.any(active):
                continue
            # Engine samplers accept arbitrary coordinate arrays. Dense zones
            # keep their fast contiguous path; partial zones sample only the
            # contributing columns, preserving world coordinates and order.
            selection = Ellipsis if np.all(active) else active
            local_height, local_moisture = sample_surface(recipe, x[selection], z[selection], self.seed)
            terrain[selection] += weight[selection] * local_height
            moisture[selection] += weight[selection] * local_moisture
        return terrain, moisture

    @cached_property
    def river_profiles(self):
        return prepare_river_profiles(self.feature_recipe, lambda x, z: self.sample_surface(x, z)[0])

    def generate(
        self,
        *,
        width: int,
        height: int,
        origin_x: float = 0,
        origin_z: float = 0,
        cell_size: float = 1,
        include_floating_islands: bool = True,
    ) -> Heightfield:
        """Sample final geometry, optionally exposing ground below island caps."""

        if width < 1 or height < 1:
            raise ValueError("heightfield dimensions must be positive")
        if cell_size <= 0 or not np.isfinite((origin_x, origin_z, cell_size)).all():
            raise ValueError("coordinates must be finite and cell_size positive")
        x, z = np.meshgrid(
            origin_x + np.arange(width, dtype=np.float64) * cell_size,
            origin_z + np.arange(height, dtype=np.float64) * cell_size,
        )
        terrain, moisture = self.sample_surface(x, z)
        features = self.feature_recipe if include_floating_islands else replace(
            self.feature_recipe, floating_islands=(),
        )
        return finish_heightfield(
            features, terrain, moisture, x, z, self.seed,
            river_profiles=self.river_profiles,
        )
