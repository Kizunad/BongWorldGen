"""Data contracts shared by the procedural engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


NoiseKind = Literal["value", "ridge", "fbm", "warp"]


@dataclass(frozen=True)
class Point:
    x: float
    z: float


@dataclass(frozen=True)
class NoiseLayer:
    """A deterministic noise contribution in world-coordinate units."""

    kind: NoiseKind = "fbm"
    scale: float = 512.0
    amplitude: float = 1.0
    octaves: int = 4
    lacunarity: float = 2.0
    gain: float = 0.5
    seed_offset: int = 0
    warp_scale: float = 1200.0
    warp_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("noise scale must be positive")
        if self.octaves < 1:
            raise ValueError("noise octaves must be at least 1")
        if self.lacunarity <= 1.0:
            raise ValueError("noise lacunarity must be greater than 1")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("noise gain must be in (0, 1]")
        if self.warp_scale <= 0 or self.warp_strength < 0:
            raise ValueError("noise warp scale must be positive and strength non-negative")


@dataclass(frozen=True)
class MountainRange:
    """A controllable ridge system described by a polyline."""

    path: tuple[Point, ...]
    width: float
    height: float
    roughness: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="ridge", scale=420.0, amplitude=1.0)
    )
    roughness_contrast: float = 0.55
    valley_depth: float = 0.0

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("mountain path needs at least two points")
        if self.width <= 0 or self.height < 0 or self.valley_depth < 0:
            raise ValueError("mountain width must be positive and heights non-negative")
        if not 0.0 <= self.roughness_contrast <= 1.0:
            raise ValueError("mountain roughness contrast must be in [0, 1]")


@dataclass(frozen=True)
class River:
    """A river carved along a polyline; width can taper from source to outlet."""

    path: tuple[Point, ...]
    width: float
    depth: float
    widening: float = 1.0

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("river path needs at least two points")
        if self.width <= 0 or self.depth < 0 or self.widening <= 0:
            raise ValueError("river width/depth must be positive")


@dataclass(frozen=True)
class Basin:
    """An elliptical depression used for lakes and lowland bowls."""

    center: Point
    radius_x: float
    radius_z: float
    depth: float

    def __post_init__(self) -> None:
        if self.radius_x <= 0 or self.radius_z <= 0 or self.depth < 0:
            raise ValueError("basin radii must be positive and depth non-negative")


@dataclass(frozen=True)
class TerrainRecipe:
    """Project data only: the engine interprets this recipe without mutation."""

    name: str
    base_height: float = 64.0
    base_noise: tuple[NoiseLayer, ...] = ()
    basins: tuple[Basin, ...] = ()
    mountains: tuple[MountainRange, ...] = ()
    rivers: tuple[River, ...] = ()
    sea_level: float = 62.0
    moisture_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(scale=1800.0, amplitude=1.0, octaves=3)
    )


@dataclass(frozen=True)
class Heightfield:
    """All scalar fields produced by one deterministic generation pass."""

    height: np.ndarray
    moisture: np.ndarray
    water_level: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.height.shape, self.moisture.shape, self.water_level.shape}
        if len(shapes) != 1:
            raise ValueError(f"heightfield layers must share a shape, got {sorted(shapes)}")
        if self.height.ndim != 2:
            raise ValueError("heightfield layers must be two-dimensional")
        for name in ("height", "moisture", "water_level"):
            value = getattr(self, name)
            if not np.isfinite(value).all():
                raise ValueError(f"heightfield layer {name} contains non-finite values")
