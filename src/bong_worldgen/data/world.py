"""Stable input data contracts for world layouts imported from Bong."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class PoiDefinition:
    kind: str
    name: str
    pos_xyz: tuple[float, float, float]
    tags: tuple[str, ...] = ()
    unlock: str = ""
    qi_affinity: float = 0.0
    danger_bias: int = 0

    def __post_init__(self) -> None:
        if len(self.pos_xyz) != 3:
            raise ValueError(f"POI {self.name!r} position must contain x, y and z")


@dataclass(frozen=True)
class ZoneDefinition:
    name: str
    display_name: str
    center_x: float
    center_z: float
    size_x: float
    size_z: float
    terrain_profile: str
    shape: str
    boundary_mode: str
    boundary_width: int
    spirit_qi: float
    danger_level: int
    pois: tuple[PoiDefinition, ...] = ()


@dataclass(frozen=True)
class WorldDefinition:
    version: int
    name: str
    spawn_zone: str
    bounds: WorldBounds
    notes: tuple[str, ...]
    zones: tuple[ZoneDefinition, ...]
