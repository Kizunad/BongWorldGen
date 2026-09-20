"""Deterministic world-coordinate footprints and overlap ownership.

Sizes are full diameters. Irregular outlines stay inside those extents. The
source schema has no rotation, so rotated_rift has a documented 30-degree
orientation. Distance is measured along the radial ray, in world blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..data.world import ZoneDefinition


SHAPES = frozenset((
    "ellipse", "circular", "massif", "basin", "elongated", "irregular_blob",
    "rotated_rift", "subterranean_cluster", "plateau",
))
BOUNDARY_MODES = frozenset(("soft", "semi_hard", "hard"))


def boundary_distance(zone: ZoneDefinition, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Positive inside; zero on the authored footprint; negative outside."""

    dx, dz = x - zone.center_x, z - zone.center_z
    if zone.shape == "rotated_rift":
        angle = math.pi / 6.0
        dx, dz = (
            dx * math.cos(angle) + dz * math.sin(angle),
            -dx * math.sin(angle) + dz * math.cos(angle),
        )
    rx, rz = zone.size_x * 0.5, zone.size_z * 0.5
    if zone.shape == "circular":
        rx = rz = min(rx, rz)
    nx, nz = dx / rx, dz / rz
    if zone.shape == "plateau":
        radius = (nx**4 + nz**4) ** 0.25
    else:
        radius = np.hypot(nx, nz)
    if zone.shape in ("massif", "irregular_blob", "subterranean_cluster"):
        angle = np.arctan2(nz, nx)
        outline = 0.90 + 0.06 * np.cos(3.0 * angle) + 0.04 * np.sin(5.0 * angle)
        radius = radius / outline
    distance = np.hypot(dx, dz)
    radial_extent = np.full_like(radius, min(rx, rz))
    np.divide(distance, radius, out=radial_extent, where=radius > 1.0e-12)
    return (1.0 - radius) * radial_extent


@dataclass(frozen=True)
class ZoneContribution:
    zone: ZoneDefinition
    weight: np.ndarray


@dataclass(frozen=True)
class ZoneBlend:
    contributions: tuple[ZoneContribution, ...]
    background: np.ndarray

    def weight_for(self, name: str) -> np.ndarray:
        for contribution in self.contributions:
            if contribution.zone.name == name:
                return contribution.weight
        return np.zeros_like(self.background)


class ZoneIndex:
    """Smaller footprints overlay larger ones; names break equal-area ties.

    Source order, tile size and Python's process-specific hash never affect
    ownership. Uncovered points retain the background recipe with weight one.
    """

    def __init__(self, zones: tuple[ZoneDefinition, ...]) -> None:
        names: set[str] = set()
        for zone in zones:
            if zone.name in names:
                raise ValueError(f"duplicate zone name {zone.name!r}")
            names.add(zone.name)
            if zone.shape not in SHAPES:
                raise ValueError(f"unknown zone shape {zone.shape!r}")
            if zone.boundary_mode not in BOUNDARY_MODES:
                raise ValueError(f"unknown boundary mode {zone.boundary_mode!r}")
            if not all(math.isfinite(value) for value in (
                zone.center_x, zone.center_z, zone.size_x, zone.size_z, zone.boundary_width,
            )) or min(zone.size_x, zone.size_z) <= 0 or zone.boundary_width < 0:
                raise ValueError(f"invalid footprint for zone {zone.name!r}")
        self.zones = tuple(sorted(zones, key=lambda zone: (zone.size_x * zone.size_z, zone.name)))

    def query(self, x: np.ndarray | float, z: np.ndarray | float) -> ZoneBlend:
        x, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64))
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError("zone coordinates must be finite")
        remaining = np.ones_like(x)
        contributions: list[ZoneContribution] = []
        for zone in self.zones:
            inside = boundary_distance(zone, x, z) >= 0.0
            weight = remaining * inside
            if np.any(weight):
                contributions.append(ZoneContribution(zone, weight))
            remaining = remaining * ~inside
        return ZoneBlend(tuple(contributions), remaining)

    def zone_at(self, x: float, z: float) -> ZoneDefinition | None:
        blend = self.query(x, z)
        best, weight = None, float(blend.background)
        for contribution in blend.contributions:
            if float(contribution.weight) > weight:
                best, weight = contribution.zone, float(contribution.weight)
        return best
