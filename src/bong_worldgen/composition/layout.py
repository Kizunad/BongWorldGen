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
BOUNDARY_SCALE = {"soft": 1.0, "semi_hard": 0.75, "hard": 0.5}


def boundary_alpha(zone: ZoneDefinition, distance: np.ndarray) -> np.ndarray:
    """A C2 transition centered on the footprint; zero width is a true step.

    Hard terrain uses half the authored transition width, semi-hard uses 3/4.
    All nonzero widths have zero endpoint derivatives to avoid cut-off walls.
    """

    width = zone.boundary_width * BOUNDARY_SCALE[zone.boundary_mode]
    if width == 0:
        return (distance >= 0).astype(np.float64)
    t = np.clip(0.5 + distance / width, 0.0, 1.0)
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


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


def _influence_bounds(zone: ZoneDefinition) -> tuple[float, float, float, float]:
    """Conservative world bounds, including the outward half of the blend band."""

    rx, rz = zone.size_x * 0.5, zone.size_z * 0.5
    if zone.shape == "circular":
        rx = rz = min(rx, rz)
    elif zone.shape == "rotated_rift":
        angle = math.pi / 6.0
        # The rotated rectangle encloses the ellipse, including its endpoints.
        rx, rz = rx * math.cos(angle) + rz * math.sin(angle), rx * math.sin(angle) + rz * math.cos(angle)
    margin = zone.boundary_width * BOUNDARY_SCALE[zone.boundary_mode] * 0.5
    return (
        math.nextafter(zone.center_x - rx - margin, -math.inf),
        math.nextafter(zone.center_x + rx + margin, math.inf),
        math.nextafter(zone.center_z - rz - margin, -math.inf),
        math.nextafter(zone.center_z + rz + margin, math.inf),
    )


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

    def dominant(self) -> tuple[np.ndarray, np.ndarray]:
        """Return contribution indices and weights; -1 means background.

        Indices address this blend's contributions, not an output palette.
        Background retains ties, followed by the index's overlay order.
        """

        indices = np.full(self.background.shape, -1, dtype=np.int32)
        weights = self.background.copy()
        for index, part in enumerate(self.contributions):
            stronger = part.weight > weights
            indices[stronger] = index
            weights = np.maximum(weights, part.weight)
        return indices, weights


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
        self._bounds = tuple(_influence_bounds(zone) for zone in self.zones)

    def query(self, x: np.ndarray | float, z: np.ndarray | float) -> ZoneBlend:
        x, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64))
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError("zone coordinates must be finite")
        remaining = np.ones_like(x)
        contributions: list[ZoneContribution] = []
        if x.size == 0:
            return ZoneBlend((), remaining)
        min_x, max_x, min_z, max_z = x.min(), x.max(), z.min(), z.max()
        for zone, (left, right, top, bottom) in zip(self.zones, self._bounds):
            if max_x < left or min_x > right or max_z < top or min_z > bottom:
                continue
            active = (x >= left) & (x <= right) & (z >= top) & (z <= bottom)
            if not np.any(active):
                continue
            if np.all(active):
                alpha = boundary_alpha(zone, boundary_distance(zone, x, z))
            else:
                alpha = np.zeros_like(x)
                alpha[active] = boundary_alpha(zone, boundary_distance(zone, x[active], z[active]))
            weight = remaining * alpha
            if np.any(weight):
                contributions.append(ZoneContribution(zone, weight))
            remaining = remaining * (1.0 - alpha)
        return ZoneBlend(tuple(contributions), remaining)

    def zone_at(self, x: float, z: float) -> ZoneDefinition | None:
        blend = self.query(x, z)
        indices, _ = blend.dominant()
        index = int(indices)
        return None if index < 0 else blend.contributions[index].zone
