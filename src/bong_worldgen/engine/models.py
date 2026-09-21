"""Data contracts shared by the procedural engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


NoiseKind = Literal["value", "ridge", "fbm", "warp"]
DEFAULT_RIVERBED_MATERIALS = ("dirt", "mud", "gravel", "sand", "clay")
SPAN_MIN_Y = -64
SPAN_MAX_SPANS = 4


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
    bed_materials: tuple[str, ...] = DEFAULT_RIVERBED_MATERIALS
    water_drop: float = 0.0
    bank_clearance: float = 0.5

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("river path needs at least two points")
        if (
            self.width <= 0
            or self.depth < 0
            or self.widening <= 0
            or self.water_drop < 0
            or self.bank_clearance < 0
        ):
            raise ValueError("river width/depth must be positive")
        if isinstance(self.bed_materials, str):
            raise ValueError("river bed_materials must be a sequence of names")
        materials = tuple(
            material.strip().lower().removeprefix("minecraft:").replace("-", "_")
            for material in self.bed_materials
        )
        if not materials or any(not material for material in materials):
            raise ValueError("river bed_materials must contain at least one named material")
        if len(set(materials)) != len(materials):
            raise ValueError("river bed_materials must not contain duplicates")
        object.__setattr__(self, "bed_materials", materials)


@dataclass(frozen=True)
class CaveNetwork:
    """沿手工路径约束的浅层 3D 噪声/SDF 洞穴网络。

    形状参考 FastNoiseLite 的 seeded 3D 噪声和 domain-warp 工作流；本项目
    保留 NumPy 实现，以便维持依赖轻量的 Python 生成管线。
    来源：https://github.com/Auburn/FastNoiseLite
    """

    name: str
    paths: tuple[tuple[Point, ...], ...]
    width: float = 2.5
    height: int = 4
    depth: float = 10.0
    noise_strength: float = 0.22
    sdf_threshold: float = 0.0
    roof_thickness: float = 3.0
    vertical_scale: float = 24.0
    worm_threshold: float = 0.55
    dead_end_strength: float = 1.15
    vertical_warp: float = 3.0
    domain_warp_scale: float = 180.0
    domain_warp_strength: float = 3.0
    branch_count: int = 8
    branch_segments: int = 5
    branch_length: float = 180.0
    branch_turn: float = 0.65
    chamber_count: int = 4
    chamber_radius: float = 7.0
    chamber_height: float = 5.0
    smooth_union: float = 0.8
    entrance_count: int = 1
    entrance_radius: float = 2.4
    chamber_centers: tuple[Point, ...] = ()
    entrance_points: tuple[Point, ...] = ()
    worm_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=150.0, octaves=2, gain=0.55)
    )
    vertical_warp_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=320.0, octaves=2, gain=0.55)
    )
    roughness: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="value", scale=72.0, amplitude=1.0, octaves=1)
    )
    # Connect this network's lowest and highest void in each column. Useful
    # for a single cave layer whose noisy walls must not leave vertical shards.
    fill_vertical_gaps: bool = False

    def __post_init__(self) -> None:
        if not self.paths or any(len(path) < 2 for path in self.paths):
            raise ValueError("cave network needs at least one path with two points")
        if (
            self.width <= 0
            or self.height < 2
            or self.depth <= 0
            or self.noise_strength < 0
            or self.roof_thickness < 1
            or self.vertical_scale <= 0
            or not 0.0 < self.worm_threshold <= 1.0
            or self.dead_end_strength < 0
            or self.vertical_warp < 0
            or self.domain_warp_scale <= 0
            or self.domain_warp_strength < 0
            or self.branch_count < 0
            or self.branch_segments < 1
            or self.branch_length < 0
            or self.branch_turn < 0
            or self.chamber_count < 0
            or self.chamber_radius <= 0
            or self.chamber_height <= 0
            or self.smooth_union < 0
            or self.entrance_count < 0
            or self.entrance_radius <= 0
        ):
            raise ValueError("cave width, height and depth must be positive")


@dataclass(frozen=True)
class UndergroundBlock:
    """由未来地下特征生成器输出的稀疏方块放置。"""

    x: int
    y: int
    z: int
    material: str


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
class Plateau:
    """A level elliptical shelf with a continuous transition at its perimeter."""

    center: Point
    radius_x: float
    radius_z: float
    height: float
    edge_width: float

    def __post_init__(self) -> None:
        if self.radius_x <= 0 or self.radius_z <= 0 or self.edge_width <= 0:
            raise ValueError("plateau radii and edge width must be positive")
        if not np.isfinite((self.center.x, self.center.z, self.radius_x,
                            self.radius_z, self.height, self.edge_width)).all():
            raise ValueError("plateau parameters must be finite")


@dataclass(frozen=True)
class FloatingIsland:
    """A detached ellipsoidal solid cap above the existing ground."""

    center: Point
    radius_x: float
    radius_z: float
    height: float = 270.0
    relief: float = 24.0
    thickness: float = 38.0

    def __post_init__(self) -> None:
        if min(self.radius_x, self.radius_z, self.thickness) <= 0 or self.relief < 0:
            raise ValueError("island radii/thickness must be positive and relief non-negative")


@dataclass(frozen=True)
class TerrainRecipe:
    """Project data only: the engine interprets this recipe without mutation."""

    name: str
    base_height: float = 64.0
    base_noise: tuple[NoiseLayer, ...] = ()
    basins: tuple[Basin, ...] = ()
    mountains: tuple[MountainRange, ...] = ()
    rivers: tuple[River, ...] = ()
    caves: tuple[CaveNetwork, ...] = ()
    sea_level: float = 62.0
    moisture_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(scale=1800.0, amplitude=1.0, octaves=3)
    )
    plateaus: tuple[Plateau, ...] = ()
    floating_islands: tuple[FloatingIsland, ...] = ()


@dataclass(frozen=True)
class Heightfield:
    """All scalar fields produced by one deterministic generation pass."""

    height: np.ndarray
    moisture: np.ndarray
    water_level: np.ndarray
    riverbed_id: np.ndarray | None = None
    riverbed_palette: tuple[str, ...] = ()
    solid_spans: np.ndarray | None = None
    underground_blocks: tuple[UndergroundBlock, ...] = ()
    cave_id: np.ndarray | None = None
    cave_palette: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.riverbed_id is None:
            object.__setattr__(
                self,
                "riverbed_id",
                np.full(self.height.shape, -1, dtype=np.int16),
            )
        if self.cave_id is None:
            object.__setattr__(self, "cave_id", np.zeros(self.height.shape, dtype=np.uint8))
        shapes = {
            self.height.shape,
            self.moisture.shape,
            self.water_level.shape,
            self.riverbed_id.shape,
            self.cave_id.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"heightfield layers must share a shape, got {sorted(shapes)}")
        if self.height.ndim != 2:
            raise ValueError("heightfield layers must be two-dimensional")
        for name in ("height", "moisture", "water_level"):
            value = getattr(self, name)
            if not np.isfinite(value).all():
                raise ValueError(f"heightfield layer {name} contains non-finite values")
        if not np.issubdtype(self.riverbed_id.dtype, np.integer):
            raise ValueError("heightfield layer riverbed_id must contain integer palette ids")
        if not np.issubdtype(self.cave_id.dtype, np.integer):
            raise ValueError("heightfield layer cave_id must contain integer palette ids")
        if np.any(self.cave_id < 0) or np.any(self.cave_id > 255):
            raise ValueError("heightfield layer cave_id must fit in uint8")
        cave_valid = (self.cave_id == 0) | (
            (self.cave_id > 0) & (self.cave_id <= len(self.cave_palette))
        )
        if not np.all(cave_valid):
            raise ValueError("cave_id contains an index outside cave_palette")
        if self.solid_spans is not None:
            expected_spans = (*self.height.shape, SPAN_MAX_SPANS, 2)
            if self.solid_spans.shape != expected_spans:
                raise ValueError(
                    f"solid_spans must have shape {expected_spans}, got {self.solid_spans.shape}"
                )
            if not np.issubdtype(self.solid_spans.dtype, np.integer):
                raise ValueError("solid_spans must contain integer world-Y bounds")
            valid = self.solid_spans[..., 0] <= self.solid_spans[..., 1]
            sentinel = self.solid_spans[..., 0] == 32767
            if not np.all(valid | sentinel):
                raise ValueError("solid_spans contains an inverted span")
        for block in self.underground_blocks:
            if not isinstance(block, UndergroundBlock):
                raise ValueError("underground_blocks must contain UndergroundBlock values")
            if not block.material.strip():
                raise ValueError("underground block material cannot be empty")
        if self.riverbed_palette:
            valid = (self.riverbed_id == -1) | (
                (self.riverbed_id >= 0) & (self.riverbed_id < len(self.riverbed_palette))
            )
        else:
            valid = self.riverbed_id == -1
        if not np.all(valid):
            raise ValueError("riverbed_id contains an index outside riverbed_palette")
