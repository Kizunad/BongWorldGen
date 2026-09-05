"""洞穴、地下河、矿物与地下输出记录的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .constants import CAVE_RARITY_MULTIPLIERS
from .terrain_config import NoiseLayer, Point
from .validation import normalize_material_names, validate_rarities


@dataclass(frozen=True)
class SolidOreSpec:
    """实心石层中的普通矿脉配置。"""

    material: str
    rarity: Literal["少", "中", "多"] = "中"
    cluster_count: int = 12
    min_depth: float = 4.0
    max_depth: float = 48.0
    vein_length: int = 5

    def __post_init__(self) -> None:
        materials = normalize_material_names((self.material,), "solid ore material")
        if (
            self.rarity not in CAVE_RARITY_MULTIPLIERS
            or self.cluster_count < 0
            or self.min_depth <= 0
            or self.max_depth < self.min_depth
            or self.vein_length < 1
        ):
            raise ValueError("solid ore parameters are out of range")
        object.__setattr__(self, "material", materials[0])


@dataclass(frozen=True)
class CaveNetwork:
    """可由 seed 生成的浅层 3D 噪声/SDF 洞穴网络。

    ``paths`` 仅用于兼容手工锁定路线的配方；留空时由 seed 生成随机游走
    洞道。3D 噪声和 domain-warp 结构参考：
    https://github.com/Auburn/FastNoiseLite
    """

    name: str
    paths: tuple[tuple[Point, ...], ...] = ()
    seed_center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    seed_extent: float = 512.0
    seed_path_count: int = 5
    seed_path_segments: int = 14
    seed_path_length: float = 420.0
    seed_path_turn: float = 0.8
    width: float = 2.5
    height: int = 4
    depth: float = 56.0
    noise_strength: float = 0.22
    sdf_threshold: float = 0.0
    roof_thickness: float = 3.0
    vertical_scale: float = 24.0
    worm_threshold: float = 0.55
    dead_end_strength: float = 1.15
    vertical_warp: float = 9.0
    vertical_tail_fraction: float = 0.05
    vertical_rise_ratio: float = 0.90
    vertical_drop_ratio: float = 0.25
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
    placeholder_count: int = 16
    placeholder_materials: tuple[str, ...] = (
        "minecraft:coal_ore",
        "minecraft:iron_ore",
        "minecraft:copper_ore",
        "minecraft:glow_lichen",
    )
    placeholder_density: str = "中"
    placeholder_rarities: tuple[str, ...] = ("多", "中", "少", "多")
    worm_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=150.0, octaves=2, gain=0.55)
    )
    vertical_warp_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=140.0, octaves=3, gain=0.55)
    )
    roughness: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="value", scale=72.0, amplitude=1.0, octaves=1)
    )

    def __post_init__(self) -> None:
        if any(len(path) < 2 for path in self.paths):
            raise ValueError("explicit cave paths need at least two points")
        if (
            self.seed_extent <= 0
            or self.seed_path_count < 1
            or self.seed_path_segments < 2
            or self.seed_path_length <= 0
            or not 0.0 <= self.seed_path_turn <= 1.0
            or self.width <= 0
            or self.height < 2
            or self.depth <= 0
            or self.noise_strength < 0
            or self.roof_thickness < 1
            or self.vertical_scale <= 0
            or not 0.0 < self.worm_threshold <= 1.0
            or self.dead_end_strength < 0
            or self.vertical_warp < 0
            or not 0.0 <= self.vertical_tail_fraction < 0.5
            or not 0.0 <= self.vertical_rise_ratio <= 1.0
            or not 0.0 <= self.vertical_drop_ratio <= 1.0
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
            or self.placeholder_count < 0
            or self.placeholder_density not in CAVE_RARITY_MULTIPLIERS
        ):
            raise ValueError("cave width, height and depth must be positive")
        if isinstance(self.placeholder_materials, str):
            raise ValueError("cave placeholder_materials must be a sequence of names")
        materials = normalize_material_names(
            self.placeholder_materials,
            "cave placeholder_materials",
            allow_empty=True,
        )
        object.__setattr__(self, "placeholder_materials", materials)
        rarities = validate_rarities(
            materials,
            self.placeholder_rarities,
            "cave placeholder_rarities",
        )
        object.__setattr__(self, "placeholder_rarities", rarities)


@dataclass(frozen=True)
class UndergroundBlock:
    """地下天然内容的刷新点。

    ``material`` 是 BlueMap/Anvil 的真实视觉占位方块；Server 应优先使用
    ``resource_id``、``source`` 和 ``rarity`` 决定真实产物。
    """

    x: int
    y: int
    z: int
    material: str
    resource_id: str = ""
    source: Literal[
        "unknown",
        "cave",
        "underground_river",
        "fracture",
        "solid_ore",
        "snow_mountain",
    ] = (
        "unknown"
    )
    rarity: Literal["少", "中", "多"] | None = None

    def __post_init__(self) -> None:
        if not self.material.strip():
            raise ValueError("underground block material cannot be empty")
        if not self.resource_id.strip() and self.resource_id != "":
            raise ValueError("underground block resource_id cannot be whitespace")
        if self.source not in (
            "unknown",
            "cave",
            "underground_river",
            "fracture",
            "solid_ore",
            "snow_mountain",
        ):
            raise ValueError("underground block source is unknown")
        if self.rarity is not None and self.rarity not in CAVE_RARITY_MULTIPLIERS:
            raise ValueError("underground block rarity must be 少, 中, or 多")


@dataclass(frozen=True)
class UndergroundWaterBlock:
    """地下河/湖的真实水方块及其语义类型。"""

    x: int
    y: int
    z: int
    kind: Literal["river", "lake"]
    flowing: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("river", "lake"):
            raise ValueError("underground water kind must be river or lake")


@dataclass(frozen=True)
class UndergroundRiverNetwork:
    """独立于普通洞穴的地下河网与地下湖配方。

    入口、出口、代价场与 conduit 合并概念参考 pyKasso；本项目使用独立的
    轻量 NumPy 实现，不复制其 GPL-3.0 代码：
    https://github.com/randlab/pyKasso
    """

    name: str
    seed_center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    seed_extent: float = 512.0
    path_count: int = 2
    path_segments: int = 18
    path_length: float = 640.0
    path_turn: float = 0.72
    source_depth: float = 42.0
    outlet_depth: float = 58.0
    width: float = 2.5
    height: int = 4
    water_depth: int = 2
    branch_count: int = 6
    branch_segments: int = 7
    branch_length: float = 220.0
    branch_turn: float = 0.8
    lake_count: int = 3
    lake_radius: float = 12.0
    lake_depth: int = 2
    river_chance: float = 0.9
    inlet_count: int = 0
    outlet_count: int = 1
    inlet_margin: float = 0.12
    outlet_margin: float = 0.18
    cost_noise_scale: float = 90.0
    cost_noise_strength: float = 0.32
    fracture_bias: float = 0.35
    fracture_isoline_scale: float = 0.0
    fracture_isoline_width: float = 0.16
    fracture_width: float = 1.2
    fracture_depth: float = 48.0
    fracture_height: int = 28
    network_iterations: int = 3
    conduit_cost: float = 0.35
    junction_radius: float = 2.0
    resource_density: str = "中"
    ore_cluster_count: int = 12
    ore_materials: tuple[str, ...] = (
        "minecraft:coal_ore",
        "minecraft:iron_ore",
        "minecraft:copper_ore",
    )
    ore_rarities: tuple[str, ...] = ("多", "中", "中")
    plant_cluster_count: int = 10
    plant_materials: tuple[str, ...] = (
        "minecraft:glow_lichen",
        "minecraft:moss_block",
        "minecraft:spore_blossom",
    )
    plant_rarities: tuple[str, ...] = ("多", "中", "少")

    def __post_init__(self) -> None:
        if (
            self.seed_extent <= 0
            or self.path_count < 1
            or self.path_segments < 2
            or self.path_length <= 0
            or not 0.0 <= self.path_turn <= 1.0
            or self.source_depth <= 0
            or self.outlet_depth <= 0
            or self.outlet_depth < self.source_depth
            or self.width <= 0
            or self.height < 2
            or self.water_depth < 1
            or self.water_depth > self.height
            or self.branch_count < 0
            or self.branch_segments < 1
            or self.branch_length < 0
            or not 0.0 <= self.branch_turn <= 1.0
            or self.lake_count < 0
            or self.lake_radius <= 0
            or self.lake_depth < 1
            or self.lake_depth > self.height
            or not 0.0 <= self.river_chance <= 1.0
            or self.inlet_count < 0
            or self.outlet_count < 1
            or not 0.0 <= self.inlet_margin < 0.5
            or not 0.0 <= self.outlet_margin < 0.5
            or self.cost_noise_scale <= 0
            or not 0.0 <= self.cost_noise_strength < 1.0
            or not 0.0 <= self.fracture_bias < 1.0
            or self.fracture_isoline_scale < 0
            or not 0.0 < self.fracture_isoline_width <= 1.0
            or self.fracture_width <= 0
            or self.fracture_depth <= 0
            or self.fracture_height < 2
            or self.network_iterations < 1
            or not 0.0 < self.conduit_cost <= 1.0
            or self.junction_radius <= 0
            or self.resource_density not in CAVE_RARITY_MULTIPLIERS
            or self.ore_cluster_count < 0
            or self.plant_cluster_count < 0
        ):
            raise ValueError(
                "underground river parameters are out of range; "
                "outlet_depth must be at least source_depth"
            )
        for field_name, values in (
            ("ore_materials", self.ore_materials),
            ("plant_materials", self.plant_materials),
        ):
            normalized = normalize_material_names(
                values,
                f"underground river {field_name}",
                allow_empty=True,
            )
            object.__setattr__(self, field_name, normalized)
        for field_name, materials, rarities in (
            ("ore", self.ore_materials, self.ore_rarities),
            ("plant", self.plant_materials, self.plant_rarities),
        ):
            validate_rarities(
                materials,
                rarities,
                f"underground river {field_name}_rarities",
            )


__all__ = [
    "CaveNetwork",
    "SolidOreSpec",
    "UndergroundBlock",
    "UndergroundRiverNetwork",
    "UndergroundWaterBlock",
]
