"""地表地形的类型化配置。

本模块只描述输入数据，不执行任何生成逻辑。世界级配方组合位于
``world_config.py``，地下配置位于 ``underground_config.py``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .constants import DEFAULT_RIVERBED_MATERIALS, MOUNTAIN_ROCK_PALETTE
from .validation import normalize_material_names


NoiseKind = Literal["value", "ridge", "fbm", "warp"]


@dataclass(frozen=True)
class Point:
    x: float
    z: float


@dataclass(frozen=True)
class NoiseLayer:
    """以世界坐标为单位的确定性噪声层。"""

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
class NaturalRelief:
    """不依赖地貌分区的全球随机崎岖场。

    每个字段都是世界坐标上的连续噪声参数；算法实现位于 ``relief.py``，
    这样配方可以调整起伏而不需要修改生成器逻辑。噪声组织方式参考：
    https://github.com/Auburn/FastNoiseLite
    """

    macro_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=2600.0, octaves=5, gain=0.56)
    )
    macro_amplitude: float = 28.0
    ridge_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=720.0, octaves=4, gain=0.57)
    )
    ridge_amplitude: float = 42.0
    ridge_power: float = 2.0
    detail_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=190.0, octaves=3, gain=0.54)
    )
    detail_amplitude: float = 10.0
    warp_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=1800.0, octaves=3, gain=0.55)
    )
    warp_strength: float = 520.0
    cliff_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=160.0, octaves=3, gain=0.56)
    )
    cliff_threshold: float = 0.04
    cliff_sharpness: float = 8.0
    cliff_amplitude: float = 24.0

    def __post_init__(self) -> None:
        if self.macro_amplitude < 0 or self.ridge_amplitude < 0 or self.detail_amplitude < 0:
            raise ValueError("relief amplitudes must be non-negative")
        if self.ridge_power <= 0:
            raise ValueError("relief ridge power must be positive")
        if self.warp_strength < 0 or self.cliff_amplitude < 0:
            raise ValueError("relief warp and cliff amplitude must be non-negative")
        if self.cliff_sharpness <= 0:
            raise ValueError("relief cliff sharpness must be positive")
        if not -1.0 <= self.cliff_threshold <= 1.0:
            raise ValueError("relief cliff threshold must be within [-1, 1]")


@dataclass(frozen=True)
class SpawnPlainSettings:
    """从自然地形中寻找并二次整形出生平原的配置。"""

    enabled: bool = True
    center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    search_extent_x: float = 700.0
    search_extent_z: float = 700.0
    search_resolution: float = 32.0
    candidate_window_radius: float = 160.0
    plain_radius_x: float = 560.0
    plain_radius_z: float = 560.0
    edge_blend: float = 140.0
    maximum_mean_slope: float = 0.18
    maximum_local_relief: float = 12.0
    minimum_elevation_above_sea: float = 3.0
    smoothing_strength: float = 0.72
    natural_variation: float = 1.8
    micro_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=220.0, octaves=3, gain=0.55)
    )

    def __post_init__(self) -> None:
        if (
            self.search_extent_x <= 0.0
            or self.search_extent_z <= 0.0
            or self.search_resolution <= 0.0
            or self.candidate_window_radius <= 0.0
            or self.plain_radius_x <= 0.0
            or self.plain_radius_z <= 0.0
            or self.edge_blend <= 0.0
            or self.edge_blend >= min(self.plain_radius_x, self.plain_radius_z)
            or self.maximum_mean_slope <= 0.0
            or self.maximum_local_relief <= 0.0
            or self.minimum_elevation_above_sea < 0.0
            or not 0.0 <= self.smoothing_strength <= 1.0
            or self.natural_variation < 0.0
        ):
            raise ValueError("spawn plain parameters are out of range")


@dataclass(frozen=True)
class TownSettings:
    """城镇生成的布局输入。

    这层只描述城镇规模、核心/外围密度、道路 palette 和结构资源目录。
    Weighted Eden/Agent Growth 位于 ``engine/town_growth.py``，方块发射位于
    ``engine/town.py``。出生平原只向城镇层提供锚点，不拥有布局算法。
    """

    enabled: bool = True
    center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    radius: float = 520.0
    core_radius: float = 180.0
    core_house_count: int = 22
    outer_house_count: int = 28
    outer_min_radius: float = 250.0
    outer_cluster_count: int = 5
    outer_cluster_spread: float = 92.0
    # 外围部落不沿用核心区的贴边间隙，避免形成第二圈城墙式建筑带。
    outer_building_gap_min: int = 7
    outer_building_gap_max: int = 15
    house_min_size: int = 7
    house_max_size: int = 11
    house_height: int = 4
    road_width: int = 3
    alley_width: int = 1
    attractor_count: int = 3
    main_road_count: int = 3
    growth_candidate_count: int = 48
    building_gap_min: int = 1
    building_gap_max: int = 4
    # 围墙矩形内真实核心建筑 footprint 的最低占地比例。达不到时，布局器会用
    # 小型模板继续填补空隙；外围建筑不会影响核心围墙的边界和利用率。
    core_wall_minimum_utilization: float = 0.80
    core_wall_infill_max_buildings: int = 0
    # 启用围墙填补时，建筑中心相对部落锚点的最大半径比例；限制链式生长
    # 漂移，保持墙内空间紧凑。
    core_growth_radius_ratio: float = 0.55
    maximum_slope: float = 0.24
    maximum_relief: float = 3.0
    house_material: str = "minecraft:stone_bricks"
    road_materials: tuple[str, ...] = ()
    road_material_weights: tuple[float, ...] = ()
    # 乡村道路参数：A* 在地形代价场上寻路，再用低频随机势场制造自然弯曲。
    road_wander: float = 0.30
    road_search_margin: float = 64.0
    road_slope_penalty: float = 8.0
    road_max_slope: float = 0.75
    road_cliff_penalty: float = 12.0
    road_smoothing_passes: int = 1
    road_path_step: int = 3
    # 跨水道路的最大连续跨度；设为 0 可强制所有道路绕开水体。
    bridge_max_span: int = 0
    bridge_deck_material: str = "minecraft:oak_planks"
    bridge_rail_material: str = "minecraft:oak_fence"
    bridge_support_material: str = "minecraft:cobblestone"
    bridge_clearance: int = 1
    bridge_approach_length: int = 12
    bridge_max_support_depth: int = 24
    bridge_cost_factor: float = 3.0
    house_schematic_directory: str | None = None
    tree_schematic_directory: str | None = None
    tree_count: int = 8
    tree_clearance: int = 4
    # 每栋建筑周围可供 NPC 刷新和防守的水平缓冲半径（方块）。
    npc_spawn_radius: int = 8
    # 建筑地基相对地形表面抬高的方块数；默认 1 表示建筑底层落在
    # 地表方块上方，避免模板的 Y=0 覆盖并隐藏地表。树木和道路不使用此偏移。
    structure_ground_offset: int = 1
    # 城门外道路使用的起始材质；城门/角塔可以独立下沉，避免把铺路层
    # 和普通房屋的地表锚点混在一起。
    gate_road_material: str = "minecraft:dirt"
    # 城门外主路比住宅巷道更宽。每座城门在此闭区间内稳定抽取一个宽度，
    # 同一 world seed 重复生成时不会改变。
    gate_road_min_width: int = 2
    gate_road_max_width: int = 5
    gate_structure_sink: int = 1
    tower_structure_sink: int = 1
    core_wall_enabled: bool = True
    core_wall_margin: int = 2

    def __post_init__(self) -> None:
        if (
            self.radius <= 0.0
            or self.core_radius <= 0.0
            or self.core_radius >= self.radius
            or self.core_house_count < 1
            or self.outer_house_count < 0
            or self.outer_min_radius <= self.core_radius
            or self.outer_min_radius >= self.radius
            or self.outer_cluster_count < 1
            or self.outer_cluster_spread <= 0.0
            or self.outer_building_gap_min < 0
            or self.outer_building_gap_max < self.outer_building_gap_min
            or self.house_min_size < 3
            or self.house_max_size < self.house_min_size
            or self.house_height < 2
            or self.road_width < 1
            or self.alley_width < 1
            or self.attractor_count < 1
            or self.main_road_count < 1
            or self.main_road_count > self.attractor_count
            or self.growth_candidate_count < 1
            or self.building_gap_min < 0
            or self.building_gap_max < self.building_gap_min
            or not 0.0 < self.core_wall_minimum_utilization <= 1.0
            or self.core_wall_infill_max_buildings < 0
            or not 0.0 < self.core_growth_radius_ratio <= 1.0
            or self.maximum_slope <= 0.0
            or self.maximum_relief < 0.0
            or not 0.0 <= self.road_wander <= 1.0
            or self.road_search_margin < 0.0
            or self.road_slope_penalty < 0.0
            or self.road_max_slope <= 0.0
            or self.road_cliff_penalty < 0.0
            or self.road_smoothing_passes < 0
            or self.road_path_step < 1
            or self.bridge_max_span < 0
            or self.bridge_clearance < 1
            or self.bridge_approach_length < 2
            or self.bridge_max_support_depth < 1
            or self.bridge_cost_factor < 1.0
            or (
                self.house_schematic_directory is not None
                and not self.house_schematic_directory.strip()
            )
            or (
                self.tree_schematic_directory is not None
                and not self.tree_schematic_directory.strip()
            )
            or self.tree_count < 0
            or self.tree_clearance < 0
            or not isinstance(self.structure_ground_offset, int)
            or self.structure_ground_offset < 0
            or self.gate_road_min_width < 1
            or self.gate_road_max_width < self.gate_road_min_width
            or not isinstance(self.gate_structure_sink, int)
            or self.gate_structure_sink < 0
            or not isinstance(self.tower_structure_sink, int)
            or self.tower_structure_sink < 0
            or not isinstance(self.core_wall_margin, int)
            or self.core_wall_margin < 0
            or self.npc_spawn_radius < 0
        ):
            raise ValueError("town parameters are out of range")
        for field_name in (
            "house_material", "gate_road_material", "bridge_deck_material",
            "bridge_rail_material", "bridge_support_material",
        ):
            material = normalize_material_names(
                (getattr(self, field_name),),
                f"town {field_name}",
            )[0]
            object.__setattr__(self, field_name, material)
        road_materials = normalize_material_names(
            self.road_materials or ("minecraft:stone_bricks",),
            "town road_materials",
        )
        road_weights = self.road_material_weights or (1.0,)
        if len(road_materials) != len(road_weights):
            raise ValueError("road material weights must match a non-empty palette")
        if any(weight < 0.0 for weight in road_weights) or sum(road_weights) <= 0.0:
            raise ValueError("road material weights must have a positive sum")
        object.__setattr__(self, "road_materials", road_materials)
        object.__setattr__(self, "road_material_weights", tuple(float(w) for w in road_weights))


@dataclass(frozen=True)
class StandaloneStructureSettings:
    """世界中独立刷新的大型结构配置。

    这层不参与城镇的 Weighted Eden 布局或围墙计算。生成器只会在干燥、
    平整且离城镇足够远的位置放置完整模板，并为 Server 输出同一套 NPC
    刷新/防守区域记录。
    """

    enabled: bool = True
    structure_directory: str | None = None
    grid_spacing: float = 2048.0
    maximum_slope: float = 0.35
    maximum_relief: float = 7.0
    exclusion_center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    exclusion_radius: float = 640.0
    npc_spawn_radius: int = 14
    structure_ground_offset: int = 1

    def __post_init__(self) -> None:
        if (
            self.grid_spacing <= 0.0
            or self.maximum_slope <= 0.0
            or self.maximum_relief < 0.0
            or self.exclusion_radius < 0.0
            or self.npc_spawn_radius < 0
            or not isinstance(self.structure_ground_offset, int)
            or self.structure_ground_offset < 0
            or (
                self.structure_directory is not None
                and not self.structure_directory.strip()
            )
        ):
            raise ValueError("standalone structure parameters are out of range")


@dataclass(frozen=True)
class HydraulicErosionSettings:
    """末端滴水侵蚀 refinement 的参数。

    每个水滴沿当前坡度移动，按速度、水量和坡度计算携沙能力，再在
    ``erosion_rate``/``deposition_rate`` 控制下下切或沉积。实现参考
    常见的 droplet hydraulic erosion 形式，未复制外部源码：
    https://github.com/Auburn/FastNoiseLite
    https://github.com/SebLague/Hydraulic-Erosion
    """

    enabled: bool = False
    droplet_count: int = 0
    max_steps: int = 48
    inertia: float = 0.30
    gravity: float = 4.0
    evaporation: float = 0.025
    initial_water: float = 1.0
    initial_speed: float = 1.0
    sediment_capacity: float = 3.0
    erosion_rate: float = 0.22
    deposition_rate: float = 0.16
    minimum_slope: float = 0.01
    max_erosion_per_step: float = 1.8
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.droplet_count < 0 or self.max_steps < 1:
            raise ValueError("hydraulic erosion counts must be non-negative and steps positive")
        if not 0.0 <= self.inertia <= 1.0:
            raise ValueError("hydraulic erosion inertia must be within [0, 1]")
        if self.gravity <= 0.0 or self.initial_water <= 0.0 or self.initial_speed <= 0.0:
            raise ValueError("hydraulic erosion gravity and initial state must be positive")
        if not 0.0 < self.evaporation < 1.0:
            raise ValueError("hydraulic erosion evaporation must be in (0, 1)")
        if self.sediment_capacity <= 0.0:
            raise ValueError("hydraulic erosion sediment capacity must be positive")
        if not 0.0 <= self.erosion_rate <= 1.0 or not 0.0 <= self.deposition_rate <= 1.0:
            raise ValueError("hydraulic erosion rates must be within [0, 1]")
        if self.minimum_slope < 0.0 or self.max_erosion_per_step < 0.0:
            raise ValueError("hydraulic erosion slope and erosion cap must be non-negative")


@dataclass(frozen=True)
class RidgedMultifractal:
    """山脉专用的多尺度尖脊场。

    ``ridge_gain`` 让粗尺度脊线控制后续细节出现的位置；``persistence``
    控制高频纹理衰减，``warp_strength`` 则打散规则的晶格方向。实现参考：
    https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h
    """

    scale: float = 260.0
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.52
    ridge_offset: float = 1.0
    ridge_gain: float = 2.0
    ridge_power: float = 2.0
    seed_offset: int = 0
    warp_scale: float = 920.0
    warp_strength: float = 90.0

    def __post_init__(self) -> None:
        if self.scale <= 0 or self.warp_scale <= 0:
            raise ValueError("ridged multifractal scales must be positive")
        if self.octaves < 1:
            raise ValueError("ridged multifractal octaves must be at least 1")
        if self.lacunarity <= 1.0:
            raise ValueError("ridged multifractal lacunarity must be greater than 1")
        if not 0.0 < self.persistence <= 1.0:
            raise ValueError("ridged multifractal persistence must be in (0, 1]")
        if self.ridge_offset <= 0 or self.ridge_gain < 0 or self.ridge_power <= 0:
            raise ValueError("ridged multifractal ridge parameters are out of range")
        if self.warp_strength < 0:
            raise ValueError("ridged multifractal warp strength must be non-negative")


@dataclass(frozen=True)
class MountainPeakSettings:
    """依附主山脊的峰值修饰配置。

    峰点不是独立的圆形山包，而是从脊柱候选中按 Ridged Multifractal、折线
    曲率和折点优先级选择，再用最小弧长间距做 Poisson 式筛选。``along_width``
    和 ``across_width`` 分别控制沿脊与横脊尺度，形成各向异性峰核。
    """

    enabled: bool = False
    count: int = 0
    candidate_spacing: float = 180.0
    minimum_spacing: float = 420.0
    amplitude: float = 36.0
    along_width: float = 180.0
    across_width: float = 64.0
    smoothness: float = 8.0
    ridge_weight: float = 0.62
    curvature_weight: float = 0.23
    junction_weight: float = 0.15
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("mountain peak count must be non-negative")
        if self.candidate_spacing <= 0.0 or self.minimum_spacing <= 0.0:
            raise ValueError("mountain peak spacings must be positive")
        if self.amplitude < 0.0 or self.along_width <= 0.0 or self.across_width <= 0.0:
            raise ValueError("mountain peak dimensions must be positive")
        if self.smoothness < 0.0:
            raise ValueError("mountain peak smoothness must be non-negative")
        weights = (self.ridge_weight, self.curvature_weight, self.junction_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("mountain peak selection weights must be non-negative")


@dataclass(frozen=True)
class MountainMaterialSettings:
    """群山专用的场驱动冰雪与裸岩材质配置。

    ``snowline_*`` 决定不规则雪线，坡度、暴露方向、气候寒冷度和冰川场
    再对积雪分数做连续修正。最终材质不是按固定 Y 层切片，而是由分数和
    坐标稳定的噪声共同选择。陡坡裸岩使用独立的坡度暴露场和噪声场，
    覆盖雪、冰分类，避免雪山变成整齐的单一材质。噪声组织方式参考
    FastNoiseLite 的 seeded 多尺度采样接口（本项目没有复制其实现）：
    https://github.com/Auburn/FastNoiseLite
    """

    enabled: bool = True
    snowline_base_elevation: float = 190.0
    snowline_large_amplitude: float = 24.0
    snowline_small_amplitude: float = 8.0
    snowline_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=900.0, octaves=3, gain=0.55)
    )
    snowline_detail_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="value", scale=190.0, octaves=1)
    )
    snowline_transition: float = 34.0
    snow_slope_full_angle: float = 20.0
    snow_slope_cutoff_angle: float = 45.0
    snow_slope_bonus: float = 0.20
    snow_slope_penalty: float = 0.30
    exposure_direction_degrees: float = 90.0
    exposure_strength: float = 0.16
    temperature_strength: float = 0.32
    glacier_bonus: float = 0.34
    minimum_mountain_weight: float = 0.12
    powder_threshold: float = 0.18
    snow_threshold: float = 0.34
    ice_threshold: float = 0.55
    packed_ice_threshold: float = 0.73
    blue_ice_threshold: float = 0.88
    boundary_noise_strength: float = 0.18
    material_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=120.0, octaves=3, gain=0.54)
    )
    # 坡度先归一到 [0, 1]（0° -> 0，90° -> 1），在该区间内逐渐增加裸岩概率。
    # 0.55/0.85 对应陡坡到悬崖，不使用固定高度，因此同一雪线两侧也会有
    # 不同的岩石暴露带。
    rock_exposure_start_slope: float = 0.55
    rock_exposure_full_slope: float = 0.85
    # 即使是悬崖也保留一部分雪/冰斑块，避免整个山面变成单一岩壁。
    rock_exposure_max_probability: float = 0.58
    rock_exposure_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="value", scale=58.0, octaves=1)
    )
    # 仅从这个子集选择裸岩变体；ID 仍由 MOUNTAIN_SURFACE_PALETTE 的稳定顺序
    # 编码，便于不同雪山 biome 通过配方选择不同基岩。
    rock_palette: tuple[str, ...] = MOUNTAIN_ROCK_PALETTE
    seed_offset: int = 0

    def __post_init__(self) -> None:
        scalar_values = (
            self.snowline_base_elevation,
            self.snowline_large_amplitude,
            self.snowline_small_amplitude,
            self.snowline_transition,
            self.snow_slope_full_angle,
            self.snow_slope_cutoff_angle,
            self.snow_slope_bonus,
            self.snow_slope_penalty,
            self.exposure_direction_degrees,
            self.exposure_strength,
            self.temperature_strength,
            self.glacier_bonus,
            self.minimum_mountain_weight,
            self.boundary_noise_strength,
            self.powder_threshold,
            self.snow_threshold,
            self.ice_threshold,
            self.packed_ice_threshold,
            self.blue_ice_threshold,
            self.rock_exposure_start_slope,
            self.rock_exposure_full_slope,
            self.rock_exposure_max_probability,
        )
        if (
            not all(math.isfinite(value) for value in scalar_values)
            or self.snowline_base_elevation < 0.0
            or self.snowline_large_amplitude < 0.0
            or self.snowline_small_amplitude < 0.0
            or self.snowline_transition <= 0.0
            or self.snow_slope_full_angle < 0.0
            or self.snow_slope_cutoff_angle <= self.snow_slope_full_angle
            or self.snow_slope_cutoff_angle > 90.0
            or self.snow_slope_bonus < 0.0
            or self.snow_slope_penalty < 0.0
            or self.exposure_strength < 0.0
            or self.temperature_strength < 0.0
            or self.glacier_bonus < 0.0
            or not 0.0 <= self.minimum_mountain_weight <= 1.0
            or not 0.0 <= self.boundary_noise_strength <= 1.0
            or not 0.0 <= self.rock_exposure_start_slope < self.rock_exposure_full_slope <= 1.0
            or not 0.0 < self.rock_exposure_max_probability <= 1.0
        ):
            raise ValueError("mountain material settings are out of range")
        thresholds = (
            self.powder_threshold,
            self.snow_threshold,
            self.ice_threshold,
            self.packed_ice_threshold,
            self.blue_ice_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("mountain material thresholds must be within [0, 1]")
        if any(previous >= current for previous, current in zip(thresholds, thresholds[1:])):
            raise ValueError("mountain material thresholds must be strictly increasing")
        allowed_rocks = set(MOUNTAIN_ROCK_PALETTE)
        normalized_rocks = tuple(
            material.strip().lower().removeprefix("minecraft:").replace("-", "_")
            for material in self.rock_palette
        )
        if not normalized_rocks or len(set(normalized_rocks)) != len(normalized_rocks):
            raise ValueError("mountain rock palette must contain unique materials")
        if any(f"minecraft:{material}" not in allowed_rocks for material in normalized_rocks):
            raise ValueError("mountain rock palette contains an unsupported Minecraft block")


@dataclass(frozen=True)
class MountainRange:
    """用山脉脊柱和有限宽度距离场描述的山脊系统。

    ``path`` 是宏观脊柱；生成器先用 domain warp 扭曲脊柱距离场，再让
    ``width`` 范围内的高度按 ``slope_power`` 从山脚升到脊线。沿脊柱的
    Ridged Multifractal 负责峰、鞍部、支山脊和岩石褶皱，宽度噪声负责
    不规则山脚。

    距离场与 domain warp 的组织方式参考公开资料，没有复制源码：
    https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
    https://github.com/Auburn/FastNoiseLite
    """

    path: tuple[Point, ...]
    width: float
    height: float
    spine_ridges: RidgedMultifractal = field(
        default_factory=lambda: RidgedMultifractal(
            scale=760.0,
            octaves=4,
            persistence=0.50,
            warp_scale=1500.0,
            warp_strength=70.0,
        )
    )
    flank_ridges: RidgedMultifractal = field(default_factory=RidgedMultifractal)
    flank_carving: float = 0.55
    rock_folds: RidgedMultifractal = field(
        default_factory=lambda: RidgedMultifractal(
            scale=86.0,
            octaves=4,
            lacunarity=2.15,
            persistence=0.46,
            ridge_gain=2.15,
            warp_scale=360.0,
            warp_strength=38.0,
        )
    )
    rock_fold_height: float = 0.0
    valley_depth: float = 0.0
    base_elevation: float | None = None
    summit_elevation: float | None = None
    slope_power: float = 1.35
    # 冠部宽度以方块为单位，负责让山脊顶部从单点变成圆钝但不平顶的区域。
    crest_width: float = 0.0
    # 仅在有限支持边缘混合回原地形，避免把山脚基准误变成环形墙体。
    edge_blend: float = 0.16
    spine_height_variation: float = 0.0
    # 沿脊柱方向平滑主峰，防止 Ridged Multifractal 的局部极值只占一个体素。
    spine_peak_width: float = 0.0
    spine_warp_strength: float = 0.0
    spine_warp_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=1100.0, octaves=3, gain=0.55)
    )
    width_variation: float = 0.0
    width_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=620.0, octaves=3, gain=0.55)
    )
    # 峰值必须依附主脊；关闭时保留只有脊柱 Ridged Multifractal 的旧行为。
    peaks: MountainPeakSettings = field(default_factory=MountainPeakSettings)

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("mountain path needs at least two points")
        if self.width <= 0 or self.height < 0 or self.valley_depth < 0:
            raise ValueError("mountain width must be positive and heights non-negative")
        if not 0.0 <= self.flank_carving <= 1.0:
            raise ValueError("mountain flank_carving must be in [0, 1]")
        if self.rock_fold_height < 0:
            raise ValueError("mountain rock_fold_height must be non-negative")
        if self.slope_power <= 0:
            raise ValueError("mountain slope_power must be positive")
        if self.crest_width < 0 or self.crest_width >= self.width:
            raise ValueError("mountain crest_width must be in [0, width)")
        if not 0.0 < self.edge_blend <= 1.0:
            raise ValueError("mountain edge_blend must be in (0, 1]")
        if (
            self.spine_height_variation < 0
            or self.spine_peak_width < 0
            or self.spine_warp_strength < 0
        ):
            raise ValueError("mountain spine variations must be non-negative")
        if not 0.0 <= self.width_variation < 1.0:
            raise ValueError("mountain width_variation must be in [0, 1)")
        if self.spine_height_variation > self.height and self.base_elevation is None:
            raise ValueError("mountain spine_height_variation cannot exceed relative height")
        if (self.base_elevation is None) != (self.summit_elevation is None):
            raise ValueError(
                "mountain base_elevation and summit_elevation must be provided together"
            )
        if self.base_elevation is not None and self.summit_elevation is not None:
            if not math.isfinite(self.base_elevation) or not math.isfinite(
                self.summit_elevation
            ):
                raise ValueError("mountain absolute elevations must be finite")
            if self.summit_elevation <= self.base_elevation:
                raise ValueError("mountain summit_elevation must exceed base_elevation")
            if self.spine_height_variation >= (
                self.summit_elevation - self.base_elevation
            ):
                raise ValueError(
                    "mountain spine_height_variation must be smaller than absolute relief"
                )


@dataclass(frozen=True)
class ValleySystem:
    """固定世界域上的汇流侵蚀山谷系统。

    ``domain_center``/``domain_extent_*`` 定义一次性水文规划域，避免每个
    raster tile 各算各的汇流方向。``minimum_drainage_area`` 决定支沟何时
    开始切蚀，``reference_drainage_area`` 则控制主谷达到多大尺度。
    ``uplift_strength``/``uplift_reference`` 把山脉 Uplift 场接入 Stream Power，
    ``coupling_iterations`` 控制下切后重新路由的反馈轮数。

    填洼、D8 汇流和 Stream Power 的合同参考以下公开实现，未复制源码：

    - https://github.com/Deltares/pyflwdir/blob/main/pyflwdir/dem.py
    - https://github.com/r-barnes/richdem/blob/master/include/richdem/depressions/
      Barnes2014.hpp
    - https://github.com/landlab/landlab/blob/master/src/landlab/components/
      stream_power/stream_power.py
    """

    name: str
    domain_center: Point
    domain_extent_x: float
    domain_extent_z: float
    planning_resolution: float = 16.0
    rainfall_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(
            kind="fbm",
            scale=1100.0,
            octaves=3,
            gain=0.55,
            seed_offset=2_701,
        )
    )
    rainfall_variation: float = 0.28
    minimum_drainage_area: float = 12_000.0
    reference_drainage_area: float = 600_000.0
    minimum_slope: float = 0.002
    reference_slope: float = 0.22
    area_exponent: float = 0.5
    slope_exponent: float = 1.0
    erosion_strength: float = 0.82
    maximum_depth: float = 58.0
    minimum_width: float = 10.0
    maximum_width: float = 72.0
    width_exponent: float = 0.38
    cross_section_power: float = 1.8
    minimum_height_above_sea_level: float = 4.0
    # 构造抬升对侵蚀效率的耦合：抬升越强，坡面可用的 Stream Power 越高。
    # 参考 Landlab 的 stream-power 形式；这里额外保留独立的 Uplift 场，
    # 让山脉骨架与水文侵蚀之间的关系可配置、可查询。
    uplift_strength: float = 0.45
    uplift_reference: float = 128.0
    # 分水岭修饰把相邻 D8 流域的边界抬成窄脊；宽度和高度都使用世界单位。
    watershed_divide_strength: float = 14.0
    watershed_divide_width: float = 48.0
    # 分水岭只能回填已切出的谷槽一小部分，避免修饰层抹平 Stream Power 地貌。
    watershed_divide_valley_fill: float = 0.35
    # 每轮先按当前地形路由，再应用 Stream Power 下切；下一轮会重新路由，
    # 让山脉抬升与山谷切割形成有限的地貌反馈。
    coupling_iterations: int = 2

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("valley system name cannot be empty")
        if self.domain_extent_x <= 0 or self.domain_extent_z <= 0:
            raise ValueError("valley domain extents must be positive")
        if self.planning_resolution <= 0:
            raise ValueError("valley planning_resolution must be positive")
        if min(self.domain_extent_x, self.domain_extent_z) * 2.0 < self.planning_resolution * 2.0:
            raise ValueError("valley planning domain needs at least three samples per axis")
        if not 0.0 <= self.rainfall_variation < 1.0:
            raise ValueError("valley rainfall_variation must be in [0, 1)")
        if self.minimum_drainage_area <= 0:
            raise ValueError("valley minimum_drainage_area must be positive")
        if self.reference_drainage_area <= self.minimum_drainage_area:
            raise ValueError(
                "valley reference_drainage_area must exceed minimum_drainage_area"
            )
        if self.minimum_slope <= 0 or self.reference_slope <= 0:
            raise ValueError("valley slope references must be positive")
        if self.area_exponent <= 0 or self.slope_exponent <= 0:
            raise ValueError("valley stream-power exponents must be positive")
        if self.erosion_strength < 0 or self.maximum_depth < 0:
            raise ValueError("valley erosion strength and depth must be non-negative")
        if self.minimum_width <= 0 or self.maximum_width < self.minimum_width:
            raise ValueError("valley widths are out of range")
        if self.width_exponent <= 0 or self.cross_section_power <= 0:
            raise ValueError("valley width and cross-section exponents must be positive")
        if self.minimum_height_above_sea_level < 0:
            raise ValueError("valley sea-level clearance must be non-negative")
        if self.uplift_strength < 0 or self.uplift_reference <= 0:
            raise ValueError(
                "valley uplift coupling must be non-negative with a positive reference"
            )
        if self.watershed_divide_strength < 0 or self.watershed_divide_width < 0:
            raise ValueError("watershed divide strength and width must be non-negative")
        if not 0.0 <= self.watershed_divide_valley_fill <= 1.0:
            raise ValueError("watershed divide valley fill must be within [0, 1]")
        if self.coupling_iterations < 1:
            raise ValueError("valley coupling_iterations must be at least one")


@dataclass(frozen=True)
class Canyon:
    """沿 seeded 折线切割的大峡谷。

    ``paths`` 为空时，拓扑模块会从 ``seed_center``/``seed_extent`` 生成
    确定性的弯曲折线。路径只使用 XZ 坐标和 seed，不读取地表高度。
    """

    name: str
    paths: tuple[tuple[Point, ...], ...] = ()
    seed_center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    seed_extent: float = 3200.0
    path_count: int = 1
    path_segments: int = 18
    path_length: float = 9000.0
    path_turn: float = 0.78
    width: float = 180.0
    floor_width: float = 28.0
    depth: float = 72.0
    bank_width: float = 48.0
    wall_power: float = 1.8
    width_variation: float = 0.20
    depth_variation: float = 0.16
    terrace_count: int = 6
    terrace_strength: float = 0.18
    domain_warp_scale: float = 950.0
    domain_warp_strength: float = 260.0
    wall_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(kind="fbm", scale=260.0, octaves=3, gain=0.55)
    )

    def __post_init__(self) -> None:
        if any(len(path) < 2 for path in self.paths):
            raise ValueError("explicit canyon paths need at least two points")
        if (
            self.seed_extent <= 0
            or self.path_count < 1
            or self.path_segments < 2
            or self.path_length <= 0
            or not 0.0 <= self.path_turn <= 1.0
            or self.width <= 0
            or self.floor_width <= 0
            or self.floor_width >= self.width
            or self.depth <= 0
            or self.bank_width < 0
            or self.wall_power <= 0
            or not 0.0 <= self.width_variation <= 1.0
            or not 0.0 <= self.depth_variation <= 1.0
            or self.terrace_count < 0
            or not 0.0 <= self.terrace_strength <= 1.0
            or self.domain_warp_scale <= 0
            or self.domain_warp_strength < 0
        ):
            raise ValueError("canyon parameters are out of range")


@dataclass(frozen=True)
class SnowMountainResourceSpec:
    """雪山地表生成物的资源刷新配置。

    ``material`` 只是 BlueMap/Anvil 使用的真实视觉方块，Server 应使用
    ``resource_id``、固定来源 ``snow_mountain`` 和 ``rarity`` 刷新实际产物。
    生成器只在群山材质、寒带权重和最终覆盖层同时满足条件时按世界坐标
    进行确定性概率判定。字段名和资源 ID 都是显式配置，不把雪山内容写死
    在生成算法中。
    """

    name: str = "雪山生成物"
    material: str = "minecraft:dead_bush"
    resource_id: str = "snow_mountain:dead_bush"
    rarity: Literal["少", "中", "多"] = "少"
    probability: float = 0.018
    min_mountain_weight: float = 0.22
    min_cold_weight: float = 0.45
    seed_offset: int = 0

    def __post_init__(self) -> None:
        materials = normalize_material_names((self.material,), "snow mountain resource material")
        if (
            not self.name.strip()
            or not self.resource_id.strip()
            or self.rarity not in ("少", "中", "多")
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
            or not math.isfinite(self.min_mountain_weight)
            or not 0.0 <= self.min_mountain_weight <= 1.0
            or not math.isfinite(self.min_cold_weight)
            or not 0.0 <= self.min_cold_weight <= 1.0
        ):
            raise ValueError("snow mountain resource parameters are out of range")
        object.__setattr__(self, "material", materials[0])


@dataclass(frozen=True)
class GlacialSystem:
    """寒带冰川地貌系统的可复现配置。

    系统先从积雪与质量平衡场选择冰斗，再沿坡面规划冰川流路，随后用
    可变宽度的 U 型截面 carve，最后叠加终碛和顺冰流方向的鼓丘沉积。
    算法结构参考：

    - https://github.com/oargudo/glaciers
    - https://github.com/TheJanusStream/symbios-ground
    """

    name: str
    seed_center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    seed_extent: float = 2400.0
    cirque_count: int = 4
    cirque_radius: float = 110.0
    cirque_depth: float = 26.0
    cirque_min_elevation: float = 92.0
    cirque_elevation_blend: float = 24.0
    valley_count: int = 4
    valley_segments: int = 16
    valley_length: float = 1800.0
    valley_turn: float = 0.72
    valley_source_width: float = 30.0
    valley_floor_width: float = 10.0
    valley_depth: float = 38.0
    valley_width_growth: float = 2.2
    valley_wall_power: float = 4.0
    moraine_height: float = 8.0
    moraine_width: float = 46.0
    moraine_length: float = 150.0
    drumlin_count: int = 18
    drumlin_length: float = 150.0
    drumlin_width: float = 34.0
    drumlin_height: float = 5.0
    drumlin_max_elevation: float = 42.0
    surface_patch_count: int = 48
    surface_patch_radius: float = 48.0
    surface_material_weights: tuple[float, ...] = (0.24, 0.14, 0.31, 0.20, 0.11, 0.0)
    freeze_thaw_iterations: int = 5
    freeze_thaw_strength: float = 0.18
    talus_slope_threshold: float = 1.8
    surface_protrusion_probability: float = 0.12
    mass_balance_enabled: bool = True
    # 风向积雪是独立的可见覆盖修饰：正向响应增加积雪层，负向响应
    # 在陡坡上削薄积雪。阈值对应法线点积，不是固定世界坐标线。
    wind_snow_enabled: bool = True
    wind_snow_accumulation_threshold: float = 0.30
    wind_snow_erosion_threshold: float = 0.50
    wind_snow_max_extra_layers: int = 1
    wind_snow_erosion_retention_scale: float = 0.70
    # 角度表示风/朝阳坡指向的世界方向：0 度为 +X，90 度为 +Z。
    prevailing_wind_angle_degrees: float = 90.0
    sun_facing_angle_degrees: float = 90.0
    # 平衡线高程相对 sea_level；过渡宽度控制积累区到消融区的渐变。
    equilibrium_line_elevation: float = 88.0
    equilibrium_transition: float = 64.0
    snowfall_rate: float = 1.0
    snowfall_variability: float = 0.35
    snowfall_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(
            kind="fbm",
            scale=420.0,
            octaves=3,
            gain=0.55,
            seed_offset=1_907,
        )
    )
    windward_accumulation_strength: float = 0.55
    leeward_drift_strength: float = 0.24
    mass_balance_slope_scale: float = 0.22
    ablation_rate: float = 0.85
    minimum_ablation: float = 0.08
    low_elevation_ablation_strength: float = 0.72
    solar_ablation_strength: float = 0.32
    transition_ablation_strength: float = 0.55
    # 流路规划在完整冰川域的低分辨率网格上执行。
    # 长度均使用世界方块单位。
    flow_planning_resolution: float = 32.0
    flow_source_separation: float = 220.0
    flow_inertia: float = 0.68
    flow_meander_strength: float = 0.22
    flow_direction_samples: int = 9
    flow_merge_distance: float = 72.0
    flow_uphill_tolerance: float = 0.0
    flow_min_length: float = 0.0
    flow_termination_balance: float = -0.08
    permafrost_enabled: bool = True
    permafrost_min_cold_weight: float = 0.18
    permafrost_full_cold_weight: float = 0.32
    permafrost_coverage: float = 0.64
    permafrost_elevation_start: float = 4.0
    permafrost_elevation_range: float = 90.0
    permafrost_patch_scale: float = 120.0
    permafrost_material_scale: float = 42.0
    permafrost_material_weights: tuple[float, ...] = (
        0.18,
        0.14,
        0.13,
        0.10,
        0.17,
        0.12,
        0.09,
        0.07,
    )
    climate_fade_scale: float = 180.0
    climate_fade_full_weight: float = 0.20
    cover_snow_weight: float = 0.24
    cover_ice_weight: float = 0.52
    cover_blue_ice_weight: float = 0.78
    cover_powder_depth: int = 1
    cover_snow_depth: int = 1
    cover_ice_depth: int = 1
    cover_blue_ice_depth: int = 1
    cover_elevation_start: float = 18.0
    cover_elevation_range: float = 110.0
    # 冰面横向裂隙独立于地下 fracture_id；它们沿冰川流线的法向生成。
    crevasses_enabled: bool = True
    crevasse_spacing: float = 150.0
    crevasse_width: float = 1.0
    crevasse_probability: float = 0.55
    crevasse_jitter: float = 0.35
    crevasse_meander: float = 0.55
    crevasse_blue_ice_probability: float = 0.18
    crevasse_min_cold_weight: float = 0.08
    meltwater_enabled: bool = True
    meltwater_ice_thickness: float = 12.0
    meltwater_rate: float = 0.001
    meltwater_max_depth: float = 3.0
    meltwater_depth_exponent: float = 0.40
    meltwater_min_depth: float = 0.15
    meltwater_min_cold_weight: float = 0.08
    # 雪山地表资源由独立模块按最终雪冰材质和世界坐标概率生成。
    snow_mountain_resources: tuple[SnowMountainResourceSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("glacial system name must not be empty")
        mass_balance_values = (
            self.wind_snow_accumulation_threshold,
            self.wind_snow_erosion_threshold,
            self.wind_snow_erosion_retention_scale,
            self.prevailing_wind_angle_degrees,
            self.sun_facing_angle_degrees,
            self.equilibrium_line_elevation,
            self.equilibrium_transition,
            self.snowfall_rate,
            self.snowfall_variability,
            self.windward_accumulation_strength,
            self.leeward_drift_strength,
            self.mass_balance_slope_scale,
            self.ablation_rate,
            self.minimum_ablation,
            self.low_elevation_ablation_strength,
            self.solar_ablation_strength,
            self.transition_ablation_strength,
            self.flow_planning_resolution,
            self.flow_source_separation,
            self.flow_inertia,
            self.flow_meander_strength,
            self.flow_merge_distance,
            self.flow_uphill_tolerance,
            self.flow_min_length,
            self.flow_termination_balance,
        )
        if (
            self.seed_extent <= 0
            or self.cirque_count < 0
            or self.cirque_radius <= 0
            or self.cirque_depth <= 0
            or self.cirque_min_elevation < 0
            or self.cirque_elevation_blend <= 0
            or self.valley_count < 0
            or self.valley_segments < 2
            or self.valley_length <= 0
            or not 0.0 <= self.valley_turn <= 1.0
            or self.valley_source_width <= 0
            or self.valley_floor_width <= 0
            or self.valley_floor_width >= self.valley_source_width
            or self.valley_depth <= 0
            or self.valley_width_growth < 0
            or self.valley_wall_power <= 1.0
            or self.moraine_height < 0
            or self.moraine_width <= 0
            or self.moraine_length <= 0
            or self.drumlin_count < 0
            or self.drumlin_length <= 0
            or self.drumlin_width <= 0
            or self.drumlin_height < 0
            or self.drumlin_max_elevation < 0
            or self.surface_patch_count < 0
            or self.surface_patch_radius <= 0
            or self.freeze_thaw_iterations < 0
            or not 0.0 <= self.freeze_thaw_strength <= 1.0
            or self.talus_slope_threshold <= 0
            or not 0.0 <= self.surface_protrusion_probability <= 1.0
            or not all(math.isfinite(value) for value in mass_balance_values)
            or not 0.0 <= self.wind_snow_accumulation_threshold < 1.0
            or not 0.0 <= self.wind_snow_erosion_threshold < 1.0
            or self.wind_snow_max_extra_layers < 0
            or self.wind_snow_max_extra_layers > 255
            or not 0.0 <= self.wind_snow_erosion_retention_scale <= 1.0
            or self.equilibrium_line_elevation < 0.0
            or self.equilibrium_transition <= 0.0
            or self.snowfall_rate < 0.0
            or not 0.0 <= self.snowfall_variability <= 1.0
            or self.windward_accumulation_strength < 0.0
            or self.leeward_drift_strength < 0.0
            or self.mass_balance_slope_scale <= 0.0
            or self.ablation_rate < 0.0
            or self.minimum_ablation < 0.0
            or self.low_elevation_ablation_strength < 0.0
            or self.solar_ablation_strength < 0.0
            or self.transition_ablation_strength < 0.0
            or self.flow_planning_resolution <= 0.0
            or self.flow_source_separation <= 0.0
            or not 0.0 <= self.flow_inertia < 1.0
            or not 0.0 <= self.flow_meander_strength <= 1.0
            or self.flow_direction_samples < 3
            or self.flow_direction_samples % 2 == 0
            or self.flow_merge_distance < 0.0
            or self.flow_uphill_tolerance < 0.0
            or self.flow_min_length < 0.0
            or self.flow_min_length > self.valley_length
            or self.flow_min_length >= self.seed_extent
            or self.permafrost_min_cold_weight < 0.0
            or self.permafrost_min_cold_weight > 1.0
            or self.permafrost_full_cold_weight <= self.permafrost_min_cold_weight
            or self.permafrost_full_cold_weight > 1.0
            or self.permafrost_coverage < 0.0
            or self.permafrost_coverage > 1.0
            or self.permafrost_elevation_start < 0.0
            or self.permafrost_elevation_range <= 0.0
            or self.permafrost_patch_scale <= 0.0
            or self.permafrost_material_scale <= 0.0
            or self.climate_fade_scale <= 0
            or not 0.0 < self.climate_fade_full_weight <= 1.0
            or not 0.0 <= self.cover_snow_weight <= 1.0
            or not 0.0 <= self.cover_ice_weight <= 1.0
            or not 0.0 <= self.cover_blue_ice_weight <= 1.0
            or self.cover_snow_weight > self.cover_ice_weight
            or self.cover_ice_weight > self.cover_blue_ice_weight
            or self.cover_powder_depth < 0
            or self.cover_snow_depth < 0
            or self.cover_ice_depth < 0
            or self.cover_blue_ice_depth < 0
            or self.cover_elevation_start < 0
            or self.cover_elevation_range <= 0
            or self.crevasse_spacing <= 0.0
            or self.crevasse_width <= 0.0
            or not 0.0 <= self.crevasse_probability <= 1.0
            or not 0.0 <= self.crevasse_jitter <= 1.0
            or not 0.0 <= self.crevasse_meander <= 1.0
            or not 0.0 <= self.crevasse_blue_ice_probability <= 1.0
            or not 0.0 <= self.crevasse_min_cold_weight <= 1.0
            or self.meltwater_ice_thickness <= 0
            or self.meltwater_rate <= 0
            or self.meltwater_max_depth <= 0
            or self.meltwater_depth_exponent <= 0
            or self.meltwater_min_depth <= 0
            or self.meltwater_min_depth > self.meltwater_max_depth
            or not 0.0 <= self.meltwater_min_cold_weight <= 1.0
        ):
            raise ValueError("glacial system parameters are out of range")
        if len(self.surface_material_weights) != 6:
            raise ValueError("surface_material_weights must contain six probabilities")
        if any(weight < 0 for weight in self.surface_material_weights):
            raise ValueError("surface_material_weights cannot contain negative values")
        if sum(self.surface_material_weights) <= 0:
            raise ValueError("surface_material_weights must have a positive sum")
        if len(self.permafrost_material_weights) != 8:
            raise ValueError("permafrost_material_weights must contain eight probabilities")
        if any(weight < 0 for weight in self.permafrost_material_weights):
            raise ValueError("permafrost_material_weights cannot contain negative values")
        if sum(self.permafrost_material_weights) <= 0:
            raise ValueError("permafrost_material_weights must have a positive sum")


@dataclass(frozen=True)
class River:
    """沿折线刻蚀的河流；河宽可以从源头向出口逐步放大。"""

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
        object.__setattr__(
            self,
            "bed_materials",
            normalize_material_names(self.bed_materials, "river bed_materials"),
        )


@dataclass(frozen=True)
class Basin:
    """用于湖泊和低地盆地的椭圆形洼地。"""

    center: Point
    radius_x: float
    radius_z: float
    depth: float

    def __post_init__(self) -> None:
        if self.radius_x <= 0 or self.radius_z <= 0 or self.depth < 0:
            raise ValueError("basin radii must be positive and depth non-negative")


__all__ = [
    "Basin",
    "Canyon",
    "GlacialSystem",
    "MountainMaterialSettings",
    "MountainRange",
    "NoiseKind",
    "NoiseLayer",
    "Point",
    "River",
    "SnowMountainResourceSpec",
    "ValleySystem",
    "TownSettings",
]
