"""一份完整程序化世界的输入配方。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .climate import ClimateWorldBounds, GlobalClimatePlan
from .terrain_config import (
    Basin,
    Canyon,
    GlacialSystem,
    HydraulicErosionSettings,
    MountainMaterialSettings,
    MountainRange,
    NoiseLayer,
    NaturalRelief,
    River,
    SpawnPlainSettings,
    TownSettings,
    StandaloneStructureSettings,
    ValleySystem,
)
from .underground_config import CaveNetwork, SolidOreSpec, UndergroundRiverNetwork


@dataclass(frozen=True)
class TerrainRecipe:
    """只存放项目输入数据；引擎解释配方但不修改配方对象。

    ``climate`` 描述归一化的全球气候带；``climate_world_bounds`` 由数据层
    显式注入后，生成器才会把它映射到世界坐标。
    """

    name: str
    base_height: float = 64.0
    base_noise: tuple[NoiseLayer, ...] = ()
    natural_relief: NaturalRelief | None = None
    spawn_plain: SpawnPlainSettings | None = None
    basins: tuple[Basin, ...] = ()
    mountains: tuple[MountainRange, ...] = ()
    valleys: tuple[ValleySystem, ...] = ()
    canyons: tuple[Canyon, ...] = ()
    rivers: tuple[River, ...] = ()
    solid_ores: tuple[SolidOreSpec, ...] = ()
    caves: tuple[CaveNetwork, ...] = ()
    underground_rivers: tuple[UndergroundRiverNetwork, ...] = ()
    climate: GlobalClimatePlan = field(default_factory=GlobalClimatePlan)
    climate_world_bounds: ClimateWorldBounds | None = None
    sea_level: float = 62.0
    moisture_noise: NoiseLayer = field(
        default_factory=lambda: NoiseLayer(scale=1800.0, amplitude=1.0, octaves=3)
    )
    glaciers: tuple[GlacialSystem, ...] = ()
    hydraulic_erosion: HydraulicErosionSettings | None = None
    mountain_materials: MountainMaterialSettings | None = None
    # 追加在末尾，避免改变已有 TerrainRecipe 位置参数的含义。
    town: TownSettings | None = None
    # 不依附城镇的大型单体结构，例如独立生存屋与废弃城堡。
    standalone_structures: StandaloneStructureSettings | None = None


__all__ = ["TerrainRecipe"]
