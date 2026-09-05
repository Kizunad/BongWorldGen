"""寒带冰川地貌子系统。

生成顺序是：冰斗 → 冰川谷 → 终碛/鼓丘。这里的实现是独立 NumPy 代码，
只参考公开的冰川模拟和热力侵蚀项目，不复制外部源码：

- https://github.com/oargudo/glaciers
- https://github.com/TheJanusStream/symbios-ground
"""

from .erosion import apply_freeze_thaw, glacial_exposure_mask, talus_surface_mask
from .crevasses import (
    GLACIAL_CREVASSE_PALETTE,
    GlacierValleyField,
    sample_glacial_crevasses,
    sample_glacier_valley_field,
)
from .generator import (
    GLACIAL_COVER_PALETTE,
    GLACIAL_LANDFORM_PALETTE,
    GLACIAL_SURFACE_PALETTE,
    apply_glaciers,
    apply_surface_protrusions,
    glacial_landform_ids,
    glacial_surface_layers,
    glacial_surface_materials,
)
from .hydrology import GLACIAL_WATER_PALETTE, GlacialWaterField, apply_glacial_meltwater
from .mass_balance import GlacialMassBalanceField, sample_glacial_mass_balance
from .wind_snow import WindSnowField, sample_wind_snow_field, surface_wind_normal_alignment
from .permafrost import PERMAFROST_PALETTE, permafrost_surface_materials
from .topology import CirqueSeed, GlacialFlowPlan, plan_glacial_flow
from .surface_resources import generate_snow_mountain_resource_blocks

__all__ = [
    "CirqueSeed",
    "GlacialFlowPlan",
    "GLACIAL_SURFACE_PALETTE",
    "GLACIAL_CREVASSE_PALETTE",
    "GLACIAL_COVER_PALETTE",
    "GLACIAL_LANDFORM_PALETTE",
    "apply_freeze_thaw",
    "apply_glaciers",
    "apply_surface_protrusions",
    "glacial_exposure_mask",
    "glacial_landform_ids",
    "glacial_surface_materials",
    "glacial_surface_layers",
    "talus_surface_mask",
    "plan_glacial_flow",
    "GLACIAL_WATER_PALETTE",
    "GlacialWaterField",
    "apply_glacial_meltwater",
    "GlacialMassBalanceField",
    "WindSnowField",
    "sample_glacial_mass_balance",
    "sample_wind_snow_field",
    "surface_wind_normal_alignment",
    "PERMAFROST_PALETTE",
    "permafrost_surface_materials",
    "GlacierValleyField",
    "sample_glacier_valley_field",
    "sample_glacial_crevasses",
    "generate_snow_mountain_resource_blocks",
]
