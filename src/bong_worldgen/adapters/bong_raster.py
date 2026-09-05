"""把通用标量场转换为 Bong raster 层数组。

只有本模块了解 Bong 的层名称和 palette id；程序化引擎仍可供非 Minecraft
消费者复用。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np

from ..data.wilderness import classify_wilderness, wilderness_palette_manifest
from ..engine.generated_world import Heightfield
from ..engine.settlement_points import (
    SettlementInterestPoint,
    SettlementSpawnArea,
    settlement_interest_point_manifest,
    settlement_spawn_area_manifest,
)
from ..engine.underground_config import UndergroundBlock, UndergroundWaterBlock
from ..engine.glaciers import PERMAFROST_PALETTE


BASE_SURFACE_PALETTE = ("stone", "coarse_dirt", "gravel", "grass_block", "sand")
RIVERBED_NONE_ID = 255
UNDERGROUND_MATERIAL_PALETTE = (
    "minecraft:coal_ore",
    "minecraft:copper_ore",
    "minecraft:glow_lichen",
    "minecraft:iron_ore",
    "minecraft:moss_block",
    "minecraft:spore_blossom",
    "minecraft:dead_bush",
)
RESOURCE_SOURCE_IDS = {
    "unknown": 0,
    "cave": 1,
    "underground_river": 2,
    "fracture": 3,
    "solid_ore": 4,
    "snow_mountain": 5,
}
RESOURCE_RARITY_IDS = {None: 0, "少": 1, "中": 2, "多": 3}


def _resource_identity(block: UndergroundBlock) -> tuple[str, str, str | None]:
    """返回逻辑产物身份；旧记录没有 resource_id 时走 legacy 命名。"""

    material = block.material.strip().lower().removeprefix("minecraft:").replace("-", "_")
    return block.resource_id or f"legacy:{material}", block.source, block.rarity


def _resource_catalog(blocks: tuple[UndergroundBlock, ...]) -> tuple[dict[str, object], ...]:
    catalog: dict[str, dict[str, object]] = {}
    for block in blocks:
        resource_id, source, rarity = _resource_identity(block)
        catalog.setdefault(
            resource_id,
            {
                "resource_id": resource_id,
                "material": block.material,
                "source": source,
                "rarity": rarity,
            },
        )
    return tuple(catalog[key] for key in sorted(catalog))


@dataclass(frozen=True)
class BongTile:
    height: np.ndarray
    surface_id: np.ndarray
    subsurface_id: np.ndarray
    water_level: np.ndarray
    biome_id: np.ndarray
    feature_mask: np.ndarray
    boundary_weight: np.ndarray
    wilderness_id: np.ndarray
    uplift: np.ndarray | None = None
    glacial_landform_id: np.ndarray | None = None
    glacial_landform_palette: tuple[str, ...] = ()
    glacial_water_id: np.ndarray | None = None
    glacial_water_palette: tuple[str, ...] = ()
    glacial_crevasse_id: np.ndarray | None = None
    glacial_crevasse_palette: tuple[str, ...] = ()
    glacial_discharge: np.ndarray | None = None
    valley_depth: np.ndarray | None = None
    valley_flow_accumulation: np.ndarray | None = None
    valley_stream_power: np.ndarray | None = None
    watershed_divide: np.ndarray | None = None
    hydraulic_erosion: np.ndarray | None = None
    hydraulic_deposition: np.ndarray | None = None
    snow_accumulation: np.ndarray | None = None
    glacial_mass_balance: np.ndarray | None = None
    wind_snow_alignment: np.ndarray | None = None
    wind_snow_response: np.ndarray | None = None
    riverbed_id: np.ndarray | None = None
    surface_palette: tuple[str, ...] = BASE_SURFACE_PALETTE
    riverbed_palette: tuple[str, ...] = ()
    solid_spans: np.ndarray | None = None
    cave_id: np.ndarray | None = None
    cave_palette: tuple[str, ...] = ()
    fracture_id: np.ndarray | None = None
    fracture_palette: tuple[str, ...] = ()
    surface_material_id: np.ndarray | None = None
    surface_material_palette: tuple[str, ...] = ()
    mountain_material_id: np.ndarray | None = None
    mountain_material_palette: tuple[str, ...] = ()
    mountain_weight: np.ndarray | None = None
    mountain_snowline: np.ndarray | None = None
    mountain_material_score: np.ndarray | None = None
    mountain_slope_angle: np.ndarray | None = None
    mountain_exposure: np.ndarray | None = None
    mountain_rock_exposure: np.ndarray | None = None
    surface_visible_id: np.ndarray | None = None
    surface_visible_palette: tuple[str, ...] = ()
    permafrost_id: np.ndarray | None = None
    permafrost_palette: tuple[str, ...] = ()
    surface_cover_layers: np.ndarray | None = None
    surface_cover_palette: tuple[str, ...] = ()
    climate_id: np.ndarray | None = None
    climate_palette: tuple[str, ...] = ()
    climate_transition_id: np.ndarray | None = None
    climate_transition_palette: tuple[str, ...] = ()
    climate_transition_weight: np.ndarray | None = None
    climate_surface_id: np.ndarray | None = None
    climate_surface_palette: tuple[str, ...] = ()
    underground_water_blocks: tuple[UndergroundWaterBlock, ...] = ()
    underground_blocks: tuple[UndergroundBlock, ...] = ()
    settlement_spawn_areas: tuple[SettlementSpawnArea, ...] = ()
    settlement_interest_points: tuple[SettlementInterestPoint, ...] = ()

    def __post_init__(self) -> None:
        shape = self.height.shape
        for name in (
            "surface_id",
            "subsurface_id",
            "water_level",
            "biome_id",
            "feature_mask",
            "boundary_weight",
            "wilderness_id",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Bong tile layer {name} shape mismatch")
        if self.riverbed_id is None:
            object.__setattr__(
                self,
                "riverbed_id",
                np.full(shape, RIVERBED_NONE_ID, dtype=np.uint8),
            )
        elif self.riverbed_id.shape != shape:
            raise ValueError("Bong tile layer riverbed_id shape mismatch")
        if self.uplift is None:
            object.__setattr__(self, "uplift", np.zeros(shape, dtype=np.float32))
        elif self.uplift.shape != shape:
            raise ValueError("Bong tile layer uplift shape mismatch")
        if not np.issubdtype(self.uplift.dtype, np.floating):
            raise ValueError("Bong tile uplift must contain floats")
        if np.any(self.uplift < 0.0) or not np.isfinite(self.uplift).all():
            raise ValueError("Bong tile uplift must be finite and non-negative")
        if self.glacial_landform_id is None:
            object.__setattr__(self, "glacial_landform_id", np.zeros(shape, dtype=np.uint8))
        elif self.glacial_landform_id.shape != shape:
            raise ValueError("Bong tile layer glacial_landform_id shape mismatch")
        if not np.issubdtype(self.glacial_landform_id.dtype, np.integer):
            raise ValueError("Bong tile glacial_landform_id must contain integer palette ids")
        if np.any(self.glacial_landform_id > len(self.glacial_landform_palette)):
            raise ValueError("Bong tile glacial_landform_id contains an unknown palette id")
        if self.glacial_water_id is None:
            object.__setattr__(self, "glacial_water_id", np.zeros(shape, dtype=np.uint8))
        elif self.glacial_water_id.shape != shape:
            raise ValueError("Bong tile layer glacial_water_id shape mismatch")
        if self.glacial_discharge is None:
            object.__setattr__(self, "glacial_discharge", np.zeros(shape, dtype=np.float32))
        elif self.glacial_discharge.shape != shape:
            raise ValueError("Bong tile layer glacial_discharge shape mismatch")
        if not np.issubdtype(self.glacial_water_id.dtype, np.integer):
            raise ValueError("Bong tile glacial_water_id must contain integer palette ids")
        if not np.issubdtype(self.glacial_discharge.dtype, np.floating):
            raise ValueError("Bong tile glacial_discharge must contain floats")
        if np.any(self.glacial_water_id > len(self.glacial_water_palette)):
            raise ValueError("Bong tile glacial_water_id contains an unknown palette id")
        if self.glacial_crevasse_id is None:
            object.__setattr__(self, "glacial_crevasse_id", np.zeros(shape, dtype=np.uint8))
        elif self.glacial_crevasse_id.shape != shape:
            raise ValueError("Bong tile glacial_crevasse_id shape mismatch")
        if not np.issubdtype(self.glacial_crevasse_id.dtype, np.integer):
            raise ValueError("Bong tile glacial_crevasse_id must contain integer palette ids")
        if np.any(self.glacial_crevasse_id > len(self.glacial_crevasse_palette)):
            raise ValueError("Bong tile glacial_crevasse_id contains an unknown palette id")
        if np.any(self.glacial_discharge < 0.0) or not np.isfinite(self.glacial_discharge).all():
            raise ValueError("Bong tile glacial_discharge must be finite and non-negative")
        for name in (
            "valley_depth",
            "valley_flow_accumulation",
            "valley_stream_power",
            "watershed_divide",
            "hydraulic_erosion",
            "hydraulic_deposition",
        ):
            values = getattr(self, name)
            if values is None:
                values = np.zeros(shape, dtype=np.float32)
                object.__setattr__(self, name, values)
            elif values.shape != shape:
                raise ValueError(f"Bong tile layer {name} shape mismatch")
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError(f"Bong tile {name} must contain floats")
            if np.any(values < 0.0) or not np.isfinite(values).all():
                raise ValueError(f"Bong tile {name} must be finite and non-negative")
        if self.snow_accumulation is None:
            object.__setattr__(self, "snow_accumulation", np.zeros(shape, dtype=np.float32))
        elif self.snow_accumulation.shape != shape:
            raise ValueError("Bong tile layer snow_accumulation shape mismatch")
        if self.glacial_mass_balance is None:
            object.__setattr__(self, "glacial_mass_balance", np.zeros(shape, dtype=np.float32))
        elif self.glacial_mass_balance.shape != shape:
            raise ValueError("Bong tile layer glacial_mass_balance shape mismatch")
        if self.wind_snow_alignment is None:
            object.__setattr__(self, "wind_snow_alignment", np.zeros(shape, dtype=np.float32))
        elif self.wind_snow_alignment.shape != shape:
            raise ValueError("Bong tile layer wind_snow_alignment shape mismatch")
        if self.wind_snow_response is None:
            object.__setattr__(self, "wind_snow_response", np.zeros(shape, dtype=np.float32))
        elif self.wind_snow_response.shape != shape:
            raise ValueError("Bong tile layer wind_snow_response shape mismatch")
        if not np.issubdtype(self.snow_accumulation.dtype, np.floating):
            raise ValueError("Bong tile snow_accumulation must contain floats")
        if not np.issubdtype(self.glacial_mass_balance.dtype, np.floating):
            raise ValueError("Bong tile glacial_mass_balance must contain floats")
        if not np.issubdtype(self.wind_snow_alignment.dtype, np.floating):
            raise ValueError("Bong tile wind_snow_alignment must contain floats")
        if not np.issubdtype(self.wind_snow_response.dtype, np.floating):
            raise ValueError("Bong tile wind_snow_response must contain floats")
        if np.any(self.snow_accumulation < 0.0) or not np.isfinite(
            self.snow_accumulation
        ).all():
            raise ValueError("Bong tile snow_accumulation must be finite and non-negative")
        if not np.isfinite(self.glacial_mass_balance).all():
            raise ValueError("Bong tile glacial_mass_balance must be finite")
        if not np.isfinite(self.wind_snow_alignment).all() or not np.isfinite(
            self.wind_snow_response
        ).all():
            raise ValueError("Bong tile wind-snow fields must be finite")
        if np.any((self.wind_snow_alignment < -1.0) | (self.wind_snow_alignment > 1.0)):
            raise ValueError("Bong tile wind_snow_alignment must be within [-1, 1]")
        if self.cave_id is None:
            object.__setattr__(self, "cave_id", np.zeros(shape, dtype=np.uint8))
        elif self.cave_id.shape != shape:
            raise ValueError("Bong tile cave_id shape mismatch")
        if not np.issubdtype(self.cave_id.dtype, np.integer):
            raise ValueError("Bong tile cave_id must contain integer palette ids")
        if np.any(self.cave_id > len(self.cave_palette)):
            raise ValueError("Bong tile cave_id contains an unknown palette id")
        if self.fracture_id is None:
            object.__setattr__(self, "fracture_id", np.zeros(shape, dtype=np.uint8))
        elif self.fracture_id.shape != shape:
            raise ValueError("Bong tile fracture_id shape mismatch")
        if not np.issubdtype(self.fracture_id.dtype, np.integer):
            raise ValueError("Bong tile fracture_id must contain integer palette ids")
        if np.any(self.fracture_id > len(self.fracture_palette)):
            raise ValueError("Bong tile fracture_id contains an unknown palette id")
        if self.surface_material_id is None:
            object.__setattr__(self, "surface_material_id", np.zeros(shape, dtype=np.uint8))
        elif self.surface_material_id.shape != shape:
            raise ValueError("Bong tile surface_material_id shape mismatch")
        if not np.issubdtype(self.surface_material_id.dtype, np.integer):
            raise ValueError("Bong tile surface_material_id must contain integer palette ids")
        if np.any(self.surface_material_id > len(self.surface_material_palette)):
            raise ValueError("Bong tile surface_material_id contains an unknown palette id")
        if self.mountain_material_id is None:
            object.__setattr__(self, "mountain_material_id", np.zeros(shape, dtype=np.uint8))
        elif self.mountain_material_id.shape != shape:
            raise ValueError("Bong tile mountain_material_id shape mismatch")
        if not np.issubdtype(self.mountain_material_id.dtype, np.integer):
            raise ValueError("Bong tile mountain_material_id must contain integer palette ids")
        if np.any(self.mountain_material_id > len(self.mountain_material_palette)):
            raise ValueError("Bong tile mountain_material_id contains an unknown palette id")
        for name in (
            "mountain_weight",
            "mountain_snowline",
            "mountain_material_score",
            "mountain_slope_angle",
            "mountain_exposure",
            "mountain_rock_exposure",
        ):
            values = getattr(self, name)
            if values is None:
                values = np.zeros(shape, dtype=np.float32)
                object.__setattr__(self, name, values)
            elif values.shape != shape:
                raise ValueError(f"Bong tile layer {name} shape mismatch")
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError(f"Bong tile {name} must contain floats")
            if not np.isfinite(values).all():
                raise ValueError(f"Bong tile {name} must be finite")
        if np.any((self.mountain_weight < 0.0) | (self.mountain_weight > 1.0)):
            raise ValueError("Bong tile mountain_weight must be within [0, 1]")
        if np.any((self.mountain_material_score < 0.0) | (self.mountain_material_score > 1.0)):
            raise ValueError("Bong tile mountain_material_score must be within [0, 1]")
        if np.any((self.mountain_exposure < 0.0) | (self.mountain_exposure > 1.0)):
            raise ValueError("Bong tile mountain_exposure must be within [0, 1]")
        if np.any(
            (self.mountain_rock_exposure < 0.0) | (self.mountain_rock_exposure > 1.0)
        ):
            raise ValueError("Bong tile mountain_rock_exposure must be within [0, 1]")
        if self.surface_visible_id is None:
            object.__setattr__(self, "surface_visible_id", np.zeros(shape, dtype=np.uint8))
        elif self.surface_visible_id.shape != shape:
            raise ValueError("Bong tile surface_visible_id shape mismatch")
        if not np.issubdtype(self.surface_visible_id.dtype, np.integer):
            raise ValueError("Bong tile surface_visible_id must contain integer palette ids")
        if np.any(self.surface_visible_id > len(self.surface_visible_palette)):
            raise ValueError("Bong tile surface_visible_id contains an unknown palette id")
        if self.permafrost_id is None:
            object.__setattr__(self, "permafrost_id", np.zeros(shape, dtype=np.uint8))
        elif self.permafrost_id.shape != shape:
            raise ValueError("Bong tile permafrost_id shape mismatch")
        if not np.issubdtype(self.permafrost_id.dtype, np.integer):
            raise ValueError("Bong tile permafrost_id must contain integer palette ids")
        if np.any(self.permafrost_id > len(self.permafrost_palette)):
            raise ValueError("Bong tile permafrost_id contains an unknown palette id")
        if self.surface_cover_layers is None:
            object.__setattr__(
                self,
                "surface_cover_layers",
                np.zeros((4, *shape), dtype=np.uint8),
            )
        elif self.surface_cover_layers.shape != (4, *shape):
            raise ValueError("Bong tile surface_cover_layers must have shape (4, height, width)")
        if not np.issubdtype(self.surface_cover_layers.dtype, np.integer):
            raise ValueError("Bong tile surface_cover_layers must contain integers")
        if np.any(self.surface_cover_layers < 0) or np.any(self.surface_cover_layers > 255):
            raise ValueError("Bong tile surface_cover_layers values must fit in uint8")
        if self.surface_cover_palette and len(self.surface_cover_palette) != 4:
            raise ValueError("Bong tile surface_cover_palette must contain four materials")
        if self.climate_id is None:
            object.__setattr__(self, "climate_id", np.zeros(shape, dtype=np.uint8))
        elif self.climate_id.shape != shape:
            raise ValueError("Bong tile climate_id shape mismatch")
        if self.climate_transition_id is None:
            object.__setattr__(self, "climate_transition_id", np.zeros(shape, dtype=np.uint8))
        elif self.climate_transition_id.shape != shape:
            raise ValueError("Bong tile climate_transition_id shape mismatch")
        if self.climate_transition_weight is None:
            object.__setattr__(
                self,
                "climate_transition_weight",
                np.zeros(shape, dtype=np.float32),
            )
        elif self.climate_transition_weight.shape != shape:
            raise ValueError("Bong tile climate_transition_weight shape mismatch")
        if self.climate_surface_id is None:
            object.__setattr__(self, "climate_surface_id", np.zeros(shape, dtype=np.uint8))
        elif self.climate_surface_id.shape != shape:
            raise ValueError("Bong tile climate_surface_id shape mismatch")
        for name, values, palette in (
            ("climate_id", self.climate_id, self.climate_palette),
            ("climate_transition_id", self.climate_transition_id, self.climate_transition_palette),
            ("climate_surface_id", self.climate_surface_id, self.climate_surface_palette),
        ):
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Bong tile {name} must contain integer palette ids")
            if name in ("climate_id", "climate_surface_id"):
                invalid = np.any((values != 0) & (values > len(palette)))
            elif palette:
                invalid = np.any(values >= len(palette))
            else:
                invalid = np.any(values != 0)
            if invalid:
                raise ValueError(f"Bong tile {name} contains an unknown palette id")
        if not np.isfinite(self.climate_transition_weight).all():
            raise ValueError("Bong tile climate_transition_weight contains non-finite values")
        if np.any((self.climate_transition_weight < 0.0) | (self.climate_transition_weight > 1.0)):
            raise ValueError("Bong tile climate_transition_weight must be within [0, 1]")
        if self.solid_spans is not None:
            expected = (*shape, 4, 2)
            if self.solid_spans.shape != expected:
                raise ValueError(f"Bong tile solid_spans must have shape {expected}")
        for block in self.underground_water_blocks:
            if not isinstance(block, UndergroundWaterBlock):
                raise ValueError(
                    "Bong tile underground_water_blocks must contain UndergroundWaterBlock values"
                )
        for block in self.underground_blocks:
            if not isinstance(block, UndergroundBlock):
                raise ValueError(
                    "Bong tile underground_blocks must contain UndergroundBlock values"
                )
        for area in self.settlement_spawn_areas:
            if not isinstance(area, SettlementSpawnArea):
                raise ValueError(
                    "Bong tile settlement_spawn_areas must contain SettlementSpawnArea values"
                )
        for point in self.settlement_interest_points:
            if not isinstance(point, SettlementInterestPoint):
                raise ValueError(
                    "Bong tile settlement_interest_points must contain SettlementInterestPoint values"
                )


def to_bong_tile(
    field: Heightfield,
    *,
    sea_level: float,
    grass_id: int = 3,
    stone_id: int = 0,
    gravel_id: int = 2,
    # 这个独立适配器在下方声明两项 palette。
    # (0=plains, 1=river); keep the default id aligned with that contract.
    river_biome_id: int = 1,
    land_biome_id: int = 0,
) -> BongTile:
    """为生成的高度场应用明确的最小 palette 策略。"""

    wet = field.water_level >= 0.0
    high = field.height >= sea_level + 52.0
    surface_palette = list(BASE_SURFACE_PALETTE)
    for material in field.riverbed_palette:
        if material not in surface_palette:
            surface_palette.append(material)
    surface_id = np.where(wet, gravel_id, np.where(high, stone_id, grass_id)).astype(np.uint8)
    if field.riverbed_palette:
        for material_index, material in enumerate(field.riverbed_palette):
            surface_id[field.riverbed_id == material_index] = surface_palette.index(material)
    if field.climate_surface_palette:
        desert_index = (
            field.climate_surface_palette.index("desert") + 1
            if "desert" in field.climate_surface_palette
            else None
        )
        if desert_index is not None:
            desert = (field.climate_surface_id == desert_index) & ~wet & ~high
            surface_id[desert] = surface_palette.index("sand")
    # 旧格式没有独立覆盖层时，才把 surface_material_id 兼容映射到顶部方块。
    # 新格式保留这个 ID 作为查询语义，真实冰雪由 surface_cover_layers 叠加。
    if field.surface_material_palette:
        for material_index, material in enumerate(field.surface_material_palette, start=1):
            if material not in surface_palette:
                surface_palette.append(material)
            # 覆盖层自身由独立的纵向数据写入；例如 gravel 这种不在覆盖
            # palette 中的材质仍然属于原地表，应继续更新 surface_id。
            if material not in field.surface_cover_palette:
                surface_id[(field.surface_material_id == material_index) & ~wet] = (
                    surface_palette.index(material)
                )
    permafrost_palette = field.permafrost_palette or PERMAFROST_PALETTE
    # 冻土材质是独立语义层，但仍要投影到 surface_id，保证旧版控制台和
    # Minecraft 表面适配器都能直接看到真实方块；permafrost_id 保留给 Server 查询。
    for material_index, material in enumerate(permafrost_palette, start=1):
        if material not in surface_palette:
            surface_palette.append(material)
        surface_id[(field.permafrost_id == material_index) & ~wet] = surface_palette.index(
            material
        )
    subsurface_id = np.where(high, stone_id, grass_id).astype(np.uint8)
    biome_id = np.where(wet, river_biome_id, land_biome_id).astype(np.uint8)
    feature_mask = np.clip(
        np.maximum(np.abs(field.moisture - 0.5) * 0.8, high.astype(np.float32) * 0.35),
        0.0,
        1.0,
    ).astype(np.float32)
    wilderness_id = classify_wilderness(field.height, field.water_level, sea_level=sea_level)
    return BongTile(
        height=np.ascontiguousarray(field.height, dtype=np.float32),
        surface_id=surface_id,
        subsurface_id=subsurface_id,
        water_level=np.ascontiguousarray(field.water_level, dtype=np.float32),
        uplift=np.ascontiguousarray(field.uplift, dtype=np.float32),
        glacial_landform_id=np.ascontiguousarray(field.glacial_landform_id, dtype=np.uint8),
        glacial_landform_palette=field.glacial_landform_palette,
        glacial_water_id=np.ascontiguousarray(field.glacial_water_id, dtype=np.uint8),
        glacial_water_palette=field.glacial_water_palette,
        glacial_crevasse_id=np.ascontiguousarray(field.glacial_crevasse_id, dtype=np.uint8),
        glacial_crevasse_palette=field.glacial_crevasse_palette,
        glacial_discharge=np.ascontiguousarray(field.glacial_discharge, dtype=np.float32),
        valley_depth=np.ascontiguousarray(field.valley_depth, dtype=np.float32),
        valley_flow_accumulation=np.ascontiguousarray(
            field.valley_flow_accumulation, dtype=np.float32
        ),
        valley_stream_power=np.ascontiguousarray(field.valley_stream_power, dtype=np.float32),
        watershed_divide=np.ascontiguousarray(field.watershed_divide, dtype=np.float32),
        hydraulic_erosion=np.ascontiguousarray(field.hydraulic_erosion, dtype=np.float32),
        hydraulic_deposition=np.ascontiguousarray(field.hydraulic_deposition, dtype=np.float32),
        snow_accumulation=np.ascontiguousarray(field.snow_accumulation, dtype=np.float32),
        glacial_mass_balance=np.ascontiguousarray(
            field.glacial_mass_balance, dtype=np.float32
        ),
        wind_snow_alignment=np.ascontiguousarray(
            field.wind_snow_alignment, dtype=np.float32
        ),
        wind_snow_response=np.ascontiguousarray(
            field.wind_snow_response, dtype=np.float32
        ),
        biome_id=biome_id,
        feature_mask=feature_mask,
        boundary_weight=np.zeros(field.height.shape, dtype=np.float32),
        wilderness_id=wilderness_id,
        riverbed_id=np.ascontiguousarray(
            np.where(field.riverbed_id >= 0, field.riverbed_id, RIVERBED_NONE_ID),
            dtype=np.uint8,
        ),
        surface_palette=tuple(surface_palette),
        riverbed_palette=field.riverbed_palette,
        solid_spans=field.solid_spans,
        cave_id=np.ascontiguousarray(field.cave_id, dtype=np.uint8),
        cave_palette=field.cave_palette,
        fracture_id=np.ascontiguousarray(field.fracture_id, dtype=np.uint8),
        fracture_palette=field.fracture_palette,
        surface_material_id=np.ascontiguousarray(field.surface_material_id, dtype=np.uint8),
        surface_material_palette=field.surface_material_palette,
        mountain_material_id=np.ascontiguousarray(field.mountain_material_id, dtype=np.uint8),
        mountain_material_palette=field.mountain_material_palette,
        mountain_weight=np.ascontiguousarray(field.mountain_weight, dtype=np.float32),
        mountain_snowline=np.ascontiguousarray(field.mountain_snowline, dtype=np.float32),
        mountain_material_score=np.ascontiguousarray(
            field.mountain_material_score, dtype=np.float32
        ),
        mountain_slope_angle=np.ascontiguousarray(field.mountain_slope_angle, dtype=np.float32),
        mountain_exposure=np.ascontiguousarray(field.mountain_exposure, dtype=np.float32),
        mountain_rock_exposure=np.ascontiguousarray(
            field.mountain_rock_exposure, dtype=np.float32
        ),
        surface_visible_id=np.ascontiguousarray(field.surface_visible_id, dtype=np.uint8),
        surface_visible_palette=field.surface_visible_palette,
        permafrost_id=np.ascontiguousarray(field.permafrost_id, dtype=np.uint8),
        permafrost_palette=permafrost_palette,
        surface_cover_layers=np.ascontiguousarray(field.surface_cover_layers, dtype=np.uint8),
        surface_cover_palette=field.surface_cover_palette,
        climate_id=np.ascontiguousarray(field.climate_id, dtype=np.uint8),
        climate_palette=field.climate_palette,
        climate_transition_id=np.ascontiguousarray(field.climate_transition_id, dtype=np.uint8),
        climate_transition_palette=field.climate_transition_palette,
        climate_transition_weight=np.ascontiguousarray(
            field.climate_transition_weight, dtype=np.float32
        ),
        climate_surface_id=np.ascontiguousarray(field.climate_surface_id, dtype=np.uint8),
        climate_surface_palette=field.climate_surface_palette,
        underground_water_blocks=field.underground_water_blocks,
        underground_blocks=field.underground_blocks,
        settlement_spawn_areas=field.settlement_spawn_areas,
        settlement_interest_points=field.settlement_interest_points,
    )


def write_bong_raster(
    tile: BongTile,
    output_dir: Path,
    *,
    tile_x: int = 0,
    tile_z: int = 0,
    world_name: str = "bong_worldgen_preview",
    write_manifest: bool = True,
    resource_palette: tuple[str, ...] | None = None,
) -> Path:
    """写出一个 loader 兼容的 v2 raster tile 及其 manifest。

    独立适配器为每列输出一个地表实心段，在保持垂直契约正确的同时，
    把洞穴和浮岛留给后续 spans 修饰阶段。
    """

    if tile.height.ndim != 2 or tile.height.shape[0] != tile.height.shape[1]:
        raise ValueError("Bong raster tiles must be square")
    tile_size = tile.height.shape[0]
    tile_dir = output_dir / f"tile_{tile_x}_{tile_z}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    surface = tile.surface_id.astype(np.uint8, copy=False)
    subsurface = tile.subsurface_id.astype(np.uint8, copy=False)
    biome = tile.biome_id.astype(np.uint8, copy=False)
    heights = np.rint(tile.height).astype(np.int16).reshape(-1)
    if tile.solid_spans is None:
        spans_count = np.ones(tile_size * tile_size, dtype=np.uint8)
        spans = np.full(tile_size * tile_size * 8, 32767, dtype="<i2")
        spans[0::8] = -64
        spans[1::8] = np.clip(heights, -64, 431)
    else:
        spans_array = np.asarray(tile.solid_spans, dtype="<i2")
        spans_count = np.count_nonzero(spans_array[..., 0] != 32767, axis=2).astype(np.uint8).reshape(-1)
        spans = spans_array.reshape(tile_size * tile_size * 8)

    binary_layers = {
        "spans_count.bin": spans_count,
        "spans.bin": spans,
        "surface_id.bin": surface,
        "subsurface_id.bin": subsurface,
        "water_level.bin": tile.water_level.astype("<f4", copy=False),
        "uplift.bin": tile.uplift.astype("<f4", copy=False),
        "glacial_landform_id.bin": tile.glacial_landform_id.astype(np.uint8, copy=False),
        "glacial_water_id.bin": tile.glacial_water_id.astype(np.uint8, copy=False),
        "glacial_crevasse_id.bin": tile.glacial_crevasse_id.astype(np.uint8, copy=False),
        "glacial_discharge.bin": tile.glacial_discharge.astype("<f4", copy=False),
        "valley_depth.bin": tile.valley_depth.astype("<f4", copy=False),
        "valley_flow_accumulation.bin": tile.valley_flow_accumulation.astype(
            "<f4", copy=False
        ),
        "valley_stream_power.bin": tile.valley_stream_power.astype("<f4", copy=False),
        "watershed_divide.bin": tile.watershed_divide.astype("<f4", copy=False),
        "hydraulic_erosion.bin": tile.hydraulic_erosion.astype("<f4", copy=False),
        "hydraulic_deposition.bin": tile.hydraulic_deposition.astype("<f4", copy=False),
        "snow_accumulation.bin": tile.snow_accumulation.astype("<f4", copy=False),
        "glacial_mass_balance.bin": tile.glacial_mass_balance.astype("<f4", copy=False),
        "wind_snow_alignment.bin": tile.wind_snow_alignment.astype("<f4", copy=False),
        "wind_snow_response.bin": tile.wind_snow_response.astype("<f4", copy=False),
        "biome_id.bin": biome,
        "feature_mask.bin": tile.feature_mask.astype("<f4", copy=False),
        "boundary_weight.bin": tile.boundary_weight.astype("<f4", copy=False),
        "wilderness_id.bin": tile.wilderness_id.astype(np.uint8, copy=False),
        "riverbed_id.bin": tile.riverbed_id.astype(np.uint8, copy=False),
        "cave_id.bin": tile.cave_id.astype(np.uint8, copy=False),
        "fracture_id.bin": tile.fracture_id.astype(np.uint8, copy=False),
        "surface_material_id.bin": tile.surface_material_id.astype(np.uint8, copy=False),
        "mountain_material_id.bin": tile.mountain_material_id.astype(np.uint8, copy=False),
        "mountain_weight.bin": tile.mountain_weight.astype("<f4", copy=False),
        "mountain_snowline.bin": tile.mountain_snowline.astype("<f4", copy=False),
        "mountain_material_score.bin": tile.mountain_material_score.astype("<f4", copy=False),
        "mountain_slope_angle.bin": tile.mountain_slope_angle.astype("<f4", copy=False),
        "mountain_exposure.bin": tile.mountain_exposure.astype("<f4", copy=False),
        "mountain_rock_exposure.bin": tile.mountain_rock_exposure.astype("<f4", copy=False),
        "surface_visible_id.bin": tile.surface_visible_id.astype(np.uint8, copy=False),
        "permafrost_id.bin": tile.permafrost_id.astype(np.uint8, copy=False),
        "surface_cover_layers.bin": tile.surface_cover_layers.astype(np.uint8, copy=False),
        "climate_id.bin": tile.climate_id.astype(np.uint8, copy=False),
        "climate_transition_id.bin": tile.climate_transition_id.astype(np.uint8, copy=False),
        "climate_transition_weight.bin": tile.climate_transition_weight.astype("<f4", copy=False),
        "climate_surface_id.bin": tile.climate_surface_id.astype(np.uint8, copy=False),
    }
    tile_origin_x = tile_x * tile_size
    tile_origin_z = tile_z * tile_size
    water_records = bytearray()
    for block in tile.underground_water_blocks:
        if not (
            tile_origin_x <= block.x < tile_origin_x + tile_size
            and tile_origin_z <= block.z < tile_origin_z + tile_size
        ):
            continue
        kind_id = 1 if block.kind == "river" else 2
        # 固定 16 字节记录：i32 world_x, i16 world_y, i32 world_z,
        # u8 kind, u8 flowing, 4 字节保留位。固定长度便于 Rust mmap。
        water_records.extend(
            struct.pack(
                "<ihiBB4x",
                block.x,
                block.y,
                block.z,
                kind_id,
                int(block.flowing),
            )
        )
    (tile_dir / "underground_water.bin").write_bytes(water_records)
    catalog = _resource_catalog(tile.underground_blocks)
    effective_resource_palette = (
        tuple(resource_palette)
        if resource_palette is not None
        else tuple(entry["resource_id"] for entry in catalog)
    )
    resource_ids = {
        resource_id: index for index, resource_id in enumerate(effective_resource_palette)
    }
    resource_records = bytearray()
    for block in tile.underground_blocks:
        if not (
            tile_origin_x <= block.x < tile_origin_x + tile_size
            and tile_origin_z <= block.z < tile_origin_z + tile_size
        ):
            continue
        resource_id, source, rarity = _resource_identity(block)
        try:
            resource_index = resource_ids[resource_id]
        except KeyError as error:
            raise ValueError(
                f"resource {resource_id!r} is missing from the world resource palette"
            ) from error
        resource_records.extend(
            struct.pack(
                "<ihiHBB2x",
                block.x,
                block.y,
                block.z,
                resource_index,
                RESOURCE_SOURCE_IDS[source],
                RESOURCE_RARITY_IDS[rarity],
            )
        )
    (tile_dir / "underground_resources.bin").write_bytes(resource_records)
    settlement_areas = [
        settlement_spawn_area_manifest(area)
        for area in tile.settlement_spawn_areas
        if not (
            area.max_x < tile_origin_x
            or area.min_x >= tile_origin_x + tile_size
            or area.max_z < tile_origin_z
            or area.min_z >= tile_origin_z + tile_size
        )
    ]
    (tile_dir / "settlement_spawn_areas.json").write_text(
        json.dumps(settlement_areas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    settlement_points = [
        settlement_interest_point_manifest(point)
        for point in tile.settlement_interest_points
        if tile_origin_x <= point.x < tile_origin_x + tile_size
        and tile_origin_z <= point.z < tile_origin_z + tile_size
    ]
    (tile_dir / "settlement_interest_points.json").write_text(
        json.dumps(settlement_points, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename, values in binary_layers.items():
        values.tofile(tile_dir / filename)

    manifest = {
        "version": 2,
        "backend": "raster",
        "world_name": world_name,
        "tile_size": tile_size,
        "world_bounds": {
            "min_x": tile_x * tile_size,
            "max_x": (tile_x + 1) * tile_size - 1,
            "min_z": tile_z * tile_size,
            "max_z": (tile_z + 1) * tile_size - 1,
        },
        "surface_palette": list(tile.surface_palette),
        "surface_material_palette": list(tile.surface_material_palette),
        "mountain_material_palette": list(tile.mountain_material_palette),
        "mountain_material_encoding": {
            "material_file": "mountain_material_id.bin",
            "weight_file": "mountain_weight.bin",
            "snowline_file": "mountain_snowline.bin",
            "score_file": "mountain_material_score.bin",
            "slope_angle_file": "mountain_slope_angle.bin",
            "exposure_file": "mountain_exposure.bin",
            "rock_exposure_file": "mountain_rock_exposure.bin",
            "id_dtype": "u8",
            "field_dtype": "f32",
            "none": 0,
            "description": "仅群山有限距离场内的雪、冰、蓝冰、裸岩材质与驱动场",
        },
        "surface_visible_palette": list(tile.surface_visible_palette),
        "surface_cover_palette": list(tile.surface_cover_palette),
        "permafrost_palette": list(tile.permafrost_palette),
        "riverbed_palette": list(tile.riverbed_palette),
        "riverbed_encoding": {"none": RIVERBED_NONE_ID, "dtype": "u8"},
        "cave_palette": list(tile.cave_palette),
        "cave_encoding": {"none": 0, "dtype": "u8", "vertical_range": "spans.bin"},
        "fracture_palette": list(tile.fracture_palette),
        "fracture_encoding": {"none": 0, "dtype": "u8", "vertical_range": "spans.bin"},
        "surface_material_encoding": {"none": 0, "dtype": "u8"},
        "surface_visible_encoding": {
            "file": "surface_visible_id.bin",
            "none": 0,
            "dtype": "u8",
            "description": "最终可见的地表方块材质；覆盖层和洞口保护已在生成阶段合并",
        },
        "permafrost_encoding": {
            "file": "permafrost_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "surface_cover_encoding": {
            "file": "surface_cover_layers.bin",
            "dtype": "u8",
            "shape": [4, "height", "width"],
            "order": list(tile.surface_cover_palette),
            "base": "height layer is retained; these blocks are stacked above it",
        },
        "glacial_landform_palette": list(tile.glacial_landform_palette),
        "glacial_landform_encoding": {
            "file": "glacial_landform_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "glacial_water_palette": list(tile.glacial_water_palette),
        "glacial_water_encoding": {
            "id_file": "glacial_water_id.bin",
            "discharge_file": "glacial_discharge.bin",
            "id_dtype": "u8",
            "discharge_dtype": "f32",
            "none": 0,
        },
        "glacial_crevasse_palette": list(tile.glacial_crevasse_palette),
        "glacial_crevasse_encoding": {
            "file": "glacial_crevasse_id.bin",
            "none": 0,
            "dtype": "u8",
            "description": "冰川表面横向裂隙；与地下 fracture_id 分离",
        },
        "valley_hydrology_encoding": {
            "depth_file": "valley_depth.bin",
            "flow_accumulation_file": "valley_flow_accumulation.bin",
            "stream_power_file": "valley_stream_power.bin",
            "dtype": "f32",
            "description": "Flow Accumulation 与 Stream Power 生成的谷地物理场",
        },
        "watershed_divide_encoding": {
            "file": "watershed_divide.bin",
            "dtype": "f32",
            "description": "D8 流域出口边界在 Uplift + Stream Power 之后形成的山脊修饰抬升量",
        },
        "hydraulic_erosion_encoding": {
            "erosion_file": "hydraulic_erosion.bin",
            "deposition_file": "hydraulic_deposition.bin",
            "dtype": "f32",
            "description": "末端 droplet hydraulic erosion refinement 的累计下切与冲积沉积",
        },
        "uplift_encoding": {
            "file": "uplift.bin",
            "dtype": "f32",
            "description": (
                "Mountain Spine 产生、供 Stream Power 耦合的正向构造抬升场"
            ),
        },
        "glacial_mass_balance_encoding": {
            "snow_accumulation_file": "snow_accumulation.bin",
            "mass_balance_file": "glacial_mass_balance.bin",
            "dtype": "f32",
            "mass_balance_sign": {"positive": "accumulation", "negative": "ablation"},
        },
        "wind_snow_encoding": {
            "alignment_file": "wind_snow_alignment.bin",
            "response_file": "wind_snow_response.bin",
            "dtype": "f32",
            "alignment": "horizontal terrain normal dot wind direction",
            "response": "positive accumulation, negative wind erosion",
        },
        "climate_palette": list(tile.climate_palette),
        "climate_encoding": {"file": "climate_id.bin", "none": 0, "dtype": "u8"},
        "climate_transition_palette": list(tile.climate_transition_palette),
        "climate_transition_encoding": {
            "id_file": "climate_transition_id.bin",
            "weight_file": "climate_transition_weight.bin",
            "none": 0,
            "id_dtype": "u8",
            "weight_dtype": "f32",
        },
        "climate_surface_palette": list(tile.climate_surface_palette),
        "climate_surface_encoding": {
            "file": "climate_surface_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "underground_water_encoding": {
            "file": "underground_water.bin",
            "record_size": 16,
            "fields": "i32 world_x, i16 world_y, i32 world_z, u8 kind, u8 flowing, 4 reserved bytes",
            "kind": {"1": "river", "2": "lake"},
        },
        "underground_resource_palette": list(effective_resource_palette),
        "underground_material_palette": sorted(
            set(UNDERGROUND_MATERIAL_PALETTE)
            | {entry["material"] for entry in catalog}
        ),
        "underground_resource_catalog": list(catalog),
        "underground_resources_encoding": {
            "file": "underground_resources.bin",
            "record_size": 16,
            "fields": "i32 world_x, i16 world_y, i32 world_z, u16 resource_id, u8 source_id, u8 rarity_id, 2 reserved bytes",
            "source": {str(value): key for key, value in RESOURCE_SOURCE_IDS.items()},
            "rarity": {str(value): key for key, value in RESOURCE_RARITY_IDS.items() if key is not None},
        },
        "settlement_spawn_area_encoding": {
            "file": "settlement_spawn_areas.json",
            "format": "json array",
            "description": "每栋建筑的 NPC 刷新与防守 AABB；spawn_bounds 从地面 Y 开始，footprint 保留完整结构高度",
        },
        "settlement_interest_point_encoding": {
            "file": "settlement_interest_points.json",
            "format": "json array",
            "description": "聚落可查询兴趣点；包含建筑中心、道路节点和可供 NPC 刷新的屋内点位",
        },
        "biome_palette": ["minecraft:plains", "minecraft:river"],
        "wilderness_palette": wilderness_palette_manifest(),
        "tiles": [
            {
                "tile_x": tile_x,
                "tile_z": tile_z,
                "dir": tile_dir.name,
                "layers": [
                    "surface_id",
                    "subsurface_id",
                    "water_level",
                    "uplift",
                    "glacial_landform_id",
                    "glacial_water_id",
                    "glacial_crevasse_id",
                    "glacial_discharge",
                    "valley_depth",
                    "valley_flow_accumulation",
                    "valley_stream_power",
                    "watershed_divide",
                    "hydraulic_erosion",
                    "hydraulic_deposition",
                    "snow_accumulation",
                    "glacial_mass_balance",
                    "wind_snow_alignment",
                    "wind_snow_response",
                    "biome_id",
                    "feature_mask",
                    "boundary_weight",
                    "wilderness_id",
                    "riverbed_id",
                    "cave_id",
                    "fracture_id",
                    "surface_material_id",
                    "mountain_material_id",
                    "mountain_weight",
                    "mountain_snowline",
                    "mountain_material_score",
                    "mountain_slope_angle",
                    "mountain_exposure",
                    "mountain_rock_exposure",
                    "surface_visible_id",
                    "permafrost_id",
                    "surface_cover_layers",
                    "climate_id",
                    "climate_transition_id",
                    "climate_transition_weight",
                    "climate_surface_id",
                    "settlement_interest_points",
                ],
                "spans": True,
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    if write_manifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10),
            encoding="utf-8",
        )
    return manifest_path
