"""一次世界生成产生的连续字段与稀疏地下记录。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import SPAN_MAX_SPANS, SPAN_MIN_Y
from .settlement_points import SettlementInterestPoint, SettlementSpawnArea
from .town import SettlementBlock
from .underground_config import UndergroundBlock, UndergroundWaterBlock


def _palette_ids_are_valid(
    values: np.ndarray,
    palette: tuple[str, ...],
    *,
    zero_is_none: bool = True,
) -> bool:
    if zero_is_none:
        valid = (values == 0) | ((values > 0) & (values <= len(palette)))
    else:
        valid = (values >= 0) & (values < len(palette)) if palette else values == 0
    return bool(np.all(valid))


@dataclass(frozen=True)
class Heightfield:
    """一次确定性生成流程产生的全部标量场和稀疏地下结果。"""

    height: np.ndarray
    moisture: np.ndarray
    water_level: np.ndarray
    # 山脉阶段产生的正向构造抬升；谷地侵蚀不会覆盖这张驱动场。
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
    # 分水岭边界在 Stream Power 谷地侵蚀之后叠加的连续山脊抬升量。
    watershed_divide: np.ndarray | None = None
    # 滴水 refinement 的累计侵蚀与沉积量；两者均为非负高度差。
    hydraulic_erosion: np.ndarray | None = None
    hydraulic_deposition: np.ndarray | None = None
    snow_accumulation: np.ndarray | None = None
    glacial_mass_balance: np.ndarray | None = None
    # 风向与地形法线耦合后的连续积雪修正；正值为迎风积雪，负值为背风削薄。
    wind_snow_alignment: np.ndarray | None = None
    wind_snow_response: np.ndarray | None = None
    riverbed_id: np.ndarray | None = None
    riverbed_palette: tuple[str, ...] = ()
    solid_spans: np.ndarray | None = None
    underground_blocks: tuple[UndergroundBlock, ...] = ()
    underground_water_blocks: tuple[UndergroundWaterBlock, ...] = ()
    cave_id: np.ndarray | None = None
    cave_palette: tuple[str, ...] = ()
    fracture_id: np.ndarray | None = None
    fracture_palette: tuple[str, ...] = ()
    surface_material_id: np.ndarray | None = None
    surface_material_palette: tuple[str, ...] = ()
    # 群山场驱动材质与连续查询场；山脉范围外均为零。
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
    # 轴顺序为 powder_snow / snow_block / ice / blue_ice。
    surface_cover_layers: np.ndarray | None = None
    surface_cover_palette: tuple[str, ...] = ()
    climate_id: np.ndarray | None = None
    climate_palette: tuple[str, ...] = ()
    climate_transition_id: np.ndarray | None = None
    climate_transition_palette: tuple[str, ...] = ()
    climate_transition_weight: np.ndarray | None = None
    climate_surface_id: np.ndarray | None = None
    climate_surface_palette: tuple[str, ...] = ()
    # 城镇结构方块；追加在末尾以保持既有位置参数兼容。
    settlement_blocks: tuple[SettlementBlock, ...] = ()
    # 每栋建筑的 NPC 刷新/防守区域，按结构真实尺寸和配置缓冲计算。
    settlement_spawn_areas: tuple[SettlementSpawnArea, ...] = ()
    # 可被 Server/NPC/任务系统直接查询的聚落兴趣点。
    settlement_interest_points: tuple[SettlementInterestPoint, ...] = ()

    def __post_init__(self) -> None:
        shape = self.height.shape
        zero_id_layers = (
            "glacial_landform_id",
            "glacial_water_id",
            "glacial_crevasse_id",
            "cave_id",
            "fracture_id",
            "surface_material_id",
            "mountain_material_id",
            "surface_visible_id",
            "permafrost_id",
            "climate_id",
            "climate_transition_id",
            "climate_surface_id",
        )
        for name in zero_id_layers:
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(shape, dtype=np.uint8))
        if self.riverbed_id is None:
            object.__setattr__(self, "riverbed_id", np.full(shape, -1, dtype=np.int16))
        for name in (
            "mountain_weight",
            "mountain_snowline",
            "mountain_material_score",
            "mountain_slope_angle",
            "mountain_exposure",
            "mountain_rock_exposure",
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(shape, dtype=np.float32))
        if self.glacial_discharge is None:
            object.__setattr__(self, "glacial_discharge", np.zeros(shape, dtype=np.float32))
        if self.uplift is None:
            object.__setattr__(self, "uplift", np.zeros(shape, dtype=np.float32))
        for name in (
            "valley_depth",
            "valley_flow_accumulation",
            "valley_stream_power",
            "watershed_divide",
            "hydraulic_erosion",
            "hydraulic_deposition",
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(shape, dtype=np.float32))
        if self.snow_accumulation is None:
            object.__setattr__(self, "snow_accumulation", np.zeros(shape, dtype=np.float32))
        if self.glacial_mass_balance is None:
            object.__setattr__(self, "glacial_mass_balance", np.zeros(shape, dtype=np.float32))
        if self.wind_snow_alignment is None:
            object.__setattr__(self, "wind_snow_alignment", np.zeros(shape, dtype=np.float32))
        if self.wind_snow_response is None:
            object.__setattr__(self, "wind_snow_response", np.zeros(shape, dtype=np.float32))
        if self.climate_transition_weight is None:
            object.__setattr__(
                self,
                "climate_transition_weight",
                np.zeros(shape, dtype=np.float32),
            )
        if self.surface_cover_layers is None:
            object.__setattr__(
                self,
                "surface_cover_layers",
                np.zeros((4, *shape), dtype=np.uint8),
            )

        layer_shapes = {
            self.height.shape,
            self.moisture.shape,
            self.water_level.shape,
            self.glacial_landform_id.shape,
            self.glacial_water_id.shape,
            self.glacial_crevasse_id.shape,
            self.glacial_discharge.shape,
            self.uplift.shape,
            self.valley_depth.shape,
            self.valley_flow_accumulation.shape,
            self.valley_stream_power.shape,
            self.watershed_divide.shape,
            self.hydraulic_erosion.shape,
            self.hydraulic_deposition.shape,
            self.snow_accumulation.shape,
            self.glacial_mass_balance.shape,
            self.wind_snow_alignment.shape,
            self.wind_snow_response.shape,
            self.riverbed_id.shape,
            self.cave_id.shape,
            self.fracture_id.shape,
            self.surface_material_id.shape,
            self.mountain_material_id.shape,
            self.mountain_weight.shape,
            self.mountain_snowline.shape,
            self.mountain_material_score.shape,
            self.mountain_slope_angle.shape,
            self.mountain_exposure.shape,
            self.mountain_rock_exposure.shape,
            self.surface_visible_id.shape,
            self.permafrost_id.shape,
            self.surface_cover_layers.shape[1:],
            self.climate_id.shape,
            self.climate_transition_id.shape,
            self.climate_transition_weight.shape,
            self.climate_surface_id.shape,
        }
        if len(layer_shapes) != 1:
            raise ValueError(f"heightfield layers must share a shape, got {sorted(layer_shapes)}")
        if self.height.ndim != 2:
            raise ValueError("heightfield layers must be two-dimensional")

        for name in (
            "height",
            "moisture",
            "water_level",
            "glacial_discharge",
            "uplift",
            "snow_accumulation",
            "glacial_mass_balance",
            "wind_snow_alignment",
            "wind_snow_response",
            "valley_depth",
            "valley_flow_accumulation",
            "valley_stream_power",
            "watershed_divide",
            "climate_transition_weight",
            "mountain_weight",
            "mountain_snowline",
            "mountain_material_score",
            "mountain_slope_angle",
            "mountain_exposure",
            "mountain_rock_exposure",
        ):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"heightfield layer {name} contains non-finite values")
        if np.any(self.glacial_discharge < 0.0):
            raise ValueError("glacial_discharge cannot be negative")
        if np.any(self.uplift < 0.0):
            raise ValueError("uplift cannot be negative")
        if np.any((self.mountain_rock_exposure < 0.0) | (self.mountain_rock_exposure > 1.0)):
            raise ValueError("mountain_rock_exposure must be within [0, 1]")
        for name in (
            "valley_depth",
            "valley_flow_accumulation",
            "valley_stream_power",
            "watershed_divide",
            "hydraulic_erosion",
            "hydraulic_deposition",
        ):
            values = getattr(self, name)
            if np.any(values < 0.0):
                raise ValueError(f"{name} cannot be negative")
        if np.any(self.snow_accumulation < 0.0):
            raise ValueError("snow_accumulation cannot be negative")
        if np.any((self.wind_snow_alignment < -1.0) | (self.wind_snow_alignment > 1.0)):
            raise ValueError("wind_snow_alignment must be within [-1, 1]")
        if np.any((self.climate_transition_weight < 0.0) | (self.climate_transition_weight > 1.0)):
            raise ValueError("climate_transition_weight must be within [0, 1]")

        palette_layers = (
            ("glacial_landform_id", self.glacial_landform_palette, True),
            ("glacial_water_id", self.glacial_water_palette, True),
            ("glacial_crevasse_id", self.glacial_crevasse_palette, True),
            ("cave_id", self.cave_palette, True),
            ("fracture_id", self.fracture_palette, True),
            ("surface_material_id", self.surface_material_palette, True),
            ("mountain_material_id", self.mountain_material_palette, True),
            ("surface_visible_id", self.surface_visible_palette, True),
            ("permafrost_id", self.permafrost_palette, True),
            ("climate_id", self.climate_palette, True),
            ("climate_transition_id", self.climate_transition_palette, False),
            ("climate_surface_id", self.climate_surface_palette, True),
        )
        for name, palette, zero_is_none in palette_layers:
            values = getattr(self, name)
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"heightfield layer {name} must contain integer palette ids")
            if np.any(values < 0) or np.any(values > 255):
                raise ValueError(f"heightfield layer {name} must fit in uint8")
            if not _palette_ids_are_valid(values, palette, zero_is_none=zero_is_none):
                raise ValueError(f"{name} contains an index outside its palette")

        if not np.issubdtype(self.riverbed_id.dtype, np.integer):
            raise ValueError("heightfield layer riverbed_id must contain integer palette ids")
        if self.riverbed_palette:
            riverbed_valid = (self.riverbed_id == -1) | (
                (self.riverbed_id >= 0) & (self.riverbed_id < len(self.riverbed_palette))
            )
        else:
            riverbed_valid = self.riverbed_id == -1
        if not np.all(riverbed_valid):
            raise ValueError("riverbed_id contains an index outside riverbed_palette")

        if self.surface_cover_layers.ndim != 3 or self.surface_cover_layers.shape[0] != 4:
            raise ValueError("surface_cover_layers must have shape (4, height, width)")
        if not np.issubdtype(self.surface_cover_layers.dtype, np.integer):
            raise ValueError("heightfield layer surface_cover_layers must contain integers")
        if np.any(self.surface_cover_layers < 0) or np.any(self.surface_cover_layers > 255):
            raise ValueError("surface_cover_layers values must fit in uint8")
        if self.surface_cover_palette and len(self.surface_cover_palette) != 4:
            raise ValueError("surface_cover_palette must contain four layer materials")

        if self.solid_spans is not None:
            expected_spans = (*shape, SPAN_MAX_SPANS, 2)
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
        for block in self.underground_water_blocks:
            if not isinstance(block, UndergroundWaterBlock):
                raise ValueError(
                    "underground_water_blocks must contain UndergroundWaterBlock values"
                )
            if block.y <= SPAN_MIN_Y:
                raise ValueError("underground water must be above the bedrock floor")
        for block in self.settlement_blocks:
            if not isinstance(block, SettlementBlock):
                raise ValueError("settlement_blocks must contain SettlementBlock values")
            if not block.material.strip() or not block.kind.strip():
                raise ValueError("settlement blocks need material and kind")
        for area in self.settlement_spawn_areas:
            if not isinstance(area, SettlementSpawnArea):
                raise ValueError(
                    "settlement_spawn_areas must contain SettlementSpawnArea values"
                )
            if not area.area_id.strip() or not area.structure_name.strip():
                raise ValueError("settlement spawn areas need an id and structure name")
            if area.min_x > area.max_x or area.min_z > area.max_z or area.min_y > area.max_y:
                raise ValueError("settlement spawn area bounds must be ordered")
            if (
                area.footprint_min_x > area.footprint_max_x
                or area.footprint_min_z > area.footprint_max_z
                or area.footprint_min_y > area.footprint_max_y
            ):
                raise ValueError("settlement spawn footprint bounds must be ordered")
            if area.defense_radius < 0:
                raise ValueError("settlement spawn defense radius cannot be negative")
        for point in self.settlement_interest_points:
            if not isinstance(point, SettlementInterestPoint):
                raise ValueError(
                    "settlement_interest_points must contain SettlementInterestPoint values"
                )
            if point.area_id is not None and not point.area_id.strip():
                raise ValueError("settlement interest point area_id cannot be blank")


__all__ = ["Heightfield"]
