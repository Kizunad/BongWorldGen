"""把汇水面积与坡度转换为河流下切深度。

核心采用 Stream Power 形式 ``E = K A^m S^n``；本项目把一次地貌演化的
累计侵蚀限制在 ``maximum_depth`` 内，再按汇水面积生成可变宽度谷槽。
公式和参数含义参考 Landlab：
https://github.com/landlab/landlab/blob/master/src/landlab/components/stream_power/stream_power.py
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain_config import ValleySystem
from .hydrology import FlowField


def _expanded_valley_depth(
    center_depth: np.ndarray,
    channel_width: np.ndarray,
    system: ValleySystem,
    cell_size_x: float,
    cell_size_z: float,
) -> np.ndarray:
    """用连续 U/V 之间的横截面扩宽 D8 中心线，不改变其水文拓扑。"""

    output = np.zeros(center_depth.shape, dtype=np.float64)
    radius_x = math.ceil(system.maximum_width / cell_size_x)
    radius_z = math.ceil(system.maximum_width / cell_size_z)
    height, width = center_depth.shape

    for row_offset in range(-radius_z, radius_z + 1):
        for column_offset in range(-radius_x, radius_x + 1):
            distance = math.hypot(
                column_offset * cell_size_x,
                row_offset * cell_size_z,
            )
            if distance > system.maximum_width:
                continue
            source_row_start = max(0, -row_offset)
            source_row_stop = min(height, height - row_offset)
            source_column_start = max(0, -column_offset)
            source_column_stop = min(width, width - column_offset)
            target_row_start = source_row_start + row_offset
            target_row_stop = source_row_stop + row_offset
            target_column_start = source_column_start + column_offset
            target_column_stop = source_column_stop + column_offset

            source = (
                slice(source_row_start, source_row_stop),
                slice(source_column_start, source_column_stop),
            )
            target = (
                slice(target_row_start, target_row_stop),
                slice(target_column_start, target_column_stop),
            )
            local_width = channel_width[source]
            normalized_distance = distance / np.maximum(local_width, 1.0e-12)
            cross_section = np.power(
                np.clip(1.0 - normalized_distance**2, 0.0, 1.0),
                system.cross_section_power,
            )
            candidate = center_depth[source] * cross_section
            output[target] = np.maximum(output[target], candidate)
    return output


def stream_power_erosion(
    terrain: np.ndarray,
    flow: FlowField,
    system: ValleySystem,
    *,
    sea_level: float,
    cell_size_x: float,
    cell_size_z: float,
    uplift: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回扩宽后的侵蚀深度，以及 Stream Power 标量场。

    基础项仍为 ``A^m S^n``。当提供山脉 Uplift 场时，使用
    ``(1 + c * U / U_ref)`` 作为构造耦合项：抬升越强的区域，河流下切越
    强；Uplift 不直接改写水文拓扑，也不会凭空抬高或降低地形。
    """

    if any(
        field.shape != terrain.shape
        for field in (flow.accumulation, flow.slope, flow.receiver)
    ):
        raise ValueError("stream-power fields must share the terrain shape")
    if uplift is None:
        uplift_field = np.zeros_like(terrain, dtype=np.float64)
    else:
        uplift_field = np.asarray(uplift, dtype=np.float64)
        if uplift_field.shape != terrain.shape:
            raise ValueError("stream-power uplift must share the terrain shape")
        if not np.isfinite(uplift_field).all() or np.any(uplift_field < 0.0):
            raise ValueError("stream-power uplift must be finite and non-negative")

    area_ratio = flow.accumulation / system.reference_drainage_area
    slope_ratio = flow.slope / system.reference_slope
    base_stream_power = (
        np.power(np.maximum(area_ratio, 0.0), system.area_exponent)
        * np.power(np.maximum(slope_ratio, 0.0), system.slope_exponent)
    )
    uplift_ratio = np.clip(uplift_field / system.uplift_reference, 0.0, 1.0)
    uplift_factor = 1.0 + system.uplift_strength * uplift_ratio
    stream_power = base_stream_power * uplift_factor
    active = (
        (flow.accumulation >= system.minimum_drainage_area)
        & (terrain >= sea_level + system.minimum_height_above_sea_level)
        & (flow.slope > 0.0)
    )
    # 指数饱和保持低阶支沟细浅，同时让大汇水主谷逐渐逼近最大深度。
    normalized_incision = 1.0 - np.exp(-system.erosion_strength * stream_power)
    center_depth = np.where(active, system.maximum_depth * normalized_incision, 0.0)

    normalized_area = np.clip(
        (flow.accumulation - system.minimum_drainage_area)
        / (system.reference_drainage_area - system.minimum_drainage_area),
        0.0,
        1.0,
    )
    channel_width = system.minimum_width + (
        system.maximum_width - system.minimum_width
    ) * np.power(normalized_area, system.width_exponent)
    channel_width = np.where(active, channel_width, 0.0)
    erosion_depth = _expanded_valley_depth(
        center_depth,
        channel_width,
        system,
        cell_size_x,
        cell_size_z,
    )
    return (
        np.ascontiguousarray(erosion_depth),
        np.ascontiguousarray(np.where(active, stream_power, 0.0)),
    )


__all__ = ["stream_power_erosion"]
