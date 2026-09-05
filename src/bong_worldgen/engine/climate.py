"""全球气候带的数据契约。

本模块描述气候带的顺序、范围和过渡配置，并提供从数据层世界边界采样气候
标量场的函数。引擎不复制固定地图坐标，所有边界通过 ``ClimateWorldBounds`` 注入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np


ClimateAxis = Literal["world_z"]
ClimateKind = Literal["cold", "temperate", "tropical"]
ClimateSurfaceFamily = Literal["snow", "temperate_grassland", "desert"]

CLIMATE_PALETTE = ("cold", "temperate", "tropical")
CLIMATE_TRANSITION_PALETTE = ("none", "cold_temperate", "temperate_tropical")
CLIMATE_SURFACE_PALETTE = ("snow", "temperate_grassland", "desert")


@dataclass(frozen=True)
class ClimateWorldBounds:
    """把归一化气候轴映射到世界坐标所需的南北边界。

    世界边界由数据层注入；引擎不假设任何固定地图大小，也不会用当前 tile
    的范围代替整个世界范围。
    """

    north_z: float
    south_z: float

    def __post_init__(self) -> None:
        if not isfinite(self.north_z) or not isfinite(self.south_z):
            raise ValueError("climate world bounds must be finite")
        if self.north_z >= self.south_z:
            raise ValueError("climate world bounds must be ordered north to south")


@dataclass(frozen=True)
class ClimateField:
    """一次采样得到的气候查询层；权重用于地貌阶段的连续门控。"""

    climate_id: np.ndarray
    transition_id: np.ndarray
    transition_weight: np.ndarray
    surface_id: np.ndarray
    cold_weight: np.ndarray


def sample_climate(z: np.ndarray, plan: "GlobalClimatePlan", bounds: ClimateWorldBounds) -> ClimateField:
    """按世界南北坐标采样气候带和相邻过渡带。

    硬分类用于 server/decorations 查询；``cold_weight`` 和
    ``transition_weight`` 保持连续，供地貌 carve 阶段避免硬切边界。
    """

    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("climate coordinates must be a finite two-dimensional array")
    normalized = np.clip(
        (z.astype(np.float64, copy=False) - bounds.north_z)
        / (bounds.south_z - bounds.north_z)
        * 2.0
        - 1.0,
        -1.0,
        1.0,
    )
    south_edges = np.asarray([band.south_edge for band in plan.bands[:-1]], dtype=np.float64)
    climate_index = np.searchsorted(south_edges, normalized, side="right")
    climate_id = (climate_index + 1).astype(np.uint8)
    transition_id = np.zeros(z.shape, dtype=np.uint8)
    transition_weight = np.zeros(z.shape, dtype=np.float32)
    cold_weight = (climate_id == 1).astype(np.float32)

    for index, transition in enumerate(plan.transitions):
        boundary = plan.bands[index].south_edge
        distance = np.abs(normalized - boundary)
        mask = distance <= transition.width
        transition_id[mask] = np.uint8(index + 1)
        transition_weight[mask] = np.maximum(
            transition_weight[mask],
            (1.0 - distance[mask] / transition.width).astype(np.float32),
        )
        if index == 0:
            local = np.clip(
                (normalized - (boundary - transition.width)) / (2.0 * transition.width),
                0.0,
                1.0,
            )
            cold_weight[mask] = (1.0 - local[mask]).astype(np.float32)

    surface_lookup = {name: index for index, name in enumerate(CLIMATE_SURFACE_PALETTE)}
    surface_palette_ids = np.asarray(
        [surface_lookup[band.surface_family] for band in plan.bands], dtype=np.uint8
    )
    surface_id = (surface_palette_ids[climate_index] + 1).astype(np.uint8)
    return ClimateField(
        climate_id=np.ascontiguousarray(climate_id),
        transition_id=np.ascontiguousarray(transition_id),
        transition_weight=np.ascontiguousarray(transition_weight),
        surface_id=np.ascontiguousarray(surface_id),
        cold_weight=np.ascontiguousarray(cold_weight),
    )


@dataclass(frozen=True)
class ClimateBand:
    """一条按世界归一化南北坐标描述的总气候带。

    ``north_edge`` 和 ``south_edge`` 使用 ``[-1, 1]``：-1 是地图北端，+1
    是地图南端；世界坐标映射由 ``sample_climate`` 按数据层边界执行。
    ``surface_family`` 是地表语义：热带默认使用沙漠，而不是热带草地。
    """

    kind: ClimateKind
    display_name: str
    surface_family: ClimateSurfaceFamily
    north_edge: float
    south_edge: float

    def __post_init__(self) -> None:
        if self.kind not in ("cold", "temperate", "tropical"):
            raise ValueError("climate band kind must be cold, temperate, or tropical")
        if self.surface_family not in ("snow", "temperate_grassland", "desert"):
            raise ValueError("unsupported climate surface family")
        if not self.display_name.strip():
            raise ValueError("climate band display_name must not be empty")
        if not isfinite(self.north_edge) or not isfinite(self.south_edge):
            raise ValueError("climate band edges must be finite")
        if not -1.0 <= self.north_edge < self.south_edge <= 1.0:
            raise ValueError("climate band edges must be ordered within [-1, 1]")


@dataclass(frozen=True)
class ClimateTransition:
    """相邻总气候带之间的过渡地形配置。

    ``width`` 是归一化坐标中的过渡半宽；它只保留参数，不执行任何混合。
    """

    from_kind: ClimateKind
    to_kind: ClimateKind
    width: float = 0.12

    def __post_init__(self) -> None:
        climate_kinds = ("cold", "temperate", "tropical")
        if self.from_kind not in climate_kinds or self.to_kind not in climate_kinds:
            raise ValueError("unsupported climate transition kind")
        if self.from_kind == self.to_kind:
            raise ValueError("climate transition must connect two different bands")
        if not isfinite(self.width) or self.width <= 0 or self.width >= 1.0:
            raise ValueError("climate transition width must be in (0, 1)")


@dataclass(frozen=True)
class GlobalClimatePlan:
    """从北向南排列的全球气候框架，默认是寒带、温带、热带。

    气候带本身仍使用归一化轴，世界坐标边界由 ``TerrainRecipe`` 显式注入。
    """

    axis: ClimateAxis = "world_z"
    bands: tuple[ClimateBand, ...] = field(
        default_factory=lambda: (
            ClimateBand("cold", "寒带", "snow", -1.0, -0.34),
            ClimateBand("temperate", "温带", "temperate_grassland", -0.34, 0.34),
            ClimateBand("tropical", "热带", "desert", 0.34, 1.0),
        )
    )
    transitions: tuple[ClimateTransition, ...] = field(
        default_factory=lambda: (
            ClimateTransition("cold", "temperate"),
            ClimateTransition("temperate", "tropical"),
        )
    )

    def __post_init__(self) -> None:
        if self.axis != "world_z":
            raise ValueError("global climate axis must be world_z")
        expected_kinds = ("cold", "temperate", "tropical")
        actual_kinds = tuple(band.kind for band in self.bands)
        if actual_kinds != expected_kinds:
            raise ValueError("global climate bands must be ordered cold, temperate, tropical")
        if self.bands[0].north_edge != -1.0 or self.bands[-1].south_edge != 1.0:
            raise ValueError("global climate bands must cover the full normalized north-south axis")
        if any(
            previous.south_edge != current.north_edge
            for previous, current in zip(self.bands, self.bands[1:])
        ):
            raise ValueError("global climate bands must be contiguous")
        if len(self.transitions) != 2:
            raise ValueError("global climate plan requires two adjacent transitions")
        expected_transitions = (("cold", "temperate"), ("temperate", "tropical"))
        actual_transitions = tuple(
            (transition.from_kind, transition.to_kind) for transition in self.transitions
        )
        if actual_transitions != expected_transitions:
            raise ValueError("global climate transitions must connect adjacent bands")


__all__ = [
    "CLIMATE_PALETTE",
    "CLIMATE_SURFACE_PALETTE",
    "CLIMATE_TRANSITION_PALETTE",
    "ClimateAxis",
    "ClimateBand",
    "ClimateField",
    "ClimateKind",
    "ClimateSurfaceFamily",
    "ClimateTransition",
    "ClimateWorldBounds",
    "GlobalClimatePlan",
    "sample_climate",
]
