"""把局部地下配方稳定地分布到整个世界。

洞穴和地下河都遵循同一条规则：配方中心为原点时，按世界网格生成
seeded 实例；显式路径或非原点中心则视为手工定位的单一实例。本模块
只负责实例位置和子 seed，不参与任何洞穴或水文几何计算。
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterator, Protocol, TypeVar

import numpy as np

from .terrain_config import Point
from .randomness import mix64, stable_text_seed, unit_interval


_MIN_SPACING = 1024.0
_CENTER_JITTER = 0.62


class WorldDistributedNetwork(Protocol):
    """全世界确定性分布所需的最小配方接口。"""

    name: str
    seed_center: Point
    seed_extent: float


NetworkT = TypeVar("NetworkT", bound=WorldDistributedNetwork)


def iter_world_network_instances(
    network: NetworkT,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    seed: int,
    network_index: int,
) -> Iterator[tuple[NetworkT, int]]:
    """产出与当前网格相交的网络实例及其稳定子 seed。"""

    explicit_paths = bool(getattr(network, "paths", ()))
    center = network.seed_center
    if explicit_paths or center.x != 0.0 or center.z != 0.0:
        yield network, seed + network_index * 9_973
        return

    spacing = max(
        _MIN_SPACING,
        network.seed_extent * 4.0,
        getattr(network, "path_length", 0.0) * 2.0,
    )
    margin = network.seed_extent + max(
        getattr(network, "branch_length", 0.0),
        getattr(network, "domain_warp_strength", 0.0),
        getattr(network, "chamber_radius", 0.0),
        getattr(network, "entrance_radius", 0.0),
    ) + 2.0
    name_seed = stable_text_seed(network.name) + network_index * 97_531
    origin_x = (unit_interval(seed, name_seed + 1) - 0.5) * spacing
    origin_z = (unit_interval(seed, name_seed + 2) - 0.5) * spacing
    min_cell_x = math.floor((float(x_axis[0]) - margin - origin_x) / spacing) - 1
    max_cell_x = math.ceil((float(x_axis[-1]) + margin - origin_x) / spacing) + 1
    min_cell_z = math.floor((float(z_axis[0]) - margin - origin_z) / spacing) - 1
    max_cell_z = math.ceil((float(z_axis[-1]) + margin - origin_z) / spacing) + 1

    for cell_z in range(min_cell_z, max_cell_z + 1):
        for cell_x in range(min_cell_x, max_cell_x + 1):
            cell_seed = mix64(
                seed
                + name_seed * 0x9E3779B1
                + cell_x * 0xBF58476D
                + cell_z * 0x94D049BB
            )
            center_x = origin_x + cell_x * spacing + (
                unit_interval(cell_seed, 11) - 0.5
            ) * spacing * _CENTER_JITTER
            center_z = origin_z + cell_z * spacing + (
                unit_interval(cell_seed, 13) - 0.5
            ) * spacing * _CENTER_JITTER
            if (
                center_x + margin < float(x_axis[0])
                or center_x - margin > float(x_axis[-1])
                or center_z + margin < float(z_axis[0])
                or center_z - margin > float(z_axis[-1])
            ):
                continue
            instance_seed = mix64(cell_seed + 0xD6E8FEB86659FD93)
            instance = replace(network, seed_center=Point(center_x, center_z))
            yield instance, instance_seed


__all__ = ["WorldDistributedNetwork", "iter_world_network_instances"]
