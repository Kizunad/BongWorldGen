"""实心石层中的普通矿脉生成。

普通矿物和洞穴内容是两个不同的生成层：这里的候选点必须落在实心段内，
因此不会把洞穴墙壁上的刷新点误当成普通矿脉。位置由世界坐标、seed 和
分区索引共同决定，跨 tile 生成时仍保持一致。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .constants import CAVE_RARITY_MULTIPLIERS, SPAN_MIN_Y
from .underground_config import SolidOreSpec, UndergroundBlock, UndergroundWaterBlock
from .randomness import mix64, unit_interval


_ORE_REGION_SIZE = 512.0
_CELL_X_SALT = 0xBF58476D
_CELL_Z_SALT = 0x94D049BB
_SPEC_SALT = 0x9E3779B9


def _nearest_axis_index(axis: np.ndarray, value: float) -> int | None:
    """返回世界轴上离坐标最近的索引；越界时返回 None。"""

    if axis.size == 0:
        return None
    if axis.size == 1:
        return 0 if abs(value - float(axis[0])) <= 0.5 else None
    if value < float(axis[0]) - 0.5 or value > float(axis[-1]) + 0.5:
        return None
    spacing = max(float(axis[1] - axis[0]), 1.0e-9)
    index = int(np.rint((value - float(axis[0])) / spacing))
    return index if 0 <= index < axis.size else None


def _is_void(
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
    surface: np.ndarray,
    x_index: int,
    z_index: int,
    world_y: int,
) -> bool:
    """判断一个世界坐标是否落在已生成的地下空腔中。"""

    offset = world_y - int(surface[z_index, x_index])
    offset_index = int(np.searchsorted(cave_offsets, offset))
    return bool(
        0 <= offset_index < cave_offsets.size
        and int(cave_offsets[offset_index]) == offset
        and cave_void[offset_index, z_index, x_index]
    )


def _cluster_seed(seed: int, spec_index: int, region_x: int, region_z: int, cluster: int) -> int:
    """为矿脉簇派生不依赖 tile 边界的子 seed。"""

    return mix64(
        seed
        + spec_index * _SPEC_SALT
        + region_x * _CELL_X_SALT
        + region_z * _CELL_Z_SALT
        + cluster * 0xD6E8FEB9
    )


def generate_solid_ore_blocks(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cave_offsets: np.ndarray,
    cave_void: np.ndarray,
    specs: tuple[SolidOreSpec, ...],
    seed: int,
    *,
    occupied_blocks: Iterable[UndergroundBlock] = (),
    water_blocks: Iterable[UndergroundWaterBlock] = (),
) -> tuple[UndergroundBlock, ...]:
    """在空腔之外的实心石层中生成确定性的普通矿脉。"""

    if not specs:
        return ()
    surface = np.rint(terrain).astype(np.int32)
    x_axis = np.asarray(x[0, :], dtype=np.float64)
    z_axis = np.asarray(z[:, 0], dtype=np.float64)
    cell_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    cell_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    min_region_x = math.floor(float(x_axis[0]) / _ORE_REGION_SIZE) - 1
    max_region_x = math.floor(float(x_axis[-1]) / _ORE_REGION_SIZE) + 1
    min_region_z = math.floor(float(z_axis[0]) / _ORE_REGION_SIZE) - 1
    max_region_z = math.floor(float(z_axis[-1]) / _ORE_REGION_SIZE) + 1
    occupied = {(block.x, block.y, block.z) for block in occupied_blocks}
    occupied.update((block.x, block.y, block.z) for block in water_blocks)
    result: dict[tuple[int, int, int], UndergroundBlock] = {}

    for spec_index, spec in enumerate(specs):
        rarity_multiplier = CAVE_RARITY_MULTIPLIERS[spec.rarity]
        cluster_count = max(0, round(spec.cluster_count * rarity_multiplier))
        if cluster_count == 0:
            continue
        depth_span = spec.max_depth - spec.min_depth
        for region_z in range(min_region_z, max_region_z + 1):
            for region_x in range(min_region_x, max_region_x + 1):
                region_origin_x = region_x * _ORE_REGION_SIZE
                region_origin_z = region_z * _ORE_REGION_SIZE
                for cluster_index in range(cluster_count):
                    cluster_seed = _cluster_seed(
                        seed,
                        spec_index,
                        region_x,
                        region_z,
                        cluster_index,
                    )
                    center_x = region_origin_x + unit_interval(cluster_seed, 1) * _ORE_REGION_SIZE
                    center_z = region_origin_z + unit_interval(cluster_seed, 2) * _ORE_REGION_SIZE
                    depth = spec.min_depth + unit_interval(cluster_seed, 3) * depth_span
                    angle = unit_interval(cluster_seed, 4) * math.tau
                    vertical_bias = (unit_interval(cluster_seed, 5) - 0.5) * 0.8
                    current_x = center_x
                    current_z = center_z
                    base_depth = depth
                    for vein_index in range(spec.vein_length):
                        if vein_index:
                            angle += (unit_interval(cluster_seed, 10 + vein_index) - 0.5) * 1.15
                            current_x += math.cos(angle) * cell_x
                            current_z += math.sin(angle) * cell_z
                        x_index = _nearest_axis_index(x_axis, current_x)
                        z_index = _nearest_axis_index(z_axis, current_z)
                        if x_index is None or z_index is None:
                            continue
                        local_depth = base_depth + (
                            unit_interval(cluster_seed, 40 + vein_index) - 0.5
                        ) * min(2.0, max(depth_span * 0.12, 0.5))
                        local_depth += vertical_bias * vein_index
                        world_y = int(surface[z_index, x_index] - round(local_depth))
                        position = (int(round(x_axis[x_index])), world_y, int(round(z_axis[z_index])))
                        if (
                            world_y <= SPAN_MIN_Y
                            or world_y >= int(surface[z_index, x_index])
                            or position in occupied
                            or position in result
                            or _is_void(cave_void, cave_offsets, surface, x_index, z_index, world_y)
                        ):
                            continue
                        result[position] = UndergroundBlock(
                            x=position[0],
                            y=position[1],
                            z=position[2],
                            material=spec.material,
                            resource_id=f"solid:{spec.material}",
                            source="solid_ore",
                            rarity=spec.rarity,
                        )
    return tuple(result.values())


__all__ = ["generate_solid_ore_blocks"]
