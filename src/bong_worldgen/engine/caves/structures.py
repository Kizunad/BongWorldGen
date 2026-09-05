"""洞穴天然内容的确定性占位方块生成。

这里输出的不是抽象素材 ID，而是可以直接写入 Anvil 的真实 Minecraft 方块名。
Server 读取这些方块后，再按 ``resource_id/source/rarity`` 刷新真正的矿物或植物产物；
``material`` 只用于预览和 Anvil 占位。

洞穴域采样沿用本项目的 seeded 设计，参考：
https://github.com/Auburn/FastNoiseLite
https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from ..constants import CAVE_RARITY_MULTIPLIERS, SPAN_MIN_Y
from ..randomness import unit_interval as _unit
from ..terrain_config import Point
from ..underground_config import CaveNetwork, UndergroundBlock
from .topology import CaveTopology


def _nearest_index(axis: np.ndarray, value: float) -> int | None:
    """返回规则世界轴上离 value 最近的列索引。"""

    if axis.size == 0 or value < axis[0] or value > axis[-1]:
        return None
    right = int(np.searchsorted(axis, value, side="left"))
    if right == 0:
        return 0
    if right == axis.size:
        return axis.size - 1
    left = right - 1
    return (
        left
        if abs(float(axis[left]) - value) <= abs(float(axis[right]) - value)
        else right
    )


def _path_candidates(
    paths: Iterable[tuple[Point, ...]], count: int, seed: int
) -> list[tuple[Point, float, float]]:
    """从主洞和支洞上取确定性采样点，并返回点及其切线方向。"""

    path_list = tuple(paths)
    candidates: list[tuple[Point, float, float]] = []
    if not path_list:
        return candidates
    for index in range(count):
        path = path_list[int(_unit(seed, index * 5) * len(path_list)) % len(path_list)]
        segment = min(int(_unit(seed, index * 5 + 1) * (len(path) - 1)), len(path) - 2)
        amount = 0.2 + 0.6 * _unit(seed, index * 5 + 2)
        start = path[segment]
        end = path[segment + 1]
        point = Point(
            start.x + (end.x - start.x) * amount,
            start.z + (end.z - start.z) * amount,
        )
        dx = end.x - start.x
        dz = end.z - start.z
        length = math.hypot(dx, dz) or 1.0
        candidates.append((point, dx / length, dz / length))
    return candidates


def _is_void(
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
    x_index: int,
    z_index: int,
    offset: int,
) -> bool:
    offset_index = int(np.searchsorted(cave_offsets, offset))
    if (
        offset_index >= cave_offsets.size
        or int(cave_offsets[offset_index]) != offset
        or not (0 <= z_index < cave_void.shape[1] and 0 <= x_index < cave_void.shape[2])
    ):
        return False
    return bool(cave_void[offset_index, z_index, x_index])


def _append_unique(
    result: dict[tuple[int, int, int], UndergroundBlock],
    x: int,
    y: int,
    z: int,
    material: str,
    *,
    resource_id: str,
    rarity: str,
) -> None:
    if y <= SPAN_MIN_Y:
        return
    result.setdefault(
        (x, y, z),
        UndergroundBlock(
            x=x,
            y=y,
            z=z,
            material=f"minecraft:{material}",
            resource_id=resource_id,
            source="cave",
            rarity=rarity,
        ),
    )


def _find_wall_cell(
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
    x_index: int,
    z_index: int,
    offset: int,
) -> tuple[int, int, int] | None:
    """在候选洞腔附近找一对“空腔 + 相邻实心墙”列。"""

    height, width = cave_void.shape[1:]
    # 目标层不存在或该层恰好没有洞壁时，沿垂直方向优先尝试最近层。
    # 这避免资源簇因为一个随机 offset 落在洞腔中心而整簇消失。
    offset_order = sorted(
        (int(value) for value in cave_offsets),
        key=lambda value: (abs(value - offset), value),
    )
    # 先找候选中心附近的空腔，再沿四个方向向外取一格实心墙。
    for target_offset in offset_order:
        offset_index = int(np.searchsorted(cave_offsets, target_offset))
        for radius in range(0, 13):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dz)) != radius:
                        continue
                    inner_x = x_index + dx
                    inner_z = z_index + dz
                    if not (0 <= inner_x < width and 0 <= inner_z < height):
                        continue
                    if not cave_void[offset_index, inner_z, inner_x]:
                        continue
                    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
                    for step_x, step_z in directions:
                        wall_x = inner_x + step_x
                        wall_z = inner_z + step_z
                        if not (0 <= wall_x < width and 0 <= wall_z < height):
                            continue
                        if not cave_void[offset_index, wall_z, wall_x]:
                            return wall_x, wall_z, target_offset
    return None


def _wall_has_cave_neighbor(
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
    x_index: int,
    z_index: int,
    offset: int,
) -> bool:
    """判断实心墙格是否至少有一个 6 邻域洞腔面。"""

    return any(
        _is_void(
            cave_void,
            cave_offsets,
            x_index + dx,
            z_index + dz,
            offset + dy,
        )
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    )


def _find_world_vein_cells(
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    crop_surface: np.ndarray,
    start: tuple[int, int, int],
    length: int,
    seed: int,
    *,
    require_cave_neighbor: bool = True,
) -> tuple[tuple[int, int, int], ...]:
    """按真实世界坐标沿 6 邻域扩展矿脉，避免坡面把矿脉错连成斜线。"""

    start_x, start_z, start_offset = start
    if not (0 <= start_x < x_axis.size and 0 <= start_z < z_axis.size):
        return ()
    world_start_x = int(round(float(x_axis[start_x])))
    world_start_z = int(round(float(z_axis[start_z])))
    start_y = int(round(float(crop_surface[start_z, start_x]))) + start_offset
    height, width = cave_void.shape[1:]
    frontier = [(world_start_x, start_y, world_start_z)]
    visited: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    while frontier and len(result) < length:
        current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        world_x, world_y, world_z = current
        x_index = _nearest_index(x_axis, world_x)
        z_index = _nearest_index(z_axis, world_z)
        if x_index is None or z_index is None:
            continue
        current_offset = world_y - int(round(float(crop_surface[z_index, x_index])))
        offset_index = int(np.searchsorted(cave_offsets, current_offset))
        if not (
            0 <= x_index < width
            and 0 <= z_index < height
            and offset_index < cave_offsets.size
            and int(cave_offsets[offset_index]) == current_offset
            and not cave_void[offset_index, z_index, x_index]
            and (
                not require_cave_neighbor
                or _wall_has_cave_neighbor(
                    cave_void,
                    cave_offsets,
                    x_index,
                    z_index,
                    current_offset,
                )
            )
        ):
            continue
        result.append(current)
        neighbours = [
            (world_x + dx, world_y + dy, world_z + dz)
            for dx, dy, dz in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            )
        ]
        neighbours.sort(
            key=lambda value: _unit(
                seed,
                value[0] * 7_919 + value[1] * 104_729 + value[2] * 1_009,
            ),
            reverse=True,
        )
        frontier.extend(neighbours[:3])
    return tuple(result)


def _resource_cluster_counts(network: CaveNetwork) -> tuple[int, ...]:
    """按全局密度和每种资源稀有度分配矿脉/植物簇数量。"""

    if network.placeholder_count <= 0 or not network.placeholder_materials:
        return ()
    base_count = max(
        1,
        round(network.placeholder_count * CAVE_RARITY_MULTIPLIERS[network.placeholder_density]),
    )
    weights = [CAVE_RARITY_MULTIPLIERS[rarity] for rarity in network.placeholder_rarities]
    total_weight = sum(weights)
    return tuple(
        max(1, round(base_count * weight / total_weight))
        for weight in weights
    )


def generate_placeholder_blocks(
    network: CaveNetwork,
    topology: CaveTopology,
    crop_x: np.ndarray,
    crop_z: np.ndarray,
    crop_surface: np.ndarray,
    cave_offsets: np.ndarray,
    cave_void: np.ndarray,
    seed: int,
) -> tuple[UndergroundBlock, ...]:
    """为已确认的洞穴空腔生成真实方块刷新点。

    矿石和荧光地衣都附着在洞壁（目标列为实心、朝洞腔一侧的邻列为空腔）。
    候选点如果没有满足几何约束就跳过，不会生成悬空方块或穿出地表的方块。
    """

    materials = network.placeholder_materials
    if network.placeholder_count <= 0 or not materials:
        return ()
    x_axis = crop_x[0, :]
    z_axis = crop_z[:, 0]
    result: dict[tuple[int, int, int], UndergroundBlock] = {}
    cluster_index = 0
    for material_index, (material, cluster_count) in enumerate(
        zip(materials, _resource_cluster_counts(network))
    ):
        rarity = network.placeholder_rarities[material_index]
        for local_index in range(cluster_count):
            candidate_seed = seed + material_index * 73_921 + local_index * 1_009
            # 一个簇不只试一个点：路径边界、洞腔中心和随机层都可能让单点
            # 不具备洞壁条件。候选回退仍由 seed 决定，不改变可复现性。
            candidates = _path_candidates(
                topology.paths,
                max(8, cluster_count * 6),
                candidate_seed,
            )
            if not candidates:
                continue
            if rarity == "少":
                vein_length = 2 + int(_unit(candidate_seed, 5) * 2)
            elif rarity == "中":
                vein_length = 3 + int(_unit(candidate_seed, 5) * 3)
            else:
                vein_length = 5 + int(_unit(candidate_seed, 5) * 5)
            placed = False
            for candidate_index, (point, _tangent_x, _tangent_z) in enumerate(candidates):
                x_index = _nearest_index(x_axis, point.x)
                z_index = _nearest_index(z_axis, point.z)
                if x_index is None or z_index is None:
                    continue
                offset = int(
                    round(
                        -network.depth
                        + (_unit(candidate_seed, 4 + candidate_index) - 0.5)
                        * network.height
                    )
                )
                wall = _find_wall_cell(
                    cave_void,
                    cave_offsets,
                    x_index,
                    z_index,
                    offset,
                )
                if wall is None:
                    continue
                before = len(result)
                for world_x, world_y, world_z in _find_world_vein_cells(
                    cave_void,
                    cave_offsets,
                    x_axis,
                    z_axis,
                    crop_surface,
                    wall,
                    vein_length,
                    candidate_seed + cluster_index * 31 + candidate_index,
                    require_cave_neighbor=material == "glow_lichen",
                ):
                    _append_unique(
                        result,
                        world_x,
                        world_y,
                        world_z,
                        material,
                        resource_id=f"cave:{material}",
                        rarity=rarity,
                    )
                if len(result) > before:
                    placed = True
                    cluster_index += 1
                    break
            if not placed:
                # 没有可见洞壁时保持确定性跳过，不向地表或洞腔空气中硬塞方块。
                continue
    return tuple(result.values())


__all__ = ["generate_placeholder_blocks"]
