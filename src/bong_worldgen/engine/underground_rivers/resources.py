"""地下河洞壁上的矿脉与湿生植物刷新点。

输出是带逻辑 ``resource_id`` 的 ``UndergroundBlock``；Server 不必从通用
Minecraft 材质反推来源。矿脉和植物都必须附着在河道
空腔旁的实心墙格，植物额外要求邻近水方块，避免悬空或出现在干燥岩层
深处。

算法只参考公开思路，不复制代码：

* FastNoiseLite 的 seed/deterministic procedural 设计：
  https://github.com/Auburn/FastNoiseLite
* Godot Voxel 的洞壁/空腔生成思路：
  https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
* pyKasso 的 conduit network 与节点/边结果：
  https://github.com/randlab/pyKasso
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..constants import CAVE_RARITY_MULTIPLIERS, SPAN_MIN_Y
from ..terrain_config import Point
from ..underground_config import (
    UndergroundBlock,
    UndergroundRiverNetwork,
    UndergroundWaterBlock,
)
from ..randomness import unit_interval as _unit


_NEIGHBOURS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _nearest_index(axis: np.ndarray, value: float) -> int | None:
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
    paths: Iterable[tuple[Point, ...]],
    count: int,
    seed: int,
) -> tuple[tuple[Point, int], ...]:
    """沿 inlet-outlet graph 边采样候选点，结果跨进程稳定。"""

    path_list = tuple(path for path in paths if len(path) >= 2)
    if not path_list:
        return ()
    result: list[tuple[Point, int]] = []
    for index in range(count):
        path = path_list[int(_unit(seed, index * 5) * len(path_list)) % len(path_list)]
        segment = min(int(_unit(seed, index * 5 + 1) * (len(path) - 1)), len(path) - 2)
        amount = 0.12 + 0.76 * _unit(seed, index * 5 + 2)
        start = path[segment]
        end = path[segment + 1]
        result.append(
            (
                Point(
                    start.x + (end.x - start.x) * amount,
                    start.z + (end.z - start.z) * amount,
                ),
                index,
            )
        )
    return tuple(result)


def _offset_index(offsets: np.ndarray, offset: int) -> int | None:
    index = int(np.searchsorted(offsets, offset))
    if index >= offsets.size or int(offsets[index]) != offset:
        return None
    return index


def _is_void_world(
    void: np.ndarray,
    offsets: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    surface: np.ndarray,
    world_x: int,
    world_y: int,
    world_z: int,
) -> bool:
    x_index = _nearest_index(x_axis, world_x)
    z_index = _nearest_index(z_axis, world_z)
    if x_index is None or z_index is None:
        return False
    offset = world_y - int(round(float(surface[z_index, x_index])))
    offset_index = _offset_index(offsets, offset)
    return bool(
        offset_index is not None
        and not (world_y <= SPAN_MIN_Y)
        and void[offset_index, z_index, x_index]
    )


def _has_void_neighbor(
    void: np.ndarray,
    offsets: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    surface: np.ndarray,
    world_x: int,
    world_y: int,
    world_z: int,
) -> bool:
    return any(
        _is_void_world(
            void,
            offsets,
            x_axis,
            z_axis,
            surface,
            world_x + dx,
            world_y + dy,
            world_z + dz,
        )
        for dx, dy, dz in _NEIGHBOURS
    )


def _find_wall_cell(
    void: np.ndarray,
    offsets: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    surface: np.ndarray,
    point: Point,
    water_positions: set[tuple[int, int, int]],
    *,
    require_water: bool,
) -> tuple[int, int, int] | None:
    """找河腔旁的实心格；植物候选还必须与水相邻。"""

    center_x = _nearest_index(x_axis, point.x)
    center_z = _nearest_index(z_axis, point.z)
    if center_x is None or center_z is None:
        return None
    candidate_offsets = np.flatnonzero(void[:, center_z, center_x])
    if not candidate_offsets.size:
        # 河道中心可能刚好落在 tile 的边界或湖水覆盖区，
        # 附近几格作为回退。
        for radius in range(1, 5):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x_index = center_x + dx
                    z_index = center_z + dz
                    if 0 <= x_index < x_axis.size and 0 <= z_index < z_axis.size:
                        candidate_offsets = np.flatnonzero(void[:, z_index, x_index])
                        if candidate_offsets.size:
                            center_x, center_z = x_index, z_index
                            break
                if candidate_offsets.size:
                    break
            if candidate_offsets.size:
                break
    for offset_index in candidate_offsets:
        cavity_y = int(round(float(surface[center_z, center_x]))) + int(offsets[offset_index])
        cavity_x = int(round(float(x_axis[center_x])))
        cavity_z = int(round(float(z_axis[center_z])))
        for dx, dy, dz in _NEIGHBOURS:
            wall = (cavity_x + dx, cavity_y + dy, cavity_z + dz)
            if wall in water_positions:
                continue
            if _is_void_world(
                void,
                offsets,
                x_axis,
                z_axis,
                surface,
                *wall,
            ):
                continue
            if not _has_void_neighbor(
                void,
                offsets,
                x_axis,
                z_axis,
                surface,
                *wall,
            ):
                continue
            if require_water and not any(
                (wall[0] + ndx, wall[1] + ndy, wall[2] + ndz) in water_positions
                for ndx, ndy, ndz in _NEIGHBOURS
            ):
                continue
            return wall
    return None


def _grow_cluster(
    start: tuple[int, int, int],
    length: int,
    seed: int,
    void: np.ndarray,
    offsets: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    surface: np.ndarray,
    water_positions: set[tuple[int, int, int]],
    *,
    require_water: bool,
) -> tuple[tuple[int, int, int], ...]:
    """在洞壁表面做确定性 6 邻域扩展，生成相连的矿脉/植物簇。"""

    frontier = [start]
    visited: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    while frontier and len(result) < length:
        current = frontier.pop(0)
        if current in visited or current in water_positions:
            continue
        visited.add(current)
        if current[1] <= SPAN_MIN_Y or _is_void_world(
            void,
            offsets,
            x_axis,
            z_axis,
            surface,
            *current,
        ):
            continue
        if not _has_void_neighbor(
            void,
            offsets,
            x_axis,
            z_axis,
            surface,
            *current,
        ):
            continue
        if require_water and not any(
            (current[0] + dx, current[1] + dy, current[2] + dz) in water_positions
            for dx, dy, dz in _NEIGHBOURS
        ):
            continue
        result.append(current)
        neighbours = [
            (current[0] + dx, current[1] + dy, current[2] + dz)
            for dx, dy, dz in _NEIGHBOURS
        ]
        neighbours.sort(
            key=lambda value: _unit(
                seed,
                value[0] * 7_919 + value[1] * 104_729 + value[2] * 1_009,
            ),
            reverse=True,
        )
        frontier.extend(neighbours)
    return tuple(result)


def _cluster_counts(
    total: int,
    density: str,
    rarities: tuple[str, ...],
) -> tuple[int, ...]:
    if total <= 0 or not rarities:
        return tuple(0 for _ in rarities)
    base = max(1, round(total * CAVE_RARITY_MULTIPLIERS[density]))
    weights = [CAVE_RARITY_MULTIPLIERS[rarity] for rarity in rarities]
    weight_sum = sum(weights)
    return tuple(max(1, round(base * weight / weight_sum)) for weight in weights)


def _cluster_length(rarity: str, seed: int) -> int:
    if rarity == "少":
        return 2 + int(_unit(seed, 3) * 3)
    if rarity == "多":
        return 6 + int(_unit(seed, 3) * 6)
    return 4 + int(_unit(seed, 3) * 4)


def _append_unique(
    result: dict[tuple[int, int, int], UndergroundBlock],
    cells: tuple[tuple[int, int, int], ...],
    material: str,
    *,
    resource_id: str,
    rarity: str,
) -> None:
    for x, y, z in cells:
        if y <= SPAN_MIN_Y:
            continue
        result.setdefault(
            (x, y, z),
            UndergroundBlock(
                x=x,
                y=y,
                z=z,
                material=f"minecraft:{material}",
                resource_id=resource_id,
                source="underground_river",
                rarity=rarity,
            ),
        )


def generate_river_resource_blocks(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    offsets: np.ndarray,
    void: np.ndarray,
    water_blocks: tuple[UndergroundWaterBlock, ...],
    paths: tuple[tuple[Point, ...], ...],
    network: UndergroundRiverNetwork,
    seed: int,
) -> tuple[UndergroundBlock, ...]:
    """生成地下河矿脉和植物，供 Server 识别并替换真实产物。"""

    if not paths or (network.ore_cluster_count <= 0 and network.plant_cluster_count <= 0):
        return ()
    x_axis = x[0, :]
    z_axis = z[:, 0]
    surface = np.floor(terrain).astype(np.int16)
    water_positions = {(block.x, block.y, block.z) for block in water_blocks}
    result: dict[tuple[int, int, int], UndergroundBlock] = {}
    categories = (
        (network.ore_cluster_count, network.ore_materials, network.ore_rarities, False, 31_001),
        (
            network.plant_cluster_count,
            network.plant_materials,
            network.plant_rarities,
            True,
            47_003,
        ),
    )
    cluster_index = 0
    for total, materials, rarities, require_water, category_seed in categories:
        counts = _cluster_counts(total, network.resource_density, rarities)
        for material_index, (material, rarity, count) in enumerate(
            zip(materials, rarities, counts)
        ):
            for local_index in range(count):
                candidate_seed = (
                    seed + category_seed + material_index * 73_921 + local_index * 1_009
                )
                candidates = _path_candidates(paths, max(10, count * 8), candidate_seed)
                for point, _candidate_index in candidates:
                    wall = _find_wall_cell(
                        void,
                        offsets,
                        x_axis,
                        z_axis,
                        surface,
                        point,
                        water_positions,
                        require_water=require_water,
                    )
                    if wall is None:
                        continue
                    cells = _grow_cluster(
                        wall,
                        _cluster_length(rarity, candidate_seed + cluster_index),
                        candidate_seed + cluster_index * 31,
                        void,
                        offsets,
                        x_axis,
                        z_axis,
                        surface,
                        water_positions,
                        require_water=require_water,
                    )
                    if not cells:
                        continue
                    _append_unique(
                        result,
                        cells,
                        material,
                        resource_id=f"underground_river:{material}",
                        rarity=rarity,
                    )
                    cluster_index += 1
                    break
    return tuple(result.values())


__all__ = ["generate_river_resource_blocks"]
