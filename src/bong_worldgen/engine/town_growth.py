"""核心城区与外围部落的增量生长布局。

这里仅返回布局记录，不写 Minecraft 方块。核心是 Weighted Eden Growth：从
已有建筑边缘选择 frontier，按地形、邻近度和随机势场提出候选，再接受有效
候选。这个方向参考了 Minecraft 聚落案例 Tome 的地块/道路分离思路：
https://github.com/Jandhi/Tome

道路只在本模块返回少量吸引点；实际寻路与方块发射由 ``town.py`` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .randomness import stable_text_seed, unit_interval
from .structures import SchematicStructure
from .terrain_config import Point, TownSettings
from .town_footprints import WallFootprint, dry_footprint


@dataclass(frozen=True)
class GrowthBuilding:
    """一个已接受的房屋候选及其真实旋转 footprint 与城镇分区。"""

    center: Point
    size_x: int
    size_z: int
    ground: int
    schematic: SchematicStructure | None
    rotation: int
    district: str


@dataclass(frozen=True)
class GrowthLayout:
    """布局阶段的纯数据输出。"""

    attractors: tuple[Point, ...]
    core_buildings: tuple[GrowthBuilding, ...]
    buildings: tuple[GrowthBuilding, ...]
    tree_centers: tuple[tuple[Point, SchematicStructure, int, int], ...]


def building_footprint_bounds(
    buildings: tuple[GrowthBuilding, ...] | list[GrowthBuilding],
    margin: int = 0,
) -> tuple[int, int, int, int] | None:
    """返回围墙包住建筑所需的 X/Z 闭区间。"""

    if not buildings:
        return None
    footprints = [_footprint(item.center, item.size_x, item.size_z) for item in buildings]
    return (
        min(bounds[0] for bounds in footprints) - margin,
        max(bounds[1] for bounds in footprints) + margin,
        min(bounds[2] for bounds in footprints) - margin,
        max(bounds[3] for bounds in footprints) + margin,
    )


def wall_utilization(
    buildings: tuple[GrowthBuilding, ...] | list[GrowthBuilding],
) -> float:
    """计算真实建筑占地在核心可用矩形中的比例。

    城墙的净空属于道路和防御通道，不应稀释核心建筑区的利用率；发射围墙时
    仍由 ``core_wall_margin`` 在外围预留对应空间。
    """

    bounds = building_footprint_bounds(buildings)
    if bounds is None:
        return 0.0
    min_x, max_x, min_z, max_z = bounds
    enclosure_area = (max_x - min_x + 1) * (max_z - min_z + 1)
    footprint_area = sum(item.size_x * item.size_z for item in buildings)
    return footprint_area / max(enclosure_area, 1)


def _structure_size(
    schematic: SchematicStructure | None,
    rotation: int,
    settings: TownSettings,
    seed: int,
    index: int,
) -> tuple[int, int, SchematicStructure | None, int]:
    if schematic is not None:
        rotated = schematic.rotated(rotation)
        return rotated.width, rotated.length, schematic, rotation
    size_x = settings.house_min_size + int(
        unit_interval(seed, index * 11 + 2)
        * (settings.house_max_size - settings.house_min_size + 1)
    )
    size_z = settings.house_min_size + int(
        unit_interval(seed, index * 11 + 3)
        * (settings.house_max_size - settings.house_min_size + 1)
    )
    return size_x, size_z, None, 0


def _choose_schematic(
    schematics: tuple[SchematicStructure, ...],
    seed: int,
    index: int,
) -> tuple[SchematicStructure | None, int]:
    if not schematics:
        return None, 0
    # 面积越大的模板越少出现，避免一栋巨型建筑吞掉整个部落。
    weights = np.asarray(
        [1.0 / math.sqrt(max(item.width * item.length, 1)) for item in schematics],
        dtype=np.float64,
    )
    target = unit_interval(seed, index * 11 + 1) * float(weights.sum())
    chosen = len(schematics) - 1
    for item_index, weight in enumerate(weights):
        target -= float(weight)
        if target <= 0.0:
            chosen = item_index
            break
    rotation = min(int(unit_interval(seed, index * 11 + 4) * 4), 3)
    return schematics[chosen], rotation


def _choose_core_infill_schematic(
    schematics: tuple[SchematicStructure, ...],
    seed: int,
    index: int,
) -> tuple[SchematicStructure | None, int]:
    """优先选择小型模板填补核心区，避免大模板不断撑大矩形城墙。"""

    if not schematics:
        return None, 0
    # 核心补位的职责是压实现有街区。面积的平方反比会显著偏向小屋、摊位
    # 与井，但仍保留较大模板偶发出现，避免核心变成规则化的小方块阵列。
    weights = np.asarray(
        [1.0 / max(item.width * item.length, 1) ** 2 for item in schematics],
        dtype=np.float64,
    )
    target = unit_interval(seed, index * 11 + 1) * float(weights.sum())
    chosen = len(schematics) - 1
    for item_index, weight in enumerate(weights):
        target -= float(weight)
        if target <= 0.0:
            chosen = item_index
            break
    rotation = min(int(unit_interval(seed, index * 11 + 4) * 4), 3)
    return schematics[chosen], rotation


def _choose_core_schematic(
    schematics: tuple[SchematicStructure, ...],
    seed: int,
) -> tuple[SchematicStructure | None, int]:
    """为部落中心稳定选择一栋 Spawn 模板。"""

    if not schematics:
        return None, 0
    weights = np.asarray(
        [1.0 / math.sqrt(max(item.width * item.length, 1)) for item in schematics],
        dtype=np.float64,
    )
    target = unit_interval(seed, 7_901) * float(weights.sum())
    chosen = len(schematics) - 1
    for index, weight in enumerate(weights):
        target -= float(weight)
        if target <= 0.0:
            chosen = index
            break
    rotation = min(int(unit_interval(seed, 7_902) * 4), 3)
    return schematics[chosen], rotation


def _nearest_cell(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    world_x: float,
    world_z: float,
) -> tuple[int, int] | None:
    if not (x_axis[0] <= world_x <= x_axis[-1] and z_axis[0] <= world_z <= z_axis[-1]):
        return None
    step_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    step_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    column = int(np.clip(np.rint((world_x - x_axis[0]) / step_x), 0, x_axis.size - 1))
    row = int(np.clip(np.rint((world_z - z_axis[0]) / step_z), 0, z_axis.size - 1))
    return row, column


def _footprint(
    center: Point,
    size_x: int,
    size_z: int,
) -> tuple[int, int, int, int]:
    half_x = size_x // 2
    half_z = size_z // 2
    left = int(round(center.x)) - half_x
    top = int(round(center.z)) - half_z
    return left, left + size_x - 1, top, top + size_z - 1


def _overlaps(
    center: Point,
    size_x: int,
    size_z: int,
    buildings: list[GrowthBuilding],
    gap: int,
) -> bool:
    left, right, top, bottom = _footprint(center, size_x, size_z)
    for other in buildings:
        other_left, other_right, other_top, other_bottom = _footprint(
            other.center, other.size_x, other.size_z
        )
        if not (
            right + gap < other_left
            or other_right + gap < left
            or bottom + gap < other_top
            or other_bottom + gap < top
        ):
            return True
    return False


def _suitable(
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    center: Point,
    size_x: int,
    size_z: int,
    settings: TownSettings,
    buildings: list[GrowthBuilding],
) -> tuple[bool, int, float]:
    left, right, top, bottom = _footprint(center, size_x, size_z)
    if not dry_footprint(water, x_axis, z_axis, (left, right, top, bottom)):
        return False, 0, -math.inf
    sample_x = np.arange(left, right + 1, dtype=np.float64)
    sample_z = np.arange(top, bottom + 1, dtype=np.float64)
    step_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    step_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    columns = np.clip(np.rint((sample_x - x_axis[0]) / step_x), 0, x_axis.size - 1).astype(int)
    rows = np.clip(np.rint((sample_z - z_axis[0]) / step_z), 0, z_axis.size - 1).astype(int)
    heights = terrain[np.ix_(rows, columns)]
    if not np.all(np.isfinite(heights)):
        return False, 0, -math.inf
    relief = float(np.ptp(heights))
    if relief > settings.maximum_relief:
        return False, 0, -math.inf
    slope_x = np.diff(heights, axis=1)
    slope_z = np.diff(heights, axis=0)
    slope = max(
        float(np.percentile(np.abs(slope_x), 90)) if slope_x.size else 0.0,
        float(np.percentile(np.abs(slope_z), 90)) if slope_z.size else 0.0,
    )
    if slope > settings.maximum_slope:
        return False, 0, -math.inf
    ground = int(round(float(np.median(heights))))
    # 平坦且接近已有建筑的候选优先，但不强制形成规则环。
    nearest = min(
        (math.hypot(center.x - item.center.x, center.z - item.center.z) for item in buildings),
        default=settings.radius,
    )
    score = -nearest - relief * 2.0 - slope * 10.0
    return True, ground, score


def _projected_utilization(
    buildings: list[GrowthBuilding],
    center: Point,
    size_x: int,
    size_z: int,
) -> float:
    """估算接受候选后的围墙利用率，用于优先压实外围轮廓。"""

    candidate = GrowthBuilding(center, size_x, size_z, 0, None, 0, "core")
    return wall_utilization([*buildings, candidate])


def _fits_walls(
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    buildings: list[GrowthBuilding],
    center: Point,
    size_x: int,
    size_z: int,
    walls: WallFootprint | None,
) -> bool:
    """核心候选还必须为对齐后的围墙、城门和角塔留出完整干地。"""

    if walls is None:
        return True
    candidate = GrowthBuilding(center, size_x, size_z, 0, None, 0, "core")
    bounds = building_footprint_bounds([*buildings, candidate])
    return all(
        dry_footprint(water, x_axis, z_axis, footprint)
        for footprint in walls.footprints(bounds)
    )


def _choose_dense_infill_candidate(
    *,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    buildings: list[GrowthBuilding],
    size_x: int,
    size_z: int,
    seed: int,
    index: int,
    walls: WallFootprint | None = None,
) -> tuple[Point, int] | None:
    """在既有核心边界内填补空位，不能扩大城墙投影。"""

    bounds = building_footprint_bounds(buildings, 0)
    if bounds is None:
        return None
    min_x, max_x, min_z, max_z = bounds
    low_x = min_x + size_x // 2
    high_x = max_x - (size_x - 1 - size_x // 2)
    low_z = min_z + size_z // 2
    high_z = max_z - (size_z - 1 - size_z // 2)
    if low_x > high_x or low_z > high_z:
        return None

    attempts = settings.growth_candidate_count * 4
    base_offset = 80_000 + index * 193
    random_centers = (
        Point(
            float(low_x + int(unit_interval(seed, base_offset + attempt * 7)
                              * (high_x - low_x + 1))),
            float(low_z + int(unit_interval(seed, base_offset + attempt * 7 + 1)
                              * (high_z - low_z + 1))),
        )
        for attempt in range(attempts)
    )
    # 窄空隙可能只剩一个合法中心，随机抽点很容易全部漏过。后备候选贴齐
    # 既有建筑边缘或核心边界，仍使用同一套碰撞、干地、坡度和围墙检查。
    edge_x = {low_x, high_x}
    edge_z = {low_z, high_z}
    for building in buildings:
        left, right, top, bottom = _footprint(
            building.center, building.size_x, building.size_z
        )
        edge_x.update((left - (size_x + 1) // 2, right + 1 + size_x // 2))
        edge_z.update((top - (size_z + 1) // 2, bottom + 1 + size_z // 2))
    xs = sorted(value for value in edge_x if low_x <= value <= high_x)
    zs = sorted(value for value in edge_z if low_z <= value <= high_z)
    boundary_centers = (Point(float(cx), float(cz)) for cz in zs for cx in xs)

    for candidates in (random_centers, boundary_centers):
        best: tuple[Point, int, float] | None = None
        for attempt, center in enumerate(candidates):
            if _overlaps(center, size_x, size_z, buildings, 0):
                continue
            suitable, ground, terrain_score = _suitable(
                terrain, water, x_axis, z_axis, center, size_x, size_z, settings, buildings
            )
            if not suitable:
                continue
            if not _fits_walls(water, x_axis, z_axis, buildings, center, size_x, size_z, walls):
                continue
            # 随机势场只负责打破同样平坦候选的平局；候选均不扩大围墙。
            score = terrain_score + unit_interval(seed, base_offset + attempt * 7 + 2)
            if best is None or score > best[2]:
                best = (center, ground, score)
        if best is not None:
            return best[0], best[1]
    return None


def _choose_compact_candidate(
    *,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    anchor: Point,
    buildings: list[GrowthBuilding],
    size_x: int,
    size_z: int,
    seed: int,
    index: int,
    attempts: int,
    maximum_radius: float,
    walls: WallFootprint | None = None,
) -> tuple[Point, int] | None:
    """在已有建筑边缘提出候选，并选择最紧凑且地形合格的一处。"""

    # 核心模板可能非常大，不能把它的完整尺寸当作候选中心间距；否则在
    # 小半径出生区内会没有任何合法住宅。核心仍参与 footprint/碰撞检查，
    # 这里只把 anchor 作为一个可生长的点。
    parents: list[Point | GrowthBuilding] = [anchor, *buildings]
    best: tuple[Point, int, float] | None = None
    for attempt in range(attempts):
        offset = index * 193 + attempt * 7
        # 一部分候选直接在现有包络内部采样，用来填补核心建筑与住宅之间
        # 的空位；其余候选仍沿父 footprint 边缘生长，保持自然的不规则轮廓。
        interior_candidate: Point | None = None
        if buildings and attempt % 3 == 0:
            current_bounds = building_footprint_bounds(buildings, 0)
            if current_bounds is not None:
                min_x, max_x, min_z, max_z = current_bounds
                low_x = min_x + (size_x + 1) // 2
                high_x = max_x - size_x // 2
                low_z = min_z + (size_z + 1) // 2
                high_z = max_z - size_z // 2
                if low_x <= high_x and low_z <= high_z:
                    interior_candidate = Point(
                        float(low_x + int(unit_interval(seed, offset + 1) * (high_x - low_x + 1))),
                        float(low_z + int(unit_interval(seed, offset + 2) * (high_z - low_z + 1))),
                    )
        parent = parents[
            min(int(unit_interval(seed, offset) * len(parents)), len(parents) - 1)
        ]
        if interior_candidate is not None:
            candidate = interior_candidate
            anchor_distance = math.hypot(candidate.x - anchor.x, candidate.z - anchor.z)
            if anchor_distance > maximum_radius:
                continue
            if _overlaps(candidate, size_x, size_z, buildings, 0):
                continue
            suitable, ground, terrain_score = _suitable(
                terrain, water, x_axis, z_axis, candidate, size_x, size_z, settings, buildings
            )
            if not suitable:
                continue
            if not _fits_walls(
                water, x_axis, z_axis, buildings, candidate, size_x, size_z, walls
            ):
                continue
            compactness = _projected_utilization(
                buildings, candidate, size_x, size_z
            )
            score = terrain_score + compactness * 1_200.0 - anchor_distance * 0.25
            if best is None or score > best[2]:
                best = (candidate, ground, score)
            continue
        gap = settings.building_gap_min + int(
            unit_interval(seed, offset + 2)
            * (settings.building_gap_max - settings.building_gap_min + 1)
        )
        # 沿父 footprint 的四条边贴边放置。候选仍由 seed 决定边和切向偏移，
        # 但不再用圆形半径推远，因此大模板周围不会形成空旷环带。
        if isinstance(parent, GrowthBuilding):
            parent_left, parent_right, parent_top, parent_bottom = _footprint(
                parent.center, parent.size_x, parent.size_z
            )
            side = int(unit_interval(seed, offset + 1) * 4.0)
            jitter = unit_interval(seed, offset + 2)
            if side < 2:
                center_x = (
                    parent_right + 1 + gap + size_x // 2
                    if side == 0
                    else parent_left - gap - (size_x + 1) // 2
                )
                span = max(parent.size_z - size_z, 0)
                center_z = parent_top + size_z // 2 + int(jitter * (span + 1))
            else:
                center_z = (
                    parent_bottom + 1 + gap + size_z // 2
                    if side == 2
                    else parent_top - gap - (size_z + 1) // 2
                )
                span = max(parent.size_x - size_x, 0)
                center_x = parent_left + size_x // 2 + int(jitter * (span + 1))
            candidate = Point(float(center_x), float(center_z))
        else:
            angle = unit_interval(seed, offset + 1) * math.tau
            distance = max(size_x, size_z) * 0.5 + gap + 0.5
            candidate = Point(
                parent.x + math.cos(angle) * distance,
                parent.z + math.sin(angle) * distance,
            )
        anchor_distance = math.hypot(candidate.x - anchor.x, candidate.z - anchor.z)
        if anchor_distance > maximum_radius:
            continue
        if _overlaps(candidate, size_x, size_z, buildings, gap):
            continue
        suitable, ground, terrain_score = _suitable(
            terrain, water, x_axis, z_axis, candidate, size_x, size_z, settings, buildings
        )
        if not suitable:
            continue
        if not _fits_walls(
            water, x_axis, z_axis, buildings, candidate, size_x, size_z, walls
        ):
            continue
        compactness = _projected_utilization(
            buildings,
            candidate,
            size_x,
            size_z,
        )
        score = terrain_score + compactness * 1_000.0 - anchor_distance * 0.35
        if best is None or score > best[2]:
            best = (candidate, ground, score)
    return None if best is None else (best[0], best[1])


def _outer_cluster_center(
    anchor: Point,
    settings: TownSettings,
    seed: int,
    cluster_index: int,
) -> Point:
    """为外围部落生成稳定、分扇区的聚落簇中心。"""

    sector = math.tau / settings.outer_cluster_count
    angle = sector * (cluster_index + 0.18 + 0.64 * unit_interval(seed, cluster_index * 7 + 1))
    minimum = settings.outer_min_radius + settings.outer_cluster_spread
    maximum = settings.radius - settings.outer_cluster_spread
    if maximum < minimum:
        minimum = settings.outer_min_radius
        maximum = settings.radius
    radius = minimum + (maximum - minimum) * unit_interval(seed, cluster_index * 7 + 2)
    return Point(
        anchor.x + math.cos(angle) * radius,
        anchor.z + math.sin(angle) * radius,
    )


def _choose_outer_candidate(
    *,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    anchor: Point,
    cluster_center: Point,
    buildings: list[GrowthBuilding],
    size_x: int,
    size_z: int,
    seed: int,
    index: int,
) -> tuple[Point, int] | None:
    """在核心墙外为外围部落选取地形合格但彼此疏离的建筑位置。"""

    best: tuple[Point, int, float] | None = None
    attempts = settings.growth_candidate_count * 2
    for attempt in range(attempts):
        offset = 40_000 + index * 193 + attempt * 7
        angle = unit_interval(seed, offset) * math.tau
        # 平方根分布使候选覆盖整个簇，而不是全部堆到簇的中心或边缘。
        distance = settings.outer_cluster_spread * math.sqrt(unit_interval(seed, offset + 1))
        candidate = Point(
            cluster_center.x + math.cos(angle) * distance,
            cluster_center.z + math.sin(angle) * distance,
        )
        radial = math.hypot(candidate.x - anchor.x, candidate.z - anchor.z)
        if radial < settings.outer_min_radius or radial > settings.radius:
            continue
        gap = settings.outer_building_gap_min + int(
            unit_interval(seed, offset + 2)
            * (settings.outer_building_gap_max - settings.outer_building_gap_min + 1)
        )
        if _overlaps(candidate, size_x, size_z, buildings, gap):
            continue
        suitable, ground, terrain_score = _suitable(
            terrain, water, x_axis, z_axis, candidate, size_x, size_z, settings, []
        )
        if not suitable:
            continue
        # 外围只按地形与微弱随机扰动择优，不奖励接近其它建筑，保持稀疏。
        score = terrain_score + unit_interval(seed, offset + 3) * 4.0
        if best is None or score > best[2]:
            best = (candidate, ground, score)
    return None if best is None else (best[0], best[1])


def _attractors(
    anchor: Point,
    settings: TownSettings,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    seed: int,
) -> tuple[Point, ...]:
    points = [anchor]
    for index in range(settings.attractor_count - 1):
        angle = unit_interval(seed, index * 5 + 21) * math.tau
        radius = settings.core_radius * (0.22 + 0.18 * unit_interval(seed, index * 5 + 22))
        candidate = Point(anchor.x + math.cos(angle) * radius, anchor.z + math.sin(angle) * radius)
        cell = _nearest_cell(x_axis, z_axis, candidate.x, candidate.z)
        if cell is None or water[cell] >= 0.0:
            continue
        points.append(candidate)
    return tuple(points)


def _trees(
    settings: TownSettings,
    anchor: Point,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    buildings: list[GrowthBuilding],
    trees: tuple[SchematicStructure, ...],
    seed: int,
) -> tuple[tuple[Point, SchematicStructure, int, int], ...]:
    if not trees or settings.tree_count == 0:
        return ()
    result: list[tuple[Point, SchematicStructure, int, int]] = []
    occupied = list(buildings)
    base_seed = seed + stable_text_seed("settlement_trees")
    for index in range(settings.tree_count):
        chosen = trees[min(int(unit_interval(base_seed, index * 7) * len(trees)), len(trees) - 1)]
        rotation = min(int(unit_interval(base_seed, index * 7 + 1) * 4), 3)
        structure = chosen.rotated(rotation)
        angle = unit_interval(base_seed, index * 7 + 2) * math.tau
        radial = settings.radius * (0.35 + 0.60 * unit_interval(base_seed, index * 7 + 3))
        center = Point(anchor.x + math.cos(angle) * radial, anchor.z + math.sin(angle) * radial)
        if _overlaps(
            center,
            structure.width,
            structure.length,
            occupied,
            settings.tree_clearance,
        ):
            continue
        suitable, ground, _ = _suitable(
            terrain, water, x_axis, z_axis, center, structure.width, structure.length,
            settings, [],
        )
        if suitable:
            result.append((center, chosen, rotation, ground))
            occupied.append(
                GrowthBuilding(center, structure.width, structure.length, ground, chosen, rotation, "tree")
            )
    return tuple(result)


def generate_town_layout(
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    anchor: Point,
    seed: int,
    house_schematics: tuple[SchematicStructure, ...],
    tree_schematics: tuple[SchematicStructure, ...],
    core_schematics: tuple[SchematicStructure, ...] = (),
    walls: WallFootprint | None = None,
) -> GrowthLayout:
    """生成核心城区、外围部落、吸引点和树木位置。"""

    base_seed = seed + stable_text_seed("weighted_eden_settlement")
    attractors = _attractors(anchor, settings, terrain, water, x_axis, z_axis, base_seed)
    core_buildings: list[GrowthBuilding] = []
    core, core_rotation = _choose_core_schematic(core_schematics, base_seed)
    if core is not None:
        rotated_core = core.rotated(core_rotation)
        suitable, core_ground, _ = _suitable(
            terrain, water, x_axis, z_axis, anchor,
            rotated_core.width, rotated_core.length, settings, [],
        )
        # Spawn 模板必须能完整落在核心区内。小型测试城镇与窄小配置不能被
        # 一栋大型模板占满，否则普通住宅没有可用的生长空间。
        core_extent = math.hypot(rotated_core.width / 2.0, rotated_core.length / 2.0)
        if (
            suitable
            and core_extent <= settings.core_radius - settings.core_wall_margin
            and _fits_walls(
                water, x_axis, z_axis, [], anchor,
                rotated_core.width, rotated_core.length, walls,
            )
        ):
            # 核心模板先占位，普通住宅的 footprint 检查会自动避让它。
            core_buildings.append(
                GrowthBuilding(
                    anchor,
                    rotated_core.width,
                    rotated_core.length,
                    core_ground,
                    core,
                    core_rotation,
                    "core",
                )
            )
    # 核心住宅在围墙内部保持高密度；核心模板是额外地标，不消耗住宅配额。
    regular_count = settings.core_house_count
    for index in range(regular_count):
        chosen, rotation = _choose_schematic(house_schematics, base_seed, index)
        size_x, size_z, chosen, rotation = _structure_size(
            chosen, rotation, settings, base_seed, index
        )
        best = _choose_compact_candidate(
            terrain=terrain,
            water=water,
            x_axis=x_axis,
            z_axis=z_axis,
            settings=settings,
            anchor=anchor,
            buildings=core_buildings,
            size_x=size_x,
            size_z=size_z,
            seed=base_seed,
            index=index,
            attempts=settings.growth_candidate_count,
            maximum_radius=settings.core_radius,
            walls=walls,
        )
        if best is None:
            continue
        center, ground = best
        building = GrowthBuilding(center, size_x, size_z, ground, chosen, rotation, "core")
        core_buildings.append(building)

    # 常规数量提供部落规模；围墙利用率再提供空间密度。这里刻意不预建地块，
    # 只在已有建筑边缘补入较小模板，保留自发生长的轮廓同时消除大面积空地。
    for infill_index in range(settings.core_wall_infill_max_buildings):
        if wall_utilization(core_buildings) >= (
            settings.core_wall_minimum_utilization
        ):
            break
        index = regular_count + infill_index
        chosen, rotation = _choose_core_infill_schematic(
            house_schematics, base_seed, 10_000 + index
        )
        size_x, size_z, chosen, rotation = _structure_size(
            chosen, rotation, settings, base_seed, 10_000 + index
        )
        candidate = _choose_dense_infill_candidate(
            terrain=terrain,
            water=water,
            x_axis=x_axis,
            z_axis=z_axis,
            settings=settings,
            buildings=core_buildings,
            size_x=size_x,
            size_z=size_z,
            seed=base_seed,
            index=10_000 + index,
            walls=walls,
        )
        if candidate is None:
            continue
        center, ground = candidate
        core_buildings.append(
            GrowthBuilding(center, size_x, size_z, ground, chosen, rotation, "core")
        )

    outer_buildings: list[GrowthBuilding] = []
    for index in range(settings.outer_house_count):
        chosen, rotation = _choose_schematic(house_schematics, base_seed, 20_000 + index)
        size_x, size_z, chosen, rotation = _structure_size(
            chosen, rotation, settings, base_seed, 20_000 + index
        )
        cluster_index = index % settings.outer_cluster_count
        candidate = _choose_outer_candidate(
            terrain=terrain,
            water=water,
            x_axis=x_axis,
            z_axis=z_axis,
            settings=settings,
            anchor=anchor,
            cluster_center=_outer_cluster_center(anchor, settings, base_seed, cluster_index),
            buildings=[*core_buildings, *outer_buildings],
            size_x=size_x,
            size_z=size_z,
            seed=base_seed,
            index=index,
        )
        if candidate is None:
            continue
        center, ground = candidate
        outer_buildings.append(
            GrowthBuilding(center, size_x, size_z, ground, chosen, rotation, "outer")
        )

    buildings = [*core_buildings, *outer_buildings]
    trees_out = _trees(
        settings,
        anchor,
        terrain,
        water,
        x_axis,
        z_axis,
        buildings,
        tree_schematics,
        base_seed,
    )
    return GrowthLayout(attractors, tuple(core_buildings), tuple(buildings), trees_out)


__all__ = [
    "GrowthBuilding",
    "GrowthLayout",
    "building_footprint_bounds",
    "generate_town_layout",
    "wall_utilization",
]
