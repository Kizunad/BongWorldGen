"""城镇的方块发射与阶段编排。

布局由 ``town_growth.py`` 独立完成：建筑先沿 frontier 增长，少量主路
和按需巷道随后接入。本模块只负责读取模板、寻路铺路和输出可查询的方块记录。

布局分层参考以下公开项目和论文思路，未复制其代码：
* Tome（Minecraft 聚落生成）：https://github.com/Jandhi/Tome
* ProceduralCityGeneration（Voronoi 道路/地块）：
  https://github.com/Grzybojad/ProceduralCityGeneration
* Parish 与 Müller 的程序化城市道路扩张模型：
  https://dl.acm.org/doi/10.1145/383259.383292
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from .randomness import stable_text_seed, unit_interval
from .bridges import bridge_blocks
from .road_terrain import RoadTerrain
from .rural_roads import plan_rural_road
from .settlement_points import (
    SettlementInterestPoint,
    SettlementSpawnArea,
)
from .town_points import building_interest_points, building_spawn_area
from .town_growth import (
    GrowthBuilding,
    building_footprint_bounds,
    generate_town_layout,
)
from .structures import (
    SchematicStructure,
    load_structure_directory,
    structure_anchor_y,
    structure_block_kind,
    structure_note,
)
from .terrain_config import Point, TownSettings


@dataclass(frozen=True)
class SettlementBlock:
    """部落结构方块；``kind`` 供后续 Server/decorations 查询。"""

    x: int
    y: int
    z: int
    material: str
    kind: str


@dataclass(frozen=True)
class SettlementGeneration:
    """部落方块与 NPC 区域标记的统一生成结果。"""

    blocks: tuple[SettlementBlock, ...]
    spawn_areas: tuple[SettlementSpawnArea, ...]
    interest_points: tuple[SettlementInterestPoint, ...] = ()


def _grid_axes(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if x.shape != z.shape or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("settlement grids must be matching two-dimensional arrays")
    x_axis = np.asarray(x[0], dtype=np.float64)
    z_axis = np.asarray(z[:, 0], dtype=np.float64)
    step_x = float(x_axis[1] - x_axis[0]) if x_axis.size > 1 else 1.0
    step_z = float(z_axis[1] - z_axis[0]) if z_axis.size > 1 else 1.0
    if step_x <= 0.0 or step_z <= 0.0:
        raise ValueError("settlement grids must use increasing world coordinates")
    return x_axis, z_axis, step_x, step_z


def _nearest_cell(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    world_x: float,
    world_z: float,
) -> tuple[int, int] | None:
    if (
        world_x < x_axis[0]
        or world_x > x_axis[-1]
        or world_z < z_axis[0]
        or world_z > z_axis[-1]
    ):
        return None
    step_x = x_axis[1] - x_axis[0] if x_axis.size > 1 else 1.0
    step_z = z_axis[1] - z_axis[0] if z_axis.size > 1 else 1.0
    col = int(np.clip(np.rint((world_x - x_axis[0]) / max(step_x, 1.0e-9)), 0, x_axis.size - 1))
    row = int(np.clip(np.rint((world_z - z_axis[0]) / max(step_z, 1.0e-9)), 0, z_axis.size - 1))
    return row, col


def _sample_cell(
    values: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    world_x: float,
    world_z: float,
) -> float | None:
    cell = _nearest_cell(x_axis, z_axis, world_x, world_z)
    return None if cell is None else float(values[cell])


def _load_house_schematics(
    settings: TownSettings,
) -> tuple[SchematicStructure, ...]:
    """读取配置的房屋资源目录；未配置时返回石砖矩形回退。"""

    if settings.house_schematic_directory is None:
        return ()
    configured = Path(settings.house_schematic_directory)
    if not configured.is_absolute():
        configured = Path(__file__).resolve().parents[3] / configured
    return load_structure_directory(configured)


def _split_settlement_schematics(
    schematics: tuple[SchematicStructure, ...],
) -> tuple[
    tuple[SchematicStructure, ...],
    tuple[SchematicStructure, ...],
    tuple[SchematicStructure, ...],
    tuple[SchematicStructure, ...],
]:
    """按目录备注拆分住宅、Spawn 核心、外围墙和城门模板。

    ``standalone`` 模板由独立地标层处理，绝不能落入部落住宅候选池。
    """

    houses: list[SchematicStructure] = []
    cores: list[SchematicStructure] = []
    walls: list[SchematicStructure] = []
    gates: list[SchematicStructure] = []
    for schematic in schematics:
        note = structure_note(schematic.name)
        if note is not None and note.category == "spawn":
            cores.append(schematic)
        elif note is not None and note.category == "wall":
            walls.append(schematic)
        elif note is not None and note.category == "gate":
            gates.append(schematic)
        elif note is None or note.category != "standalone":
            houses.append(schematic)
    return tuple(houses), tuple(cores), tuple(walls), tuple(gates)


def _load_tree_schematics(settings: TownSettings) -> tuple[SchematicStructure, ...]:
    """读取树木模板；资源不存在时不阻止部落生成。"""

    if settings.tree_schematic_directory is None:
        return ()
    configured = Path(settings.tree_schematic_directory)
    if not configured.is_absolute():
        configured = Path(__file__).resolve().parents[3] / configured
    if not configured.is_dir():
        return ()
    try:
        return load_structure_directory(configured)
    except ValueError:
        return ()


def _road_edges(nodes: list[Point], seed: int) -> tuple[tuple[int, int], ...]:
    """用 Prim MST 保证连通，再按 seed 恢复少量环路。"""

    if len(nodes) < 2:
        return ()
    connected = {0}
    edges: list[tuple[int, int]] = []
    while len(connected) < len(nodes):
        candidate: tuple[float, int, int] | None = None
        for left in sorted(connected):
            for right in range(len(nodes)):
                if right in connected:
                    continue
                distance = math.hypot(
                    nodes[left].x - nodes[right].x,
                    nodes[left].z - nodes[right].z,
                )
                item = (distance, left, right)
                if candidate is None or item < candidate:
                    candidate = item
        if candidate is None:
            break
        _, left, right = candidate
        connected.add(right)
        edges.append((left, right))
    existing = {tuple(sorted(edge)) for edge in edges}
    edge_index = 0
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            edge = (left, right)
            if edge in existing:
                continue
            if unit_interval(seed, edge_index + 401) < 0.22:
                edges.append(edge)
            edge_index += 1
    return tuple(edges)


def _add_block(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    world_x: int,
    world_z: int,
    world_y: int,
    material: str,
    kind: str,
) -> None:
    if _nearest_cell(x_axis, z_axis, world_x, world_z) is None:
        return
    water_level = _sample_cell(water, x_axis, z_axis, world_x, world_z)
    if water_level is None or water_level >= 0.0:
        return
    blocks[(world_x, world_y, world_z)] = SettlementBlock(
        world_x,
        world_y,
        world_z,
        material,
        kind,
    )


def _emit_road(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    start: Point,
    end: Point,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    seed: int,
    *,
    kind: str = "road",
    width: int | None = None,
    material_override: str | None = None,
) -> tuple[Point, ...]:
    """铺设道路并返回其中心线，供后续建筑接入已有路网。"""

    road_width = settings.road_width if width is None else width
    route_settings = replace(settings, road_width=road_width)
    plan = plan_rural_road(
        start,
        end,
        terrain,
        water,
        x_axis,
        z_axis,
        route_settings,
        seed,
    )
    path = plan.path
    bridge_edges = {(bridge.start, bridge.end) for bridge in plan.bridges}
    # 保留配置给出的真实宽度。旧的 radius 写法会把 2 格道路扩成 3 格，
    # 不能表达城门道路的 2 至 5 格范围。
    negative_extent = road_width // 2
    positive_extent = road_width - negative_extent
    if material_override is None:
        materials = settings.road_materials
        weights = np.asarray(settings.road_material_weights, dtype=np.float64)
        cumulative = np.cumsum(weights / weights.sum())
    else:
        # 城门外第一段路是明确的土路，不能被普通道路的随机材质覆盖。
        materials = (material_override,)
        cumulative = np.asarray((1.0,), dtype=np.float64)
    for first, second in zip(path, path[1:]):
        if (first, second) in bridge_edges:
            continue
        distance = math.hypot(second.x - first.x, second.z - first.z)
        steps = max(1, int(math.ceil(distance * 2.0)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            point_x = first.x + (second.x - first.x) * float(t)
            point_z = first.z + (second.z - first.z) * float(t)
            for offset_x in range(-negative_extent, positive_extent):
                for offset_z in range(-negative_extent, positive_extent):
                    world_x = int(round(point_x + offset_x))
                    world_z = int(round(point_z + offset_z))
                    ground = _sample_cell(terrain, x_axis, z_axis, world_x, world_z)
                    if ground is not None:
                        material_pick = unit_interval(
                            seed + world_x * 73856093 + world_z * 19349663,
                            13,
                        )
                        material_index = int(
                            np.searchsorted(cumulative, material_pick, side="right")
                        )
                        material_index = min(material_index, len(materials) - 1)
                        _add_block(
                            blocks,
                            terrain,
                            water,
                            x_axis,
                            z_axis,
                            world_x,
                            world_z,
                            int(round(ground)),
                            materials[material_index],
                            kind,
                        )
    # 双岸、宽度和净空已经在寻路阶段一起验证。桥梁必须允许向水上发射，
    # 不走拒绝一切湿列的普通建筑/陆路 helper，也不修改 terrain/water。
    grid = RoadTerrain(terrain, water, x_axis, z_axis, road_width)
    for bridge in plan.bridges:
        for bx, by, bz, material, block_kind in bridge_blocks(bridge, grid, route_settings):
            blocks[(bx, by, bz)] = SettlementBlock(bx, by, bz, material, block_kind)
    return path


def _emit_house(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    center: Point,
    size_x: int,
    size_z: int,
    ground: int,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    schematic: SchematicStructure | None = None,
    rotation: int = 0,
) -> None:
    if schematic is not None:
        structure = schematic.rotated(rotation)
        origin_x = int(round(center.x)) - structure.width // 2
        origin_z = int(round(center.z)) - structure.length // 2
        # Anvil 的 surface_y 是地表方块所在层；建筑基准要抬到其上方。
        # 井模板的资源原点可能位于地下，Fence/枯木锚点仍以该基准对齐。
        structure_ground = ground + settings.structure_ground_offset
        origin_y = structure_ground - structure_anchor_y(schematic.name, structure.blocks)
        for block in structure.blocks:
            _add_block(
                blocks, terrain, water, x_axis, z_axis,
                origin_x + block.x, origin_z + block.z,
                origin_y + block.y,
                block.blockstate,
                structure_block_kind(schematic.name, block.blockstate),
            )
        return
    half_x = size_x // 2
    half_z = size_z // 2
    min_x, max_x = int(round(center.x)) - half_x, int(round(center.x)) + half_x
    min_z, max_z = int(round(center.z)) - half_z, int(round(center.z)) + half_z
    structure_ground = ground + settings.structure_ground_offset
    for world_z in range(min_z, max_z + 1):
        for world_x in range(min_x, max_x + 1):
            _add_block(
                blocks, terrain, water, x_axis, z_axis,
                world_x, world_z, structure_ground, settings.house_material, "house_foundation",
            )
            wall = world_x in (min_x, max_x) or world_z in (min_z, max_z)
            for offset_y in range(1, settings.house_height + 1):
                if wall:
                    _add_block(
                        blocks, terrain, water, x_axis, z_axis,
                        world_x, world_z, structure_ground + offset_y, settings.house_material, "house_wall",
                    )
            _add_block(
                blocks, terrain, water, x_axis, z_axis,
                world_x, world_z, structure_ground + settings.house_height + 1,
                settings.house_material, "house_roof",
            )


def _emit_tree(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    center: Point,
    ground: int,
    schematic: SchematicStructure,
    rotation: int,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
) -> None:
    structure = schematic.rotated(rotation)
    origin_x = int(round(center.x)) - structure.width // 2
    origin_z = int(round(center.z)) - structure.length // 2
    for block in structure.blocks:
        _add_block(
            blocks,
            terrain,
            water,
            x_axis,
            z_axis,
            origin_x + block.x,
            origin_z + block.z,
            ground + block.y,
            block.blockstate,
            "tree_schematic",
        )


def _emit_settlement_wall(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    buildings: tuple[GrowthBuilding, ...],
    wall_schematics: tuple[SchematicStructure, ...],
    gate_schematics: tuple[SchematicStructure, ...],
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: TownSettings,
    seed: int,
    tower_schematics: tuple[SchematicStructure, ...] = (),
) -> tuple[Point, ...]:
    """沿核心建筑包围盒发射完整墙段、城门和四角石塔。

    城墙模板的水平步长和垂直步长都来自模板尺寸，边界先向外对齐到该
    步长的整数倍，因此不会再用 ``min`` 截断最后一段墙。城门开口是唯一
    的例外：墙段仍保持完整，只跳过其落在门体包围盒内的方块。返回的点是
    每座城门外侧一格，供外城区土路作为固定起点。
    """

    if not buildings or not wall_schematics:
        return ()
    wall = wall_schematics[
        min(int(unit_interval(seed, 9_101) * len(wall_schematics)), len(wall_schematics) - 1)
    ]
    wall_name = wall.name
    margin = settings.core_wall_margin
    bounds = building_footprint_bounds(buildings, margin)
    if bounds is None:
        return ()
    min_x, max_x, min_z, max_z = bounds

    gate = None
    if gate_schematics:
        gate = gate_schematics[
            min(int(unit_interval(seed, 9_102) * len(gate_schematics)), len(gate_schematics) - 1)
        ]

    raw_span_x = max_x - min_x + 1
    raw_span_z = max_z - min_z + 1
    horizontal_wall = wall.rotated(1)
    vertical_wall = wall.rotated(0)
    horizontal_step = max(horizontal_wall.width, 1)
    vertical_step = max(vertical_wall.length, 1)

    # 先确定城门方向，再确保带城门的长边至少能容纳门体和两段完整墙。
    long_axis_x = raw_span_x >= raw_span_z
    gate_width = 0
    if gate is not None:
        gate_width = gate.rotated(0).width if long_axis_x else gate.rotated(1).length
    required_long_span = gate_width + 2 * (horizontal_step if long_axis_x else vertical_step)

    def align_bounds(
        low: int,
        high: int,
        step: int,
        minimum_span: int = 0,
    ) -> tuple[int, int]:
        """把闭区间向外扩展为完整模板步长，保留所有建筑在墙内。"""

        span = max(high - low + 1, minimum_span, step)
        count = max(1, int(math.ceil(span / step)))
        aligned_low = math.floor(low / step) * step
        aligned_high = aligned_low + count * step - 1
        while aligned_high < high:
            count += 1
            aligned_high = aligned_low + count * step - 1
        return aligned_low, aligned_high

    min_x, max_x = align_bounds(
        min_x,
        max_x,
        horizontal_step,
        required_long_span if long_axis_x else 0,
    )
    min_z, max_z = align_bounds(
        min_z,
        max_z,
        vertical_step,
        required_long_span if not long_axis_x else 0,
    )
    span_x = max_x - min_x + 1
    span_z = max_z - min_z + 1
    # 城门落在围墙的长轴两端，成对提供出入口；短轴只保留连续城墙。
    gate_sides: dict[tuple[bool, int], tuple[SchematicStructure, int]] = {}
    if gate is not None:
        if span_x >= span_z:
            gate_sides[(True, min_z)] = (gate.rotated(0), (min_x + max_x) // 2)
            gate_sides[(True, max_z)] = (gate.rotated(2), (min_x + max_x) // 2)
        else:
            gate_sides[(False, min_x)] = (gate.rotated(1), (min_z + max_z) // 2)
            gate_sides[(False, max_x)] = (gate.rotated(3), (min_z + max_z) // 2)

    gate_road_starts: list[Point] = []

    def emit_side(rotation: int, horizontal: bool, side: int, start: int, end: int) -> None:
        structure = wall.rotated(rotation)
        gate_entry = gate_sides.get((horizontal, side))
        gate_structure = None if gate_entry is None else gate_entry[0]
        gate_center = None if gate_entry is None else gate_entry[1]
        if gate_structure is None:
            gate_start = gate_end = -1
        else:
            gate_span = gate_structure.width if horizontal else gate_structure.length
            gate_start = gate_center - gate_span // 2
            gate_end = gate_start + gate_span - 1
        cursor = start
        step = max(structure.width if horizontal else structure.length, 1)
        gate_origin_x = gate_origin_z = 0
        if gate_structure is not None:
            if horizontal:
                gate_origin_x = gate_start
                gate_origin_z = side - gate_structure.length // 2
            else:
                gate_origin_x = side - gate_structure.width // 2
                gate_origin_z = gate_start

        def inside_gate(world_x: int, world_z: int) -> bool:
            return (
                gate_structure is not None
                and gate_origin_x <= world_x < gate_origin_x + gate_structure.width
                and gate_origin_z <= world_z < gate_origin_z + gate_structure.length
            )

        while cursor <= end:
            if horizontal:
                world_x, world_z = cursor, side
                sample_x, sample_z = min(cursor + structure.width // 2, end), side
            else:
                world_x, world_z = side, cursor
                sample_x, sample_z = side, min(cursor + structure.length // 2, end)
            ground = _sample_cell(terrain, x_axis, z_axis, sample_x, sample_z)
            if ground is not None and _sample_cell(water, x_axis, z_axis, sample_x, sample_z) < 0.0:
                origin_x = world_x if horizontal else side - structure.width // 2
                origin_z = side - structure.length // 2 if horizontal else world_z
                origin_y = int(round(ground)) + settings.structure_ground_offset
                for block in structure.blocks:
                    block_x = origin_x + block.x
                    block_z = origin_z + block.z
                    if inside_gate(block_x, block_z):
                        continue
                    _add_block(
                        blocks, terrain, water, x_axis, z_axis,
                        block_x, block_z, origin_y + block.y,
                        block.blockstate, structure_block_kind(wall_name, block.blockstate),
                    )
            cursor += step

        if gate_structure is None:
            return
        if horizontal:
            world_x = gate_start
            world_z = side - gate_structure.length // 2
            sample_x, sample_z = gate_center, side
        else:
            world_x = side - gate_structure.width // 2
            world_z = gate_start
            sample_x, sample_z = side, gate_center
        ground = _sample_cell(terrain, x_axis, z_axis, sample_x, sample_z)
        if ground is None or _sample_cell(water, x_axis, z_axis, sample_x, sample_z) >= 0.0:
            return
        origin_y = (
            int(round(ground))
            + settings.structure_ground_offset
            - settings.gate_structure_sink
        )
        for block in gate_structure.blocks:
            _add_block(
                blocks,
                terrain,
                water,
                x_axis,
                z_axis,
                world_x + block.x,
                world_z + block.z,
                origin_y + block.y,
                block.blockstate,
                structure_block_kind(gate_structure.name, block.blockstate),
            )
        if horizontal:
            outward = -1 if side == min_z else 1
            gate_road_starts.append(
                Point(
                    float(gate_center),
                    float(side + outward * (gate_structure.length // 2 + 1)),
                )
            )
        else:
            outward = -1 if side == min_x else 1
            gate_road_starts.append(
                Point(
                    float(side + outward * (gate_structure.width // 2 + 1)),
                    float(gate_center),
                )
            )

    emit_side(1, True, min_z, min_x, max_x)
    emit_side(1, True, max_z, min_x, max_x)
    emit_side(0, False, min_x, min_z, max_z)
    emit_side(0, False, max_x, min_z, max_z)

    # 角塔中心落在两条墙的交点，塔体覆盖墙端的 9x9 交接区；发射顺序在
    # 墙和门之后，保证角部不会留下可见缝隙。Simple Stone Tower（31122）
    # 是首选，目录中没有它时才回退到其它 tower 类模板。
    tower = next((item for item in tower_schematics if item.name == "31122"), None)
    if tower is None and tower_schematics:
        tower = tower_schematics[0]
    if tower is not None:
        for corner_x, corner_z in (
            (min_x, min_z),
            (max_x, min_z),
            (min_x, max_z),
            (max_x, max_z),
        ):
            ground = _sample_cell(terrain, x_axis, z_axis, corner_x, corner_z)
            if ground is None or _sample_cell(water, x_axis, z_axis, corner_x, corner_z) >= 0.0:
                continue
            structure = tower.rotated(0)
            origin_x = corner_x - structure.width // 2
            origin_z = corner_z - structure.length // 2
            origin_y = (
                int(round(ground))
                + settings.structure_ground_offset
                - settings.tower_structure_sink
            )
            for block in structure.blocks:
                _add_block(
                    blocks,
                    terrain,
                    water,
                    x_axis,
                    z_axis,
                    origin_x + block.x,
                    origin_z + block.z,
                    origin_y + block.y,
                    block.blockstate,
                    structure_block_kind(tower.name, block.blockstate),
                )

    return tuple(gate_road_starts)


def generate_town_result(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: TownSettings | None,
    anchor: Point,
    seed: int,
) -> SettlementGeneration:
    """生成一份与 tile 无关的城镇片段。

    ``anchor`` 可以来自出生平原，也可以由任意调用方提供；城镇布局不依赖
    ``spawn_plain``。不同 tile 只会截取自己范围内的方块，不会重新选择中心。
    """

    if settings is None or not settings.enabled:
        return SettlementGeneration((), ())
    if terrain.shape != water.shape or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("town fields must share a shape")
    x_axis, z_axis, _, _ = _grid_axes(x, z)
    blocks: dict[tuple[int, int, int], SettlementBlock] = {}
    schematics = _load_house_schematics(settings)
    house_schematics, core_schematics, wall_schematics, gate_schematics = _split_settlement_schematics(
        schematics
    )
    tower_schematics = tuple(
        schematic
        for schematic in schematics
        if (structure_note(schematic.name) is not None)
        and structure_note(schematic.name).category == "tower"
    )
    tree_schematics = _load_tree_schematics(settings)
    layout = generate_town_layout(
        terrain,
        water,
        x_axis,
        z_axis,
        settings,
        anchor,
        seed,
        house_schematics,
        tree_schematics,
        core_schematics,
    )
    nodes = list(layout.attractors)
    for edge_index, (left, right) in enumerate(
        _road_edges(nodes, seed + stable_text_seed("settlement_roads"))
    ):
        if edge_index >= settings.main_road_count:
            break
        _emit_road(
            blocks,
            nodes[left],
            nodes[right],
            terrain,
            water,
            x_axis,
            z_axis,
            settings,
            seed + edge_index,
            kind="road",
            width=settings.road_width,
        )
    gate_road_starts: tuple[Point, ...] = ()
    if settings.core_wall_enabled:
        gate_road_starts = _emit_settlement_wall(
            blocks,
            layout.core_buildings,
            wall_schematics,
            gate_schematics,
            terrain,
            water,
            x_axis,
            z_axis,
            settings,
            seed,
            tower_schematics,
        )
    # 核心区保持高密度建筑和城墙，不再为每栋核心建筑单独向外拉巷道。
    # 城门是核心与外围部落唯一的道路接口；外围建筑只向已经发射的门外
    # 路网接入，因而不会从城墙四角穿出。
    outer_buildings = tuple(
        building for building in layout.buildings if building.district == "outer"
    )
    outer_road_network: list[Point] = []
    if gate_road_starts and outer_buildings:
        for road_index, start in enumerate(gate_road_starts):
            target_building = min(
                outer_buildings,
                key=lambda building: math.hypot(
                    start.x - building.center.x,
                    start.z - building.center.z,
                ),
            )
            width_span = settings.gate_road_max_width - settings.gate_road_min_width + 1
            width = settings.gate_road_min_width + min(
                int(
                    unit_interval(
                        seed + stable_text_seed("settlement_gate_road_width"),
                        road_index,
                    )
                    * width_span
                ),
                width_span - 1,
            )
            outer_road_network.extend(
                _emit_road(
                    blocks,
                    start,
                    target_building.center,
                    terrain,
                    water,
                    x_axis,
                    z_axis,
                    settings,
                    seed + stable_text_seed("settlement_gate_road") + road_index,
                    kind="gate_road",
                    width=width,
                    material_override=settings.gate_road_material,
                )
            )
    # 外围房屋按布局顺序接到最近的门外路网。每条新巷道也会加入网络，
    # 后续房屋自然共享已有路径，不会各自直线冲向核心区。
    for index, building in enumerate(outer_buildings):
        if not outer_road_network:
            break
        target = min(
            outer_road_network,
            key=lambda point: math.hypot(
                building.center.x - point.x,
                building.center.z - point.z,
            ),
        )
        if math.hypot(building.center.x - target.x, building.center.z - target.z) <= 8.0:
            continue
        outer_road_network.extend(
            _emit_road(
                blocks,
                building.center,
                target,
                terrain,
                water,
                x_axis,
                z_axis,
                settings,
                seed + stable_text_seed("settlement_outer_alley") + index,
                kind="alley",
                width=settings.alley_width,
            )
        )
    spawn_areas = tuple(
        building_spawn_area(building, settings) for building in layout.buildings
    )
    interest_points: list[SettlementInterestPoint] = []
    center_cell = _nearest_cell(x_axis, z_axis, anchor.x, anchor.z)
    if center_cell is not None and water[center_cell] < 0.0:
        interest_points.append(
            SettlementInterestPoint(
                point_id=f"town-center-{int(round(anchor.x))}-{int(round(anchor.z))}",
                kind="settlement_center",
                role="settlement_center",
                district="core",
                x=int(round(anchor.x)),
                y=int(round(float(terrain[center_cell]))) + 1,
                z=int(round(anchor.z)),
                tags=("center", "settlement"),
            )
        )
    # 吸引点是道路图的逻辑节点；中心点已有独立语义，其余节点作为路网兴趣点。
    for node_index, node in enumerate(layout.attractors[1:], start=1):
        node_cell = _nearest_cell(x_axis, z_axis, node.x, node.z)
        if node_cell is None or water[node_cell] >= 0.0:
            continue
        interest_points.append(
            SettlementInterestPoint(
                point_id=f"town-road-junction-{node_index}-{int(round(node.x))}-{int(round(node.z))}",
                kind="road_junction",
                role="route_node",
                district="core",
                x=int(round(node.x)),
                y=int(round(float(terrain[node_cell]))) + 1,
                z=int(round(node.z)),
                tags=("road", "junction"),
            )
        )
    for building, area in zip(layout.buildings, spawn_areas):
        interest_points.extend(building_interest_points(building, area, settings))
    for index, (center, schematic, _, ground) in enumerate(layout.tree_centers):
        interest_points.append(
            SettlementInterestPoint(
                point_id=f"town-tree-{index}-{int(round(center.x))}-{int(round(center.z))}",
                kind="tree",
                role="decoration",
                district="outer",
                x=int(round(center.x)),
                y=ground + 1,
                z=int(round(center.z)),
                structure_name=schematic.name,
                tags=("decoration", "tree"),
            )
        )
    for building in layout.buildings:
        _emit_house(
            blocks,
            building.center,
            building.size_x,
            building.size_z,
            building.ground,
            terrain,
            water,
            x_axis,
            z_axis,
            settings,
            building.schematic,
            building.rotation,
        )
    for center, schematic, rotation, ground in layout.tree_centers:
        _emit_tree(
            blocks,
            center,
            ground,
            schematic,
            rotation,
            terrain,
            water,
            x_axis,
            z_axis,
        )
    return SettlementGeneration(
        blocks=tuple(blocks[key] for key in sorted(blocks)),
        spawn_areas=spawn_areas,
        interest_points=tuple(
            sorted(interest_points, key=lambda point: point.point_id)
        ),
    )


def generate_town(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: TownSettings | None,
    anchor: Point,
    seed: int,
) -> tuple[SettlementBlock, ...]:
    """兼容只需要结构方块的调用方；完整结果见 ``generate_town_result``。"""

    return generate_town_result(
        terrain,
        water,
        x,
        z,
        settings,
        anchor,
        seed,
    ).blocks


__all__ = [
    "SettlementBlock",
    "SettlementGeneration",
    "generate_town",
    "generate_town_result",
]
