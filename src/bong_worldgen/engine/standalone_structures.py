"""世界级独立结构刷新。

大型单体模板不应参与城镇的增量布局。本模块以世界坐标网格为稳定
候选域：同一个 ``(world seed, grid cell, template)`` 无论在哪个 tile 生成，
都会提出同一个位置和旋转。只有完整 footprint 落在当前地形窗口、且满足
干地和地形平整度约束时才写入方块。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .randomness import stable_text_seed, unit_interval
from .settlement_points import SettlementInterestPoint, SettlementSpawnArea
from .town import SettlementBlock
from .structures import (
    SchematicStructure,
    load_structure_directory,
    structure_anchor_y,
    structure_block_kind,
    structure_note,
)
from .terrain_config import Point, StandaloneStructureSettings


@dataclass(frozen=True)
class StandaloneStructureGeneration:
    """独立结构层的真实方块和 Server 区域记录。"""

    blocks: tuple[SettlementBlock, ...]
    spawn_areas: tuple[SettlementSpawnArea, ...]
    interest_points: tuple[SettlementInterestPoint, ...] = ()


def _grid_axes(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape != z.shape or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("standalone structure grids must be matching two-dimensional arrays")
    x_axis = np.asarray(x[0], dtype=np.float64)
    z_axis = np.asarray(z[:, 0], dtype=np.float64)
    if (x_axis.size > 1 and np.any(np.diff(x_axis) <= 0.0)) or (
        z_axis.size > 1 and np.any(np.diff(z_axis) <= 0.0)
    ):
        raise ValueError("standalone structure grids must use increasing world coordinates")
    return x_axis, z_axis


def _footprint(center: Point, structure: SchematicStructure) -> tuple[int, int, int, int]:
    min_x = int(round(center.x)) - structure.width // 2
    min_z = int(round(center.z)) - structure.length // 2
    return min_x, min_x + structure.width - 1, min_z, min_z + structure.length - 1


def _indices(
    axis: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray | None:
    if start < axis[0] or end > axis[-1]:
        return None
    step = float(axis[1] - axis[0]) if axis.size > 1 else 1.0
    positions = np.arange(start, end + 1, dtype=np.float64)
    return np.clip(np.rint((positions - axis[0]) / step), 0, axis.size - 1).astype(int)


def _suitable_location(
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    center: Point,
    structure: SchematicStructure,
    settings: StandaloneStructureSettings,
) -> tuple[int, tuple[int, int, int, int]] | None:
    min_x, max_x, min_z, max_z = _footprint(center, structure)
    columns = _indices(x_axis, min_x, max_x)
    rows = _indices(z_axis, min_z, max_z)
    if columns is None or rows is None:
        return None
    heights = terrain[np.ix_(rows, columns)]
    if np.any(water[np.ix_(rows, columns)] >= 0.0):
        return None
    if float(np.ptp(heights)) > settings.maximum_relief:
        return None
    slope_x = np.diff(heights, axis=1)
    slope_z = np.diff(heights, axis=0)
    maximum_slope = max(
        float(np.percentile(np.abs(slope_x), 90)) if slope_x.size else 0.0,
        float(np.percentile(np.abs(slope_z), 90)) if slope_z.size else 0.0,
    )
    if maximum_slope > settings.maximum_slope:
        return None
    return int(round(float(np.median(heights)))), (min_x, max_x, min_z, max_z)


def _candidate_centers(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    structure: SchematicStructure,
    settings: StandaloneStructureSettings,
    exclusion_center: Point,
    seed: int,
    unit_key: str | None = None,
    unit_keys: tuple[str, ...] = (),
) -> tuple[tuple[Point, int], ...]:
    """从与 tile 无关的世界网格提出本窗口可能包含的候选。"""

    half_extent = max(structure.width, structure.length) * 0.5
    spacing = settings.grid_spacing
    min_cell_x = math.floor((x_axis[0] - half_extent) / spacing)
    max_cell_x = math.floor((x_axis[-1] + half_extent) / spacing)
    min_cell_z = math.floor((z_axis[0] - half_extent) / spacing)
    max_cell_z = math.floor((z_axis[-1] + half_extent) / spacing)
    edge = half_extent + 2.0
    usable_span = spacing - edge * 2.0
    if usable_span <= 0.0:
        raise ValueError("standalone structure grid_spacing is too small for its template")

    if (unit_key is None) != (not unit_keys):
        raise ValueError("standalone placement unit selection must provide both key and pool")
    if unit_key is not None and unit_key not in unit_keys:
        raise ValueError("standalone placement unit key must belong to its pool")
    template_seed = seed + stable_text_seed(structure.name)
    selection_seed = seed + stable_text_seed("standalone_structure_selection")
    candidates: list[tuple[Point, int]] = []
    for cell_z in range(min_cell_z, max_cell_z + 1):
        for cell_x in range(min_cell_x, max_cell_x + 1):
            cell_seed = cell_x * 73_856_093 + cell_z * 19_349_663
            if unit_key is not None:
                selected = min(
                    int(unit_interval(selection_seed, cell_seed + 3) * len(unit_keys)),
                    len(unit_keys) - 1,
                )
                if unit_keys[selected] != unit_key:
                    continue
            center = Point(
                cell_x * spacing + edge + unit_interval(template_seed, cell_seed) * usable_span,
                cell_z * spacing
                + edge
                + unit_interval(template_seed, cell_seed + 1) * usable_span,
            )
            if math.hypot(
                center.x - exclusion_center.x,
                center.z - exclusion_center.z,
            ) < settings.exclusion_radius + half_extent:
                continue
            rotation = min(int(unit_interval(template_seed, cell_seed + 2) * 4.0), 3)
            candidates.append((center, rotation))
    return tuple(candidates)


def _overlaps(
    footprint: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> bool:
    min_x, max_x, min_z, max_z = footprint
    for other_min_x, other_max_x, other_min_z, other_max_z in occupied:
        if not (
            max_x < other_min_x
            or other_max_x < min_x
            or max_z < other_min_z
            or other_max_z < min_z
        ):
            return True
    return False


def _rotate_group_offset(offset: tuple[int, int], rotation: int) -> tuple[int, int]:
    """按模板旋转同步旋转结构组成员相对锚点的位置。"""

    x, z = offset
    turns = rotation % 4
    if turns == 1:
        return -z, x
    if turns == 2:
        return -x, -z
    if turns == 3:
        return z, -x
    return x, z


def _group_member_center(
    anchor: Point,
    structure: SchematicStructure,
    rotation: int,
) -> Point:
    """返回结构组成员相对组锚点的世界坐标中心。"""

    note = structure_note(structure.name)
    if note is None:
        return anchor
    offset_x, offset_z = _rotate_group_offset(note.group_offset, rotation)
    return Point(anchor.x + offset_x, anchor.z + offset_z)


def _group_candidate_shape(
    group_name: str,
    members: tuple[SchematicStructure, ...],
) -> SchematicStructure:
    """创建覆盖整个结构组的虚拟候选 footprint。

    结构组在选择候选网格时必须使用整体外包尺寸；否则锚点在 tile 外、但某
    个成员落进 tile 时，生成器可能只看见半组结构。
    """

    half_extent = 0.0
    for member in members:
        note = structure_note(member.name)
        if note is None:
            continue
        offset = note.group_offset
        half_extent = max(
            half_extent,
            abs(offset[0]) + max(member.width, member.length) * 0.5,
            abs(offset[1]) + max(member.width, member.length) * 0.5,
        )
    side = max(1, int(math.ceil(half_extent * 2.0)))
    return SchematicStructure(f"group:{group_name}", side, 1, side, ())


def _emit_group(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    areas: list[SettlementSpawnArea],
    occupied: list[tuple[int, int, int, int]],
    members: tuple[SchematicStructure, ...],
    anchor: Point,
    rotation: int,
    terrain: np.ndarray,
    water: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    settings: StandaloneStructureSettings,
) -> bool:
    """仅在全部成员可放置时，原子地写入一个结构组。"""

    candidates: list[
        tuple[SchematicStructure, Point, int, tuple[int, int, int, int]]
    ] = []
    for source in members:
        structure = source.rotated(rotation)
        center = _group_member_center(anchor, source, rotation)
        suitable = _suitable_location(
            terrain,
            water,
            x_axis,
            z_axis,
            center,
            structure,
            settings,
        )
        if suitable is None:
            return False
        ground, footprint = suitable
        if _overlaps(footprint, occupied) or _overlaps(
            footprint,
            [candidate[3] for candidate in candidates],
        ):
            return False
        candidates.append((structure, center, ground, footprint))
    grounds = [candidate[2] for candidate in candidates]
    if max(grounds) - min(grounds) > settings.maximum_relief:
        return False
    ground = int(round(float(np.median(grounds))))
    for structure, center, _, footprint in candidates:
        bounds = _emit_structure(blocks, structure, center, ground, settings)
        occupied.append(footprint)
        areas.append(_spawn_area(structure, center, ground, bounds, settings))
    return True


def _emit_structure(
    blocks: dict[tuple[int, int, int], SettlementBlock],
    structure: SchematicStructure,
    center: Point,
    ground: int,
    settings: StandaloneStructureSettings,
) -> tuple[int, int, int, int, int, int]:
    min_x, max_x, min_z, max_z = _footprint(center, structure)
    structure_ground = ground + settings.structure_ground_offset
    origin_y = structure_ground - structure_anchor_y(structure.name, structure.blocks)
    kind = structure_block_kind(structure.name, "minecraft:stone")
    for block in structure.blocks:
        world_x = min_x + block.x
        world_y = origin_y + block.y
        world_z = min_z + block.z
        blocks[(world_x, world_y, world_z)] = SettlementBlock(
            world_x,
            world_y,
            world_z,
            block.blockstate,
            kind,
        )
    return min_x, max_x, min_z, max_z, origin_y, origin_y + structure.height - 1


def _spawn_area(
    structure: SchematicStructure,
    center: Point,
    ground: int,
    bounds: tuple[int, int, int, int, int, int],
    settings: StandaloneStructureSettings,
) -> SettlementSpawnArea:
    min_x, max_x, min_z, max_z, min_y, max_y = bounds
    radius = settings.npc_spawn_radius
    note = structure_note(structure.name)
    category = "standalone" if note is None else note.category
    return SettlementSpawnArea(
        area_id=f"standalone-{structure.name}-{min_x}-{min_z}",
        structure_name=structure.name,
        category=category,
        district="standalone",
        center_x=int(round(center.x)),
        center_z=int(round(center.z)),
        min_x=min_x - radius,
        max_x=max_x + radius,
        min_z=min_z - radius,
        max_z=max_z + radius,
        min_y=ground,
        max_y=max(ground, max_y),
        ground_y=ground,
        footprint_min_x=min_x,
        footprint_max_x=max_x,
        footprint_min_z=min_z,
        footprint_max_z=max_z,
        footprint_min_y=min_y,
        footprint_max_y=max_y,
        defense_radius=radius,
    )


def generate_standalone_structures(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    settings: StandaloneStructureSettings | None,
    seed: int,
    exclusion_center: Point | None = None,
) -> StandaloneStructureGeneration:
    """在当前窗口生成满足条件的独立大型结构。

    ``exclusion_center`` 由调用方传入实际出生平原锚点。未传入时使用配方
    中的静态默认值，保持单独调用和旧配方的确定性。
    """

    if settings is None or not settings.enabled or settings.structure_directory is None:
        return StandaloneStructureGeneration((), ())
    if terrain.shape != water.shape or terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("standalone structure fields must share a shape")
    x_axis, z_axis = _grid_axes(x, z)
    effective_exclusion_center = (
        settings.exclusion_center if exclusion_center is None else exclusion_center
    )
    directory = Path(settings.structure_directory)
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[3] / directory
    structures = load_structure_directory(directory)
    blocks: dict[tuple[int, int, int], SettlementBlock] = {}
    areas: list[SettlementSpawnArea] = []
    occupied: list[tuple[int, int, int, int]] = []
    individual_structures: list[SchematicStructure] = []
    grouped_structures: dict[str, list[SchematicStructure]] = {}
    for source_structure in structures:
        note = structure_note(source_structure.name)
        if note is not None and note.category != "standalone":
            continue
        if note is not None and note.placement_group is not None:
            grouped_structures.setdefault(note.placement_group, []).append(source_structure)
        else:
            individual_structures.append(source_structure)

    groups = {
        group_name: tuple(members)
        for group_name, members in grouped_structures.items()
    }
    if any(len(members) < 2 for members in groups.values()):
        incomplete = ", ".join(
            sorted(name for name, members in groups.items() if len(members) < 2)
        )
        raise ValueError(f"standalone structure groups are incomplete: {incomplete}")
    unit_keys = tuple(
        sorted(
            [f"group:{name}" for name in groups]
            + [f"structure:{structure.name}" for structure in individual_structures]
        )
    )

    # 一个世界网格只选择一个结构单位；组成员共享同一候选锚点，避免目录
    # 扩充后每个网格同时出现所有大型模板。
    for group_name, members in sorted(groups.items()):
        group_shape = _group_candidate_shape(group_name, members)
        for center, rotation in _candidate_centers(
            x_axis,
            z_axis,
            group_shape,
            settings,
            effective_exclusion_center,
            seed,
            unit_key=f"group:{group_name}",
            unit_keys=unit_keys,
        ):
            _emit_group(
                blocks,
                areas,
                occupied,
                members,
                center,
                rotation,
                terrain,
                water,
                x_axis,
                z_axis,
                settings,
            )

    for source_structure in individual_structures:
        for center, rotation in _candidate_centers(
            x_axis,
            z_axis,
            source_structure,
            settings,
            effective_exclusion_center,
            seed,
            unit_key=f"structure:{source_structure.name}",
            unit_keys=unit_keys,
        ):
            structure = source_structure.rotated(rotation)
            suitable = _suitable_location(
                terrain,
                water,
                x_axis,
                z_axis,
                center,
                structure,
                settings,
            )
            if suitable is None:
                continue
            ground, footprint = suitable
            if _overlaps(footprint, occupied):
                continue
            bounds = _emit_structure(blocks, structure, center, ground, settings)
            occupied.append(footprint)
            areas.append(_spawn_area(structure, center, ground, bounds, settings))
    sorted_areas = tuple(sorted(areas, key=lambda area: area.area_id))
    interest_points = tuple(
        SettlementInterestPoint(
            point_id=f"{area.area_id}-interest",
            kind="standalone_structure",
            role="structure_center",
            district="standalone",
            x=area.center_x,
            y=area.ground_y + 1,
            z=area.center_z,
            area_id=area.area_id,
            structure_name=area.structure_name,
            tags=(
                "standalone",
                "structure",
                *(
                    structure_note(area.structure_name).roles
                    if structure_note(area.structure_name) is not None
                    else ()
                ),
            ),
        )
        for area in sorted_areas
    )
    return StandaloneStructureGeneration(
        tuple(blocks[key] for key in sorted(blocks)),
        sorted_areas,
        interest_points,
    )


__all__ = ["StandaloneStructureGeneration", "generate_standalone_structures"]
