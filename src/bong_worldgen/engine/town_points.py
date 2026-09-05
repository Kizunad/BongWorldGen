"""城镇建筑的区域标记与 NPC 室内选点，不负责方块发射。"""

from __future__ import annotations

from .settlement_points import SettlementInterestPoint, SettlementSpawnArea
from .structures import schematic_interior_spawn, structure_anchor_y, structure_note
from .terrain_config import Point, TownSettings
from .town_growth import GrowthBuilding


_INTERIOR_SPAWN_CATEGORIES = frozenset({"house", "spawn"})
def _building_footprint(
    center: Point,
    size_x: int,
    size_z: int,
) -> tuple[int, int, int, int]:
    """计算回退建筑的 X/Z 闭区间。"""

    half_x = size_x // 2
    half_z = size_z // 2
    min_x = int(round(center.x)) - half_x
    min_z = int(round(center.z)) - half_z
    return min_x, min_x + size_x - 1, min_z, min_z + size_z - 1


def building_spawn_area(
    building: GrowthBuilding,
    settings: TownSettings,
) -> SettlementSpawnArea:
    """按真实模板尺寸建立一条可序列化的 NPC 刷新区域。"""

    if building.schematic is None:
        structure_name = "fallback_house"
        category = "house"
        footprint_min_x, footprint_max_x, footprint_min_z, footprint_max_z = _building_footprint(
            building.center,
            building.size_x,
            building.size_z,
        )
        structure_ground = building.ground + settings.structure_ground_offset
        footprint_min_y = structure_ground
        footprint_max_y = structure_ground + settings.house_height + 1
    else:
        structure = building.schematic.rotated(building.rotation)
        structure_name = building.schematic.name
        note = structure_note(structure_name)
        category = "house" if note is None else note.category
        footprint_min_x = int(round(building.center.x)) - structure.width // 2
        footprint_min_z = int(round(building.center.z)) - structure.length // 2
        footprint_max_x = footprint_min_x + structure.width - 1
        footprint_max_z = footprint_min_z + structure.length - 1
        anchor_y = structure_anchor_y(structure_name, structure.blocks)
        structure_ground = building.ground + settings.structure_ground_offset
        origin_y = structure_ground - anchor_y
        footprint_min_y = origin_y
        footprint_max_y = origin_y + structure.height - 1

    radius = settings.npc_spawn_radius
    center_x = int(round(building.center.x))
    center_z = int(round(building.center.z))
    return SettlementSpawnArea(
        area_id=(
            f"town-{building.district}-{structure_name}-{footprint_min_x}-{footprint_min_z}"
        ),
        structure_name=structure_name,
        category=category,
        district=building.district,
        center_x=center_x,
        center_z=center_z,
        min_x=footprint_min_x - radius,
        max_x=footprint_max_x + radius,
        min_z=footprint_min_z - radius,
        max_z=footprint_max_z + radius,
        # NPC 刷新高度从地面开始，避免井水等地下结构被当作站立层；
        # 完整结构高度仍在 footprint_min_y/max_y 中保留。
        min_y=building.ground,
        max_y=max(building.ground, footprint_max_y),
        ground_y=building.ground,
        footprint_min_x=footprint_min_x,
        footprint_max_x=footprint_max_x,
        footprint_min_z=footprint_min_z,
        footprint_max_z=footprint_max_z,
        footprint_min_y=footprint_min_y,
        footprint_max_y=footprint_max_y,
        defense_radius=radius,
    )


def _building_interest_point(
    building: GrowthBuilding,
    area: SettlementSpawnArea,
) -> SettlementInterestPoint:
    """将建筑映射为一个稳定的设施/住宅兴趣点。"""

    category = area.category
    if category == "house":
        kind = "core_building" if building.district == "core" else "outer_building"
        role = "residential"
    elif category == "spawn":
        kind = "core_building"
        role = "settlement_core"
    else:
        kind = category if category in {"tower", "well", "stall", "farm", "camp", "tree"} else "structure"
        role = "facility"
    return SettlementInterestPoint(
        point_id=f"{area.area_id}-interest",
        kind=kind,
        role=role,
        district=building.district,
        x=area.center_x,
        y=building.ground + 1,
        z=area.center_z,
        area_id=area.area_id,
        structure_name=area.structure_name,
        tags=(category, building.district),
    )


def _building_interior_spawn_point(
    building: GrowthBuilding,
    area: SettlementSpawnArea,
    settings: TownSettings,
) -> SettlementInterestPoint | None:
    """将模板内预先识别的脚点映射到已放置建筑的世界坐标。"""

    if building.schematic is None:
        # 回退住宅的中心有地基与屋顶；至少两格净空才允许成人 NPC。
        if settings.house_height < 2:
            return None
        x, y, z = area.center_x, area.footprint_min_y + 1, area.center_z
    else:
        structure = building.schematic
        local = schematic_interior_spawn(structure)
        if local is None:
            return None
        local_x, local_y, local_z = structure.rotate_local_point(local, building.rotation)
        x = area.footprint_min_x + local_x
        y = area.footprint_min_y + local_y
        z = area.footprint_min_z + local_z
    return SettlementInterestPoint(
        point_id=f"{area.area_id}-interior-spawn",
        kind=f"{area.category}_interior_spawn",
        role="npc_spawn",
        district=building.district,
        x=x,
        y=y,
        z=z,
        area_id=area.area_id,
        structure_name=area.structure_name,
        tags=(area.category, "interior", "npc_spawn"),
    )


def building_interest_points(
    building: GrowthBuilding,
    area: SettlementSpawnArea,
    settings: TownSettings,
) -> tuple[SettlementInterestPoint, ...]:
    """返回建筑定位点及可选的室内 NPC 点；类别只在此处统一筛选。"""

    center = _building_interest_point(building, area)
    if building.district == "core" and area.category in _INTERIOR_SPAWN_CATEGORIES:
        interior = _building_interior_spawn_point(building, area, settings)
        if interior is not None:
            return center, interior
    return (center,)
