"""聚落兴趣点、建筑防守区域及其 JSON 数据契约。

兴趣点与建筑防守区域是两种不同语义：区域描述一块 AABB，兴趣点描述一个
可被 Server、NPC 或任务系统直接定位的世界坐标。这样住宅内部点位不会被
迫塞进 ``SettlementSpawnArea``。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettlementInterestPoint:
    """聚落中的一个可查询坐标。

    ``y`` 是实体脚部所在的方块坐标，而不是地板方块坐标。屋内点因此可以
    直接交给 Server 的 NPC 生成逻辑；``area_id`` 将它关联回建筑防守区域。
    """

    point_id: str
    kind: str
    role: str
    district: str
    x: int
    y: int
    z: int
    area_id: str | None = None
    structure_name: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("point_id", "kind", "role", "district"):
            if not getattr(self, name).strip():
                raise ValueError(f"settlement interest point {name} cannot be empty")
        if self.area_id is not None and not self.area_id.strip():
            raise ValueError("settlement interest point area_id cannot be blank")
        if self.structure_name is not None and not self.structure_name.strip():
            raise ValueError("settlement interest point structure_name cannot be blank")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("settlement interest point tags must be non-empty strings")


def settlement_interest_point_manifest(point: SettlementInterestPoint) -> dict[str, object]:
    """转换为 Server/BlueMap 共用的 JSON 记录。"""

    return {
        "point_id": point.point_id,
        "kind": point.kind,
        "role": point.role,
        "district": point.district,
        "position": {"x": point.x, "y": point.y, "z": point.z},
        "area_id": point.area_id,
        "structure_name": point.structure_name,
        "tags": list(point.tags),
    }


@dataclass(frozen=True)
class SettlementSpawnArea:
    """一栋建筑对应的 NPC 刷新/防守区域。

    ``min_x..max_z`` 包含水平防守缓冲；``min_y..max_y`` 是从地面开始的
    可刷新高度范围，``ground_y`` 是优先寻找站立点的地面层。``footprint_*``
    保留模板完整三维范围，包括井底等地下部分。
    """

    area_id: str
    structure_name: str
    category: str
    # ``core`` 表示城墙内的高密度核心，``outer`` 表示无墙外围部落；
    # 独立地标使用 ``standalone``。Server 可据此选择不同的防守和刷新策略。
    district: str
    center_x: int
    center_z: int
    min_x: int
    max_x: int
    min_z: int
    max_z: int
    min_y: int
    max_y: int
    ground_y: int
    footprint_min_x: int
    footprint_max_x: int
    footprint_min_z: int
    footprint_max_z: int
    footprint_min_y: int
    footprint_max_y: int
    defense_radius: int


def settlement_spawn_area_manifest(area: SettlementSpawnArea) -> dict[str, object]:
    """把 NPC 区域转换成 raster/Server 共用的 JSON 记录。"""

    return {
        "area_id": area.area_id,
        "structure_name": area.structure_name,
        "category": area.category,
        "district": area.district,
        "center": {"x": area.center_x, "z": area.center_z},
        "spawn_bounds": {
            "min_x": area.min_x,
            "max_x": area.max_x,
            "min_z": area.min_z,
            "max_z": area.max_z,
            "min_y": area.min_y,
            "max_y": area.max_y,
        },
        "footprint": {
            "min_x": area.footprint_min_x,
            "max_x": area.footprint_max_x,
            "min_z": area.footprint_min_z,
            "max_z": area.footprint_max_z,
            "min_y": area.footprint_min_y,
            "max_y": area.footprint_max_y,
        },
        "ground_y": area.ground_y,
        "defense_radius": area.defense_radius,
    }


__all__ = [
    "SettlementInterestPoint",
    "SettlementSpawnArea",
    "settlement_interest_point_manifest",
    "settlement_spawn_area_manifest",
]
