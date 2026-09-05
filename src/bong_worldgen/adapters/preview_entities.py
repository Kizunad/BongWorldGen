"""把 NPC 刷新点编码为仅供离线预览使用的真实村民实体。

按 Minecraft 1.20.1 的实体区域格式写入独立 entities/r.*.*.mca，
BlueMap 读取协议参考：
https://github.com/BlueMap-Minecraft/BlueMap/blob/v5.23/core/src/main/java/de/bluecolored/bluemap/core/world/mca/entity/chunk/MCAEntityChunk.java
"""

from __future__ import annotations

from io import BytesIO
import struct
from uuid import NAMESPACE_URL, uuid5

from ..engine.settlement_points import SettlementInterestPoint
from .anvil_nbt import (
    DATA_VERSION,
    TAG_BYTE,
    TAG_COMPOUND,
    TAG_DOUBLE,
    TAG_FLOAT,
    TAG_INT,
    TAG_INT_ARRAY,
    TAG_LIST,
    TAG_STRING,
    _byte,
    _end,
    _int,
    _list_header,
    _named,
    _string,
)


def encode_preview_villagers(
    chunk_x: int,
    chunk_z: int,
    points: tuple[SettlementInterestPoint, ...],
) -> bytes:
    """编码实体区块；脚部 Y 不偏移，XZ 放在点位对应方块的中心。"""

    buffer = BytesIO()
    _named(buffer, TAG_COMPOUND, "")
    _named(buffer, TAG_INT, "DataVersion")
    _int(buffer, DATA_VERSION)
    _named(buffer, TAG_INT_ARRAY, "Position")
    _int(buffer, 2)
    _int(buffer, chunk_x)
    _int(buffer, chunk_z)
    _named(buffer, TAG_LIST, "Entities")
    _list_header(buffer, TAG_COMPOUND, len(points))
    for point in sorted(points, key=lambda item: item.point_id):
        if (point.x // 16, point.z // 16) != (chunk_x, chunk_z):
            raise ValueError(f"preview NPC {point.point_id!r} is outside entity chunk")
        _named(buffer, TAG_STRING, "id")
        _string(buffer, "minecraft:villager")
        _named(buffer, TAG_INT_ARRAY, "UUID")
        _int(buffer, 4)
        # 固定 UUID 使同一刷新点在重复导出时仍然是同一个预览实体。
        identity = uuid5(NAMESPACE_URL, f"bong-worldgen:preview-villager:{point.point_id}")
        buffer.write(identity.bytes)
        for name, values in (
            ("Pos", (point.x + 0.5, float(point.y), point.z + 0.5)),
            ("Motion", (0.0, 0.0, 0.0)),
        ):
            _named(buffer, TAG_LIST, name)
            _list_header(buffer, TAG_DOUBLE, 3)
            buffer.write(struct.pack(">ddd", *values))
        _named(buffer, TAG_LIST, "Rotation")
        _list_header(buffer, TAG_FLOAT, 2)
        buffer.write(struct.pack(">ff", 0.0, 0.0))
        for name in ("NoAI", "NoGravity", "Silent", "Invulnerable", "PersistenceRequired"):
            _named(buffer, TAG_BYTE, name)
            _byte(buffer, 1)
        _named(buffer, TAG_FLOAT, "Health")
        buffer.write(struct.pack(">f", 20.0))
        _named(buffer, TAG_INT, "Age")
        _int(buffer, 0)
        _named(buffer, TAG_LIST, "Tags")
        _list_header(buffer, TAG_STRING, 2)
        _string(buffer, "bong_worldgen_preview")
        _string(buffer, point.point_id)
        _named(buffer, TAG_COMPOUND, "VillagerData")
        for name, value in (("type", "minecraft:plains"), ("profession", "minecraft:none")):
            _named(buffer, TAG_STRING, name)
            _string(buffer, value)
        _named(buffer, TAG_INT, "level")
        _int(buffer, 1)
        _end(buffer)
        _end(buffer)
    _end(buffer)
    return buffer.getvalue()
