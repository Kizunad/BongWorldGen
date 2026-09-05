"""把生成的高度场转换为有边界的 Minecraft Anvil 世界。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

import numpy as np

from ..data.wilderness import LAKE, MOUNTAINS, RIVER
from ..engine.generated_world import Heightfield
from ..engine.settlement_points import SettlementInterestPoint
from .anvil_nbt import (
    AIR,
    CLAY,
    COAL_ORE,
    COARSE_DIRT,
    COPPER_ORE,
    CHEST,
    DIRT,
    GLOW_LICHEN,
    MOSS_BLOCK,
    SPORE_BLOSSOM,
    GRASS_BLOCK,
    GRAVEL,
    IRON_ORE,
    MUD,
    MUD_BRICKS,
    PACKED_MUD,
    SAND,
    SNOW_BLOCK,
    ICE,
    PACKED_ICE,
    BLUE_ICE,
    POWDER_SNOW,
    ANDESITE,
    CALCITE,
    TUFF,
    DEEPSLATE,
    DEAD_BUSH,
    STONE,
    STONE_BRICKS,
    WATER,
    FLOWING_WATER_LEVEL_1,
    OAK_LOG,
    OAK_PLANKS,
    TORCH,
    WORLD_MAX_Y,
    WORLD_MIN_Y,
    encode_chunk_nbt,
    encode_level_dat,
    register_blockstate,
)
from .bong_raster import to_bong_tile
from .preview_entities import encode_preview_villagers


CHUNKS_PER_REGION = 32
SECTOR_SIZE = 4096
REGION_HEADER_SIZE = 2 * SECTOR_SIZE

RIVERBED_BLOCKS = {
    "dirt": DIRT,
    "mud": MUD,
    "gravel": GRAVEL,
    "sand": SAND,
    "clay": CLAY,
    "coarse_dirt": COARSE_DIRT,
    "packed_mud": PACKED_MUD,
    "mud_bricks": MUD_BRICKS,
    "mud_brick": MUD_BRICKS,
}

UNDERGROUND_BLOCKS = {
    "oak_log": OAK_LOG,
    "oak_planks": OAK_PLANKS,
    "chest": CHEST,
    "torch": TORCH,
    "coal_ore": COAL_ORE,
    "iron_ore": IRON_ORE,
    "copper_ore": COPPER_ORE,
    "glow_lichen": GLOW_LICHEN,
    "moss_block": MOSS_BLOCK,
    "spore_blossom": SPORE_BLOSSOM,
    "dead_bush": DEAD_BUSH,
}

SETTLEMENT_BLOCKS = {
    "stone_bricks": STONE_BRICKS,
}

GLACIAL_SURFACE_BLOCKS = {
    "minecraft:snow_block": SNOW_BLOCK,
    "minecraft:powder_snow": POWDER_SNOW,
    "minecraft:ice": ICE,
    "minecraft:packed_ice": PACKED_ICE,
    "minecraft:blue_ice": BLUE_ICE,
    "minecraft:gravel": GRAVEL,
    "minecraft:stone": STONE,
    "minecraft:andesite": ANDESITE,
    "minecraft:calcite": CALCITE,
    "minecraft:tuff": TUFF,
    "minecraft:deepslate": DEEPSLATE,
}

PERMAFROST_BLOCKS = {
    "minecraft:powder_snow": POWDER_SNOW,
    "minecraft:snow_block": SNOW_BLOCK,
    "minecraft:ice": ICE,
    "minecraft:packed_ice": PACKED_ICE,
    "minecraft:gravel": GRAVEL,
    "minecraft:coarse_dirt": COARSE_DIRT,
    "minecraft:dirt": DIRT,
    "minecraft:stone": STONE,
}


@dataclass(frozen=True)
class MinecraftWorldExport:
    output_dir: Path
    chunks_written: int
    regions_written: int
    min_chunk_x: int
    max_chunk_x: int
    min_chunk_z: int
    max_chunk_z: int
    preview_villagers_written: int = 0


def region_for_chunk(chunk_x: int, chunk_z: int) -> tuple[int, int]:
    return chunk_x >> 5, chunk_z >> 5


def chunk_index_in_region(chunk_x: int, chunk_z: int) -> int:
    return ((chunk_z & 31) << 5) | (chunk_x & 31)


def _region_payload(compressed_nbt: bytes) -> bytes:
    payload = struct.pack(">I", len(compressed_nbt) + 1) + b"\x02" + compressed_nbt
    return payload + bytes((-len(payload)) % SECTOR_SIZE)


def write_region(
    region_x: int,
    region_z: int,
    chunks: dict[tuple[int, int], bytes],
    output_dir: Path,
) -> Path:
    """从 zlib 压缩的区块 NBT 写出一个确定性的 `.mca` 文件。"""

    locations = bytearray(SECTOR_SIZE)
    timestamps = bytearray(SECTOR_SIZE)
    payloads: list[bytes] = []
    next_sector = REGION_HEADER_SIZE // SECTOR_SIZE

    for chunk_x, chunk_z in sorted(chunks, key=lambda pos: chunk_index_in_region(*pos)):
        if region_for_chunk(chunk_x, chunk_z) != (region_x, region_z):
            raise ValueError(f"chunk {(chunk_x, chunk_z)} is outside region {(region_x, region_z)}")
        payload = _region_payload(chunks[(chunk_x, chunk_z)])
        sectors = len(payload) // SECTOR_SIZE
        if sectors > 255:
            raise ValueError(f"chunk {(chunk_x, chunk_z)} exceeds the Anvil 255-sector limit")
        index = chunk_index_in_region(chunk_x, chunk_z)
        locations[index * 4 : index * 4 + 3] = next_sector.to_bytes(3, "big")
        locations[index * 4 + 3] = sectors
        payloads.append(payload)
        next_sector += sectors

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"r.{region_x}.{region_z}.mca"
    with path.open("wb") as output:
        output.write(locations)
        output.write(timestamps)
        for payload in payloads:
            output.write(payload)
    return path


def _surface_blocks(field: Heightfield, sea_level: float) -> np.ndarray:
    tile = to_bong_tile(field, sea_level=sea_level)
    wet = field.water_level >= 0.0
    blocks = np.full(field.height.shape, GRASS_BLOCK, dtype=np.uint8)
    blocks[tile.surface_id == 0] = STONE
    blocks[tile.surface_id == 1] = COARSE_DIRT
    blocks[tile.surface_id == 2] = GRAVEL
    blocks[tile.surface_id == 4] = SAND
    blocks[(tile.wilderness_id == LAKE) | (tile.wilderness_id == RIVER)] = GRAVEL
    if tile.riverbed_palette:
        for material_index, material in enumerate(tile.riverbed_palette):
            material = material.removeprefix("minecraft:").replace("-", "_")
            try:
                block_id = RIVERBED_BLOCKS[material]
            except KeyError as exc:
                raise ValueError(
                    f"river bed material {material!r} has no Minecraft block mapping"
                ) from exc
            blocks[tile.riverbed_id == material_index] = block_id
    # 只有手工构造、没有最终可见层的最小 Heightfield 才使用这个保底映射。
    # 正常生成结果始终携带 surface_visible_palette，不会按固定高度刷雪。
    if not tile.surface_visible_palette:
        blocks[(tile.wilderness_id == MOUNTAINS) & (field.height >= sea_level + 92.0)] = (
            SNOW_BLOCK
        )
    # 覆盖层 palette 内的冰雪材质由独立纵向层叠加；不在其中的材质（例如
    # 冻融产生的 gravel）仍属于原地表，需要在这里写入真实方块。旧 raster
    # 没有覆盖层 palette 时则保持所有 surface_material_id 的兼容映射。
    for material_index, material in enumerate(tile.surface_material_palette, start=1):
        if tile.surface_cover_palette and material in tile.surface_cover_palette:
            continue
        try:
            block_id = GLACIAL_SURFACE_BLOCKS[material]
        except KeyError as exc:
            raise ValueError(
                f"surface material {material!r} has no Minecraft block mapping"
            ) from exc
        blocks[tile.surface_material_id == material_index] = block_id
    # 群山材质独立于普通 surface_material_id；裸岩必须先写入覆盖层下方的
    # 真实基底，才能在没有雪冰覆盖时由 Anvil/BlueMap 正确显示。
    for material_index, material in enumerate(tile.mountain_material_palette, start=1):
        if tile.surface_cover_palette and material in tile.surface_cover_palette:
            continue
        try:
            block_id = GLACIAL_SURFACE_BLOCKS[material]
        except KeyError as exc:
            raise ValueError(
                f"mountain material {material!r} has no Minecraft block mapping"
            ) from exc
        blocks[(tile.mountain_material_id == material_index) & ~wet] = block_id
    for material_index, material in enumerate(tile.permafrost_palette, start=1):
        try:
            block_id = PERMAFROST_BLOCKS[material]
        except KeyError as exc:
            raise ValueError(
                f"permafrost material {material!r} has no Minecraft block mapping"
            ) from exc
        blocks[(tile.permafrost_id == material_index) & ~wet] = block_id
    # 统一的最终可见层只在该列没有垂直覆盖层时作为底层输出；有覆盖层的
    # 列仍需保留真实基底，随后由 Anvil 层堆叠逻辑写出最高方块。
    if tile.surface_visible_palette:
        visible_blocks = {
            "minecraft:air": AIR,
            **GLACIAL_SURFACE_BLOCKS,
            **{f"minecraft:{name}": block_id for name, block_id in RIVERBED_BLOCKS.items()},
            "minecraft:stone": STONE,
            "minecraft:grass_block": GRASS_BLOCK,
            "minecraft:coarse_dirt": COARSE_DIRT,
            "minecraft:gravel": GRAVEL,
            "minecraft:sand": SAND,
            "minecraft:dirt": DIRT,
            "minecraft:mud": MUD,
            "minecraft:clay": CLAY,
            "minecraft:andesite": ANDESITE,
            "minecraft:calcite": CALCITE,
            "minecraft:tuff": TUFF,
            "minecraft:deepslate": DEEPSLATE,
        }
        uncovered = np.sum(tile.surface_cover_layers, axis=0) == 0
        for material_index, material in enumerate(tile.surface_visible_palette, start=1):
            try:
                block_id = visible_blocks[material]
            except KeyError as exc:
                raise ValueError(
                    f"visible surface material {material!r} has no Minecraft block mapping"
                ) from exc
            blocks[(tile.surface_visible_id == material_index) & uncovered] = block_id
    return blocks


def _block_heights(
    field: Heightfield,
    sea_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface = np.clip(
        np.rint(field.height), WORLD_MIN_Y + 1, WORLD_MAX_Y - 1
    ).astype(np.int16)
    # -64..431 是当前维度的合法方块范围；不能为了给覆盖层预留空间而
    # 提前把地表压到 430，否则高山会在顶端形成一条人工平切。
    water_plane = np.ceil(field.water_level).astype(np.int16)
    # 水不能与岸边的地表方块占据同一个 Y 格。此前用
    # ``max(water_plane, surface + 1)`` 会把接近湖岸的水抬高一格：
    # 连续地形 60.8 经四舍五入成为地表 61，水面 61 又被强制放到 62。
    # 这里保留地表方块作为同高岸线，只给确实低于水面的柱体写水。
    wet = (field.water_level >= 0.0) & (surface < water_plane)
    water = np.full(field.height.shape, -1, dtype=np.int16)
    water_flow = np.full(field.height.shape, -1, dtype=np.int8)
    water[wet] = water_plane[wet]
    # 海水/湖水保留 source block；抬高的配方河道使用 Minecraft 流动水状态。
    # 连续水面高度仍通过 Heightfield.water_level 提供给 raster 和查看器。
    river = wet & (field.water_level > sea_level + 1.0e-3)
    water_flow[wet] = 0
    water_flow[river] = 1
    water = np.clip(water, -1, WORLD_MAX_Y - 1).astype(np.int16)
    return surface, water, water_flow


def _fallback_solid_spans(surface: np.ndarray) -> np.ndarray:
    spans = np.full((*surface.shape, 4, 2), 32767, dtype=np.int16)
    spans[:, :, 0] = np.stack((np.full(surface.shape, WORLD_MIN_Y, dtype=np.int16), surface), axis=-1)
    return spans


def _chunk_structure_blocks(
    field: Heightfield,
    *,
    world_min_x: int,
    world_min_z: int,
) -> np.ndarray:
    rows_by_position: dict[tuple[int, int, int], int] = {}
    for block in field.underground_blocks:
        if not (
            world_min_x <= block.x < world_min_x + 16
            and world_min_z <= block.z < world_min_z + 16
        ):
            continue
        material = block.material.strip().lower().removeprefix("minecraft:").replace("-", "_")
        try:
            block_id = UNDERGROUND_BLOCKS[material]
        except KeyError as exc:
            raise ValueError(
                f"underground block material {block.material!r} has no Minecraft block mapping"
            ) from exc
        rows_by_position[(block.x - world_min_x, block.z - world_min_z, block.y)] = block_id
    for block in field.underground_water_blocks:
        if not (
            world_min_x <= block.x < world_min_x + 16
            and world_min_z <= block.z < world_min_z + 16
        ):
            continue
        block_id = FLOWING_WATER_LEVEL_1 if block.flowing else WATER
        rows_by_position[(block.x - world_min_x, block.z - world_min_z, block.y)] = block_id
    for block in field.settlement_blocks:
        if not (
            world_min_x <= block.x < world_min_x + 16
            and world_min_z <= block.z < world_min_z + 16
        ):
            continue
        material = block.material.strip().lower()
        normalized = material.removeprefix("minecraft:").replace("-", "_")
        if "[" not in material and normalized in SETTLEMENT_BLOCKS:
            block_id = SETTLEMENT_BLOCKS[normalized]
        else:
            # Schematic blockstate（楼梯朝向、门、栅栏连接等）由 Anvil
            # 编码器动态注册；未知的 Minecraft 原版方块也无需再维护一张
            # 会不断膨胀的手写映射表。
            try:
                block_id = register_blockstate(material)
            except ValueError as exc:
                raise ValueError(
                    f"settlement block material {block.material!r} is not a valid Minecraft blockstate"
                ) from exc
        rows_by_position[(block.x - world_min_x, block.z - world_min_z, block.y)] = block_id
    if not rows_by_position:
        return np.empty((0, 4), dtype=np.int32)
    rows = [
        (local_x, local_z, block_y, block_id)
        for (local_x, local_z, block_y), block_id in sorted(rows_by_position.items())
    ]
    return np.asarray(rows, dtype=np.int32)


def export_minecraft_world(
    field: Heightfield,
    output_dir: Path,
    *,
    origin_x: int,
    origin_z: int,
    sea_level: float,
    seed: int,
    world_name: str,
    preview_npc_spawns: bool = False,
) -> MinecraftWorldExport:
    """把高度场的每个网格单元写成 Minecraft 1.20.1 世界中的方块。"""

    height, width = field.height.shape
    if width % 16 or height % 16:
        raise ValueError("heightfield width and height must be multiples of 16")
    if origin_x % 16 or origin_z % 16:
        raise ValueError("world origins must be aligned to a 16-block chunk boundary")

    surface_y, water_y, water_flow = _block_heights(field, sea_level)
    surface_blocks = _surface_blocks(field, sea_level)
    solid_spans = field.solid_spans
    if solid_spans is None:
        solid_spans = _fallback_solid_spans(surface_y)
    min_chunk_x = origin_x // 16
    min_chunk_z = origin_z // 16
    max_chunk_x = min_chunk_x + width // 16 - 1
    max_chunk_z = min_chunk_z + height // 16 - 1

    # 预览村民由调用方明确开启，避免普通世界导出提前占用 Server 的 NPC 点位。
    npc_chunks: dict[tuple[int, int], dict[str, SettlementInterestPoint]] = {}
    if preview_npc_spawns:
        for point in field.settlement_interest_points:
            if (
                point.role == "npc_spawn"
                and origin_x <= point.x < origin_x + width
                and origin_z <= point.z < origin_z + height
                and WORLD_MIN_Y <= point.y < WORLD_MAX_Y
            ):
                npc_chunks.setdefault((point.x // 16, point.z // 16), {})[point.point_id] = point

    region_dir = output_dir / "region"
    regions_written = 0
    chunks_written = 0
    min_region_x, min_region_z = region_for_chunk(min_chunk_x, min_chunk_z)
    max_region_x, max_region_z = region_for_chunk(max_chunk_x, max_chunk_z)

    for region_z in range(min_region_z, max_region_z + 1):
        for region_x in range(min_region_x, max_region_x + 1):
            chunks: dict[tuple[int, int], bytes] = {}
            entity_chunks: dict[tuple[int, int], bytes] = {}
            first_x = max(min_chunk_x, region_x * CHUNKS_PER_REGION)
            last_x = min(max_chunk_x, region_x * CHUNKS_PER_REGION + 31)
            first_z = max(min_chunk_z, region_z * CHUNKS_PER_REGION)
            last_z = min(max_chunk_z, region_z * CHUNKS_PER_REGION + 31)
            for chunk_z in range(first_z, last_z + 1):
                local_z = (chunk_z - min_chunk_z) * 16
                for chunk_x in range(first_x, last_x + 1):
                    local_x = (chunk_x - min_chunk_x) * 16
                    z_slice = slice(local_z, local_z + 16)
                    x_slice = slice(local_x, local_x + 16)
                    nbt = encode_chunk_nbt(
                        chunk_x,
                        chunk_z,
                        surface_y[z_slice, x_slice],
                        water_y[z_slice, x_slice],
                        surface_blocks[z_slice, x_slice],
                        water_flow[z_slice, x_slice],
                        solid_spans[z_slice, x_slice],
                        _chunk_structure_blocks(
                            field,
                            world_min_x=origin_x + local_x,
                            world_min_z=origin_z + local_z,
                        ),
                        field.surface_cover_layers[:, z_slice, x_slice],
                    )
                    chunks[(chunk_x, chunk_z)] = zlib.compress(nbt, level=6)
                    if preview_npc_spawns:
                        points = tuple(npc_chunks.get((chunk_x, chunk_z), {}).values())
                        entity_chunks[(chunk_x, chunk_z)] = zlib.compress(
                            encode_preview_villagers(chunk_x, chunk_z, points), level=6
                        )
            write_region(region_x, region_z, chunks, region_dir)
            if preview_npc_spawns:
                # 空实体区块也写出，重复生成后不会遗留已经移走的刷新点。
                write_region(region_x, region_z, entity_chunks, output_dir / "entities")
            regions_written += 1
            chunks_written += len(chunks)

    spawn_x = origin_x + width // 2
    spawn_z = origin_z + height // 2
    spawn_y = int(surface_y[height // 2, width // 2]) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "level.dat").write_bytes(
        encode_level_dat(world_name, seed, spawn_x, spawn_y, spawn_z)
    )
    return MinecraftWorldExport(
        output_dir=output_dir,
        chunks_written=chunks_written,
        regions_written=regions_written,
        min_chunk_x=min_chunk_x,
        max_chunk_x=max_chunk_x,
        min_chunk_z=min_chunk_z,
        max_chunk_z=max_chunk_z,
        preview_villagers_written=sum(len(points) for points in npc_chunks.values()),
    )


__all__ = [
    "MinecraftWorldExport",
    "chunk_index_in_region",
    "export_minecraft_world",
    "region_for_chunk",
    "write_region",
]
