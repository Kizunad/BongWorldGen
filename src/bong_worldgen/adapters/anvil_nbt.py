"""Anvil 适配器使用的 Minecraft 1.20.1 最小 NBT 编码器。"""

from __future__ import annotations

import gzip
from io import BytesIO
import struct
from typing import Callable, Iterable

import numpy as np


DATA_VERSION = 3465
# Bong server 的 overworld 使用 -64..431（496 格），与
# server/src/world/terrain/mod.rs 的 WORLD_HEIGHT/MIN_Y 保持一致。
WORLD_MIN_Y = -64
WORLD_MAX_Y = WORLD_MIN_Y + 496
SECTION_HEIGHT = 16
CHUNK_WIDTH = 16

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12

AIR = 0
STONE = 1
BEDROCK = 2
GRASS_BLOCK = 3
COARSE_DIRT = 4
GRAVEL = 5
WATER = 6
SNOW_BLOCK = 7
DIRT = 8
MUD = 9
SAND = 10
CLAY = 11
PACKED_MUD = 12
MUD_BRICKS = 13
FLOWING_WATER_LEVEL_1 = 14
OAK_LOG = 15
OAK_PLANKS = 16
CHEST = 17
TORCH = 18
COAL_ORE = 19
IRON_ORE = 20
COPPER_ORE = 21
GLOW_LICHEN = 22
MOSS_BLOCK = 23
SPORE_BLOSSOM = 24
ICE = 25
PACKED_ICE = 26
BLUE_ICE = 27
POWDER_SNOW = 28
ANDESITE = 29
CALCITE = 30
TUFF = 31
DEEPSLATE = 32
DEAD_BUSH = 33
STONE_BRICKS = 34

BLOCK_NAMES = (
    "minecraft:air",
    "minecraft:stone",
    "minecraft:bedrock",
    "minecraft:grass_block",
    "minecraft:coarse_dirt",
    "minecraft:gravel",
    "minecraft:water",
    "minecraft:snow_block",
    "minecraft:dirt",
    "minecraft:mud",
    "minecraft:sand",
    "minecraft:clay",
    "minecraft:packed_mud",
    "minecraft:mud_bricks",
    "minecraft:water",
    "minecraft:oak_log",
    "minecraft:oak_planks",
    "minecraft:chest",
    "minecraft:torch",
    "minecraft:coal_ore",
    "minecraft:iron_ore",
    "minecraft:copper_ore",
    "minecraft:glow_lichen",
    "minecraft:moss_block",
    "minecraft:spore_blossom",
    "minecraft:ice",
    "minecraft:packed_ice",
    "minecraft:blue_ice",
    "minecraft:powder_snow",
    "minecraft:andesite",
    "minecraft:calcite",
    "minecraft:tuff",
    "minecraft:deepslate",
    "minecraft:dead_bush",
    "minecraft:stone_bricks",
)

# 导入的 Schematic 会携带完整 blockstate。基础地形继续使用上面的固定 ID，
# 结构方块则按首次出现顺序注册到扩展 palette；最终每个 section 仍写成
# Minecraft 原生的局部 palette，不会把这些内部 ID 暴露到世界文件。
_DYNAMIC_BLOCK_STATES: list[tuple[str, dict[str, str] | None]] = []
_DYNAMIC_BLOCK_IDS: dict[str, int] = {}


def _split_blockstate(blockstate: str) -> tuple[str, dict[str, str] | None]:
    state = blockstate.strip().lower()
    if not state.startswith("minecraft:"):
        state = f"minecraft:{state}"
    if "[" not in state:
        return state, None
    if not state.endswith("]"):
        raise ValueError(f"invalid Minecraft blockstate {blockstate!r}")
    name, raw_properties = state[:-1].split("[", 1)
    properties: dict[str, str] = {}
    for item in raw_properties.split(","):
        if not item or "=" not in item:
            raise ValueError(f"invalid Minecraft blockstate property in {blockstate!r}")
        key, value = item.split("=", 1)
        properties[key] = value
    return name, properties or None


def register_blockstate(blockstate: str) -> int:
    """把结构 blockstate 注册为 Anvil 编码器内部 ID。"""

    name, properties = _split_blockstate(blockstate)
    canonical = name
    if properties:
        canonical += "[" + ",".join(
            f"{key}={value}" for key, value in sorted(properties.items())
        ) + "]"
    existing = _DYNAMIC_BLOCK_IDS.get(canonical)
    if existing is not None:
        return existing
    block_id = len(BLOCK_NAMES) + len(_DYNAMIC_BLOCK_STATES)
    _DYNAMIC_BLOCK_IDS[canonical] = block_id
    _DYNAMIC_BLOCK_STATES.append((name, properties))
    return block_id


def _registered_blockstate(block_id: int) -> tuple[str, dict[str, str] | None]:
    if 0 <= block_id < len(BLOCK_NAMES):
        return BLOCK_NAMES[block_id], _block_properties(block_id)
    dynamic_index = block_id - len(BLOCK_NAMES)
    if not 0 <= dynamic_index < len(_DYNAMIC_BLOCK_STATES):
        raise ValueError(f"unknown Anvil block id {block_id}")
    return _DYNAMIC_BLOCK_STATES[dynamic_index]


def _block_properties(block_id: int) -> dict[str, str] | None:
    """返回非默认方块 palette 项需要写出的显式状态属性。"""

    if block_id == GRASS_BLOCK:
        # BlueMap 按 blockstate 条件选择模型；省略 snowy 会无法匹配
        # grass_block.json 的 snowy=false 变体，导致渲染时露出下方石块。
        return {"snowy": "false"}
    if block_id == FLOWING_WATER_LEVEL_1:
        return {"level": "1"}
    if block_id == GLOW_LICHEN:
        # Glow lichen 的默认状态可能没有任何附着面，渲染器会把它当作
        # 不可见方块。洞穴占位点已经保证它落在洞壁上，这里显式打开六面
        # 使 BlueMap 和 Minecraft 都能看到植物；后续 Server 可按真实墙面
        # 朝向收窄为单面状态。
        return {
            "down": "true",
            "east": "true",
            "north": "true",
            "south": "true",
            "up": "true",
            "waterlogged": "false",
            "west": "true",
        }
    return None


def _byte(buffer: BytesIO, value: int) -> None:
    buffer.write(struct.pack(">b", value))


def _int(buffer: BytesIO, value: int) -> None:
    buffer.write(struct.pack(">i", value))


def _long(buffer: BytesIO, value: int) -> None:
    buffer.write(struct.pack(">q", value))


def _string(buffer: BytesIO, value: str) -> None:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFF:
        raise ValueError("NBT strings cannot exceed 65535 encoded bytes")
    buffer.write(struct.pack(">H", len(encoded)))
    buffer.write(encoded)


def _named(buffer: BytesIO, tag: int, name: str) -> None:
    _byte(buffer, tag)
    _string(buffer, name)


def _end(buffer: BytesIO) -> None:
    _byte(buffer, TAG_END)


def _list_header(buffer: BytesIO, element_tag: int, count: int) -> None:
    _byte(buffer, element_tag)
    _int(buffer, count)


def _long_array(buffer: BytesIO, values: Iterable[int]) -> None:
    items = list(values)
    _int(buffer, len(items))
    for value in items:
        _long(buffer, value)


def _signed_long(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _pack_values(values: Iterable[int], bits: int) -> list[int]:
    if bits < 1 or bits > 64:
        raise ValueError("packed value width must be in [1, 64]")
    per_long = 64 // bits
    mask = (1 << bits) - 1
    packed: list[int] = []
    current = 0
    used = 0
    for value in values:
        value = int(value)
        if value < 0 or value > mask:
            raise ValueError(f"value {value} does not fit in {bits} bits")
        current |= value << (used * bits)
        used += 1
        if used == per_long:
            packed.append(_signed_long(current))
            current = 0
            used = 0
    if used:
        packed.append(_signed_long(current))
    return packed


def _write_palette(buffer: BytesIO, block_ids: np.ndarray) -> None:
    palette_ids = np.unique(block_ids)
    _named(buffer, TAG_LIST, "palette")
    _list_header(buffer, TAG_COMPOUND, len(palette_ids))
    for block_id in palette_ids:
        _named(buffer, TAG_STRING, "Name")
        block_id = int(block_id)
        name, properties = _registered_blockstate(block_id)
        _string(buffer, name)
        if properties is not None:
            _named(buffer, TAG_COMPOUND, "Properties")
            for name, value in properties.items():
                _named(buffer, TAG_STRING, name)
                _string(buffer, value)
            _end(buffer)
        _end(buffer)

    if len(palette_ids) > 1:
        bits = max(4, (len(palette_ids) - 1).bit_length())
        indexes = np.searchsorted(palette_ids, block_ids.reshape(-1))
        _named(buffer, TAG_LONG_ARRAY, "data")
        _long_array(buffer, _pack_values(indexes, bits))
    _end(buffer)


def _write_biomes(buffer: BytesIO) -> None:
    _named(buffer, TAG_LIST, "palette")
    _list_header(buffer, TAG_STRING, 1)
    _string(buffer, "minecraft:plains")
    _end(buffer)


def _section_blocks(
    section_y: int,
    surface_y: np.ndarray,
    water_y: np.ndarray,
    surface_blocks: np.ndarray,
    water_flow: np.ndarray | None = None,
    solid_spans: np.ndarray | None = None,
    structure_blocks: np.ndarray | None = None,
    surface_cover_layers: np.ndarray | None = None,
) -> np.ndarray:
    world_y = (
        section_y * SECTION_HEIGHT + np.arange(SECTION_HEIGHT, dtype=np.int16)
    )[:, None, None]
    heights = surface_y[None, :, :]
    water = water_y[None, :, :]

    if solid_spans is None:
        blocks = np.where(world_y < heights, STONE, AIR).astype(np.uint16)
    else:
        blocks = np.full((SECTION_HEIGHT, *surface_y.shape), AIR, dtype=np.uint16)
        for slot in range(solid_spans.shape[2]):
            floor = solid_spans[:, :, slot, 0][None, :, :]
            ceiling = solid_spans[:, :, slot, 1][None, :, :]
            valid = (floor != 32767) & (ceiling != 32767)
            blocks = np.where(valid & (world_y >= floor) & (world_y <= ceiling), STONE, blocks)
    # 洞穴入口的顶层实心 span 可能低于 surface_y；此时 surface_y 是
    # 空气，而不是应该被恢复的地表方块。只有确实有 span 覆盖到地表的
    # 列才写回 surface_blocks，避免把入口重新封死。
    if solid_spans is None:
        surface_columns = np.ones(surface_y.shape, dtype=bool)
    else:
        top_ceiling = solid_spans[:, :, 0, 1]
        surface_columns = top_ceiling == surface_y
    blocks = np.where(
        (world_y == heights) & surface_columns[None, :, :],
        surface_blocks[None, :, :],
        blocks,
    )
    if surface_cover_layers is not None:
        if surface_cover_layers.shape != (4, *surface_y.shape):
            raise ValueError(
                "surface_cover_layers must have shape (4, height, width) for a chunk"
            )
        # 数组顺序是 powder/snow/ice/blue，实际堆叠从底部 blue 向上到
        # 顶部 powder。每一层都只占用原地形上方的格子，因此不会覆盖石头。
        cursor = np.zeros(surface_y.shape, dtype=np.int16)
        for material_index, block_id in (
            (3, BLUE_ICE),
            (2, ICE),
            (1, SNOW_BLOCK),
            (0, POWDER_SNOW),
        ):
            count = surface_cover_layers[material_index].astype(np.int16, copy=False)
            # ``heights`` 已经带有广播用的首轴；这里改用二维高度，
            # 避免再次加轴后让 blocks 变成四维数组。
            start = surface_y + 1 + cursor
            end = start + count
            cover_mask = (
                surface_columns[None, :, :]
                & (count[None, :, :] > 0)
                & (world_y >= start[None, :, :])
                & (world_y < end[None, :, :])
            )
            blocks = np.where(cover_mask, block_id, blocks)
            cursor += count
    water_mask = (water >= 0) & (world_y > heights) & (world_y <= water)
    if water_flow is None:
        water_ids = np.full(blocks.shape, WATER, dtype=np.uint16)
    else:
        flowing = water_flow[None, :, :] == 1
        water_ids = np.where(flowing, FLOWING_WATER_LEVEL_1, WATER).astype(np.uint16)
    blocks = np.where(water_mask, water_ids, blocks)
    if structure_blocks is not None:
        for local_x, local_z, block_y, block_id in structure_blocks:
            if 0 <= local_x < CHUNK_WIDTH and 0 <= local_z < CHUNK_WIDTH:
                section_floor = section_y * SECTION_HEIGHT
                if section_floor <= block_y < section_floor + SECTION_HEIGHT:
                    blocks[int(block_y - section_floor), int(local_z), int(local_x)] = int(block_id)
    blocks = np.where(world_y == WORLD_MIN_Y, BEDROCK, blocks)
    return blocks


def _write_section(
    buffer: BytesIO,
    section_y: int,
    surface_y: np.ndarray,
    water_y: np.ndarray,
    surface_blocks: np.ndarray,
    water_flow: np.ndarray | None = None,
    solid_spans: np.ndarray | None = None,
    structure_blocks: np.ndarray | None = None,
    surface_cover_layers: np.ndarray | None = None,
) -> None:
    _named(buffer, TAG_BYTE, "Y")
    _byte(buffer, section_y)
    _named(buffer, TAG_COMPOUND, "block_states")
    _write_palette(
        buffer,
        _section_blocks(
            section_y,
            surface_y,
            water_y,
            surface_blocks,
            water_flow,
            solid_spans,
            structure_blocks,
            surface_cover_layers,
        ),
    )
    _named(buffer, TAG_COMPOUND, "biomes")
    _write_biomes(buffer)
    _end(buffer)


def _heightmap(values: np.ndarray) -> list[int]:
    stored = values.astype(np.int32).reshape(-1) + 1 - WORLD_MIN_Y
    if np.any(stored < 0) or np.any(stored >= 512):
        raise ValueError("heightmap value is outside the Minecraft 1.20.1 build range")
    return _pack_values(stored, 9)


def _validate_chunk_arrays(
    surface_y: np.ndarray,
    water_y: np.ndarray,
    surface_blocks: np.ndarray,
    water_flow: np.ndarray | None = None,
    solid_spans: np.ndarray | None = None,
    structure_blocks: np.ndarray | None = None,
    surface_cover_layers: np.ndarray | None = None,
) -> None:
    expected = (CHUNK_WIDTH, CHUNK_WIDTH)
    for name, values in (
        ("surface_y", surface_y),
        ("water_y", water_y),
        ("surface_blocks", surface_blocks),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if water_flow is not None and water_flow.shape != expected:
        raise ValueError(f"water_flow must have shape {expected}, got {water_flow.shape}")
    if water_flow is not None and not np.isin(water_flow, (-1, 0, 1)).all():
        raise ValueError("water_flow must contain only -1, 0, or 1")
    if solid_spans is not None:
        expected_spans = (*expected, 4, 2)
        if solid_spans.shape != expected_spans:
            raise ValueError(f"solid_spans must have shape {expected_spans}, got {solid_spans.shape}")
    if surface_cover_layers is not None:
        expected_cover = (4, *expected)
        if surface_cover_layers.shape != expected_cover:
            raise ValueError(
                f"surface_cover_layers must have shape {expected_cover}, got {surface_cover_layers.shape}"
            )
        if not np.issubdtype(surface_cover_layers.dtype, np.integer):
            raise ValueError("surface_cover_layers must contain integer layer counts")
    if structure_blocks is not None:
        if structure_blocks.ndim != 2 or structure_blocks.shape[1] != 4:
            raise ValueError("structure_blocks must have shape (N, 4): local_x, local_z, y, block_id")
        if len(structure_blocks) and np.any(structure_blocks[:, 3] < 0):
            raise ValueError("structure_blocks contains a negative block id")
        if len(structure_blocks) and np.any(
            (structure_blocks[:, 2] < WORLD_MIN_Y)
            | (structure_blocks[:, 2] >= WORLD_MAX_Y)
        ):
            raise ValueError("structure_blocks contains a y coordinate outside the world height")
    if np.any(surface_y < WORLD_MIN_Y) or np.any(surface_y >= WORLD_MAX_Y):
        raise ValueError("surface_y is outside the Minecraft 1.20.1 build range")
    if np.any(water_y >= WORLD_MAX_Y):
        raise ValueError("water_y is outside the Minecraft 1.20.1 build range")
    allowed_surface_blocks = (
        AIR,
        STONE,
        GRASS_BLOCK,
        COARSE_DIRT,
        GRAVEL,
        SNOW_BLOCK,
        DIRT,
        MUD,
        SAND,
        CLAY,
        PACKED_MUD,
            MUD_BRICKS,
            ICE,
            PACKED_ICE,
            BLUE_ICE,
            POWDER_SNOW,
            ANDESITE,
            CALCITE,
            TUFF,
            DEEPSLATE,
        )
    if not np.isin(surface_blocks, allowed_surface_blocks).all():
        raise ValueError("surface_blocks contains an unsupported block id")


def encode_chunk_nbt(
    chunk_x: int,
    chunk_z: int,
    surface_y: np.ndarray,
    water_y: np.ndarray,
    surface_blocks: np.ndarray,
    water_flow: np.ndarray | None = None,
    solid_spans: np.ndarray | None = None,
    structure_blocks: np.ndarray | None = None,
    surface_cover_layers: np.ndarray | None = None,
) -> bytes:
    """把一个完整的 1.20.1 区块编码为未压缩 NBT。"""

    if not isinstance(chunk_x, int) or not isinstance(chunk_z, int):
        raise TypeError("chunk coordinates must be integers")
    _validate_chunk_arrays(
        surface_y,
        water_y,
        surface_blocks,
        water_flow,
        solid_spans,
        structure_blocks,
        surface_cover_layers,
    )

    top_y = np.maximum(surface_y, water_y)
    if surface_cover_layers is not None:
        if solid_spans is None:
            cover_columns = np.ones(surface_y.shape, dtype=bool)
        else:
            cover_columns = solid_spans[:, :, 0, 1] == surface_y
        cover_top = np.where(
            cover_columns,
            np.minimum(
                surface_y + np.sum(surface_cover_layers, axis=0),
                WORLD_MAX_Y - 1,
            ),
            surface_y,
        )
        top_y = np.maximum(top_y, cover_top)
    if structure_blocks is not None:
        # section 列表与 WORLD_SURFACE 高度图必须包含建筑最高方块。此前这里只
        # 看地形、水面和冰雪覆盖，地表 Y=92 时最多写出 80..95 section，
        # 导致位于 Y=96 以上的屋顶在 Anvil 文件中被整段截断。
        top_y = top_y.copy()
        for local_x, local_z, block_y, _ in structure_blocks:
            if 0 <= local_x < CHUNK_WIDTH and 0 <= local_z < CHUNK_WIDTH:
                top_y[int(local_z), int(local_x)] = max(
                    int(top_y[int(local_z), int(local_x)]),
                    int(block_y),
                )
    max_section_y = min(
        (WORLD_MAX_Y - 1) // SECTION_HEIGHT,
        int(top_y.max()) // SECTION_HEIGHT,
    )
    section_ys = range(WORLD_MIN_Y // SECTION_HEIGHT, max_section_y + 1)

    buffer = BytesIO()
    _named(buffer, TAG_COMPOUND, "")
    for name, value in (("DataVersion", DATA_VERSION), ("xPos", chunk_x), ("yPos", -4), ("zPos", chunk_z)):
        _named(buffer, TAG_INT, name)
        _int(buffer, value)
    _named(buffer, TAG_STRING, "Status")
    _string(buffer, "minecraft:full")
    for name in ("LastUpdate", "InhabitedTime"):
        _named(buffer, TAG_LONG, name)
        _long(buffer, 0)
    _named(buffer, TAG_BYTE, "isLightOn")
    _byte(buffer, 0)

    _named(buffer, TAG_LIST, "sections")
    _list_header(buffer, TAG_COMPOUND, len(section_ys))
    for section_y in section_ys:
        _write_section(
            buffer,
            section_y,
            surface_y,
            water_y,
            surface_blocks,
            water_flow,
            solid_spans,
            structure_blocks,
            surface_cover_layers,
        )

    for name in ("block_entities", "block_ticks", "fluid_ticks"):
        _named(buffer, TAG_LIST, name)
        _list_header(buffer, TAG_END, 0)

    _named(buffer, TAG_COMPOUND, "Heightmaps")
    ground_map = _heightmap(surface_y)
    surface_map = _heightmap(top_y)
    for name in ("OCEAN_FLOOR", "OCEAN_FLOOR_WG"):
        _named(buffer, TAG_LONG_ARRAY, name)
        _long_array(buffer, ground_map)
    for name in (
        "MOTION_BLOCKING",
        "MOTION_BLOCKING_NO_LEAVES",
        "WORLD_SURFACE",
        "WORLD_SURFACE_WG",
    ):
        _named(buffer, TAG_LONG_ARRAY, name)
        _long_array(buffer, surface_map)
    _end(buffer)
    _end(buffer)
    return buffer.getvalue()


def encode_level_dat(
    world_name: str,
    seed: int,
    spawn_x: int,
    spawn_y: int,
    spawn_z: int,
) -> bytes:
    """编码 BlueMap 识别世界所需的最小 level.dat 子集。"""

    buffer = BytesIO()
    _named(buffer, TAG_COMPOUND, "")
    _named(buffer, TAG_COMPOUND, "Data")
    _named(buffer, TAG_INT, "DataVersion")
    _int(buffer, DATA_VERSION)
    _named(buffer, TAG_STRING, "LevelName")
    _string(buffer, world_name)
    _named(buffer, TAG_LONG, "RandomSeed")
    _long(buffer, seed)
    for name, value in (("SpawnX", spawn_x), ("SpawnY", spawn_y), ("SpawnZ", spawn_z)):
        _named(buffer, TAG_INT, name)
        _int(buffer, value)
    _named(buffer, TAG_COMPOUND, "Version")
    _named(buffer, TAG_INT, "Id")
    _int(buffer, DATA_VERSION)
    _named(buffer, TAG_STRING, "Name")
    _string(buffer, "1.20.1")
    _named(buffer, TAG_STRING, "Series")
    _string(buffer, "main")
    _named(buffer, TAG_BYTE, "Snapshot")
    _byte(buffer, 0)
    _end(buffer)
    _end(buffer)
    _end(buffer)
    return gzip.compress(buffer.getvalue(), mtime=0)


def read_root_compound(data: bytes) -> dict[str, object]:
    """读取本模块写出的 NBT 子集，用于校验和测试。"""

    source = BytesIO(data)

    def read_exact(size: int) -> bytes:
        value = source.read(size)
        if len(value) != size:
            raise ValueError("truncated NBT payload")
        return value

    def read_byte() -> int:
        return struct.unpack(">b", read_exact(1))[0]

    def read_int() -> int:
        return struct.unpack(">i", read_exact(4))[0]

    def read_long() -> int:
        return struct.unpack(">q", read_exact(8))[0]

    def read_string() -> str:
        size = struct.unpack(">H", read_exact(2))[0]
        return read_exact(size).decode("utf-8")

    def read_payload(tag: int) -> object:
        readers: dict[int, Callable[[], object]] = {
            TAG_BYTE: read_byte,
            TAG_INT: read_int,
            TAG_LONG: read_long,
            TAG_FLOAT: lambda: struct.unpack(">f", read_exact(4))[0],
            TAG_DOUBLE: lambda: struct.unpack(">d", read_exact(8))[0],
            TAG_STRING: read_string,
            TAG_COMPOUND: read_compound,
        }
        if tag == TAG_LIST:
            element_tag = read_byte()
            return [read_payload(element_tag) for _ in range(read_int())]
        if tag == TAG_LONG_ARRAY:
            return [read_long() for _ in range(read_int())]
        if tag == TAG_INT_ARRAY:
            return [read_int() for _ in range(read_int())]
        try:
            return readers[tag]()
        except KeyError as error:
            raise ValueError(f"unsupported NBT tag {tag}") from error

    def read_compound() -> dict[str, object]:
        result: dict[str, object] = {}
        while True:
            tag = read_byte()
            if tag == TAG_END:
                return result
            name = read_string()
            result[name] = read_payload(tag)

    if read_byte() != TAG_COMPOUND or read_string() != "":
        raise ValueError("expected an unnamed root compound")
    return read_compound()


__all__ = [
    "BLOCK_NAMES",
    "COARSE_DIRT",
    "DATA_VERSION",
    "FLOWING_WATER_LEVEL_1",
    "GRASS_BLOCK",
    "GRAVEL",
    "SNOW_BLOCK",
    "ICE",
    "PACKED_ICE",
    "BLUE_ICE",
    "DEAD_BUSH",
    "POWDER_SNOW",
    "STONE",
    "WATER",
    "WORLD_MAX_Y",
    "WORLD_MIN_Y",
    "encode_chunk_nbt",
    "encode_level_dat",
    "read_root_compound",
    "register_blockstate",
]
