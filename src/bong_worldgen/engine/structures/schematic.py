"""Sponge Schematic v2/v3 的最小读取器。

这里直接读取 gzip + NBT，不把 Schematics 变成一份第二套世界数据。结构文件
仍是外部资源，读取后只转换为 ``SchematicBlock``，由 settlement 模块负责
seed 选型、旋转和放置。

格式参考：
* Sponge Schematic specification: https://github.com/SpongePowered/Schematic-Specification
* SpongePowered/Schematic-Specification: https://github.com/SpongePowered/Schematic-Specification
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
from pathlib import Path
import struct


TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


# 这些模板混用了少量新版本或 WorldEdit 别名。目标世界固定为 1.20.1，
# 因此在资源边界做显式替换，生成器内部只看合法的目标版本方块。
MINECRAFT_1201_REPLACEMENTS = {
    "minecraft:bamboo_shelf": "minecraft:bamboo_planks",
    "minecraft:bush": "minecraft:dead_bush",
    "minecraft:cactus_flower": "minecraft:poppy",
    "minecraft:closed_eyeblossom": "minecraft:brown_mushroom",
    "minecraft:firefly_bush": "minecraft:fern",
    "minecraft:iron_chain": "minecraft:chain",
    "minecraft:pale_moss_block": "minecraft:moss_block",
    "minecraft:pale_oak_fence_gate": "minecraft:birch_fence_gate",
    "minecraft:pale_oak_pressure_plate": "minecraft:birch_pressure_plate",
    "minecraft:pale_oak_slab": "minecraft:birch_slab",
    "minecraft:pale_oak_stairs": "minecraft:birch_stairs",
    "minecraft:pale_oak_trapdoor": "minecraft:birch_trapdoor",
    "minecraft:pale_oak_wall_sign": "minecraft:birch_wall_sign",
    "minecraft:polished_tuff": "minecraft:tuff",
    "minecraft:polished_tuff_stairs": "minecraft:stone_brick_stairs",
    "minecraft:short_grass": "minecraft:grass",
    "minecraft:short_dry_grass": "minecraft:dead_bush",
    "minecraft:tuff_slab": "minecraft:stone_brick_slab",
    "minecraft:tuff_stairs": "minecraft:stone_brick_stairs",
    "minecraft:tuff_wall": "minecraft:cobblestone_wall",
    "minecraft:warped_shelf": "minecraft:warped_planks",
    "minecraft:waxed_copper_chest": "minecraft:chest",
    "minecraft:oxidized_copper_chest": "minecraft:chest",
    "minecraft:waxed_exposed_copper_chain": "minecraft:chain",
    "minecraft:waxed_weathered_copper_lantern": "minecraft:lantern",
}


class _NbtReader:
    """读取本地结构资源所需的完整 NBT 基础类型。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = memoryview(payload)
        self._offset = 0

    def _read(self, format_string: str) -> tuple[object, ...]:
        size = struct.calcsize(format_string)
        if self._offset + size > len(self._payload):
            raise ValueError("schematic NBT ended before a complete value was read")
        values = struct.unpack_from(format_string, self._payload, self._offset)
        self._offset += size
        return values

    def byte(self) -> int:
        return int(self._read(">b")[0])

    def unsigned_byte(self) -> int:
        return int(self._read(">B")[0])

    def short(self) -> int:
        return int(self._read(">h")[0])

    def int(self) -> int:
        return int(self._read(">i")[0])

    def long(self) -> int:
        return int(self._read(">q")[0])

    def float(self) -> float:
        return float(self._read(">f")[0])

    def double(self) -> float:
        return float(self._read(">d")[0])

    def string(self) -> str:
        length = int(self._read(">H")[0])
        end = self._offset + length
        if end > len(self._payload):
            raise ValueError("schematic NBT string extends past the payload")
        value = self._payload[self._offset:end].tobytes().decode("utf-8")
        self._offset = end
        return value

    def payload(self, tag: int) -> object:
        if tag == TAG_BYTE:
            return self.byte()
        if tag == TAG_SHORT:
            return self.short()
        if tag == TAG_INT:
            return self.int()
        if tag == TAG_LONG:
            return self.long()
        if tag == TAG_FLOAT:
            return self.float()
        if tag == TAG_DOUBLE:
            return self.double()
        if tag == TAG_BYTE_ARRAY:
            length = self.int()
            if length < 0 or self._offset + length > len(self._payload):
                raise ValueError("invalid schematic NBT byte array length")
            value = self._payload[self._offset : self._offset + length].tobytes()
            self._offset += length
            return value
        if tag == TAG_STRING:
            return self.string()
        if tag == TAG_LIST:
            element_tag = self.unsigned_byte()
            length = self.int()
            if length < 0:
                raise ValueError("invalid schematic NBT list length")
            return [self.payload(element_tag) for _ in range(length)]
        if tag == TAG_COMPOUND:
            return self.compound()
        if tag == TAG_INT_ARRAY:
            length = self.int()
            if length < 0:
                raise ValueError("invalid schematic NBT int array length")
            return [self.int() for _ in range(length)]
        if tag == TAG_LONG_ARRAY:
            length = self.int()
            if length < 0:
                raise ValueError("invalid schematic NBT long array length")
            return [self.long() for _ in range(length)]
        raise ValueError(f"unsupported schematic NBT tag {tag}")

    def compound(self) -> dict[str, object]:
        result: dict[str, object] = {}
        while True:
            tag = self.unsigned_byte()
            if tag == TAG_END:
                return result
            name = self.string()
            result[name] = self.payload(tag)

    def root(self) -> dict[str, object]:
        if self.unsigned_byte() != TAG_COMPOUND:
            raise ValueError("schematic NBT root must be a compound")
        self.string()
        return self.compound()


@dataclass(frozen=True)
class SchematicBlock:
    """结构中的一个非空气方块，坐标以结构原点为基准。"""

    x: int
    y: int
    z: int
    blockstate: str

    @property
    def material(self) -> str:
        """返回不含 blockstate 属性的方块名。"""

        return self.blockstate.split("[", 1)[0]


@dataclass(frozen=True)
class SchematicStructure:
    """已解码的 Sponge 结构。"""

    name: str
    width: int
    height: int
    length: int
    blocks: tuple[SchematicBlock, ...]

    def rotate_local_point(
        self,
        point: tuple[int, int, int],
        quarter_turns: int,
    ) -> tuple[int, int, int]:
        """旋转局部方块索引；结构体素与 NPC 脚点必须使用同一变换。

        输入原点为未旋转模板的最小角，输出原点为旋转后模板的最小角。
        宽长不同时 90°/270° 会交换尺寸；Y 不变。此处不加入实体居中用的
        半格偏移，XZ 的 +0.5 由实体导出层统一处理。
        """

        x, y, z = point
        turns = quarter_turns % 4
        if turns == 1:
            return self.length - 1 - z, y, x
        if turns == 2:
            return self.width - 1 - x, y, self.length - 1 - z
        if turns == 3:
            return z, y, self.width - 1 - x
        return x, y, z

    @lru_cache(maxsize=4)
    def rotated(self, quarter_turns: int) -> "SchematicStructure":
        """按顺时针 90 度的整数倍旋转结构及其方向状态。"""

        turns = quarter_turns % 4
        width, length = self.width, self.length
        transformed: list[SchematicBlock] = []
        for block in self.blocks:
            rotated_x, rotated_y, rotated_z = self.rotate_local_point(
                (block.x, block.y, block.z), turns
            )
            transformed.append(
                SchematicBlock(
                    rotated_x,
                    rotated_y,
                    rotated_z,
                    _rotate_blockstate(block.blockstate, turns),
                )
            )
        if turns % 2:
            width, length = length, width
        return SchematicStructure(
            name=f"{self.name}@rot{turns * 90}",
            width=width,
            height=self.height,
            length=length,
            blocks=tuple(transformed),
        )


def _rotate_blockstate(blockstate: str, turns: int) -> str:
    if turns == 0 or "[" not in blockstate:
        return blockstate
    name, raw_properties = blockstate.split("[", 1)
    properties = raw_properties.rstrip("]").split(",")
    parsed: dict[str, str] = {}
    for property_value in properties:
        if "=" not in property_value:
            continue
        key, value = property_value.split("=", 1)
        parsed[key] = value
    direction_order = ("north", "east", "south", "west")

    def rotate_direction(value: str) -> str:
        if value not in direction_order:
            return value
        return direction_order[(direction_order.index(value) + turns) % 4]

    if "facing" in parsed:
        parsed["facing"] = rotate_direction(parsed["facing"])
    if "axis" in parsed and turns % 2:
        if parsed["axis"] == "x":
            parsed["axis"] = "z"
        elif parsed["axis"] == "z":
            parsed["axis"] = "x"
    if "rotation" in parsed:
        try:
            parsed["rotation"] = str((int(parsed["rotation"]) + turns * 4) % 16)
        except ValueError:
            pass
    directional_values = {key: parsed[key] for key in direction_order if key in parsed}
    for key in directional_values:
        del parsed[key]
    for old_key, value in directional_values.items():
        new_key = direction_order[(direction_order.index(old_key) + turns) % 4]
        parsed[new_key] = value
    return f"{name}[{','.join(f'{key}={value}' for key, value in parsed.items())}]"


def _decode_varints(data: bytes, expected: int) -> list[int]:
    values: list[int] = []
    value = 0
    shift = 0
    for raw in data:
        value |= (raw & 0x7F) << shift
        if raw & 0x80:
            shift += 7
            if shift > 63:
                raise ValueError("schematic BlockData contains an overlong varint")
            continue
        values.append(value)
        value = 0
        shift = 0
        if len(values) == expected:
            break
    if shift:
        raise ValueError("schematic BlockData ends in an incomplete varint")
    if len(values) != expected:
        raise ValueError(
            f"schematic BlockData contains {len(values)} values, expected {expected}"
        )
    return values


def _minecraft_1201_blockstate(blockstate: str) -> str:
    """把已知的新版本方块映射到 Minecraft 1.20.1 等价物。"""

    if "[" in blockstate:
        name, properties = blockstate.split("[", 1)
        replacement = MINECRAFT_1201_REPLACEMENTS.get(name, name)
        # 新版竹架/书架的状态字段在 1.20.1 的普通木板上不存在；
        # 替代材质必须回到默认状态，避免 Anvil 注册出非法 blockstate。
        if name in {"minecraft:bamboo_shelf", "minecraft:warped_shelf"}:
            return replacement
        return f"{replacement}[{properties}"
    return MINECRAFT_1201_REPLACEMENTS.get(blockstate, blockstate)


def _sponge_blocks(root: dict[str, object]) -> tuple[int, int, int, dict[str, int], bytes]:
    """提取 Sponge v2/v3 文档或 WorldEdit v3 包装文档的方块数据。

    WorldEdit 有时会把完整 Sponge v3 文档放进根 ``Schematic`` compound；
    其尺寸和 palette 语义与标准 v3 相同，因此只在资源读取边界展开包装，
    后续坐标和方块状态处理保持完全一致。
    """

    document = root.get("Schematic", root)
    if not isinstance(document, dict):
        raise ValueError("schematic root does not contain a compound document")
    blocks = document.get("Blocks")
    if isinstance(blocks, dict):
        palette_source = blocks.get("Palette")
        block_data_source = blocks.get("Data")
    else:
        palette_source = document.get("Palette")
        block_data_source = document.get("BlockData")
    try:
        width = int(document["Width"])
        height = int(document["Height"])
        length = int(document["Length"])
        palette = {
            str(name): int(index) for name, index in dict(palette_source).items()
        }
        block_data = bytes(block_data_source)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("schematic is missing a valid Sponge palette or dimensions") from exc
    return width, height, length, palette, block_data


@lru_cache(maxsize=256)
def _load_schematic_cached(
    resolved_path: str,
    modified_ns: int,
    file_size: int,
) -> SchematicStructure:
    """按路径和文件指纹缓存解析结果；尺寸和 palette 每进程只读取一次。"""

    del modified_ns, file_size
    source = Path(resolved_path)
    try:
        with gzip.open(source, "rb") as stream:
            root = _NbtReader(stream.read()).root()
    except OSError as exc:
        raise ValueError(f"could not read schematic {source}") from exc
    try:
        width, height, length, palette, block_data = _sponge_blocks(root)
    except ValueError as exc:
        raise ValueError(
            f"schematic {source} is missing a valid Sponge palette or dimensions"
        ) from exc
    if width < 1 or height < 1 or length < 1:
        raise ValueError(f"schematic {source} has non-positive dimensions")
    expected = width * height * length
    indexes = _decode_varints(block_data, expected)
    by_index = {index: state for state, index in palette.items()}
    blocks: list[SchematicBlock] = []
    for index, palette_index in enumerate(indexes):
        blockstate = by_index.get(palette_index)
        if blockstate is None:
            raise ValueError(f"schematic {source} references unknown palette index {palette_index}")
        blockstate = _minecraft_1201_blockstate(blockstate)
        if blockstate.split("[", 1)[0] in {
            "minecraft:air",
            "minecraft:cave_air",
            "minecraft:void_air",
        }:
            continue
        x = index % width
        z = (index // width) % length
        y = index // (width * length)
        blocks.append(SchematicBlock(x, y, z, blockstate))
    return SchematicStructure(source.stem, width, height, length, tuple(blocks))


def load_schematic(path: Path) -> SchematicStructure:
    """读取并缓存一个 gzip 压缩的 Sponge ``.schem`` 文件。"""

    source = Path(path).resolve()
    try:
        stat = source.stat()
    except OSError as exc:
        raise ValueError(f"could not read schematic {source}") from exc
    return _load_schematic_cached(str(source), stat.st_mtime_ns, stat.st_size)


def load_schematic_directory(directory: Path) -> tuple[SchematicStructure, ...]:
    """按文件名稳定排序读取一个结构目录。"""

    source = Path(directory)
    if not source.is_dir():
        raise ValueError(f"schematic directory does not exist: {source}")
    paths = tuple(sorted(source.glob("*.schem"), key=lambda item: item.name.lower()))
    if not paths:
        raise ValueError(f"schematic directory contains no .schem files: {source}")
    return tuple(load_schematic(path) for path in paths)


def _litematic_blockstate(entry: object) -> str:
    """把 Litematica palette compound 转成 Sponge 使用的 blockstate 字符串。"""

    if not isinstance(entry, dict) or "Name" not in entry:
        raise ValueError("litematic palette contains an invalid blockstate")
    name = str(entry["Name"])
    properties = entry.get("Properties")
    if not isinstance(properties, dict) or not properties:
        return _minecraft_1201_blockstate(name)
    raw = ",".join(f"{key}={value}" for key, value in sorted(properties.items()))
    return _minecraft_1201_blockstate(f"{name}[{raw}]")


def load_litematic(path: Path) -> SchematicStructure:
    """读取单区域 Litematica 文件并转换为统一结构。

    Litematica 使用按位打包的 long 数组；这里按官方区域线性索引
    ``x + z * size_x + y * size_x * size_z`` 解码。多区域文件会合并到同一
    footprint，区域位置仅用于稳定排序，不改变相对模板原点。
    """

    source = Path(path).resolve()
    try:
        stat = source.stat()
        return _load_litematic_cached(str(source), stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        raise ValueError(f"could not read litematic {source}") from exc


@lru_cache(maxsize=128)
def _load_litematic_cached(
    resolved_path: str,
    modified_ns: int,
    file_size: int,
) -> SchematicStructure:
    del modified_ns, file_size
    source = Path(resolved_path)
    try:
        with gzip.open(source, "rb") as stream:
            root = _NbtReader(stream.read()).root()
    except OSError as exc:
        raise ValueError(f"could not read litematic {source}") from exc
    regions = root.get("Regions")
    if not isinstance(regions, dict) or not regions:
        raise ValueError(f"litematic {source} contains no regions")
    decoded: list[SchematicBlock] = []
    extent_min: tuple[int, int, int] | None = None
    extent_max: tuple[int, int, int] | None = None
    for region_name, region in sorted(regions.items(), key=lambda item: str(item[0]).lower()):
        if not isinstance(region, dict):
            raise ValueError(f"litematic {source} region is invalid")
        size = region.get("Size")
        position = region.get("Position")
        palette = region.get("BlockStatePalette")
        packed = region.get("BlockStates")
        if not (
            isinstance(size, dict)
            and isinstance(position, dict)
            and isinstance(palette, list)
            and isinstance(packed, list)
        ):
            raise ValueError(f"litematic {source} region is missing size, palette, or block states")
        signed_size_x, signed_size_y, signed_size_z = (int(size[key]) for key in ("x", "y", "z"))
        size_x, size_y, size_z = (
            abs(value) for value in (signed_size_x, signed_size_y, signed_size_z)
        )
        if min(size_x, size_y, size_z) < 1:
            continue
        pos_x, pos_y, pos_z = (int(position[key]) for key in ("x", "y", "z"))
        region_min = tuple(
            position_value if signed_size >= 0 else position_value - abs(signed_size) + 1
            for position_value, signed_size in (
                (pos_x, signed_size_x),
                (pos_y, signed_size_y),
                (pos_z, signed_size_z),
            )
        )
        region_max = tuple(
            position_value + abs(signed_size) - 1 if signed_size >= 0 else position_value
            for position_value, signed_size in (
                (pos_x, signed_size_x),
                (pos_y, signed_size_y),
                (pos_z, signed_size_z),
            )
        )
        extent_min = (
            region_min
            if extent_min is None
            else tuple(min(a, b) for a, b in zip(extent_min, region_min))
        )
        extent_max = (
            region_max
            if extent_max is None
            else tuple(max(a, b) for a, b in zip(extent_max, region_max))
        )
        states = tuple(_litematic_blockstate(item) for item in palette)
        bits = max(2, (len(states) - 1).bit_length())
        mask = (1 << bits) - 1
        unsigned_words = tuple(int(word) & ((1 << 64) - 1) for word in packed)
        total = size_x * size_y * size_z
        for index in range(total):
            bit_index = index * bits
            word_index = bit_index >> 6
            offset = bit_index & 63
            value = (unsigned_words[word_index] >> offset) & mask
            if offset + bits > 64:
                value |= (unsigned_words[word_index + 1] << (64 - offset)) & mask
            if value >= len(states):
                raise ValueError(f"litematic {source} contains an unknown palette index")
            state = states[value]
            if state.split("[", 1)[0] in {
                "minecraft:air",
                "minecraft:cave_air",
                "minecraft:void_air",
            }:
                continue
            local_x = index % size_x
            local_z = (index // size_x) % size_z
            local_y = index // (size_x * size_z)
            # Litematica 的容器始终按包围盒最小角存储局部坐标；负尺寸只表示
            # Position 是该轴的另一端点，并不表示 BlockStates 要按该轴反向读取。
            # 参考官方读取/放置流程：
            # https://github.com/maruohon/litematica/blob/master/src/main/java/litematica/schematic/LitematicaSchematic.java
            world_x = region_min[0] + local_x
            world_y = region_min[1] + local_y
            world_z = region_min[2] + local_z
            decoded.append(SchematicBlock(world_x, world_y, world_z, state))
    if not decoded:
        raise ValueError(f"litematic {source} contains no non-air blocks")
    if extent_min is None or extent_max is None:
        raise ValueError(f"litematic {source} contains no valid regions")
    min_x, min_y, min_z = extent_min
    max_x, max_y, max_z = extent_max
    normalized = tuple(
        SchematicBlock(block.x - min_x, block.y - min_y, block.z - min_z, block.blockstate)
        for block in decoded
    )
    return SchematicStructure(
        source.stem,
        max_x - min_x + 1,
        max_y - min_y + 1,
        max_z - min_z + 1,
        normalized,
    )


def load_structure_directory(directory: Path) -> tuple[SchematicStructure, ...]:
    """读取目录中的 Sponge ``.schem`` 和 Litematica ``.litematic`` 模板。"""

    source = Path(directory)
    if not source.is_dir():
        raise ValueError(f"structure directory does not exist: {source}")
    paths = tuple(
        sorted(
            (*source.glob("*.schem"), *source.glob("*.litematic")),
            key=lambda item: item.name.lower(),
        )
    )
    if not paths:
        raise ValueError(f"structure directory contains no supported structure files: {source}")
    return tuple(
        load_schematic(path) if path.suffix.lower() == ".schem" else load_litematic(path)
        for path in paths
    )


__all__ = [
    "SchematicBlock",
    "SchematicStructure",
    "load_schematic",
    "load_schematic_directory",
    "load_litematic",
    "load_structure_directory",
]
