"""在原始 Schematic 的局部坐标中识别室内 NPC 脚点。

分析结果与世界坐标、放置高度和朝向无关；同一模板在城镇中多次复用时，
只需将局部点旋转和平移。仅缓存解码模板的分析结果，不修改原始结构文件。

局部坐标沿用结构加载器归一化后的坐标；Sponge 格式参考：
https://github.com/SpongePowered/Schematic-Specification
室内支撑面与覆盖检测为本项目实现，不是模板格式自带的 NPC 数据。
"""

from __future__ import annotations

from functools import lru_cache
import math

from .catalog import structure_anchor_y
from .schematic import SchematicStructure


_FULL_FLOOR_MATERIALS = frozenset({
    "stone", "cobblestone", "mossy_cobblestone", "smooth_stone",
    "granite", "diorite", "andesite", "polished_granite", "polished_diorite",
    "polished_andesite", "deepslate", "cobbled_deepslate", "polished_deepslate",
    "chiseled_deepslate", "tuff", "calcite", "sandstone", "red_sandstone",
    "smooth_sandstone", "smooth_red_sandstone", "cut_sandstone", "cut_red_sandstone",
    "chiseled_sandstone", "chiseled_red_sandstone", "bricks", "terracotta",
    "quartz_pillar", "smooth_quartz", "blackstone", "polished_blackstone",
    "chiseled_polished_blackstone", "obsidian", "crying_obsidian", "end_stone",
    "prismarine", "dark_prismarine", "dirt", "coarse_dirt", "rooted_dirt",
    "grass_block", "podzol", "mycelium", "clay", "packed_mud", "bedrock",
})
_FULL_FLOOR_SUFFIXES = (
    "_planks", "_bricks", "_tiles", "_wool", "_concrete", "_terracotta", "_ore",
    "_log", "_wood", "_stem", "_hyphae",
)
_TREE_TRUNK_SUFFIXES = ("_log", "_wood", "_stem", "_hyphae")


def _full_floor(blockstate: str) -> bool:
    """保守识别脚部能站在整数 Y 上的支撑面；未知/装饰方块不冒充地板。

    草、地毯、围栏及下半砖等不能按完整一格抬升脚点。上半砖/双半砖可以
    使用其方块顶面；这里无需复制 Minecraft 的完整碰撞形状系统。
    """

    material, _, properties = blockstate.partition("[")
    name = material.removeprefix("minecraft:")
    if name.endswith("_slab"):
        values = properties.rstrip("]").split(",")
        return "type=top" in values or "type=double" in values
    if name.endswith("_block"):
        return name not in {"magma_block", "honey_block", "slime_block"}
    return name in _FULL_FLOOR_MATERIALS or name.endswith(_FULL_FLOOR_SUFFIXES)


def _roof_cover(blockstate: str) -> bool:
    """屋顶允许斜楼梯/半砖，但树冠内的原木不能单独被当作屋顶。"""

    material = blockstate.split("[", 1)[0]
    return not material.endswith(_TREE_TRUNK_SUFFIXES) and (
        _full_floor(blockstate) or material.endswith(("_stairs", "_slab"))
    )


def _scan_interior_feet(
    structure: SchematicStructure,
    first_y: int,
) -> tuple[int, int, int] | None:
    """从模板地板上选一个两格净空、上方有覆盖的局部脚点。

    模板只存非空气体素，逐个检查地板即可。保守识别完整支撑面，并排除
    树冠/树枝作为屋顶，避免庭院抢占室内点；这不是完整的寻路检查。
    """

    occupied = {(block.x, block.y, block.z): block.blockstate for block in structure.blocks}
    supports = {position for position, state in occupied.items() if _full_floor(state)}
    roof_y: dict[tuple[int, int], int] = {}
    for (x, y, z), state in occupied.items():
        if _roof_cover(state):
            roof_y[x, z] = max(roof_y.get((x, z), y), y)

    best: tuple[int, int, float, int, int] | None = None
    for x, floor_y, z in supports:
        y = floor_y + 1
        if not (
            first_y <= y < structure.height - 1
            and 0 < x < structure.width - 1
            and 0 < z < structure.length - 1
        ):
            continue
        if (x, y, z) in occupied or (x, y + 1, z) in occupied:
            continue
        if roof_y.get((x, z), y) <= y + 1:
            continue
        # 首层优先，其次远离模板边界、靠近中心；坐标用于稳定打破平局。
        edge_clearance = min(x, structure.width - 1 - x, z, structure.length - 1 - z)
        center_distance = math.hypot(
            x - (structure.width - 1) / 2.0,
            z - (structure.length - 1) / 2.0,
        )
        score = (y, -edge_clearance, center_distance, x, z)
        if best is None or score < best:
            best = score
    if best is None:
        return None
    y, _, _, x, z = best
    return x, y, z


@lru_cache(maxsize=128)
def schematic_interior_spawn(structure: SchematicStructure) -> tuple[int, int, int] | None:
    """返回未旋转模板中的脚部方块坐标；无候选的结果同样缓存。

    缓存键包含模板尺寸和方块内容；文件更新并重新解码后不会仅凭文件名
    误用旧点位。首层锚点取自模板自身，不把世界地面高度带入分析阶段。
    """

    first_y = max(1, structure_anchor_y(structure.name, structure.blocks) + 1)
    return _scan_interior_feet(structure, first_y)
