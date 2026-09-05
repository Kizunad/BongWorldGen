"""地形引擎跨模块共享的稳定常量。"""

from __future__ import annotations


DEFAULT_RIVERBED_MATERIALS = ("dirt", "mud", "gravel", "sand", "clay")
# 群山裸岩的受支持 Minecraft 1.20.1 方块；配方可以选择其中的子集。
MOUNTAIN_ROCK_PALETTE = (
    "minecraft:stone",
    "minecraft:andesite",
    "minecraft:calcite",
    "minecraft:tuff",
    "minecraft:deepslate",
)
SPAN_MIN_Y = -64
SPAN_MAX_SPANS = 4
CAVE_RARITY_MULTIPLIERS = {"少": 0.5, "中": 1.0, "多": 2.0}


__all__ = [
    "CAVE_RARITY_MULTIPLIERS",
    "DEFAULT_RIVERBED_MATERIALS",
    "MOUNTAIN_ROCK_PALETTE",
    "SPAN_MAX_SPANS",
    "SPAN_MIN_Y",
]
