"""BlueMap 离线 NPC 预览模型，复用用户已同意下载的原版村民纹理。

使用 BlueMap 5.23 的 entitystates + 方盒模型接口：
https://github.com/BlueMap-Minecraft/BlueMap/blob/v5.23/core/src/main/java/de/bluecolored/bluemap/core/map/hires/entity/ResourceModelRenderer.java
不重新分发 Mojang 纹理；模型仅用于标识 NPC 刷新点的脚部位置和成人体积。
"""

from __future__ import annotations

import json
from pathlib import Path


def _villager_cuboid(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    texture_origin: tuple[int, int],
    *,
    texture_size: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    """把 64×64 实体贴图的展开盒 UV 转成 BlueMap 的 0..16 UV。"""

    width, height, depth = texture_size or tuple(b - a for a, b in zip(lower, upper))
    u, v = texture_origin
    rectangles = {
        "west": (u, v + depth, u + depth, v + depth + height),
        "north": (u + depth, v + depth, u + depth + width, v + depth + height),
        "east": (u + depth + width, v + depth, u + 2 * depth + width, v + depth + height),
        "south": (u + 2 * depth + width, v + depth, u + 2 * (depth + width), v + depth + height),
        "up": (u + depth, v, u + depth + width, v + depth),
        "down": (u + depth + width, v + depth, u + depth + 2 * width, v),
    }
    # 原型高 34 像素，统一缩放到成人村民的 1.95 格；脚底始终保持 Y=0。
    scale = 1.95 * 16.0 / 34.0
    return {
        "from": [value * scale for value in lower],
        "to": [value * scale for value in upper],
        "faces": {
            face: {"uv": [value / 4.0 for value in uv], "texture": "#skin"}
            for face, uv in rectangles.items()
        },
    }


def write_villager_preview_model(pack_dir: Path) -> None:
    """安装站立、抱臂的村民模型；只影响 BlueMap 对村民实体的显示。"""

    model = {
        "textures": {"skin": "minecraft:entity/villager/villager"},
        "elements": [
            _villager_cuboid((-4, 24, -4), (4, 34, 4), (0, 0)),  # 头
            _villager_cuboid((-1, 24, -6), (1, 28, -4), (24, 0)),  # 鼻子
            _villager_cuboid((-4, 12, -3), (4, 24, 3), (16, 20)),  # 身体
            _villager_cuboid(
                (-4.5, 4, -3.5), (4.5, 24, 3.5), (0, 38), texture_size=(8, 18, 6)
            ),  # 长袍
            _villager_cuboid((-4, 0, -2), (0, 12, 2), (0, 22)),
            _villager_cuboid((0, 0, -2), (4, 12, 2), (0, 22)),
            _villager_cuboid((-6, 15, -4), (-2, 23, 0), (40, 38)),
            _villager_cuboid((2, 15, -4), (6, 23, 0), (40, 38)),
            _villager_cuboid((-4, 15, -5), (4, 19, -1), (40, 46)),  # 抱臂
        ],
    }
    entitystate = {"parts": [{"model": "bong:entity/preview_villager"}]}
    for relative, payload in (
        ("assets/minecraft/entitystates/villager.json", entitystate),
        ("assets/bong/models/entity/preview_villager.json", model),
    ):
        target = pack_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
