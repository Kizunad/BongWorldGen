"""把各个地表语义层合成为最终可见的地表材质。

``surface_id`` 仍表示地表基底，便于 Server 做地质和装饰判断；本模块输出的
``surface_visible_id`` 则表示当前列最上方实际应显示的方块。这样 BlueMap、Anvil
和 raster 查看器不必各自重新推断冰雪覆盖、冻土或河床的优先级。
"""

from __future__ import annotations

import numpy as np


BASE_SURFACE_PALETTE = (
    "minecraft:stone",
    "minecraft:coarse_dirt",
    "minecraft:gravel",
    "minecraft:grass_block",
    "minecraft:sand",
)


def _minecraft_name(material: str) -> str:
    name = material.strip().lower().removeprefix("minecraft:").replace("-", "_")
    if not name:
        raise ValueError("surface material names cannot be empty")
    return f"minecraft:{name}"


def build_visible_surface_layer(
    terrain: np.ndarray,
    water_level: np.ndarray,
    *,
    sea_level: float,
    riverbed_id: np.ndarray,
    riverbed_palette: tuple[str, ...],
    surface_material_id: np.ndarray,
    surface_material_palette: tuple[str, ...],
    permafrost_id: np.ndarray,
    permafrost_palette: tuple[str, ...],
    surface_cover_layers: np.ndarray,
    surface_cover_palette: tuple[str, ...],
    mountain_material_id: np.ndarray | None = None,
    mountain_material_palette: tuple[str, ...] = (),
    surface_columns: np.ndarray | None = None,
    climate_surface_id: np.ndarray | None = None,
    climate_surface_palette: tuple[str, ...] = (),
    glacial_crevasse_id: np.ndarray | None = None,
    glacial_crevasse_palette: tuple[str, ...] = (),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """按统一优先级计算最终可见地表材质。

    优先级从底到顶为：基础地表、河床、寒带基底材质、冻土材质、群山裸岩、
    覆盖层顶部。
    覆盖层只在真实地表列生效；洞穴入口的顶层实心段低于地表时会被排除，避免
    在入口上方生成看似悬空的雪冰。水位仍是独立字段，水面不会被编码进此层。
    """

    if terrain.ndim != 2:
        raise ValueError("visible surface fields must be two-dimensional")
    if surface_cover_layers.ndim != 3 or surface_cover_layers.shape[0] != 4:
        raise ValueError("surface_cover_layers must have shape (4, height, width)")
    fields = (
        terrain,
        water_level,
        riverbed_id,
        surface_material_id,
        permafrost_id,
        surface_cover_layers[0],
    )
    if glacial_crevasse_id is not None:
        fields = (*fields, glacial_crevasse_id)
    if mountain_material_id is not None:
        fields = (*fields, mountain_material_id)
    if any(field.shape != terrain.shape for field in fields):
        raise ValueError("visible surface fields must share a shape")
    if len(surface_cover_palette) != 4:
        raise ValueError("surface_cover_palette must contain four materials")
    if glacial_crevasse_id is not None:
        if not glacial_crevasse_palette:
            raise ValueError("glacial_crevasse_palette is required when ids are provided")
        if not np.issubdtype(glacial_crevasse_id.dtype, np.integer):
            raise ValueError("glacial_crevasse_id must contain integer palette ids")
        if np.any(glacial_crevasse_id < 0) or np.any(
            glacial_crevasse_id > len(glacial_crevasse_palette)
        ):
            raise ValueError("glacial_crevasse_id contains an unknown palette id")
    if surface_columns is None:
        surface_columns = np.ones(terrain.shape, dtype=bool)
    elif surface_columns.shape != terrain.shape:
        raise ValueError("surface_columns must have the same shape as terrain")

    wet = water_level >= 0.0
    high = terrain >= sea_level + 52.0
    names = np.full(terrain.shape, "minecraft:grass_block", dtype=object)
    names[high & ~wet] = "minecraft:stone"
    names[wet] = "minecraft:gravel"

    if climate_surface_id is not None:
        if climate_surface_id.shape != terrain.shape:
            raise ValueError("climate_surface_id must have the same shape as terrain")
        if "desert" in climate_surface_palette:
            desert_id = climate_surface_palette.index("desert") + 1
            names[(climate_surface_id == desert_id) & ~wet & ~high] = "minecraft:sand"

    for material_index, material in enumerate(riverbed_palette):
        names[riverbed_id == material_index] = _minecraft_name(material)

    cover_names = {_minecraft_name(material) for material in surface_cover_palette}
    for material_index, material in enumerate(surface_material_palette, start=1):
        name = _minecraft_name(material)
        if name not in cover_names:
            names[(surface_material_id == material_index) & ~wet] = name

    for material_index, material in enumerate(permafrost_palette, start=1):
        names[(permafrost_id == material_index) & ~wet] = _minecraft_name(material)

    # 群山材质是独立语义层：它覆盖普通冰川基底和冻土，但仍让最终覆盖层
    # （松雪/雪块/冰/蓝冰）在上方保持统一优先级。裸岩 ID 因此能直接进入
    # BlueMap 和 Anvil，而不会污染 surface_material_id 的旧 palette。
    if mountain_material_id is not None:
        if mountain_material_id.shape != terrain.shape:
            raise ValueError("mountain material ids must have the same shape as terrain")
        for material_index, material in enumerate(mountain_material_palette, start=1):
            name = _minecraft_name(material)
            if name not in cover_names:
                names[(mountain_material_id == material_index) & ~wet] = name

    # 数组顺序是从最上层松雪到最下层蓝冰；从底部蓝冰向上扫描，后写入的
    # 材质才会覆盖较低层，最终留下该列真正可见的最高材质。
    for material_index, material in reversed(tuple(enumerate(surface_cover_palette))):
        count = surface_cover_layers[material_index]
        names[(count > 0) & surface_columns & ~wet] = _minecraft_name(material)

    # 冰裂隙是覆盖层之上的最终可见修饰：中央空气会露出下方冰体，边缘可见
    # packed ice/blue ice。真实裂缝由 pipeline 清空对应 cover counts，避免
    # Anvil 堆叠逻辑把裂缝重新封住。
    if glacial_crevasse_id is not None:
        for material_index, material in enumerate(glacial_crevasse_palette, start=1):
            names[
                (glacial_crevasse_id == material_index)
                & surface_columns
                & ~wet
            ] = _minecraft_name(material)

    palette = tuple(
        dict.fromkeys(
            (
                *BASE_SURFACE_PALETTE,
                *(_minecraft_name(material) for material in riverbed_palette),
                *(_minecraft_name(material) for material in surface_material_palette),
                *(_minecraft_name(material) for material in mountain_material_palette),
                *(_minecraft_name(material) for material in permafrost_palette),
                *(_minecraft_name(material) for material in surface_cover_palette),
                *(_minecraft_name(material) for material in glacial_crevasse_palette),
            )
        )
    )
    ids = {name: index for index, name in enumerate(palette, start=1)}
    visible = np.empty(terrain.shape, dtype=np.uint8)
    for name, index in ids.items():
        visible[names == name] = index
    # 洞口列的顶层是空气，不把地下入口伪装成草地或石头。
    visible[~surface_columns] = 0
    return np.ascontiguousarray(visible), palette


__all__ = ["BASE_SURFACE_PALETTE", "build_visible_surface_layer"]
