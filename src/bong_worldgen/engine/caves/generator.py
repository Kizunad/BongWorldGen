"""洞穴网络和地下结构的生成编排。

洞穴几何采用 seeded 3D noise 与路径 SDF 的组合。接口形态参考
FastNoiseLite 的公开设计（3D noise、seed、domain warp）：
https://github.com/Auburn/FastNoiseLite

这里只保存与洞穴相关的中间数据，最终由 ``Heightfield`` 和适配器负责
序列化，服务端可通过 ``cave_id`` 判断区域，再通过 ``spans`` 找到实际空腔。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..distribution import iter_world_network_instances
from ..geometry import polyline_distance_and_progress
from ..constants import SPAN_MIN_Y
from ..noise import sample_noise, sample_noise_3d
from ..ores import generate_solid_ore_blocks
from ..underground_config import UndergroundBlock, UndergroundWaterBlock
from ..underground_rivers import generate_underground_rivers
from ..world_config import TerrainRecipe
from .density import (
    axis_slice,
    domain_warp_coordinates,
    network_bounds,
    smooth_density_union,
    stable_vertical_tail_thresholds,
    vertical_tail_lift,
)
from .spans import build_solid_spans
from .structures import generate_placeholder_blocks
from .topology import generate_cave_topology
from .worms import sample_worm_field


@dataclass(frozen=True)
class UndergroundResult:
    """地下生成阶段的完整输出，供主 pipeline 一次性装入 Heightfield。"""

    solid_spans: np.ndarray
    blocks: tuple[UndergroundBlock, ...]
    water_blocks: tuple[UndergroundWaterBlock, ...]
    cave_id: np.ndarray
    cave_palette: tuple[str, ...]
    fracture_id: np.ndarray
    fracture_palette: tuple[str, ...]


def _empty_underground(terrain: np.ndarray) -> UndergroundResult:
    """创建没有洞穴时的结果，交给 spans 模块走向量化快速路径。"""

    height, width = terrain.shape
    empty_void = np.empty((0, height, width), dtype=bool)
    return UndergroundResult(
        build_solid_spans(terrain, empty_void, np.empty(0, dtype=np.int16)),
        (),
        (),
        np.zeros((height, width), dtype=np.uint8),
        (),
        np.zeros((height, width), dtype=np.uint8),
        (),
    )


def generate_underground(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
    seed: int,
) -> UndergroundResult:
    """生成 worm 洞穴空腔、地下结构、洞穴 ID 和垂直实心段。"""

    if not recipe.caves and not recipe.underground_rivers and not recipe.solid_ores:
        return _empty_underground(terrain)

    x_axis = x[0, :]
    z_axis = z[:, 0]
    cave_instances = [
        (network_index, instance, instance_seed)
        for network_index, network in enumerate(recipe.caves)
        for instance, instance_seed in iter_world_network_instances(
            network, x_axis, z_axis, seed, network_index
        )
    ]
    river_instances = [
        (river_index, instance, instance_seed)
        for river_index, network in enumerate(recipe.underground_rivers)
        for instance, instance_seed in iter_world_network_instances(
            network, x_axis, z_axis, seed + 31_337, river_index
        )
    ]

    feature_bounds: list[tuple[int, int]] = []
    for network in recipe.caves:
        radius = max(
            network.height * 0.5,
            network.chamber_height,
            network.vertical_warp,
            network.entrance_radius,
            1.0,
        )
        vertical_radius = max(network.height * 0.5, 1.0)
        surface_lift = network.depth - vertical_radius * 0.5
        vertical_min = (
            -network.depth
            - network.depth * network.vertical_drop_ratio
            - radius
            - 1.0
        )
        vertical_max = (
            -network.depth
            + max(network.depth * network.vertical_rise_ratio, surface_lift)
            + radius
            + 1.0
        )
        feature_bounds.append(
            (
                math.floor(vertical_min),
                math.ceil(vertical_max),
            )
        )
        if network.entrance_count:
            feature_bounds.append(
                (
                    math.floor(-network.depth - radius - 1.0),
                    1,
                )
            )
    for network in recipe.underground_rivers:
        vertical_radius = max(network.height * 0.75, 1.0)
        deepest = max(network.source_depth, network.outlet_depth)
        shallowest = min(network.source_depth, network.outlet_depth)
        feature_bounds.append(
            (
                math.floor(-deepest - vertical_radius - 1.0),
                math.ceil(-shallowest + vertical_radius + 1.0),
            )
        )
        fracture_radius = max(network.fracture_height * 0.5, 1.0)
        feature_bounds.append(
            (
                math.floor(-network.fracture_depth - fracture_radius - 1.0),
                math.ceil(-network.fracture_depth + fracture_radius + 1.0),
            )
        )
    for ore in recipe.solid_ores:
        # 普通矿物位于地表下 min_depth..max_depth 格，先把这个垂直范围
        # 纳入统一 offsets，才能与洞穴空腔使用同一套排除判断。
        feature_bounds.append(
            (
                math.floor(-ore.max_depth - 1.0),
                math.ceil(-ore.min_depth + 1.0),
            )
        )
    height, width = terrain.shape

    offset_min = min(bounds[0] for bounds in feature_bounds)
    offset_max = max(bounds[1] for bounds in feature_bounds)
    cave_offsets = np.arange(offset_min, offset_max + 1, dtype=np.int16)
    cave_void = np.zeros((cave_offsets.size, height, width), dtype=bool)
    cave_id_only = np.zeros((height, width), dtype=np.uint8)
    cave_id = np.zeros((height, width), dtype=np.uint8)
    fracture_id = np.zeros((height, width), dtype=np.uint8)
    cave_palette = tuple(network.name for network in recipe.caves)
    fracture_palette = tuple(network.name for network in recipe.underground_rivers)
    underground_blocks: list[UndergroundBlock] = []
    underground_water_blocks: list[UndergroundWaterBlock] = []

    for network_index, network, network_seed in cave_instances:
        topology = generate_cave_topology(network, network_seed)
        min_x, max_x, min_z, max_z = network_bounds(network, topology)
        x_slice = axis_slice(x_axis, min_x, max_x)
        z_slice = axis_slice(z_axis, min_z, max_z)
        if x_slice is None or z_slice is None:
            # 当前 tile 与网络的确定性世界范围没有交集，不做任何 3D
            # noise/SDF 计算；最终 spans 模块会走全空腔快速路径。
            continue

        crop_x = x[z_slice, x_slice]
        crop_z = z[z_slice, x_slice]
        crop_surface = np.floor(terrain[z_slice, x_slice]).astype(np.int16)
        network_void = np.zeros(
            (cave_offsets.size, crop_surface.shape[0], crop_surface.shape[1]),
            dtype=bool,
        )
        warped_x, warped_z = domain_warp_coordinates(
            crop_x,
            crop_z,
            network.name,
            network.domain_warp_scale,
            network.domain_warp_strength,
            network_seed,
        )
        worm = sample_worm_field(warped_x, warped_z, network, network_seed)
        # 低频分带让大多数洞道留在主深度，只有约 5% 的区域向上抬升、约
        # 5% 向下沉降。这个分带参考 Godot Voxel 的低频调制思路，不改变
        # XZ 拓扑，只改变局部洞道的垂直中心：
        # https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
        tail_noise = sample_noise(
            warped_x,
            warped_z,
            network.vertical_warp_noise,
            network_seed + 92_117,
        )
        lower_start, upper_start, lower_end, upper_end = stable_vertical_tail_thresholds(
            network,
            topology,
            network_seed,
        )
        vertical_lift = vertical_tail_lift(
            tail_noise,
            depth=network.depth,
            tail_fraction=network.vertical_tail_fraction,
            rise_ratio=network.vertical_rise_ratio,
            drop_ratio=network.vertical_drop_ratio,
            upper_start=upper_start,
            lower_start=lower_start,
            upper_end=upper_end,
            lower_end=lower_end,
        )
        # 90% 的主洞仍保留屋顶；上抬尾部的极端 20% 允许真正接触地表，
        # 形成可见的天然天窗/入口，而不是只在地下抬高几格。
        surface_break = vertical_lift >= (
            network.depth * network.vertical_rise_ratio * 0.8
        )
        # 上抬尾部的极端区段把洞道中心抬到地表下半径的一半处，保证
        # 截面真正穿出地表；普通中段仍由 roof_thickness 保持封闭。
        vertical_lift = np.where(
            surface_break,
            np.maximum(vertical_lift, surface_lift),
            vertical_lift,
        )
        network_density = np.full(network_void.shape, -np.inf, dtype=np.float64)
        entrance_density = np.full(network_void.shape, -np.inf, dtype=np.float64)
        for path in topology.paths:
            distance, _ = polyline_distance_and_progress(warped_x, warped_z, path)
            for level_index, offset in enumerate(cave_offsets):
                world_y = crop_surface + int(offset)
                vertical_coordinate = (
                    float(offset) + network.depth - vertical_lift - worm.vertical_offset
                ) / vertical_radius
                # Godot Voxel 的核心技巧：2D 噪声平方后按阈值截取 worm，
                # 再用 y²-1 的抛物线调制阈值，使同一条 XZ 通道拥有圆润的
                # 3D 截面，而不是一张“贴在平面上的 2D 洞穴贴图”。
                parabolic_profile = 1.0 - np.square(vertical_coordinate)
                radial_profile = 1.0 - distance / np.maximum(
                    network.width * worm.radius_scale,
                    1.0e-6,
                )
                godot_worm_density = np.minimum(radial_profile, parabolic_profile)
                threshold_modulation = 0.55 + 0.45 * worm.corridor
                noise = sample_noise_3d(
                    warped_x,
                    world_y * (network.roughness.scale / network.vertical_scale),
                    warped_z,
                    network.roughness,
                    # 不能使用全局 cave_offsets 的 level_index：加入独立河网
                    # 会扩大 Y 范围并改变普通洞穴的噪声 seed。使用世界 offset
                    # 作为稳定坐标，保证干洞与地下河开关相互独立。
                    network_seed + (int(offset) + 10_000) * 1_013 + 40_001,
                )
                # 低频阈值调制负责制造死胡同；3D 噪声只扰动洞壁细节，
                # 不负责凭空创造一条远离拓扑路径的洞道。
                path_density = (
                    godot_worm_density * threshold_modulation * (0.65 + 0.35 * worm.dead_end)
                    + network.noise_strength * noise
                    - network.dead_end_strength * (1.0 - worm.dead_end)
                )
                network_density[level_index] = smooth_density_union(
                    network_density[level_index],
                    path_density,
                    network.smooth_union,
                )

        for level_index, offset in enumerate(cave_offsets):
            world_y = crop_surface + int(offset)
            # 将这一层的路径并集与洞室节点合并。洞室使用独立的椭球
            # SDF，能够打破“所有地方都是细管”的视觉单调性。
            for chamber in topology.chambers:
                chamber_sdf = np.sqrt(
                    ((crop_x - chamber.x) / network.chamber_radius) ** 2
                    + ((crop_z - chamber.z) / network.chamber_radius) ** 2
                    + ((float(offset) + network.depth) / network.chamber_height) ** 2
                ) - 1.0
                network_density[level_index] = smooth_density_union(
                    network_density[level_index],
                    -chamber_sdf,
                    network.smooth_union,
                )
            roof_limit = np.where(
                surface_break,
                crop_surface,
                crop_surface - network.roof_thickness,
            )
            network_void[level_index] = (
                (network_density[level_index] >= network.sdf_threshold)
                & (world_y <= roof_limit)
                & (world_y > SPAN_MIN_Y)
            )

            # 洞口是独立的第二阶段：用一个从地表延伸到主洞的椭球通道，
            # 只放宽洞口自身的屋顶限制，不让普通洞道穿出地表。
            entrance_radius = max(
                network.entrance_radius,
                network.domain_warp_strength + network.width * 0.5,
            )
            for entrance in topology.entrances:
                entrance_sdf = np.sqrt(
                    ((crop_x - entrance.x) / entrance_radius) ** 2
                    + ((crop_z - entrance.z) / entrance_radius) ** 2
                    + (
                        (float(offset) + network.depth * 0.5)
                        / max(network.depth * 0.5 + 1.0, 1.0)
                    )
                    ** 2
                ) - 1.0
                entrance_density[level_index] = smooth_density_union(
                    entrance_density[level_index],
                    -entrance_sdf,
                    network.smooth_union,
                )
            network_void[level_index] |= (
                (entrance_density[level_index] >= network.sdf_threshold)
                & (world_y <= crop_surface)
                & (world_y > SPAN_MIN_Y)
            )
        cave_void[:, z_slice, x_slice] |= network_void
        target_cave_id = cave_id_only[z_slice, x_slice]
        target_cave_id[np.any(network_void, axis=0)] = np.uint8(network_index + 1)
        underground_blocks.extend(
            generate_placeholder_blocks(
                network,
                topology,
                crop_x,
                crop_z,
                crop_surface,
                cave_offsets,
                network_void,
                network_seed + 81_337,
            )
        )
    if recipe.caves:
        cave_id = cave_id_only
    # 地下河单独生成自己的三维河道，不读取也不覆盖普通洞穴的空腔。
    # 最终只在 spans 阶段与普通洞穴做几何并集，因此两者的生成概率仍然独立。
    for river_index, river, river_seed in river_instances:
        river_result = generate_underground_rivers(
            terrain,
            x,
            z,
            cave_offsets,
            river,
            river_seed,
        )
        cave_void |= river_result.void
        if river_result.fracture_void is not None:
            # 普通洞穴、裂隙、地下河最后才合并实心段；河水和资源仍只
            # 使用 river_result.void，因此裂隙不会被误判成河道。
            cave_void |= river_result.fracture_void
            target_fracture_id = fracture_id
            target_fracture_id[np.any(river_result.fracture_void, axis=0)] = np.uint8(
                river_index + 1
            )
        underground_blocks.extend(river_result.resource_blocks)
        underground_water_blocks.extend(river_result.water_blocks)

    # 普通矿脉最后生成：此时普通洞穴、地下河和裂隙的空腔都已经合并，
    # 候选点可以严格排除空腔和已有的地下内容，保证它确实替换实心石。
    underground_blocks.extend(
        generate_solid_ore_blocks(
            terrain,
            x,
            z,
            cave_offsets,
            cave_void,
            recipe.solid_ores,
            seed + 144_709,
            occupied_blocks=underground_blocks,
            water_blocks=underground_water_blocks,
        )
    )

    return UndergroundResult(
        build_solid_spans(terrain, cave_void, cave_offsets),
        tuple(underground_blocks),
        tuple(underground_water_blocks),
        cave_id,
        cave_palette,
        fracture_id,
        fracture_palette,
    )
