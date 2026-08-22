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

from ..geometry import polyline_distance_and_progress
from ..models import (
    NoiseLayer,
    SPAN_MIN_Y,
    TerrainRecipe,
    UndergroundBlock,
)
from ..noise import sample_noise, sample_noise_3d
from .spans import build_solid_spans
from .topology import generate_cave_topology
from .worms import sample_worm_field


@dataclass(frozen=True)
class UndergroundResult:
    """地下生成阶段的完整输出，供主 pipeline 一次性装入 Heightfield。"""

    solid_spans: np.ndarray
    blocks: tuple[UndergroundBlock, ...]
    cave_id: np.ndarray
    cave_palette: tuple[str, ...]


def _smooth_max(left: np.ndarray, right: np.ndarray, radius: float) -> np.ndarray:
    """对两个密度场做平滑并集，避免洞道交汇处出现硬切接缝。"""

    if radius <= 0:
        return np.maximum(left, right)
    blend = np.clip(0.5 + 0.5 * (right - left) / radius, 0.0, 1.0)
    return np.maximum(left, right) + radius * blend * (1.0 - blend)


def _domain_warp_cave_coordinates(
    x: np.ndarray,
    z: np.ndarray,
    network_name: str,
    scale: float,
    strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对洞穴的 XZ 域做确定性扭曲，避免通道保持笔直的人工形状。

    这是 FastNoiseLite domain-warp 工作流的 NumPy 版本：先用两个独立的
    噪声场得到 X/Z 位移，再在扭曲后的坐标上评估洞穴 SDF。网络名称只用于
    形成稳定的 seed 偏移，不能使用 Python 的 ``hash``（它跨进程不稳定）。

    参考：
    https://github.com/Auburn/FastNoiseLite
    https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
    """

    if strength <= 0.0:
        return x, z
    name_seed = sum((index + 1) * ord(char) for index, char in enumerate(network_name))
    warp_layer = NoiseLayer(
        kind="fbm",
        scale=scale,
        octaves=2,
        gain=0.5,
        seed_offset=name_seed & 0x7FFF,
    )
    offset_x = sample_noise(x, z, warp_layer, seed + 101_021)
    offset_z = sample_noise(x, z, warp_layer, seed + 101_053)
    return x + offset_x * strength, z + offset_z * strength


def generate_underground(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
    seed: int,
) -> UndergroundResult:
    """生成 worm 洞穴空腔、地下结构、洞穴 ID 和垂直实心段。"""

    feature_bounds: list[tuple[int, int]] = []
    for network in recipe.caves:
        radius = max(
            network.height * 0.5,
            network.chamber_height,
            network.vertical_warp,
            network.entrance_radius,
            1.0,
        )
        feature_bounds.append(
            (
                math.floor(-network.depth - radius - 1.0),
                math.ceil(-network.depth + radius + 1.0),
            )
        )
        if network.entrance_count:
            feature_bounds.append(
                (
                    math.floor(-network.depth - radius - 1.0),
                    1,
                )
            )
    height, width = terrain.shape
    if not feature_bounds:
        empty_void = np.zeros((0, height, width), dtype=bool)
        return UndergroundResult(
            build_solid_spans(terrain, empty_void, np.empty(0, dtype=np.int16)),
            (),
            np.zeros((height, width), dtype=np.uint8),
            (),
        )

    offset_min = min(bounds[0] for bounds in feature_bounds)
    offset_max = max(bounds[1] for bounds in feature_bounds)
    cave_offsets = np.arange(offset_min, offset_max + 1, dtype=np.int16)
    cave_void = np.zeros((cave_offsets.size, height, width), dtype=bool)
    cave_id = np.zeros((height, width), dtype=np.uint8)
    cave_palette = tuple(network.name for network in recipe.caves)
    surface = np.floor(terrain).astype(np.int16)

    for network_index, network in enumerate(recipe.caves):
        network_void = np.zeros_like(cave_void)
        network_seed = seed + network_index * 9_973
        topology = generate_cave_topology(network, network_seed)
        warped_x, warped_z = _domain_warp_cave_coordinates(
            x,
            z,
            network.name,
            network.domain_warp_scale,
            network.domain_warp_strength,
            network_seed,
        )
        worm = sample_worm_field(warped_x, warped_z, network, network_seed)
        network_density = np.full(cave_void.shape, -np.inf, dtype=np.float64)
        entrance_density = np.full(cave_void.shape, -np.inf, dtype=np.float64)
        for path in topology.paths:
            distance, _ = polyline_distance_and_progress(warped_x, warped_z, path)
            vertical_radius = max(network.height * 0.5, 1.0)
            for level_index, offset in enumerate(cave_offsets):
                world_y = surface + int(offset)
                vertical_coordinate = (
                    float(offset) + network.depth - worm.vertical_offset
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
                    network_seed + level_index * 1013 + 40_001,
                )
                # 低频阈值调制负责制造死胡同；3D 噪声只扰动洞壁细节，
                # 不负责凭空创造一条远离拓扑路径的洞道。
                path_density = (
                    godot_worm_density * threshold_modulation * (0.65 + 0.35 * worm.dead_end)
                    + network.noise_strength * noise
                    - network.dead_end_strength * (1.0 - worm.dead_end)
                )
                network_density[level_index] = _smooth_max(
                    network_density[level_index],
                    path_density,
                    network.smooth_union,
                )

        for level_index, offset in enumerate(cave_offsets):
            world_y = surface + int(offset)
            # 将这一层的路径并集与洞室节点合并。洞室使用独立的椭球
            # SDF，能够打破“所有地方都是细管”的视觉单调性。
            for chamber in topology.chambers:
                chamber_sdf = np.sqrt(
                    ((x - chamber.x) / network.chamber_radius) ** 2
                    + ((z - chamber.z) / network.chamber_radius) ** 2
                    + ((float(offset) + network.depth) / network.chamber_height) ** 2
                ) - 1.0
                network_density[level_index] = _smooth_max(
                    network_density[level_index],
                    -chamber_sdf,
                    network.smooth_union,
                )
            network_void[level_index] = (
                (network_density[level_index] >= network.sdf_threshold)
                & (world_y <= surface - network.roof_thickness)
                & (world_y > SPAN_MIN_Y)
            )

            # 洞口是独立的第二阶段：用一个从地表延伸到主洞的椭球通道，
            # 只放宽洞口自身的屋顶限制，不让普通洞道穿出地表。
            for entrance in topology.entrances:
                entrance_sdf = np.sqrt(
                    ((warped_x - entrance.x) / network.entrance_radius) ** 2
                    + ((warped_z - entrance.z) / network.entrance_radius) ** 2
                    + (
                        (float(offset) + network.depth * 0.5)
                        / max(network.depth * 0.5 + 1.0, 1.0)
                    )
                    ** 2
                ) - 1.0
                entrance_density[level_index] = _smooth_max(
                    entrance_density[level_index],
                    -entrance_sdf,
                    network.smooth_union,
                )
            network_void[level_index] |= (
                (entrance_density[level_index] >= network.sdf_threshold)
                & (world_y <= surface)
                & (world_y > SPAN_MIN_Y)
            )
        cave_void |= network_void
        cave_id[np.any(network_void, axis=0)] = np.uint8(network_index + 1)

    return UndergroundResult(
        build_solid_spans(terrain, cave_void, cave_offsets),
        (),
        cave_id,
        cave_palette,
    )
