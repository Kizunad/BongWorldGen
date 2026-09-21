"""洞穴网络和地下结构的生成编排。

洞穴几何采用 seeded 3D noise 与路径 SDF 的组合。接口形态参考
FastNoiseLite 的公开设计（3D noise、seed、domain warp）：
https://github.com/Auburn/FastNoiseLite

这里只保存与洞穴相关的中间数据，最终由 ``Heightfield`` 和适配器负责
序列化，服务端可通过 ``cave_id`` 判断区域，再通过 ``spans`` 找到实际空腔。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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


def _axis_slice(axis: np.ndarray, lower: float, upper: float) -> slice | None:
    """返回覆盖世界坐标区间的最小网格切片，并保留一格边界余量。"""

    if axis[-1] < lower or axis[0] > upper:
        return None
    start = max(0, int(np.searchsorted(axis, lower, side="left")) - 1)
    stop = min(axis.size, int(np.searchsorted(axis, upper, side="right")) + 1)
    return slice(start, stop) if start < stop else None


def _network_bounds(network, topology) -> tuple[float, float, float, float]:
    """计算网络在 XZ 平面的保守影响范围，用于裁剪 3D 噪声计算。"""

    points = [point for path in topology.paths for point in path]
    points.extend(topology.chambers)
    points.extend(topology.entrances)
    margin = (
        max(network.width, network.chamber_radius, network.entrance_radius)
        + network.domain_warp_strength
        + network.smooth_union
        + 2.0
    )
    return (
        min(point.x for point in points) - margin,
        max(point.x for point in points) + margin,
        min(point.z for point in points) - margin,
        max(point.z for point in points) + margin,
    )


def _empty_underground(terrain: np.ndarray) -> UndergroundResult:
    """创建没有洞穴时的结果，交给 spans 模块走向量化快速路径。"""

    height, width = terrain.shape
    empty_void = np.empty((0, height, width), dtype=bool)
    return UndergroundResult(
        build_solid_spans(terrain, empty_void, np.empty(0, dtype=np.int16)),
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

    if not recipe.caves:
        return _empty_underground(terrain)

    active_networks = []
    for network_index, network in enumerate(recipe.caves):
        network_seed = seed + network_index * 9_973
        topology = generate_cave_topology(network, network_seed)
        min_x, max_x, min_z, max_z = _network_bounds(network, topology)
        x_slice = _axis_slice(x[0, :], min_x, max_x)
        z_slice = _axis_slice(z[:, 0], min_z, max_z)
        if x_slice is not None and z_slice is not None:
            active_networks.append((network_index, network, topology, x_slice, z_slice))
    if not active_networks:
        # Keep global palette IDs while skipping the entire 3D allocation for
        # tiles outside every cave. Terrain and palette remain crop-independent.
        return replace(_empty_underground(terrain), cave_palette=tuple(n.name for n in recipe.caves))

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
        if network.entrance_count or network.entrance_points:
            feature_bounds.append(
                (
                    math.floor(-network.depth - radius - 1.0),
                    1,
                )
            )
    height, width = terrain.shape

    offset_min = min(bounds[0] for bounds in feature_bounds)
    offset_max = max(bounds[1] for bounds in feature_bounds)
    cave_offsets = np.arange(offset_min, offset_max + 1, dtype=np.int16)
    cave_void = np.zeros((cave_offsets.size, height, width), dtype=bool)
    cave_id = np.zeros((height, width), dtype=np.uint8)
    cave_palette = tuple(network.name for network in recipe.caves)
    for network_index, network, topology, x_slice, z_slice in active_networks:
        network_seed = seed + network_index * 9_973
        crop_x = x[z_slice, x_slice]
        crop_z = z[z_slice, x_slice]
        crop_surface = np.floor(terrain[z_slice, x_slice]).astype(np.int16)
        network_void = np.zeros(
            (cave_offsets.size, crop_surface.shape[0], crop_surface.shape[1]),
            dtype=bool,
        )
        warped_x, warped_z = _domain_warp_cave_coordinates(
            crop_x,
            crop_z,
            network.name,
            network.domain_warp_scale,
            network.domain_warp_strength,
            network_seed,
        )
        worm = sample_worm_field(warped_x, warped_z, network, network_seed)
        network_density = np.full(network_void.shape, -np.inf, dtype=np.float64)
        entrance_density = np.full(network_void.shape, -np.inf, dtype=np.float64)
        path_vertical_limit = (
            network.vertical_warp
            + max(network.height * 0.5, 1.0)
            * math.sqrt(1.0 + (network.noise_strength + len(topology.paths) * network.smooth_union)
                        / (0.55 * 0.65))
            + 1.0
        )
        for path in topology.paths:
            distance, _ = polyline_distance_and_progress(warped_x, warped_z, path)
            vertical_radius = max(network.height * 0.5, 1.0)
            for level_index, offset in enumerate(cave_offsets):
                if abs(float(offset) + network.depth) > path_vertical_limit:
                    continue
                world_y = crop_surface + int(offset)
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
                    # A single 3D field must keep the same seed across Y.
                    # Reseeding each slice creates detached one-block voids
                    # near walls and can overflow the four-span contract.
                    network_seed + 40_001,
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
            world_y = crop_surface + int(offset)
            # 将这一层的路径并集与洞室节点合并。洞室使用独立的椭球
            # SDF，能够打破“所有地方都是细管”的视觉单调性。
            for chamber in topology.chambers:
                chamber_sdf = np.sqrt(
                    ((crop_x - chamber.x) / network.chamber_radius) ** 2
                    + ((crop_z - chamber.z) / network.chamber_radius) ** 2
                    + ((float(offset) + network.depth) / network.chamber_height) ** 2
                ) - 1.0
                network_density[level_index] = _smooth_max(
                    network_density[level_index],
                    -chamber_sdf,
                    network.smooth_union,
                )
            network_void[level_index] = (
                (network_density[level_index] >= network.sdf_threshold)
                & (world_y <= crop_surface - network.roof_thickness)
                & (world_y > SPAN_MIN_Y)
            )

            # 洞口是独立的第二阶段：用从地表延伸到主洞的胶囊形竖井，
            # 只放宽洞口自身的屋顶限制，不让普通洞道穿出地表。
            for entrance in topology.entrances:
                entrance_sdf = np.sqrt(
                    ((warped_x - entrance.x) / network.entrance_radius) ** 2
                    + ((warped_z - entrance.z) / network.entrance_radius) ** 2
                    + (max(-network.depth - float(offset), float(offset), 0.0)
                       / network.entrance_radius) ** 2
                ) - 1.0
                entrance_density[level_index] = _smooth_max(
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
        network_columns = np.any(network_void, axis=0)
        cave_id[z_slice, x_slice][network_columns] = np.uint8(network_index + 1)

    return UndergroundResult(
        build_solid_spans(terrain, cave_void, cave_offsets),
        (),
        cave_id,
        cave_palette,
    )
