"""独立地下河网生成器。

这里实现的是 pyKasso 的轻量、可复现子集：在固定三维世界的 XZ 投影域中
生成 inlet/outlet，构造独立的水文 travel cost，用多轮 geodesic 路径合并成
conduit graph；zero-isoline 裂隙作为另一套无水空腔生成，最后只在 spans 几何层合并。

参考资料（只移植算法思想，不复制代码）：

* pyKasso 的 domain、fracture、入口/出口和 conduit network：
  https://github.com/randlab/pyKasso
* pyKasso 的 geodesic/fast-marching 网络思想：
  https://github.com/randlab/pyKasso/blob/master/pykasso/model/sks.py
* Tokunaga 树的分支级别与汇流结构：
  https://en.wikipedia.org/wiki/Tokunaga%E2%80%93Tokunaga_model
* Priority-Flood/D8 水文网络参考：
  https://github.com/semenovi/landscape-gen

pyKasso 为 GPL-3.0 项目，本模块保持 BongWorldGen 当前的 NumPy-only 依赖，
没有直接引入 AGD/HFM，也没有把 pyKasso 代码复制进来。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from typing import Literal

import numpy as np

from ..constants import SPAN_MIN_Y
from ..terrain_config import NoiseLayer, Point
from ..underground_config import (
    UndergroundBlock,
    UndergroundRiverNetwork,
    UndergroundWaterBlock,
)
from ..noise import sample_noise
from ..randomness import unit_interval as _unit
from .fractures import zero_isoline_band
from .resources import generate_river_resource_blocks


_NEIGHBOURS = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class _RiverPath:
    """一条 inlet 到 outlet 的世界坐标 conduit 边。"""

    points: tuple[Point, ...]
    inlet_index: int


@dataclass(frozen=True)
class _NetworkGeometry:
    """网络图的可审计中间结果；不会改变既有 Heightfield 输出契约。"""

    paths: tuple[_RiverPath, ...]
    inlets: tuple[Point, ...]
    outlets: tuple[Point, ...]
    junctions: tuple[Point, ...]
    unreachable_inlets: tuple[Point, ...]


@dataclass(frozen=True)
class UndergroundRiverResult:
    """独立地下河网的空腔、水块和网络诊断信息。"""

    void: np.ndarray
    water_blocks: tuple[UndergroundWaterBlock, ...]
    # 裂隙与河道是两个独立的空腔场，只在上层 spans 合并几何。
    fracture_void: np.ndarray | None = None
    resource_blocks: tuple[UndergroundBlock, ...] = ()
    paths: tuple[tuple[Point, ...], ...] = ()
    inlets: tuple[Point, ...] = ()
    outlets: tuple[Point, ...] = ()
    junctions: tuple[Point, ...] = ()
    unreachable_inlets: tuple[Point, ...] = ()


def _grid_cell_size(x: np.ndarray, z: np.ndarray) -> float:
    """读取 tile 网格步长；单列/单行 tile 使用 1 格默认值。"""

    cell_x = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0
    cell_z = float(z[1, 0] - z[0, 0]) if z.shape[0] > 1 else 1.0
    if cell_x <= 0 or cell_z <= 0 or not math.isclose(cell_x, cell_z, rel_tol=1e-6):
        raise ValueError("underground river grid must be a regular square grid")
    return cell_x


def _network_grid(
    network: UndergroundRiverNetwork,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """建立与 tile 无关的固定世界域，避免裁剪边界改变河网。"""

    minimum_x = network.seed_center.x - network.seed_extent
    minimum_z = network.seed_center.z - network.seed_extent
    count = int(math.ceil(network.seed_extent * 2.0 / cell_size)) + 1
    x_axis = minimum_x + np.arange(count, dtype=np.float64) * cell_size
    z_axis = minimum_z + np.arange(count, dtype=np.float64) * cell_size
    grid_x, grid_z = np.meshgrid(x_axis, z_axis, indexing="xy")
    return grid_x, grid_z, x_axis, z_axis


def _endpoint_points(
    network: UndergroundRiverNetwork,
    seed: int,
) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """在固定域内确定性放置入口弧和下游出口弧。"""

    inlet_count = network.inlet_count or max(
        1, network.path_count + network.branch_count
    )
    direction = _unit(seed, 10_001) * math.tau
    inlets: list[Point] = []
    for index in range(inlet_count):
        centered = (index - (inlet_count - 1) * 0.5) / max(inlet_count, 1)
        jitter = (_unit(seed, 10_100 + index * 7) - 0.5) * 0.32
        angle = direction + math.pi + (centered * 1.15 + jitter)
        radius = network.seed_extent * (
            1.0 - network.inlet_margin - 0.12 * _unit(seed, 10_101 + index * 7)
        )
        inlets.append(
            Point(
                network.seed_center.x + math.cos(angle) * radius,
                network.seed_center.z + math.sin(angle) * radius,
            )
        )

    outlets: list[Point] = []
    for index in range(network.outlet_count):
        centered = (index - (network.outlet_count - 1) * 0.5) / max(
            network.outlet_count, 1
        )
        angle = direction + centered * 0.7
        radius = network.seed_extent * (
            1.0 - network.outlet_margin - 0.08 * _unit(seed, 11_001 + index * 5)
        )
        outlets.append(
            Point(
                network.seed_center.x + math.cos(angle) * radius,
                network.seed_center.z + math.sin(angle) * radius,
            )
        )
    return tuple(inlets), tuple(outlets)


def _nearest_cell(point: Point, x_axis: np.ndarray, z_axis: np.ndarray) -> tuple[int, int]:
    cell_x = max(float(x_axis[1] - x_axis[0]), 1.0e-9)
    cell_z = max(float(z_axis[1] - z_axis[0]), 1.0e-9)
    x_index = int(
        np.clip(np.rint((point.x - x_axis[0]) / cell_x), 0, x_axis.size - 1)
    )
    z_index = int(
        np.clip(np.rint((point.z - z_axis[0]) / cell_z), 0, z_axis.size - 1)
    )
    return z_index, x_index


def _travel_cost(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    network: UndergroundRiverNetwork,
    seed: int,
) -> np.ndarray:
    """生成地下河自己的水文通行代价场。

    pyKasso 会把地质域、裂隙方向和 conduit 代价送入 fast marching。这里的
    河流路径只使用独立的低频水文起伏；zero-isoline 裂隙由 ``_fracture_void``
    单独生成，不会被当成河道中心线。
    参考：
    https://github.com/randlab/pyKasso/blob/master/pykasso/model/geologic_features/fractures.py
    """

    noise = sample_noise(
        grid_x,
        grid_z,
        NoiseLayer(
            kind="fbm",
            scale=network.cost_noise_scale,
            octaves=3,
            gain=0.55,
            seed_offset=17,
        ),
        seed + 12_017,
    )
    # 河流路径不再直接跟随 zero-isoline；裂隙会在独立的 fracture_void
    # 中生成。这里保留第二个独立的地质起伏层，只表达河道的水文通行差异。
    hydrology = sample_noise(
        grid_x,
        grid_z,
        NoiseLayer(
            kind="fbm",
            scale=network.cost_noise_scale * 0.58,
            octaves=2,
            gain=0.5,
            seed_offset=31,
        ),
        seed + 12_071,
    )
    cost = (
        1.0
        + network.cost_noise_strength * (noise + 1.0) * 0.5
        + 0.08 * (hydrology + 1.0) * 0.5
    )
    return np.maximum(cost, 0.08).astype(np.float64, copy=False)


def _fracture_void(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    offsets: np.ndarray,
    network: UndergroundRiverNetwork,
    seed: int,
) -> np.ndarray:
    """生成独立的构造裂隙，不读取地下河路径或水体。

    裂隙与普通洞穴的区别是几何约束，而不是换一个 seed：zero-isoline
    只提供平面上的长裂缝中心线；梯度估计把它换算为方块单位的窄带，
    再沿 Y 方向拉成长而窄的竖向切面。上下端逐渐收尖，中心线做小幅
    确定性错动，因此不会退化成水平 worm 洞道。

    zero-isoline 的思路参考：
    https://github.com/randlab/pyKasso/blob/master/pykasso/model/geologic_features/fractures.py
    噪声与 3D 体积场的组织参考：
    https://github.com/Auburn/FastNoiseLite
    """

    scale = (
        network.fracture_isoline_scale
        if network.fracture_isoline_scale > 0.0
        else network.cost_noise_scale * 0.72
    )
    field = sample_noise(
        x,
        z,
        NoiseLayer(kind="fbm", scale=scale, octaves=2, gain=0.5, seed_offset=31),
        seed + 71_071,
    )
    band = zero_isoline_band(field, network.fracture_isoline_width)
    # bias 只控制零等值线的连续保留率；实际宽度由 fracture_width（方块）
    # 决定，避免噪声值域的参数把裂隙意外放大成矿洞。
    threshold = 0.56 + 0.24 * (1.0 - network.fracture_bias)
    domain_mask = (
        (np.abs(x - network.seed_center.x) <= network.seed_extent)
        & (np.abs(z - network.seed_center.z) <= network.seed_extent)
    )
    cell_size = _grid_cell_size(x, z)
    gradient_z, gradient_x = np.gradient(field, cell_size, cell_size)
    gradient_length = np.hypot(gradient_x, gradient_z)
    distance_to_isoline = np.abs(field) / np.maximum(gradient_length, 1.0e-6)
    width_noise = sample_noise(
        x,
        z,
        NoiseLayer(
            kind="fbm",
            scale=max(scale * 1.7, 32.0),
            octaves=2,
            gain=0.5,
            seed_offset=73,
        ),
        seed + 71_109,
    )
    width_map = network.fracture_width * (0.78 + 0.34 * (width_noise + 1.0) * 0.5)
    fracture_centerline = (band >= threshold) & domain_mask
    # 采用零等值线到采样点的局部距离，而不是直接使用 band 值域，
    # 这样 fracture_width 才是真正可读的世界方块宽度。
    base_distance_mask = distance_to_isoline <= width_map
    surface_y = np.floor(terrain).astype(np.int16)
    center_offset = -network.fracture_depth
    vertical_radius = max(network.fracture_height * 0.5, 1.0)
    vertical_shift = sample_noise(
        x,
        z,
        NoiseLayer(
            kind="fbm",
            scale=max(scale * 1.8, 40.0),
            octaves=2,
            gain=0.5,
            seed_offset=97,
        ),
        seed + 71_151,
    ) * min(vertical_radius * 0.18, 4.0)
    void = np.zeros((offsets.size, *terrain.shape), dtype=bool)
    for level_index, offset in enumerate(offsets):
        vertical = (float(offset) - center_offset - vertical_shift) / vertical_radius
        vertical_profile = np.clip(1.0 - np.abs(vertical) ** 1.35, 0.0, 1.0)
        # 裂隙顶部和底部渐缩，中段才达到最大窄缝宽度；这与普通洞穴的
        # 圆形横截面相反，侧视图会呈现“岩层被劈开”的构造形态。
        local_width = width_map * (0.42 + 0.58 * vertical_profile)
        void[level_index] = (
            fracture_centerline
            & base_distance_mask
            & (distance_to_isoline <= local_width)
            & (vertical_profile > 0.0)
            & (int(offset) < 0)
        )
    # 裂隙必须留在地表下方，避免 zero-isoline 直接把地表割开。
    return void & (offsets[:, None, None] < 0) & (
        offsets[:, None, None] >= -surface_y[None, :, :] + int(SPAN_MIN_Y)
    )


def _dijkstra(cost: np.ndarray, outlet_cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    """从多个出口反向传播累计代价，等价于轻量 fast-marching。"""

    distance = np.full(cost.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for z_index, x_index in outlet_cells:
        if distance[z_index, x_index] == 0.0:
            continue
        distance[z_index, x_index] = 0.0
        heapq.heappush(queue, (0.0, z_index, x_index))

    height, width = cost.shape
    while queue:
        current_distance, z_index, x_index = heapq.heappop(queue)
        if current_distance != distance[z_index, x_index]:
            continue
        for dz, dx, length in _NEIGHBOURS:
            next_z = z_index + dz
            next_x = x_index + dx
            if not (0 <= next_z < height and 0 <= next_x < width):
                continue
            candidate = current_distance + (
                cost[z_index, x_index] + cost[next_z, next_x]
            ) * 0.5 * length
            if candidate + 1.0e-12 < distance[next_z, next_x]:
                distance[next_z, next_x] = candidate
                heapq.heappush(queue, (candidate, next_z, next_x))
    return distance


def _backtrack(
    distance: np.ndarray,
    start: tuple[int, int],
    outlet_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """沿严格降低的累计代价回溯，保证路径无环并最终到达出口。"""

    current = start
    result = [current]
    for _ in range(distance.size):
        if current in outlet_cells:
            return tuple(result)
        z_index, x_index = current
        candidates: list[tuple[float, int, int]] = []
        for dz, dx, _length in _NEIGHBOURS:
            next_z = z_index + dz
            next_x = x_index + dx
            if 0 <= next_z < distance.shape[0] and 0 <= next_x < distance.shape[1]:
                candidate = float(distance[next_z, next_x])
                if candidate + 1.0e-9 < distance[z_index, x_index]:
                    candidates.append((candidate, next_z, next_x))
        if not candidates:
            return ()
        _, next_z, next_x = min(candidates)
        current = (next_z, next_x)
        result.append(current)
    return ()


def _lower_conduit_cost(
    cost: np.ndarray,
    path: tuple[tuple[int, int], ...],
    network: UndergroundRiverNetwork,
    cell_size: float,
) -> None:
    """降低已有 conduit 的周边代价，让后续入口偏向共享主干。"""

    radius = max(1, int(math.ceil(network.junction_radius / cell_size)))
    for z_index, x_index in path:
        z_start = max(0, z_index - radius)
        z_stop = min(cost.shape[0], z_index + radius + 1)
        x_start = max(0, x_index - radius)
        x_stop = min(cost.shape[1], x_index + radius + 1)
        local_z, local_x = np.ogrid[z_start:z_stop, x_start:x_stop]
        distance = np.hypot(local_z - z_index, local_x - x_index)
        falloff = np.clip(1.0 - distance / max(radius, 1), 0.0, 1.0)
        target = 1.0 - (1.0 - network.conduit_cost) * falloff
        cost[z_start:z_stop, x_start:x_stop] = np.minimum(
            cost[z_start:z_stop, x_start:x_stop], target
        )


def _network_geometry(
    network: UndergroundRiverNetwork,
    seed: int,
    cell_size: float,
) -> _NetworkGeometry:
    """生成入口到出口的多轮合并 conduit graph。"""

    grid_x, grid_z, x_axis, z_axis = _network_grid(network, cell_size)
    inlets, outlets = _endpoint_points(network, seed)
    outlet_cells = tuple(_nearest_cell(point, x_axis, z_axis) for point in outlets)
    outlet_cell_set = set(outlet_cells)
    inlet_cells = tuple(_nearest_cell(point, x_axis, z_axis) for point in inlets)
    cost = _travel_cost(grid_x, grid_z, network, seed)
    selected_inlets = tuple(
        point
        for index, point in enumerate(inlets)
        if index == 0 or _unit(seed, 13_000 + index) <= network.river_chance
    )
    if not selected_inlets:
        selected_inlets = (inlets[0],)
    selected_cells = tuple(_nearest_cell(point, x_axis, z_axis) for point in selected_inlets)

    paths: list[_RiverPath] = []
    seen_paths: set[tuple[tuple[int, int], ...]] = set()
    unreachable: set[Point] = set()
    for _iteration in range(network.network_iterations):
        # 当前轮次的出口累计代价对所有入口共用；只有已有 conduit 降价后
        # 才进入下一轮重算，避免对同一张 cost map 重复执行 Dijkstra。
        distance = _dijkstra(cost, outlet_cells)
        for inlet_index, (inlet, cell) in enumerate(zip(selected_inlets, selected_cells)):
            route = _backtrack(distance, cell, outlet_cell_set)
            if not route:
                unreachable.add(inlet)
                continue
            if route not in seen_paths:
                seen_paths.add(route)
                paths.append(
                    _RiverPath(
                        tuple(Point(float(x_axis[x]), float(z_axis[z])) for z, x in route),
                        inlet_index,
                    )
                )
            _lower_conduit_cost(cost, route, network, cell_size)

    counts: Counter[tuple[int, int]] = Counter()
    for path in paths:
        counts.update(_nearest_cell(point, x_axis, z_axis) for point in path.points)
    junction_cells = [cell for cell, count in counts.items() if count > 1]
    junctions = tuple(Point(float(x_axis[x]), float(z_axis[z])) for z, x in junction_cells)
    return _NetworkGeometry(
        paths=tuple(paths),
        inlets=tuple(selected_inlets),
        outlets=outlets,
        junctions=junctions,
        unreachable_inlets=tuple(sorted(unreachable, key=lambda point: (point.x, point.z))),
    )


def _distance_and_progress(
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """计算网格到 conduit 折线的距离及从 inlet 到 outlet 的进度。"""

    distance = np.full(x.shape, np.inf, dtype=np.float64)
    progress_distance = np.zeros(x.shape, dtype=np.float64)
    cumulative = [0.0]
    for start, end in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + math.hypot(end.x - start.x, end.z - start.z))
    total = max(cumulative[-1], 1.0e-9)
    for segment_index, (start, end) in enumerate(zip(path, path[1:])):
        dx = end.x - start.x
        dz = end.z - start.z
        length_sq = max(dx * dx + dz * dz, 1.0e-12)
        amount = np.clip(
            ((x - start.x) * dx + (z - start.z) * dz) / length_sq,
            0.0,
            1.0,
        )
        closest_x = start.x + amount * dx
        closest_z = start.z + amount * dz
        candidate_distance = np.hypot(x - closest_x, z - closest_z)
        better = candidate_distance < distance
        distance = np.where(better, candidate_distance, distance)
        progress_distance = np.where(
            better,
            cumulative[segment_index] + amount * math.sqrt(length_sq),
            progress_distance,
        )
    return distance, np.clip(progress_distance / total, 0.0, 1.0)


def _emit_water(
    blocks: dict[tuple[int, int, int], UndergroundWaterBlock],
    *,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    surface: np.ndarray,
    offsets: np.ndarray,
    void: np.ndarray,
    z_index: int,
    x_index: int,
    depth: int,
    kind: Literal["river", "lake"],
    flowing: bool,
) -> None:
    levels = np.flatnonzero(void[:, z_index, x_index])
    if not levels.size:
        return
    # offset 已按从地表向下排序；先填洞底，再逐格填到水面，
    # 避免水悬在腔顶。
    for level_index in levels[:depth]:
        y = int(round(float(surface[z_index, x_index]) + float(offsets[level_index])))
        if y <= SPAN_MIN_Y:
            continue
        key = (int(round(x_axis[x_index])), y, int(round(z_axis[z_index])))
        blocks[key] = UndergroundWaterBlock(
            x=key[0], y=key[1], z=key[2], kind=kind, flowing=flowing
        )


def _path_void_and_water(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    offsets: np.ndarray,
    network: UndergroundRiverNetwork,
    path: tuple[Point, ...],
    path_index: int,
    water: dict[tuple[int, int, int], UndergroundWaterBlock],
) -> np.ndarray:
    distance, progress = _distance_and_progress(x, z, path)
    width = network.width * max(0.62, 1.0 - path_index * 0.06)
    channel = distance <= width
    surface_y = np.floor(terrain).astype(np.int16)
    vertical_radius = max(network.height * 0.5, 1.0)
    void = np.zeros((offsets.size, *terrain.shape), dtype=bool)
    center_offset = -(
        network.source_depth
        + (network.outlet_depth - network.source_depth) * progress
    )
    for level_index, offset in enumerate(offsets):
        vertical = (float(offset) - center_offset) / vertical_radius
        void[level_index] = channel & (np.abs(vertical) <= 1.0) & (int(offset) < 0)
    for z_index, x_index in zip(*np.where(channel)):
        _emit_water(
            water,
            x_axis=x[0, :],
            z_axis=z[:, 0],
            surface=surface_y,
            offsets=offsets,
            void=void,
            z_index=int(z_index),
            x_index=int(x_index),
            depth=network.water_depth,
            kind="river",
            flowing=True,
        )
    return void


def _lake_center_candidates(
    geometry: _NetworkGeometry,
    network: UndergroundRiverNetwork,
    seed: int,
) -> tuple[Point, ...]:
    """优先在共享 conduit junction 放湖，退化时才使用河道站点。"""

    candidates = list(geometry.junctions)
    for path in geometry.paths:
        if len(path.points) > 2:
            candidates.append(path.points[len(path.points) // 2])
    if not candidates:
        return ()
    ordered: list[Point] = []
    for index in range(network.lake_count):
        choice = int(_unit(seed, 20_001 + index * 11) * len(candidates)) % len(candidates)
        point = candidates[choice]
        if all(
            math.hypot(point.x - other.x, point.z - other.z) >= network.lake_radius
            for other in ordered
        ):
            ordered.append(point)
    return tuple(ordered)


def _lake_void_and_water(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    offsets: np.ndarray,
    network: UndergroundRiverNetwork,
    center: Point,
    seed: int,
    water: dict[tuple[int, int, int], UndergroundWaterBlock],
) -> np.ndarray:
    radius = network.lake_radius * (0.75 + 0.5 * _unit(seed, 21_001))
    normalized = np.hypot((x - center.x) / radius, (z - center.z) / radius)
    lake_mask = normalized <= 1.0
    lake_depth = network.source_depth + (network.outlet_depth - network.source_depth) * 0.65
    center_offset = -lake_depth
    vertical_radius = max(network.height * 0.75, 1.0)
    void = np.zeros((offsets.size, *terrain.shape), dtype=bool)
    for level_index, offset in enumerate(offsets):
        vertical = (float(offset) - center_offset) / vertical_radius
        void[level_index] = lake_mask & (np.abs(vertical) <= 1.0) & (int(offset) < 0)
    surface_y = np.floor(terrain).astype(np.int16)
    for z_index, x_index in zip(*np.where(lake_mask)):
        _emit_water(
            water,
            x_axis=x[0, :],
            z_axis=z[:, 0],
            surface=surface_y,
            offsets=offsets,
            void=void,
            z_index=int(z_index),
            x_index=int(x_index),
            depth=network.lake_depth,
            kind="lake",
            flowing=False,
        )
    return void


def generate_underground_rivers(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    offsets: np.ndarray,
    network: UndergroundRiverNetwork,
    seed: int,
) -> UndergroundRiverResult:
    """生成不读取普通洞穴空腔的地下河/地下湖。"""

    river_void = np.zeros((offsets.size, *terrain.shape), dtype=bool)
    water: dict[tuple[int, int, int], UndergroundWaterBlock] = {}
    cell_size = _grid_cell_size(x, z)
    geometry = _network_geometry(network, seed, cell_size)
    for path_index, path in enumerate(geometry.paths):
        river_void |= _path_void_and_water(
            terrain,
            x,
            z,
            offsets,
            network,
            path.points,
            path_index,
            water,
        )

    for lake_index, center in enumerate(_lake_center_candidates(geometry, network, seed)):
        river_void |= _lake_void_and_water(
            terrain,
            x,
            z,
            offsets,
            network,
            center,
            seed + lake_index * 101,
            water,
        )

    resource_blocks = generate_river_resource_blocks(
        terrain,
        x,
        z,
        offsets,
        river_void,
        tuple(water.values()),
        tuple(path.points for path in geometry.paths),
        network,
        seed + 91_007,
    )
    fracture_void = _fracture_void(
        terrain,
        x,
        z,
        offsets,
        network,
        seed,
    )
    return UndergroundRiverResult(
        void=river_void,
        water_blocks=tuple(water.values()),
        fracture_void=fracture_void,
        resource_blocks=resource_blocks,
        paths=tuple(path.points for path in geometry.paths),
        inlets=geometry.inlets,
        outlets=geometry.outlets,
        junctions=geometry.junctions,
        unreachable_inlets=geometry.unreachable_inlets,
    )


__all__ = ["UndergroundRiverResult", "generate_underground_rivers"]
