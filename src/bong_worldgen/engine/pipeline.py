"""独立程序化地形引擎的配方解释器。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .caves import generate_underground
from .geometry import polyline_distance_and_progress
from .models import Heightfield, NoiseLayer, Point, River, TerrainRecipe
from .noise import sample_noise
from .solids import add_floating_islands


SurfaceSampler = Callable[[np.ndarray, np.ndarray], np.ndarray]
RiverProfile = tuple[np.ndarray, np.ndarray, np.ndarray]


def _coordinate_grid(
    width: int, height: int, origin_x: float, origin_z: float, cell_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if width < 1 or height < 1:
        raise ValueError("heightfield dimensions must be positive")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    xs = origin_x + np.arange(width, dtype=np.float64) * cell_size
    zs = origin_z + np.arange(height, dtype=np.float64) * cell_size
    return np.meshgrid(xs, zs, indexing="xy")


def _apply_mountains(
    terrain: np.ndarray, x: np.ndarray, z: np.ndarray, recipe: TerrainRecipe, seed: int
) -> np.ndarray:
    output = terrain
    for index, mountain in enumerate(recipe.mountains):
        distance, _ = polyline_distance_and_progress(x, z, mountain.path)
        ridge_mask = np.exp(-((distance / mountain.width) ** 2))
        roughness = sample_noise(x, z, mountain.roughness, seed + index * 7919)
        contrast = mountain.roughness_contrast
        roughness = np.clip((1.0 - contrast) + contrast * roughness, 0.0, 1.0)
        uplift = ridge_mask * roughness * mountain.height
        if mountain.valley_depth:
            valley_mask = np.exp(-((distance / (mountain.width * 0.24)) ** 2))
            uplift -= valley_mask * mountain.valley_depth
        output = output + uplift
    return output


def _apply_basins(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
) -> np.ndarray:
    output = terrain
    for basin in recipe.basins:
        normalized = np.hypot(
            (x - basin.center.x) / basin.radius_x,
            (z - basin.center.z) / basin.radius_z,
        )
        bowl = np.exp(-(normalized**2.4))
        output = output - bowl * basin.depth
    return output


def _polyline_stations(
    path: tuple[Point, ...], station_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回沿折线采样的归一化进度和坐标。"""

    if station_count < 2:
        raise ValueError("river profile needs at least two stations")
    lengths = np.asarray(
        [
            np.hypot(path[index + 1].x - point.x, path[index + 1].z - point.z)
            for index, point in enumerate(path[:-1])
        ],
        dtype=np.float64,
    )
    total = max(float(lengths.sum()), 1.0e-9)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, total, station_count)
    segment_ids = np.searchsorted(cumulative[1:], distances, side="right")
    segment_ids = np.minimum(segment_ids, len(path) - 2)
    segment_start = cumulative[segment_ids]
    segment_length = np.maximum(lengths[segment_ids], 1.0e-9)
    local_t = np.clip((distances - segment_start) / segment_length, 0.0, 1.0)
    start = path[:-1]
    end = path[1:]
    start_x = np.asarray([point.x for point in start], dtype=np.float64)
    start_z = np.asarray([point.z for point in start], dtype=np.float64)
    end_x = np.asarray([point.x for point in end], dtype=np.float64)
    end_z = np.asarray([point.z for point in end], dtype=np.float64)
    station_x = start_x[segment_ids] + (end_x[segment_ids] - start_x[segment_ids]) * local_t
    station_z = start_z[segment_ids] + (end_z[segment_ids] - start_z[segment_ids]) * local_t
    return distances / total, station_x, station_z


def _sample_regular_grid(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    sample_x: np.ndarray,
    sample_z: np.ndarray,
) -> np.ndarray:
    """在规则世界坐标网格上双线性采样，并把边界坐标钳制到网格内。"""

    height, width = values.shape
    cell_x = float(grid_x[0, 1] - grid_x[0, 0]) if width > 1 else 1.0
    cell_z = float(grid_z[1, 0] - grid_z[0, 0]) if height > 1 else 1.0
    grid_origin_x = float(grid_x[0, 0])
    grid_origin_z = float(grid_z[0, 0])
    gx = np.clip((sample_x - grid_origin_x) / cell_x, 0.0, width - 1.0)
    gz = np.clip((sample_z - grid_origin_z) / cell_z, 0.0, height - 1.0)
    x0 = np.floor(gx).astype(np.int64)
    z0 = np.floor(gz).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    z1 = np.minimum(z0 + 1, height - 1)
    tx = gx - x0
    tz = gz - z0
    top = values[z0, x0] * (1.0 - tx) + values[z0, x1] * tx
    bottom = values[z1, x0] * (1.0 - tx) + values[z1, x1] * tx
    return top * (1.0 - tz) + bottom * tz


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    """消除单个站点的地形尖峰，同时不引入过冲。"""

    radius = min(4, max(1, profile.size // 32))
    kernel = np.full(2 * radius + 1, 1.0 / (2 * radius + 1), dtype=np.float64)
    padded = np.pad(profile, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _river_surface_profile(
    river: River,
    sample: SurfaceSampler,
    station_count: int,
) -> RiverProfile:
    """沿一条手工河流构建单调下降的水面和两岸上限。"""

    station_t, station_x, station_z = _polyline_stations(river.path, station_count)
    tangent_x = np.gradient(station_x)
    tangent_z = np.gradient(station_z)
    tangent_length = np.maximum(np.hypot(tangent_x, tangent_z), 1.0e-9)
    normal_x = -tangent_z / tangent_length
    normal_z = tangent_x / tangent_length
    width_profile = river.width * (1.0 + (river.widening - 1.0) * station_t)
    terrain_profile = sample(station_x, station_z)
    center_profile = _smooth_profile(terrain_profile)
    bank_offset = width_profile * 1.2
    left_bank = sample(
        station_x + normal_x * bank_offset,
        station_z + normal_z * bank_offset,
    )
    right_bank = sample(
        station_x - normal_x * bank_offset,
        station_z - normal_z * bank_offset,
    )

    bank_profile = np.minimum(left_bank, right_bank)
    bank_profile = _smooth_profile(bank_profile)
    bank_cap = np.floor(bank_profile) - river.bank_clearance
    surface_profile = np.minimum(
        center_profile,
        bank_profile,
    ) - river.bank_clearance
    surface_profile = np.minimum(surface_profile, bank_cap)
    surface_profile -= river.water_drop * station_t
    # 河流剖面必须向下游流动：可以下降，但不能爬过路径上的横向山脊。
    surface_profile = np.minimum.accumulate(surface_profile)
    return station_t, surface_profile, bank_cap


def _river_station_count(river: River, spacing: float, limit: int) -> int:
    length = sum(float(np.hypot(end.x - start.x, end.z - start.z))
                 for start, end in zip(river.path, river.path[1:]))
    return max(32, min(limit, int(length / spacing) + 1))


def prepare_river_profiles(recipe: TerrainRecipe, sampler: SurfaceSampler) -> tuple[RiverProfile, ...]:
    """Precompute reusable river sections from a fixed world-coordinate sampler."""

    return tuple(_river_surface_profile(river, sampler, _river_station_count(river, 8.0, 4096))
                 for river in recipe.rivers)


def _apply_rivers(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
    seed: int,
    riverbed_id: np.ndarray,
    riverbed_palette: tuple[str, ...],
    surface_sampler: SurfaceSampler | None = None,
    river_profiles: tuple[RiverProfile, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_height = terrain
    output_water = water
    output_riverbed = riverbed_id
    if river_profiles is None and surface_sampler is not None:
        river_profiles = prepare_river_profiles(recipe, surface_sampler)
    if river_profiles is not None and len(river_profiles) != len(recipe.rivers):
        raise ValueError("river profiles must match the recipe rivers")
    palette_index = {name: index for index, name in enumerate(riverbed_palette)}
    for river_index, river in enumerate(recipe.rivers):
        distance, progress = polyline_distance_and_progress(x, z, river.path)
        width = river.width * (1.0 + (river.widening - 1.0) * progress)
        channel_mask = distance <= width
        cell_size = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0
        riverbed_mask = distance <= width + (1.0 if river_profiles is not None else cell_size)
        if river_profiles is not None:
            station_t, water_profile, bank_profile = river_profiles[river_index]
        else:
            station_t, water_profile, bank_profile = _river_surface_profile(
                river, lambda sx, sz: _sample_regular_grid(output_height, x, z, sx, sz),
                _river_station_count(river, max(cell_size * 4.0, 1.0), 256),
            )
        water_surface = np.floor(np.interp(progress, station_t, water_profile))
        bank_cap = np.floor(np.interp(progress, station_t, bank_profile))
        normalized_distance = np.clip(distance / np.maximum(width, 1.0e-6), 0.0, 1.0)
        cross_section = np.power(np.maximum(1.0 - normalized_distance**2, 0.0), 1.5)
        target_bed = water_surface - river.depth * cross_section
        output_height = np.where(
            channel_mask,
            np.minimum(output_height, target_bed),
            output_height,
        )

        # 水体是纵向剖面；已有湖水可以并入河道，但仍受剖面级
        # 两岸上限约束。
        existing_water = np.where(output_water >= 0.0, output_water, -np.inf)
        channel_water = np.maximum(existing_water, water_surface)
        channel_water = np.minimum(channel_water, bank_cap)
        channel_water = np.where(
            channel_mask & (channel_water > output_height), channel_water, -1.0
        )
        output_water = np.where(channel_mask, channel_water, output_water)

        material_count = len(river.bed_materials)
        material_noise = sample_noise(
            x,
            z,
            # 低频材质斑块让相邻河床方块保持连贯，避免每列都随机
            # 换材质。
            layer=NoiseLayer(
                kind="value",
                scale=max(river.width * 3.5, 8.0),
                octaves=1,
                seed_offset=river_index * 1543,
            ),
            seed=seed + 271_828,
        )
        material_choice = np.minimum(
            ((material_noise + 1.0) * 0.5 * material_count).astype(np.int64),
            material_count - 1,
        )
        material_ids = np.asarray(
            [palette_index[name] for name in river.bed_materials], dtype=np.int16
        )
        output_riverbed = np.where(
            riverbed_mask,
            material_ids[material_choice],
            output_riverbed,
        )
    return output_height, output_water, output_riverbed


def sample_surface(
    recipe: TerrainRecipe,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample dry terrain and moisture at arbitrary world-coordinate arrays.

    Layout composition can mix these continuous fields before rivers and
    underground geometry are evaluated. No world layout is known here.
    """

    x, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64))
    terrain = np.full(x.shape, recipe.base_height, dtype=np.float64)
    for layer in recipe.base_noise:
        terrain += layer.amplitude * sample_noise(x, z, layer, seed)
    terrain = _apply_basins(terrain, x, z, recipe)
    terrain = _apply_mountains(terrain, x, z, recipe, seed)
    for plateau in recipe.plateaus:
        dx, dz = x - plateau.center.x, z - plateau.center.z
        if plateau.rotation:
            cosine, sine = np.cos(plateau.rotation), np.sin(plateau.rotation)
            dx, dz = dx * cosine + dz * sine, -dx * sine + dz * cosine
        if plateau.shape == "rectangle":
            distance = np.minimum(plateau.radius_x - np.abs(dx), plateau.radius_z - np.abs(dz))
        else:
            radius = np.hypot(dx / plateau.radius_x, dz / plateau.radius_z)
            distance = (1.0 - radius) * min(plateau.radius_x, plateau.radius_z)
        t = np.clip(distance / plateau.edge_width, 0.0, 1.0)
        weight = t**3 * (t * (t * 6.0 - 15.0) + 10.0)
        terrain = terrain * (1.0 - weight) + plateau.height * weight
    moisture = sample_noise(x, z, recipe.moisture_noise, seed + 100_003)
    return terrain, np.clip((moisture + 1.0) * 0.5, 0.0, 1.0)


def finish_heightfield(
    recipe: TerrainRecipe,
    terrain: np.ndarray,
    moisture: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
    *,
    surface_sampler: SurfaceSampler | None = None,
    river_profiles: tuple[RiverProfile, ...] | None = None,
) -> Heightfield:
    """Apply water and underground geometry to an already composed surface."""

    water = np.where(terrain < recipe.sea_level, recipe.sea_level, -1.0)
    riverbed_palette = tuple(
        dict.fromkeys(material for river in recipe.rivers for material in river.bed_materials)
    )
    riverbed = np.full(terrain.shape, -1, dtype=np.int16)
    terrain, water, riverbed = _apply_rivers(
        terrain, water, x, z, recipe, seed, riverbed, riverbed_palette, surface_sampler, river_profiles,
    )
    water = np.where(water >= 0.0, np.maximum(water, terrain), -1.0)
    underground = generate_underground(terrain, x, z, recipe, seed)
    field = Heightfield(
        height=np.ascontiguousarray(terrain, dtype=np.float32),
        moisture=np.ascontiguousarray(moisture, dtype=np.float32),
        water_level=np.ascontiguousarray(water, dtype=np.float32),
        riverbed_id=np.ascontiguousarray(riverbed, dtype=np.int16),
        riverbed_palette=riverbed_palette,
        solid_spans=np.ascontiguousarray(underground.solid_spans, dtype=np.int16),
        underground_blocks=underground.blocks,
        cave_id=np.ascontiguousarray(underground.cave_id, dtype=np.uint8),
        cave_palette=underground.cave_palette,
    )
    return add_floating_islands(field, x, z, recipe, seed)


def generate_heightfield(
    recipe: TerrainRecipe,
    *,
    width: int,
    height: int,
    seed: int,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
    cell_size: float = 1.0,
) -> Heightfield:
    """将一份配方计算为连续存储的 float32 标量场。"""

    x, z = _coordinate_grid(width, height, origin_x, origin_z, cell_size)
    terrain, moisture = sample_surface(recipe, x, z, seed)
    return finish_heightfield(recipe, terrain, moisture, x, z, seed)
