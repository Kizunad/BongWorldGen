"""地表水体阶段：河流水面、河床和河床材质。"""

from __future__ import annotations

import numpy as np

from .geometry import polyline_distance_and_progress, polyline_stations, sample_regular_grid
from .terrain_config import NoiseLayer, River
from .world_config import TerrainRecipe
from .noise import sample_noise


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    """消除单个站点的尖峰，不引入新的过冲。"""

    radius = min(4, max(1, profile.size // 32))
    kernel = np.full(2 * radius + 1, 1.0 / (2 * radius + 1), dtype=np.float64)
    padded = np.pad(profile, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _river_surface_profile(
    terrain: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    river: River,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿河流生成单调水面，并返回两岸水位上限。"""

    total_length = sum(
        float(np.hypot(end.x - start.x, end.z - start.z))
        for start, end in zip(river.path, river.path[1:])
    )
    station_count = max(32, min(256, int(total_length / max(cell_size * 4.0, 1.0)) + 1))
    station_t, station_x, station_z = polyline_stations(river.path, station_count)
    tangent_x = np.gradient(station_x)
    tangent_z = np.gradient(station_z)
    tangent_length = np.maximum(np.hypot(tangent_x, tangent_z), 1.0e-9)
    normal_x = -tangent_z / tangent_length
    normal_z = tangent_x / tangent_length
    width_profile = river.width * (1.0 + (river.widening - 1.0) * station_t)
    terrain_profile = sample_regular_grid(
        terrain,
        grid_x,
        grid_z,
        station_x,
        station_z,
    )
    center_profile = _smooth_profile(terrain_profile)
    bank_offset = width_profile * 1.2
    left_bank = sample_regular_grid(
        terrain,
        grid_x,
        grid_z,
        station_x + normal_x * bank_offset,
        station_z + normal_z * bank_offset,
    )
    right_bank = sample_regular_grid(
        terrain,
        grid_x,
        grid_z,
        station_x - normal_x * bank_offset,
        station_z - normal_z * bank_offset,
    )
    bank_profile = _smooth_profile(np.minimum(left_bank, right_bank))
    bank_cap = np.floor(bank_profile) - river.bank_clearance
    surface_profile = np.minimum(center_profile, bank_profile) - river.bank_clearance
    surface_profile = np.minimum(surface_profile, bank_cap)
    surface_profile -= river.water_drop * station_t
    # 河流剖面必须向下游流动：可以下降，但不能爬过路径上的横向山脊。
    surface_profile = np.minimum.accumulate(surface_profile)
    return station_t, surface_profile, bank_cap


def apply_surface_rivers(
    terrain: np.ndarray,
    water: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    recipe: TerrainRecipe,
    seed: int,
    riverbed_id: np.ndarray,
    riverbed_palette: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把配方中的地表河流刻入高度、水面和河床材质场。"""

    output_height = terrain
    output_water = water
    output_riverbed = riverbed_id
    palette_index = {name: index for index, name in enumerate(riverbed_palette)}
    cell_size = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0

    for river_index, river in enumerate(recipe.rivers):
        distance, progress = polyline_distance_and_progress(x, z, river.path)
        width = river.width * (1.0 + (river.widening - 1.0) * progress)
        channel_mask = distance <= width
        # 河床额外覆盖一格，防止连续场量化到方块坐标后漏出裸露边缘。
        riverbed_mask = distance <= width + cell_size
        station_t, water_profile, bank_profile = _river_surface_profile(
            output_height, x, z, river, cell_size
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

        # 已有湖水可以并入河道，但水面不能超过同一纵向剖面的两岸上限。
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
            # 低频材质斑块让相邻河床方块保持连贯，避免每列都随机换
            # 材质。
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


__all__ = ["apply_surface_rivers"]
