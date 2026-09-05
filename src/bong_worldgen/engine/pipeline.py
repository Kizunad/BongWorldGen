"""独立程序化地形引擎的配方解释器。"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .canyons import apply_canyons
from .caves import generate_underground
from .climate import (
    CLIMATE_PALETTE,
    CLIMATE_SURFACE_PALETTE,
    CLIMATE_TRANSITION_PALETTE,
    sample_climate,
)
from .glaciers import (
    GLACIAL_COVER_PALETTE,
    GLACIAL_CREVASSE_PALETTE,
    GLACIAL_LANDFORM_PALETTE,
    GLACIAL_SURFACE_PALETTE,
    GLACIAL_WATER_PALETTE,
    PERMAFROST_PALETTE,
    GlacialFlowPlan,
    apply_glaciers,
    apply_glacial_meltwater,
    apply_surface_protrusions,
    glacial_landform_ids,
    glacial_surface_layers,
    glacial_surface_materials,
    generate_snow_mountain_resource_blocks,
    sample_glacial_crevasses,
    permafrost_surface_materials,
    plan_glacial_flow,
    sample_glacial_mass_balance,
    sample_wind_snow_field,
)
from .generated_world import Heightfield
from .hydraulic import apply_hydraulic_erosion
from .mountains import (
    MOUNTAIN_SURFACE_PALETTE,
    apply_mountains_with_uplift,
    sample_mountain_material_field,
)
from .noise import sample_noise
from .relief import apply_natural_relief
from .surface_water import apply_surface_rivers
from .surface import build_visible_surface_layer
from .town import generate_town_result
from .standalone_structures import generate_standalone_structures
from .terrain_config import NoiseLayer, Point
from .spawn_plain import SpawnPlainSelection, apply_spawn_plain, select_spawn_plain_anchor
from .valleys import (
    ValleyErosionPlan,
    apply_valley_erosion,
    apply_watershed_divide,
    plan_valley_erosion,
    sample_valley_fields,
)
from .world_config import TerrainRecipe


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


def _generate_uplift_terrain(
    recipe: TerrainRecipe,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """生成冰川阶段的输入地形，供主流程和坡向 halo 共同复用。"""

    terrain, _ = _generate_uplift_fields(recipe, x, z, seed)
    return terrain


def _generate_natural_terrain(
    recipe: TerrainRecipe,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """生成未加入山脉等显式结构的自然基底场。

    出生平原的选址只能观察这层连续自然场；如果把已声明的山脉也带入
    评分，后续整形就可能把一条本应保留的山脉夷平。
    """

    terrain = np.full(x.shape, recipe.base_height, dtype=np.float64)
    for layer in recipe.base_noise:
        terrain += layer.amplitude * sample_noise(x, z, layer, seed)
    if recipe.natural_relief is not None:
        terrain = apply_natural_relief(terrain, x, z, recipe.natural_relief, seed)
    return _apply_basins(terrain, x, z, recipe)


def _apply_spawn_plain_if_enabled(
    recipe: TerrainRecipe,
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """在显式山脉/峡谷之前，对自然基底做一次出生平原二次加工。"""

    settings = recipe.spawn_plain
    if settings is None or not settings.enabled:
        return terrain
    return apply_spawn_plain(
        terrain,
        x,
        z,
        settings,
        _select_spawn_plain(recipe, seed),
        seed,
    )


def _generate_uplift_fields(
    recipe: TerrainRecipe,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(抬升后地形, 构造 Uplift 场)``，供水文阶段消费。"""

    terrain = _generate_natural_terrain(recipe, x, z, seed)
    terrain = _apply_spawn_plain_if_enabled(recipe, terrain, x, z, seed)
    return apply_mountains_with_uplift(terrain, x, z, recipe.mountains, seed)


@lru_cache(maxsize=4)
def _plan_valley_systems(recipe: TerrainRecipe, seed: int) -> tuple[ValleyErosionPlan, ...]:
    """在每个配方水文域上规划一次，缓存结果供所有 tile 采样。"""

    plans: list[ValleyErosionPlan] = []
    for index, system in enumerate(recipe.valleys):
        sample_count_x = max(
            2,
            math.ceil(system.domain_extent_x * 2.0 / system.planning_resolution),
        )
        sample_count_z = max(
            2,
            math.ceil(system.domain_extent_z * 2.0 / system.planning_resolution),
        )
        axis_x = np.linspace(
            system.domain_center.x - system.domain_extent_x,
            system.domain_center.x + system.domain_extent_x,
            sample_count_x + 1,
            dtype=np.float64,
        )
        axis_z = np.linspace(
            system.domain_center.z - system.domain_extent_z,
            system.domain_center.z + system.domain_extent_z,
            sample_count_z + 1,
            dtype=np.float64,
        )
        planning_x, planning_z = np.meshgrid(axis_x, axis_z, indexing="xy")
        planning_terrain, planning_uplift = _generate_uplift_fields(
            recipe,
            planning_x,
            planning_z,
            seed,
        )
        plans.append(
            plan_valley_erosion(
                system,
                planning_terrain,
                planning_x,
                planning_z,
                seed=seed + index * 1_009_003,
                sea_level=recipe.sea_level,
                uplift=planning_uplift,
            )
        )
    return tuple(plans)


def _generate_pre_glacial_terrain(
    recipe: TerrainRecipe,
    x: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """生成冰川阶段的输入地形，并把固定域山谷侵蚀采样回当前 tile。"""

    terrain = _generate_uplift_terrain(recipe, x, z, seed)
    if recipe.valleys:
        valley_plans = _plan_valley_systems(recipe, seed)
        terrain = apply_valley_erosion(terrain, x, z, valley_plans)
        terrain = apply_watershed_divide(terrain, x, z, valley_plans)
    return terrain


@lru_cache(maxsize=4)
def _select_spawn_plain(recipe: TerrainRecipe, seed: int) -> SpawnPlainSelection:
    """在自然宏观场中选择一次出生平原锚点，所有 tile 复用同一结果。"""

    settings = recipe.spawn_plain
    if settings is None or not settings.enabled:
        center_x = 0.0 if settings is None else settings.center.x
        center_z = 0.0 if settings is None else settings.center.z
        return SpawnPlainSelection(center_x, center_z, recipe.base_height, 0.0)

    def sample_macro_terrain(sample_x: np.ndarray, sample_z: np.ndarray) -> np.ndarray:
        return _generate_natural_terrain(recipe, sample_x, sample_z, seed)

    return select_spawn_plain_anchor(
        settings,
        seed,
        sample_macro_terrain,
        sea_level=recipe.sea_level,
    )


def _sample_mass_balance_with_halo(
    recipe: TerrainRecipe,
    *,
    width: int,
    height: int,
    seed: int,
    origin_x: float,
    origin_z: float,
    cell_size: float,
):
    """在目标范围外多取一格，避免坡向在 raster tile 边缘产生接缝。"""

    if not recipe.glaciers:
        shape = (height, width)
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    halo_x, halo_z = _coordinate_grid(
        width + 2,
        height + 2,
        origin_x - cell_size,
        origin_z - cell_size,
        cell_size,
    )
    halo_terrain = _generate_pre_glacial_terrain(recipe, halo_x, halo_z, seed)
    halo_climate = (
        None
        if recipe.climate_world_bounds is None
        else sample_climate(halo_z, recipe.climate, recipe.climate_world_bounds)
    )
    field = sample_glacial_mass_balance(
        halo_terrain,
        halo_x,
        halo_z,
        recipe.glaciers,
        seed,
        recipe.sea_level,
        cold_weight=None if halo_climate is None else halo_climate.cold_weight,
    )
    return field.snow_accumulation[1:-1, 1:-1], field.mass_balance[1:-1, 1:-1]


def _plan_glacial_flows(
    recipe: TerrainRecipe,
    seed: int,
) -> tuple[GlacialFlowPlan, ...]:
    """在每个冰川系统的完整生成域上规划全局流路。

    规划网格不依赖当前输出 tile，所以同一 seed 下的冰斗、支流和共享主干
    在整图导出、分块导出和 Server 按需生成时保持一致。
    """

    plans: list[GlacialFlowPlan] = []
    for index, system in enumerate(recipe.glaciers):
        step_count = max(
            2,
            math.ceil(system.seed_extent * 2.0 / system.flow_planning_resolution),
        )
        spacing = system.seed_extent * 2.0 / step_count
        planning_x, planning_z = _coordinate_grid(
            step_count + 1,
            step_count + 1,
            system.seed_center.x - system.seed_extent,
            system.seed_center.z - system.seed_extent,
            spacing,
        )
        planning_terrain = _generate_pre_glacial_terrain(
            recipe,
            planning_x,
            planning_z,
            seed,
        )
        planning_climate = (
            None
            if recipe.climate_world_bounds is None
            else sample_climate(
                planning_z,
                recipe.climate,
                recipe.climate_world_bounds,
            )
        )
        system_seed = seed + index * 1_301_071
        mass_balance = sample_glacial_mass_balance(
            planning_terrain,
            planning_x,
            planning_z,
            (system,),
            system_seed,
            recipe.sea_level,
            cold_weight=(
                None if planning_climate is None else planning_climate.cold_weight
            ),
        )
        plans.append(
            plan_glacial_flow(
                system,
                planning_terrain,
                planning_x,
                planning_z,
                mass_balance.snow_accumulation,
                mass_balance.mass_balance,
                system_seed,
            )
        )
    return tuple(plans)


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
    terrain, uplift = _generate_uplift_fields(recipe, x, z, seed)
    valley_plans = _plan_valley_systems(recipe, seed)
    valley_field = sample_valley_fields(x, z, valley_plans)
    terrain = apply_valley_erosion(terrain, x, z, valley_plans)
    terrain = apply_watershed_divide(terrain, x, z, valley_plans)
    climate = None
    if recipe.climate_world_bounds is not None:
        climate = sample_climate(z, recipe.climate, recipe.climate_world_bounds)
    snow_accumulation, glacial_mass_balance = _sample_mass_balance_with_halo(
        recipe,
        width=width,
        height=height,
        seed=seed,
        origin_x=origin_x,
        origin_z=origin_z,
        cell_size=cell_size,
    )
    glacial_flows = _plan_glacial_flows(recipe, seed)
    glacial_landform_id = glacial_landform_ids(
        terrain,
        x,
        z,
        glacial_flows,
        recipe.sea_level,
        climate_cold_weight=None if climate is None else climate.cold_weight,
    )
    glacial_terrain = apply_glaciers(terrain, x, z, glacial_flows, recipe.sea_level)
    if climate is None:
        terrain = glacial_terrain
    else:
        # 冰川的几何仍由 recipe.glaciers 控制，但其影响按寒带权重渐入，
        # 这样寒带到温带不会出现一条人工直切的 carve 边界。
        terrain = terrain + (glacial_terrain - terrain) * climate.cold_weight
    surface_material_id = glacial_surface_materials(
        terrain,
        x,
        z,
        glacial_flows,
        seed,
        recipe.sea_level,
        climate_cold_weight=None if climate is None else climate.cold_weight,
    )
    if climate is not None and recipe.mountain_materials is None:
        # 没有群山材质场的自定义配方才使用寒带高地兜底；默认配方由
        # mountain_material_id 决定，不能再让气候带直接刷出一条直线。
        coverage_noise = (
            sample_noise(
                x,
                z,
                NoiseLayer(kind="value", scale=320.0, octaves=1, seed_offset=6_701),
                seed + 91_003,
            )
            + 1.0
        ) * 0.5
        snow_id = GLACIAL_SURFACE_PALETTE.index("minecraft:snow_block") + 1
        cold_land = (terrain >= recipe.sea_level) & (coverage_noise <= climate.cold_weight)
        surface_material_id[cold_land & (surface_material_id == 0)] = np.uint8(snow_id)
    terrain = apply_canyons(terrain, x, z, recipe.canyons, seed)
    terrain, _ = apply_surface_protrusions(
        terrain,
        x,
        z,
        surface_material_id,
        recipe.glaciers,
        seed,
        recipe.sea_level,
    )
    if recipe.hydraulic_erosion is None:
        hydraulic_erosion = np.zeros_like(terrain)
        hydraulic_deposition = np.zeros_like(terrain)
    else:
        hydraulic_result = apply_hydraulic_erosion(
            terrain,
            x,
            z,
            recipe.hydraulic_erosion,
            seed + 73_001,
        )
        terrain = hydraulic_result.height
        hydraulic_erosion = hydraulic_result.erosion
        hydraulic_deposition = hydraulic_result.deposition
    moisture = sample_noise(x, z, recipe.moisture_noise, seed + 100_003)
    moisture = np.clip((moisture + 1.0) * 0.5, 0.0, 1.0)
    water = np.where(terrain < recipe.sea_level, recipe.sea_level, -1.0)
    riverbed_palette = tuple(
        dict.fromkeys(material for river in recipe.rivers for material in river.bed_materials)
    )
    riverbed = np.full(terrain.shape, -1, dtype=np.int16)
    terrain, water, riverbed = apply_surface_rivers(
        terrain,
        water,
        x,
        z,
        recipe,
        seed,
        riverbed,
        riverbed_palette,
    )
    water = np.where(water >= 0.0, np.maximum(water, terrain), -1.0)
    glacial_water = apply_glacial_meltwater(
        terrain,
        water,
        x,
        z,
        glacial_flows,
        recipe.sea_level,
        cold_weight=None if climate is None else climate.cold_weight,
        cell_size=cell_size,
    )
    water = np.where(glacial_water.water_level >= 0.0, glacial_water.water_level, water)
    mountain_material_field = None
    if recipe.mountain_materials is not None:
        # 材质场必须读取河流完成后的最终高度；冰川材质只作为一个
        # glacier bonus 输入，实际写回仍由山脉距离场严格门控。
        mountain_material_field = sample_mountain_material_field(
            terrain,
            x,
            z,
            recipe.mountains,
            seed,
            recipe.mountain_materials,
            climate_cold_weight=None if climate is None else climate.cold_weight,
            glacier_weight=(glacial_landform_id > 0).astype(np.float64),
        )
    # 地下结果先于覆盖层确定；入口列的顶层实心段低于地表，后面会禁止在
    # 入口上方堆叠雪冰，避免 BlueMap 出现封口或悬空覆盖。
    underground = generate_underground(terrain, x, z, recipe, seed)
    settlement_anchor = (
        _select_spawn_plain(recipe, seed)
        if recipe.spawn_plain is not None
        else None
    )
    settlement_settings = recipe.town
    if settlement_settings is None:
        settlement_blocks = ()
        settlement_spawn_areas = ()
        settlement_interest_points = ()
    else:
        anchor = (
            Point(settlement_anchor.center_x, settlement_anchor.center_z)
            if settlement_anchor is not None
            else settlement_settings.center
        )
        settlement_result = generate_town_result(
            terrain,
            water,
            x,
            z,
            settlement_settings,
            anchor,
            seed,
        )
        settlement_blocks = settlement_result.blocks
        settlement_spawn_areas = settlement_result.spawn_areas
        settlement_interest_points = settlement_result.interest_points
    standalone_result = generate_standalone_structures(
        terrain,
        water,
        x,
        z,
        recipe.standalone_structures,
        seed,
        exclusion_center=(
            None
            if settlement_anchor is None
            else Point(settlement_anchor.center_x, settlement_anchor.center_z)
        ),
    )
    settlement_blocks = settlement_blocks + standalone_result.blocks
    settlement_spawn_areas = settlement_spawn_areas + standalone_result.spawn_areas
    settlement_interest_points = settlement_interest_points + standalone_result.interest_points
    surface_y = np.rint(terrain).astype(np.int16)
    surface_columns = underground.solid_spans[:, :, 0, 1] == surface_y

    # 覆盖层必须在河流和湖泊完成后计算；它只叠加方块，不修改 terrain，
    # 因此石头、草地和河床仍然是独立的原始基底。水面列不再生成雪/冰覆盖。
    # 风雪场在所有地形 carve、河流和冰川融水完成后采样，确保法线对应最终
    # 可见山坡；它同时供覆盖层和 Server 查询，避免两条路径使用不同坡向。
    wind_snow_field = sample_wind_snow_field(terrain, x, z, recipe.glaciers)
    surface_cover_layers = glacial_surface_layers(
        terrain,
        x,
        z,
        recipe.glaciers,
        seed,
        recipe.sea_level,
        climate_cold_weight=None if climate is None else climate.cold_weight,
        surface_material_id=surface_material_id,
        mountain_material_id=(
            None
            if mountain_material_field is None
            else mountain_material_field.material_id
        ),
        wind_snow_field=wind_snow_field,
        plans=glacial_flows,
    )
    surface_cover_layers[:, (water >= 0.0) | ~surface_columns] = 0
    glacial_crevasse_id = sample_glacial_crevasses(
        terrain,
        x,
        z,
        glacial_flows,
        seed + 88_001,
        recipe.sea_level,
        climate_cold_weight=None if climate is None else climate.cold_weight,
        surface_material_id=surface_material_id,
    )
    # 裂隙中央是空气，不能再让覆盖层从上方把裂缝封死；河湖列和洞口列
    # 同样不生成地表裂隙。
    crevasse_columns = glacial_crevasse_id > 0
    surface_cover_layers[:, crevasse_columns | (water >= 0.0) | ~surface_columns] = 0
    permafrost_id = permafrost_surface_materials(
        terrain,
        x,
        z,
        recipe.glaciers,
        seed,
        recipe.sea_level,
        cold_weight=None if climate is None else climate.cold_weight,
        water_level=water,
        existing_surface_material_id=surface_material_id,
        existing_surface_material_palette=GLACIAL_SURFACE_PALETTE,
    )
    if mountain_material_field is not None:
        # 裸岩是陡坡上有意保留的基岩暴露带，不能在最终合成时被冻土
        # 斑块重新盖住；Server 仍可通过 mountain_material_id 查询该列。
        rock_material_start = 1 + len(GLACIAL_SURFACE_PALETTE)
        rock_columns = mountain_material_field.material_id >= rock_material_start
        permafrost_id = np.where(rock_columns, 0, permafrost_id).astype(np.uint8, copy=False)
    surface_visible_id, surface_visible_palette = build_visible_surface_layer(
        terrain,
        water,
        sea_level=recipe.sea_level,
        riverbed_id=riverbed,
        riverbed_palette=riverbed_palette,
        surface_material_id=surface_material_id,
        surface_material_palette=MOUNTAIN_SURFACE_PALETTE,
        permafrost_id=permafrost_id,
        permafrost_palette=PERMAFROST_PALETTE,
        surface_cover_layers=surface_cover_layers,
        surface_cover_palette=GLACIAL_COVER_PALETTE,
        mountain_material_id=(
            None
            if mountain_material_field is None
            else mountain_material_field.material_id
        ),
        mountain_material_palette=(
            () if mountain_material_field is None else MOUNTAIN_SURFACE_PALETTE
        ),
        surface_columns=surface_columns,
        climate_surface_id=None if climate is None else climate.surface_id,
        climate_surface_palette=CLIMATE_SURFACE_PALETTE,
        glacial_crevasse_id=glacial_crevasse_id,
        glacial_crevasse_palette=GLACIAL_CREVASSE_PALETTE,
    )
    snow_mountain_blocks = generate_snow_mountain_resource_blocks(
        terrain,
        x,
        z,
        recipe.glaciers,
        seed + 191_003,
        mountain_weight=(
            np.zeros(terrain.shape, dtype=np.float64)
            if mountain_material_field is None
            else mountain_material_field.mountain_weight
        ),
        mountain_material_id=(
            np.zeros(terrain.shape, dtype=np.uint8)
            if mountain_material_field is None
            else mountain_material_field.material_id
        ),
        surface_material_id=surface_material_id,
        surface_cover_layers=surface_cover_layers,
        water_level=water,
        glacial_crevasse_id=glacial_crevasse_id,
        occupied_blocks=underground.blocks,
        climate_cold_weight=None if climate is None else climate.cold_weight,
    )

    return Heightfield(
        height=np.ascontiguousarray(terrain, dtype=np.float32),
        moisture=np.ascontiguousarray(moisture, dtype=np.float32),
        water_level=np.ascontiguousarray(water, dtype=np.float32),
        uplift=np.ascontiguousarray(uplift, dtype=np.float32),
        glacial_landform_id=np.ascontiguousarray(glacial_landform_id, dtype=np.uint8),
        glacial_landform_palette=GLACIAL_LANDFORM_PALETTE,
        glacial_water_id=np.ascontiguousarray(glacial_water.water_id, dtype=np.uint8),
        glacial_water_palette=GLACIAL_WATER_PALETTE,
        glacial_discharge=np.ascontiguousarray(glacial_water.discharge, dtype=np.float32),
        valley_depth=np.ascontiguousarray(valley_field.depth, dtype=np.float32),
        valley_flow_accumulation=np.ascontiguousarray(
            valley_field.flow_accumulation, dtype=np.float32
        ),
        valley_stream_power=np.ascontiguousarray(
            valley_field.stream_power, dtype=np.float32
        ),
        watershed_divide=np.ascontiguousarray(
            valley_field.watershed_divide, dtype=np.float32
        ),
        hydraulic_erosion=np.ascontiguousarray(hydraulic_erosion, dtype=np.float32),
        hydraulic_deposition=np.ascontiguousarray(hydraulic_deposition, dtype=np.float32),
        snow_accumulation=np.ascontiguousarray(snow_accumulation, dtype=np.float32),
        glacial_mass_balance=np.ascontiguousarray(glacial_mass_balance, dtype=np.float32),
        wind_snow_alignment=np.ascontiguousarray(
            wind_snow_field.normal_alignment, dtype=np.float32
        ),
        wind_snow_response=np.ascontiguousarray(
            wind_snow_field.response, dtype=np.float32
        ),
        riverbed_id=np.ascontiguousarray(riverbed, dtype=np.int16),
        riverbed_palette=riverbed_palette,
        solid_spans=np.ascontiguousarray(underground.solid_spans, dtype=np.int16),
        underground_blocks=underground.blocks + snow_mountain_blocks,
        underground_water_blocks=underground.water_blocks,
        settlement_blocks=settlement_blocks,
        settlement_spawn_areas=settlement_spawn_areas,
        settlement_interest_points=settlement_interest_points,
        cave_id=np.ascontiguousarray(underground.cave_id, dtype=np.uint8),
        cave_palette=underground.cave_palette,
        fracture_id=np.ascontiguousarray(underground.fracture_id, dtype=np.uint8),
        fracture_palette=underground.fracture_palette,
        surface_material_id=np.ascontiguousarray(surface_material_id, dtype=np.uint8),
        surface_material_palette=GLACIAL_SURFACE_PALETTE,
        mountain_material_id=(
            np.zeros(terrain.shape, dtype=np.uint8)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.material_id, dtype=np.uint8)
        ),
        mountain_material_palette=MOUNTAIN_SURFACE_PALETTE,
        mountain_weight=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.mountain_weight, dtype=np.float32)
        ),
        mountain_snowline=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.snowline, dtype=np.float32)
        ),
        mountain_material_score=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.snow_score, dtype=np.float32)
        ),
        mountain_slope_angle=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.slope_angle, dtype=np.float32)
        ),
        mountain_exposure=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(mountain_material_field.exposure, dtype=np.float32)
        ),
        mountain_rock_exposure=(
            np.zeros(terrain.shape, dtype=np.float32)
            if mountain_material_field is None
            else np.ascontiguousarray(
                mountain_material_field.rock_exposure, dtype=np.float32
            )
        ),
        surface_visible_id=np.ascontiguousarray(surface_visible_id, dtype=np.uint8),
        surface_visible_palette=surface_visible_palette,
        permafrost_id=np.ascontiguousarray(permafrost_id, dtype=np.uint8),
        permafrost_palette=PERMAFROST_PALETTE,
        surface_cover_layers=np.ascontiguousarray(surface_cover_layers, dtype=np.uint8),
        surface_cover_palette=GLACIAL_COVER_PALETTE,
        glacial_crevasse_id=np.ascontiguousarray(glacial_crevasse_id, dtype=np.uint8),
        glacial_crevasse_palette=GLACIAL_CREVASSE_PALETTE,
        climate_id=None if climate is None else np.ascontiguousarray(climate.climate_id),
        climate_palette=() if climate is None else CLIMATE_PALETTE,
        climate_transition_id=(
            None if climate is None else np.ascontiguousarray(climate.transition_id)
        ),
        climate_transition_palette=(
            () if climate is None else CLIMATE_TRANSITION_PALETTE
        ),
        climate_transition_weight=(
            None if climate is None else np.ascontiguousarray(climate.transition_weight)
        ),
        climate_surface_id=(
            None if climate is None else np.ascontiguousarray(climate.surface_id)
        ),
        climate_surface_palette=() if climate is None else CLIMATE_SURFACE_PALETTE,
    )
