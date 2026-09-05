"""导出完整世界的 raster 数据和语义清单。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .adapters import to_bong_tile, write_bong_raster
from .adapters.bong_raster import BASE_SURFACE_PALETTE, UNDERGROUND_MATERIAL_PALETTE
from .engine.glaciers import (
    GLACIAL_COVER_PALETTE,
    GLACIAL_CREVASSE_PALETTE,
    GLACIAL_LANDFORM_PALETTE,
    GLACIAL_SURFACE_PALETTE,
    PERMAFROST_PALETTE,
)
from .engine.mountains import MOUNTAIN_SURFACE_PALETTE
from .engine.climate import (
    CLIMATE_PALETTE,
    CLIMATE_SURFACE_PALETTE,
    CLIMATE_TRANSITION_PALETTE,
)
from .data.world import PoiDefinition, ZoneDefinition
from .data.world_definition import WORLD
from .data.recipes import DEFAULT_RECIPE
from .data.wilderness import wilderness_palette_manifest
from .engine import (
    TerrainRecipe,
    generate_heightfield,
    settlement_interest_point_manifest,
    settlement_spawn_area_manifest,
)


SPAN_ENCODING = {
    "max_spans": 4,
    "bytes_per_column": 16,
    "sentinel": 32767,
    "count_file": "spans_count.bin",
    "spans_file": "spans.bin",
    "slot_layout": "i16_le(floor_y, ceiling_y) x max_spans, unused slots = sentinel",
}
OVERVIEW_STRIDE = 32


def _zone_manifest(zone: ZoneDefinition) -> dict[str, object]:
    """把配方区域元数据投影为控制台 manifest 结构。"""

    # 适配逻辑保持在这里，让 Python 类型化区域模型独立于浏览器 manifest 契约。
    return {
        "name": zone.name,
        "display_name": zone.display_name,
        "terrain_profile": zone.terrain_profile,
        "spirit_qi": zone.spirit_qi,
        "danger_level": zone.danger_level,
        "worldgen": {
            "center_xz": [zone.center_x, zone.center_z],
            "size_xz": [zone.size_x, zone.size_z],
            "shape": zone.shape,
            "boundary": {"mode": zone.boundary_mode, "width": zone.boundary_width},
            "source": "authored_zone_data",
        },
    }


def _poi_manifest(zone_name: str, poi: PoiDefinition) -> dict[str, object]:
    return {
        "zone": zone_name,
        "kind": poi.kind,
        "name": poi.name,
        "pos_xyz": list(poi.pos_xyz),
        "tags": list(poi.tags),
        "unlock": poi.unlock,
        "qi_affinity": poi.qi_affinity,
        "danger_bias": poi.danger_bias,
    }


def _tile_range(minimum: int, maximum: int, tile_size: int) -> range:
    first = math.floor(minimum / tile_size)
    last = math.floor(maximum / tile_size)
    return range(first, last + 1)


def export_preview_world(
    output_dir: Path,
    *,
    recipe: TerrainRecipe = DEFAULT_RECIPE,
    seed: int = 812731,
    min_x: int = -4096,
    max_x: int = 4095,
    min_z: int = -4096,
    max_z: int = 4095,
    tile_size: int = 256,
) -> Path:
    """生成有边界世界中的全部 tile，并返回 manifest 路径。"""

    if tile_size < 1 or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a positive power of two")
    if min_x > max_x or min_z > max_z:
        raise ValueError("world bounds must be ordered")
    rasters_dir = output_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    tile_entries: list[dict[str, object]] = []
    overview_width = (max_x - min_x + 1 + OVERVIEW_STRIDE - 1) // OVERVIEW_STRIDE
    overview_height = (max_z - min_z + 1 + OVERVIEW_STRIDE - 1) // OVERVIEW_STRIDE
    overview_elevation = np.full((overview_height, overview_width), np.nan, dtype=np.float32)
    overview_uplift = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_surface = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_surface_material = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_mountain_material = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_mountain_weight = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_mountain_snowline = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_mountain_score = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_mountain_slope = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_mountain_exposure = np.zeros((overview_height, overview_width), dtype=np.float32)
    overview_mountain_rock_exposure = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_surface_visible = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_surface_cover_depth = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_permafrost = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_glacial_landform = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_glacial_crevasse = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_snow_accumulation = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_glacial_mass_balance = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_watershed_divide = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_hydraulic_erosion = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_hydraulic_deposition = np.zeros(
        (overview_height, overview_width), dtype=np.float32
    )
    overview_wilderness = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_climate = np.zeros((overview_height, overview_width), dtype=np.uint8)
    overview_climate_transition = np.zeros((overview_height, overview_width), dtype=np.uint8)
    resource_palette = tuple(
        sorted(
            {
                f"underground_river:{material}"
                for network in recipe.underground_rivers
                for material in (*network.ore_materials, *network.plant_materials)
            }
            | {f"solid:{spec.material}" for spec in recipe.solid_ores}
            | {
                f"cave:{material}"
                for network in recipe.caves
                for material in network.placeholder_materials
            }
            | {
                spec.resource_id
                for system in recipe.glaciers
                for spec in system.snow_mountain_resources
            }
        )
    )
    resource_catalog: dict[str, dict[str, object]] = {}
    settlement_spawn_areas: dict[str, dict[str, object]] = {}
    settlement_interest_points: dict[str, dict[str, object]] = {}
    for network in recipe.caves:
        for material, rarity in zip(network.placeholder_materials, network.placeholder_rarities):
            resource_catalog.setdefault(
                f"cave:{material}",
                {
                    "resource_id": f"cave:{material}",
                    "material": f"minecraft:{material}",
                    "source": "cave",
                    "rarity": rarity,
                },
            )
    for network in recipe.underground_rivers:
        for material, rarity in (
            *zip(network.ore_materials, network.ore_rarities),
            *zip(network.plant_materials, network.plant_rarities),
        ):
            resource_catalog.setdefault(
                f"underground_river:{material}",
                {
                    "resource_id": f"underground_river:{material}",
                    "material": f"minecraft:{material}",
                    "source": "underground_river",
                    "rarity": rarity,
                },
            )
    for spec in recipe.solid_ores:
        resource_catalog.setdefault(
            f"solid:{spec.material}",
            {
                "resource_id": f"solid:{spec.material}",
                "material": f"minecraft:{spec.material}",
                "source": "solid_ore",
                "rarity": spec.rarity,
            },
        )
    for system in recipe.glaciers:
        for spec in system.snow_mountain_resources:
            resource_catalog.setdefault(
                spec.resource_id,
                {
                    "resource_id": spec.resource_id,
                    "material": f"minecraft:{spec.material}",
                    "source": "snow_mountain",
                    "rarity": spec.rarity,
                },
            )
    resource_catalog_entries = [resource_catalog[key] for key in sorted(resource_catalog)]

    for tile_z in _tile_range(min_z, max_z, tile_size):
        for tile_x in _tile_range(min_x, max_x, tile_size):
            origin_x = tile_x * tile_size
            origin_z = tile_z * tile_size
            field = generate_heightfield(
                recipe,
                width=tile_size,
                height=tile_size,
                seed=seed,
                origin_x=origin_x,
                origin_z=origin_z,
            )
            tile = to_bong_tile(field, sea_level=recipe.sea_level)
            for area in tile.settlement_spawn_areas:
                settlement_spawn_areas.setdefault(
                    area.area_id,
                    settlement_spawn_area_manifest(area),
                )
            for point in tile.settlement_interest_points:
                settlement_interest_points.setdefault(
                    point.point_id,
                    settlement_interest_point_manifest(point),
                )
            for local_z in range(0, tile_size, OVERVIEW_STRIDE):
                world_z = origin_z + local_z
                oz = (world_z - min_z) // OVERVIEW_STRIDE
                if oz < 0 or oz >= overview_height:
                    continue
                for local_x in range(0, tile_size, OVERVIEW_STRIDE):
                    world_x = origin_x + local_x
                    ox = (world_x - min_x) // OVERVIEW_STRIDE
                    if 0 <= ox < overview_width:
                        overview_elevation[oz, ox] = tile.height[local_z, local_x]
                        overview_uplift[oz, ox] = tile.uplift[local_z, local_x]
                        overview_surface[oz, ox] = tile.surface_id[local_z, local_x]
                        overview_surface_material[oz, ox] = tile.surface_material_id[local_z, local_x]
                        overview_mountain_material[oz, ox] = tile.mountain_material_id[local_z, local_x]
                        overview_mountain_weight[oz, ox] = tile.mountain_weight[local_z, local_x]
                        overview_mountain_snowline[oz, ox] = tile.mountain_snowline[local_z, local_x]
                        overview_mountain_score[oz, ox] = tile.mountain_material_score[local_z, local_x]
                        overview_mountain_slope[oz, ox] = tile.mountain_slope_angle[local_z, local_x]
                        overview_mountain_exposure[oz, ox] = tile.mountain_exposure[local_z, local_x]
                        overview_mountain_rock_exposure[oz, ox] = tile.mountain_rock_exposure[
                            local_z, local_x
                        ]
                        overview_surface_visible[oz, ox] = tile.surface_visible_id[local_z, local_x]
                        overview_surface_cover_depth[oz, ox] = np.sum(
                            tile.surface_cover_layers[:, local_z, local_x], dtype=np.uint16
                        )
                        overview_permafrost[oz, ox] = tile.permafrost_id[local_z, local_x]
                        overview_glacial_landform[oz, ox] = tile.glacial_landform_id[
                            local_z, local_x
                        ]
                        overview_glacial_crevasse[oz, ox] = tile.glacial_crevasse_id[
                            local_z, local_x
                        ]
                        overview_snow_accumulation[oz, ox] = tile.snow_accumulation[
                            local_z, local_x
                        ]
                        overview_glacial_mass_balance[oz, ox] = tile.glacial_mass_balance[
                            local_z, local_x
                        ]
                        overview_watershed_divide[oz, ox] = tile.watershed_divide[
                            local_z, local_x
                        ]
                        overview_hydraulic_erosion[oz, ox] = tile.hydraulic_erosion[
                            local_z, local_x
                        ]
                        overview_hydraulic_deposition[oz, ox] = tile.hydraulic_deposition[
                            local_z, local_x
                        ]
                        overview_wilderness[oz, ox] = tile.wilderness_id[local_z, local_x]
                        overview_climate[oz, ox] = tile.climate_id[local_z, local_x]
                        overview_climate_transition[oz, ox] = tile.climate_transition_id[
                            local_z, local_x
                        ]
            write_bong_raster(
                tile,
                rasters_dir,
                tile_x=tile_x,
                tile_z=tile_z,
                world_name=recipe.name,
                write_manifest=False,
                resource_palette=resource_palette,
            )
            tile_entries.append(
                {
                    "tile_x": tile_x,
                    "tile_z": tile_z,
                    "dir": f"tile_{tile_x}_{tile_z}",
                    "zones": ["procedural_world"],
                    "layers": [
                        "surface_id",
                        "subsurface_id",
                        "water_level",
                        "uplift",
                        "glacial_landform_id",
                        "glacial_water_id",
                        "glacial_discharge",
                        "valley_depth",
                        "valley_flow_accumulation",
                        "valley_stream_power",
            "watershed_divide",
            "hydraulic_erosion",
            "hydraulic_deposition",
                        "snow_accumulation",
                        "glacial_mass_balance",
                        "biome_id",
                        "feature_mask",
                        "boundary_weight",
                        "wilderness_id",
                        "riverbed_id",
                        "cave_id",
                        "fracture_id",
                        "surface_material_id",
                        "mountain_material_id",
                        "mountain_weight",
                        "mountain_snowline",
                        "mountain_material_score",
                        "mountain_slope_angle",
                        "mountain_exposure",
                        "mountain_rock_exposure",
                        "permafrost_id",
                        "surface_cover_layers",
                        "climate_id",
                        "climate_transition_id",
                        "climate_transition_weight",
                        "climate_surface_id",
                        "settlement_spawn_areas",
                        "settlement_interest_points",
                    ],
                    "spans": True,
                }
            )

    if not np.isfinite(overview_elevation).all():
        raise ValueError("overview sampling left uncovered cells; choose aligned world bounds")
    overview_elevation.tofile(rasters_dir / "overview_height.bin")
    overview_uplift.tofile(rasters_dir / "overview_uplift.bin")
    overview_surface.tofile(rasters_dir / "overview_surface_id.bin")
    overview_surface_material.tofile(rasters_dir / "overview_surface_material_id.bin")
    overview_mountain_material.tofile(rasters_dir / "overview_mountain_material_id.bin")
    overview_mountain_weight.tofile(rasters_dir / "overview_mountain_weight.bin")
    overview_mountain_snowline.tofile(rasters_dir / "overview_mountain_snowline.bin")
    overview_mountain_score.tofile(rasters_dir / "overview_mountain_material_score.bin")
    overview_mountain_slope.tofile(rasters_dir / "overview_mountain_slope_angle.bin")
    overview_mountain_exposure.tofile(rasters_dir / "overview_mountain_exposure.bin")
    overview_mountain_rock_exposure.tofile(
        rasters_dir / "overview_mountain_rock_exposure.bin"
    )
    overview_surface_visible.tofile(rasters_dir / "overview_surface_visible_id.bin")
    overview_surface_cover_depth.tofile(rasters_dir / "overview_surface_cover_depth.bin")
    overview_permafrost.tofile(rasters_dir / "overview_permafrost_id.bin")
    overview_glacial_landform.tofile(rasters_dir / "overview_glacial_landform_id.bin")
    overview_glacial_crevasse.tofile(rasters_dir / "overview_glacial_crevasse_id.bin")
    overview_snow_accumulation.tofile(rasters_dir / "overview_snow_accumulation.bin")
    overview_glacial_mass_balance.tofile(
        rasters_dir / "overview_glacial_mass_balance.bin"
    )
    overview_watershed_divide.tofile(rasters_dir / "overview_watershed_divide.bin")
    overview_hydraulic_erosion.tofile(rasters_dir / "overview_hydraulic_erosion.bin")
    overview_hydraulic_deposition.tofile(rasters_dir / "overview_hydraulic_deposition.bin")
    overview_wilderness.tofile(rasters_dir / "overview_wilderness_id.bin")
    overview_climate.tofile(rasters_dir / "overview_climate_id.bin")
    overview_climate_transition.tofile(rasters_dir / "overview_climate_transition_id.bin")
    (rasters_dir / "settlement_spawn_areas.json").write_text(
        json.dumps(
            [settlement_spawn_areas[key] for key in sorted(settlement_spawn_areas)],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (rasters_dir / "settlement_interest_points.json").write_text(
        json.dumps(
            [
                settlement_interest_points[key]
                for key in sorted(settlement_interest_points)
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    surface_palette = list(BASE_SURFACE_PALETTE)
    riverbed_palette = list(
        dict.fromkeys(material for river in recipe.rivers for material in river.bed_materials)
    )
    for material in riverbed_palette:
        if material not in surface_palette:
            surface_palette.append(material)

    manifest = {
        "version": 2,
        "backend": "raster",
        "world_name": recipe.name,
        "tile_size": tile_size,
        "spans_encoding": SPAN_ENCODING,
        "world_bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
        },
        "overview": {
            "width": overview_width,
            "height": overview_height,
            "origin_x": min_x,
            "origin_z": min_z,
            "cell_size": OVERVIEW_STRIDE,
            "height_file": "overview_height.bin",
            "uplift_file": "overview_uplift.bin",
            "surface_file": "overview_surface_id.bin",
            "surface_material_file": "overview_surface_material_id.bin",
            "mountain_material_file": "overview_mountain_material_id.bin",
            "mountain_weight_file": "overview_mountain_weight.bin",
            "mountain_snowline_file": "overview_mountain_snowline.bin",
            "mountain_material_score_file": "overview_mountain_material_score.bin",
            "mountain_slope_angle_file": "overview_mountain_slope_angle.bin",
            "mountain_exposure_file": "overview_mountain_exposure.bin",
            "mountain_rock_exposure_file": "overview_mountain_rock_exposure.bin",
            "surface_visible_file": "overview_surface_visible_id.bin",
            "surface_cover_depth_file": "overview_surface_cover_depth.bin",
            "permafrost_file": "overview_permafrost_id.bin",
            "glacial_landform_file": "overview_glacial_landform_id.bin",
            "glacial_crevasse_file": "overview_glacial_crevasse_id.bin",
            "snow_accumulation_file": "overview_snow_accumulation.bin",
            "glacial_mass_balance_file": "overview_glacial_mass_balance.bin",
            "watershed_divide_file": "overview_watershed_divide.bin",
            "hydraulic_erosion_file": "overview_hydraulic_erosion.bin",
            "hydraulic_deposition_file": "overview_hydraulic_deposition.bin",
            "wilderness_file": "overview_wilderness_id.bin",
            "climate_file": "overview_climate_id.bin",
            "climate_transition_file": "overview_climate_transition_id.bin",
            "note": "Display-only overview; full-resolution tile rasters remain authoritative.",
        },
        "surface_palette": surface_palette,
        "surface_material_palette": list(GLACIAL_SURFACE_PALETTE),
        "mountain_material_palette": list(MOUNTAIN_SURFACE_PALETTE),
        "mountain_material_encoding": {
            "material_file": "mountain_material_id.bin",
            "weight_file": "mountain_weight.bin",
            "snowline_file": "mountain_snowline.bin",
            "score_file": "mountain_material_score.bin",
            "slope_angle_file": "mountain_slope_angle.bin",
            "exposure_file": "mountain_exposure.bin",
            "rock_exposure_file": "mountain_rock_exposure.bin",
            "id_dtype": "u8",
            "field_dtype": "f32",
            "none": 0,
            "description": "仅群山有限距离场内的雪、冰、蓝冰、裸岩材质与驱动场",
        },
        "surface_visible_palette": [
            *dict.fromkeys(
                [
                    "minecraft:stone",
                    "minecraft:coarse_dirt",
                    "minecraft:gravel",
                    "minecraft:grass_block",
                    "minecraft:sand",
                    *[
                        f"minecraft:{material.removeprefix('minecraft:').replace('-', '_')}"
                        for material in (*riverbed_palette, *MOUNTAIN_SURFACE_PALETTE)
                    ],
                    *PERMAFROST_PALETTE,
                    *GLACIAL_COVER_PALETTE,
                    *GLACIAL_CREVASSE_PALETTE,
                ]
            )
        ],
        "surface_cover_palette": list(GLACIAL_COVER_PALETTE),
        "surface_cover_encoding": {
            "file": "surface_cover_layers.bin",
            "dtype": "u8",
            "shape": [4, "height", "width"],
            "order": list(GLACIAL_COVER_PALETTE),
            "base": "height layer is retained; these blocks are stacked above it",
        },
        "permafrost_palette": list(PERMAFROST_PALETTE),
        "permafrost_encoding": {
            "file": "permafrost_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "glacial_landform_palette": list(GLACIAL_LANDFORM_PALETTE),
        "glacial_landform_encoding": {
            "file": "glacial_landform_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "glacial_crevasse_palette": list(GLACIAL_CREVASSE_PALETTE),
        "glacial_crevasse_encoding": {
            "file": "glacial_crevasse_id.bin",
            "none": 0,
            "dtype": "u8",
            "description": "冰川表面横向裂隙；与地下 fracture_id 分离",
        },
        "climate_palette": list(CLIMATE_PALETTE),
        "climate_transition_palette": list(CLIMATE_TRANSITION_PALETTE),
        "climate_surface_palette": list(CLIMATE_SURFACE_PALETTE),
        "climate_encoding": {"file": "climate_id.bin", "none": 0, "dtype": "u8"},
        "climate_transition_encoding": {
            "id_file": "climate_transition_id.bin",
            "weight_file": "climate_transition_weight.bin",
            "none": 0,
            "id_dtype": "u8",
            "weight_dtype": "f32",
        },
        "climate_surface_encoding": {
            "file": "climate_surface_id.bin",
            "none": 0,
            "dtype": "u8",
        },
        "glacial_water_palette": ["glacial_meltwater"],
        "glacial_water_encoding": {
            "id_file": "glacial_water_id.bin",
            "discharge_file": "glacial_discharge.bin",
            "id_dtype": "u8",
            "discharge_dtype": "f32",
            "none": 0,
        },
        "glacial_mass_balance_encoding": {
            "snow_accumulation_file": "snow_accumulation.bin",
            "mass_balance_file": "glacial_mass_balance.bin",
            "dtype": "f32",
            "mass_balance_sign": {"positive": "accumulation", "negative": "ablation"},
        },
        "uplift_encoding": {
            "file": "uplift.bin",
            "dtype": "f32",
            "description": (
                "Mountain Spine 产生、供 Stream Power 耦合的正向构造抬升场"
            ),
        },
        "watershed_divide_encoding": {
            "file": "watershed_divide.bin",
            "dtype": "f32",
            "description": "D8 流域出口边界在 Uplift + Stream Power 之后形成的山脊修饰抬升量",
        },
        "hydraulic_erosion_encoding": {
            "erosion_file": "hydraulic_erosion.bin",
            "deposition_file": "hydraulic_deposition.bin",
            "dtype": "f32",
            "description": "末端 droplet hydraulic erosion refinement 的累计下切与冲积沉积",
        },
        "riverbed_palette": riverbed_palette,
        "cave_palette": [network.name for network in recipe.caves],
        "fracture_palette": [network.name for network in recipe.underground_rivers],
        "fracture_encoding": {
            "file": "fracture_id.bin",
            "none": 0,
            "dtype": "u8",
            "vertical_range": "spans.bin",
        },
        "underground_water_encoding": {
            "file": "underground_water.bin",
            "record_size": 16,
            "fields": "i32 world_x, i16 world_y, i32 world_z, u8 kind, u8 flowing, 4 reserved bytes",
            "kind": {"1": "river", "2": "lake"},
        },
        "underground_resource_palette": list(resource_palette),
        "underground_resource_catalog": resource_catalog_entries,
        "underground_material_palette": sorted(
            set(UNDERGROUND_MATERIAL_PALETTE)
            | {entry["material"] for entry in resource_catalog_entries}
        ),
        "underground_resources_encoding": {
            "file": "underground_resources.bin",
            "record_size": 16,
            "fields": "i32 world_x, i16 world_y, i32 world_z, u16 resource_id, u8 source_id, u8 rarity_id, 2 reserved bytes",
        },
        "settlement_spawn_area_encoding": {
            "file": "settlement_spawn_areas.json",
            "format": "json array",
            "description": "每栋建筑的 NPC 刷新/防守区域；spawn_bounds 从 ground_y 开始，footprint 保留完整结构高度",
        },
        "settlement_spawn_areas": [
            settlement_spawn_areas[key] for key in sorted(settlement_spawn_areas)
        ],
        "settlement_interest_point_encoding": {
            "file": "settlement_interest_points.json",
            "format": "json array",
            "description": "聚落可查询兴趣点；包含建筑中心、道路节点和可供 NPC 刷新的屋内点位",
        },
        "settlement_interest_points": [
            settlement_interest_points[key]
            for key in sorted(settlement_interest_points)
        ],
        "biome_palette": ["minecraft:plains", "minecraft:river"],
        "wilderness_palette": wilderness_palette_manifest(),
        "tiles": tile_entries,
        "pois": [
            _poi_manifest(zone.name, poi)
            for zone in WORLD.zones
            for poi in zone.pois
        ],
        "poi_connections": [],
        "zones": [
            {
                "name": "procedural_world",
                "display_name": "程序山河",
                "terrain_profile": recipe.name,
                "spirit_qi": 0.35,
                "danger_level": 2,
                "worldgen": {"generator": "bong_worldgen", "seed": seed},
            },
            *(_zone_manifest(zone) for zone in WORLD.zones),
        ],
        "semantic_layers": [
            "cave_id",
            "fracture_id",
            "glacial_landform_id",
            "glacial_crevasse_id",
            "glacial_water_id",
            "glacial_discharge",
            "valley_depth",
            "valley_flow_accumulation",
            "valley_stream_power",
                        "watershed_divide",
                        "hydraulic_erosion",
                        "hydraulic_deposition",
            "uplift",
            "snow_accumulation",
            "glacial_mass_balance",
            "surface_material_id",
            "surface_cover_layers",
            "permafrost_id",
            "climate_id",
            "climate_transition_id",
            "climate_transition_weight",
            "climate_surface_id",
            "underground_water",
            "underground_resources",
            "settlement_spawn_areas",
            "settlement_interest_points",
        ],
        "vertical_layers": ["spans"],
        "notes": [
            "Generated by standalone BongWorldGen.",
            "BlueMap 使用同一份程序化世界数据生成 Anvil 预览。",
        ],
    }
    manifest_path = rasters_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["SPAN_ENCODING", "export_preview_world"]
