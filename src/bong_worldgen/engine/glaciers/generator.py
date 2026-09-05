"""编排冰斗、U 型冰川谷和沉积地貌。"""

from __future__ import annotations

import numpy as np

from ..geometry import polyline_distance_and_progress
from ..terrain_config import GlacialSystem, NoiseLayer
from ..noise import sample_noise
from ..randomness import stable_text_seed, unit_interval
from .carving import (
    carve_cirque,
    carve_glacier_valley,
    cirque_carve_depth,
    glacier_valley_carve_depth,
)
from .deposits import (
    deposit_drumlins,
    deposit_terminal_moraine,
    drumlin_deposition,
    terminal_moraine_deposition,
)
from .erosion import apply_freeze_thaw, glacial_exposure_mask, talus_surface_mask
from .crevasses import sample_glacier_valley_field
from .topology import GlacialFlowPlan
from .wind_snow import WindSnowField, sample_wind_snow_field


# 0 保留给“无额外寒带覆盖”；其余值会写入 Heightfield 的独立材质层。
GLACIAL_SURFACE_PALETTE = (
    "minecraft:snow_block",
    "minecraft:powder_snow",
    "minecraft:ice",
    "minecraft:packed_ice",
    "minecraft:blue_ice",
    "minecraft:gravel",
)

# 覆盖层按“从地表向上”的语义保存；写入 Minecraft 时会反向堆叠，
# 使蓝冰在底部、松雪在最上方。分层结构参考公开冰川特征放置思路：
# https://github.com/oargudo/glaciers
GLACIAL_COVER_PALETTE = (
    "minecraft:powder_snow",
    "minecraft:snow_block",
    "minecraft:ice",
    "minecraft:blue_ice",
)

# 0 表示无冰川地貌；ID 顺序是稳定的 raster/Server 契约。
GLACIAL_LANDFORM_PALETTE = (
    "cirque",
    "u_valley",
    "terminal_moraine",
    "drumlin_field",
)


def glacial_landform_ids(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    sea_level: float,
    *,
    climate_cold_weight: np.ndarray | None = None,
) -> np.ndarray:
    """生成冰川宏观地貌语义 ID，不修改地形。

    ID 直接来自各算法的实际侵蚀/沉积贡献，而不是从最终高度反推。
    重叠时可见沉积地貌优先于侵蚀地貌，终碛脊优先于鼓丘；冰斗优先于
    谷源重叠区。
    """

    if terrain.shape != x.shape or terrain.shape != z.shape:
        raise ValueError("glacial landform fields must share a shape")
    if climate_cold_weight is not None and climate_cold_weight.shape != terrain.shape:
        raise ValueError("climate cold weight must have the same shape as terrain")

    output = np.zeros(terrain.shape, dtype=np.uint8)
    climate_mask = (
        np.ones(terrain.shape, dtype=bool)
        if climate_cold_weight is None
        else climate_cold_weight > 0.0
    )
    for plan in plans:
        system = plan.system
        system_seed = plan.system_seed
        cirques = plan.cirques
        paths = plan.paths

        valley_mask = np.zeros(terrain.shape, dtype=bool)
        for path in paths:
            carve = glacier_valley_carve_depth(x, z, path, system)
            valley_mask |= carve >= max(0.5, system.valley_depth * 0.03)
        output[valley_mask & climate_mask] = 2

        cirque_mask = np.zeros(terrain.shape, dtype=bool)
        for cirque in cirques:
            carve = cirque_carve_depth(terrain, x, z, cirque, system, sea_level)
            cirque_mask |= carve >= max(0.5, system.cirque_depth * 0.03)
        output[cirque_mask & climate_mask] = 1

        drumlin = drumlin_deposition(terrain, x, z, system, system_seed, sea_level)
        # 语义范围只标记沉积主体，不把高斯尾部无限扩成整片鼓丘区。
        drumlin_mask = drumlin >= max(0.5, system.drumlin_height * 0.35)
        output[drumlin_mask & climate_mask] = 4

        moraine_mask = np.zeros(terrain.shape, dtype=bool)
        for path in paths:
            deposition = terminal_moraine_deposition(x, z, path, system)
            moraine_mask |= deposition >= max(0.5, system.moraine_height * 0.35)
        output[moraine_mask & climate_mask] = 3
    return output


def apply_glaciers(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    sea_level: float,
) -> np.ndarray:
    """按配方顺序应用寒带冰川地貌。"""

    output = np.asarray(terrain, dtype=np.float64).copy()
    for plan in plans:
        system = plan.system
        system_seed = plan.system_seed
        cirques = plan.cirques
        for cirque in cirques:
            output = carve_cirque(output, x, z, cirque, system, sea_level)
        for path in plan.paths:
            output = carve_glacier_valley(output, x, z, path, system)
        for path in plan.paths:
            output = deposit_terminal_moraine(output, x, z, path, system)
        output = deposit_drumlins(output, x, z, system, system_seed, sea_level)
        exposure = glacial_exposure_mask(output, x, z, plan, sea_level, margin=1.0)
        output = apply_freeze_thaw(output, x, z, exposure, system, system_seed)
    return output


def apply_surface_protrusions(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    surface_material_id: np.ndarray,
    systems: tuple[GlacialSystem, ...],
    seed: int,
    sea_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在寒带地表按概率抬高零散的一格方块。

    结果返回 ``(terrain, raised_mask)``。每个候选列最多增加一个方块高度，
    材质仍由传入的 ``surface_material_id`` 决定，因此不会把冰块凸起误换成
    另一种方块。低频噪声负责形成局部块团，细噪声只负责打散边缘。
    """

    if terrain.shape != surface_material_id.shape:
        raise ValueError("surface material ids must have the same shape as terrain")
    output = np.asarray(terrain, dtype=np.float64).copy()
    raised = np.zeros(terrain.shape, dtype=bool)
    for index, system in enumerate(systems):
        probability = system.surface_protrusion_probability
        if probability <= 0:
            continue
        system_seed = seed + index * 1_301_071 + 4_001
        material_mask = surface_material_id > 0
        macro = sample_noise(
            x,
            z,
            NoiseLayer(
                kind="fbm",
                scale=max(system.surface_patch_radius * 0.55, 8.0),
                octaves=2,
                gain=0.55,
                seed_offset=811,
            ),
            system_seed,
        )
        micro = sample_noise(
            x,
            z,
            NoiseLayer(
                kind="value",
                scale=max(system.surface_patch_radius * 0.18, 4.0),
                octaves=1,
                seed_offset=823,
            ),
            system_seed,
        )
        # value noise 近似均匀落在 [-1, 1]，将概率映射为阈值；微噪声保留
        # 一点空隙，使团块边缘不成为完整的棋盘格。
        threshold = 1.0 - 2.0 * probability
        candidate = (
            material_mask
            & ~raised
            & (macro >= threshold)
            & (micro > -0.35)
            & (output >= sea_level + system.cirque_min_elevation * 0.30)
        )
        output[candidate] += 1.0
        raised |= candidate
    return output, raised


def glacial_surface_materials(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    plans: tuple[GlacialFlowPlan, ...],
    seed: int,
    sea_level: float,
    climate_cold_weight: np.ndarray | None = None,
) -> np.ndarray:
    """生成寒带冰雪地表材质 ID。

    这不是把整张地图按高度简单涂白，而是复用冰斗和冰川谷的几何范围：
    外缘使用雪块/粉雪，谷内依次使用冰、浮冰和蓝冰。这样 BlueMap 能直接
    呈现出“寒带冰川”而不是普通高山积雪。

    材质层的组织方式参考公开冰川项目的 ice/snow feature placement：
    https://github.com/oargudo/glaciers
    Minecraft 方块仅作为本项目 Anvil 适配器的真实占位，不复制外部源码。
    """

    if climate_cold_weight is not None and climate_cold_weight.shape != x.shape:
        raise ValueError("climate cold weight must have the same shape as terrain")
    output = np.zeros(x.shape, dtype=np.uint8)
    systems = tuple(plan.system for plan in plans)
    valley_field = sample_glacier_valley_field(x, z, plans)
    glacier_valley_mask = valley_field.mask
    glacier_pressure = valley_field.pressure
    for plan in plans:
        system = plan.system
        system_seed = plan.system_seed
        cirques = plan.cirques
        paths = plan.paths

        # 冰斗及冰川外缘先铺雪；高程门控避免冰川系统在低地留下不合理冰盖。
        snow_mask = np.zeros(x.shape, dtype=bool)
        for cirque in cirques:
            dx = x - cirque.center.x
            dz = z - cirque.center.z
            radial = np.hypot(dx, dz) / max(system.cirque_radius * cirque.radius_scale, 1.0e-6)
            snow_mask |= radial <= 1.28
        for path in paths:
            distance, _ = polyline_distance_and_progress(x, z, path)
            width = system.valley_source_width * (1.0 + system.valley_width_growth)
            snow_mask |= distance <= width * 1.45
        snow_mask &= terrain >= sea_level + system.cirque_min_elevation * 0.35
        output[snow_mask] = 1

        # 粉雪只占外缘的一部分，使用低频 seeded 噪声打破整齐的同心环。
        snow_noise = sample_noise(
            x,
            z,
            layer=NoiseLayer(kind="fbm", scale=180.0, octaves=2, gain=0.55, seed_offset=73),
            seed=system_seed,
        )
        powder_mask = snow_mask & (snow_noise > 0.05)
        output[powder_mask] = 2

        for path_index, path in enumerate(paths):
            distance, _ = polyline_distance_and_progress(x, z, path)
            width = system.valley_source_width * (1.0 + system.valley_width_growth)
            in_valley = distance <= width
            output[in_valley & (output < 3)] = 3
            packed = in_valley & (
                distance <= width * 0.62
            )
            output[packed] = 4
            # 蓝冰只出现在冰川谷的高压核心。压力由谷宽、距流线距离共同给出，
            # 再叠加低频噪声抽样成 veins；因此蓝冰不会形成完整覆盖层。
            blue_noise = sample_noise(
                x,
                z,
                layer=NoiseLayer(
                    kind="fbm", scale=75.0, octaves=2, gain=0.5, seed_offset=path_index * 19 + 91
                ),
                seed=system_seed,
            )
            pressure_noise = (blue_noise + 1.0) * 0.5
            vein_probability = np.clip(
                system.crevasse_blue_ice_probability * 0.55
                + system.cover_blue_ice_weight * 0.22 * glacier_pressure,
                0.0,
                0.62,
            )
            blue = (
                in_valley
                & (glacier_pressure >= 0.35)
                & (pressure_noise <= vein_probability)
            )
            output[blue] = 5

        # 用连续的局部团块覆盖几何基底。每个团块独立抽取材质和半径，
        # 团块之间允许重叠，后生成的团块覆盖前一个团块，形成自然的混合边界。
        if system.surface_patch_count > 0:
            weights = np.asarray(system.surface_material_weights, dtype=np.float64)
            cumulative = np.cumsum(weights / weights.sum())
            name_seed = sum(
                (char_index + 1) * ord(char)
                for char_index, char in enumerate(system.name)
            )
            cold_mask = output > 0
            # 团块锚定在冰斗和冰川路径上，再做局部偏移；这样每个实际的冰川
            # 窗口都有可见的混合材质，而不是把所有概率样本撒到生成域外。
            anchors = [cirque.center for cirque in cirques]
            for path in paths:
                stride = max(1, len(path) // 8)
                anchors.extend(path[::stride])
            if not anchors:
                anchors = [system.seed_center]
            best_patch_weight = np.zeros(x.shape, dtype=np.float64)
            best_patch_material = np.zeros(x.shape, dtype=np.uint8)
            for patch_index in range(system.surface_patch_count):
                base = name_seed + 150_000 + patch_index * 23_011
                # 轮询锚点保证每条冰川都能获得局部团块，随机数只负责锚点
                # 内的偏移和材质抽签，不会把一条短冰川完全漏掉。
                anchor = anchors[patch_index % len(anchors)]
                center_angle = unit_interval(system_seed, base + 2) * np.pi * 2.0
                center_radius = system.surface_patch_radius * (
                    0.15 + 1.15 * unit_interval(system_seed, base + 3)
                )
                center_x = anchor.x + np.cos(center_angle) * center_radius
                center_z = anchor.z + np.sin(center_angle) * center_radius
                patch_radius = system.surface_patch_radius * (
                    0.55 + 0.95 * unit_interval(system_seed, base + 4)
                )
                material_roll = unit_interval(system_seed, base + 5)
                material_index = int(np.searchsorted(cumulative, material_roll, side="right"))
                material_index = min(material_index, len(GLACIAL_SURFACE_PALETTE) - 1)
                distance = np.hypot(x - center_x, z - center_z)
                # smoothstep 外缘让团块不是硬圆形贴图；阈值控制有效覆盖面积。
                patch_t = np.clip((patch_radius - distance) / max(patch_radius * 0.30, 1.0e-6), 0.0, 1.0)
                patch_weight = patch_t * patch_t * (3.0 - 2.0 * patch_t)
                patch_mask = patch_weight > best_patch_weight
                best_patch_weight[patch_mask] = patch_weight[patch_mask]
                best_patch_material[patch_mask] = material_index + 1
            patch_mask = cold_mask & (best_patch_weight > 0.42)
            # 冰/浮冰/蓝冰团块必须落在真实冰川谷内；雪和碎石团块仍可在冰斗
            # 与高山外缘出现。这样局部概率团块不会把整片山坡变成蓝冰。
            glacier_material = best_patch_material >= 3
            patch_mask &= (~glacier_material) | glacier_valley_mask
            output[patch_mask] = best_patch_material[patch_mask]
    # 碎石坡是冻融后坡壁下侧的沉积结果，不参与冰雪概率抽签。
    talus = talus_surface_mask(terrain, x, z, plans, sea_level)
    output[talus] = len(GLACIAL_SURFACE_PALETTE)
    if climate_cold_weight is not None:
        # 过渡带不再使用 ``cold_weight <= 0.05`` 的硬阈值。那会让相邻
        # 两行方块从“整行冰川材质”跳成“整行普通地表”。这里用 seeded
        # value-noise 配合可配置的 smoothstep 概率曲线，既保持 seed 可复现，
        # 也让冰雪团块随纬度逐渐稀疏而不牺牲过渡中部的覆盖率。
        if climate_cold_weight.shape != output.shape:
            raise ValueError("climate cold weight must have the same shape as materials")
        if not systems:
            return output
        fade_scale = min(system.climate_fade_scale for system in systems)
        fade_full_weight = min(system.climate_fade_full_weight for system in systems)
        fade_noise = (
            sample_noise(
                x,
                z,
                NoiseLayer(kind="value", scale=fade_scale, octaves=1, seed_offset=97_001),
                seed + 97_003,
            )
            + 1.0
        ) * 0.5
        # 直接使用 cold_weight 会让过渡尾部的冰材质只剩 5% 左右，视觉上
        # 像刷新率突然降低。这里在配置的尾部范围内使用 smoothstep，达到
        # full_weight 后恢复完整覆盖，同时保持 cold_weight=0 完全清除。
        fade_t = np.clip(climate_cold_weight / fade_full_weight, 0.0, 1.0)
        keep_probability = fade_t * fade_t * (3.0 - 2.0 * fade_t)
        keep = fade_noise <= keep_probability
        output = np.where(keep, output, 0).astype(np.uint8, copy=False)
    return output


def glacial_surface_layers(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    systems: tuple[GlacialSystem, ...],
    seed: int,
    sea_level: float,
    surface_material_id: np.ndarray,
    climate_cold_weight: np.ndarray | None = None,
    mountain_material_id: np.ndarray | None = None,
    wind_snow_field: WindSnowField | None = None,
    plans: tuple[GlacialFlowPlan, ...] | None = None,
) -> np.ndarray:
    """生成寒带地表的垂直覆盖层数量，不改写原始高度场。

    寒带权重从低到高逐步解锁覆盖层：过渡带只有松雪，寒带内部再增加
    雪块、冰和蓝冰。传入 ``mountain_material_id`` 时，覆盖范围先由群山
    材质场决定，气候权重只负责层级强度，避免世界坐标气候带直接形成直线。
    高程会对冰层略作加权，但不会成为硬边界。风雪场按地形法线与主风向的
    点积调整覆盖强度：迎风坡会概率性增加一层松雪，背风坡会概率性削薄
    覆盖。返回数组的轴顺序固定为
    ``powder_snow / snow_block / ice / blue_ice``，每个值表示该材质要在
    原地形顶部叠加多少个方块。
    """

    if terrain.shape != x.shape or z.shape != x.shape:
        raise ValueError("terrain and coordinates must share a shape")
    if climate_cold_weight is not None and climate_cold_weight.shape != terrain.shape:
        raise ValueError("climate cold weight must have the same shape as terrain")
    if surface_material_id.shape != terrain.shape:
        raise ValueError("surface material ids must have the same shape as terrain")
    if mountain_material_id is not None and mountain_material_id.shape != terrain.shape:
        raise ValueError("mountain material ids must have the same shape as terrain")
    if wind_snow_field is None:
        wind_snow_field = sample_wind_snow_field(
            terrain,
            x,
            z,
            systems,
        )
    else:
        for name in ("normal_alignment", "accumulation", "erosion", "response"):
            values = getattr(wind_snow_field, name)
            if values.shape != terrain.shape:
                raise ValueError(f"wind snow field {name} must have the same shape as terrain")
            if not np.isfinite(values).all():
                raise ValueError(f"wind snow field {name} must be finite")

    output = np.zeros((4, *terrain.shape), dtype=np.uint8)
    if not systems:
        return output
    if climate_cold_weight is None:
        cold_weight = np.ones(terrain.shape, dtype=np.float64)
    else:
        cold_weight = np.clip(climate_cold_weight, 0.0, 1.0).astype(np.float64)

    if mountain_material_id is None:
        eligible = (
            (surface_material_id > 0)
            & (surface_material_id != len(GLACIAL_SURFACE_PALETTE))
            & (terrain >= sea_level)
        )
    else:
        # 最终材质场存在时，覆盖层必须消费它，而不能再次按 world_z 的
        # cold_weight 生成一条横向雪线。否则材质场已经打散的边界会在
        # Anvil/BlueMap 导出前被覆盖层恢复成直线。
        # 岩石 ID（稳定 palette 的第 7 项及以后）是裸露基底，不能再被
        # 雪冰覆盖；否则岩石暴露场只会在查询层可见，实际渲染仍会被雪盖住。
        eligible = np.isin(mountain_material_id, (1, 2, 3, 4, 5)) & (
            terrain >= sea_level
        )
    # 未提供群山材质场时，旧的独立冰川调用仍允许寒带高地生成覆盖；
    # 正常流水线会传入 mountain_material_id，因此不会走这个全局兜底。
    coverage_noise = (
        sample_noise(
            x,
            z,
            NoiseLayer(kind="value", scale=260.0, octaves=1, seed_offset=9_701),
            seed + 97_003,
        )
        + 1.0
    ) * 0.5
    if mountain_material_id is None:
        eligible |= (
            (cold_weight > 0.0)
            & (terrain >= sea_level)
            & (coverage_noise <= cold_weight)
        )
        eligible &= cold_weight > 0.0
        cover_cold_weight = cold_weight
    else:
        # 材质场已经在 altitude/slope/exposure/noise 上完成了群山判定。
        # 把材质等级映射为最低覆盖强度，避免 cold_weight 在过渡带末端
        # 归零后把整片山坡的覆盖层突然截断。气候仍可提高层数，但不再
        # 覆盖材质场的正面判断。
        material_strength = np.select(
            (
                mountain_material_id == 2,  # powder_snow
                mountain_material_id == 1,  # snow_block
                mountain_material_id == 3,  # ice
                mountain_material_id == 4,  # packed_ice
                mountain_material_id == 5,  # blue_ice
            ),
            (0.18, 0.40, 0.60, 0.80, 1.0),
            default=0.0,
        )
        cover_cold_weight = np.maximum(cold_weight, material_strength)

    if not np.any(eligible):
        return output

    # 正常流水线传入完整规划结果，冰层只在真实冰川谷内解锁。保留 None
    # 的直接调用行为，方便只测试覆盖层阈值而不必构造拓扑计划。
    valley_mask = np.ones(terrain.shape, dtype=bool)
    valley_pressure = np.ones(terrain.shape, dtype=np.float64)
    if plans is not None:
        valley_field = sample_glacier_valley_field(x, z, plans)
        valley_mask = valley_field.mask
        valley_pressure = valley_field.pressure

    # 第一层始终是松雪；更深材质按寒带权重和高程连续增加。
    powder_noise = sample_noise(
        x,
        z,
        NoiseLayer(kind="value", scale=90.0, octaves=1, seed_offset=9_703),
        seed + 97_007,
    )
    powder_uniform = (powder_noise + 1.0) * 0.5
    # accumulation 为正、erosion 为负的响应已经带有 system 域权重；这里
    # 只在 eligible 地表消费，避免把风雪直接传播到山脉支持域外。
    wind_response = np.asarray(wind_snow_field.response, dtype=np.float64)
    wind_accumulation = np.asarray(wind_snow_field.accumulation, dtype=np.float64)
    wind_erosion = np.asarray(wind_snow_field.erosion, dtype=np.float64)
    for system in systems:
        # 每个系统按自己的配置写层数；多个系统叠加时取更厚的一层。
        # 使用连续高程因子，而不是按某一条等高线切断冰层。
        elevation_factor = np.clip(
            (np.asarray(terrain, dtype=np.float64) - sea_level - system.cover_elevation_start)
            / max(system.cover_elevation_range, 1.0e-6),
            0.0,
            1.0,
        )
        effective_cold = cover_cold_weight * (0.68 + 0.32 * elevation_factor)
        # 风向只改变局部积雪厚度，不改变寒带/温带的地理范围；因此响应
        # 叠加在连续的覆盖强度上，而不是新增一条世界坐标边界。
        effective_cold = np.clip(effective_cold + wind_response, 0.0, 1.0)
        # 风蚀压力是连续场，但可见方块需要更明显的概率差异；平方根把
        # 中等风蚀从“几乎不可见”提升为稳定的薄雪斑块，同时保留配方可调的
        # 保留比例，避免背风面被一次性削成完全裸岩。
        retention = np.clip(
            1.0
            - system.wind_snow_erosion_retention_scale
            * np.sqrt(np.clip(wind_erosion, 0.0, 1.0)),
            0.0,
            1.0,
        )
        retained = eligible & (powder_uniform <= retention)
        snow_gate = retained & (effective_cold >= system.cover_snow_weight)
        ice_gate = retained & (effective_cold >= system.cover_ice_weight) & valley_mask
        blue_gate = retained & (effective_cold >= system.cover_blue_ice_weight) & valley_mask

        base_powder_depth = np.uint8(min(system.cover_powder_depth, 255))
        drift_extra = np.where(
            eligible & (wind_accumulation > powder_uniform),
            np.uint8(min(system.wind_snow_max_extra_layers, 255)),
            0,
        )
        powder_depth = np.minimum(
            base_powder_depth.astype(np.uint16) + drift_extra.astype(np.uint16),
            255,
        ).astype(np.uint8)
        output[0] = np.maximum(output[0], np.where(retained, powder_depth, 0))
        if system.cover_powder_depth > 1:
            output[0] = np.maximum(
                output[0],
                np.where(
                    retained & (powder_noise > 0.15),
                    np.uint8(min(system.cover_powder_depth, 255)),
                    0,
                ),
            )

        output[1] = np.maximum(
            output[1],
            np.where(snow_gate, np.uint8(min(system.cover_snow_depth, 255)), 0),
        )
        output[2] = np.maximum(
            output[2],
            np.where(ice_gate, np.uint8(min(system.cover_ice_depth, 255)), 0),
        )

        if plans is None:
            # 只测试覆盖阈值的旧式直接调用没有流路上下文，保留其原有的
            # 四层契约；正常流水线总是传入 plans，走下方稀疏 veins 分支。
            deep_ice = np.isin(surface_material_id, (4, 5)) | (elevation_factor >= 0.40)
        else:
            # 蓝冰是高压老冰的 veins，不是完整的一层：只有冰川谷中心、足够
            # 厚的覆盖和低频噪声同时命中才写入蓝冰轴。概率上限 62% 仍保留
            # 能看到普通冰主体。低频采样方式参考 FastNoiseLite：
            # https://github.com/Auburn/FastNoiseLite
            deep_ice = valley_mask & (
                (valley_pressure >= 0.35)
                | np.isin(surface_material_id, (4, 5))
                | (elevation_factor >= 0.40)
            )
            blue_noise = (
                sample_noise(
                    x,
                    z,
                    NoiseLayer(
                        kind="fbm",
                        scale=max(system.valley_floor_width * 8.0, 36.0),
                        octaves=2,
                        gain=0.55,
                        seed_offset=stable_text_seed(system.name) % 997 + 9_811,
                    ),
                    seed + 9_817,
                )
                + 1.0
            ) * 0.5
            blue_probability = np.clip(
                system.crevasse_blue_ice_probability * 0.55
                + 0.52 * valley_pressure
                + 0.16 * elevation_factor,
                0.0,
                0.62,
            )
            blue_gate &= deep_ice & (blue_noise <= blue_probability)
        output[3] = np.maximum(
            output[3],
            np.where(
                blue_gate,
                np.uint8(min(system.cover_blue_ice_depth, 255)),
                0,
            ),
        )

    return output


__all__ = [
    "GLACIAL_COVER_PALETTE",
    "GLACIAL_LANDFORM_PALETTE",
    "GLACIAL_SURFACE_PALETTE",
    "apply_glaciers",
    "apply_surface_protrusions",
    "glacial_surface_materials",
    "glacial_surface_layers",
    "glacial_landform_ids",
]
