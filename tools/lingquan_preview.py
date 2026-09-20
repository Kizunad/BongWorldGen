#!/usr/bin/env python3
"""分 Anvil region 导出完整灵泉湿地，保留 1 格精度并生成 BlueMap。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bong_worldgen.adapters import export_minecraft_world
from bong_worldgen.adapters.anvil_nbt import encode_level_dat
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config
from bong_worldgen.data.recipes import DEFAULT_RECIPE
from bong_worldgen.data.zone_recipes import LINGQUAN_TERRAIN as ZONE
from bong_worldgen.engine import TerrainRecipe, generate_heightfield
from bluemap import ensure_bluemap, java_path
from river_gallery import _terrain_image, _write_png


def terrain_recipe() -> TerrainRecipe:
    """恢复全球自然 wilderness 基底，同时保持建筑层关闭。

    通过 ``replace`` 继承默认配方的盆地、山脉、水文、冰川和地下内容，
    避免预览脚本维护一份容易漏项的“半配方”。建筑和森林模板仍关闭；
    它们需要跨 Anvil tile 的归属裁剪，不能在当前单 tile 导出循环中直接开启。
    """

    return replace(
        DEFAULT_RECIPE,
        name="lingquan_marsh_wilderness_terrain_preview",
        terrain_zones=(ZONE,),
        town=None,
        standalone_structures=None,
        wilderness_tribes=None,
        wilderness_sites=(),
        wilderness_loot=None,
        wilderness_exclusions=(),
        spawn_pois=None,
        forest=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=812731)
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = ROOT / ".bluemap" / output.name
    world = output / "world"
    # 每次只导出一个完整 .mca，避免多次导出同一 region 覆盖此前的区块。
    tile_size = 512
    min_x = math.floor((ZONE.center.x - ZONE.radius_x) / tile_size) * tile_size
    min_z = math.floor((ZONE.center.z - ZONE.radius_z) / tile_size) * tile_size
    max_x = math.ceil((ZONE.center.x + ZONE.radius_x) / tile_size) * tile_size
    max_z = math.ceil((ZONE.center.z + ZONE.radius_z) / tile_size) * tile_size
    width, height = max_x - min_x, max_z - min_z
    if not args.render_only:
        if world.exists():
            raise SystemExit("已有预览世界保留；新一版请指定新输出目录，重渲染使用 --render-only。")
        output.mkdir(parents=True, exist_ok=True)
        recipe = terrain_recipe()
        started = time.monotonic()
        overview = generate_heightfield(
            recipe, width=width // 4, height=height // 4, seed=args.seed,
            origin_x=min_x, origin_z=min_z, cell_size=4,
        )
        overview_rgb = _terrain_image(overview)
        _write_png(output / "overview.png", overview_rgb)
        # 500 格网格与 BlueMap 当前 lowres.tileSize 对齐，便于比较实际区域尺度。
        grid_rgb = overview_rgb.copy()
        for coord in range(math.ceil(min_x / 500) * 500, max_x, 500):
            column = (coord - min_x) // 4
            grid_rgb[:, column] = (grid_rgb[:, column].astype(float) * .4 + 255 * .6).astype(np.uint8)
        for coord in range(math.ceil(min_z / 500) * 500, max_z, 500):
            row = (coord - min_z) // 4
            grid_rgb[row] = (grid_rgb[row].astype(float) * .4 + 255 * .6).astype(np.uint8)
        _write_png(output / "overview-grid.png", grid_rgb)
        del overview, overview_rgb, grid_rgb
        core_count, water_count, chunks_written = 0, 0, 0
        depths, materials = Counter(), Counter()
        core_min, core_max = math.inf, -math.inf
        total = width * height // tile_size**2
        for index, (z0, x0) in enumerate(
            ((z0, x0) for z0 in range(min_z, max_z, tile_size)
             for x0 in range(min_x, max_x, tile_size)), start=1,
        ):
            field = generate_heightfield(recipe, width=tile_size, height=tile_size, seed=args.seed,
                                         origin_x=x0, origin_z=z0)
            x, z = np.meshgrid(x0 + np.arange(tile_size), z0 + np.arange(tile_size))
            core = ZONE.weight(x, z) == 1
            wet = (field.water_level >= 0) & core
            core_count += int(core.sum())
            water_count += int(wet.sum())
            if core.any():
                core_min = min(core_min, float(field.height[core].min()))
                core_max = max(core_max, float(field.height[core].max()))
                values, counts = np.unique(
                    field.water_level[wet] - np.rint(field.height[wet]), return_counts=True,
                )
                depths.update({str(int(v)): int(c) for v, c in zip(values, counts)})
                values, counts = np.unique(field.surface_visible_id[core], return_counts=True)
                materials.update({field.surface_visible_palette[int(v) - 1]: int(c)
                                  for v, c in zip(values, counts)})
            assert not field.settlement_blocks and not field.settlement_interest_points
            result = export_minecraft_world(
                field, world, origin_x=x0, origin_z=z0, sea_level=recipe.sea_level,
                seed=args.seed, world_name="灵泉湿地 · 扩域地貌预览",
            )
            assert result.regions_written == 1 and result.chunks_written == 1024
            chunks_written += result.chunks_written
            print(f"导出 {index}/{total}：({x0}, {z0})，累计 {time.monotonic() - started:.0f} 秒", flush=True)
            del field
        # 最后一次 region 导出的默认出生点位于边角，统一改为主泉旁的固定预览点。
        (world / "level.dat").write_bytes(encode_level_dat(
            "灵泉湿地 · 扩域地貌预览", args.seed,
            int(ZONE.center.x), int(ZONE.wetland.water_level) + 1, int(ZONE.center.z),
        ))
        stats = {
            "seed": args.seed, "origin": [min_x, min_z], "size": [width, height],
            "zone_size": [ZONE.radius_x * 2, ZONE.radius_z * 2],
            "core_water_fraction": water_count / core_count,
            "core_height_range": [core_min, core_max],
            "core_blocks": core_count, "water_depth_counts": dict(sorted(depths.items())),
            "surface_materials": dict(materials), "regions": total, "chunks": chunks_written,
            "settlement_blocks": 0, "settlement_interest_points": 0,
            "generation_export_seconds": time.monotonic() - started,
            "bluemap_lowres_tile_size": 500,
            "core_contains_centered_4_by_4_lowres_tiles": (
                min(ZONE.radius_x, ZONE.radius_z) - ZONE.boundary_width - ZONE.wetland.rim_width
                >= math.hypot(1000, 1000)
            ),
        }
        assert stats["core_contains_centered_4_by_4_lowres_tiles"]
        assert "0" not in depths and .20 < stats["core_water_fraction"] < .70
        (output / "verification.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    elif not (output / "verification.json").exists():
        raise SystemExit("未找到完整导出的核验记录。")
    (runtime / "data").mkdir(parents=True, exist_ok=True)
    cache = runtime / "data/minecraft-client-1.20.1.jar"
    if not cache.exists():
        cache.symlink_to(ROOT / ".bluemap/data/minecraft-client-1.20.1.jar")
    write_bluemap_config(BlueMapConfig(
        config_dir=runtime / "config", data_dir=runtime / "data", web_dir=runtime / "web",
        world_dir=world, min_x=min_x, max_x=max_x - 1, min_z=min_z, max_z=max_z - 1,
        start_x=int(ZONE.center.x), start_z=int(ZONE.center.z), start_y=90,
        start_distance=width * 1.1, port=args.port,
    ))
    core_config = runtime / "config/core.conf"
    core_config.write_text(core_config.read_text().replace("render-thread-count: 4", "render-thread-count: 8"))
    command = [str(java_path()), "-Xmx4G", "-jar", str(ensure_bluemap()),
               "--config", str(runtime / "config"), "--mc-version", "1.20.1"]
    # 所有路径由 shlex 引号保护；用户可以直接 bash 运行，无需修改执行权限。
    import shlex
    (output / "start.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " +
                                     shlex.join([*command, "--webserver"]) + "\n")
    print("开始渲染扩大后的完整湿地", flush=True)
    with (output / "render.log").open("w") as log:
        subprocess.run([*command, "--generate-webapp", "--render", "--generate-websettings"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("扩域地貌导出与 BlueMap 渲染完成", flush=True)


if __name__ == "__main__":
    main()
