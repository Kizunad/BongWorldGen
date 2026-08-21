# BongWorldGen

Bong 的独立程序化地形生成器。项目只保留一条清晰的数据流：

    Python 地貌配方
            ↓
    NumPy procedural engine
            ↓
    统一 Heightfield
            ↓
    Bong raster 转换层
            ↓
    height / surface_id / water_level / biome_id / feature_mask

fork/ 下的仓库只用于算法对照，不参与运行时导入，也不会提交到本仓库。当前参考
仓库为 Kizunad/Procedural-Maps，其上游没有声明许可证，因此核心实现采用独立代码，
避免把无许可证代码直接纳入发布产物。

## 数据布局

src/bong_worldgen/data/ 是类型化输入数据，不是生成器实现：

- world.py：PoiDefinition、ZoneDefinition、WorldDefinition
- world_metadata.py：世界名、边界、spawn 和说明
- world_definition.py：聚合入口 WORLD
- zones/*.py：每个 zone 一个文件，包含自己的 POI
- imported_world.py：旧入口兼容转发，不再存储数据

修改单个区域时只改对应的 zones/<zone>.py；修改世界范围时改
world_metadata.py。重新从 Bong JSON 同步时运行导入工具即可重新生成这些 Python
数据文件。

## 开发

    cd ~/Code/BongWorldGen
    python3 -m venv .venv
    .venv/bin/pip install -e '.[dev]'
    .venv/bin/pytest -q

生成一个小型高度场预览：

    .venv/bin/bong-worldgen --width 256 --height 256 --seed 812731 --output generated/demo.npz

如果需要给现有 Rust loader 做本地 smoke test，可以调用转换层写出单 tile v2 raster：

    from pathlib import Path
    from bong_worldgen.adapters import to_bong_tile, write_bong_raster
    from bong_worldgen.data.recipes import DEFAULT_RECIPE
    from bong_worldgen.engine import generate_heightfield

    field = generate_heightfield(DEFAULT_RECIPE, width=512, height=512, seed=812731)
    write_bong_raster(
        to_bong_tile(field, sea_level=DEFAULT_RECIPE.sea_level),
        Path('generated/raster'),
    )

从 Bong 旧版 JSON 导入 Python 数据模块：

    .venv/bin/python tools/import_legacy_world.py \
      ../Bong/server/zones.worldview.example.json \
      src/bong_worldgen/data/imported_world.py

导入工具只负责格式转换，不参与生成逻辑。后续可以逐步把旧 JSON 的字段迁移成类型化
的 Python 数据对象。
