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
    height / surface_id / water_level / biome_id / feature_mask / wilderness_id / riverbed_id / zone_id

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

## Zone 地形生成

`src/bong_worldgen/composition/` 是位于通用 engine 之上的区域合成层。它读取
`data/zones/*.py` 的中心、完整尺寸、形状、边界模式和 `terrain_profile`，在世界坐标
中得到每列的 dominant zone 与权重，再混合普通高度场。`engine/` 只接受纯
`TerrainRecipe` 和坐标采样，不导入 zone、POI 或 raster 类型。

当前 27 个 zone 覆盖 15 种 profile：平原、残峰、高原、湿地、裂谷、灰烬地、渊口、
劫坑、战场、宗门遗址、药园、王印台、洞穴、深渊和浮岛。边界用 `soft`、`semi_hard`
或 `hard` 的连续权重带混合；非零 `boundary_width` 的过渡端点没有高度断崖。
洞穴和浮岛在混合地表之后生成，因此 `spans.bin` 仍然是实际可行走的垂直实体范围。

`export_preview_world()` 额外写出 `zone_id.bin` 和 manifest 的 `zone_palette`：每列
使用 dominant zone 的稳定 ID，`255` 表示背景；`boundary_weight.bin` 保留该列的
混合强度。下游可以同时读取高度、zone 归属和边界权重，不需要从图像颜色反推区域。

生成区域画廊和验收报告：

```bash
.venv/bin/python tools/bluemap.py render --accept-minecraft-eula --zone-gallery --width 256 --height 256
.venv/bin/python tools/zone_evidence.py
MPLCONFIGDIR="$PWD/generated/.mplcache" python3 tools/plot_zone_evidence.py generated/zone-evidence
```

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

## 荒野类型数据

### 河床材质

河流的河床材质是配方数据，不写死在生成器里。直接在 `River` 上配置候选集合：

```python
River(
    path=(Point(0, 0), Point(800, 600)),
    width=18.0,
    depth=6.0,
    widening=2.5,
    bed_materials=("mud", "gravel", "sand", "clay"),
)
```

同一 seed 会用低频噪声在河道内稳定地混合这些材质。`Heightfield` 保存
`riverbed_id` 和 `riverbed_palette`；raster 转换层另外写出 `riverbed_id.bin`，BlueMap
转换层则把这些名字映射为 Minecraft 方块。支持 `dirt`、`mud`、`gravel`、`sand`、
`clay`、`coarse_dirt`、`packed_mud` 和 `mud_bricks`。未知材质会在 Anvil 导出时明确报错。

### 地下洞穴

`TerrainRecipe.caves` 使用 Godot Voxel 风格的确定性洞穴图、2D worm 场、路径 SDF 和
3D seeded noise 组合生成浅层洞穴：先对 2D 噪声平方并按阈值截取 worm，再用 `y²-1`
抛物线调制出圆形截面；低频调制让通道自然收缩成死胡同，垂直扰动让洞穴上下起伏，
FastNoiseLite 风格的 XZ domain warp 改变通道走向，3D 噪声再改变洞壁细节，屋顶厚度参数
阻止洞穴穿出地表。最终通过 `SDF smooth union` 合并支洞和洞室，再由 `solid_spans` 完成
等价于 Godot `terrain - caves` 的体素空腔扣除。
`solid_spans` 会把每列的地下实心段写入控制台 `spans.bin` 和 Minecraft Anvil，因而
控制台可以直接看到洞顶、洞底和连通空腔，而不是只看地表高度。

洞穴实现按职责拆分在 `src/bong_worldgen/engine/caves/`：

- `generator.py`：编排洞穴网络、可选地下结构、`cave_id` 和最终地下结果；
- `spans.py`：把三维空腔折叠为服务端消费的 top-first `solid_spans`；
- `structures.py`：预留未来天然地下特征的稀疏方块扩展边界；当前不生成人工建筑；
- `worms.py`：实现 Godot Voxel 文档中的 2D worm、Y 轴扰动和死胡同调制；
- `topology.py`：根据 seed 和配方锚点生成确定性的支洞边与洞室节点；
- `__init__.py`：只暴露洞穴子系统的公共入口。

`engine/pipeline.py` 只调用 `generate_underground()`，不再持有洞穴几何或地下建筑细节。
这样修改洞穴噪声、垂直范围或后续地下结构时，不会把地表河流和基础高度场一起改动。

服务端不需要从几何反推矿洞：每个 tile 额外写出 `cave_id.bin`（`0` 表示无矿洞，
`1..N` 对应 manifest 的 `cave_palette`），`spans.bin` 则提供该列实际的地下垂直范围。
Rust loader 读取这两个层即可判断某个 chunk/列是否属于矿洞，以及可进入的 Y 区间。

洞穴噪声接口和 domain-warp 组织方式参考 FastNoiseLite 的公开 API 设计；本项目使用
独立 NumPy 实现，没有复制其 C++ 源码，也不增加 C++ 运行时依赖：

<https://github.com/Auburn/FastNoiseLite>

默认配方会为洞穴图生成确定性的地表入口。入口是独立阶段，只放宽入口柱的屋顶限制，
不会让普通洞道随机穿出地表。当前洞室直接作为自然洞穴的一部分生成，不输出人工矿厅或
固定建筑方块；未来天然拱门、沉积物等地下特征可以复用稀疏方块输出，不需要修改
Three.js 解码器。

每个 raster tile 现在额外输出 `wilderness_id.bin`，每个地表柱一个 `uint8` id。
类型表写在 manifest 的 `wilderness_palette` 中，当前 id 是稳定契约：

- `0 grassland`：草地，可放草、灌木和树
- `1 mountains`：群山，可放石头、松树和矿物
- `2 lake`：湖泊，可放芦苇、水生植物和岸线内容
- `3 river`：河流，可放河岸内容和桥

类型表同时携带 `decoration_tags`，后续 `decorations.rs` 应按 id/tag 读取，
不要依赖前端颜色或中文显示名。

完整 raster 仍然是权威数据；manifest 另外提供 `overview`，指向一份仅用于控制台
快速定位的低分辨率高度/地表材质/荒野/区域索引。采样严格对应 manifest 声明的
origin 与 32 方块步长，支持未对齐的世界边界和小于步长的 tile。控制台先显示 overview，再在后台加载完整 tiles，
不会用 overview 替代服务器或 `decorations.rs` 使用的详细数据。

## 最小 Three.js 控制台

控制台已经迁移到本项目的 `console/`，不再依赖 Bong 原仓库的 FastAPI
`console_server`、zone 参数编辑或 regen 接口。它只读取：

```text
/world/manifest.json
/world/tile_<x>_<z>/*.bin
```

默认使用地表材质视图，并提供区域归属、水体、POI 和荒野类型高亮。区域归属开关
同时显示 overview 和详细 tile 的 `zone_id`，区域按名称配色，背景保留地表色；
overview 仅表示每 32 方块采样的粗略范围。点击荒野图例中的某一类，
只会在真实地表上叠加对应 `wilderness_id` 的半透明高亮。启动：

```bash
cd console
npm install
npm run dev
```

默认读取 `../generated/console-world/rasters`；可以用 `BONG_WORLD_DIR` 指向另一份
已生成 raster。控制台按 manifest 的 `zone_id` 层加载实际 dominant zone 归属，
不把 zone 名称或高度颜色当作生成结果的替代品。

## BlueMap 地图

BlueMap 是当前的完整地图查看器。转换层直接把 `Heightfield` 写成 Minecraft 1.20.1
Anvil world，地表使用草地、岩石、积雪、砂砾和真实水方块；BlueMap 再负责方块模型、
多级细节、俯视、透视和自由飞行。

首次渲染会下载固定版本 BlueMap 5.23，并校验 SHA-256。BlueMap 还需要下载 Mojang 的
1.20.1 客户端资源，所以必须显式确认持有 Minecraft Java Edition 并接受 EULA：

```bash
.venv/bin/python tools/bluemap.py render --accept-minecraft-eula
.venv/bin/python tools/bluemap.py serve
```

访问 `http://127.0.0.1:8100/`。每次 `render` 都会先替换
`generated/bluemap-world` 和 `.bluemap/web`，不会保留按版本命名的整份历史世界；
BlueMap 的客户端资源缓存保留在 `.bluemap/data`，避免每次重复下载。
