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
    height / uplift / surface_id / water_level / biome_id / feature_mask / wilderness_id / riverbed_id
    glacial_landform_id / glacial_water_id / glacial_discharge
    valley_depth / valley_flow_accumulation / valley_stream_power
    hydraulic_erosion / hydraulic_deposition
    snow_accumulation / glacial_mass_balance
    surface_material_id / mountain_material_id / surface_visible_id / permafrost_id
    mountain_weight / mountain_snowline / mountain_material_score
    mountain_slope_angle / mountain_exposure
    climate_id / climate_transition_id / climate_transition_weight / climate_surface_id
natural relief is evaluated before glacier, river and underground stages
hydraulic erosion is the final continuous-height refinement before surface water and underground voxelization

完整流程图见 [docs/worldgen-pipeline.svg](docs/worldgen-pipeline.svg)。

## 全球气候框架

`engine/climate.py` 声明总气候类型，并在生成时按显式注入的世界南北边界采样。
默认 `GlobalClimatePlan` 沿 `world_z` 从北到南排列为：

1. 寒带 `cold`：未来地表语义为雪地；
2. 温带 `temperate`：气温舒适，未来地表语义为温带草地；
3. 热带 `tropical`：未来地表语义为沙漠，不使用热带草地。

寒带→温带、温带→热带各自有独立的 `ClimateTransition` 过渡宽度配置。
`TerrainRecipe.climate_world_bounds` 必须由数据层提供 `ClimateWorldBounds`；默认配方
从 `data/world_metadata.py` 的 `WORLD_BOUNDS` 注入，不在引擎内重复地图范围常量。

生成结果包含：

- `climate_id`：硬分类（`0=none`，`1=cold`，`2=temperate`，`3=tropical`），供
  server/decorations 查询；
- `climate_transition_id` 与 `climate_transition_weight`：相邻气候带的过渡信息；
- `climate_surface_id`：地表家族（`0=none`，其余按 palette 为雪地、温带草地、沙漠）。

冰川 carve 使用连续的寒带权重，寒带高地会补充雪地覆盖，热带陆地在转换层映射为
真实的 `minecraft:sand`。未提供世界边界的自定义配方不会假设当前 tile 是整个世界，
因此不会生成伪造的全球气候层。

fork/ 下的仓库只用于算法对照，不参与运行时导入，也不会提交到本仓库。当前参考
仓库为 Kizunad/Procedural-Maps，其上游没有声明许可证，因此核心实现采用独立代码，
避免把无许可证代码直接纳入发布产物。

## 连续崎岖地形

`TerrainRecipe.natural_relief` 是可选的全局连续崎岖场。它不把地图预先分成平原、
山地或悬崖区，而是用同一个 seed 叠加低频宏观起伏、域扭曲后的 ridged 场、中尺度
细节和随机陡壁。地貌类型只在生成完成后由 `wilderness.py` 识别，因此不会出现固定
数量的地貌模板拼接。

生成阶段一直保留 `float64` 连续高度；只有 Anvil/raster 转换层才把高度量化为方块
坐标。陡壁不设置全局坡度上限，局部可以自然产生多格落差；这仍然是高度场表达，
真正的悬挑、倒挂和天然拱门继续通过 `solid_spans` 的三维实心区间表达。噪声和域
扭曲的组织方式参考 FastNoiseLite 的公开 API：
<https://github.com/Auburn/FastNoiseLite>。

`MountainRange` 支持两种明确语义：仅设置 `height` 时是相对抬升，适合局部山脊；
同时设置 `base_elevation` 与 `summit_elevation` 时是绝对高度剖面，适合需要明确
山脚到峰顶跨度的大型山脉。默认主山脉使用绝对高度模式，当前 seed 的 BlueMap
窗口实测方块地表为 `Y=123..424`，跨度 `301` 格。

`TerrainRecipe.valleys` 消费山脉阶段产生的 `uplift` 场，而不是另行猜测山峰高度。
每个 `ValleySystem` 在固定世界域上做有限轮次的 `Uplift → Flow Accumulation →
Stream Power → terrain` 反馈：抬升后的地形决定 D8 流向，抬升强度按
`1 + uplift_strength * uplift / uplift_reference` 增强侵蚀势能，下切后的地形再参与下一
轮汇流。累计谷深仍受 `maximum_depth` 限制，避免迭代次数改变配方尺度上限。汇水量同时
决定谷槽宽度，规划结果按世界坐标采样回 tile，因此 Server 分块生成与整图导出使用同一
套谷网，不会在 tile 边缘临时改道。实现位于 `engine/mountains/` 与 `engine/valleys/`，
参考 pyflwdir、RichDEM 和 Landlab 的公开算法合同，运行时仍只依赖 NumPy。

生成结果中的 `Heightfield.uplift` 和 raster `uplift.bin` 是正向构造抬升物理层；
`valley_depth`、`valley_flow_accumulation`、`valley_stream_power` 则记录同一耦合过程的
水文结果，便于 Server 和后续 decorations 判断山峰、山脊与谷地关系。

## 水力侵蚀 refinement

`TerrainRecipe.hydraulic_erosion` 是宏观山脉、主谷和冰川完成后的末端 refinement。
每个 seeded 水滴从内部网格起步，按当前高度梯度更新方向、速度和水量：下坡时依据
携沙能力下切，上坡或携沙过量时沉积。侵蚀和沉积都按双线性权重写回四个邻格，因而
不会产生“整格刷掉”的规则条纹。它只负责小沟壑、支流、局部 ridge 和冲积扇，不能
替代 `ValleySystem` 的 Flow Accumulation + Stream Power 主地貌布局。

配置为 `None` 或 `enabled=False` 时是精确 no-op；默认配方使用中等强度的 900 个水滴。
连续结果保存在 `Heightfield.hydraulic_erosion` 与 `Heightfield.hydraulic_deposition`，
并由 raster 的 `hydraulic_erosion.bin`、`hydraulic_deposition.bin` 导出，供 Server
按区域判断 refinement 强度。实现和外部算法依据写在
`engine/hydraulic/erosion.py`，参考：

- <https://github.com/SebLague/Hydraulic-Erosion>
- <https://github.com/Auburn/FastNoiseLite>

配置示例：

```python
from bong_worldgen.engine import NaturalRelief, TerrainRecipe

TerrainRecipe(
    name="rugged_world",
    natural_relief=NaturalRelief(
        macro_amplitude=32.0,
        ridge_amplitude=48.0,
        detail_amplitude=11.0,
        cliff_amplitude=30.0,
    ),
)
```

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

### 城镇

`TerrainRecipe.town` 是独立的城镇层。它可以复用 `spawn_plain` 的世界级
锚点，也可由任意调用方传入锚点；布局算法不属于出生平原。城镇由高密度的
核心区与分簇、稀疏的外围部落构成：核心区生成 Spawn 地标、住宅和城墙，外围
部落只生成住宅与小径，不参与围墙边界或利用率计算。
默认配方从 `assets/structures/houses/*.schem` 读取 Sponge Schematic 房屋，按真实尺寸
选址并随机旋转；未配置 `house_schematic_directory` 的测试配方仍使用石砖矩形回退。
生成结果存放在 `Heightfield.settlement_blocks`，每条记录包含世界坐标、方块状态和
`spawn_core`、`road`、`house_schematic`、`settlement_wall` 等结构类型，以及
`core` / `outer` 分区，Anvil 适配器再按区块切分写入；石砖广场不再生成。
资源解析器位于 `engine/structures/schematic.py`，目标版本为 Minecraft 1.20.1，
对模板中的少量新版本方块执行显式等价替换。布局参考 Tome
（<https://github.com/Jandhi/Tome>）与 ProceduralCityGeneration
（<https://github.com/Grzybojad/ProceduralCityGeneration>），实现为本项目独立代码。

模板读取器同时接受标准 Sponge `.schem` 以及 WorldEdit 输出的 v3 包装形式；后者将
Sponge 文档放在根 `Schematic` compound 内，读取时只在资源边界展开，不改变下游
方块、旋转和 footprint 逻辑。树木、装饰和归档资源分别位于
`assets/structures/trees/`、`assets/structures/decorations/` 与
`assets/structures/excluded/`；只有已接入的住宅、树木和独立结构目录会被生成层扫描。

结构名称、类别和特殊放置规则集中记录在
[`docs/structures-catalog.md`](docs/structures-catalog.md)。其中 `31280` 石砖井按
Fence 层对齐地表，`31101` 沙漠井按枯木层对齐地表；两者井水都保留为真实的
`minecraft:water`，并以 `well_water` 类型交给 Server 查询。

每栋已放置建筑还会生成一条 `settlement_spawn_area`：记录结构 ID、类别、真实
三维 footprint、带 `npc_spawn_radius` 缓冲的刷新/防守 AABB，以及 `ground_y`。
建筑默认使用 `structure_ground_offset=1`：模板底层位于地表方块上方一格，避免
Anvil 的地表层语义让房屋整体下沉；井的 Fence/枯木锚点仍可保留地下井水。
完整世界清单和每个 raster tile 的 `settlement_spawn_areas.json` 都会提供这些记录，
Server 可直接按 `area_id` 或坐标筛选 NPC 刷新区域，无需反向解析建筑方块。

### 乡村道路与跨河桥梁

城镇道路使用低频 seed 势场、八方向 A* 和受约束的折线平滑。寻路会检查
整段线路及实际路宽，不会在失败后直接画直线穿水；平滑也不能切入水体或悬崖。
跨河时，独立的 `engine/bridges.py` 把可施工的双岸桥段作为高代价边交给 A*，
由路径成本决定绕行还是架桥。默认配方允许最长 24 格连续水面的桥，使用木板、
木栅栏和圆石边墩，两侧桥头逐格接坡，保留中央水道。首版桥梁沿 X/Z 方向直跨，
桥外陆路可以弯曲；不支持斜桥或长距离海上高架。

`TownSettings` 的 `bridge_max_span=0` 禁用桥梁；`bridge_clearance` 控制桥面
底面高于最高水位的净空，`bridge_approach_length`、`bridge_max_support_depth`
限制接坡与桥墩深度，`bridge_cost_factor` 控制相对绕行的成本。材质分别由
`bridge_deck_material` / `bridge_rail_material` / `bridge_support_material` 配置。
生成的真实方块分别标记为 `bridge_deck`、`bridge_rail`、`bridge_support`。

### 独立世界结构

`TerrainRecipe.standalone_structures` 是城镇之外的大型地标层。默认读取
`assets/structures/standalone/` 中的 `31382` 深板岩生存屋和 `31279` 蔓生生存城堡；
二者绝不参与城镇住宅、道路或围墙布局。刷新位置由世界 seed、坐标网格和模板名
共同确定，并排除实际城镇半径。只有完整 footprint 落在当前窗口且地面干燥、
坡度与高差合格时才写入真实 schematic 方块。它们同样以 `standalone` 类别写入
`settlement_spawn_areas.json`，Server 可按该类别设置独立的 NPC 刷新和防守策略。

`TerrainRecipe.town` 是独立的城镇层；出生平原只为默认配方选择其锚点。城镇拆分
为两个明确分区：核心区以 Weighted Eden 方式紧凑生长，且仅核心建筑被城墙包围；
外围部落按稳定扇区和簇中心散布在城墙外，不会影响墙体边界。
`core_wall_minimum_utilization` 默认要求核心建筑投影至少占围墙矩形的 80%，
`core_wall_margin=2` 只保留两格巡逻缓冲，因此不会再生成围墙内的大块空地。该约束
只在默认配方启用填补上限时生效；特殊剧情城镇可以将
`core_wall_infill_max_buildings` 设为 0。
随机补位没有命中时，布局器会沿现有建筑边界寻找可容纳模板的窄空隙；后备候选
仍检查完整占地、干地、坡度和城墙边界，只在既有核心范围内填充。

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

## 代码结构

运行时代码按“数据、连续场、编排、转换”分层：

```text
src/bong_worldgen/
├── data/                       Python 世界与地形配方
├── engine/
│   ├── terrain_config.py       地表地形输入契约
│   ├── underground_config.py   地下输入与稀疏记录契约
│   ├── world_config.py         完整 TerrainRecipe
│   ├── relief.py               多尺度随机崎岖场与陡壁
│   ├── generated_world.py      Heightfield 生成结果
│   ├── distribution.py         洞穴/地下河的世界级 seeded 分布
│   ├── pipeline.py             地表与地下阶段总编排
│   ├── valleys/                填洼、D8 汇流与 Stream Power 山谷
│   ├── canyons/                 峡谷拓扑、距离场、刻蚀
│   ├── caves/                   洞穴拓扑、密度场、spans、刷新点
│   ├── glaciers/               积雪平衡、冰川刻蚀、沉积、冻土与融水
│   ├── hydraulic/              末端 droplet 水力侵蚀与冲积沉积
│   └── underground_rivers/     独立地下河、裂隙和地下产物
├── adapters/                   Bong raster 与 Minecraft Anvil 转换
└── bluemap_config.py           BlueMap 配置和默认观察窗口
```

`engine/caves/generator.py` 只编排地下阶段。连续洞穴场集中在
`engine/caves/density.py`，全世界实例位置集中在 `engine/distribution.py`；两者均不
写文件。所有随机模块统一使用 `engine/randomness.py`，禁止使用跨进程不稳定的
Python `hash()`。

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

### 实心石层矿脉

`TerrainRecipe.solid_ores` 定义空腔之外的普通矿物。它们不是洞壁刷新点，而是
直接替换地下实心石段中的石头方块；每个 `SolidOreSpec.cluster_count` 表示一个
512×512 世界分区中的矿脉簇数量，矿脉沿确定性的相邻路径展开。

生成顺序保证普通矿脉不会落入普通洞穴、地下河或构造裂隙，也不会覆盖已经生成的
洞穴资源和地下水。输出仍使用 `UndergroundBlock`，但来源和逻辑 ID 独立：
`source="solid_ore"`、`resource_id="solid:coal_ore"`。因此 Server 可以把普通
煤矿、洞穴煤矿和地下河煤矿分别映射到不同的刷新规则，即使它们使用同一个
`minecraft:coal_ore` 视觉占位方块。

默认配方包含煤矿、铁矿和铜矿三类实心矿脉；可以在 Python 配方中调整矿脉数量、
深度范围、稀有度和簇长度：

```python
SolidOreSpec(
    material="coal_ore",
    rarity="多",
    cluster_count=28,
    min_depth=4.0,
    max_depth=48.0,
vein_length=7,
)
```

### 寒带冰川地貌

`TerrainRecipe.glaciers` 是独立的寒带地表阶段，当前已实现三部分：

- `cirque`：从正质量平衡区的积雪极大值选择源头，再在高程门槛以上 carve 椭圆冰斗；
- `terrain-guided flow`：在完整冰川域的规划网格上沿坡面下行，使用可配置的路径惯性、
  seed 扭曲和转向采样生成流路；不在边界反弹、不无故爬坡，靠近低处既有路径时让支流
  复用主干后缀。规划结果不依赖当前 tile；
- `U-shaped valley`：统一消费上述流路，谷宽按下游进度增长，中央保留宽平底，谷壁使用
  高次幂截面，区别于河流的 V 型切割；
- `moraine/drumlin`：在冰川末端生成终碛脊，在较低地形生成沿冰流方向拉长的鼓丘沉积。
- `snow accumulation / mass balance`：独立计算高程、寒带权重、降雪噪声、迎风坡、
  背风积雪、低海拔和向阳坡消融。输出 `snow_accumulation.bin` 与
  `glacial_mass_balance.bin` 两个 `float32` 连续场；质量平衡大于零表示净积累，
  小于零表示净消融。坡向计算使用一格 halo，整块和分 tile 生成不会在边缘产生接缝。
- `glacial landform semantic`：独立输出 `glacial_landform_id.bin`，稳定 palette 为
  `cirque / u_valley / terminal_moraine / drumlin_field`。ID 直接来自各算法的实际
  侵蚀/沉积 mask，不从最终高度反推，供 Server 和 decorations 查询。
- `surface material`：寒带地表额外输出独立材质层，按冰川几何和 seeded 噪声混合
  `snow_block`、`powder_snow`、`ice`、`packed_ice` 和深部的 `blue_ice`；冻融侵蚀在
  坡壁下侧另外生成 `gravel` 碎石坡团块。这层不改写
  历史 `surface_id`，而是通过 `surface_material_id.bin` 和 palette 交给 raster/Anvil
  适配器，因此 Server 也能直接判断寒带地表覆盖。
- `mountain material field`：群山内部的材质不再按固定海拔切成“雪/冰/蓝冰”三明治。
  `engine/mountains/material_field.py` 在最终地形上计算 Mountain Spine 距离权重、
  不规则雪线、坡度角、朝向暴露、Climate 寒冷度和冰川加成，再用世界坐标稳定的
  材质噪声选择五种冰雪材质。它输出 `mountain_material_id` 以及
  `mountain_weight`、`mountain_snowline`、`mountain_material_score`、
  `mountain_slope_angle`、`mountain_exposure`，山脉有限支持域外全部为零；这些层
  写入 raster manifest，供 Server/decorations 直接查询。
- `freeze-thaw/talus`：在冰斗和冰川谷暴露区按临界坡度迭代转移岩屑，保留总高度量，
  并把低侧沉积带标记为砂砾。实现参考 <https://github.com/TheJanusStream/symbios-ground>，
  不复制外部代码。
- `surface protrusions`：寒带材质层上的局部概率团块会额外抬高少量列一格，抬高列
  继续使用原本的冰、蓝冰、雪块或碎石材质；每列最多一格，避免把冰川变成尖刺地形。
- `surface cover layers`：寒带覆盖不再替换石头/草地基底，而是独立输出四组叠加层数
  `powder_snow / snow_block / ice / blue_ice`。寒带权重从低到高逐步解锁材质：过渡带
  只有松雪，寒带内部再增加雪块、冰和蓝冰；Anvil 导出时按蓝冰在底、松雪在顶的顺序
  写到原地形上方，`WORLD_SURFACE` 高度图也包含覆盖层顶部。raster 文件为
  `surface_cover_layers.bin`，轴顺序和基底保留规则写在 manifest 的
  `surface_cover_encoding` 中。
- `surface_visible_id`：生成阶段把基础地表、河床、寒带基底、冻土和覆盖层合成为
  一份稳定的最终可见材质索引。它只描述当前列最上方应该显示的方块，不替代
  `surface_id` 等语义层；水体列和洞穴入口列分别保持水体/空气语义。BlueMap、Anvil
  和 raster overview 都消费这一层，避免各适配器用不同规则重新推断地表材质。
- `glacial meltwater`：冰川路径把冰厚和融水率转换成沿程累计 `discharge`，再投影成连续
  水面；它不覆盖或降低已有湖泊/普通河流水位。`glacial_water_id.bin` 使用 `0=普通水体、
 1=glacial_meltwater`，`glacial_discharge.bin` 保存非负 `float32` 流量，二者都按每列
  输出并在 manifest 的 `glacial_water_encoding` 中声明，供 Server 与 decorations 直接查询。

融水阶段是独立的冰川水文层，不会把普通河流、地下河或冰裂隙混为同一个 ID。实现位于
`engine/glaciers/hydrology.py`，路径汇流结构参考：
<https://github.com/alexanderpino/skills/blob/main/terrain-architect/reference-impl/flow.py>、
<https://github.com/alexanderpino/skills/blob/main/terrain-architect/reference-impl/shallow_water.py>；
冰厚与融化分层参考 <https://github.com/oargudo/glaciers>。代码为独立 NumPy 实现。

代码按职责放在 `engine/glaciers/`：`topology.py` 负责积累峰选源、下坡流路和支流汇合，
`mass_balance.py` 负责积雪与消融场，`carving.py` 负责冰斗/U 型谷，`deposits.py` 负责冰碛和
鼓丘，`generator.py` 负责主冰川阶段编排，`hydrology.py` 负责融水阶段。
算法结构参考公开项目，但实现为独立 NumPy 代码：

- 冰川流动与地貌特征：<https://github.com/oargudo/glaciers>
- 热力侵蚀/碎石坡参考：<https://github.com/TheJanusStream/symbios-ground>

默认配方的 `north_cold_glaciers` 使用显式冰川生成域，并受全球寒带权重门控。BlueMap
的 Anvil 导出会把上述材质映射成真实的 Minecraft 方块；默认冰斗窗口已经实际包含五种
冰雪材质。全球气候层同时写入 raster tile 和 overview，供控制台和 server 后续读取。

### 大峡谷

`TerrainRecipe.canyons` 是独立的地表 carve 阶段。`Canyon.paths` 可以锁定一条
手写路线；留空时 `engine/canyons/topology.py` 从 seed 在 XZ 平面生成长距离随机游走，
不读取或筛选地表高度，所以任何高度的地表都可以成为峡谷起点。默认配方只生成一条
大峡谷，宽度约 180 格、深度约 72 格，底部保留平坦河谷，墙面用 7 级轻微台阶和噪声
扰动模拟分层岩壁。

`engine/canyons/distance.py` 先对查询坐标做双通道 domain warp，再计算到折线的连续
距离和弧长进度（warped polyline distance）；`generator.py` 使用距离场的 smoothstep
横截面，核心墙体采用 carve-only 的 `minimum` 合并，不会把原地形抬高。相关算法思路
参考以下 GitHub 文档，代码为本项目独立 NumPy 实现：

- 距离场与折线距离：<https://github.com/Tomen/colonies/blob/0bb686f3e5b5c25cc6a9325b116cfd088e946a20/docs/world/01_physical_layer/distance-fields.md>
- 弧长采样与 carve channel：<https://github.com/alexanderpino/skills/blob/09f8696a3d2040d372bdd282fedddc9c895e8929/terrain-architect/reference-impl/meander.py>
- domain warp 噪声接口：<https://github.com/Auburn/FastNoiseLite>
- 大峡谷分层墙体设计：<https://github.com/Sharks820/veilbreakers-gamedev-toolkit/blob/1516c802c487c74a0f4e4aec42d8e42930d75e24/.planning/research/MOUNTAIN_PASS_CANYON_DESIGN.md>

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
- `structures.py`：在洞壁和洞底生成真实 `minecraft:*` 方块刷新点；当前不生成人工矿厅；
- `engine/underground_rivers/`：独立地下河网；它不读取 `CaveNetwork`，自己生成 3D
  河道空腔、主流/支流和汇水湖盆，再与普通洞穴在 spans 层做几何并集；
- `worms.py`：实现 Godot Voxel 文档中的 2D worm、Y 轴扰动和死胡同调制；
- `topology.py`：根据 seed 和洞网区域生成随机游走主洞、支洞边与洞室节点；显式路径只作兼容覆盖；
- `__init__.py`：只暴露洞穴子系统的公共入口。

`engine/pipeline.py` 只调用 `generate_underground()`，不再持有洞穴几何或地下建筑细节。
这样修改洞穴噪声、垂直范围或后续地下结构时，不会把地表河流和基础高度场一起改动。

服务端不需要从几何反推地下内容：每个 tile 额外写出 `cave_id.bin` 和
`fracture_id.bin`（都以 `0` 表示无该类型，`1..N` 对应各自 palette），`spans.bin`
则提供该列实际的地下垂直范围。普通洞穴与构造裂隙因此有独立的类型判断，Rust loader
可以分别判断某个 chunk/列属于矿洞还是裂隙，以及可进入的 Y 区间。
地下水另外写入每个 tile 的 `underground_water.bin`，每条固定 16 字节记录包含世界坐标、
`kind`（1=river、2=lake）和 `flowing` 状态；地下河矿脉/植物写入同样固定 16 字节的
`underground_resources.bin`，通过 manifest 的 `underground_resource_palette`、
`underground_material_palette` 和
`underground_resource_catalog` 解释逻辑产物索引。资源记录同时带有来源和稀有度，
而 `material` 只代表 BlueMap/Anvil 的真实 Minecraft 占位方块。
这些数据不会被压成二维表面水位。

洞穴噪声接口和 domain-warp 组织方式参考 FastNoiseLite 的公开 API 设计；本项目使用
独立 NumPy 实现，没有复制其 C++ 源码，也不增加 C++ 运行时依赖：

<https://github.com/Auburn/FastNoiseLite>

默认配方会为洞穴图生成确定性的地表入口，洞穴中心距地表 56 格。入口是独立阶段，只放宽入口柱的屋顶限制，
不会让普通洞道随机穿出地表。当前洞室直接作为自然洞穴的一部分生成，不输出人工矿厅或
固定建筑方块。洞穴天然内容先输出煤矿、铁矿、铜矿和荧光地衣等真实占位方块，Server
可以按刷新点规则替换为真正产物。

洞穴资源使用三档中文稀有度，默认总密度为“中”：

| 稀有度 | 倍率 | 默认资源 |
| --- | ---: | --- |
| 少 | 0.5 | 铜矿 |
| 中 | 1.0 | 铁矿 |
| 多 | 2.0 | 煤矿、荧光地衣 |

倍率先分配资源簇数量，再决定每簇的相邻扩展长度：少为 2-3 格，中为 3-5 格，多为 5-9 格。
资源只会落在已确认的洞壁上；每个簇会沿洞壁的 6 邻域扩展，避免只生成一个孤立方块。
荧光地衣也是洞壁附着的真实 `minecraft:glow_lichen`，不是控制台标记或抽象数据。

地下水同样是生成结果而不是前端效果：`Heightfield.underground_water_blocks` 保存每个
真实水方块的世界坐标、`kind`（`river`/`lake`）和 `flowing` 状态。地下河现在由
`UndergroundRiverNetwork` 独立生成：固定世界域中的多入口/出口先经过自己的水文起伏
代价场，再用 geodesic 路径多轮复用已有 conduit，形成共享主干和 junction；湖盆优先放在
网络节点，不要求附近存在普通洞穴。Anvil 转换层会把这些结果写成 `minecraft:water` 或
level=1 的流动水，Server 可以按 `kind` 识别并替换成真实的地下水内容。
默认地下河水深为 2 格、地下湖水深为 2 格；可在 `UndergroundRiverNetwork` 上分别调整
`water_depth` 和 `lake_depth`。地下河道自身使用独立的随机游走中心线、分支起点进度和
纵向深度剖面；每个地下河网络还会在河腔洞壁生成煤矿、铁矿、铜矿和荧光地衣/苔藓等
真实刷新点。矿脉和植物按 `少/中/多` 稀有度形成相连簇，植物必须邻近水体；Server 可
通过 `UndergroundBlock.resource_id/source/rarity` 或 `underground_resources.bin`
识别后替换真正产物。例如 `cave:iron_ore` 与 `underground_river:iron_ore` 可以
使用同一个 `minecraft:iron_ore` 占位方块，但由 Server 映射到不同的逻辑产物。

地下水算法的结构参考：

- pyKasso：<https://github.com/randlab/pyKasso>
- Tokunaga 树与 Barnes depression filling：<https://github.com/r-barnes/Barnes2013-Depressions>
- RichDEM：<https://github.com/r-barnes/richdem>
- pysheds：<https://github.com/mdbartos/pysheds>

裂隙、普通洞穴和地下河是三个不同的生成结果：`engine/underground_rivers/fractures.py`
对 seeded 噪声场使用 `abs(noise)` 提取 `noise == 0` 的 zero-isoline，再用噪声梯度把
中心线换算成方块单位的窄带。裂隙默认半宽约 1.2 格、高 28 格，上下端渐缩并带小幅
竖向错动；普通洞穴仍是 4-6 格高的圆形 worm/洞室。因此裂隙是“岩层被劈开的细长竖缝”，
不是另一种矿洞。地下河不读取这条裂隙场，只使用自己的水文代价场。三者最后只在
`solid_spans` 的几何并集阶段相交。`fracture_isoline_scale=0` 时使用
`cost_noise_scale * 0.72`，`fracture_isoline_width` 控制 zero-isoline 的连续保留，
`fracture_width` 控制真实方块半宽，`fracture_depth`/`fracture_height` 控制裂隙的垂直层。
它们都是 Python 配方参数，不是硬编码的四条通道。实现只借鉴公开项目的算法思路：

- line-based Simplex 洞穴：<https://github.com/sachPico/lineBasedCave_Simplex>
- Godot Voxel 程序化洞穴：<https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md>
- FastNoiseLite seeded noise：<https://github.com/Auburn/FastNoiseLite>

每个 raster tile 现在额外输出 `wilderness_id.bin`，每个地表柱一个 `uint8` id。
类型表写在 manifest 的 `wilderness_palette` 中，当前 id 是稳定契约：

- `0 grassland`：草地，可放草、灌木和树
- `1 mountains`：群山，可放石头、松树和矿物
- `2 lake`：湖泊，可放芦苇、水生植物和岸线内容
- `3 river`：河流，可放河岸内容和桥

类型表同时携带 `decoration_tags`，后续 `decorations.rs` 应按 id/tag 读取，
不要依赖前端颜色或中文显示名。

完整 raster 仍然是权威数据；manifest 另外提供 `overview`，指向一份用于快速定位的
低分辨率高度、地表材质、荒野与冰川地貌索引。Server 和 `decorations.rs` 应读取
完整 tile 数据，不使用 overview 代替详细数据。

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

默认视口围绕默认 seed 实际选中的出生平原，生成 1024×1024 的城镇观察窗口。
需要查看其它区域或指定其他 seed 的窗口时，再传入对应的
`--origin-x/--origin-z`。访问
`http://127.0.0.1:8100/`。每次 `render` 都会先替换
`generated/bluemap-world` 和 `.bluemap/web`，不会保留按版本命名的整份历史世界；
BlueMap 的客户端资源缓存保留在 `.bluemap/data`，避免每次重复下载。

核心区住宅和 Spawn 核心建筑（`19212` / `19237`）共用室内选点，每栋最多一个，
分别标记为 `house_interior_spawn` 和 `spawn_interior_spawn`。候选需要完整支撑面、
两格净空以及上方覆盖，排除草丛、地毯、下半砖和树冠/树枝误判；找不到时不生成点位。
区域与 JSON 契约集中在 `engine/settlement_points.py`。`engine/structures/interiors.py`
在未旋转的 Schematic 局部坐标中识别并缓存脚点，同名模板内容更新后会重新分析。
`engine/town_points.py` 只负责映射：局部点与结构方块共用旋转公式，再加上旋转后
模板的世界原点；Y 同时计入模板首层锚点与建筑下沉/抬升偏移。同一模板换位置或朝向
不会重新选择另一个房间。

BlueMap 预览会在已有 `role=npc_spawn` 兴趣点处生成成年村民，脚部 Y 保持原值，
XZ 对齐方块中心。村民使用真实的 1.20.1 实体存档，关闭 AI 和重力，便于查看刷新位置；
专用预览资源包为 BlueMap 提供村民模型，并复用已下载的原版纹理。普通世界导出默认
关闭该功能，Server 的兴趣点和防守区域数据不受影响。村民会被屋顶和墙壁正常遮挡，
查看室内点位需使用自由飞行进入房屋。
