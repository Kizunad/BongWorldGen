# Zone 实现记录

分支：`feat/zone-impl`。目标来源：本地 `.goal-zone-impl.md`。不创建 PR。

## 分层与步骤

`data/zones` → `composition`（世界坐标查询、配方编译、混合）→ `engine`
（通用标量场与洞穴）→ `adapters`（raster / Anvil）。引擎不导入区域数据。
连续地表先混合，洞穴随后生成；浮岛使用实际实心段，不能用抬高实心山代替。

1. 区域形状、重叠优先级、坐标查询；初醒原、残峰、北荒三种纯配方。
2. 接入 preview、CLI、BlueMap；拆出引擎通用地表采样和后处理接口。
3. 实现边界带；验证整体/分块、负坐标、河道跨边界完全一致。
4. 分批实现湿地、裂谷、焦土、战场、遗迹、药园、王印台、洞穴、深渊、浮岛。
5. POI 从生成结果选择地表/洞底/浮岛高度，保留原始坐标供追溯。
6. BlueMap 实际出图、视觉检查、测量报告和复现命令。

现有 43 个测试的断言保持不变；每个实现提交同时记录行为验证。

## 空间约定

- `size_x/z` 是完整尺寸；`circular` 用较短直径，`ellipse` 和 `basin` 用椭圆。
- `plateau` 为四次超椭圆；`massif`、`irregular_blob`、`subterranean_cluster`
  使用可复现的三/五瓣扰动，轮廓不超出声明尺寸。
- 数据没有旋转角字段，`rotated_rift` 固定旋转 30°；不从 POI 猜方向。
- 小范围区域覆盖大范围区域，同面积按名称排序，结果不依赖数据加载次序。
- 空白地带使用背景配方。边界带的距离沿中心射线按世界方块计量。
- 后续非零 `boundary_width` 保持几何连续，hard 带比 soft 更陡；宽度为零才是精确阶跃。

## 证据

初始基线：`.venv/bin/pytest -q` → **43 passed**（2026-09-20）。

第一步建立 27 个区域的空间查询和三种地貌配方。契约验证覆盖所有 9 种 shape、
负坐标、完整直径、嵌套小区优先、权重守恒、源顺序无关，以及三个 seed 下
残峰高度标准差超过初醒原 8 倍、北荒平均高度高出 40 方块且起伏小于 15 方块。
此步查询先采用精确轮廓，边界带在第三步实现；尚未接入生产入口。

第一步验证：`.venv/bin/pytest -q` → **59 passed**。

第二步：新增 `ZoneTerrain`，preview、CLI、BlueMap 共用合成结果。引擎只增加
`sample_surface` / `finish_heightfield` 两个通用阶段；地表先混合，水体与洞穴后处理。
未实现的 12 个已知 profile 暂用背景，并在 manifest 的 `generation.pending_profiles`
显式列出；未知名字报错。生成入口契约检查真实高度和实心段，导出契约从
`spans.bin` 读回出生区高度 66–74（测试背景设为 180），排除只更新元数据的假实现。
验证：`.venv/bin/pytest -q` → **62 passed**。

第三步：边界采用五次 smoothstep（端点一、二阶导数均为零），soft 使用完整
`boundary_width`，semi_hard 使用 3/4，hard 使用 1/2；过渡带以轮廓为中心向内外展开。
叠加后各区域与背景权重之和为一。`boundary_weight.bin` 写入 `1 - 最大贡献权重`。
河流后处理新增纯坐标采样回调：沿整条路径按固定世界间距取站点，从最终混合地形
读取中心与两岸，避免裁剪窗口改变水面。引擎不知道回调背后的 zone 数据。
契约验证 50 格高差在宽度 128 的最陡模式中每格变化小于 1.6；跨区河流整体、
不等宽切块、负坐标、单列和步长 4 采样在共同坐标的高度、水位、河床与 spans 一致。
验证：`.venv/bin/pytest -q` → **71 passed**，未修改任何原有断言。

第四步 A：实现 `spring_marsh`、`rift_valley`、`ash_dead_zone`、
`rift_mouth_barrens`、`tribulation_scorch`。湿地使用带起伏的浅盆地，裂谷的路径
与轮廓同向旋转，灰烬地使用侵蚀状脊线与凹地，渊口/劫坑使用盆地与环形隆起。
三个 seed 的湿地样本含 15%–85% 水面且水深中位数 < 6；裂谷两岸比谷底高 > 30；
三个实际坑地中心比边缘低 > 25、边缘比外侧高 > 8；灰烬地为基本干燥且起伏 > 20。
验证：`.venv/bin/pytest -q` → **81 passed**。

第四步 B：战魂平野增加浅弹坑与低矮土垄；七处宗门遗址共享平整的抬升中心和
破碎外台；丹宗遗园有 92/86/80 三层药圃；王印台有 104/91 两层同心台地。
引擎增加通用 `Plateau` 参数，只表达椭圆平面和连续边缘，不含区域名称或方块知识。
七处遗址分别验证中央高度变化 < 0.01 且高出外围 > 12；药园和王印台验证实际层高。
验证：`.venv/bin/pytest -q` → **91 passed**。

第四步 C：`cave_network`、`abyssal_maze` 编译为通用 CaveNetwork 参数。地下 POI
作为天然洞室坐标，洞口/坠渊井作为入口；地穴深度参考原始地下 Y，深渊采用
34/72/112 三层深度。引擎新增纯 Point 类型的可指定洞室与入口，不认识 POI 或 zone。
真实地下 POI 列验证存在至少 4 格净空，深渊列有四段实心体/三层洞室；入口穿透
地表。整块与切块的 spans 和 cave_id 逐项相等。raster 使用所有网络的稳定全局色表。
验证：`.venv/bin/pytest -q` → **96 passed**。

第四步 D：`sky_isle` 输出一座主岛和两座卫星岛，底部使用椭球厚度收薄，原地面
继续保留在下一段。引擎新增通用 `FloatingIsland` 实心体参数，最高地表 < 318，适配
Minecraft 1.20.1 的高度范围。方块契约验证 y=160–175 全为空气，y=272–287 含岛体；
岛缘分块逐字节一致。27 个 zone / 15 个 profile 全部有实现，已移除暂用背景的名单。
当前 Heightfield 水层只能表示最高地表上方的一层水，因此浮岛下方的地面水不会输出；
本组主岛地面高于海平面。该限制不影响实心体、洞穴与 POI 的垂直范围。
验证：`.venv/bin/pytest -q` → **100 passed**。

第五步：78 个 POI 保持原始 X/Z，按导出的实心段选择脚部 Y = 支撑块顶 Y + 1。
地下点选最接近原始 Y 且有净空的洞底；浮岛点使用岛面，带 ground 标签的遗迹保留
地面定位；井口柱已被挖通则使用实际入口底面，水下点标为 underwater_surface。
输入 dataclass 不变，manifest 增加 `authored_pos_xyz` 和 `placement`。分数负坐标
按 floor 选 Minecraft 列，不能用四舍五入跨到另一列。
78 点逐一验证脚下实心、头顶两格为空、X/Z 不变；raster 回读验证与 POI Y 相符。
验证：`.venv/bin/pytest -q` → **103 passed**。

第六步（断点收拾与验收工具）：

- 河流剖面改为按 `ZoneTerrain` 的固定世界坐标预计算并缓存，分块生成不再重复
  采样长河；洞穴 3D 噪声在所有 Y 层复用同一 seed，洞口改为连续竖井，路径层
  做保守垂直裁剪。此前 256 方块样区会溢出四段 spans；修复后地穴和深渊窗口
  均满足 `max_solid_spans <= 4`，新增回归测试。
- Anvil 增加 `append=True`，按现有 region 的 zlib chunk 合并稀疏 gallery patch，
  不覆盖已写 chunk；BlueMap 增加 `--zone-gallery`，每个 profile 在真实世界坐标
  输出 256×256 patch，POI marker 写入配置，gallery 对洞穴保留完整 Y 范围。
- `tools/zone_evidence.py` 重算每个 zone 的均值、标准差、湿地比例、spans 上限、
  高度 hash，并以 48/49 列切块逐项比较高度、水位、河床、spans、cave_id；
  `tools/plot_zone_evidence.py` 输出 `profiles.png` 与 `island-section.png`。
  绘图依赖系统 matplotlib，运行前可设 `MPLCONFIGDIR` 到 `generated/.mplcache`。

证据命令：

```bash
.venv/bin/pytest -q
.venv/bin/python tools/zone_evidence.py
MPLCONFIGDIR="$PWD/generated/.mplcache" python3 tools/plot_zone_evidence.py generated/zone-evidence
.venv/bin/python tools/bluemap.py render --accept-minecraft-eula --zone-gallery --width 256 --height 256
```

2026-09-21 验收结果：`107 passed`；报告为 27 zones / 15 profiles / 78 POI，
所有 27 个 zone 的 split_equal 为 true，最大实心段数为 4。BlueMap 5.23 使用 Java 25
完成渲染，资源下载和 map update 日志均到 100%；证据图经人工查看，平原、残峰、
高原、湿地、裂谷、遗址台地、洞穴和悬空岛在同一高度标尺下有可见差异，岛体下方
保持空气层。gallery 是稀疏真实坐标样区，空白区域是未生成区，不代表背景地貌。

第七步：把 dominant zone 归属写进 raster 契约。每列新增 `zone_id.bin`，manifest
声明稳定的 `zone_palette` 和 `zone_encoding.none = 255`；边界带内按实际最大权重
选择 dominant zone，仍保留 `boundary_weight.bin` 供下游识别过渡区。Three.js 解码器
按 tile layers 声明加载 `zone_id`，不改变现有地形网格行为。测试覆盖跨 tile 的两个
zone、边界竞争、背景哨兵和 palette 范围。验证：`.venv/bin/pytest -q` → **108 passed**。

第八步：固定 `zone_palette` 的规范顺序。导出时按 zone 名称排序，因而生成模块的
装配顺序变化不会重新编号已存在区域；`255` 继续专用于背景，最多允许 255 个区域。
适配器在转换前拒绝负值和超出 `u8` 范围的区域 ID。反向输入顺序的两个导出结果逐 tile
比较 `zone_id.bin` 完全一致。
验证：`.venv/bin/pytest -q` → **110 passed**。

第九步：控制台消费 `zone_id`。Three.js 解码器新增 `zone` 配色模式，区域开关把每列
的 canonical palette 名称映射成稳定色板；背景哨兵 `255` 回退到真实地形色。区域层
使用透明叠加材质，不改变地形几何、水体或荒野高亮层。
验证：`cd console && npm test` → **61 passed**；`npm run build`（包含 TypeScript
类型检查）通过。

第十步：修复导出区域归属时遗漏背景权重的错误。此前 tile 只要与某个 zone 相交，
该 tile 的零权重列也会收到 zone ID；边界外以背景为主的列也被错误归属。现在背景
参与最大权重选择，并与 `ZoneIndex.zone_at` 一样在平权时保留背景；边界强度复用
同一最大权重。负坐标、两个重叠 zone 和背景构成的 64×32 样区逐列对照查询结果，
分别按 32 和 64 大小导出 tile，结果完全一致。此回归在修复前有 1443/2048 列错误。
复现：`.venv/bin/pytest -q tests/test_zone_raster.py -k matches_point_queries`。
验证：`.venv/bin/pytest -q` → **111 passed**。

第十一步：全图 overview 新增 `overview_zone_id.bin`，与详细 tile 共用 `zone_palette`。
修复原有 tile 原点采样造成的偏移与覆盖：所有 overview 层现在按 manifest 的世界
坐标取值。未对齐负坐标样区在 tile_size=16/64 时，与逐列生成及详细 raster 相等。
控制台区域开关同时控制 overview 叠加；旧 manifest 不请求未声明的区域文件，声明
但缺失或截断的文件报错，overview 与详细 tile 使用相同配色。overview 的线性颜色
插值仅供定位，精确区域边界以详细 tile 为准。

验证：Python **113 passed**；控制台 **66 passed**；`npm run build` 通过。
离线证据工具直接拼接导出的 `zone_id.bin` 并逐点核对 overview，真实血谷渊口
576×576 样区（非对齐边界）**324/324** 采样点一致，输出对照 PNG：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from bong_worldgen.preview_world import export_preview_world
export_preview_world(Path('generated/zone-overview-evidence'), min_x=2917, max_x=3492,
                     min_z=-3075, max_z=-2500, tile_size=64)
PY
MPLCONFIGDIR="$PWD/generated/.mplcache" python3 tools/plot_zone_overview.py \
  generated/zone-overview-evidence/rasters/manifest.json \
  --output generated/zone-overview-evidence/zone-overview.png
```

第十二步：把非中心洞口接入主洞路径。无垠深渊的坠渊井 (5600, 2100) 原先距最近
主路径约 304.63 方块，仅生成孤立竖井。合成层现在将洞口与洞室一起编译成各层的
路径端点，引擎仍只接收通用洞穴参数。真实 32×48 体素样区排除地表空气，按脚部
与头部两格净空构建六邻接通路，从深井底检查向主洞方向 40 方块外的浅/中/深三层
开口均可达。seed=7/812731/2026 在修复前全部失败，修复后全部通过；该证据验证
井口附近三层连接，不等同于验证整个深渊所有洞室的可达性。

验证：`.venv/bin/pytest -q` → **116 passed**；
`.venv/bin/python tools/zone_evidence.py --output generated/zone-evidence-cave-connection`
重算 27 zones / 15 profiles / 78 POI，全部 split_equal=true，最大实心段数仍为 4。
定向复现：`.venv/bin/pytest -q tests/test_zone_cave_connections.py`。

第十三步：CLI 的 NPZ 保存实际区域与垂直几何。新增实心段、区域 ID/palette、
边界权重、材质 palette、世界原点和步长，支持 `np.load(..., allow_pickle=False)`。
合成层 `ZoneBlend.dominant()` 统一标量/批量查询的最大权重规则；独立 adapter
将贡献索引映射到 raster palette，CLI 和完整世界导出共用该转换。
契约测试覆盖负分数坐标、2.5 方块步长、深渊四段/浮岛两段，以及同一窗口的 NPZ
与 raster 归属、边界权重、水位、材质、荒野、洞穴 ID 和实心段逐项相等。
验证：`.venv/bin/pytest -q` → **120 passed**，compileall 通过。实际 CLI 导出命令：

```bash
.venv/bin/bong-worldgen --width 32 --height 48 --origin-x 5576 --origin-z 2060 \
  --cell-size 1 --output generated/zone-evidence-cave-connection/well.npz
```

第十四步：消除荒野分类的 tile 接缝。原来坡度使用 tile 内的 `np.gradient`，窗口
边缘改成单边差分，单列则一律当成平地；南荒余烬 64×64 样区左右切块有 2 列分类
不一致。zone adapter 现在多生成一圈邻列，中心差分后裁剪，CLI 与完整 raster
共用这一入口。坡度除以实际 `cell_size`，语义层保持相同采样步长下的分块一致性；
不同步长的坡度是不同尺度的近似。引擎、几何高度和实心段规则没有改变。
新增契约覆盖步长 1/4、单列/单行、负坐标、32/64 tile raster 与 overview 一致性。
定向复现：`.venv/bin/pytest -q tests/test_zone_wilderness.py`。
验证：`.venv/bin/pytest -q` → **124 passed**，compileall 通过；
`generated/zone-evidence-cave-connection/wilderness-split.json` 记录默认 seed 下
(-1232, 7968) 起始 64×64 样区左右分块的荒野分类差异为 **0**。
