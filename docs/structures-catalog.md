# 结构目录

城镇结构文件位于 `assets/structures/houses/`；独立世界结构位于
`assets/structures/standalone/`。两类资源都由 Python 目录
`src/bong_worldgen/engine/structures/catalog.py` 提供中文备注和放置语义。
结构本身仍按 `.schem` / `.litematic` 原文件导入，目标 Minecraft 版本为 1.20.1。

## 建筑与地标

| 文件 | 名称与备注 | 类别 |
| --- | --- | --- |
| `31232.schem` | Medieval Stone Base A：中世纪石制基础建筑 | house |
| `31062.schem` | Mediterranean Tower：地中海瞭望塔 | tower |
| `31368.schem` / `CH2.schem` | Country House 2：乡村住宅，CH 代表 Country House | house |
| `31233.schem` | Small Fantasy House：小型奇幻住宅 | house |
| `31270.litematic` | FREE Medieval House With Garden：带花园的中世纪住宅 | house |
| `31304.schem` | Medieval Small House：中世纪小屋 | house |
| `31085.schem` | Little Medieval House：小型中世纪住宅 | house |
| `22031.litematic` | Medieval House and Barn Connecting：相连住宅与谷仓 | house |
| `18812.schem` | Rustic Inn：乡村旅店 | house |
| `18804.schem`、`18805.schem`、`18816.schem`、`18817.schem`、`18818.schem` | Rustic Build 1/2/3/4/5：乡村建筑系列 | house |
| `18909.schem` | Rustic Build 6：乡村建筑六号 | house |
| `18911.schem`、`18921.schem`、`18922.schem`、`18923.schem`、`19996.schem` | Rustic Build Granite & Bricks Versions：花岗岩与砖石变体 | house |
| `31403.litematic` | Medieval House Free：免费中世纪住宅 | house |
| `Hut.schem` | Rustic Hut：简易乡村小屋 | house |
| `RJH.schem`、`RJH1.schem`…`RJH7.schem` | Rural Japanese House 系列，RJH 代表 Rural Japanese House | house |
| `31122.schem` | Simple Stone Tower：简洁石塔 | tower |
| `19212.schem` | FREE Medium Rustic Spawn：中型乡村出生点 | spawn |
| `19237.schem` | FREE Small Rustic Spawn：小型乡村出生点 | spawn |
| `31406.schem` | Stone City Wall：石制城市外围墙段，拼接为矩形外围墙 | wall |
| `31293.schem` | Medieval City Gate：中世纪城门，成对嵌入外围城墙长轴两端的预留门洞 | gate |
| `31244.schem` | Farmhouse：农舍，作为城镇的普通住宅模板 | house |

## 独立世界结构

下列模板不进入 Weighted Eden 部落住宅池，也不会被围墙利用率逻辑处理。
生成器以 `seed + 世界坐标网格` 稳定选点，每个网格只会选择一个结构单位；结构组视为
一个单位，只有全部成员都能落在干地、满足坡度与高差约束并且不重叠时才会整体放置。
每座结构都会产生类别为 `standalone` 的 `settlement_spawn_area`，供 Server 使用真实
三维 footprint 与 `defense_radius` 进行 NPC 刷新和防守判定。

| 文件 | 名称与备注 | 放置语义 |
| --- | --- | --- |
| `31382.schem` | Deepslate Survival House：深板岩生存屋 | 独立地标 |
| `31279.schem` | Overgrown Survival Castle：蔓生生存城堡 | 独立地标 |
| `29753.schem` | Greek Parthenon：希腊帕特农神庙 | 大型城镇核心候选 |
| `28204.litematic` | Asian Castle/House：亚洲城堡住宅 | 大型城镇核心候选 |
| `27125.schem` | Medieval Church：中世纪教堂 | 城镇核心候选 |
| `26820.schem`、`26476.litematic` | Japanese Temple、Dune Awakening Temple | 大型独立地标 |
| `20493.litematic`、`20503.litematic` | Tango Tek Hermitcraft Season 9 双塔 | 不可拆分结构组 |
| `26360.litematic`、`26269.litematic` | Old and Tiny Church、Medieval Cathedral | 教堂地标 |
| `25729.litematic`、`25502.schem` | A Coruna City Hall、Chapel and Cemetery | 城市/宗教地标 |
| `23461.litematic`、`23437.litematic`、`23160.litematic` | Tam Chuc Pagoda、Temple of Amogus、Crimson Temple | 大型神殿地标 |
| `23076.litematic`、`23075.litematic`、`23074.litematic` | Japanese Castle、Chinese Castle、Japanese Shrine | 东亚主题地标 |
| `22434.litematic` | WOW Antorus The Burning Throne：燃烧王座 | 巨型独立地标 |
| `22407.schem`、`22249.litematic`、`22101.litematic`、`21961.schem` | Wolf Shrine、小灰教堂、樱花神殿、Tao-Koi Pagoda | 宗教地标 |
| `21923.schem`、`21876.litematic`、`21877.litematic` | Neoclassical Building、Ancient Tower、Enchanting Table | 建筑/装饰地标 |
| `31112.litematic`、`30896.litematic`、`28644.litematic` | Earth Kingdom House、Suncrest Spire、Dornogal Foundationhall | 主题/大型地标 |
| `26109.schem`、`26108.schem`、`ed-mereldar-church-hallowfall.schem` | Suramar House、Night Elven Treehouse、Hallowfall Church | 主题地标 |

## 摊位、农田与道具

| 文件 | 名称与备注 | 类别 |
| --- | --- | --- |
| `31087.schem` | Red Food Stand：红色食品摊 | stall |
| `31281.schem` | Yellow Plants Stand：黄色植物摊 | stall |
| `31302.schem` | Cyan Fish Stand：青色鱼摊 | stall |
| `22072.litematic` | Market Stall：集市摊位，纳入部落和城镇设施候选池 | stall |
| `31301.schem` | Small Carts Pack：小型手推车与货车组合 | prop |
| `31350.schem` | Rural Japanese Paddie 1：日本乡村稻田第一段 | farm |
| `31358.schem` | Rural Japanese Paddie 3：日本乡村稻田第三段 | farm |
| `31359.schem` | Japanese Rural Paddie 4：日本乡村稻田第四段 | farm |
| `31212.schem` | Pointed Pillar：尖顶石柱，已分类为装饰资源 | prop |
| `31056.schem` | Statue：雕像，已分类为装饰资源 | prop |
| `31075.schem` | Embertrail Campsite：余烬小径营地，已分类为装饰资源 | camp |

## 树木资源

树木模板位于 `assets/structures/trees/`，由城镇树木层按现有数量与间距配置
随机选择；它们不进入住宅池。

| 文件 | 名称与备注 | 类别 |
| --- | --- | --- |
| `31211.schem` | Cool Small Tree：小型树木 | tree |
| `31052.schem` | Dry Tree：枯树 | tree |

## 已归档模板

`assets/structures/excluded/` 只保存暂不适合主世界风格的资源，任何生成层都不会扫描
该目录。`29642.litematic`（Kizumonogatari Eikou Cram School，`174 x 176 x 214`）目前
仅归档，默认禁用。

## 水井特殊规则

| 文件 | 锚点规则 | 水体语义 |
| --- | --- | --- |
| `31280.schem` | 最低的 Fence 方块所在层是第一层，井水向下保留 | `well_water` |
| `31101.schem` | 最低的枯木（`dead_bush`）所在层是第一层，井水向下保留 | `well_water` |
| `31303.schem` | Abandoned Forest Well：废弃森林井，按普通结构原点放置 | `house_schematic` |

井的锚点只影响 Y 原点，不改动模板的 X/Z footprint。井水仍是实际的
`minecraft:water` 方块，并在 `SettlementBlock.kind` 中标为 `well_water`，Server
可以按类型替换或填充后续真实内容。
