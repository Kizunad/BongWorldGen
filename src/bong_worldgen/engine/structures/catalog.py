"""城镇结构资源目录。

这里保存资源的人类可读备注和少量放置语义，不把说明散落在布局算法中。
文件名仍是结构的稳定标识；新增模板只需在本目录登记，不需要修改加载器。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureNote:
    """一个结构模板的中文备注和放置约束。"""

    name_zh: str
    description_zh: str
    category: str
    # 若设置，包含该关键词的最低 Y 层就是模板的地表锚点。
    anchor_marker: str | None = None
    water_kind: str | None = None
    # 结构组用于要求多个模板在同一候选点成组放置，例如 Tango 的双塔。
    placement_group: str | None = None
    group_offset: tuple[int, int] = (0, 0)
    roles: tuple[str, ...] = ()


# 资源名称来自 building_schems；RJH=Rural Japanese House，CH=Country House。
# 这些备注也作为生成器和 Server 共享的稳定语义入口，避免依赖文件名猜测用途。
STRUCTURE_NOTES: dict[str, StructureNote] = {
    "31232": StructureNote("Medieval Stone Base A", "适合冒险或角色扮演的中世纪石制基础建筑。", "house"),
    "31062": StructureNote("Mediterranean Tower", "地中海风格瞭望塔，适合村落边缘或入口。", "tower"),
    "31368": StructureNote("Country House 2", "乡村住宅模板，兼作 CH2 的标准别名。", "house"),
    "CH2": StructureNote("Country House 2", "Country House 系列的第二座乡村住宅。", "house"),
    "31233": StructureNote("Small Fantasy House", "小型奇幻住宅，适合部落密集区。", "house"),
    "31270": StructureNote("FREE Medieval House With Garden", "带花园的中世纪住宅。", "house"),
    "31280": StructureNote(
        "Stonebricks Well",
        "石砖井；Fence 围栏层是地表首层，井水位于其下方。",
        "well",
        anchor_marker="fence",
        water_kind="well_water",
    ),
    "31304": StructureNote("Medieval Small House", "中世纪小屋住宅。", "house"),
    "31101": StructureNote(
        "Desert Well",
        "沙漠井；带枯木的那一层作为地表首层，井水向下布置。",
        "well",
        anchor_marker="dead_bush",
        water_kind="well_water",
    ),
    "31122": StructureNote("Simple Stone Tower", "简洁石塔，适合防御点或聚落地标。", "tower"),
    "31087": StructureNote("Red Food Stand", "红色食品摊位。", "stall"),
    "31085": StructureNote("Little Medieval House", "小型中世纪住宅。", "house"),
    "31281": StructureNote("Yellow Plants Stand", "黄色植物摊位。", "stall"),
    "31301": StructureNote("Small Carts Pack", "小型手推车与货车组合。", "prop"),
    "31302": StructureNote("Cyan Fish Stand", "青色鱼摊位。", "stall"),
    "31303": StructureNote("Abandoned Forest Well", "废弃森林井，带苔藓、植被和枯败装饰。", "well"),
    "31350": StructureNote("Rural Japanese Paddie 1", "日本乡村稻田第一段。", "farm"),
    "31358": StructureNote("Rural Japanese Paddie 3", "日本乡村稻田第三段。", "farm"),
    "31359": StructureNote("Japanese Rural Paddie 4", "日本乡村稻田第四段。", "farm"),
    "22031": StructureNote("Medieval House and Barn Connecting", "相连的中世纪住宅与谷仓。", "house"),
    "18812": StructureNote("Rustic Inn", "乡村旅店。", "house"),
    "18805": StructureNote("Rustic Build 2", "乡村建筑系列二号。", "house"),
    "31403": StructureNote("Medieval House Free", "免费中世纪住宅模板。", "house"),
    "18804": StructureNote("Rustic Build 1", "乡村建筑系列一号。", "house"),
    "18816": StructureNote("Rustic Build 3", "乡村建筑系列三号。", "house"),
    "18817": StructureNote("Rustic Build 4", "乡村建筑系列四号。", "house"),
    "18818": StructureNote("Rustic Build 5", "乡村建筑系列五号。", "house"),
    "18909": StructureNote("Rustic Build 6", "乡村建筑系列六号。", "house"),
    "18911": StructureNote("Rustic Build 1 - Granite & Bricks Version", "花岗岩与砖石乡村建筑一号。", "house"),
    "18921": StructureNote("Rustic Build 2 - Granite & Bricks Version", "花岗岩与砖石乡村建筑二号。", "house"),
    "18922": StructureNote("Rustic Build 3 - Granite & Bricks Version", "花岗岩与砖石乡村建筑三号。", "house"),
    "18923": StructureNote("Rustic Build 4 - Granite & Bricks Version", "花岗岩与砖石乡村建筑四号。", "house"),
    "19212": StructureNote("FREE Medium Rustic Spawn", "中型乡村出生点模板。", "spawn"),
    "19237": StructureNote("FREE Small Rustic Spawn", "小型乡村出生点模板。", "spawn"),
    "31406": StructureNote("Stone City Wall", "石制城市外围墙；按聚落边界重复拼接成长方体围墙。", "wall"),
    "31293": StructureNote(
        "Medieval City Gate",
        "中世纪城门；嵌入城镇核心区外围城墙的预留门洞，不参与住宅随机刷新。",
        "gate",
    ),
    "31244": StructureNote("Farmhouse", "农舍；作为城镇的普通住宅模板。", "house"),
    "31211": StructureNote("Cool Small Tree", "小型树木；用于聚落树木装饰池。", "tree"),
    "31052": StructureNote("Dry Tree", "枯树；用于聚落树木装饰池。", "tree"),
    "31212": StructureNote("Pointed Pillar", "尖顶石柱；保留给未来的聚落装饰层。", "prop"),
    "31056": StructureNote("Statue", "雕像；保留给未来的聚落装饰层。", "prop"),
    "31075": StructureNote("Embertrail Campsite", "余烬小径营地；保留给未来的聚落装饰层。", "camp"),
    "29642": StructureNote(
        "Kizumonogatari Eikou Cram School",
        "风格不适合主世界；仅归档，默认不参与任何刷新池。",
        "excluded",
    ),
    "31382": StructureNote(
        "Deepslate Survival House",
        "深板岩生存屋；作为远离城镇的独立地标刷新。",
        "standalone",
    ),
    "31279": StructureNote(
        "Overgrown Survival Castle",
        "蔓生生存城堡；作为远离城镇的独立地标刷新。",
        "standalone",
    ),
    "29753": StructureNote(
        "Greek Parthenon",
        "希腊帕特农神庙；适合独立地标或大型城镇核心地标。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "28204": StructureNote(
        "Asian Castle/House",
        "亚洲城堡住宅；适合独立刷新或大型城镇核心。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "27125": StructureNote(
        "Medieval Church",
        "中世纪教堂；适合独立刷新或城镇核心地标。",
        "standalone",
        roles=("standalone", "city_core_landmark"),
    ),
    "26820": StructureNote(
        "Japanese Temple",
        "大型日本寺庙；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "26476": StructureNote(
        "Dune Awakening Temple",
        "Dune Awakening 神殿；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "20493": StructureNote(
        "Tango Tek Hermitcraft Season 9 Tower 1",
        "Tango Tek Hermitcraft Season 9 双塔系列第一座；必须与同组塔一起独立刷新。",
        "standalone",
        placement_group="tango_hermitcraft_s9_towers",
        group_offset=(-24, 0),
        roles=("standalone", "structure_group_member"),
    ),
    "20503": StructureNote(
        "Tango Tek Hermitcraft Season 9 Tower 2",
        "Tango Tek Hermitcraft Season 9 双塔系列第二座；必须与同组塔一起独立刷新。",
        "standalone",
        placement_group="tango_hermitcraft_s9_towers",
        group_offset=(24, 0),
        roles=("standalone", "structure_group_member"),
    ),
    "26360": StructureNote(
        "Old and Tiny Church",
        "古老小教堂；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "26269": StructureNote(
        "Medieval Cathedral",
        "中世纪大教堂；适合独立刷新或大型城镇核心。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "25729": StructureNote(
        "A Coruna City Hall",
        "A Coruna 市政厅；适合大型城镇核心。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "25502": StructureNote(
        "Chapel and Cemetery",
        "小教堂与墓地组合；适合独立刷新。",
        "standalone",
        roles=("standalone", "city_core_landmark"),
    ),
    "23461": StructureNote(
        "Tam Chuc Pagoda",
        "巨型 Tam Chuc 宝塔；适合独立刷新或大型城镇核心。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "23437": StructureNote(
        "DerpGTX Temple of Amogus",
        "巨型 DerpGTX Amogus 神殿；作为独立结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "23160": StructureNote(
        "Crimson Temple",
        "绯红神殿；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "23076": StructureNote(
        "Japanese Castle",
        "巨型日本城堡；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "23075": StructureNote(
        "Chinese Castle with Interior",
        "带内部结构的巨型中式城堡；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "city_core_landmark", "large_structure"),
    ),
    "23074": StructureNote(
        "Japanese Shrine",
        "日本神社；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "22407": StructureNote(
        "Wolf Shrine",
        "狼神社；原模板来自 1.21，导入时会转换为目标 1.20.1 可用方块。",
        "standalone",
        roles=("standalone",),
    ),
    "22249": StructureNote(
        "Small Gray Church",
        "灰色小教堂；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "22101": StructureNote(
        "Cherry Blossom Temple",
        "樱花神庙；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "21961": StructureNote(
        "Tao-Koi Pagoda",
        "道鲤宝塔；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "21923": StructureNote(
        "Neoclassical Styled Building",
        "新古典主义建筑；适合独立刷新或城镇核心地标。",
        "standalone",
        roles=("standalone", "city_core_landmark"),
    ),
    "21876": StructureNote(
        "Ancient Tower",
        "古代高塔；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "21877": StructureNote(
        "Enchanting Table",
        "附魔台地标；适合独立刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "31112": StructureNote(
        "Earth Kingdom Inspired House",
        "土强国风格住宅；作为独立主题建筑刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "30896": StructureNote(
        "Suncrest Spire",
        "巨型 Suncrest 尖塔；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "28644": StructureNote(
        "WOW Dornogal Foundationhall",
        "巨型 WOW Dornogal Foundationhall；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "26109": StructureNote(
        "Suramar Inspired House",
        "苏拉玛风格住宅；作为独立主题建筑刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "26108": StructureNote(
        "Night Elven Inspired Treehouse",
        "暗夜精灵风格树屋；作为独立主题建筑刷新。",
        "standalone",
        roles=("standalone",),
    ),
    "ed-mereldar-church-hallowfall": StructureNote(
        "Ed Mereldar Church Hallowfall",
        "巨型 Hallowfall 教堂；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "22434": StructureNote(
        "WOW Antorus The Burning Throne",
        "巨型 WOW Antorus 燃烧王座；作为独立大型结构刷新。",
        "standalone",
        roles=("standalone", "large_structure"),
    ),
    "22072": StructureNote(
        "Market Stall",
        "集市摊位；纳入部落和城镇的非住宅建筑候选池。",
        "stall",
        roles=("settlement", "market_stall"),
    ),
    "19996": StructureNote("Rustic Build 6 - Granite & Bricks Version", "花岗岩与砖石乡村建筑六号。", "house"),
    "RJH": StructureNote("Rural Japanese House", "Rural Japanese House，日本乡村住宅主模板。", "house"),
    "RJH1": StructureNote("Rural Japanese House 1", "RJH 系列日本乡村住宅一号。", "house"),
    "RJH2": StructureNote("Rural Japanese House 2", "RJH 系列日本乡村住宅二号。", "house"),
    "RJH3": StructureNote("Rural Japanese House 3", "RJH 系列日本乡村住宅三号。", "house"),
    "RJH4": StructureNote("Rural Japanese House 4", "RJH 系列日本乡村住宅四号。", "house"),
    "RJH5": StructureNote("Rural Japanese House 5", "RJH 系列日本乡村住宅五号。", "house"),
    "RJH6": StructureNote("Rural Japanese House 6", "RJH 系列日本乡村住宅六号。", "house"),
    "RJH7": StructureNote("Rural Japanese House 7", "RJH 系列日本乡村住宅七号。", "house"),
    "Hut": StructureNote("Rustic Hut", "简易乡村小屋。", "house"),
}


def structure_note(name: str) -> StructureNote | None:
    """按结构 stem 返回备注；旋转后的模板沿用原始语义。"""

    return STRUCTURE_NOTES.get(name.split("@rot", 1)[0])


def structure_anchor_y(name: str, blocks: tuple[object, ...]) -> int:
    """计算结构相对地表的 Y 锚点。

    井模板的截图原点不在围栏层，因此按备注中的材料标记寻找首层；普通
    结构沿用 y=0，兼容没有登记备注的第三方模板。
    """

    note = structure_note(name)
    if note is None or note.anchor_marker is None:
        return 0
    marker = note.anchor_marker.lower()
    candidates = [
        int(getattr(block, "y"))
        for block in blocks
        if marker in str(getattr(block, "material", "")).lower()
    ]
    return min(candidates, default=0)


def structure_block_kind(name: str, material: str) -> str:
    """返回 Server 可查询的结构方块类型。"""

    note = structure_note(name)
    if note is not None and note.water_kind is not None and material.split("[", 1)[0] == "minecraft:water":
        return note.water_kind
    if note is not None and note.category == "spawn":
        return "spawn_core"
    if note is not None and note.category == "wall":
        return "settlement_wall"
    if note is not None and note.category == "gate":
        return "settlement_gate"
    if note is not None and note.category == "tower":
        return "settlement_tower"
    if note is not None and note.category == "stall":
        return "settlement_stall"
    if note is not None and note.category == "standalone":
        return "standalone_structure"
    return "house_schematic"


__all__ = [
    "STRUCTURE_NOTES",
    "StructureNote",
    "structure_anchor_y",
    "structure_block_kind",
    "structure_note",
]
