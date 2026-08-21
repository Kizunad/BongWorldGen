"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_bloodstream_ruin',
    display_name='血溪故地',
    center_x=5500.0,
    center_z=-1000.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='血溪阵核残柱', pos_xyz=(5500.0, 84.0, -1000.0), tags=('wild_formation', 'zong_core', 'origin:bloodstream'), unlock='投入灵草 / 骨币 / 真元可短时活化', qi_affinity=0.4, danger_bias=2),
        PoiDefinition(kind='stele', name='万血斗台残碑', pos_xyz=(5360.0, 82.0, -1160.0), tags=('lore', 'zong_stele', 'style:baomai'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='血溪守墓人', pos_xyz=(5680.0, 80.0, -820.0), tags=('zong_keeper', 'origin:bloodstream', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
