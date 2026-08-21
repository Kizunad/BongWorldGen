"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_youan_ruin',
    display_name='幽暗故地',
    center_x=2800.0,
    center_z=4500.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='幽暗影壁阵核', pos_xyz=(2800.0, 84.0, 4500.0), tags=('wild_formation', 'zong_core', 'origin:youan', 'style:tuike'), unlock='影壁灯火自亮时可感知替尸残法', qi_affinity=0.4, danger_bias=2),
        PoiDefinition(kind='stele', name='幽暗影壁残铭', pos_xyz=(2540.0, 80.0, 4240.0), tags=('lore', 'zong_stele', 'style:tuike'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='幽暗守墓人', pos_xyz=(3060.0, 80.0, 4760.0), tags=('zong_keeper', 'origin:youan', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
