"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_xuanshui_ruin',
    display_name='玄水故地',
    center_x=-6500.0,
    center_z=1500.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='玄水试剑阵核', pos_xyz=(-6500.0, 86.0, 1500.0), tags=('wild_formation', 'zong_core', 'origin:xuanshui', 'style:zhenmai'), unlock='剑痕石碑共鸣后可短时稳住局部灵气', qi_affinity=0.4, danger_bias=2),
        PoiDefinition(kind='stele', name='玄水剑痕碑', pos_xyz=(-6740.0, 84.0, 1260.0), tags=('lore', 'zong_stele', 'style:zhenmai'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='玄水守墓人', pos_xyz=(-6260.0, 84.0, 1740.0), tags=('zong_keeper', 'origin:xuanshui', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
