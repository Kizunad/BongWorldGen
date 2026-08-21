"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_beiling_ruin',
    display_name='北陵故地',
    center_x=-1000.0,
    center_z=-8500.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='北陵阵眼锚柱', pos_xyz=(-1000.0, 84.0, -8500.0), tags=('wild_formation', 'zong_core', 'origin:beiling', 'style:zhenfa'), unlock='破解残阵可临时稳住局部灵气', qi_affinity=0.45, danger_bias=2),
        PoiDefinition(kind='stele', name='北陵量天碑', pos_xyz=(-1210.0, 84.0, -8660.0), tags=('lore', 'zong_stele', 'style:zhenfa'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='北陵守墓人', pos_xyz=(-760.0, 82.0, -8260.0), tags=('zong_keeper', 'origin:beiling', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
