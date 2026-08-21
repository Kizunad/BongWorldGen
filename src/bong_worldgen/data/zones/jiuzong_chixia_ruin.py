"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_chixia_ruin',
    display_name='赤霞故地',
    center_x=6000.0,
    center_z=4000.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='赤霞引雷阵核', pos_xyz=(6000.0, 88.0, 4000.0), tags=('wild_formation', 'zong_core', 'origin:chixia', 'style:anqi'), unlock='雷雨或投料可短时激活', qi_affinity=0.4, danger_bias=3),
        PoiDefinition(kind='stele', name='赤霞雷纹碑', pos_xyz=(5740.0, 82.0, 3740.0), tags=('lore', 'zong_stele', 'style:anqi'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='赤霞守墓人', pos_xyz=(6260.0, 82.0, 4260.0), tags=('zong_keeper', 'origin:chixia', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
