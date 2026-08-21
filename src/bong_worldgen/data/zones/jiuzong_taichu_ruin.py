"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_taichu_ruin',
    display_name='太初故地',
    center_x=0.0,
    center_z=-10000.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='太初太极阵核', pos_xyz=(0.0, 88.0, -10000.0), tags=('wild_formation', 'zong_core', 'origin:taichu', 'style:multi_style'), unlock='黑白阵盘短时合拢时可投料活化', qi_affinity=0.45, danger_bias=2),
        PoiDefinition(kind='stele', name='太初任督图', pos_xyz=(-260.0, 84.0, -10260.0), tags=('lore', 'zong_stele', 'style:multi_style'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='太初守墓人', pos_xyz=(260.0, 84.0, -9740.0), tags=('zong_keeper', 'origin:taichu', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
