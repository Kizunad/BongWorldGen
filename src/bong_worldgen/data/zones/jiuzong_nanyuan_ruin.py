"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='jiuzong_nanyuan_ruin',
    display_name='南渊故地',
    center_x=0.0,
    center_z=6000.0,
    size_x=800.0,
    size_z=800.0,
    terrain_profile='jiu_zong_ruin',
    shape='irregular_blob',
    boundary_mode='semi_hard',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=6,
    pois=(
        PoiDefinition(kind='formation', name='南渊蛊池阵核', pos_xyz=(0.0, 82.0, 6000.0), tags=('wild_formation', 'zong_core', 'origin:nanyuan', 'style:dugu'), unlock='阵核活化后才可能从藏经残基取得蛊术残卷', qi_affinity=0.42, danger_bias=2),
        PoiDefinition(kind='stele', name='南渊蛊皿铭', pos_xyz=(-240.0, 78.0, 5740.0), tags=('lore', 'zong_stele', 'style:dugu'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='npc_anchor', name='南渊守墓人', pos_xyz=(260.0, 78.0, 6260.0), tags=('zong_keeper', 'origin:nanyuan', 'neutral_until_core_activated'), unlock='', qi_affinity=0.0, danger_bias=2),
    ),
)
