"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='youan_depths',
    display_name='幽暗地穴',
    center_x=2000.0,
    center_z=3000.0,
    size_x=1200.0,
    size_z=1200.0,
    terrain_profile='cave_network',
    shape='subterranean_cluster',
    boundary_mode='semi_hard',
    boundary_width=88,
    spirit_qi=0.4,
    danger_level=4,
    pois=(
        PoiDefinition(kind='cave_mouth', name='陷穴主门', pos_xyz=(2000.0, 60.0, 3000.0), tags=('entrance', 'main'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='forbidden_hall', name='封禁厅', pos_xyz=(2120.0, 20.0, 3080.0), tags=('secret_realm', 'locked'), unlock='破除三层封禁阵', qi_affinity=0.45, danger_bias=2),
        PoiDefinition(kind='spirit_font', name='暗河灵泉', pos_xyz=(1820.0, 8.0, 3160.0), tags=('hidden', 'water', 'cultivation_site'), unlock='潜入暗河尽头', qi_affinity=0.55, danger_bias=0),
        PoiDefinition(kind='altar', name='镇邪古坛', pos_xyz=(2420.0, 16.0, 3400.0), tags=('ward', 'ancient'), unlock='持净灵玉近前', qi_affinity=0.2, danger_bias=2),
    ),
)
