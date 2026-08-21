"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='wuxing_abyss',
    display_name='无垠深渊',
    center_x=5250.0,
    center_z=1400.0,
    size_x=1500.0,
    size_z=1800.0,
    terrain_profile='abyssal_maze',
    shape='subterranean_cluster',
    boundary_mode='semi_hard',
    boundary_width=104,
    spirit_qi=0.55,
    danger_level=5,
    pois=(
        PoiDefinition(kind='cave_mouth', name='玄渊主门', pos_xyz=(5250.0, 60.0, 1400.0), tags=('entrance', 'main'), unlock='', qi_affinity=0.2, danger_bias=0),
        PoiDefinition(kind='forbidden_hall', name='三重镇禁殿', pos_xyz=(5200.0, -30.0, 1480.0), tags=('secret_realm', 'locked', 'high_tier'), unlock='依次破除三层禁阵（浅→中→深）', qi_affinity=0.6, danger_bias=3),
        PoiDefinition(kind='spirit_font', name='深渊藏脉眼', pos_xyz=(5480.0, -38.0, 1820.0), tags=('cultivation_site', 'tier_3', 'hidden'), unlock='仅深渊层可达', qi_affinity=0.85, danger_bias=0),
        PoiDefinition(kind='tomb', name='无名修士骸骨阵', pos_xyz=(4880.0, 28.0, 780.0), tags=('battlefield', 'tier_1'), unlock='', qi_affinity=-0.1, danger_bias=2),
        PoiDefinition(kind='anomaly', name='坠渊井', pos_xyz=(5600.0, 68.0, 2100.0), tags=('bottomless', 'hazard'), unlock='一跃直达 tier 3（损伤极高）', qi_affinity=0.3, danger_bias=4),
    ),
)
