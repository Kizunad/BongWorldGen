"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='blood_valley',
    display_name='血谷',
    center_x=3000.0,
    center_z=-2500.0,
    size_x=800.0,
    size_z=1500.0,
    terrain_profile='rift_valley',
    shape='rotated_rift',
    boundary_mode='hard',
    boundary_width=72,
    spirit_qi=0.3,
    danger_level=4,
    pois=(
        PoiDefinition(kind='altar', name='血月祭坛', pos_xyz=(3000.0, 48.0, -2500.0), tags=('blood_moon', 'event_anchor', 'ritual'), unlock='血月之夜激活', qi_affinity=-0.2, danger_bias=2),
        PoiDefinition(kind='ruin', name='骨冢台地', pos_xyz=(3180.0, 62.0, -1980.0), tags=('battlefield', 'loot'), unlock='', qi_affinity=-0.1, danger_bias=1),
        PoiDefinition(kind='gate', name='裂隙之门', pos_xyz=(2720.0, 55.0, -3080.0), tags=('portal', 'rift'), unlock='需残破令牌', qi_affinity=0.15, danger_bias=3),
        PoiDefinition(kind='stele', name='残缺战纪碑', pos_xyz=(3280.0, 80.0, -2300.0), tags=('lore', 'history'), unlock='识古文者可读', qi_affinity=0.0, danger_bias=0),
    ),
)
