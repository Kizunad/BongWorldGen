"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='celestial_isles',
    display_name='九霄浮岛',
    center_x=-4400.0,
    center_z=1200.0,
    size_x=1600.0,
    size_z=1600.0,
    terrain_profile='sky_isle',
    shape='ellipse',
    boundary_mode='soft',
    boundary_width=160,
    spirit_qi=0.85,
    danger_level=3,
    pois=(
        PoiDefinition(kind='shrine', name='悬空观道坛', pos_xyz=(-4400.0, 320.0, 1200.0), tags=('cultivation_site', 'sky_isle', 'high_tier'), unlock='需御风诀或灵鸟坐骑抵达', qi_affinity=0.7, danger_bias=0),
        PoiDefinition(kind='spirit_font', name='天脉垂露', pos_xyz=(-4200.0, 288.0, 1440.0), tags=('cultivation_site', 'sky_isle', 'water'), unlock='位于浮岛底部凹坑', qi_affinity=0.6, danger_bias=0),
        PoiDefinition(kind='ruin', name='坠落仙桥残基', pos_xyz=(-4800.0, 68.0, 800.0), tags=('ruin', 'legacy', 'ground'), unlock='地面残片可拾取仙桥碎铁', qi_affinity=0.25, danger_bias=0),
    ),
)
