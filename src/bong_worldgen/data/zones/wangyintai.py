"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='wangyintai',
    display_name='王印台',
    center_x=4000.0,
    center_z=-1650.0,
    size_x=1000.0,
    size_z=1000.0,
    terrain_profile='wangyintai',
    shape='ellipse',
    boundary_mode='soft',
    boundary_width=64,
    spirit_qi=-0.15,
    danger_level=3,
    pois=(
        PoiDefinition(kind='guantiantai', name='观天台', pos_xyz=(4000.0, 92.0, -1650.0), tags=('wangyintai', 'vortex_formation', 'woliu_path'), unlock='found_by_exploration', qi_affinity=-0.15, danger_bias=1),
    ),
)
