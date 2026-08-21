"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='blood_valley_east_scorch',
    display_name='血谷东陲焦土',
    center_x=4000.0,
    center_z=-2500.0,
    size_x=1000.0,
    size_z=400.0,
    terrain_profile='tribulation_scorch',
    shape='elongated',
    boundary_mode='hard',
    boundary_width=80,
    spirit_qi=0.3,
    danger_level=6,
    pois=(
        PoiDefinition(kind='stele', name='焦土雷痕碑', pos_xyz=(4060.0, 82.0, -2520.0), tags=('tribulation_scorch', 'warning', 'blood_valley'), unlock='雷雨天接近可见碑面蓝白电痕', qi_affinity=-0.1, danger_bias=2),
    ),
)
