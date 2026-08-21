"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='drift_scorch_001',
    display_name='游离焦土',
    center_x=-4000.0,
    center_z=4000.0,
    size_x=1000.0,
    size_z=1000.0,
    terrain_profile='tribulation_scorch',
    shape='irregular_blob',
    boundary_mode='hard',
    boundary_width=80,
    spirit_qi=0.32,
    danger_level=5,
    pois=(
        PoiDefinition(kind='ruin', name='游雷玻璃滩', pos_xyz=(-4000.0, 78.0, 4000.0), tags=('tribulation_scorch', 'glass_fulgurite'), unlock='', qi_affinity=-0.05, danger_bias=1),
    ),
)
