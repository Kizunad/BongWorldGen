"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='north_waste_east_scorch',
    display_name='北荒东陲焦土',
    center_x=2100.0,
    center_z=-8000.0,
    size_x=1200.0,
    size_z=1000.0,
    terrain_profile='tribulation_scorch',
    shape='irregular_blob',
    boundary_mode='hard',
    boundary_width=80,
    spirit_qi=0.28,
    danger_level=7,
    pois=(
        PoiDefinition(kind='tianjie_ascension_pit', name='北荒东陲渡虚劫坑', pos_xyz=(2100.0, 80.0, -8000.0), tags=('tribulation_scorch', 'ascension_pit', 'xujie_canxie'), unlock='凝脉以上灵识扫过时，可读到残留的渡虚劫回声', qi_affinity=-0.4, danger_bias=3),
    ),
)
