"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='dan_zong_yi_yuan',
    display_name='丹宗遗园',
    center_x=-1600.0,
    center_z=4000.0,
    size_x=1600.0,
    size_z=1600.0,
    terrain_profile='dan_zong_yi_yuan',
    shape='ellipse',
    boundary_mode='soft',
    boundary_width=96,
    spirit_qi=0.4,
    danger_level=4,
    pois=(
        PoiDefinition(kind='ruin', name='百草丹殿', pos_xyz=(-1600.0, 82.0, 4000.0), tags=('dandao_path', 'alchemy', 'boss_lore', 'baolongwang_prequel'), unlock='found_by_exploration', qi_affinity=-0.1, danger_bias=2),
    ),
)
