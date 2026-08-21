"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='qingyun_peaks',
    display_name='青云残峰',
    center_x=-3000.0,
    center_z=-2000.0,
    size_x=1200.0,
    size_z=1200.0,
    terrain_profile='broken_peaks',
    shape='massif',
    boundary_mode='soft',
    boundary_width=128,
    spirit_qi=0.5,
    danger_level=2,
    pois=(
        PoiDefinition(kind='ruin', name='青云宗废殿', pos_xyz=(-3000.0, 180.0, -2000.0), tags=('sect_ruin', 'qingyun'), unlock='登顶主峰可见殿门', qi_affinity=0.3, danger_bias=1),
        PoiDefinition(kind='cave_mouth', name='采脉矿洞', pos_xyz=(-2820.0, 140.0, -1780.0), tags=('mining', 'old_sect'), unlock='', qi_affinity=0.2, danger_bias=0),
        PoiDefinition(kind='tomb', name='无名剑冢', pos_xyz=(-3400.0, 160.0, -2420.0), tags=('relic', 'legacy'), unlock='佩剑或持灵铁器者感应', qi_affinity=0.1, danger_bias=2),
    ),
)
