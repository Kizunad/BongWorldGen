"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='lingquan_marsh',
    display_name='灵泉湿地',
    center_x=-2500.0,
    center_z=2500.0,
    size_x=1000.0,
    size_z=1000.0,
    terrain_profile='spring_marsh',
    shape='basin',
    boundary_mode='soft',
    boundary_width=128,
    spirit_qi=0.7,
    danger_level=3,
    pois=(
        PoiDefinition(kind='spirit_font', name='主灵泉眼', pos_xyz=(-2500.0, 46.0, 2500.0), tags=('cultivation_site', 'water'), unlock='水下静坐可调息', qi_affinity=0.6, danger_bias=0),
        PoiDefinition(kind='shrine', name='苇渡神龛', pos_xyz=(-2760.0, 52.0, 2320.0), tags=('offering', 'blessing'), unlock='', qi_affinity=0.25, danger_bias=0),
        PoiDefinition(kind='herb_patch', name='灵草甸', pos_xyz=(-2240.0, 54.0, 2760.0), tags=('gather', 'alchemy'), unlock='识药者可采集', qi_affinity=0.3, danger_bias=0),
        PoiDefinition(kind='tomb', name='隐修遗冢', pos_xyz=(-2080.0, 56.0, 2900.0), tags=('hidden', 'legacy'), unlock='夜间月光强时显现', qi_affinity=0.2, danger_bias=1),
    ),
)
