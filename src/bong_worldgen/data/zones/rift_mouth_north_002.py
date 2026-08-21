"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='rift_mouth_north_002',
    display_name='渊口荒丘·北荒东陲',
    center_x=2000.0,
    center_z=-7300.0,
    size_x=300.0,
    size_z=300.0,
    terrain_profile='rift_mouth_barrens',
    shape='circular',
    boundary_mode='hard',
    boundary_width=48,
    spirit_qi=0.05,
    danger_level=5,
    pois=(
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·北荒东陲', pos_xyz=(2000.0, 74.0, -7300.0), tags=('direction:entry',
 'kind:main',
 'family_id:zongmen_01',
 'target_family_pos_xyz:253,100,250',
 'trigger_radius:2.0',
 'orientation:vertical',
 'facing:east'), unlock='裂缝周围寒气凝白，靠近才察觉真元正被向下抽走', qi_affinity=-0.8, danger_bias=4),
    ),
)
