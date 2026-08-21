"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='rift_mouth_west_001',
    display_name='渊口荒丘·西南遗宗',
    center_x=-3500.0,
    center_z=5500.0,
    size_x=300.0,
    size_z=300.0,
    terrain_profile='rift_mouth_barrens',
    shape='circular',
    boundary_mode='hard',
    boundary_width=48,
    spirit_qi=0.05,
    danger_level=5,
    pois=(
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·西南遗宗', pos_xyz=(-3500.0, 74.0, 5500.0), tags=('direction:entry',
 'kind:main',
 'family_id:zongmen_01',
 'target_family_pos_xyz:250,100,250',
 'trigger_radius:2.0',
 'orientation:vertical',
 'facing:west'), unlock='焦碑碎片之间似有宗门钟声倒灌，裂缝本身仍像普通地裂', qi_affinity=-0.8, danger_bias=4),
    ),
)
