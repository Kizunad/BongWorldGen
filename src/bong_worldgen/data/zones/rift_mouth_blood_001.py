"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='rift_mouth_blood_001',
    display_name='渊口荒丘·血谷露头',
    center_x=3200.0,
    center_z=-2800.0,
    size_x=300.0,
    size_z=300.0,
    terrain_profile='rift_mouth_barrens',
    shape='circular',
    boundary_mode='hard',
    boundary_width=48,
    spirit_qi=0.05,
    danger_level=5,
    pois=(
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·血谷露头', pos_xyz=(3200.0, 74.0, -2800.0), tags=('direction:entry',
 'kind:main',
 'family_id:daneng_01',
 'target_family_pos_xyz:550,100,550',
 'trigger_radius:2.0',
 'orientation:vertical',
 'facing:south'), unlock='血谷裂纹在灵识中呈现断续的寒白回声', qi_affinity=-0.8, danger_bias=4),
    ),
)
