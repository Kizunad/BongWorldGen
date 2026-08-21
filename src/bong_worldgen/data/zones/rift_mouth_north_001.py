"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='rift_mouth_north_001',
    display_name='渊口荒丘·枯木崖',
    center_x=-500.0,
    center_z=-8500.0,
    size_x=300.0,
    size_z=300.0,
    terrain_profile='rift_mouth_barrens',
    shape='circular',
    boundary_mode='hard',
    boundary_width=48,
    spirit_qi=0.05,
    danger_level=5,
    pois=(
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·枯木崖', pos_xyz=(-500.0, 74.0, -8500.0), tags=('direction:entry',
 'kind:main',
 'family_id:daneng_01',
 'target_family_pos_xyz:550,100,550',
 'trigger_radius:2.0',
 'orientation:vertical',
 'facing:north'), unlock='凝脉以上灵识扫过时，焦黑裂纹下传来极远的坠响', qi_affinity=-0.8, danger_bias=4),
    ),
)
