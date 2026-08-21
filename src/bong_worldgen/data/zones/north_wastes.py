"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='north_wastes',
    display_name='北荒',
    center_x=0.0,
    center_z=-7000.0,
    size_x=3000.0,
    size_z=2000.0,
    terrain_profile='waste_plateau',
    shape='plateau',
    boundary_mode='hard',
    boundary_width=96,
    spirit_qi=0.05,
    danger_level=5,
    pois=(
        PoiDefinition(kind='ruin', name='沉寂仙宫碎片', pos_xyz=(0.0, 90.0, -7000.0), tags=('sunken_palace', 'relic', 'high_tier'), unlock='携带仙宫残符者显现', qi_affinity=0.3, danger_bias=3),
        PoiDefinition(kind='tomb', name='鲸坠骸骨', pos_xyz=(-620.0, 76.0, -7440.0), tags=('fossil', 'lore'), unlock='', qi_affinity=-0.15, danger_bias=1),
        PoiDefinition(kind='anomaly', name='虚压涡', pos_xyz=(580.0, 70.0, -6710.0), tags=('null_pressure', 'hazard'), unlock='持护身灵玉可近', qi_affinity=-0.6, danger_bias=3),
        PoiDefinition(kind='stele', name='末法纪年碑', pos_xyz=(940.0, 82.0, -6360.0), tags=('lore', 'history'), unlock='识古文且灵识清明者可读', qi_affinity=0.0, danger_bias=0),
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·宗门遗迹', pos_xyz=(200.0, 100.0, -7200.0), tags=('direction:entry',
 'kind:main',
 'family_id:zongmen_01',
 'target_family_pos_xyz:250,100,250',
 'orientation:vertical',
 'facing:north'), unlock='灵识扫过裂隙，听见远处宗门钟鸣残响', qi_affinity=-0.3, danger_bias=2),
    ),
)
