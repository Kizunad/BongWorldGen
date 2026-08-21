"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='spawn',
    display_name='初醒原',
    center_x=0.0,
    center_z=0.0,
    size_x=1500.0,
    size_z=1500.0,
    terrain_profile='spawn_plain',
    shape='ellipse',
    boundary_mode='soft',
    boundary_width=96,
    spirit_qi=0.35,
    danger_level=1,
    pois=(
        PoiDefinition(kind='shrine', name='断碑观星台', pos_xyz=(-180.0, 72.0, -240.0), tags=('meditation', 'starter_landmark'), unlock='初次到达即可感知灵气涌动', qi_affinity=0.15, danger_bias=0),
        PoiDefinition(kind='ruin', name='残破村落', pos_xyz=(320.0, 70.0, 180.0), tags=('loot', 'starter_landmark'), unlock='', qi_affinity=-0.05, danger_bias=1),
        PoiDefinition(kind='cave_mouth', name='泥泞避难洞', pos_xyz=(-50.0, 66.0, 420.0), tags=('shelter', 'hidden'), unlock='日落后可见微光', qi_affinity=0.05, danger_bias=0),
        PoiDefinition(kind='spawn_tutorial_coffin', name='半埋石棺', pos_xyz=(0.0, 69.0, 0.0), tags=('spawn_tutorial', 'coffin', 'loot:niche_base'), unlock='', qi_affinity=0.05, danger_bias=0),
        PoiDefinition(kind='tutorial_lingquan', name='教学灵泉 #1', pos_xyz=(50.0, 65.0, 100.0), tags=('spawn_tutorial', 'index:1', 'qi:0.5'), unlock='', qi_affinity=0.35, danger_bias=0),
        PoiDefinition(kind='tutorial_lingquan', name='教学灵泉 #2', pos_xyz=(-30.0, 65.0, -80.0), tags=('spawn_tutorial', 'index:2', 'qi:0.5'), unlock='', qi_affinity=0.35, danger_bias=0),
        PoiDefinition(kind='tutorial_chest', name='灵泉边小匣', pos_xyz=(55.0, 65.0, 100.0), tags=('spawn_tutorial', 'loot:kaimai_dan', 'near_lingquan:1'), unlock='', qi_affinity=0.1, danger_bias=0),
        PoiDefinition(kind='tutorial_rogue_anchor', name='踽行散修', pos_xyz=(35.0, 70.0, -45.0), tags=('spawn_tutorial', 'rogue', 'killable'), unlock='', qi_affinity=0.02, danger_bias=0),
        PoiDefinition(kind='tutorial_rat_path', name='鼠群擦痕', pos_xyz=(25.0, 65.0, 50.0), tags=('spawn_tutorial', 'rat_swarm', 'placeholder:zombie'), unlock='', qi_affinity=0.0, danger_bias=1),
    ),
)
