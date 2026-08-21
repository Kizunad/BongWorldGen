"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='zhanhun_plain',
    display_name='战魂平野',
    center_x=2500.0,
    center_z=-4000.0,
    size_x=2000.0,
    size_z=1600.0,
    terrain_profile='ancient_battlefield',
    shape='ellipse',
    boundary_mode='semi_hard',
    boundary_width=128,
    spirit_qi=0.4,
    danger_level=4,
    pois=(
        PoiDefinition(kind='altar', name='血月召引阵', pos_xyz=(2500.0, 76.0, -4000.0), tags=('blood_moon', 'event_anchor', 'ritual', 'high_tier'), unlock='血月之夜自动激活；携血月令牌可提前触发', qi_affinity=-0.1, danger_bias=3),
        PoiDefinition(kind='formation', name='破碎阵眼', pos_xyz=(2100.0, 78.0, -4300.0), tags=('wild_formation', 'anomaly', 'event_anchor'), unlock='靠近触发灵气紊乱；破解可获阵核', qi_affinity=0.4, danger_bias=2),
        PoiDefinition(kind='gate', name='裂隙之痕', pos_xyz=(2900.0, 76.0, -3700.0), tags=('spacetime_rift', 'portal', 'anomaly'), unlock='持破界符可短暂稳定', qi_affinity=0.3, danger_bias=4),
        PoiDefinition(kind='stele', name='无名将军碑', pos_xyz=(2300.0, 82.0, -3800.0), tags=('lore', 'history', 'battlefield'), unlock='读碑可解阵营背景', qi_affinity=0.0, danger_bias=0),
        PoiDefinition(kind='tomb', name='万骨冢', pos_xyz=(2700.0, 74.0, -4200.0), tags=('battlefield', 'loot', 'cursed'), unlock='掘地三尺得骨粉/残甲', qi_affinity=-0.15, danger_bias=2),
        PoiDefinition(kind='shrine', name='残旗祭台', pos_xyz=(2200.0, 80.0, -3600.0), tags=('cursed_echo', 'offering'), unlock='供奉武器可得亡灵记忆', qi_affinity=0.1, danger_bias=1),
        PoiDefinition(kind='rift_portal', name='塌缩裂缝·大能陨落', pos_xyz=(3300.0, 90.0, -4500.0), tags=('direction:entry',
 'kind:main',
 'family_id:daneng_01',
 'target_family_pos_xyz:550,100,550',
 'orientation:vertical',
 'facing:east'), unlock='陨石坑底紫晶共鸣，引动空间裂痕', qi_affinity=-0.35, danger_bias=3),
    ),
)
