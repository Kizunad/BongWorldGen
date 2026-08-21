"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='south_ash_dead_zone',
    display_name='南荒余烬',
    center_x=-1200.0,
    center_z=8000.0,
    size_x=2000.0,
    size_z=2000.0,
    terrain_profile='ash_dead_zone',
    shape='irregular_blob',
    boundary_mode='hard',
    boundary_width=64,
    spirit_qi=0.0,
    danger_level=5,
    pois=(
        PoiDefinition(kind='stele', name='无声碑', pos_xyz=(-1380.0, 78.0, 8120.0), tags=('no_cadence', 'dead_zone', 'lore'), unlock='凝脉以上灵识扫过时只感到一片空白', qi_affinity=-1.0, danger_bias=1),
        PoiDefinition(kind='corpse_mound', name='干尸堆', pos_xyz=(-1060.0, 74.0, 7820.0), tags=('loot', 'dead_zone', 'surface_exposed'), unlock='搜刮 3-5 秒；仅有凡铁、退活骨币、干灵草', qi_affinity=-0.8, danger_bias=1),
    ),
)
