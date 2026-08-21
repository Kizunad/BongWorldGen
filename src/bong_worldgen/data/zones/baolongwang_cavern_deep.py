"""Generated zone data. Edit this file for zone-local changes."""

from ..world import PoiDefinition, ZoneDefinition

ZONE = ZoneDefinition(
    name='baolongwang_cavern_deep',
    display_name='暴龙王巢穴',
    center_x=1750.0,
    center_z=-5150.0,
    size_x=500.0,
    size_z=700.0,
    terrain_profile='cave_network',
    shape='circular',
    boundary_mode='hard',
    boundary_width=48,
    spirit_qi=-0.8,
    danger_level=5,
    pois=(
        PoiDefinition(kind='boss_arena', name='暴龙王丹炉', pos_xyz=(1750.0, -45.0, -5100.0), tags=('baolongwang', 'boss_furnace', 'dandao_path'), unlock='击破外层甲壳可摧毁丹炉，切断续命丹供给', qi_affinity=-0.9, danger_bias=4),
        PoiDefinition(kind='boss_spawn', name='暴龙王', pos_xyz=(1750.0, -40.0, -5200.0), tags=('baolongwang', 'boss', 'dandao_path', 'unique'), unlock='进入巢穴即触发BOSS战', qi_affinity=-0.8, danger_bias=5),
    ),
)
