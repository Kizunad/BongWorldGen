"""All authored zones in deterministic source order."""

from .spawn import ZONE as SPAWN
from .qingyun_peaks import ZONE as QINGYUN_PEAKS
from .lingquan_marsh import ZONE as LINGQUAN_MARSH
from .blood_valley_east_scorch import ZONE as BLOOD_VALLEY_EAST_SCORCH
from .north_waste_east_scorch import ZONE as NORTH_WASTE_EAST_SCORCH
from .drift_scorch_001 import ZONE as DRIFT_SCORCH_001
from .rift_mouth_north_001 import ZONE as RIFT_MOUTH_NORTH_001
from .rift_mouth_north_002 import ZONE as RIFT_MOUTH_NORTH_002
from .rift_mouth_blood_001 import ZONE as RIFT_MOUTH_BLOOD_001
from .rift_mouth_west_001 import ZONE as RIFT_MOUTH_WEST_001
from .blood_valley import ZONE as BLOOD_VALLEY
from .youan_depths import ZONE as YOUAN_DEPTHS
from .north_wastes import ZONE as NORTH_WASTES
from .celestial_isles import ZONE as CELESTIAL_ISLES
from .south_ash_dead_zone import ZONE as SOUTH_ASH_DEAD_ZONE
from .zhanhun_plain import ZONE as ZHANHUN_PLAIN
from .wuxing_abyss import ZONE as WUXING_ABYSS
from .jiuzong_bloodstream_ruin import ZONE as JIUZONG_BLOODSTREAM_RUIN
from .jiuzong_beiling_ruin import ZONE as JIUZONG_BEILING_RUIN
from .jiuzong_nanyuan_ruin import ZONE as JIUZONG_NANYUAN_RUIN
from .jiuzong_chixia_ruin import ZONE as JIUZONG_CHIXIA_RUIN
from .jiuzong_xuanshui_ruin import ZONE as JIUZONG_XUANSHUI_RUIN
from .jiuzong_taichu_ruin import ZONE as JIUZONG_TAICHU_RUIN
from .jiuzong_youan_ruin import ZONE as JIUZONG_YOUAN_RUIN
from .dan_zong_yi_yuan import ZONE as DAN_ZONG_YI_YUAN
from .wangyintai import ZONE as WANGYINTAI
from .baolongwang_cavern_deep import ZONE as BAOLONGWANG_CAVERN_DEEP

ALL_ZONES = (
    SPAWN,
    QINGYUN_PEAKS,
    LINGQUAN_MARSH,
    BLOOD_VALLEY_EAST_SCORCH,
    NORTH_WASTE_EAST_SCORCH,
    DRIFT_SCORCH_001,
    RIFT_MOUTH_NORTH_001,
    RIFT_MOUTH_NORTH_002,
    RIFT_MOUTH_BLOOD_001,
    RIFT_MOUTH_WEST_001,
    BLOOD_VALLEY,
    YOUAN_DEPTHS,
    NORTH_WASTES,
    CELESTIAL_ISLES,
    SOUTH_ASH_DEAD_ZONE,
    ZHANHUN_PLAIN,
    WUXING_ABYSS,
    JIUZONG_BLOODSTREAM_RUIN,
    JIUZONG_BEILING_RUIN,
    JIUZONG_NANYUAN_RUIN,
    JIUZONG_CHIXIA_RUIN,
    JIUZONG_XUANSHUI_RUIN,
    JIUZONG_TAICHU_RUIN,
    JIUZONG_YOUAN_RUIN,
    DAN_ZONG_YI_YUAN,
    WANGYINTAI,
    BAOLONGWANG_CAVERN_DEEP,
)

ZONE_BY_NAME = {zone.name: zone for zone in ALL_ZONES}
