"""洞穴生成子系统。

洞穴逻辑按职责拆分为：

* ``generator``：根据配方生成洞穴空腔和洞穴分类；
* ``spans``：把三维空腔折叠为服务端可消费的垂直实心段；
* ``structures``：在洞腔和洞壁上生成真实 Minecraft 方块刷新点；不生成人工矿厅。
* ``worms``：实现 Godot Voxel 风格的 2D worm、死胡同和垂直扰动场。
* ``topology``：从 seed 和锚点路径生成确定性的支洞与洞室节点。
"""

from .generator import UndergroundResult, generate_underground
from .spans import build_solid_spans
from .structures import generate_placeholder_blocks
from .topology import CaveTopology, generate_cave_topology
from .worms import WormField, sample_worm_field

__all__ = [
    "UndergroundResult",
    "build_solid_spans",
    "generate_placeholder_blocks",
    "generate_underground",
    "CaveTopology",
    "generate_cave_topology",
    "sample_worm_field",
    "WormField",
]
