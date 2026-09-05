"""由噪声零等值线生成可复现的地下裂隙带。

裂隙场的核心是 ``abs(noise)``：噪声场的零等值线满足 ``noise == 0``，
因此 ``abs(noise)`` 越小，越接近一条连续的裂缝。这里把它转换为 [0, 1]
的独立裂隙权重，再体素化成不带水的窄缝空腔。地下河只在自己的水文代价场
上寻路，二者可以在最终 spans 中相交，但不会共享同一条中心线。

参考了公开的程序化洞穴思路（只实现数学原语，不复制代码）：

* line-based Simplex 洞穴：
  https://github.com/sachPico/lineBasedCave_Simplex
* Godot Voxel 的噪声洞穴/体积场：
  https://github.com/Zylann/godot_voxel/blob/master/doc/source/procedural_generation.md
* FastNoiseLite 的 seeded noise 工作流：
  https://github.com/Auburn/FastNoiseLite

``zero_isoline_band`` 不依赖 scipy，保持 worldgen 的 NumPy-only 运行时契约。
"""

from __future__ import annotations

import numpy as np


def zero_isoline_band(field: np.ndarray, width: float) -> np.ndarray:
    """把标量噪声的零等值线变成平滑的裂缝权重。

    ``width`` 是噪声值域中的半宽，而不是世界坐标中的方块宽度。结果在
    零等值线处为 1，距离达到 ``width`` 后为 0。使用 smoothstep 让裂隙
    边界平滑，体素化时不会出现突兀的锯齿带。
    """

    if width <= 0.0:
        raise ValueError("zero-isoline width must be positive")
    distance = np.clip(np.abs(np.asarray(field, dtype=np.float64)) / width, 0.0, 1.0)
    # 1 - smoothstep(distance)：中心线最便宜，带外不改变原代价。
    smooth = distance * distance * (3.0 - 2.0 * distance)
    return 1.0 - smooth


__all__ = ["zero_isoline_band"]
