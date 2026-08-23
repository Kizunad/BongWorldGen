"""洞穴空腔到垂直实心段的转换。

控制台和 Minecraft 适配器都需要同一个 top-first spans 合同，因此这部分
不应该和具体的洞穴噪声或矿厅装饰耦合。
"""

from __future__ import annotations

import numpy as np

from ..models import SPAN_MAX_SPANS, SPAN_MIN_Y


def build_solid_spans(
    terrain: np.ndarray,
    cave_void: np.ndarray,
    cave_offsets: np.ndarray,
) -> np.ndarray:
    """把三维空腔掩码折叠为每列最多四段的实心范围。

    ``cave_void`` 的第一维对应 ``cave_offsets``，每个空腔连续区间会从
    地表下方切出。输出按顶部到基岩方向排列，空槽使用 ``32767`` 哨兵。
    """

    height, width = terrain.shape
    sentinel = np.int16(32767)
    spans = np.full((height, width, SPAN_MAX_SPANS, 2), sentinel, dtype=np.int16)
    surface = np.rint(terrain).astype(np.int16)

    # 没有空腔时所有列都是一段从基岩到地表的实心范围。这个分支同时
    # 覆盖无洞穴配方和洞穴范围之外的 tile，避免为每一列启动 Python
    # 区间扫描循环。
    if cave_void.size == 0 or not np.any(cave_void):
        spans[:, :, 0] = np.stack(
            (np.full((height, width), SPAN_MIN_Y, dtype=np.int16), surface),
            axis=-1,
        )
        return spans

    for z_index in range(height):
        for x_index in range(width):
            void_levels = np.flatnonzero(cave_void[:, z_index, x_index])
            intervals: list[tuple[int, int]] = []
            if void_levels.size:
                run_start = int(void_levels[0])
                previous = run_start
                for level in void_levels[1:]:
                    level = int(level)
                    if level != previous + 1:
                        intervals.append(
                            (
                                int(surface[z_index, x_index] + cave_offsets[run_start]),
                                int(surface[z_index, x_index] + cave_offsets[previous]),
                            )
                        )
                        run_start = level
                    previous = level
                intervals.append(
                    (
                        int(surface[z_index, x_index] + cave_offsets[run_start]),
                        int(surface[z_index, x_index] + cave_offsets[previous]),
                    )
                )

            cursor = int(surface[z_index, x_index])
            solid: list[tuple[int, int]] = []
            for floor, ceiling in reversed(intervals):
                if cursor >= ceiling + 1:
                    solid.append((ceiling + 1, cursor))
                cursor = min(cursor, floor - 1)
            if cursor >= SPAN_MIN_Y:
                solid.append((SPAN_MIN_Y, cursor))

            if len(solid) > SPAN_MAX_SPANS:
                raise ValueError(
                    "洞穴布局超过四段实心范围合同，请减少重叠隧道"
                )
            for slot, (floor, ceiling) in enumerate(solid):
                spans[z_index, x_index, slot] = (floor, ceiling)
    return spans
