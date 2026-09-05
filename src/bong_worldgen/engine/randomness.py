"""地形引擎共享的确定性随机函数。

所有生成器都必须通过这里把世界 seed 映射到稳定的 64 位值，避免各模块
各自复制一套混合算法后出现行为漂移。函数不维护全局状态，因此同一个
``(seed, index)`` 在分块生成和整图生成中始终得到相同结果。
"""

from __future__ import annotations


_MASK_64 = (1 << 64) - 1
_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15


def mix64(value: int) -> int:
    """把任意整数稳定地混合为无符号 64 位值。"""

    value &= _MASK_64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK_64
    value ^= value >> 31
    return value & _MASK_64


def unit_interval(seed: int, index: int) -> float:
    """返回 ``[0, 1)`` 内的无状态确定性随机数。"""

    mixed = mix64(seed + index * _GOLDEN_RATIO_64)
    return float(mixed >> 11) / float(1 << 53)


def stable_text_seed(value: str) -> int:
    """把文本转换为跨进程稳定的整数 seed。

    Python 的内置 ``hash`` 默认会在每个进程中随机化，不能用于需要按
    world seed 分块重建的地形数据。这里保留简单、可审计的字符加权规则，
    供所有子系统统一使用。
    """

    return sum((index + 1) * ord(char) for index, char in enumerate(value))


__all__ = ["mix64", "stable_text_seed", "unit_interval"]
