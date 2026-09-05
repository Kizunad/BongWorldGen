"""生成配方的共用输入校验。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from .constants import CAVE_RARITY_MULTIPLIERS


def normalize_material_names(
    values: Iterable[str],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """规范化 Minecraft 方块名，并拒绝空值和重复值。"""

    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a sequence of names")
    normalized = tuple(
        value.strip().lower().removeprefix("minecraft:").replace("-", "_")
        for value in values
    )
    if (not allow_empty and not normalized) or any(not value for value in normalized):
        raise ValueError(f"{field_name} must contain at least one named material")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def validate_rarities(
    materials: tuple[str, ...],
    rarities: Iterable[str],
    field_name: str,
) -> tuple[Literal["少", "中", "多"], ...]:
    """校验与材料一一对应的中文稀有度。"""

    normalized = tuple(rarities)
    if len(materials) != len(normalized) or any(
        rarity not in CAVE_RARITY_MULTIPLIERS for rarity in normalized
    ):
        raise ValueError(f"{field_name} must match materials and use 少/中/多")
    return normalized  # type: ignore[return-value]


__all__ = ["normalize_material_names", "validate_rarities"]
