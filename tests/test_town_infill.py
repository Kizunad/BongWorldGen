"""核心补位不能因随机抽点漏掉窄空隙，也不能绕过完整占地检查。"""

import numpy as np
import pytest

from bong_worldgen.engine.terrain_config import Point, TownSettings
from bong_worldgen.engine.town_footprints import WallFootprint
from bong_worldgen.engine.town_growth import (
    GrowthBuilding,
    _choose_dense_infill_candidate,
    building_footprint_bounds,
)


def _building(left, right, top, bottom):
    width, length = right - left + 1, bottom - top + 1
    return GrowthBuilding(
        Point(left + width // 2, top + length // 2), width, length, 80, None, 0, "core",
    )


def _infill_case(width, length, left=-17, top=13):
    """四栋建筑围出恰好容纳一栋模板的空位；随机下界落在已占用处。"""

    right, bottom = left + width - 1, top + length - 1
    buildings = [
        _building(left - 8, left - 1, top - 8, bottom + 8),
        _building(right + 1, right + 8, top - 8, bottom + 8),
        _building(left, right, top - 8, top - 1),
        _building(left, right, bottom + 1, bottom + 8),
    ]
    axis = np.arange(-64, 65, dtype=float)
    terrain = np.full((axis.size, axis.size), 80.0)
    return dict(
        terrain=terrain, water=np.full_like(terrain, -1.0), x_axis=axis, z_axis=axis,
        settings=TownSettings(growth_candidate_count=1), buildings=buildings,
        size_x=width, size_z=length, seed=812731, index=10022,
    )


@pytest.mark.parametrize(("width", "length"), [(5, 7), (6, 10), (6, 9)])
@pytest.mark.parametrize(("left", "top"), [(-17, 13), (11, -23)])
def test_infill_finds_exact_gap_after_random_candidates_miss(
    monkeypatch, width, length, left, top,
):
    monkeypatch.setattr("bong_worldgen.engine.town_growth.unit_interval", lambda *_args: 0.0)
    case = _infill_case(width, length, left, top)
    original_bounds = building_footprint_bounds(case["buildings"])

    candidate = _choose_dense_infill_candidate(**case)

    assert candidate == (Point(left + width // 2, top + length // 2), 80)
    center, ground = candidate
    filled = [*case["buildings"], GrowthBuilding(center, width, length, ground, None, 0, "core")]
    assert building_footprint_bounds(filled) == original_bounds
    assert len(case["buildings"]) == 4


@pytest.mark.parametrize("obstacle", ["water", "slope", "wall", "crop", "building"])
def test_infill_fallback_keeps_terrain_and_footprint_constraints(monkeypatch, obstacle):
    monkeypatch.setattr("bong_worldgen.engine.town_growth.unit_interval", lambda *_args: 0.0)
    case = _infill_case(6, 10)
    left, right, top, bottom = -17, -12, 13, 22
    if obstacle == "water":
        case["water"][top + 64, left + 64] = 80
    elif obstacle == "slope":
        case["terrain"][top + 64:bottom + 65, left + 64:right + 65] += np.arange(6) * 0.5
    elif obstacle == "wall":
        case["walls"] = WallFootprint(wall_span=1, wall_thickness=1, margin=2)
        min_x, _, min_z, _ = building_footprint_bounds(case["buildings"])
        case["water"][min_z - 2 + 64, min_x - 2 + 64] = 80
    elif obstacle == "crop":
        first = left + 65
        case["x_axis"] = case["x_axis"][first:]
        case["terrain"] = case["terrain"][:, first:]
        case["water"] = case["water"][:, first:]
    else:
        case["buildings"].append(_building(left, right, top, bottom))

    assert _choose_dense_infill_candidate(**case) is None
