"""冰碛和鼓丘的沉积高度场。"""

from __future__ import annotations

import math

import numpy as np

from ..terrain_config import GlacialSystem, Point
from ..randomness import stable_text_seed, unit_interval


def deposit_terminal_moraine(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
    system: GlacialSystem,
) -> np.ndarray:
    """在冰川末端放置横跨谷口的终碛脊。"""

    return terrain + terminal_moraine_deposition(x, z, path, system)


def terminal_moraine_deposition(
    x: np.ndarray,
    z: np.ndarray,
    path: tuple[Point, ...],
    system: GlacialSystem,
) -> np.ndarray:
    """返回终碛脊在每个地表列堆积的连续高度。"""

    if len(path) < 2 or system.moraine_height <= 0:
        return np.zeros(x.shape, dtype=np.float64)
    previous, endpoint = path[-2], path[-1]
    dx = endpoint.x - previous.x
    dz = endpoint.z - previous.z
    length = max(math.hypot(dx, dz), 1.0e-6)
    dx /= length
    dz /= length
    local_x = x - endpoint.x
    local_z = z - endpoint.z
    along = local_x * dx + local_z * dz
    across = -local_x * dz + local_z * dx
    ridge = np.exp(
        -0.5
        * (
            (along / system.moraine_length) ** 2
            + (across / system.moraine_width) ** 2
        )
    )
    return system.moraine_height * ridge


def _drumlin_centers(system: GlacialSystem, seed: int) -> tuple[tuple[Point, float], ...]:
    """生成鼓丘中心和冰流方向；方向有小幅随机偏转。"""

    name_seed = stable_text_seed(system.name)
    centers: list[tuple[Point, float]] = []
    for index in range(system.drumlin_count):
        base = name_seed + 90_000 + index * 19_007
        radius = system.seed_extent * (0.12 + 0.78 * unit_interval(seed, base + 1))
        angle = unit_interval(seed, base + 2) * math.tau
        center = Point(
            system.seed_center.x + math.cos(angle) * radius,
            system.seed_center.z + math.sin(angle) * radius,
        )
        flow_angle = math.pi * 0.5 + (unit_interval(seed, base + 3) - 0.5) * math.pi * 0.30
        centers.append((center, flow_angle))
    return tuple(centers)


def deposit_drumlins(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    system: GlacialSystem,
    seed: int,
    sea_level: float,
) -> np.ndarray:
    """在低地放置沿冰流方向拉长的鼓丘群。

    椭圆高斯是鼓丘的可控近似；真实冰流模拟可在后续替换此模块。参考：
    https://github.com/oargudo/glaciers
    """

    return terrain + drumlin_deposition(terrain, x, z, system, seed, sea_level)


def drumlin_deposition(
    terrain: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    system: GlacialSystem,
    seed: int,
    sea_level: float,
) -> np.ndarray:
    """返回鼓丘群的总沉积高度，并保持原有逐丘低地门控行为。"""

    output = np.asarray(terrain, dtype=np.float64).copy()
    deposited = np.zeros(terrain.shape, dtype=np.float64)
    if system.drumlin_height <= 0:
        return deposited
    for center, flow_angle in _drumlin_centers(system, seed):
        dx = x - center.x
        dz = z - center.z
        along = dx * math.cos(flow_angle) + dz * math.sin(flow_angle)
        across = -dx * math.sin(flow_angle) + dz * math.cos(flow_angle)
        ridge = np.exp(
            -0.5
            * (
                (along / system.drumlin_length) ** 2
                + (across / system.drumlin_width) ** 2
            )
        )
        lowland_gate = 1.0 - np.clip(
            (output - (sea_level + system.drumlin_max_elevation))
            / max(system.drumlin_max_elevation, 1.0),
            0.0,
            1.0,
        )
        delta = system.drumlin_height * ridge * lowland_gate
        output += delta
        deposited += delta
    return deposited


__all__ = [
    "deposit_drumlins",
    "deposit_terminal_moraine",
    "drumlin_deposition",
    "terminal_moraine_deposition",
]
