from collections import deque

import numpy as np
import pytest

from bong_worldgen.composition import ZoneTerrain


@pytest.mark.parametrize("seed", (7, 812731, 2026))
def test_bottomless_shaft_connects_to_three_underground_levels(seed):
    # Real Wuxing abyss well: the old compiler carved a shaft 305 blocks from
    # the nearest authored path, leaving rock between the well and the maze.
    composer = ZoneTerrain(seed=seed)
    field = composer.generate(width=32, height=48, origin_x=5576, origin_z=2060)
    bottom = -63
    top = int(np.ceil(field.height.max()))
    levels = np.arange(bottom, top + 2)[:, None, None]
    air = levels <= np.floor(field.height)[None, ...]
    for slot in range(4):
        floor, ceiling = field.solid_spans[..., slot, 0], field.solid_spans[..., slot, 1]
        air &= ~((levels >= floor) & (levels <= ceiling))
    # Feet and head must both fit: a one-block fissure is not a usable passage.
    clearance = air[:-1] & air[1:]
    shaft_z, shaft_x = 40, 24  # World (5600, 2100).
    surface = int(np.floor(field.height[shaft_z, shaft_x]))
    start = (surface - 112 - bottom, shaft_z, shaft_x)
    assert clearance[start]

    seen = np.zeros_like(clearance)
    seen[start] = True
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for axis in range(3):
            for delta in (-1, 1):
                adjacent = list(point)
                adjacent[axis] += delta
                if not 0 <= adjacent[axis] < clearance.shape[axis]:
                    continue
                neighbor = tuple(adjacent)
                if clearance[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)

    # A separate, roofed opening must be reachable on every level 40 blocks
    # toward the main entrance, well beyond the shaft's four-block radius.
    target_z, target_x = 0, 4  # World (5580, 2060).
    column = field.solid_spans[target_z, target_x]
    assert np.count_nonzero(column[:, 0] != 32767) == 4
    surface = int(np.floor(field.height[target_z, target_x]))
    for depth in (34, 72, 112):
        center = surface - depth - bottom
        assert seen[center - 3:center + 4, target_z, target_x].any(), (seed, depth)
