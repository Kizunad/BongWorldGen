"""Guard the evidence tool against falsely certifying disconnected cavities."""

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from zone_cave_evidence import air_intervals, connected_intervals  # noqa: E402


@pytest.mark.parametrize("gate,reachable", (((2, 3), True), ((2, 2), False), ((3, 4), False)))
def test_interval_evidence_agrees_with_six_neighbor_voxels(gate, reachable):
    # The middle column either leaves two-block headroom, only one block, or
    # touches adjacent cavities at head height without a possible horizontal step.
    surface = np.full((1, 5), 10.0)
    spans = np.full((1, 5, 4, 2), 32767, dtype=np.int16)
    for x in range(5):
        low, high = gate if x == 2 else (1, 3)
        spans[0, x, :2] = ((high + 1, 10), (-64, low - 1))
    lower, upper = air_intervals(surface, spans)
    connected = connected_intervals(lower, upper, (0, 0, 1))
    assert bool(connected[0, 4, 1]) is reachable

    # Independent voxel flood: exclude the sky even though the real world
    # would let it bypass any underground obstacle by going over the surface.
    y = np.arange(-63, 12)[:, None, None]
    air = (y <= surface[None, ...]) & ~np.any(
        (y[..., None] >= spans[None, ..., 0]) & (y[..., None] <= spans[None, ..., 1]), axis=-1)
    clearance = air[:-1] & air[1:]
    visited = np.zeros_like(clearance)
    start = (64, 0, 0)  # Feet Y = 1.
    visited[start] = True
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = list(point)
                neighbor[axis] += delta
                if not 0 <= neighbor[axis] < clearance.shape[axis]:
                    continue
                neighbor = tuple(neighbor)
                if clearance[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    expanded = np.any(connected[None, ...] & (y[:-1, ..., None] >= lower[None, ...])
                      & (y[:-1, ..., None] <= upper[None, ...]), axis=-1)
    np.testing.assert_array_equal(expanded, visited)
