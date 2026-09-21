"""Guard the evidence tool against falsely certifying disconnected cavities."""

from collections import deque
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import zone_cave_evidence as evidence  # noqa: E402
from zone_cave_evidence import air_intervals, connected_intervals, tile_intervals  # noqa: E402
from bong_worldgen.composition import ZoneTerrain
from bong_worldgen.engine import CaveNetwork, Point, TerrainRecipe
from bong_worldgen.data.world_definition import WORLD


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


def test_checkpoint_reuses_geometry_and_recovers_from_interrupted_tiles(tmp_path):
    cave = CaveNetwork(name="test", paths=((Point(-20, 0), Point(20, 0)),),
                       depth=30, width=8, height=8, noise_strength=0,
                       branch_count=0, chamber_count=0, entrance_count=0)
    composer = ZoneTerrain(replace(WORLD, zones=()),
                           background=TerrainRecipe(name="test", caves=(cave,)))
    cache = tmp_path / evidence.generation_fingerprint(composer, 8)
    expected = tile_intervals(composer, -4, -4, 8, None)
    assert np.any(expected[0] <= expected[1])  # Actual cave air, not an empty tile.
    for reused in (False, True):
        actual = tile_intervals(composer, -4, -4, 8, cache)
        np.testing.assert_array_equal(actual[:2], expected[:2])
        assert actual[2] is reused
    (cache / "-4_-4.npz").write_bytes(b"truncated checkpoint")
    actual = tile_intervals(composer, -4, -4, 8, cache)
    np.testing.assert_array_equal(actual[:2], expected[:2])
    assert actual[2] is False
    assert tile_intervals(composer, -4, -4, 8, cache)[2] is True


def test_checkpoint_identity_changes_with_world_seed_grid_and_generator(tmp_path, monkeypatch):
    # Simulate a code update without editing the repository's actual generator.
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    source = tmp_path / "src/bong_worldgen/engine.py"
    source.parent.mkdir(parents=True)
    source.write_text("version one\n")
    first = ZoneTerrain(seed=7)
    key = evidence.generation_fingerprint(first, 8)
    assert key != evidence.generation_fingerprint(ZoneTerrain(seed=8), 8)
    assert key != evidence.generation_fingerprint(first, 16)
    assert key != evidence.generation_fingerprint(ZoneTerrain(replace(WORLD, zones=()), seed=7), 8)
    source.write_text("version two\n")
    assert key != evidence.generation_fingerprint(first, 8)
