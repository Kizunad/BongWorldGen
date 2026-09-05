from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from bluemap import _default_spawn_window, _validate_render_args, build_parser  # noqa: E402
from bong_worldgen.bluemap_entities import write_villager_preview_model  # noqa: E402


@pytest.mark.parametrize(
    ("width", "height"),
    ((0, 16), (16, 0), (-16, 16), (16, -16)),
)
def test_bluemap_rejects_non_positive_dimensions_before_render_cleanup(
    width: int,
    height: int,
) -> None:
    with pytest.raises(SystemExit, match="must be positive"):
        _validate_render_args(
            Namespace(
                width=width,
                height=height,
                origin_x=0,
                origin_z=0,
            )
        )


def test_bluemap_defaults_to_the_generated_spawn_window() -> None:
    args = build_parser().parse_args(["render"])
    expected_x, expected_z = _default_spawn_window(tile_size=1024, seed=812731)

    assert (args.origin_x, args.origin_z) == (expected_x, expected_z)
    assert args.origin_x % 16 == 0
    assert args.origin_z % 16 == 0
    assert args.width == 1024
    assert args.height == 1024


def test_villager_model_resolves_texture_and_keeps_feet_at_entity_origin(tmp_path: Path) -> None:
    write_villager_preview_model(tmp_path)
    state = json.loads((tmp_path / "assets/minecraft/entitystates/villager.json").read_text())
    model_id = state["parts"][0]["model"]
    namespace, name = model_id.split(":")
    model = json.loads((tmp_path / "assets" / namespace / "models" / f"{name}.json").read_text())
    assert model["textures"]["skin"] == "minecraft:entity/villager/villager"
    elements = model["elements"]
    assert min(element["from"][1] for element in elements) == 0.0
    assert max(element["to"][1] for element in elements) / 16 == pytest.approx(1.95)
    for element in elements:
        assert all(low < high for low, high in zip(element["from"], element["to"]))
        for face in element["faces"].values():
            assert face["texture"] == "#skin"
            assert all(0 <= coordinate <= 16 for coordinate in face["uv"])
