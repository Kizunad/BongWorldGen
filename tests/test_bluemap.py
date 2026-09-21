from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from bluemap import _validate_render_args, build_parser  # noqa: E402
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config


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


def test_bluemap_cutaway_limits_rendering_and_keeps_outputs_separate(tmp_path):
    args = build_parser().parse_args([
        "render", "--width", "32", "--height", "48", "--origin-x", "5568",
        "--origin-z", "2064", "--min-y", "-64", "--max-y", "-38",
        "--output", str(tmp_path / "cutaway"),
    ])
    _validate_render_args(args)
    config_dir = args.output / "config"
    config = BlueMapConfig(
        config_dir=config_dir, data_dir=tmp_path / "data", web_dir=args.output / "web",
        world_dir=args.output / "world", min_x=5568, max_x=5599, min_z=2064, max_z=2111,
        start_x=5584, start_z=2080, min_y=args.min_y, max_y=args.max_y,
        remove_caves_below_y=-64,
    )
    write_bluemap_config(config)
    contents = (config_dir / "maps/bong.conf").read_text()
    assert "min-y: -64" in contents and "max-y: -38" in contents
    assert str((args.output / "world").resolve()) in contents
    assert str((args.output / "web/maps").resolve()) in (config_dir / "storages/file.conf").read_text()
    assert not (args.output / "world").exists()  # A render mask does not rewrite world blocks.


def test_bluemap_rejects_inverted_cutaway_bounds_before_cleanup():
    args = build_parser().parse_args(["render", "--min-y", "30", "--max-y", "-38"])
    with pytest.raises(SystemExit, match="must not exceed"):
        _validate_render_args(args)
