from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from bluemap import _validate_render_args  # noqa: E402


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
