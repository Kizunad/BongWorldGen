"""Generate the small BlueMap configuration used by local terrain previews."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BlueMapConfig:
    config_dir: Path
    data_dir: Path
    web_dir: Path
    world_dir: Path
    min_x: int
    max_x: int
    min_z: int
    max_z: int
    start_x: int
    start_z: int
    port: int = 8100
    accept_download: bool = False

    def __post_init__(self) -> None:
        if self.min_x > self.max_x or self.min_z > self.max_z:
            raise ValueError("BlueMap bounds must be ordered")
        if not 1 <= self.port <= 65535:
            raise ValueError("BlueMap web port must be in [1, 65535]")


def _quoted(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def write_bluemap_config(config: BlueMapConfig) -> None:
    """Replace the generated config files for one bounded overworld map."""

    maps_dir = config.config_dir / "maps"
    storages_dir = config.config_dir / "storages"
    maps_dir.mkdir(parents=True, exist_ok=True)
    storages_dir.mkdir(parents=True, exist_ok=True)

    (config.config_dir / "core.conf").write_text(
        "\n".join(
            (
                f"accept-download: {str(config.accept_download).lower()}",
                f"data: {_quoted(config.data_dir)}",
                "render-thread-count: 4",
                "scan-for-mod-resources: false",
                "metrics: false",
                "",
            )
        ),
        encoding="utf-8",
    )
    (storages_dir / "file.conf").write_text(
        "\n".join(
            (
                "storage-type: file",
                f"root: {_quoted(config.web_dir / 'maps')}",
                "compression: gzip",
                "",
            )
        ),
        encoding="utf-8",
    )
    (config.config_dir / "webapp.conf").write_text(
        "\n".join(
            (
                "enabled: true",
                f"webroot: {_quoted(config.web_dir)}",
                "update-settings-file: true",
                "default-to-flat-view: false",
                (
                    'start-location: "bong:'
                    f"{config.start_x}:80:{config.start_z}:420:0.1:0.19:0:0:perspective\""
                ),
                "min-zoom-distance: 5",
                "max-zoom-distance: 100000",
                "resolution-default: 1",
                "hires-slider-max: 800",
                "hires-slider-default: 180",
                "hires-slider-min: 0",
                "lowres-slider-max: 7000",
                "lowres-slider-default: 2000",
                "lowres-slider-min: 250",
                "",
            )
        ),
        encoding="utf-8",
    )
    (config.config_dir / "webserver.conf").write_text(
        "\n".join(
            (
                "enabled: true",
                f"webroot: {_quoted(config.web_dir)}",
                f"port: {config.port}",
                "sse-enabled: true",
                "",
            )
        ),
        encoding="utf-8",
    )
    (maps_dir / "bong.conf").write_text(
        "\n".join(
            (
                f"world: {_quoted(config.world_dir)}",
                'dimension: "minecraft:overworld"',
                'name: "Bong WorldGen"',
                "sorting: 0",
                f"start-pos: {{ x: {config.start_x}, z: {config.start_z} }}",
                'sky-color: "#8fb9d9"',
                'void-color: "#202428"',
                "sky-light: 1",
                "ambient-light: 0.15",
                "remove-caves-below-y: 55",
                "cave-detection-ocean-floor: -5",
                "min-inhabited-time: 0",
                "render-mask: [",
                "  {",
                f"    min-x: {config.min_x}",
                f"    max-x: {config.max_x}",
                f"    min-z: {config.min_z}",
                f"    max-z: {config.max_z}",
                "  }",
                "]",
                "render-edges: true",
                "edge-light-strength: 12",
                "enable-perspective-view: true",
                "enable-flat-view: true",
                "enable-free-flight-view: true",
                "enable-hires: true",
                'storage: "file"',
                "ignore-missing-light-data: true",
                "marker-sets: {}",
                "",
            )
        ),
        encoding="utf-8",
    )


__all__ = ["BlueMapConfig", "write_bluemap_config"]
