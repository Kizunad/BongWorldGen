"""生成本地地形预览使用的最小 BlueMap 配置。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import zlib

from .bluemap_entities import write_villager_preview_model


# BlueMap 的原版 grass_block 顶面纹理是灰度图，正常情况下由 biome
# colormap 着色；无服务器运行时的 Anvil 预览无法提供该色图上下文。
# 覆盖包只改变 BlueMap 的预览资源，不改变 Anvil 中的真实 Minecraft 方块。
_PREVIEW_RESOURCE_PACK_NAME = "99_bong_worldgen_preview"
_PREVIEW_RESOURCE_PACK_META = {
    "pack": {
        "pack_format": 15,
        "description": "Bong WorldGen BlueMap preview colors",
    }
}
_PREVIEW_BLOCK_COLORS = {
    "minecraft:grass_block": "#6f9f5e",
    "minecraft:grass": "#6f9f5e",
    "minecraft:short_grass": "#6f9f5e",
    "minecraft:tall_grass": "#6f9f5e",
}


def _solid_png(*, width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """用标准库生成无额外依赖的 RGB PNG，供 BlueMap 预览资源包使用。"""

    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    if any(not 0 <= channel <= 255 for channel in rgb):
        raise ValueError("PNG color channels must be in [0, 255]")

    # PNG 每行前的 0 表示无滤波；16x16 固色纹理足够表达草地方块，
    # 同时避免让 BlueMap 预览工具额外依赖 Pillow。
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


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
    start_y: int = 220
    start_distance: float | None = None
    port: int = 8100
    accept_download: bool = False

    def __post_init__(self) -> None:
        if self.min_x > self.max_x or self.min_z > self.max_z:
            raise ValueError("BlueMap bounds must be ordered")
        if self.start_distance is not None and self.start_distance <= 0:
            raise ValueError("BlueMap start distance must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("BlueMap web port must be in [1, 65535]")


def _quoted(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def _write_preview_resource_pack(config_dir: Path) -> None:
    """写入 BlueMap CLI 识别的本地资源包，避免草地渲染成灰色石头。"""

    # BlueMap CLI 会扫描 config/packs 下的目录和 zip；使用固定目录名，
    # 让每次 render 都能幂等重建，而无需把资源包加入生成世界。
    pack_dir = config_dir / "packs" / _PREVIEW_RESOURCE_PACK_NAME
    (pack_dir / "assets" / "minecraft").mkdir(parents=True, exist_ok=True)
    (pack_dir / "pack.mcmeta").write_text(
        json.dumps(_PREVIEW_RESOURCE_PACK_META, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pack_dir / "assets" / "minecraft" / "blockColors.json").write_text(
        json.dumps(_PREVIEW_BLOCK_COLORS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 原版顶面是灰度纹理；BlueMap 5.23 在离线 Anvil 世界中不会自动套用
    # 生物群系色图，因此直接覆盖顶面纹理。真实 Minecraft 世界仍使用原版资源。
    texture_dir = pack_dir / "assets" / "minecraft" / "textures" / "block"
    texture_dir.mkdir(parents=True, exist_ok=True)
    (texture_dir / "grass_block_top.png").write_bytes(
        _solid_png(width=16, height=16, rgb=(111, 159, 94))
    )
    write_villager_preview_model(pack_dir)


def write_bluemap_config(config: BlueMapConfig) -> None:
    """为一个有边界的主世界地图替换生成的配置文件。"""

    # 视角距离随静态渲染窗口缩放；否则大窗口仍会以固定近景打开，
    # 用户会误以为只生成了当前坐标附近的一小块地图。
    window_span = max(config.max_x - config.min_x, config.max_z - config.min_z)
    start_distance = config.start_distance or max(420.0, window_span * 1.35)

    maps_dir = config.config_dir / "maps"
    storages_dir = config.config_dir / "storages"
    maps_dir.mkdir(parents=True, exist_ok=True)
    storages_dir.mkdir(parents=True, exist_ok=True)
    _write_preview_resource_pack(config.config_dir)

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
                    f"{config.start_x}:{config.start_y}:{config.start_z}:"
                    f"{start_distance:g}:0.1:0.19:0:0:perspective\""
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
                "ambient-light: 1.0",
                # BlueMap 文档约定：极低值表示不移除洞穴；否则 56 格深的浅层洞会被隐藏。
                "remove-caves-below-y: -10000",
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
