#!/usr/bin/env python3
"""Install, render and serve the local BlueMap terrain preview."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bong_worldgen.adapters import export_minecraft_world  # noqa: E402
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config  # noqa: E402
from bong_worldgen.data.recipes import DEFAULT_RECIPE  # noqa: E402
from bong_worldgen.engine import generate_heightfield  # noqa: E402
from bong_worldgen.engine.pipeline import (  # noqa: E402
    _generate_pre_glacial_terrain,
    _select_spawn_plain,
)


BLUEMAP_VERSION = "5.23"
BLUEMAP_SHA256 = "ebdb33821b127505be94599555cb16bb9b46c8f70aa22283d314cdb829b47a54"
BLUEMAP_URL = (
    "https://github.com/BlueMap-Minecraft/BlueMap/releases/download/"
    f"v{BLUEMAP_VERSION}/bluemap-{BLUEMAP_VERSION}-cli.jar"
)
JAR_PATH = ROOT / ".tools" / "bluemap" / f"bluemap-{BLUEMAP_VERSION}-cli.jar"
RUNTIME_DIR = ROOT / ".bluemap"
CONFIG_DIR = RUNTIME_DIR / "config"
DATA_DIR = RUNTIME_DIR / "data"
WEB_DIR = RUNTIME_DIR / "web"
WORLD_DIR = ROOT / "generated" / "bluemap-world"


def _default_mountain_window(tile_size: int = 512, seed: int = 812731) -> tuple[int, int]:
    """扫描默认配方的主山脉，推导包含实际峰值的 BlueMap 窗口。"""

    if tile_size < 16 or tile_size % 16:
        raise ValueError("BlueMap preview tile size must be a positive multiple of 16")
    mountains = DEFAULT_RECIPE.mountains
    if not mountains:
        return (-704, -3264)
    # 绝对高度山脉用 summit_elevation 参与选窗；不能因为兼容字段
    # ``height`` 在该模式下为 0，就误选到较小的相对山脊。
    mountain = max(
        mountains,
        key=lambda item: (
            item.summit_elevation
            if item.summit_elevation is not None
            else item.base_elevation
            if item.base_elevation is not None
            else item.height
        ),
    )
    points = mountain.path
    minimum_x = math.floor(min(point.x for point in points) / 32.0) * 32.0
    maximum_x = math.ceil(max(point.x for point in points) / 32.0) * 32.0
    minimum_z = math.floor(min(point.z for point in points) / 32.0) * 32.0
    maximum_z = math.ceil(max(point.z for point in points) / 32.0) * 32.0
    step = 32.0
    sample_x, sample_z = np.meshgrid(
        np.arange(minimum_x, maximum_x + step, step, dtype=np.float64),
        np.arange(minimum_z, maximum_z + step, step, dtype=np.float64),
        indexing="xy",
    )
    sample_height = _generate_pre_glacial_terrain(
        DEFAULT_RECIPE,
        sample_x,
        sample_z,
        seed,
    )
    # 绝对高度模式下，沿整条山脊可能有很多同高的峰。只取全局最高点
    # 会把窗口吸到路径端点，导致看不到山脚；改为在候选峰附近选择
    # ``窗口内最大高差`` 最大的位置，保证预览同时包含山脚和峰顶。
    summit = mountain.summit_elevation
    if summit is None:
        summit = float(np.max(sample_height))
    candidate_mask = sample_height >= summit - max(8.0, abs(summit) * 0.04)
    candidates = np.argwhere(candidate_mask)
    half = tile_size * 0.5
    valid_candidates: list[tuple[float, float, float, float]] = []
    for candidate_z, candidate_x in candidates:
        anchor_x = float(sample_x[candidate_z, candidate_x])
        anchor_z = float(sample_z[candidate_z, candidate_x])
        if not (
            minimum_x + half <= anchor_x <= maximum_x - half
            and minimum_z + half <= anchor_z <= maximum_z - half
        ):
            continue
        window = (
            (np.abs(sample_x - anchor_x) <= half)
            & (np.abs(sample_z - anchor_z) <= half)
        )
        values = sample_height[window]
        if values.size:
            valid_candidates.append(
                (float(values.max() - values.min()), float(values.max()), anchor_x, anchor_z)
            )
    if valid_candidates:
        _, _, anchor_x, anchor_z = max(valid_candidates)
    else:
        peak = np.unravel_index(np.argmax(sample_height), sample_height.shape)
        anchor_x = float(sample_x[peak])
        anchor_z = float(sample_z[peak])
    # 用一个采样步长留出边缘余量，保证峰顶附近的山壁也在窗口内。
    half += step
    return (
        math.floor((anchor_x - half) / 16.0) * 16,
        math.floor((anchor_z - half) / 16.0) * 16,
    )


def _default_spawn_window(tile_size: int = 1024, seed: int = 812731) -> tuple[int, int]:
    """返回围绕 seed 实际出生平原锚点的 BlueMap 窗口左上角。"""

    if tile_size < 16 or tile_size % 16:
        raise ValueError("BlueMap preview tile size must be a positive multiple of 16")
    selection = _select_spawn_plain(DEFAULT_RECIPE, seed)
    half = tile_size * 0.5
    return (
        math.floor((selection.center_x - half) / 16.0) * 16,
        math.floor((selection.center_z - half) / 16.0) * 16,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_bluemap() -> Path:
    if JAR_PATH.is_file():
        actual = _sha256(JAR_PATH)
        if actual != BLUEMAP_SHA256:
            raise RuntimeError(f"BlueMap checksum mismatch: expected {BLUEMAP_SHA256}, got {actual}")
        return JAR_PATH

    JAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = JAR_PATH.with_suffix(".jar.part")
    urllib.request.urlretrieve(BLUEMAP_URL, partial)
    actual = _sha256(partial)
    if actual != BLUEMAP_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"BlueMap checksum mismatch: expected {BLUEMAP_SHA256}, got {actual}")
    partial.replace(JAR_PATH)
    return JAR_PATH


def java_path() -> Path:
    override = os.environ.get("BLUEMAP_JAVA")
    candidates = (
        Path(override) if override else None,
        Path.home() / ".sdkman" / "candidates" / "java" / "25.0.4-tem" / "bin" / "java",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    executable = shutil.which("java")
    if executable:
        return Path(executable)
    raise FileNotFoundError("Java 25 was not found; set BLUEMAP_JAVA to its executable")


def _replace_directory(path: Path) -> None:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError(f"refusing to clear path outside BongWorldGen: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_bluemap(*args: str) -> None:
    command = [
        str(java_path()),
        "-jar",
        str(ensure_bluemap()),
        "--config",
        str(CONFIG_DIR),
        "--mc-version",
        "1.20.1",
        *args,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _validate_render_args(args: argparse.Namespace) -> None:
    """在清理已有预览前验证所有会影响输出范围的参数。"""

    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be positive")
    if args.width % 16 or args.height % 16:
        raise SystemExit("--width and --height must be multiples of 16")
    if args.origin_x % 16 or args.origin_z % 16:
        raise SystemExit("--origin-x and --origin-z must be multiples of 16")


def render(args: argparse.Namespace) -> None:
    _validate_render_args(args)
    if not args.accept_minecraft_eula:
        raise SystemExit(
            "render requires --accept-minecraft-eula: BlueMap downloads Mojang's 1.20.1 "
            "client resources and requires a licensed Java Edition account"
        )

    ensure_bluemap()
    _replace_directory(WORLD_DIR)
    _replace_directory(WEB_DIR)
    _replace_directory(CONFIG_DIR)
    field = generate_heightfield(
        DEFAULT_RECIPE,
        width=args.width,
        height=args.height,
        seed=args.seed,
        origin_x=args.origin_x,
        origin_z=args.origin_z,
        cell_size=1.0,
    )
    result = export_minecraft_world(
        field,
        WORLD_DIR,
        origin_x=args.origin_x,
        origin_z=args.origin_z,
        sea_level=DEFAULT_RECIPE.sea_level,
        seed=args.seed,
        world_name=DEFAULT_RECIPE.name,
        preview_npc_spawns=True,
    )
    start_y = max(160, int(math.ceil(float(np.percentile(field.height, 99.0)) + 36.0)))
    write_bluemap_config(
        BlueMapConfig(
            config_dir=CONFIG_DIR,
            data_dir=DATA_DIR,
            web_dir=WEB_DIR,
            world_dir=WORLD_DIR,
            min_x=args.origin_x,
            max_x=args.origin_x + args.width - 1,
            min_z=args.origin_z,
            max_z=args.origin_z + args.height - 1,
            start_x=args.origin_x + args.width // 2,
            start_z=args.origin_z + args.height // 2,
            start_y=start_y,
            port=args.port,
            accept_download=True,
        )
    )
    print(
        f"Generated {result.chunks_written} chunks in {result.regions_written} regions at "
        f"{WORLD_DIR}"
    )
    print(f"Placed {result.preview_villagers_written} preview villagers at NPC spawn points")
    bridge_decks = [block for block in field.settlement_blocks if block.kind == "bridge_deck"]
    print(f"Placed {len(bridge_decks)} bridge deck blocks")
    if bridge_decks:
        block = bridge_decks[len(bridge_decks) // 2]
        print(f"Bridge preview position: {block.x} {block.y + 1} {block.z}")
    _run_bluemap("--generate-webapp", "--render", "--force-render", "--generate-websettings")


def serve(args: argparse.Namespace) -> None:
    if not (CONFIG_DIR / "maps" / "bong.conf").is_file():
        raise SystemExit("BlueMap config is missing; run the render command first")
    try:
        _run_bluemap("--webserver")
    except KeyboardInterrupt:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download and verify the pinned BlueMap CLI")

    render_parser = commands.add_parser("render", help="replace and render one preview world")
    default_width = 1024
    default_height = 1024
    render_parser.add_argument("--width", type=int, default=default_width)
    render_parser.add_argument("--height", type=int, default=default_height)
    # 默认窗口围绕默认 seed 实际选中的出生平原；命令行仍可以覆盖到任意
    # 世界窗口。坐标按 16 格对齐，便于 Anvil 导出。
    default_origin_x, default_origin_z = _default_spawn_window(
        tile_size=default_width,
        seed=812731,
    )
    render_parser.add_argument("--origin-x", type=int, default=default_origin_x)
    render_parser.add_argument("--origin-z", type=int, default=default_origin_z)
    render_parser.add_argument("--seed", type=int, default=812731)
    render_parser.add_argument("--port", type=int, default=8100)
    render_parser.add_argument("--accept-minecraft-eula", action="store_true")

    commands.add_parser("serve", help="serve the most recently rendered map")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "setup":
        print(f"BlueMap {BLUEMAP_VERSION}: {ensure_bluemap()}")
    elif args.command == "render":
        render(args)
    else:
        serve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
