#!/usr/bin/env python3
"""Install, render and serve the local BlueMap terrain preview."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bong_worldgen.adapters import export_minecraft_world  # noqa: E402
from bong_worldgen.bluemap_config import BlueMapConfig, write_bluemap_config  # noqa: E402
from bong_worldgen.data.recipes import DEFAULT_RECIPE  # noqa: E402
from bong_worldgen.composition import ZoneTerrain  # noqa: E402


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
    field = ZoneTerrain(seed=args.seed).generate(
        width=args.width,
        height=args.height,
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
    )
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
            port=args.port,
            accept_download=True,
        )
    )
    print(
        f"Generated {result.chunks_written} chunks in {result.regions_written} regions at "
        f"{WORLD_DIR}"
    )
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
    render_parser.add_argument("--width", type=int, default=512)
    render_parser.add_argument("--height", type=int, default=512)
    render_parser.add_argument("--origin-x", type=int, default=-960)
    render_parser.add_argument("--origin-z", type=int, default=-2272)
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
