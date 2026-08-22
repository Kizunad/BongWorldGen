"""Convert a generated heightfield into a bounded Minecraft Anvil world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

import numpy as np

from ..data.wilderness import LAKE, MOUNTAINS, RIVER
from ..engine.models import Heightfield
from .anvil_nbt import (
    CLAY,
    COARSE_DIRT,
    DIRT,
    GRASS_BLOCK,
    GRAVEL,
    CHEST,
    MUD,
    MUD_BRICKS,
    PACKED_MUD,
    SAND,
    SNOW_BLOCK,
    STONE,
    OAK_LOG,
    OAK_PLANKS,
    TORCH,
    WORLD_MAX_Y,
    WORLD_MIN_Y,
    encode_chunk_nbt,
    encode_level_dat,
)
from .bong_raster import to_bong_tile


CHUNKS_PER_REGION = 32
SECTOR_SIZE = 4096
REGION_HEADER_SIZE = 2 * SECTOR_SIZE

RIVERBED_BLOCKS = {
    "dirt": DIRT,
    "mud": MUD,
    "gravel": GRAVEL,
    "sand": SAND,
    "clay": CLAY,
    "coarse_dirt": COARSE_DIRT,
    "packed_mud": PACKED_MUD,
    "mud_bricks": MUD_BRICKS,
    "mud_brick": MUD_BRICKS,
}

UNDERGROUND_BLOCKS = {
    "oak_log": OAK_LOG,
    "oak_planks": OAK_PLANKS,
    "chest": CHEST,
    "torch": TORCH,
}


@dataclass(frozen=True)
class MinecraftWorldExport:
    output_dir: Path
    chunks_written: int
    regions_written: int
    min_chunk_x: int
    max_chunk_x: int
    min_chunk_z: int
    max_chunk_z: int


def region_for_chunk(chunk_x: int, chunk_z: int) -> tuple[int, int]:
    return chunk_x >> 5, chunk_z >> 5


def chunk_index_in_region(chunk_x: int, chunk_z: int) -> int:
    return ((chunk_z & 31) << 5) | (chunk_x & 31)


def _region_payload(compressed_nbt: bytes) -> bytes:
    payload = struct.pack(">I", len(compressed_nbt) + 1) + b"\x02" + compressed_nbt
    return payload + bytes((-len(payload)) % SECTOR_SIZE)


def write_region(
    region_x: int,
    region_z: int,
    chunks: dict[tuple[int, int], bytes],
    output_dir: Path,
) -> Path:
    """Write one deterministic `.mca` file from zlib-compressed chunk NBT."""

    locations = bytearray(SECTOR_SIZE)
    timestamps = bytearray(SECTOR_SIZE)
    payloads: list[bytes] = []
    next_sector = REGION_HEADER_SIZE // SECTOR_SIZE

    for chunk_x, chunk_z in sorted(chunks, key=lambda pos: chunk_index_in_region(*pos)):
        if region_for_chunk(chunk_x, chunk_z) != (region_x, region_z):
            raise ValueError(f"chunk {(chunk_x, chunk_z)} is outside region {(region_x, region_z)}")
        payload = _region_payload(chunks[(chunk_x, chunk_z)])
        sectors = len(payload) // SECTOR_SIZE
        if sectors > 255:
            raise ValueError(f"chunk {(chunk_x, chunk_z)} exceeds the Anvil 255-sector limit")
        index = chunk_index_in_region(chunk_x, chunk_z)
        locations[index * 4 : index * 4 + 3] = next_sector.to_bytes(3, "big")
        locations[index * 4 + 3] = sectors
        payloads.append(payload)
        next_sector += sectors

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"r.{region_x}.{region_z}.mca"
    with path.open("wb") as output:
        output.write(locations)
        output.write(timestamps)
        for payload in payloads:
            output.write(payload)
    return path


def _surface_blocks(field: Heightfield, sea_level: float) -> np.ndarray:
    tile = to_bong_tile(field, sea_level=sea_level)
    blocks = np.full(field.height.shape, GRASS_BLOCK, dtype=np.uint8)
    blocks[tile.surface_id == 0] = STONE
    blocks[tile.surface_id == 1] = COARSE_DIRT
    blocks[tile.surface_id == 2] = GRAVEL
    blocks[(tile.wilderness_id == LAKE) | (tile.wilderness_id == RIVER)] = GRAVEL
    if tile.riverbed_palette:
        for material_index, material in enumerate(tile.riverbed_palette):
            material = material.removeprefix("minecraft:").replace("-", "_")
            try:
                block_id = RIVERBED_BLOCKS[material]
            except KeyError as exc:
                raise ValueError(
                    f"river bed material {material!r} has no Minecraft block mapping"
                ) from exc
            blocks[tile.riverbed_id == material_index] = block_id
    blocks[(tile.wilderness_id == MOUNTAINS) & (field.height >= sea_level + 92.0)] = SNOW_BLOCK
    return blocks


def _block_heights(
    field: Heightfield,
    sea_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface = np.clip(
        np.rint(field.height), WORLD_MIN_Y + 1, WORLD_MAX_Y - 2
    ).astype(np.int16)
    wet = field.water_level >= 0.0
    water = np.full(field.height.shape, -1, dtype=np.int16)
    water_flow = np.full(field.height.shape, -1, dtype=np.int8)
    water[wet] = np.maximum(
        np.ceil(field.water_level[wet]).astype(np.int16),
        surface[wet] + 1,
    )
    # Sea/lake water remains a source block; raised authored channels use
    # Minecraft's flowing-water state. Continuous surface elevation remains
    # available through Heightfield.water_level for raster/viewer consumers.
    river = wet & (field.water_level > sea_level + 1.0e-3)
    water_flow[wet] = 0
    water_flow[river] = 1
    water = np.clip(water, -1, WORLD_MAX_Y - 1).astype(np.int16)
    return surface, water, water_flow


def _fallback_solid_spans(surface: np.ndarray) -> np.ndarray:
    spans = np.full((*surface.shape, 4, 2), 32767, dtype=np.int16)
    spans[:, :, 0] = np.stack((np.full(surface.shape, WORLD_MIN_Y, dtype=np.int16), surface), axis=-1)
    return spans


def _chunk_structure_blocks(
    field: Heightfield,
    *,
    local_x: int,
    local_z: int,
) -> np.ndarray:
    rows: list[tuple[int, int, int, int]] = []
    for block in field.underground_blocks:
        if not (local_x <= block.x < local_x + 16 and local_z <= block.z < local_z + 16):
            continue
        try:
            block_id = UNDERGROUND_BLOCKS[block.material]
        except KeyError as exc:
            raise ValueError(
                f"underground block material {block.material!r} has no Minecraft block mapping"
            ) from exc
        rows.append((block.x - local_x, block.z - local_z, block.y, block_id))
    if not rows:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def export_minecraft_world(
    field: Heightfield,
    output_dir: Path,
    *,
    origin_x: int,
    origin_z: int,
    sea_level: float,
    seed: int,
    world_name: str,
) -> MinecraftWorldExport:
    """Write one block-per-heightfield-cell as a Minecraft 1.20.1 world."""

    height, width = field.height.shape
    if width % 16 or height % 16:
        raise ValueError("heightfield width and height must be multiples of 16")
    if origin_x % 16 or origin_z % 16:
        raise ValueError("world origins must be aligned to a 16-block chunk boundary")

    surface_y, water_y, water_flow = _block_heights(field, sea_level)
    surface_blocks = _surface_blocks(field, sea_level)
    solid_spans = field.solid_spans
    if solid_spans is None:
        solid_spans = _fallback_solid_spans(surface_y)
    min_chunk_x = origin_x // 16
    min_chunk_z = origin_z // 16
    max_chunk_x = min_chunk_x + width // 16 - 1
    max_chunk_z = min_chunk_z + height // 16 - 1

    region_dir = output_dir / "region"
    regions_written = 0
    chunks_written = 0
    min_region_x, min_region_z = region_for_chunk(min_chunk_x, min_chunk_z)
    max_region_x, max_region_z = region_for_chunk(max_chunk_x, max_chunk_z)

    for region_z in range(min_region_z, max_region_z + 1):
        for region_x in range(min_region_x, max_region_x + 1):
            chunks: dict[tuple[int, int], bytes] = {}
            first_x = max(min_chunk_x, region_x * CHUNKS_PER_REGION)
            last_x = min(max_chunk_x, region_x * CHUNKS_PER_REGION + 31)
            first_z = max(min_chunk_z, region_z * CHUNKS_PER_REGION)
            last_z = min(max_chunk_z, region_z * CHUNKS_PER_REGION + 31)
            for chunk_z in range(first_z, last_z + 1):
                local_z = (chunk_z - min_chunk_z) * 16
                for chunk_x in range(first_x, last_x + 1):
                    local_x = (chunk_x - min_chunk_x) * 16
                    z_slice = slice(local_z, local_z + 16)
                    x_slice = slice(local_x, local_x + 16)
                    nbt = encode_chunk_nbt(
                        chunk_x,
                        chunk_z,
                        surface_y[z_slice, x_slice],
                        water_y[z_slice, x_slice],
                        surface_blocks[z_slice, x_slice],
                        water_flow[z_slice, x_slice],
                        solid_spans[z_slice, x_slice],
                        _chunk_structure_blocks(field, local_x=local_x, local_z=local_z),
                    )
                    chunks[(chunk_x, chunk_z)] = zlib.compress(nbt, level=6)
            write_region(region_x, region_z, chunks, region_dir)
            regions_written += 1
            chunks_written += len(chunks)

    spawn_x = origin_x + width // 2
    spawn_z = origin_z + height // 2
    spawn_y = int(surface_y[height // 2, width // 2]) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "level.dat").write_bytes(
        encode_level_dat(world_name, seed, spawn_x, spawn_y, spawn_z)
    )
    return MinecraftWorldExport(
        output_dir=output_dir,
        chunks_written=chunks_written,
        regions_written=regions_written,
        min_chunk_x=min_chunk_x,
        max_chunk_x=max_chunk_x,
        min_chunk_z=min_chunk_z,
        max_chunk_z=max_chunk_z,
    )


__all__ = [
    "MinecraftWorldExport",
    "chunk_index_in_region",
    "export_minecraft_world",
    "region_for_chunk",
    "write_region",
]
