export interface SpansEncoding {
  /** MAX_SPANS — slots per column (4). */
  max_spans: number;
  /** SPAN_BYTES_PER_COLUMN — stride in bytes per column (16). */
  bytes_per_column: number;
  /** i16::MAX (32767) — marks an unused slot. */
  sentinel: number;
  count_file: string;
  spans_file: string;
  slot_layout: string;
}

export interface WorldBounds {
  min_x: number;
  max_x: number;
  min_z: number;
  max_z: number;
}

export interface ManifestTile {
  tile_x: number;
  tile_z: number;
  dir: string;
  /** Zone names contributing to this tile. */
  zones: string[];
  /** Semantic / vertical layer names actually written for this tile. */
  layers: string[];
  /** Always true in P0 — every tile carries spans_count.bin + spans.bin. */
  spans: boolean;
}

export interface OverviewManifest {
  width: number;
  height: number;
  origin_x: number;
  origin_z: number;
  cell_size: number;
  height_file: string;
  surface_file: string;
  wilderness_file: string;
}

export interface WildernessType {
  id: number;
  key: string;
  display_name: string;
  color: string;
  decoration_tags: string[];
}

export interface ManifestPoi {
  zone: string;
  kind: string;
  name: string;
  pos_xyz: [number, number, number];
  tags: string[];
  unlock: string;
  qi_affinity: number;
  danger_bias: number;
}

export interface ManifestZone {
  name: string;
  display_name: string;
  terrain_profile: string;
  spirit_qi: number;
  danger_level: number;
  worldgen: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Manifest {
  version: number;
  backend: string;
  world_name: string;
  tile_size: number;
  spans_encoding: SpansEncoding;
  world_bounds: WorldBounds;
  overview: OverviewManifest;
  /** surface_id index -> block name; the viewer maps name -> RGB. */
  surface_palette: string[];
  biome_palette: string[];
  wilderness_palette: WildernessType[];
  /** Riverbed material names; IDs are stored in riverbed_id.bin. */
  riverbed_palette?: string[];
  /** Cave network names; IDs are stored in cave_id.bin (0 = no cave). */
  cave_palette?: string[];
  /** Authored zone names; 255 in zone_id.bin means background. */
  zone_palette?: string[];
  tiles: ManifestTile[];
  pois: ManifestPoi[];
  zones: ManifestZone[];
  semantic_layers: string[];
  vertical_layers: string[];
  [key: string]: unknown;
}

/** A single solid vertical range in a column: blocks [floor_y, ceiling_y] inclusive. */
export interface Span {
  floorY: number;
  ceilingY: number;
}

/** Decoded vertical structure of one column. Empty array = full-void column. */
export type ColumnSpans = Span[];

/** A decoded tile: per-column spans + the surface palette indices for coloring. */
export interface DecodedTile {
  tileSize: number;
  tileX: number;
  tileZ: number;
  /** Length tileSize*tileSize, row-major (col_idx = local_z*tileSize + local_x). */
  columns: ColumnSpans[];
  /** Optional surface_id per column (uint8), same indexing as columns. */
  surfaceId?: Uint8Array;
  /** Optional water_level per column (float32); < 0 means no water. */
  waterLevel?: Float32Array;
  /** Optional wilderness_id per column (uint8), indexed by stable palette id. */
  wildernessId?: Uint8Array;
  /** Optional riverbed material id per column; 255 means no riverbed. */
  riverbedId?: Uint8Array;
  /** Optional cave network id per column; 0 means no underground feature. */
  caveId?: Uint8Array;
  /** Optional dominant authored zone id per column; 255 means background. */
  zoneId?: Uint8Array;
  /** Optional qi_density per column (float32), clamped [0,1]. */
  qiDensity?: Float32Array;
  /** Optional flora_variant_id per column (uint8); 0 = no flora. */
  floraVariantId?: Uint8Array;
}
