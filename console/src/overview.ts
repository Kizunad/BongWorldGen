import type { Manifest } from "./types";
import { surfaceColorForId, zoneColorForId } from "./palette";

const WORLD_ROOT = "/world";

export interface OverviewData {
  width: number;
  height: number;
  originX: number;
  originZ: number;
  cellSize: number;
  elevation: Float32Array;
  surfaceId: Uint8Array;
  wildernessId: Uint8Array;
  zoneId?: Uint8Array;
}

/** Vertex colors shared by overview terrain and its optional zone overlay. */
export function overviewColors(
  data: OverviewData,
  surfacePalette: string[],
  zonePalette?: string[],
): Float32Array {
  const colors = new Float32Array(data.width * data.height * 3);
  for (let i = 0; i < data.width * data.height; i += 1) {
    const zone = data.zoneId && zonePalette
      ? zoneColorForId(data.zoneId[i], zonePalette)
      : undefined;
    const [r, g, b] = zone ?? surfaceColorForId(data.surfaceId[i], surfacePalette);
    colors.set([r / 255, g / 255, b / 255], i * 3);
  }
  return colors;
}

export async function loadOverview(manifest: Manifest): Promise<OverviewData> {
  const overview = manifest.overview;
  const [heightResponse, surfaceResponse, wildernessResponse, zoneResponse] = await Promise.all([
    fetch(`${WORLD_ROOT}/${overview.height_file}`),
    fetch(`${WORLD_ROOT}/${overview.surface_file}`),
    fetch(`${WORLD_ROOT}/${overview.wilderness_file}`),
    overview.zone_file ? fetch(`${WORLD_ROOT}/${overview.zone_file}`) : Promise.resolve(null),
  ]);
  if (!heightResponse.ok || !surfaceResponse.ok || !wildernessResponse.ok ||
      (zoneResponse !== null && !zoneResponse.ok)) {
    throw new Error("overview raster is incomplete");
  }
  const elevation = new Float32Array(await heightResponse.arrayBuffer());
  const surfaceId = new Uint8Array(await surfaceResponse.arrayBuffer());
  const wildernessId = new Uint8Array(await wildernessResponse.arrayBuffer());
  const zoneId = zoneResponse ? new Uint8Array(await zoneResponse.arrayBuffer()) : undefined;
  const expected = overview.width * overview.height;
  if (
    elevation.length !== expected ||
    surfaceId.length !== expected ||
    wildernessId.length !== expected ||
    (zoneId !== undefined && zoneId.length !== expected)
  ) {
    throw new Error(
      `overview dimensions ${overview.width}x${overview.height} do not match binary layers`,
    );
  }
  return {
    width: overview.width,
    height: overview.height,
    originX: overview.origin_x,
    originZ: overview.origin_z,
    cellSize: overview.cell_size,
    elevation,
    surfaceId,
    wildernessId,
    zoneId,
  };
}
