import type { Manifest } from "./types";

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
}

export async function loadOverview(manifest: Manifest): Promise<OverviewData> {
  const overview = manifest.overview;
  const [heightResponse, surfaceResponse, wildernessResponse] = await Promise.all([
    fetch(`${WORLD_ROOT}/${overview.height_file}`),
    fetch(`${WORLD_ROOT}/${overview.surface_file}`),
    fetch(`${WORLD_ROOT}/${overview.wilderness_file}`),
  ]);
  if (!heightResponse.ok || !surfaceResponse.ok || !wildernessResponse.ok) {
    throw new Error("overview raster is incomplete");
  }
  const elevation = new Float32Array(await heightResponse.arrayBuffer());
  const surfaceId = new Uint8Array(await surfaceResponse.arrayBuffer());
  const wildernessId = new Uint8Array(await wildernessResponse.arrayBuffer());
  const expected = overview.width * overview.height;
  if (
    elevation.length !== expected ||
    surfaceId.length !== expected ||
    wildernessId.length !== expected
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
  };
}
