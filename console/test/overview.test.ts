import { afterEach, describe, expect, it, vi } from "vitest";

import { decodeTile, spansToVoxelGeometry } from "../src/decode";
import { loadOverview, overviewColors, type OverviewData } from "../src/overview";
import type { Manifest, OverviewManifest } from "../src/types";
import { encodeColumns, fillColumns } from "./fixtures";

const OVERVIEW: OverviewManifest = {
  width: 2, height: 2, origin_x: -32, origin_z: -64, cell_size: 32,
  height_file: "overview_height.bin", surface_file: "overview_surface_id.bin",
  wilderness_file: "overview_wilderness_id.bin", zone_file: "overview_zone_id.bin",
};

function mockRasters(zone: number[] | null = [0, 255, 1, 0]) {
  const buffers: Record<string, ArrayBuffer> = {
    "/world/overview_height.bin": new Float32Array([71, 71, 71, 71]).buffer,
    "/world/overview_surface_id.bin": new Uint8Array([1, 1, 1, 1]).buffer,
    "/world/overview_wilderness_id.bin": new Uint8Array([0, 0, 0, 0]).buffer,
  };
  if (zone !== null) buffers["/world/overview_zone_id.bin"] = new Uint8Array(zone).buffer;
  const fetchMock = vi.fn(async (url: string) => buffers[url]
    ? new Response(buffers[url]) : new Response(null, { status: 404 }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe("overview zone raster", () => {
  it("loads the declared bytes and uses the same per-column colors as detailed tiles", async () => {
    mockRasters();
    const data = await loadOverview({ overview: OVERVIEW } as Manifest);
    expect(data.zoneId).toEqual(new Uint8Array([0, 255, 1, 0]));
    expect([data.originX, data.originZ, data.cellSize]).toEqual([-32, -64, 32]);
    const surfacePalette = ["stone", "grass_block"];
    const zonePalette = ["first", "second"];
    const colors = overviewColors(data, surfacePalette, zonePalette);
    const tile = decodeTile({
      tileSize: 2, ...encodeColumns(2, fillColumns(2, [[-64, 71]])),
      surfaceId: data.surfaceId, zoneId: data.zoneId,
    });
    const geometry = spansToVoxelGeometry(tile, surfacePalette, {
      colorMode: "zone", zonePalette, cullTileEdges: true,
    });
    expect(geometry.faceCount).toBe(4);
    for (let i = 0; i < 4; i += 1) {
      expect(colors.slice(i * 3, i * 3 + 3)).toEqual(geometry.colors.slice(i * 12, i * 12 + 3));
    }
    expect(colors.slice(0, 3)).not.toEqual(colors.slice(6, 9));
    expect(colors[3]).toBeCloseTo(93 / 255, 5); // Background keeps the grass material.
  });

  it("keeps older manifests working without requesting an undeclared zone file", async () => {
    const fetchMock = mockRasters(null);
    const data = await loadOverview({ overview: { ...OVERVIEW, zone_file: undefined } } as Manifest);
    expect(data.zoneId).toBeUndefined();
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/world/overview_height.bin", "/world/overview_surface_id.bin",
      "/world/overview_wilderness_id.bin",
    ]);
    expect(overviewColors(data, ["stone", "grass_block"], ["first"]))
      .toEqual(overviewColors(data, ["stone", "grass_block"]));
  });

  it("rejects a truncated declared zone raster", async () => {
    mockRasters([0]);
    await expect(loadOverview({ overview: OVERVIEW } as Manifest)).rejects.toThrow(/dimensions/);
  });

  it("rejects a missing declared zone raster", async () => {
    mockRasters(null);
    await expect(loadOverview({ overview: OVERVIEW } as Manifest)).rejects.toThrow(/incomplete/);
  });

  it("keeps zone colors stable when palette IDs change", () => {
    const data: OverviewData = {
      width: 2, height: 1, originX: 0, originZ: 0, cellSize: 32,
      elevation: new Float32Array([70, 70]), surfaceId: new Uint8Array([0, 0]),
      wildernessId: new Uint8Array([0, 0]), zoneId: new Uint8Array([0, 1]),
    };
    expect(overviewColors(data, ["stone"], ["first", "second"]))
      .toEqual(overviewColors({ ...data, zoneId: new Uint8Array([1, 0]) },
                             ["stone"], ["second", "first"]));
  });
});
