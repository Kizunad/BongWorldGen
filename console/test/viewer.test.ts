import { describe, expect, it } from "vitest";

import type { OverviewData } from "../src/overview";
import { lodStrideFor, overviewViewSettings } from "../src/viewer";

function overview(
  width: number,
  height: number,
  originX: number,
  originZ: number,
  cellSize: number,
  elevation: number[],
): OverviewData {
  return {
    width,
    height,
    originX,
    originZ,
    cellSize,
    elevation: new Float32Array(elevation),
    surfaceId: new Uint8Array(width * height),
    wildernessId: new Uint8Array(width * height),
  };
}

function distance(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

describe("overviewViewSettings", () => {
  it("centres a square overview on its actual vertex bounds", () => {
    const view = overviewViewSettings(
      overview(3, 3, -32, -32, 32, [0, 0, 0, 10, 10, 10, 20, 20, 20]),
    );

    expect(view.target).toEqual([0, 10, 0]);
    expect(view.position[0]).toBeGreaterThan(view.target[0]);
    expect(view.position[1]).toBeGreaterThan(view.target[1]);
    expect(view.position[2]).toBeGreaterThan(view.target[2]);
  });

  it("keeps every corner of a rectangular overview before fog and far clipping", () => {
    const data = overview(5, 3, -20, -10, 10, [
      -5, -5, -5, -5, -5,
      10, 10, 10, 10, 10,
      25, 25, 25, 25, 25,
    ]);
    const view = overviewViewSettings(data);
    const corners = [
      [-20, -5, -10],
      [-20, 25, 10],
      [20, -5, -10],
      [20, 25, 10],
    ];
    const farthestCorner = Math.max(...corners.map((corner) => distance(view.position, corner)));

    expect(view.target).toEqual([0, 10, 0]);
    expect(view.fogNear).toBeLessThan(view.fogFar);
    expect(view.fogFar).toBeGreaterThan(farthestCorner);
    expect(view.farPlane).toBeGreaterThan(view.fogFar);
  });

  it("falls back to sea level when elevation contains no finite values", () => {
    const view = overviewViewSettings(overview(2, 2, 0, 0, 32, [NaN, NaN, NaN, NaN]));

    expect(view.target[1]).toBe(0);
    expect(view.position.every(Number.isFinite)).toBe(true);
    expect(view.fogNear).toBeLessThan(view.fogFar);
  });
});

describe("lodStrideFor", () => {
  it("preserves the original console detail rings", () => {
    expect(lodStrideFor(0, 0)).toBe(1);
    expect(lodStrideFor(1, 0)).toBe(1);
    expect(lodStrideFor(2, 0)).toBe(2);
    expect(lodStrideFor(3, 0)).toBe(2);
    expect(lodStrideFor(4, 0)).toBe(4);
  });
});
