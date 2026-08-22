// three.js scene glue: renderer, lights (hillshade-aligned), a fly camera, and
// the tile -> mesh -> scene wiring. This is the only module that imports three;
// the data + mesh law live in decode.ts (unit-tested) and api/tile-loader.

import * as THREE from "three";
import { spansToVoxelGeometry, surfaceY, type ColorMode } from "./decode";
import { buildFloraPoints, buildPoiMarkers } from "./decorations";
import { LIGHT_DIR, surfaceColorForId } from "./palette";
import type { DecodedTile, Manifest } from "./types";
import type { OverviewData } from "./overview";

const WILDERNESS_HIGHLIGHT_COLOR = 0xffd166;

/**
 * Recursively detach an object from its parent AND free its GPU resources.
 *
 * Walks the whole subtree (Group -> Mesh/Points/...): for every node carrying a
 * `geometry` and/or `material`, dispose them (materials can be an array — a
 * multi-material mesh), then remove the root from its parent. Plain
 * `removeFromParent()` only unhooks the scene-graph link and leaks the
 * underlying WebGL buffers/textures, which accumulate across regens.
 */
function disposeObject3D(root: THREE.Object3D): void {
  root.traverse((node) => {
    const withGeo = node as Partial<THREE.Mesh>;
    withGeo.geometry?.dispose?.();
    const material = (node as Partial<THREE.Mesh>).material;
    if (Array.isArray(material)) {
      for (const m of material) m.dispose();
    } else {
      material?.dispose?.();
    }
  });
  root.removeFromParent();
}

export interface ViewerLayers {
  terrain: boolean;
  wilderness: boolean;
  water: boolean;
  qi: boolean;
  decorations: boolean;
}

/** Pick a per-tile LOD stride by distance to spawn so far tiles stay cheap. */
export function lodStrideFor(tileX: number, tileZ: number): number {
  const d2 = tileX * tileX + tileZ * tileZ;
  if (d2 <= 1) return 1; // spawn + ring 1: full res
  if (d2 <= 9) return 2; // mid ring
  return 4; // far ring
}

export interface OverviewViewSettings {
  position: [number, number, number];
  target: [number, number, number];
  farPlane: number;
  fogNear: number;
  fogFar: number;
}

/** Derive a complete-world camera and fog range from the overview bounds. */
export function overviewViewSettings(data: OverviewData): OverviewViewSettings {
  const minX = data.originX;
  const maxX = minX + Math.max(0, data.width - 1) * data.cellSize;
  const minZ = data.originZ;
  const maxZ = minZ + Math.max(0, data.height - 1) * data.cellSize;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const elevation of data.elevation) {
    minY = Math.min(minY, elevation);
    maxY = Math.max(maxY, elevation);
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
    minY = 0;
    maxY = 0;
  }

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const worldSpan = Math.max(maxX - minX, maxZ - minZ, data.cellSize);
  const distance = worldSpan * 0.72;
  const position: [number, number, number] = [
    centerX + distance,
    centerY + distance * 0.95,
    centerZ + distance,
  ];

  let farthestPoint = 0;
  for (const x of [minX, maxX]) {
    for (const y of [minY, maxY]) {
      for (const z of [minZ, maxZ]) {
        farthestPoint = Math.max(
          farthestPoint,
          Math.hypot(x - position[0], y - position[1], z - position[2]),
        );
      }
    }
  }

  // Keep the complete raster inside both the camera frustum and the fog range.
  // A fixed 400..1400 fog range makes an 8192-block overview indistinguishable
  // from the background even though its geometry loaded successfully.
  const fogNear = Math.max(400, farthestPoint * 0.85);
  const fogFar = Math.max(1400, farthestPoint * 1.35);
  return {
    position,
    target: [centerX, centerY, centerZ],
    farPlane: Math.max(8000, fogFar * 1.05),
    fogNear,
    fogFar,
  };
}

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly terrainGroup = new THREE.Group();
  private readonly wildernessGroup = new THREE.Group();
  private readonly overviewTerrainGroup = new THREE.Group();
  private readonly overviewWildernessGroup = new THREE.Group();
  private readonly waterGroup = new THREE.Group();
  private readonly qiGroup = new THREE.Group();
  private readonly decorGroup = new THREE.Group();
  /** tile dir -> meshes, so a regen can dispose+replace just those tiles. */
  private readonly tileMeshes = new Map<string, THREE.Object3D[]>();
  /** Decoded tile cache lets layer toggles rebuild without refetching bytes. */
  private readonly tileData = new Map<
    string,
    { tile: DecodedTile; surfacePalette: string[]; wildernessPalette: Manifest["wilderness_palette"] }
  >();
  private overviewData: {
    data: OverviewData;
    surfacePalette: string[];
    wildernessPalette: Manifest["wilderness_palette"];
  } | null = null;
  private highlightWildernessId: number | null = null;

  private layers: ViewerLayers = {
    terrain: true,
    wilderness: false,
    water: false,
    qi: false,
    decorations: false,
  };

  // Fly-camera state.
  private readonly keys = new Set<string>();
  private yaw = -Math.PI / 4;
  private pitch = -0.5;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private clock = new THREE.Clock();
  private disposed = false;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(0x0a0d14);
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x0a0d14);
    this.scene.fog = new THREE.Fog(0x0a0d14, 400, 1400);

    this.camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.5,
      8000,
    );
    this.camera.position.set(120, 160, 120);

    // Hillshade-aligned key light + soft ambient (mirrors exporters.py _hillshade).
    const sun = new THREE.DirectionalLight(0xfff4e0, 1.05);
    sun.position.set(-LIGHT_DIR[0], -LIGHT_DIR[1] + 1, -LIGHT_DIR[2]).multiplyScalar(500);
    this.scene.add(sun);
    this.scene.add(new THREE.HemisphereLight(0xbcc8ff, 0x202020, 0.65));

    this.scene.add(
      this.terrainGroup,
      this.wildernessGroup,
      this.overviewTerrainGroup,
      this.overviewWildernessGroup,
      this.waterGroup,
      this.qiGroup,
      this.decorGroup,
    );
    this.applyLayerVisibility();

    this.bindInput();
    window.addEventListener("resize", this.onResize);
    this.animate();
  }

  setLayers(layers: ViewerLayers): void {
    this.layers = layers;
    this.applyLayerVisibility();
    this.rebuildAllTiles();
  }

  setWildernessHighlight(wildernessId: number | null): void {
    if (this.highlightWildernessId === wildernessId) return;
    this.highlightWildernessId = wildernessId;
    this.rebuildOverviewHighlight();
    this.rebuildAllTiles();
  }

  /** Render the display-only full-world overview below detailed tile meshes. */
  setOverview(
    data: OverviewData,
    surfacePalette: string[],
    wildernessPalette: Manifest["wilderness_palette"],
  ): void {
    for (const child of [...this.overviewTerrainGroup.children]) disposeObject3D(child);
    const vertexCount = data.width * data.height;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);
    const indexCount = (data.width - 1) * (data.height - 1) * 6;
    const indices = new Uint32Array(indexCount);
    for (let z = 0; z < data.height; z += 1) {
      for (let x = 0; x < data.width; x += 1) {
        const index = z * data.width + x;
        const offset = index * 3;
        positions[offset] = data.originX + x * data.cellSize;
        positions[offset + 1] = data.elevation[index] - 2.0;
        positions[offset + 2] = data.originZ + z * data.cellSize;
        const [r, g, b] = surfaceColorForId(data.surfaceId[index], surfacePalette);
        colors[offset] = r / 255;
        colors[offset + 1] = g / 255;
        colors[offset + 2] = b / 255;
      }
    }
    let cursor = 0;
    for (let z = 0; z < data.height - 1; z += 1) {
      for (let x = 0; x < data.width - 1; x += 1) {
        const a = z * data.width + x;
        const b = a + 1;
        const c = a + data.width;
        const d = c + 1;
        indices.set([a, c, b, b, c, d], cursor);
        cursor += 6;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true }));
    this.overviewTerrainGroup.add(mesh);
    this.overviewData = { data, surfacePalette, wildernessPalette };
    this.rebuildOverviewHighlight();

    const view = overviewViewSettings(data);
    this.camera.far = view.farPlane;
    this.camera.updateProjectionMatrix();
    this.camera.position.set(...view.position);
    const forward = new THREE.Vector3(...view.target).sub(this.camera.position).normalize();
    this.yaw = Math.atan2(forward.x, forward.z);
    this.pitch = Math.asin(forward.y);
    this.scene.fog = new THREE.Fog(0x0a0d14, view.fogNear, view.fogFar);
  }

  private rebuildOverviewHighlight(): void {
    for (const child of [...this.overviewWildernessGroup.children]) disposeObject3D(child);
    if (this.highlightWildernessId === null || !this.overviewData) return;

    const { data, wildernessPalette } = this.overviewData;
    const wilderness = wildernessPalette.find((entry) => entry.id === this.highlightWildernessId);
    if (!wilderness) return;
    const positions: number[] = [];
    const indices: number[] = [];
    let vertex = 0;
    for (let z = 0; z < data.height - 1; z += 1) {
      for (let x = 0; x < data.width - 1; x += 1) {
        const a = z * data.width + x;
        if (data.wildernessId[a] !== this.highlightWildernessId) continue;
        const b = a + 1;
        const c = a + data.width;
        const d = c + 1;
        const x0 = data.originX + x * data.cellSize;
        const x1 = x0 + data.cellSize;
        const z0 = data.originZ + z * data.cellSize;
        const z1 = z0 + data.cellSize;
        positions.push(
          x0, data.elevation[a] - 1.7, z0,
          x1, data.elevation[b] - 1.7, z0,
          x0, data.elevation[c] - 1.7, z1,
          x1, data.elevation[d] - 1.7, z1,
        );
        indices.push(vertex, vertex + 2, vertex + 1, vertex + 1, vertex + 2, vertex + 3);
        vertex += 4;
      }
    }
    if (positions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshBasicMaterial({
      color: WILDERNESS_HIGHLIGHT_COLOR,
      transparent: true,
      opacity: 0.68,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
    });
    this.overviewWildernessGroup.add(new THREE.Mesh(geometry, material));
  }

  /** Build POI cones from the manifest (once; replaces any prior markers). */
  setPoiMarkers(manifest: Manifest): void {
    for (let i = this.decorGroup.children.length - 1; i >= 0; i--) {
      const child = this.decorGroup.children[i];
      if (child.userData?.kind === "poi") {
        // dispose the old markers' GPU resources, not just detach them — a
        // re-`setPoiMarkers` (e.g. after regen) would otherwise leak the cone
        // geometry + per-POI materials.
        disposeObject3D(child);
      }
    }
    const markers = buildPoiMarkers(manifest);
    markers.userData = { kind: "poi" };
    this.decorGroup.add(markers);
  }

  private applyLayerVisibility(): void {
    this.terrainGroup.visible = this.layers.terrain;
    this.wildernessGroup.visible = this.layers.wilderness;
    this.overviewTerrainGroup.visible = this.layers.terrain;
    this.overviewWildernessGroup.visible = this.layers.wilderness;
    this.waterGroup.visible = this.layers.water;
    this.qiGroup.visible = this.layers.qi;
    this.decorGroup.visible = this.layers.decorations;
  }

  /** Build (or rebuild) all renderables for one decoded tile. */
  setTile(tile: DecodedTile, surfacePalette: string[], wildernessPalette: Manifest["wilderness_palette"]): void {
    const key = `tile_${tile.tileX}_${tile.tileZ}`;
    this.tileData.set(key, { tile, surfacePalette, wildernessPalette });
    this.rebuildTile(tile, surfacePalette, wildernessPalette);
  }

  private rebuildAllTiles(): void {
    for (const entry of this.tileData.values()) {
      this.rebuildTile(entry.tile, entry.surfacePalette, entry.wildernessPalette);
    }
  }

  private rebuildTile(
    tile: DecodedTile,
    surfacePalette: string[],
    wildernessPalette: Manifest["wilderness_palette"],
  ): void {
    this.disposeTile(tile);
    const stride = lodStrideFor(tile.tileX, tile.tileZ);
    const meshes: THREE.Object3D[] = [];
    const key = `tile_${tile.tileX}_${tile.tileZ}`;

    if (this.layers.terrain) {
      meshes.push(this.buildTerrainMesh(tile, surfacePalette, stride, "terrain", wildernessPalette));
    }
    if (
      this.layers.wilderness &&
      this.highlightWildernessId !== null &&
      tile.wildernessId
    ) {
      const wilderness = this.buildTerrainMesh(
        tile,
        surfacePalette,
        stride,
        "wilderness",
        wildernessPalette,
        this.highlightWildernessId,
      );
      this.wildernessGroup.add(wilderness);
      meshes.push(wilderness);
    }
    // Only build the qi heatmap mesh when the tile actually carries qi_density;
    // without it the mesher falls back to surface_id == 0 and paints a flat
    // (and misleading) heat color over the whole tile. Absent layer -> no qi
    // mesh, the qi toggle simply shows nothing for that tile.
    if (this.layers.qi && tile.qiDensity) {
      const qi = this.buildTerrainMesh(tile, surfacePalette, stride, "qi", wildernessPalette);
      this.qiGroup.add(qi);
      meshes.push(qi);
    }

    const water = this.layers.water ? this.buildWaterMesh(tile, stride) : null;
    if (water) {
      this.waterGroup.add(water);
      meshes.push(water);
    }

    const flora = this.layers.decorations
      ? buildFloraPoints(
          tile.tileX,
          tile.tileZ,
          tile.tileSize,
          tile.floraVariantId,
          (colIdx) => surfaceY(tile.columns[colIdx]),
          stride,
        )
      : null;
    if (flora) {
      flora.userData = { kind: "flora" };
      this.decorGroup.add(flora);
      meshes.push(flora);
    }

    this.tileMeshes.set(key, meshes);
  }

  private buildTerrainMesh(
    tile: DecodedTile,
    surfacePalette: string[],
    stride: number,
    colorMode: ColorMode,
    wildernessPalette: Manifest["wilderness_palette"],
    highlightWildernessId?: number,
  ): THREE.Mesh {
    const geo = spansToVoxelGeometry(tile, surfacePalette, {
      stride,
      colorMode,
      wildernessPalette,
      highlightWildernessId,
      cullTileEdges: true,
    });
    const bg = new THREE.BufferGeometry();
    bg.setAttribute("position", new THREE.BufferAttribute(geo.positions, 3));
    bg.setAttribute("normal", new THREE.BufferAttribute(geo.normals, 3));
    bg.setAttribute("color", new THREE.BufferAttribute(geo.colors, 3));
    bg.setIndex(new THREE.BufferAttribute(geo.indices, 1));
    const mat =
      colorMode === "wilderness"
        ? new THREE.MeshBasicMaterial({
            color: WILDERNESS_HIGHLIGHT_COLOR,
            transparent: true,
            opacity: 0.68,
            depthWrite: false,
            polygonOffset: true,
            polygonOffsetFactor: -1,
          })
        : new THREE.MeshLambertMaterial({ vertexColors: true });
    const mesh = new THREE.Mesh(bg, mat);
    if (colorMode === "terrain") this.terrainGroup.add(mesh);
    return mesh;
  }

  /** Translucent water plane per column where water_level >= 0. */
  private buildWaterMesh(tile: DecodedTile, stride: number): THREE.Mesh | null {
    if (!tile.waterLevel) return null;
    const ts = tile.tileSize;
    const baseX = tile.tileX * ts;
    const baseZ = tile.tileZ * ts;
    const positions: number[] = [];
    const indices: number[] = [];
    let v = 0;
    for (let lz = 0; lz < ts; lz += stride) {
      for (let lx = 0; lx < ts; lx += stride) {
        const wl = tile.waterLevel[lz * ts + lx];
        if (wl < 0) continue; // -1 sentinel = no water
        const x0 = baseX + lx;
        const x1 = x0 + stride;
        const z0 = baseZ + lz;
        const z1 = z0 + stride;
        const y = wl + 1;
        positions.push(x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1);
        indices.push(v, v + 1, v + 2, v, v + 2, v + 3);
        v += 4;
      }
    }
    if (positions.length === 0) return null;
    const bg = new THREE.BufferGeometry();
    bg.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    bg.setIndex(indices);
    bg.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({
      color: 0x4a70a8,
      transparent: true,
      opacity: 0.55,
    });
    return new THREE.Mesh(bg, mat);
  }

  /** Drop every renderable for a tile (used before rebuild / on regen). */
  private disposeTile(tile: DecodedTile): void {
    const key = `tile_${tile.tileX}_${tile.tileZ}`;
    const meshes = this.tileMeshes.get(key);
    if (!meshes) return;
    for (const m of meshes) disposeObject3D(m);
    this.tileMeshes.delete(key);
  }

  // ---- fly camera input ----------------------------------------------------

  /** True when the focused element is a text-editing control. The fly camera
   *  must NOT consume WASD/QE while the user types in the param textarea. */
  private static isEditingTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      el.isContentEditable === true
    );
  }

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    // Ignore movement keys while editing a control (textarea/input/select) so
    // typing parameters doesn't fly the camera around. Drop any keys already
    // held so the camera doesn't keep drifting once focus enters the textarea.
    if (Viewer.isEditingTarget(e.target)) {
      this.keys.clear();
      return;
    }
    this.keys.add(e.key.toLowerCase());
  };

  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.keys.delete(e.key.toLowerCase());
  };

  private readonly onMouseUp = (): void => {
    this.dragging = false;
  };

  private readonly onMouseMove = (e: MouseEvent): void => {
    if (!this.dragging) return;
    this.yaw -= (e.clientX - this.lastX) * 0.005;
    this.pitch -= (e.clientY - this.lastY) * 0.005;
    this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch));
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private bindInput(): void {
    const el = this.renderer.domElement;
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    el.addEventListener("mousedown", (e) => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
    window.addEventListener("mouseup", this.onMouseUp);
    window.addEventListener("mousemove", this.onMouseMove);
    el.addEventListener("wheel", (e) => {
      const dir = this.forward();
      this.camera.position.addScaledVector(dir, -e.deltaY * 1.2);
      e.preventDefault();
    });
  }

  private forward(): THREE.Vector3 {
    return new THREE.Vector3(
      Math.cos(this.pitch) * Math.sin(this.yaw),
      Math.sin(this.pitch),
      Math.cos(this.pitch) * Math.cos(this.yaw),
    ).normalize();
  }

  private updateCamera(dt: number): void {
    const speed = (this.keys.has("shift") ? 800 : 260) * dt;
    const fwd = this.forward();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    if (this.keys.has("w")) this.camera.position.addScaledVector(fwd, speed);
    if (this.keys.has("s")) this.camera.position.addScaledVector(fwd, -speed);
    if (this.keys.has("d")) this.camera.position.addScaledVector(right, speed);
    if (this.keys.has("a")) this.camera.position.addScaledVector(right, -speed);
    if (this.keys.has("e")) this.camera.position.y += speed;
    if (this.keys.has("q")) this.camera.position.y -= speed;
    this.camera.lookAt(this.camera.position.clone().add(fwd));
  }

  private onResize = (): void => {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  private animate = (): void => {
    if (this.disposed) return;
    requestAnimationFrame(this.animate);
    this.updateCamera(this.clock.getDelta());
    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    this.disposed = true;
    window.removeEventListener("resize", this.onResize);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("mouseup", this.onMouseUp);
    window.removeEventListener("mousemove", this.onMouseMove);
    this.renderer.dispose();
  }
}
