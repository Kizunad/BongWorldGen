import fs from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const worldRoot = path.resolve(
  process.env.BONG_WORLD_DIR ?? path.join(here, "../generated/console-world/rasters"),
);

function generatedWorldPlugin(): Plugin {
  const serve = (request: IncomingMessage, response: ServerResponse, next: (error?: unknown) => void): void => {
    const rawUrl = request.url?.split("?", 1)[0] ?? "/";
    const relative = decodeURIComponent(rawUrl).replace(/^\/+/, "");
    const file = path.resolve(worldRoot, relative);
    const outsideRoot = path.relative(worldRoot, file).startsWith("..");
    if (outsideRoot) {
      response.statusCode = 400;
      response.end("invalid world path");
      return;
    }
    if (!fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
      next();
      return;
    }
    const extension = path.extname(file);
    const contentType = extension === ".json" ? "application/json" : "application/octet-stream";
    response.statusCode = 200;
    response.setHeader("Content-Type", contentType);
    fs.createReadStream(file).pipe(response);
  };

  return {
    name: "serve-generated-world",
    configureServer(server) {
      server.middlewares.use("/world", serve);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/world", serve);
    },
  };
}

export default defineConfig({
  plugins: [generatedWorldPlugin()],
  server: { port: 5173 },
  preview: { port: 4173 },
});
