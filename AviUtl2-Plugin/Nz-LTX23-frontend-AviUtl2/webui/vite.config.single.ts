import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

/**
 * M7b: a second build target that inlines every JS/CSS asset directly into
 * `dist-single/index.html` (`npm run build:single`), so the whole WebUI can
 * be shipped/previewed as one file with no `assets/` directory to wire up a
 * virtual host mapping for. The normal `npm run build` (`vite.config.ts`,
 * `dist/`) is unaffected — this is a wholly separate config/output dir, not
 * a variant of it, so nothing here changes what native's WebView2 virtual
 * host mapping (`webui/dist`) actually loads.
 */
export default defineConfig({
  base: "./",
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: "dist-single",
    // A single HTML file has no benefit from chunk splitting, and
    // `viteSingleFile`'s recommended config already disables it — being
    // explicit here too so this file stays correct even if that default
    // ever changes upstream.
    cssCodeSplit: false,
  },
});
