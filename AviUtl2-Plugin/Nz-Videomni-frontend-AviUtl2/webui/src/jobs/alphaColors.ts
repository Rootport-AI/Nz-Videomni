import type { AlphaChromaColor } from "../api/types";

/**
 * Alpha Gen 第 3 弾: the 🪣 color dropdown's choices, in the order shown
 * (Docs/ALPHAGEN_DESIGN.md §10.2 — the names match AviUtl2's
 * `data\Default\default.palette` presets). Names only: the hex values live in
 * one place on the server (`ALPHA_CHROMA_COLORS`, `api/models.py`), and the
 * dropdown shows the names as they are.
 */
export const ALPHA_CHROMA_COLORS: readonly AlphaChromaColor[] = [
  "white",
  "red",
  "yellow",
  "green",
  "aqua",
  "blue",
  "magenta",
  "black",
];

/** The color the dropdown starts on (§10.2). */
export const DEFAULT_ALPHA_CHROMA_COLOR: AlphaChromaColor = "green";
