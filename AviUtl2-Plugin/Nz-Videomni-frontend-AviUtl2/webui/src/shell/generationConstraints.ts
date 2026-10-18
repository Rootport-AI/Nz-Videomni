import type { BaseModelConstraints, SamplingConstraints } from "../api/types";
import {
  floorToGrid,
  FRAME_RATE_MAX,
  FRAME_RATE_MIN,
  LTX_DIMENSION_MULTIPLE,
  LTX_FRAME_GRID,
} from "../modes/single/paramUtils";
import type { FrameGrid } from "../modes/single/paramUtils";
import { MIN_NUM_FRAMES } from "../modes/single/defaultConfig";

/**
 * The loaded base model's numeric rules, in the shape Create (and the
 * right-click seeds that land in Create) read them.
 *
 * The SERVER is the source of these rules — `GET /models` publishes them per
 * base model as `base_models[].constraints`, built from the engine adapter's
 * own rule table. This module only (1) flattens that block into the form's
 * vocabulary and (2) holds the LTX values to use while it is absent (a backend
 * that predates the key, or before the first `GET /models` lands) — the same
 * constants Chained, Edit and Batch use.
 *
 * The rule the shell follows for engines holds here too: no screen ever reads
 * an engine id. A screen is handed a {@link GenerationConstraints} and reads
 * numbers off it; which engine produced them is the server's fact.
 *
 * Only Create (Single) and its right-click seeds read this. Chained, Edit and
 * Batch keep their LTX constants on purpose: an engine whose rules differ
 * closes those screens through its `unsupported_features`.
 */
export interface GenerationConstraints {
  /** The `num_frames` lattice (`step * n + offset`). */
  frameGrid: FrameGrid;
  /** The engine's own `num_frames` floor (on the grid). */
  minNumFrames: number;
  /** The engine's own `num_frames` ceiling. `Infinity` in the LTX fallback,
   * meaning "no cap beyond `config.limits.max_num_frames`" — see
   * {@link frameLimitsFor}. */
  maxNumFrames: number;
  /** Width/height multiple outside IC-LoRA mode. */
  dimensionMultiple: number;
  /** Inclusive frame-rate range. `min === max` = fixed; see
   * {@link fixedFrameRate}. */
  frameRate: { min: number; max: number };
  /** The three sampling sliders, or `null` for an engine that has none (LTX).
   * Verbatim from the server — the slider ranges and defaults are its. */
  sampling: SamplingConstraints | null;
}

/** What the WebUI uses while the server publishes no `constraints` — the LTX
 * rules, built from the same constants the other forms use. */
export const LTX_GENERATION_CONSTRAINTS: Readonly<GenerationConstraints> = Object.freeze({
  frameGrid: LTX_FRAME_GRID,
  minNumFrames: MIN_NUM_FRAMES,
  maxNumFrames: Number.POSITIVE_INFINITY,
  dimensionMultiple: LTX_DIMENSION_MULTIPLE,
  frameRate: Object.freeze({ min: FRAME_RATE_MIN, max: FRAME_RATE_MAX }),
  sampling: null,
});

/** Flattens a served `constraints` block; `null`/absent → the LTX fallback.
 * Pure. Values are taken as served (the server's rule table is the one place
 * they are decided). */
export function resolveGenerationConstraints(
  raw: BaseModelConstraints | null | undefined,
): GenerationConstraints {
  if (!raw) return LTX_GENERATION_CONSTRAINTS;
  return {
    frameGrid: { step: raw.num_frames.step, offset: raw.num_frames.offset },
    minNumFrames: raw.num_frames.min,
    maxNumFrames: raw.num_frames.max,
    dimensionMultiple: raw.dimension_multiple,
    frameRate: { min: raw.frame_rate.min, max: raw.frame_rate.max },
    sampling: raw.sampling ?? null,
  };
}

/** The frame rate the engine is fixed at, or `null` when the user may choose
 * (`min !== max`). A fixed rate disables Create's fps field and is what every
 * request carries, ahead of the right-click fps policy. */
export function fixedFrameRate(constraints: GenerationConstraints): number | null {
  return constraints.frameRate.min === constraints.frameRate.max ? constraints.frameRate.min : null;
}

/** Create's `num_frames` bounds: the engine's floor, and the smaller of the
 * engine's ceiling and the app config's, floored onto the grid so
 * `snapNumFrames` can rely on both ends being grid values. With the LTX
 * fallback this is `[MIN_NUM_FRAMES, config max]`. */
export function frameLimitsFor(
  constraints: GenerationConstraints,
  configMaxNumFrames: number,
): { minNumFrames: number; maxNumFrames: number } {
  const ceiling = Math.min(configMaxNumFrames, constraints.maxNumFrames);
  return {
    minNumFrames: constraints.minNumFrames,
    maxNumFrames: Math.max(constraints.minNumFrames, floorToGrid(ceiling, constraints.frameGrid)),
  };
}
