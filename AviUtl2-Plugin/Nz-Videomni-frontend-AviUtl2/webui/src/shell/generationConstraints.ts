import type { BaseModelConstraints, SamplingConstraints, SamplingParamConstraint } from "../api/types";
import {
  floorToGrid,
  FRAME_RATE_MAX,
  FRAME_RATE_MIN,
  LTX_DIMENSION_MULTIPLE,
  LTX_FRAME_GRID,
} from "../modes/single/paramUtils";
import type { FrameGrid } from "../modes/single/paramUtils";
import { MIN_NUM_FRAMES } from "../modes/single/defaultConfig";
import type { ReferenceRules } from "../modes/single/referenceRules";

/**
 * The loaded base model's numeric rules, in the shape Create, Chained (and the
 * right-click seeds that land in them) read them.
 *
 * The SERVER is the source of these rules — `GET /models` publishes them per
 * base model as `base_models[].constraints`, built from the engine adapter's
 * own rule table. This module only (1) flattens that block into the form's
 * vocabulary and (2) holds the LTX values to use while it is absent (a backend
 * that predates the key, or before the first `GET /models` lands) — the same
 * constants Edit and Batch use.
 *
 * The rule the shell follows for engines holds here too: no screen ever reads
 * an engine id. A screen is handed a {@link GenerationConstraints} and reads
 * numbers off it; which engine produced them is the server's fact.
 *
 * The numeric rules are read by Create (Single), Chained (§1-106), Edit's
 * Retake (§1-123), Edit's Outpainting (§1-125) and their right-click seeds;
 * the other Edit sub-tabs and
 * Batch keep their LTX constants on purpose (an engine whose rules differ
 * closes those screens through its `unsupported_features`). {@link GenerationConstraints.promptTokenLimit} is
 * the exception: the shell reads it for the prompt token counters of every
 * screen.
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
  /** Where a keyframe may sit: `0` and `offset + k * step` (LTX 8/1, MiniMax
   * H3 1/0 = any frame). Read by the keyframe panel (`keyframeGrid.ts`). */
  keyframeGrid: FrameGrid;
  /** Prompt-token limit shown beside the token counters (§1-111), or `null`
   * when the limit is not confirmed for the engine (H3) or the server has not
   * said (fallback). */
  promptTokenLimit: number | null;
  /** Single-clip A2V's Strength slider (range, step, default), or `null` when
   * the engine takes no source audio on `/generate` (LTX, fallback) — A2V
   * then goes through `/generate/chain`. Verbatim from the server. */
  singleSourceAudio: { strength: SamplingParamConstraint } | null;
  /** §1-107: the Ref2VA reference limits, camelCased, or `null`
   * when the engine takes no references (LTX, fallback) — Create then shows
   * no Reference accordion. Its presence alone decides that (no featureScope
   * entry, like {@link singleSourceAudio}). */
  reference: ReferenceRules | null;
  /** §1-106: the server-fixed chain seam, flattened from `constraints.chain`
   * (`overlapFrames` in PIXEL frames), or `null` when the server publishes no
   * `chain` block (LTX, fallback) — Chained then keeps its own seam-blend
   * controls. Its presence alone switches Chained into the fixed-overlap mode
   * (no featureScope entry, like {@link reference}). §1-120: `sourceAudio`
   * (flattened from `constraints.chain.source_audio`, MiniMax H3) is present
   * only when the server takes a per-clip A2V on the chain; its presence opens
   * the long-audio card's "distribute" mode. The key is left out entirely when
   * the server does not publish it. */
  chain: {
    overlapFrames: number;
    sourceAudio?: { strength: SamplingParamConstraint };
  } | null;
  /** §1-123: the Retake middle-audio Strength slider (range, step, default),
   * flattened from `constraints.retake`, or `null` when the server publishes
   * no `retake` block (LTX, fallback). Its presence alone switches the Retake
   * form into that engine's shape — Strength slider instead of the audio
   * radio, no Stage-2 window, sampling sliders (no featureScope entry, like
   * {@link chain}). */
  retake: { audioStrength: SamplingParamConstraint } | null;
  /** §1-125: the Outpainting source-audio Strength slider and the mask-blur
   * (stage 1) slider (range, step, default), flattened from
   * `constraints.outpaint`, or `null` when the server publishes no `outpaint`
   * block (LTX, fallback). Its presence alone switches the Outpainting form
   * into that engine's shape — no pinned control LoRA, `source_video_id`
   * instead of `reference_video_id`, the canvas on {@link dimensionMultiple},
   * the frames on {@link frameGrid}, sampling sliders (no featureScope entry,
   * like {@link retake}). */
  outpaint: { audioStrength: SamplingParamConstraint; blendDilationStage1: SamplingParamConstraint } | null;
  /** §1-124: the Inpainting source-audio Strength slider and the feather
   * (stage 1) slider (range, step, default), flattened from
   * `constraints.inpaint`, or `null` when the server publishes no `inpaint`
   * block (LTX, fallback). Its presence alone switches the Inpainting form
   * into that engine's shape (no featureScope entry, like {@link outpaint}). */
  inpaint: { audioStrength: SamplingParamConstraint; blendDilationStage1: SamplingParamConstraint } | null;
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
  keyframeGrid: Object.freeze({ step: 8, offset: 1 }),
  promptTokenLimit: null,
  singleSourceAudio: null,
  reference: null,
  chain: null,
  retake: null,
  outpaint: null,
  inpaint: null,
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
    keyframeGrid: raw.keyframes
      ? { step: raw.keyframes.step, offset: raw.keyframes.offset }
      : LTX_GENERATION_CONSTRAINTS.keyframeGrid,
    promptTokenLimit: raw.prompt_token_limit ?? null,
    singleSourceAudio: raw.source_audio ?? null,
    reference: raw.reference
      ? {
          maxImages: raw.reference.max_images,
          maxVideos: raw.reference.max_videos,
          maxAudios: raw.reference.max_audios,
          maxTotal: raw.reference.max_total,
          videoTotalSec: raw.reference.video_total_sec,
          audioTotalSec: raw.reference.audio_total_sec,
          clipMinSec: raw.reference.clip_min_sec,
        }
      : null,
    chain: raw.chain
      ? raw.chain.source_audio
        ? {
            overlapFrames: raw.chain.overlap_frames,
            sourceAudio: { strength: raw.chain.source_audio.strength },
          }
        : { overlapFrames: raw.chain.overlap_frames }
      : null,
    retake: raw.retake ? { audioStrength: raw.retake.audio_strength } : null,
    outpaint: raw.outpaint
      ? {
          audioStrength: raw.outpaint.audio_strength,
          blendDilationStage1: raw.outpaint.blend_dilation_stage1,
        }
      : null,
    inpaint: raw.inpaint
      ? {
          audioStrength: raw.inpaint.audio_strength,
          blendDilationStage1: raw.inpaint.blend_dilation_stage1,
        }
      : null,
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
