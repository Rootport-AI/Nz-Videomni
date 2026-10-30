import { describe, expect, it } from "vitest";
import type { BaseModelConstraints } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import {
  isNumFramesOnGrid,
  isValidDimension,
  isValidNumFrames,
  LTX_FRAME_GRID,
  snapNumFrames,
} from "../modes/single/paramUtils";
import { floorToFrameGrid } from "../timeline/deriveDuration";
import { comfortProfileMissing } from "./comfortTable";
import { disabledModesFor, editSubTabsDisabledFor, nagDisabledFor, singleSectionsDisabledFor } from "./featureScope";
import {
  fixedFrameRate,
  frameLimitsFor,
  LTX_GENERATION_CONSTRAINTS,
  resolveGenerationConstraints,
} from "./generationConstraints";
import { effectiveNag, NAG_OFF } from "./nagSettings";
import type { NagSettings } from "./nagSettings";

// MiniMax H3 第 2 弾 (2026-10-17): the served `constraints` block, the grid
// helpers that take it, the "no comfort profile" test and the feature names H3
// publishes — the pure half of 2c-G1.

/** MiniMax H3's block, verbatim from the plan's §3.3 JSON. */
const H3: BaseModelConstraints = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  keyframes: { step: 1, offset: 0 },
};

/** LTX's block, as the server will publish it. */
const LTX: BaseModelConstraints = {
  num_frames: { step: 8, offset: 1, min: 9, max: 481 },
  dimension_multiple: 64,
  frame_rate: { min: 1, max: 60 },
  sampling: null,
  keyframes: { step: 8, offset: 1 },
};

/** MiniMax H3's `unsupported_features`, derived the way the plan's §3.4 says
 * the adapter derives it, plus `retake` (supervisor ruling 2026-10-17).
 * §1-106: `chain` left (H3 chains from text) and the chain material it cannot
 * take joined: `v2v`, `end_source`, `a2v`, `chain_start_image`. */
const H3_UNSUPPORTED = [
  "loras",
  "reference_video",
  "nag",
  "two_stage_hq",
  "guidance_scale",
  "outpaint",
  "inpaint",
  "alpha_gen",
  "prune_vaed",
  "keep_resident_embeddings",
  "keyframe_strength",
  "v2v",
  "end_source",
  "a2v",
  "chain_start_image",
  "retake",
  "fused_gguf_dequant_kernel",
];

/** Engine-neutral: any base model that cannot chain at all publishes `chain`. */
const A_BASE_MODEL_THAT_CANNOT_CHAIN = ["chain"];

describe("paramUtils grid helpers (17n+5 / 32)", () => {
  const grid = { step: 17, offset: 5 };

  it("snapNumFrames lands on 17n+5 within [22, 481]", () => {
    expect(snapNumFrames(121, 22, 481, grid)).toBe(124);
    expect(snapNumFrames(1, 22, 481, grid)).toBe(22);
    expect(snapNumFrames(10_000, 22, 481, grid)).toBe(481);
    expect(snapNumFrames(345, 22, 481, grid)).toBe(345);
  });

  it("keeps 8n+1 as the default grid (every LTX caller unchanged)", () => {
    expect(snapNumFrames(50, 9, 481)).toBe(49);
    expect(snapNumFrames(50, 9, 481, LTX_FRAME_GRID)).toBe(49);
    expect(isValidNumFrames(121)).toBe(true);
    expect(isValidNumFrames(124)).toBe(false);
    expect(isValidDimension(1280)).toBe(true);
    expect(isValidDimension(1312)).toBe(false);
  });

  it("isValidNumFrames / isNumFramesOnGrid read the passed grid", () => {
    expect(isValidNumFrames(124, grid)).toBe(true);
    expect(isValidNumFrames(121, grid)).toBe(false);
    expect(isNumFramesOnGrid(5, 22, 481, grid)).toBe(false); // on the lattice, below the floor
    expect(isNumFramesOnGrid(22, 22, 481, grid)).toBe(true);
  });

  it("floorToFrameGrid floors onto the passed grid", () => {
    expect(floorToFrameGrid(130, grid)).toBe(124);
    expect(floorToFrameGrid(130)).toBe(129);
  });
});

describe("generationConstraints", () => {
  it("falls back to the LTX rules when the server publishes none", () => {
    expect(resolveGenerationConstraints(undefined)).toBe(LTX_GENERATION_CONSTRAINTS);
    expect(resolveGenerationConstraints(null)).toBe(LTX_GENERATION_CONSTRAINTS);
    expect(LTX_GENERATION_CONSTRAINTS.frameGrid).toEqual({ step: 8, offset: 1 });
    expect(LTX_GENERATION_CONSTRAINTS.minNumFrames).toBe(9);
    expect(LTX_GENERATION_CONSTRAINTS.dimensionMultiple).toBe(64);
    expect(LTX_GENERATION_CONSTRAINTS.sampling).toBeNull();
    expect(LTX_GENERATION_CONSTRAINTS.keyframeGrid).toEqual({ step: 8, offset: 1 });
    expect(fixedFrameRate(LTX_GENERATION_CONSTRAINTS)).toBeNull();
  });

  it("reads the served keyframe grid, and the LTX one when a block predates the key", () => {
    expect(resolveGenerationConstraints(H3).keyframeGrid).toEqual({ step: 1, offset: 0 });
    expect(resolveGenerationConstraints(LTX).keyframeGrid).toEqual({ step: 8, offset: 1 });
    const { keyframes: _omit, ...older } = H3;
    expect(resolveGenerationConstraints(older).keyframeGrid).toEqual({ step: 8, offset: 1 });
  });

  it("copies the served source_audio rule; absent or null (LTX, the fallback) means no single-clip A2V", () => {
    const rule = { strength: { min: 0, max: 1, step: 0.05, default: 1 } };
    expect(resolveGenerationConstraints({ ...H3, source_audio: rule }).singleSourceAudio).toEqual(rule);
    expect(resolveGenerationConstraints({ ...H3, source_audio: null }).singleSourceAudio).toBeNull();
    expect(resolveGenerationConstraints(H3).singleSourceAudio).toBeNull();
    expect(LTX_GENERATION_CONSTRAINTS.singleSourceAudio).toBeNull();
  });

  it("camelCases the served reference limits; absent or null (LTX, the fallback) means no Reference", () => {
    const reference = {
      max_images: 9,
      max_videos: 3,
      max_audios: 3,
      max_total: 12,
      video_total_sec: 15,
      audio_total_sec: 15,
      clip_min_sec: 2,
    };
    expect(resolveGenerationConstraints({ ...H3, reference }).reference).toEqual({
      maxImages: 9,
      maxVideos: 3,
      maxAudios: 3,
      maxTotal: 12,
      videoTotalSec: 15,
      audioTotalSec: 15,
      clipMinSec: 2,
    });
    expect(resolveGenerationConstraints({ ...H3, reference: null }).reference).toBeNull();
    expect(resolveGenerationConstraints(LTX).reference).toBeNull();
    expect(LTX_GENERATION_CONSTRAINTS.reference).toBeNull();
  });

  it("reads the served prompt-token limit; null (H3) and the fallback have none", () => {
    expect(resolveGenerationConstraints({ ...LTX, prompt_token_limit: 1024 }).promptTokenLimit).toBe(1024);
    expect(resolveGenerationConstraints({ ...H3, prompt_token_limit: null }).promptTokenLimit).toBeNull();
    // A block that predates the key reads as "no limit known".
    expect(resolveGenerationConstraints(LTX).promptTokenLimit).toBeNull();
    expect(LTX_GENERATION_CONSTRAINTS.promptTokenLimit).toBeNull();
  });

  it("flattens MiniMax H3's block", () => {
    const c = resolveGenerationConstraints(H3);
    expect(c.frameGrid).toEqual({ step: 17, offset: 5 });
    expect(c.minNumFrames).toBe(22);
    expect(c.maxNumFrames).toBe(481);
    expect(c.dimensionMultiple).toBe(32);
    expect(fixedFrameRate(c)).toBe(24);
    expect(c.sampling?.num_inference_steps.default).toBe(20);
  });

  it("a served LTX block behaves exactly like the fallback", () => {
    const c = resolveGenerationConstraints(LTX);
    expect(fixedFrameRate(c)).toBeNull();
    expect(frameLimitsFor(c, 481)).toEqual(frameLimitsFor(LTX_GENERATION_CONSTRAINTS, 481));
    expect(c.sampling).toBeNull();
  });

  it("frameLimitsFor takes the engine floor and the smaller ceiling, floored onto the grid", () => {
    expect(frameLimitsFor(LTX_GENERATION_CONSTRAINTS, 481)).toEqual({ minNumFrames: 9, maxNumFrames: 481 });
    // The fallback adds no cap of its own (a test config of 4801 keeps it).
    expect(frameLimitsFor(LTX_GENERATION_CONSTRAINTS, 4801)).toEqual({ minNumFrames: 9, maxNumFrames: 4801 });
    const h3 = resolveGenerationConstraints(H3);
    expect(frameLimitsFor(h3, 481)).toEqual({ minNumFrames: 22, maxNumFrames: 481 });
    expect(frameLimitsFor(h3, 4801)).toEqual({ minNumFrames: 22, maxNumFrames: 481 });
    expect(frameLimitsFor(h3, 300)).toEqual({ minNumFrames: 22, maxNumFrames: 294 });
  });
});

describe("comfortProfileMissing", () => {
  const limits = FALLBACK_APP_CONFIG.limits;

  it("is true only for a known family the served table has no profile for", () => {
    expect(comfortProfileMissing(limits, "h3")).toBe(true);
    expect(comfortProfileMissing(limits, "ltx")).toBe(false);
    expect(comfortProfileMissing(limits, "ltx25")).toBe(false);
  });

  it("keeps the legacy fallback for an unknown family or an older backend", () => {
    expect(comfortProfileMissing(limits, "")).toBe(false);
    expect(comfortProfileMissing(limits, undefined)).toBe(false);
    const { comfort_budgets: _omit, ...older } = limits;
    expect(comfortProfileMissing(older, "h3")).toBe(false);
  });

  it("does not read Object.prototype as a profile", () => {
    expect(comfortProfileMissing(limits, "constructor")).toBe(true);
  });
});

describe("MiniMax H3's feature names on screen", () => {
  it("keeps Chained open (§1-106), closes the whole Edit tab and the Create sections H3 cannot use", () => {
    // Supervisor ruling 2026-10-17: the adapter publishes `retake` too (a
    // Retake is a chain request it refuses), so all four Edit sub-tabs close.
    // §1-106: no `chain` any more, so the Chained tab stays open.
    expect(disabledModesFor(H3_UNSUPPORTED)).toEqual(["edit"]);
    expect(editSubTabsDisabledFor(H3_UNSUPPORTED)).toEqual({
      retake: true,
      outpainting: true,
      inpainting: true,
      alphaGen: true,
    });
    // Create's A2V card: H3 publishes `constraints.source_audio`, so it rides
    // `/generate` and stays open (no `chain` closes it any more either).
    expect(singleSectionsDisabledFor(H3_UNSUPPORTED, true)).toEqual({
      referenceVideo: true,
      a2v: false,
      keyframeStrength: true,
    });
    expect(nagDisabledFor(H3_UNSUPPORTED)).toBe(true);
  });

  it("a base model that publishes `chain` loses the Chained tab (and Create's chain-routed A2V card)", () => {
    expect(disabledModesFor(A_BASE_MODEL_THAT_CANNOT_CHAIN)).toEqual(["chained"]);
    expect(singleSectionsDisabledFor(A_BASE_MODEL_THAT_CANNOT_CHAIN).a2v).toBe(true);
  });

  it("closes nothing new on either LTX engine", () => {
    for (const list of [
      ["keep_resident_embeddings", "alpha_gen", "vram16_mode"],
      ["two_stage_hq", "prune_vaed", "vram16_mode"],
      [],
    ]) {
      expect(singleSectionsDisabledFor(list)).toEqual({
        referenceVideo: false,
        a2v: false,
        keyframeStrength: false,
      });
      expect(nagDisabledFor(list)).toBe(false);
    }
  });
});

describe("effectiveNag", () => {
  const on: NagSettings = { ...NAG_OFF, enabled: true, text: "blurry" };

  it("returns the user's settings while NAG is available (same object)", () => {
    expect(effectiveNag(on, true)).toBe(on);
  });

  it("returns NAG_OFF while the engine cannot run NAG", () => {
    expect(effectiveNag(on, false)).toBe(NAG_OFF);
  });
});
