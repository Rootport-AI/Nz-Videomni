import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { BaseModelConstraints } from "../api/types";
import { ACCELERATION_DEFAULTS, fastVaeSettings } from "./accelerationSettings";
import type { AccelerationSettings } from "./accelerationSettings";
import { resolveGenerationConstraints } from "./generationConstraints";
import { useGenerationForm } from "../modes/single/useGenerationForm";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { buildChainRequest } from "../modes/chained/chainUtils";
import { buildI2vGeneratePayload } from "../modes/batch/buildI2vGeneratePayload";
import { piH3FixedSampling, samplingRequestFields } from "../modes/single/samplingValues";
import type { SamplingValues } from "../modes/single/samplingValues";

/**
 * Ruling 107 (2026-11-23) — the cross-path regression test for `pi_h3`, the
 * `vram16Mode.paths.test.ts` shape. Each submission path builds its body with
 * its own code, so the pair of rules can still fail to reach the wire on one
 * of them:
 *
 *   pi-H3 on  → the body carries `pi_h3` and NONE of the three sampling values
 *               (the server fills them with the choice's);
 *   pi-H3 off → no `pi_h3` key, and the three values exactly as before.
 *
 * Create (single), Chained and Batch i2v are here; Retake
 * (`useRetakeForm.test.tsx`), Outpainting (`OutpaintingPanel.h3.test.tsx`) and
 * Inpainting (`useInpaintForm.h3.test.tsx`) carry the same pair next to their
 * own H3 fixtures.
 */

const H3_BLOCK: BaseModelConstraints = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  pi_h3: {
    "4steps": { num_inference_steps: 4, shift_video: 12, shift_audio: 3 },
    "2steps": { num_inference_steps: 2, shift_video: 12, shift_audio: 3 },
  },
};
const H3 = resolveGenerationConstraints(H3_BLOCK);

const ON: AccelerationSettings = { ...ACCELERATION_DEFAULTS, piH3: "4steps" };
const SAMPLING: SamplingValues = { numInferenceSteps: 30, shiftVideo: 10, shiftAudio: 4 };
const SAMPLING_KEYS = ["num_inference_steps", "shift_video", "shift_audio"] as const;

/** Path 1: Create's single `/generate`. */
function singleBody(acceleration: AccelerationSettings): Record<string, unknown> {
  return renderHook(() =>
    useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration, constraints: H3 }),
  ).result.current.toGenerateRequest() as unknown as Record<string, unknown>;
}

/** Path 2: Chained's `/generate/chain` (H3's fixed-overlap mode). */
function chainBody(acceleration: AccelerationSettings): Record<string, unknown> {
  return buildChainRequest({
    prompt: "a cat",
    width: 640,
    height: 384,
    frameRate: 24,
    seed: 42,
    overlapFrames: 3,
    overlapStrength: 0.5,
    chunkedUpsample: true,
    clips: [
      { id: "c0", prompt: "", numFrames: 124 },
      { id: "c1", prompt: "", numFrames: 124 },
    ],
    fixedOverlap: true,
    sampling: SAMPLING,
    acceleration,
  }) as unknown as Record<string, unknown>;
}

/** Path 3: Batch i2v's (and H3 Batch A2V's) per-row `/generate`. */
function batchBody(acceleration: AccelerationSettings | undefined): Record<string, unknown> {
  return buildI2vGeneratePayload({
    prompt: "a cat",
    width: 768,
    height: 512,
    cropOutput: null,
    numFrames: 124,
    frameRate: 24,
    seed: 42,
    sampling: SAMPLING,
    ...(acceleration ? { acceleration } : {}),
  }) as unknown as Record<string, unknown>;
}

describe("pi_h3 reaches the submission paths and drops the three sampling values (ruling 107)", () => {
  it("on: every path's body carries pi_h3 and none of the three sampling values", () => {
    for (const body of [singleBody(ON), chainBody(ON), batchBody(ON)]) {
      expect(body.pi_h3).toBe("4steps");
      for (const key of SAMPLING_KEYS) expect(body).not.toHaveProperty(key);
    }
    expect(chainBody({ ...ACCELERATION_DEFAULTS, piH3: "2steps" }).pi_h3).toBe("2steps");
  });

  it("off: no path's body carries pi_h3, and the three values ride along as before", () => {
    const single = singleBody(ACCELERATION_DEFAULTS);
    expect(single).not.toHaveProperty("pi_h3");
    expect(single).toMatchObject({ num_inference_steps: 20, shift_video: 12, shift_audio: 3 });
    for (const body of [chainBody(ACCELERATION_DEFAULTS), batchBody(ACCELERATION_DEFAULTS), batchBody(undefined)]) {
      expect(body).not.toHaveProperty("pi_h3");
      expect(body).toMatchObject({ num_inference_steps: 30, shift_video: 10, shift_audio: 4 });
    }
  });

  it("off: the single body is byte-identical to one built with no acceleration at all", () => {
    const withDefaults = JSON.stringify(singleBody(ACCELERATION_DEFAULTS));
    const without = JSON.stringify(
      renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { constraints: H3 }),
      ).result.current.toGenerateRequest(),
    );
    expect(withDefaults).toBe(without);
  });
});

/** Sol-H3-VAE (2026-11-24): the same three paths. Sol keeps `piH3` at off,
 * so the three sampling values ride along as before, and only `sol_vae: true`
 * is added; while off, no path's body has the key at all. */
describe("sol_vae reaches the submission paths and keeps the three sampling values (Sol-H3-VAE)", () => {
  const SOL: AccelerationSettings = { ...ACCELERATION_DEFAULTS, ...fastVaeSettings("sol") };

  it("on: every path's body carries sol_vae: true, no pi_h3, and the three sampling values", () => {
    const single = singleBody(SOL);
    expect(single.sol_vae).toBe(true);
    expect(single).not.toHaveProperty("pi_h3");
    expect(single).toMatchObject({ num_inference_steps: 20, shift_video: 12, shift_audio: 3 });
    for (const body of [chainBody(SOL), batchBody(SOL)]) {
      expect(body.sol_vae).toBe(true);
      expect(body).not.toHaveProperty("pi_h3");
      expect(body).toMatchObject({ num_inference_steps: 30, shift_video: 10, shift_audio: 4 });
    }
  });

  it("off and pi-H3: no path's body carries the sol_vae key", () => {
    for (const acceleration of [ACCELERATION_DEFAULTS, ON]) {
      for (const body of [singleBody(acceleration), chainBody(acceleration), batchBody(acceleration)]) {
        expect(body).not.toHaveProperty("sol_vae");
      }
    }
  });
});

describe("samplingRequestFields / piH3FixedSampling", () => {
  it("sends the three values only while pi-H3 is off", () => {
    expect(samplingRequestFields(SAMPLING, undefined)).toEqual({
      num_inference_steps: 30,
      shift_video: 10,
      shift_audio: 4,
    });
    expect(samplingRequestFields(SAMPLING, ACCELERATION_DEFAULTS)).toEqual({
      num_inference_steps: 30,
      shift_video: 10,
      shift_audio: 4,
    });
    expect(samplingRequestFields(SAMPLING, ON)).toEqual({});
    expect(samplingRequestFields(null, ACCELERATION_DEFAULTS)).toEqual({});
  });

  it("fixes the sliders at the served values of the choice, and at nothing while off or unpublished", () => {
    expect(piH3FixedSampling("4steps", H3.piH3)).toEqual({ numInferenceSteps: 4, shiftVideo: 12, shiftAudio: 3 });
    expect(piH3FixedSampling("2steps", H3.piH3)).toEqual({ numInferenceSteps: 2, shiftVideo: 12, shiftAudio: 3 });
    expect(piH3FixedSampling("off", H3.piH3)).toBeNull();
    // LTX (no `pi_h3` block) and the fallback.
    expect(piH3FixedSampling("4steps", null)).toBeNull();
  });

  it("resolveGenerationConstraints flattens constraints.pi_h3, and leaves it null where it is absent", () => {
    expect(H3.piH3).toEqual({
      "4steps": { numInferenceSteps: 4, shiftVideo: 12, shiftAudio: 3 },
      "2steps": { numInferenceSteps: 2, shiftVideo: 12, shiftAudio: 3 },
    });
    expect(resolveGenerationConstraints({ ...H3_BLOCK, pi_h3: null }).piH3).toBeNull();
    expect(resolveGenerationConstraints(null).piH3).toBeNull();
  });
});
