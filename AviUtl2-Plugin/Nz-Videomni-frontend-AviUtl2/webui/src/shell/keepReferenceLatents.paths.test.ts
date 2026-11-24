import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ACCELERATION_DEFAULTS, accelerationRequestFields } from "./accelerationSettings";
import type { AccelerationSettings } from "./accelerationSettings";
import { useGenerationForm } from "../modes/single/useGenerationForm";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { buildChainRequest } from "../modes/chained/chainUtils";
import { buildI2vGeneratePayload } from "../modes/batch/buildI2vGeneratePayload";

/**
 * 台帳 §1-297 — the cross-path regression test for `keep_reference_latents`
 * (keep reference latents resident), the `vram16Mode.paths.test.ts` shape.
 * Each submission path builds its body with its own code, so a field wired
 * into `accelerationRequestFields` can still fail to reach the wire on one of
 * them:
 *
 *   ON  → every body carries `keep_reference_latents: true`;
 *   OFF (the server default) → no body carries the key at all.
 *
 * The second half is what keeps LTX requests byte-identical: on LTX the row is
 * hidden and the stored choice is written back to OFF (see
 * `AppShell.featureScope.test.tsx`), so an LTX body never mentions the key.
 */
describe("keep_reference_latents reaches the submission paths (§1-297)", () => {
  const ON: AccelerationSettings = { ...ACCELERATION_DEFAULTS, keepReferenceLatents: true };

  /** Path 1: Create's single `/generate`. */
  function singleBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration }),
    ).result.current.toGenerateRequest() as unknown as Record<string, unknown>;
  }

  /** Path 2: Chained's `/generate/chain`. */
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
      acceleration,
    }) as unknown as Record<string, unknown>;
  }

  /** Path 3: Batch i2v's (and H3 Batch A2V's) per-row `/generate`. */
  function batchBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return buildI2vGeneratePayload({
      prompt: "a cat",
      width: 768,
      height: 512,
      cropOutput: null,
      numFrames: 124,
      frameRate: 24,
      seed: 42,
      acceleration,
    }) as unknown as Record<string, unknown>;
  }

  it("on: every path's body carries keep_reference_latents: true", () => {
    for (const body of [singleBody(ON), chainBody(ON), batchBody(ON)]) {
      expect(body.keep_reference_latents).toBe(true);
    }
  });

  it("off (server default): no path's body carries the key at all", () => {
    for (const body of [
      singleBody(ACCELERATION_DEFAULTS),
      chainBody(ACCELERATION_DEFAULTS),
      batchBody(ACCELERATION_DEFAULTS),
    ]) {
      expect(body).not.toHaveProperty("keep_reference_latents");
    }
  });

  it("all paths agree with the single builder they share", () => {
    expect(accelerationRequestFields(ON)).toEqual({ keep_reference_latents: true });
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
  });
});
