import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ACCELERATION_DEFAULTS, accelerationRequestFields } from "./accelerationSettings";
import type { AccelerationSettings } from "./accelerationSettings";
import { useGenerationForm } from "../modes/single/useGenerationForm";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { buildI2vGeneratePayload } from "../modes/batch/buildI2vGeneratePayload";

/**
 * 台帳 §1-101 (2026-10-18) — the cross-path regression test for `vram16_mode`
 * (VRAM 16GB mode), the same shape as `keepResidentEmbeddings.paths.test.ts`.
 *
 * The paths are the `POST /generate` bodies MiniMax H3 can be sent (its chain
 * body, which carries `vram16_mode` since §1-106, is covered by the Chained
 * tests). Each path builds its body with
 * its own code, so a field wired into `accelerationRequestFields` can still
 * fail to reach the wire on one of them:
 *
 *   OFF → every body carries `vram16_mode: false`; at the server default (ON)
 *   no body carries the key at all.
 *
 * The second half is what keeps LTX requests byte-identical: on LTX the row is
 * hidden and the stored choice is written back to ON (see
 * `AppShell.featureScope.test.tsx`), so an LTX body never mentions the key.
 *
 * (`useBatchForm`'s OR-list — which decides whether `acceleration` reaches the
 * Batch runner at all — is covered by `useBatchForm.test.ts`'s i2v case.)
 */
describe("vram16_mode reaches the /generate submission paths (§1-101)", () => {
  const OFF: AccelerationSettings = { ...ACCELERATION_DEFAULTS, vram16Mode: false };

  /** Path 1: Create's single `/generate`. */
  function singleBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration }),
    ).result.current.toGenerateRequest() as unknown as Record<string, unknown>;
  }

  /** Path 2: Batch i2v's per-row `/generate` (its own builder). */
  function batchI2vBody(acceleration: AccelerationSettings): Record<string, unknown> {
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

  it("off: every path's body carries vram16_mode: false", () => {
    for (const body of [singleBody(OFF), batchI2vBody(OFF)]) {
      expect(body.vram16_mode).toBe(false);
    }
  });

  it("server default (on): no path's body carries the key at all", () => {
    for (const body of [singleBody(ACCELERATION_DEFAULTS), batchI2vBody(ACCELERATION_DEFAULTS)]) {
      expect(body).not.toHaveProperty("vram16_mode");
    }
  });

  it("both paths agree with the single builder they share", () => {
    expect(accelerationRequestFields(OFF)).toEqual({ vram16_mode: false });
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
  });
});
