import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ACCELERATION_DEFAULTS, accelerationRequestFields } from "./accelerationSettings";
import type { AccelerationSettings } from "./accelerationSettings";
import { useGenerationForm } from "../modes/single/useGenerationForm";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { buildChainRequest } from "../modes/chained/chainUtils";
import { buildA2vChainPayload } from "../modes/batch/buildA2vChainPayload";
import type { BuildA2vChainPayloadParams } from "../modes/batch/buildA2vChainPayload";

/**
 * 台帳 §3-114 (2026-09-03) — the ONE cross-path regression test for
 * `keep_resident_embeddings`. A direct copy of `vaeMode.paths.test.ts`'s shape,
 * and for the same reason: each of the three submission paths builds its
 * request body with its own code, so a field wired into
 * `accelerationRequestFields` can still fail to reach the wire on one of them
 * (Batch A2V has its own hand-written "is anything off its default" OR-list in
 * `useBatchForm`, and its payload type in `buildA2vChainPayload` is a separate
 * local snake_case interface whose EXCESS properties would be silently dropped
 * by TypeScript's structural check on a spread).
 *
 * The per-path suites each own their key-ORDER assertions; this file asserts
 * the single end-to-end fact:
 *
 *   with the toggle on, ALL THREE bodies carry
 *   `keep_resident_embeddings: true`, and at the server default (`false`) NONE
 *   of them carry the key.
 *
 * That second half is what keeps LTX 2.3 out of a 422: the engine that refuses
 * this field refuses a `true` and ignores an absent key, so a body that never
 * mentions it is a body 2.3 can run. (Which engine is loaded is not this
 * layer's business at all — `AppShell` hides the Settings row and writes the
 * stored choice back to `false`; see `AppShell.featureScope.test.tsx`.)
 *
 * (`useBatchForm`'s OR-list — the one place a batch-only regression can hide
 * that this file cannot see, because it decides whether `acceleration` is
 * handed to the runner at all — is covered by `useBatchForm.test.ts`'s own
 * "every row body carries keep_resident_embeddings" case.)
 */
describe("keep_resident_embeddings reaches all three submission paths (§3-114)", () => {
  const RESIDENT: AccelerationSettings = { ...ACCELERATION_DEFAULTS, keepResidentEmbeddings: true };

  const CHAIN_BASE = {
    prompt: "a cat riding a skateboard",
    width: 512,
    height: 320,
    frameRate: 24,
    seed: -1,
    overlapFrames: 3,
    overlapStrength: 0.5,
    chunkedUpsample: true,
    clips: [{ id: "c0", prompt: "", numFrames: 49 }],
  };

  const A2V_BASE: BuildA2vChainPayloadParams = {
    audioId: "audio-1",
    numFrames: 121,
    prompt: "a cat talking",
    width: 768,
    height: 512,
    frameRate: 24,
    seed: 42,
    chunkedUpsample: true,
  };

  /** Path 1 of 3: Create's single `/generate`. */
  function singleBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration }),
    ).result.current.toGenerateRequest() as unknown as Record<string, unknown>;
  }

  /** Path 2 of 3: Chain's `/generate/chain`. */
  function chainBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return buildChainRequest({ ...CHAIN_BASE, acceleration }) as unknown as Record<string, unknown>;
  }

  /** Path 3 of 3: Batch A2V's per-row `/generate/chain` (its own builder and
   * its own local payload type — the historical weak spot). */
  function batchA2vBody(acceleration: AccelerationSettings): Record<string, unknown> {
    return buildA2vChainPayload({ ...A2V_BASE, acceleration }) as unknown as Record<string, unknown>;
  }

  it("on: every path's body carries keep_resident_embeddings: true", () => {
    for (const body of [singleBody(RESIDENT), chainBody(RESIDENT), batchA2vBody(RESIDENT)]) {
      expect(body.keep_resident_embeddings).toBe(true);
    }
  });

  it("server default: no path's body carries the key at all", () => {
    for (const body of [
      singleBody(ACCELERATION_DEFAULTS),
      chainBody(ACCELERATION_DEFAULTS),
      batchA2vBody(ACCELERATION_DEFAULTS),
    ]) {
      expect(body).not.toHaveProperty("keep_resident_embeddings");
    }
  });

  it("all three paths agree with the single builder they share", () => {
    expect(accelerationRequestFields(RESIDENT)).toEqual({ keep_resident_embeddings: true });
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
  });
});
