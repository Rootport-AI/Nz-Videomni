import { renderHook, waitFor, act } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import {
  ACCELERATION_DEFAULTS,
  ACCELERATION_STORAGE_KEY,
  SPARK_H3_MIN_TOKENS,
  accelerationRequestFields,
  readStoredAcceleration,
  sparkH3Availability,
  sparkH3StepsTooFew,
  sparkH3VideoTooShort,
  withServerDefaults,
} from "./accelerationSettings";
import type { AccelerationSettings } from "./accelerationSettings";
import { accelerationResetsFor, settingsRowsHiddenFor } from "./featureScope";
import { useAccelerationSettings } from "./useAccelerationSettings";
import { useGenerationForm } from "../modes/single/useGenerationForm";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { buildChainRequest } from "../modes/chained/chainUtils";
import { buildI2vGeneratePayload } from "../modes/batch/buildI2vGeneratePayload";
import type { StatusResponse } from "../api/types";

/**
 * 裁定 113 (MINIMAX_H3_DESIGN §6.11) — Spark-H3 in the panel's store: what is
 * persisted, what rides on a request, which engines hide the row, and the two
 * generate-button notes' rules.
 */

const ON: AccelerationSettings = { ...ACCELERATION_DEFAULTS, sparkH3: true };

/** Create's `/generate` body for `acceleration`. */
function singleBody(acceleration: AccelerationSettings): Record<string, unknown> {
  return renderHook(() =>
    useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration }),
  ).result.current.toGenerateRequest() as unknown as Record<string, unknown>;
}

beforeEach(() => {
  window.localStorage.clear();
});

describe("Spark-H3 persistence", () => {
  it("reads the defaults (off, 20%, 2 steps) when nothing is stored", () => {
    const stored = readStoredAcceleration();
    expect(stored.sparkH3).toBe(false);
    expect(stored.sparkH3TopkPercent).toBe(20);
    expect(stored.sparkH3FullSteps).toBe(2);
  });

  it("reads the defaults when the stored object predates Spark-H3 (no keys at all)", () => {
    window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
    const stored = readStoredAcceleration();
    expect(stored.attentionBackend).toBe("sage");
    expect(stored.sparkH3).toBe(false);
    expect(stored.sparkH3TopkPercent).toBe(20);
    expect(stored.sparkH3FullSteps).toBe(2);
  });

  it("restores stored values inside the ranges, ends included", () => {
    for (const [percent, steps] of [
      [1, 0],
      [50, 10],
      [13, 4],
    ] as const) {
      window.localStorage.setItem(
        ACCELERATION_STORAGE_KEY,
        JSON.stringify({ sparkH3: true, sparkH3TopkPercent: percent, sparkH3FullSteps: steps }),
      );
      expect(readStoredAcceleration()).toMatchObject({
        sparkH3: true,
        sparkH3TopkPercent: percent,
        sparkH3FullSteps: steps,
      });
    }
  });

  it("falls back to the default for a value out of range, a fraction, or the wrong type", () => {
    for (const bad of [
      { sparkH3: "yes", sparkH3TopkPercent: 0, sparkH3FullSteps: -1 },
      { sparkH3: 1, sparkH3TopkPercent: 51, sparkH3FullSteps: 11 },
      { sparkH3: null, sparkH3TopkPercent: 12.5, sparkH3FullSteps: 1.5 },
      { sparkH3TopkPercent: "20", sparkH3FullSteps: "2" },
    ]) {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify(bad));
      expect(readStoredAcceleration()).toMatchObject({ sparkH3: false, sparkH3TopkPercent: 20, sparkH3FullSteps: 2 });
    }
  });

  it("the setters write through to localStorage, rounded and clamped into range", async () => {
    const { result } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setSparkH3(true));
    act(() => result.current.setSparkH3TopkPercent(12.4));
    act(() => result.current.setSparkH3FullSteps(4));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toMatchObject({
        sparkH3: true,
        sparkH3TopkPercent: 12,
        sparkH3FullSteps: 4,
      });
    });
    act(() => result.current.setSparkH3TopkPercent(99));
    act(() => result.current.setSparkH3FullSteps(-3));
    expect(result.current.acceleration.sparkH3TopkPercent).toBe(50);
    expect(result.current.acceleration.sparkH3FullSteps).toBe(0);
    // Off keeps the two values stored.
    act(() => result.current.setSparkH3(false));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toMatchObject({
        sparkH3: false,
        sparkH3TopkPercent: 50,
        sparkH3FullSteps: 0,
      });
    });
    const { result: reloaded } = renderHook(() => useAccelerationSettings());
    expect(reloaded.current.acceleration).toMatchObject({
      sparkH3: false,
      sparkH3TopkPercent: 50,
      sparkH3FullSteps: 0,
    });
  });
});

describe("Spark-H3 request fields", () => {
  it("off: not one key is added", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    const offWithValues: AccelerationSettings = {
      ...ACCELERATION_DEFAULTS,
      sparkH3TopkPercent: 13,
      sparkH3FullSteps: 7,
    };
    expect(accelerationRequestFields(offWithValues)).toEqual({});
  });

  it("on: exactly the three keys, the ratio as percent / 100", () => {
    expect(accelerationRequestFields(ON)).toEqual({
      spark_h3: true,
      spark_h3_topk_ratio: 0.2,
      spark_h3_full_steps: 2,
    });
    expect(accelerationRequestFields({ ...ON, sparkH3TopkPercent: 13, sparkH3FullSteps: 0 })).toEqual({
      spark_h3: true,
      spark_h3_topk_ratio: 13 / 100,
      spark_h3_full_steps: 0,
    });
  });

  /** Each submission path builds its body with its own code (the
   * `keepTextConditioning.paths.test.ts` shape); `singleBody` is above. */
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

  it("on: every path's body carries the three keys", () => {
    for (const body of [singleBody(ON), chainBody(ON), batchBody(ON)]) {
      expect(body.spark_h3).toBe(true);
      expect(body.spark_h3_topk_ratio).toBe(0.2);
      expect(body.spark_h3_full_steps).toBe(2);
    }
  });

  it("off: every path's body is exactly what it is without Spark-H3's fields", () => {
    const offWithValues: AccelerationSettings = { ...ACCELERATION_DEFAULTS, sparkH3TopkPercent: 9, sparkH3FullSteps: 9 };
    for (const build of [singleBody, chainBody, batchBody]) {
      const body = build(offWithValues);
      expect(body).not.toHaveProperty("spark_h3");
      expect(body).not.toHaveProperty("spark_h3_topk_ratio");
      expect(body).not.toHaveProperty("spark_h3_full_steps");
      // The whole body equals the defaults' body: the two stored numbers do not leak.
      expect(body).toEqual(build(ACCELERATION_DEFAULTS));
    }
  });
});

describe("Spark-H3 scope and availability", () => {
  it("spark_h3 hides its own row only, and writes back sparkH3 alone", () => {
    expect(settingsRowsHiddenFor(["spark_h3"])).toEqual({
      vae: false,
      keepResidentEmbeddings: false,
      vram16: false,
      fastVae: false,
      keepReferenceLatents: false,
      keepTextConditioning: false,
      teCombinedRead: false,
      sparkH3: true,
      fusedGgufDequantKernel: false,
    });
    expect(settingsRowsHiddenFor([]).sparkH3).toBe(false);
    expect(accelerationResetsFor(["spark_h3"])).toEqual(["sparkH3"]);
  });

  it("the write-back turns Spark-H3 off and keeps the two slider values", () => {
    const stored: AccelerationSettings = { ...ON, sparkH3TopkPercent: 11, sparkH3FullSteps: 5 };
    expect(withServerDefaults(stored, accelerationResetsFor(["spark_h3"]))).toEqual({
      ...stored,
      sparkH3: false,
    });
    // Nothing to move: the same reference comes back.
    const off: AccelerationSettings = { ...ACCELERATION_DEFAULTS, sparkH3TopkPercent: 11 };
    expect(withServerDefaults(off, ["sparkH3"])).toBe(off);
  });

  function statusWith(available: boolean | undefined): StatusResponse {
    return {
      server: "running",
      version: "0.4.0-test",
      host: "127.0.0.1",
      port: 18620,
      pipeline_loaded: false,
      pipeline_type: null,
      gpu: { available: true, name: "test", vram_total_mb: 16000, vram_used_mb: 0, vram_free_mb: 16000 },
      queue: { mode: "single_job_in_memory", pending: 0, running: 0, completed: 0, failed: 0 },
      acceleration: {
        sage_available: true,
        ...(available === undefined ? {} : { spark_h3_available: available }),
      },
    };
  }

  it("an explicit spark_h3_available:false leaves a stored on in the request, all three keys (the Sage rule)", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ sparkH3: true, sparkH3TopkPercent: 15, sparkH3FullSteps: 1 }),
    );
    expect(sparkH3Availability(statusWith(false))).toBe(false);
    // The availability feeds only the Settings button; the settings every
    // submission path reads keep the stored choice, so the server decides
    // (it runs without Spark-H3 and records `spark_h3_used: "on->off"`).
    const { result } = renderHook(() => useAccelerationSettings(false));
    expect(accelerationRequestFields(result.current.acceleration)).toEqual({
      spark_h3: true,
      spark_h3_topk_ratio: 0.15,
      spark_h3_full_steps: 1,
    });
    expect(singleBody(result.current.acceleration)).toMatchObject({
      spark_h3: true,
      spark_h3_topk_ratio: 0.15,
      spark_h3_full_steps: 1,
    });
  });

  it("availability is three-valued: true / false, and null when unknown", () => {
    expect(sparkH3Availability(statusWith(true))).toBe(true);
    expect(sparkH3Availability(statusWith(false))).toBe(false);
    expect(sparkH3Availability(statusWith(undefined))).toBeNull();
    expect(sparkH3Availability(null)).toBeNull();
  });
});

describe("sparkH3StepsTooFew (the steps note)", () => {
  const steps = (n: number) => ({ numInferenceSteps: n });
  const fullSteps = (n: number): AccelerationSettings => ({ ...ON, sparkH3FullSteps: n });

  it("shows when the total is at or below full-attention steps (<, =), not above (>)", () => {
    expect(sparkH3StepsTooFew(fullSteps(2), steps(1), null)).toBe(true);
    expect(sparkH3StepsTooFew(fullSteps(2), steps(2), null)).toBe(true);
    expect(sparkH3StepsTooFew(fullSteps(2), steps(3), null)).toBe(false);
  });

  it("never shows while Spark-H3 is off, or without acceleration", () => {
    expect(sparkH3StepsTooFew({ ...ACCELERATION_DEFAULTS, sparkH3FullSteps: 5 }, steps(1), null)).toBe(false);
    expect(sparkH3StepsTooFew(undefined, steps(1), null)).toBe(false);
  });

  it("never shows when full-attention steps is 0", () => {
    expect(sparkH3StepsTooFew(fullSteps(0), steps(1), null)).toBe(false);
  });

  it("reads pi-H3's fixed steps over the sliders' when they are set", () => {
    // pi-H3 2 steps with full-attention 2: shown although the slider says 8.
    expect(sparkH3StepsTooFew(fullSteps(2), steps(8), steps(2))).toBe(true);
    // pi-H3 4 steps with full-attention 2: not shown although the slider says 1.
    expect(sparkH3StepsTooFew(fullSteps(2), steps(1), steps(4))).toBe(false);
  });

  it("never shows when the screen has no step count at all", () => {
    expect(sparkH3StepsTooFew(fullSteps(2), null, null)).toBe(false);
  });
});

describe("sparkH3VideoTooShort (the short-video note)", () => {
  const rows = (total: number) => ({ total });

  it("names the engine's gate", () => {
    expect(SPARK_H3_MIN_TOKENS).toBe(12288);
  });

  it("shows below 12288 and not at or above it (12287 / 12288)", () => {
    expect(sparkH3VideoTooShort(ON, [rows(12287)])).toBe(true);
    expect(sparkH3VideoTooShort(ON, [rows(12288)])).toBe(false);
    expect(sparkH3VideoTooShort(ON, [rows(12289)])).toBe(false);
  });

  it("never shows while Spark-H3 is off, or without acceleration", () => {
    expect(sparkH3VideoTooShort(ACCELERATION_DEFAULTS, [rows(100)])).toBe(false);
    expect(sparkH3VideoTooShort(undefined, [rows(100)])).toBe(false);
  });

  it("a null estimate (unreadable size or frames) never shows it", () => {
    expect(sparkH3VideoTooShort(ON, [null])).toBe(false);
    expect(sparkH3VideoTooShort(ON, [])).toBe(false);
  });

  it("Chained: shows when any one clip is short", () => {
    expect(sparkH3VideoTooShort(ON, [rows(50000), rows(12287), null])).toBe(true);
    expect(sparkH3VideoTooShort(ON, [rows(50000), rows(12288), null])).toBe(false);
  });
});
