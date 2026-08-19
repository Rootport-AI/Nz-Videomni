import { describe, expect, it } from "vitest";
import {
  ACCELERATION_DEFAULTS,
  effectiveAcceleration,
  isFullAcceleration,
  type AccelerationSettings,
} from "./accelerationSettings";

/** All five toggles at the "smart comfort marker" all-on configuration —
 * `attentionBackend: "sage"` and `vaeMode: "prune_vaed"` are the two that
 * DIFFER from `ACCELERATION_DEFAULTS` (the rest are already server-default
 * `true`/`false` as appropriate — see that constant's own doc comment). */
function makeAllOn(overrides: Partial<AccelerationSettings> = {}): AccelerationSettings {
  return {
    attentionBackend: "sage",
    blockSwapPrefetch: true,
    keepResident: true,
    fusedGgufDequantKernel: true,
    vaeMode: "prune_vaed",
    ...overrides,
  };
}

describe("isFullAcceleration", () => {
  it("is true when all five toggles are on and sage is known available", () => {
    expect(isFullAcceleration(makeAllOn(), true)).toBe(true);
  });

  it("is true when sage availability is unknown (null) — never disables on unknown", () => {
    expect(isFullAcceleration(makeAllOn(), null)).toBe(true);
  });

  it("is false when sage is explicitly reported unavailable, even with sage selected", () => {
    // The server would silently fall back to sdpa for this job, so the
    // all-on gate must not claim sage is really running.
    expect(isFullAcceleration(makeAllOn(), false)).toBe(false);
  });

  it("is false for each toggle individually left off, all others on", () => {
    expect(isFullAcceleration(makeAllOn({ attentionBackend: "sdpa" }), true)).toBe(false);
    expect(isFullAcceleration(makeAllOn({ blockSwapPrefetch: false }), true)).toBe(false);
    expect(isFullAcceleration(makeAllOn({ keepResident: false }), true)).toBe(false);
    expect(isFullAcceleration(makeAllOn({ fusedGgufDequantKernel: false }), true)).toBe(false);
    expect(isFullAcceleration(makeAllOn({ vaeMode: "default" }), true)).toBe(false);
  });

  it("is false for the frozen ACCELERATION_DEFAULTS sentinel (sage/keepResident/vaeMode are not all-on there)", () => {
    expect(isFullAcceleration(ACCELERATION_DEFAULTS, true)).toBe(false);
    expect(isFullAcceleration(ACCELERATION_DEFAULTS, null)).toBe(false);
  });

  it("must be called with the EFFECTIVE settings object, not the raw stored choice", () => {
    // Raw: block-swap prefetch and keep-resident both stored ON, but the
    // server reports the prefetch CAPABILITY itself unavailable
    // (prefetchAvailable=false). `isFullAcceleration` only reads
    // `blockSwapPrefetch`/`keepResident` off the object it's handed — it has
    // no capability flag of its own — so the RAW object (still `keepResident:
    // true`) over-reports "full acceleration".
    const raw = makeAllOn(); // blockSwapPrefetch: true, keepResident: true
    expect(isFullAcceleration(raw, true)).toBe(true); // misjudged: doesn't see the capability gap

    // The effective object (what useAccelerationSettings actually hands to
    // every reader) folds keepResident down to false in this combination
    // (the backend's own engine/worker.py guard does the same), and the gate
    // correctly reports false.
    const effective = effectiveAcceleration(raw, false);
    expect(effective.keepResident).toBe(false);
    expect(isFullAcceleration(effective, true)).toBe(false);
  });
});
