import { describe, expect, it } from "vitest";
import {
  ACCELERATION_DEFAULTS,
  effectiveAcceleration,
  effectiveAccelerationFields,
  type AccelerationSettings,
} from "./accelerationSettings";

/** Every toggle on — `attentionBackend: "sage"`, `vaeMode: "prune_vaed"` and,
 * since 台帳 §3-114, `keepResidentEmbeddings: true` are the three that DIFFER
 * from `ACCELERATION_DEFAULTS` (the rest are already server-default
 * `true`/`false` as appropriate — see that constant's own doc comment).
 *
 * Note that this is NO LONGER identical to the "smart comfort marker" all-on
 * configuration: the served `comfort_budgets` rows' `requires` maps name the
 * original FIVE keys and say nothing about `keep_resident_embeddings`, so a row
 * matches whatever this sixth field holds (`shell/comfortTable.ts`'s
 * `matchesRequires` iterates the row's keys, not the field bag's). */
function makeAllOn(overrides: Partial<AccelerationSettings> = {}): AccelerationSettings {
  return {
    attentionBackend: "sage",
    blockSwapPrefetch: true,
    keepResident: true,
    fusedGgufDequantKernel: true,
    vaeMode: "prune_vaed",
    keepResidentEmbeddings: true,
    ...overrides,
  };
}

describe("effectiveAccelerationFields", () => {
  it("maps all six toggles onto the server's own request-field vocabulary", () => {
    expect(effectiveAccelerationFields(makeAllOn(), true)).toEqual({
      attention_backend: "sage",
      block_swap_prefetch: true,
      keep_resident: true,
      fused_gguf_dequant_kernel: true,
      vae_mode: "prune_vaed",
      keep_resident_embeddings: true,
    });
  });

  it("always carries every key, unlike accelerationRequestFields' omit-the-default shape", () => {
    // The matcher in `shell/comfortTable.ts` compares key by key, so a field
    // sitting on the server default must still be PRESENT here (it is exactly
    // the field `accelerationRequestFields` would drop).
    const fields = effectiveAccelerationFields(ACCELERATION_DEFAULTS, null);
    expect(Object.keys(fields).sort()).toEqual([
      "attention_backend",
      "block_swap_prefetch",
      "fused_gguf_dequant_kernel",
      "keep_resident",
      "keep_resident_embeddings",
      "vae_mode",
    ]);
    expect(fields).toEqual({
      attention_backend: "sdpa",
      block_swap_prefetch: true,
      keep_resident: false,
      fused_gguf_dequant_kernel: true,
      vae_mode: "default",
      keep_resident_embeddings: false,
    });
  });

  it("keeps sage when availability is unknown (null) — never demotes on unknown", () => {
    expect(effectiveAccelerationFields(makeAllOn(), null).attention_backend).toBe("sage");
  });

  it("demotes sage to sdpa only when the server explicitly reports it unavailable", () => {
    // The server would silently fall back to sdpa for this job, so a row that
    // requires `attention_backend: "sage"` must not match.
    expect(effectiveAccelerationFields(makeAllOn(), false).attention_backend).toBe("sdpa");
  });

  it("reflects each toggle individually left off", () => {
    expect(effectiveAccelerationFields(makeAllOn({ blockSwapPrefetch: false }), true).block_swap_prefetch).toBe(false);
    expect(effectiveAccelerationFields(makeAllOn({ keepResident: false }), true).keep_resident).toBe(false);
    expect(
      effectiveAccelerationFields(makeAllOn({ fusedGgufDequantKernel: false }), true).fused_gguf_dequant_kernel,
    ).toBe(false);
    expect(effectiveAccelerationFields(makeAllOn({ vaeMode: "default" }), true).vae_mode).toBe("default");
    expect(
      effectiveAccelerationFields(makeAllOn({ keepResidentEmbeddings: false }), true).keep_resident_embeddings,
    ).toBe(false);
  });

  it("must be called with the EFFECTIVE settings object, not the raw stored choice", () => {
    // Raw: block-swap prefetch and keep-resident both stored ON, but the
    // server reports the prefetch CAPABILITY itself unavailable
    // (prefetchAvailable=false). This function only reads
    // `blockSwapPrefetch`/`keepResident` off the object it is handed — it has
    // no capability flag of its own — so the RAW object (still `keepResident:
    // true`) over-reports what would actually run.
    const raw = makeAllOn(); // blockSwapPrefetch: true, keepResident: true
    expect(effectiveAccelerationFields(raw, true).keep_resident).toBe(true); // misjudged

    // The effective object (what useAccelerationSettings actually hands to
    // every reader) folds keepResident down to false in this combination
    // (the backend's own engine/worker.py guard does the same), so a row
    // requiring `keep_resident: true` correctly stops matching.
    const effective = effectiveAcceleration(raw, false);
    expect(effective.keepResident).toBe(false);
    expect(effectiveAccelerationFields(effective, true).keep_resident).toBe(false);
  });
});
