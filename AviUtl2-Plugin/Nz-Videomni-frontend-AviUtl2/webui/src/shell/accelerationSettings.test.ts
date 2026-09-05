import { describe, expect, it } from "vitest";
import {
  ACCELERATION_DEFAULTS,
  effectiveAcceleration,
  effectiveAccelerationFields,
  withServerDefaults,
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

// §3-135: the "cleanup" half of `shell/featureScope.ts`'s table — the field
// NAMES come from the table, the VALUES from `ACCELERATION_DEFAULTS`.
describe("withServerDefaults", () => {
  it("returns the SAME reference for an empty field list", () => {
    // The ordinary case by far: no loaded base model hides a Settings row, so
    // `accelerationResetsFor` answers `[]` and this must cost nothing.
    const before = makeAllOn();
    expect(withServerDefaults(before, [])).toBe(before);
  });

  it("returns the SAME reference when every named field already sits on its server default", () => {
    // This is the bail-out `AppShell`'s cleanup effect depends on: it re-fires
    // whenever the memoized field list is rebuilt, so a fresh object here would
    // re-render every consumer on each `/models` refresh.
    const before = makeAllOn({ vaeMode: "default", keepResidentEmbeddings: false });
    expect(withServerDefaults(before, ["vaeMode"])).toBe(before);
    expect(withServerDefaults(before, ["vaeMode", "keepResidentEmbeddings"])).toBe(before);
  });

  it("writes the named field back to the server default and leaves the other five untouched", () => {
    const before = makeAllOn(); // vaeMode: "prune_vaed"
    const after = withServerDefaults(before, ["vaeMode"]);

    expect(after).not.toBe(before);
    expect(after.vaeMode).toBe(ACCELERATION_DEFAULTS.vaeMode);
    expect(after).toEqual(makeAllOn({ vaeMode: ACCELERATION_DEFAULTS.vaeMode }));
    // Named explicitly too, because "leaves the others alone" is the whole
    // contract: hiding one row must not reset a choice another row still owns.
    expect(after.attentionBackend).toBe("sage");
    expect(after.blockSwapPrefetch).toBe(true);
    expect(after.keepResident).toBe(true);
    expect(after.fusedGgufDequantKernel).toBe(true);
    expect(after.keepResidentEmbeddings).toBe(true);
  });

  it("writes several named fields back in one call", () => {
    // Two features hiding two rows at once — the table's `resets` set is
    // deduplicated and handed over whole, so this is one write, not two.
    const after = withServerDefaults(makeAllOn(), ["vaeMode", "keepResidentEmbeddings"]);
    expect(after.vaeMode).toBe(ACCELERATION_DEFAULTS.vaeMode);
    expect(after.keepResidentEmbeddings).toBe(ACCELERATION_DEFAULTS.keepResidentEmbeddings);
    expect(after).toEqual(
      makeAllOn({
        vaeMode: ACCELERATION_DEFAULTS.vaeMode,
        keepResidentEmbeddings: ACCELERATION_DEFAULTS.keepResidentEmbeddings,
      }),
    );
  });

  it("moves only the fields that are off their default when a list mixes both", () => {
    // `vaeMode` is already default here, `keepResidentEmbeddings` is not: the
    // whole-list "nothing to move" test must not short-circuit on the first
    // field it happens to look at.
    const before = makeAllOn({ vaeMode: "default" });
    const after = withServerDefaults(before, ["vaeMode", "keepResidentEmbeddings"]);
    expect(after).not.toBe(before);
    expect(after.vaeMode).toBe("default");
    expect(after.keepResidentEmbeddings).toBe(ACCELERATION_DEFAULTS.keepResidentEmbeddings);
  });
});
