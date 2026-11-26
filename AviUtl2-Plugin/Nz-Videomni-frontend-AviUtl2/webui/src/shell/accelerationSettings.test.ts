import { beforeEach, describe, expect, it } from "vitest";
import {
  ACCELERATION_DEFAULTS,
  ACCELERATION_STORAGE_KEY,
  KEEP_REFERENCE_LATENTS_SERVER_DEFAULT,
  KEEP_TEXT_CONDITIONING_SERVER_DEFAULT,
  accelerationRequestFields,
  readStoredAcceleration,
  writeStoredAcceleration,
  effectiveAcceleration,
  effectiveAccelerationFields,
  fastVaeChoice,
  fastVaeSettings,
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
    // §3-164: the Output row sharing this store, at its server default.
    vram16Mode: true,
    piH3: "off",
    solVae: false,
    keepReferenceLatents: false,
    keepTextConditioning: false,
    sparkH3: false,
    sparkH3TopkPercent: 20,
    sparkH3FullSteps: 2,
    embedMp4Metadata: true,
    ...overrides,
  };
}

describe("effectiveAccelerationFields", () => {
  it("maps all seven toggles onto the server's own request-field vocabulary", () => {
    expect(effectiveAccelerationFields(makeAllOn(), true)).toEqual({
      attention_backend: "sage",
      block_swap_prefetch: true,
      keep_resident: true,
      fused_gguf_dequant_kernel: true,
      vae_mode: "prune_vaed",
      keep_resident_embeddings: true,
      vram16_mode: true,
      // Ruling 107: pi-H3 at its server default. No served row's `requires`
      // names it, so it takes no part in the match.
      pi_h3: "off",
      // Sol-H3-VAE: at its server default, and likewise named by no row.
      sol_vae: false,
      // 台帳 §1-297: likewise named by no row.
      keep_reference_latents: false,
      // 台帳 §1-298: likewise named by no row.
      keep_text_conditioning: false,
      // 裁定 113: likewise named by no row (its two number keys are left out).
      spark_h3: false,
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
      "keep_reference_latents",
      "keep_resident",
      "keep_resident_embeddings",
      "keep_text_conditioning",
      "pi_h3",
      "sol_vae",
      "spark_h3",
      "vae_mode",
      "vram16_mode",
    ]);
    expect(fields).toEqual({
      attention_backend: "sdpa",
      block_swap_prefetch: true,
      keep_resident: false,
      fused_gguf_dequant_kernel: true,
      vae_mode: "default",
      keep_resident_embeddings: false,
      vram16_mode: true,
      pi_h3: "off",
      sol_vae: false,
      keep_reference_latents: false,
      keep_text_conditioning: false,
      spark_h3: false,
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
    expect(effectiveAccelerationFields(makeAllOn({ vram16Mode: false }), true).vram16_mode).toBe(false);
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

/** 台帳 §1-101 (2026-10-18): VRAM 16GB mode (MiniMax H3 only). Default ON, so
 * the key rides along only as `false` — the fused-kernel direction. */
describe("vram16Mode (§1-101)", () => {
  it("defaults to on, matching the H3 server default", () => {
    expect(ACCELERATION_DEFAULTS.vram16Mode).toBe(true);
  });

  it("sends nothing while on", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).not.toHaveProperty("vram16_mode");
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("vram16_mode");
  });

  it("sends vram16_mode: false only while off", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, vram16Mode: false })).toEqual({
      vram16_mode: false,
    });
  });

  it("withServerDefaults writes it back to on (the feature-scope reset on LTX)", () => {
    const after = withServerDefaults(makeAllOn({ vram16Mode: false }), ["vram16Mode"]);
    expect(after.vram16Mode).toBe(true);
    expect(after).toEqual(makeAllOn());
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back an off value", () => {
      writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, vram16Mode: false });
      expect(readStoredAcceleration().vram16Mode).toBe(false);
    });

    it("reads a JSON blob without the key (written before §1-101) as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
      expect(readStoredAcceleration().vram16Mode).toBe(true);
    });

    it("reads the pre-2026-08-01 bare string form as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sdpa");
      expect(readStoredAcceleration().vram16Mode).toBe(true);
    });

    it("reads a malformed value as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ vram16Mode: "no" }));
      expect(readStoredAcceleration().vram16Mode).toBe(true);
    });
  });
});

/** Ruling 107 (2026-11-23): pi-H3 — MiniMax H3's few-step sampler + DWD. */
describe("piH3 (ruling 107)", () => {
  it("defaults to off, matching the server default", () => {
    expect(ACCELERATION_DEFAULTS.piH3).toBe("off");
  });

  it("sends nothing while off", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("pi_h3");
  });

  it("sends pi_h3 only while a choice is selected", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, piH3: "4steps" })).toEqual({ pi_h3: "4steps" });
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, piH3: "2steps" })).toEqual({ pi_h3: "2steps" });
  });

  it("effectiveAccelerationFields carries the choice", () => {
    expect(effectiveAccelerationFields(makeAllOn({ piH3: "2steps" }), true).pi_h3).toBe("2steps");
  });

  it("withServerDefaults writes it back to off (the feature-scope reset on LTX)", () => {
    const after = withServerDefaults(makeAllOn({ piH3: "4steps" }), ["piH3"]);
    expect(after.piH3).toBe("off");
    expect(after).toEqual(makeAllOn());
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back each of the three values", () => {
      for (const piH3 of ["off", "4steps", "2steps"] as const) {
        writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, piH3 });
        expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).piH3).toBe(piH3);
        expect(readStoredAcceleration().piH3).toBe(piH3);
      }
    });

    it("reads a JSON blob without the key (written before ruling 107) as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
      expect(readStoredAcceleration().piH3).toBe("off");
      expect(readStoredAcceleration().attentionBackend).toBe("sage");
    });

    it("reads the pre-2026-08-01 bare string form as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sage");
      expect(readStoredAcceleration().piH3).toBe("off");
    });

    it("reads an unknown value as off", () => {
      for (const bad of ["8steps", "4", true, 4, null, "OFF"]) {
        window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ piH3: bad }));
        expect(readStoredAcceleration().piH3).toBe("off");
      }
    });
  });
});

/** 台帳 §1-297: keep reference latents resident (MiniMax H3). */
describe("keepReferenceLatents (§1-297)", () => {
  it("defaults to off, matching the server default", () => {
    expect(ACCELERATION_DEFAULTS.keepReferenceLatents).toBe(false);
    expect(KEEP_REFERENCE_LATENTS_SERVER_DEFAULT).toBe(false);
  });

  it("sends nothing while off — not even keep_reference_latents: false", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("keep_reference_latents");
  });

  it("sends keep_reference_latents: true only while on, and nothing else", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, keepReferenceLatents: true })).toEqual({
      keep_reference_latents: true,
    });
  });

  it("withServerDefaults writes it back to off (the feature-scope reset on LTX)", () => {
    const after = withServerDefaults(makeAllOn({ keepReferenceLatents: true }), ["keepReferenceLatents"]);
    expect(after.keepReferenceLatents).toBe(false);
    expect(after).toEqual(makeAllOn());
  });

  it("withServerDefaults leaves it alone when another field is the one reset", () => {
    const before = makeAllOn({ keepReferenceLatents: true });
    expect(withServerDefaults(before, ["solVae"]).keepReferenceLatents).toBe(true);
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back an on value", () => {
      writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, keepReferenceLatents: true });
      expect(readStoredAcceleration().keepReferenceLatents).toBe(true);
    });

    it("reads a JSON blob without the key (written before §1-297) as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
      expect(readStoredAcceleration().keepReferenceLatents).toBe(false);
    });

    it("reads the pre-2026-08-01 bare string form as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sdpa");
      expect(readStoredAcceleration().keepReferenceLatents).toBe(false);
    });

    it("reads a malformed value as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ keepReferenceLatents: "yes" }));
      expect(readStoredAcceleration().keepReferenceLatents).toBe(false);
    });
  });
});

/** 台帳 §1-298: text encoder conditioning cache (MiniMax H3). */
describe("keepTextConditioning (§1-298)", () => {
  it("defaults to off, matching the server default", () => {
    expect(ACCELERATION_DEFAULTS.keepTextConditioning).toBe(false);
    expect(KEEP_TEXT_CONDITIONING_SERVER_DEFAULT).toBe(false);
  });

  it("sends nothing while off — not even keep_text_conditioning: false", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("keep_text_conditioning");
  });

  it("sends keep_text_conditioning: true only while on, and nothing else", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, keepTextConditioning: true })).toEqual({
      keep_text_conditioning: true,
    });
  });

  it("is independent of keepReferenceLatents in the request fields", () => {
    expect(
      accelerationRequestFields({ ...ACCELERATION_DEFAULTS, keepReferenceLatents: true, keepTextConditioning: true }),
    ).toEqual({ keep_reference_latents: true, keep_text_conditioning: true });
    expect(
      accelerationRequestFields({ ...ACCELERATION_DEFAULTS, keepReferenceLatents: true, keepTextConditioning: false }),
    ).toEqual({ keep_reference_latents: true });
    expect(
      accelerationRequestFields({ ...ACCELERATION_DEFAULTS, keepReferenceLatents: false, keepTextConditioning: true }),
    ).toEqual({ keep_text_conditioning: true });
  });

  it("withServerDefaults writes it back to off (the feature-scope reset on LTX) and leaves keepReferenceLatents alone", () => {
    const before = makeAllOn({ keepReferenceLatents: true, keepTextConditioning: true });
    const after = withServerDefaults(before, ["keepTextConditioning"]);
    expect(after.keepTextConditioning).toBe(false);
    expect(after.keepReferenceLatents).toBe(true);
  });

  it("withServerDefaults leaves it alone when another field is the one reset", () => {
    const before = makeAllOn({ keepReferenceLatents: true, keepTextConditioning: true });
    expect(withServerDefaults(before, ["keepReferenceLatents"]).keepTextConditioning).toBe(true);
    expect(withServerDefaults(before, ["solVae"]).keepTextConditioning).toBe(true);
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back an on value", () => {
      writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, keepTextConditioning: true });
      expect(readStoredAcceleration().keepTextConditioning).toBe(true);
      expect(readStoredAcceleration().keepReferenceLatents).toBe(false);
    });

    it("reads a JSON blob without the key (written before §1-298) as off, keeping the other value", () => {
      window.localStorage.setItem(
        ACCELERATION_STORAGE_KEY,
        JSON.stringify({ attentionBackend: "sage", keepReferenceLatents: true }),
      );
      expect(readStoredAcceleration().keepTextConditioning).toBe(false);
      expect(readStoredAcceleration().keepReferenceLatents).toBe(true);
    });

    it("reads the pre-2026-08-01 bare string form as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sdpa");
      expect(readStoredAcceleration().keepTextConditioning).toBe(false);
    });

    it("reads a malformed value as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ keepTextConditioning: "yes" }));
      expect(readStoredAcceleration().keepTextConditioning).toBe(false);
    });
  });
});

/** 2026-11-24: Sol-H3-VAE, and the "Fast VAE" row's one rule over `piH3` +
 * `solVae`. */
describe("solVae (Sol-H3-VAE) and the Fast VAE row", () => {
  it("defaults to off, matching the server default", () => {
    expect(ACCELERATION_DEFAULTS.solVae).toBe(false);
  });

  it("sends nothing while off — not even sol_vae: false", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("sol_vae");
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, piH3: "4steps" })).toEqual({ pi_h3: "4steps" });
  });

  it("sends sol_vae: true only while on, without pi_h3", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, solVae: true })).toEqual({ sol_vae: true });
  });

  it("withServerDefaults writes it back to off (the feature-scope reset on LTX)", () => {
    const after = withServerDefaults(makeAllOn({ solVae: true }), ["solVae"]);
    expect(after.solVae).toBe(false);
    expect(after).toEqual(makeAllOn());
  });

  it("fastVaeSettings: Sol turns pi-H3 off, any other choice turns Sol off", () => {
    expect(fastVaeSettings("sol")).toEqual({ piH3: "off", solVae: true });
    expect(fastVaeSettings("4steps")).toEqual({ piH3: "4steps", solVae: false });
    expect(fastVaeSettings("2steps")).toEqual({ piH3: "2steps", solVae: false });
    expect(fastVaeSettings("off")).toEqual({ piH3: "off", solVae: false });
  });

  it("fastVaeChoice: the pi-H3 choice first, then Sol, else off", () => {
    expect(fastVaeChoice({ piH3: "off", solVae: false })).toBe("off");
    expect(fastVaeChoice({ piH3: "4steps", solVae: false })).toBe("4steps");
    expect(fastVaeChoice({ piH3: "2steps", solVae: false })).toBe("2steps");
    expect(fastVaeChoice({ piH3: "off", solVae: true })).toBe("sol");
    // A pair the row cannot write: the pi-H3 choice is what shows.
    expect(fastVaeChoice({ piH3: "2steps", solVae: true })).toBe("2steps");
  });

  it("fastVaeChoice reads back what fastVaeSettings wrote, for every choice", () => {
    for (const choice of ["off", "4steps", "2steps", "sol"] as const) {
      expect(fastVaeChoice(fastVaeSettings(choice))).toBe(choice);
    }
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back an on value", () => {
      writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, solVae: true });
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).solVae).toBe(true);
      expect(readStoredAcceleration().solVae).toBe(true);
    });

    it("reads a JSON blob without the key (written before Sol) as off, keeping its pi-H3 choice", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage", piH3: "4steps" }));
      const stored = readStoredAcceleration();
      expect(stored.solVae).toBe(false);
      expect(stored.piH3).toBe("4steps");
      expect(stored.attentionBackend).toBe("sage");
    });

    it("reads the pre-2026-08-01 bare string form as off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sage");
      expect(readStoredAcceleration().solVae).toBe(false);
    });

    it("reads a malformed value as off", () => {
      for (const bad of ["yes", 1, null, "true"]) {
        window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ solVae: bad }));
        expect(readStoredAcceleration().solVae).toBe(false);
      }
    });

    it("folds a stored pi-H3 choice + Sol pair to the pi-H3 choice with Sol off", () => {
      for (const piH3 of ["4steps", "2steps"] as const) {
        window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ piH3, solVae: true }));
        const stored = readStoredAcceleration();
        expect(stored.piH3).toBe(piH3);
        expect(stored.solVae).toBe(false);
      }
    });

    it("keeps Sol on when the stored pi-H3 choice is off", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ piH3: "off", solVae: true }));
      expect(readStoredAcceleration().solVae).toBe(true);
    });
  });
});

/** §3-164 (2026-09-24): the Settings panel's Output row, stored alongside the
 * acceleration choices but kept out of the comfort matcher's field bag. */
describe("embedMp4Metadata (§3-164)", () => {
  it("defaults to on, matching the server default", () => {
    expect(ACCELERATION_DEFAULTS.embedMp4Metadata).toBe(true);
  });

  it("sends nothing while on — requests stay byte-identical to before the row existed", () => {
    expect(accelerationRequestFields(ACCELERATION_DEFAULTS)).toEqual({});
    expect(accelerationRequestFields(makeAllOn())).not.toHaveProperty("embed_mp4_metadata");
  });

  it("sends embed_mp4_metadata: false only while off", () => {
    expect(accelerationRequestFields({ ...ACCELERATION_DEFAULTS, embedMp4Metadata: false })).toEqual({
      embed_mp4_metadata: false,
    });
  });

  it("is not part of effectiveAccelerationFields (the comfort matcher's key set)", () => {
    const fields = effectiveAccelerationFields(makeAllOn({ embedMp4Metadata: false }), true);
    expect(fields).not.toHaveProperty("embed_mp4_metadata");
  });

  it("withServerDefaults carries it over unchanged (no feature-scope row resets it)", () => {
    const before = makeAllOn({ embedMp4Metadata: false });
    expect(withServerDefaults(before, ["vaeMode"]).embedMp4Metadata).toBe(false);
  });

  describe("localStorage round trip", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("writes and reads back an off value", () => {
      writeStoredAcceleration({ ...ACCELERATION_DEFAULTS, embedMp4Metadata: false });
      expect(readStoredAcceleration().embedMp4Metadata).toBe(false);
    });

    it("reads a JSON blob without the key (written before §3-164) as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
      expect(readStoredAcceleration().embedMp4Metadata).toBe(true);
    });

    it("reads the pre-2026-08-01 bare string form as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sdpa");
      expect(readStoredAcceleration().embedMp4Metadata).toBe(true);
    });

    it("reads a malformed value as on", () => {
      window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ embedMp4Metadata: "no" }));
      expect(readStoredAcceleration().embedMp4Metadata).toBe(true);
    });
  });
});
