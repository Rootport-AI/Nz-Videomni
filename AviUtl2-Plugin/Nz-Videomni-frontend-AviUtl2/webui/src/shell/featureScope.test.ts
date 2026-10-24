import { describe, expect, it } from "vitest";
import {
  accelerationResetsFor,
  batchA2vDisabledFor,
  chainPanelsDisabledFor,
  disabledModesFor,
  disabledUiTargets,
  editSubTabsDisabledFor,
  FEATURE_UI,
  loraAudioStrengthHiddenFor,
  settingsRowsHiddenFor,
} from "./featureScope";

/** The set as a sorted array, so a comparison reads as a value rather than as a
 * sequence of `.has()` calls. */
function targets(unsupportedFeatures: readonly string[]): string[] {
  return [...disabledUiTargets(unsupportedFeatures)].sort();
}

describe("FEATURE_UI (the declaration table)", () => {
  it("names exactly the features that close something", () => {
    // Spelled out rather than counted: a name silently dropped from the table
    // would take its UI's greying with it, and a count would still pass.
    expect(Object.keys(FEATURE_UI).sort()).toEqual(
      [
        "a2v",
        "alpha_gen",
        "chain",
        "end_source",
        "inpaint",
        "keep_resident_embeddings",
        "keyframe_strength",
        "lora_audio_strength",
        "nag",
        "outpaint",
        "prune_vaed",
        "reference_video",
        "retake",
        "v2v",
        "vram16_mode",
        "fused_gguf_dequant_kernel",
      ].sort(),
    );
  });

  it("gives every row at least one target to close", () => {
    // A row with an empty `disables` would be a feature name that reads as
    // "handled" while closing nothing — the one failure mode of a table this
    // shape. A feature with no UI consequence belongs OUT of the table.
    for (const [name, entry] of Object.entries(FEATURE_UI)) {
      expect(entry.disables.length, name).toBeGreaterThan(0);
    }
  });
});

describe("disabledUiTargets", () => {
  it("closes nothing for the ordinary case", () => {
    expect(targets([])).toEqual([]);
  });

  it("ignores names it has never heard of, prototype keys included", () => {
    // `two_stage_hq` is a REAL server feature with no UI of its own; the other
    // two are why the lookup is a `Map` and not a plain object — `FEATURE_UI`
    // indexed by `"constructor"` would answer with `Object`'s.
    expect(targets(["two_stage_hq", "constructor", "__proto__"])).toEqual([]);
  });

  it("closes the Chained tab, the Batch A2V section and Create's A2V card for `chain`", () => {
    // MiniMax H3 第 2 弾: Create's A2V card joined — an A2V submit from Create
    // IS a `POST /generate/chain`.
    expect(targets(["chain"])).toEqual(["createSection.a2v", "createSection.batchA2v", "mode.chained"]);
    // And NOT the Edit tab: a container target closes only when everything it
    // contains has, and `chain` contains neither Edit sub-tab.
    expect(disabledUiTargets(["chain"]).has("mode.edit")).toBe(false);
  });

  it("closes the Edit tab only once ALL FOUR of its sub-tabs have", () => {
    // §3-55 (2026-09-14): Inpainting joined the container's `whenAll` the day
    // it stopped being a mock. A mock closes nothing — a real sub-tab keeps the
    // tab alive on its own — so `retake` + `outpaint` alone no longer settle
    // the tab. Since §3-150 (2026-09-15) no shipped engine publishes any of the
    // three, so the names below are a MECHANISM fixture rather than any
    // engine's list: what is pinned is the rule, which the next engine will
    // need. Alpha Gen 第 2 弾 (2026-10-13) made it four: `alpha_gen` IS
    // published (by LTX 2.3), and 2.3 keeps the tab because the other three run.
    expect(disabledUiTargets(["retake"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["outpaint"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["inpaint"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["alpha_gen"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["retake", "outpaint"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["retake", "outpaint", "inpaint"]).has("mode.edit")).toBe(false);
    expect(disabledUiTargets(["retake", "outpaint", "inpaint", "alpha_gen"]).has("mode.edit")).toBe(true);
    expect(targets(["retake", "outpaint", "inpaint", "alpha_gen"])).toEqual([
      "editSubTab.alphaGen",
      "editSubTab.inpainting",
      "editSubTab.outpainting",
      "editSubTab.retake",
      "mode.edit",
    ]);
  });
});

describe("settingsRowsHiddenFor", () => {
  it("hides no row for the ordinary case", () => {
    expect(settingsRowsHiddenFor([])).toEqual({
      vae: false,
      keepResidentEmbeddings: false,
      vram16: false,
      fusedGgufDequantKernel: false,
    });
    expect(settingsRowsHiddenFor(["chain", "retake", "two_stage_hq"])).toEqual({
      vae: false,
      keepResidentEmbeddings: false,
      vram16: false,
      fusedGgufDequantKernel: false,
    });
  });

  it("maps each feature name onto exactly its own row", () => {
    expect(settingsRowsHiddenFor(["prune_vaed"])).toEqual({
      vae: true,
      keepResidentEmbeddings: false,
      vram16: false,
      fusedGgufDequantKernel: false,
    });
    expect(settingsRowsHiddenFor(["keep_resident_embeddings"])).toEqual({
      vae: false,
      keepResidentEmbeddings: true,
      vram16: false,
      fusedGgufDequantKernel: false,
    });
    expect(settingsRowsHiddenFor(["vram16_mode"])).toEqual({
      vae: false,
      keepResidentEmbeddings: false,
      vram16: true,
      fusedGgufDequantKernel: false,
    });
    expect(settingsRowsHiddenFor(["fused_gguf_dequant_kernel"])).toEqual({
      vae: false,
      keepResidentEmbeddings: false,
      vram16: false,
      fusedGgufDequantKernel: true,
    });
  });

  it("hides all four when all four are refused", () => {
    // Not a hypothetical pairing to guard against — just the honest answer, the
    // same way `editSubTabsDisabledFor` answers two `true`s.
    expect(
      settingsRowsHiddenFor(["prune_vaed", "keep_resident_embeddings", "vram16_mode", "fused_gguf_dequant_kernel"]),
    ).toEqual({
      vae: true,
      keepResidentEmbeddings: true,
      vram16: true,
      fusedGgufDequantKernel: true,
    });
  });
});

describe("accelerationResetsFor", () => {
  it("asks for no write-back when nothing is hidden", () => {
    expect(accelerationResetsFor([])).toEqual([]);
    expect(accelerationResetsFor(["quantum_upscale", "chain", "retake"])).toEqual([]);
  });

  it("names the settings field behind each hidden row", () => {
    expect(accelerationResetsFor(["prune_vaed"])).toEqual(["vaeMode"]);
    expect(accelerationResetsFor(["keep_resident_embeddings"])).toEqual(["keepResidentEmbeddings"]);
    expect(accelerationResetsFor(["vram16_mode"])).toEqual(["vram16Mode"]);
    expect(accelerationResetsFor(["fused_gguf_dequant_kernel"])).toEqual(["fusedGgufDequantKernel"]);
  });

  it("collects both without duplicates", () => {
    const resets = accelerationResetsFor(["prune_vaed", "keep_resident_embeddings", "prune_vaed"]);
    expect(resets).toHaveLength(2);
    expect([...resets].sort()).toEqual(["keepResidentEmbeddings", "vaeMode"]);
  });
});

// The four pure mappers are exported and tested directly: they are the single
// place a server-side FEATURE name becomes a greyed control, and driving them
// through the hook would only make a wrong mapping harder to read. Moved here
// from `useBaseModels.test.ts` with §3-135's table, assertion for assertion —
// unchanged, which is what proves the refactor was an equivalence.

describe("disabledModesFor", () => {
  it("answers [] for the ordinary case", () => {
    expect(disabledModesFor([])).toEqual([]);
  });

  it("chain alone takes down Chained", () => {
    expect(disabledModesFor(["chain"])).toEqual(["chained"]);
  });

  it("Edit survives while ONE of its panels is runnable", () => {
    // The tab hosts Retake, Outpainting, (since §3-55) Inpainting and (since
    // Alpha Gen 第 2 弾) Alpha Gen; losing one — or three — is not losing the tab.
    expect(disabledModesFor(["retake"])).toEqual([]);
    expect(disabledModesFor(["outpaint"])).toEqual([]);
    expect(disabledModesFor(["inpaint"])).toEqual([]);
    expect(disabledModesFor(["alpha_gen"])).toEqual([]);
    expect(disabledModesFor(["retake", "outpaint"])).toEqual([]);
    expect(disabledModesFor(["retake", "outpaint", "inpaint"])).toEqual([]);
    expect(disabledModesFor(["retake", "outpaint", "inpaint", "alpha_gen"])).toEqual(["edit"]);
  });

  it("never disables Single or Inventory, whatever the server sends", () => {
    const everything = [
      "chain", "retake", "end_source", "v2v", "a2v", "two_stage_hq", "outpaint", "inpaint", "alpha_gen",
      "loras", "reference_video", "nag", "prune_vaed", "sage_attention", "keep_resident",
    ];
    expect(disabledModesFor(everything)).toEqual(["chained", "edit"]);
  });

  it("ignores names it has never heard of", () => {
    // This WebUI is older than the server it talks to more often than the
    // reverse; an unknown feature must not disable anything by accident.
    expect(disabledModesFor(["quantum_upscale", "chain"])).toEqual(["chained"]);
  });
});

describe("batchA2vDisabledFor", () => {
  it("is false when nothing is restricted", () => {
    expect(batchA2vDisabledFor([])).toBe(false);
    expect(batchA2vDisabledFor(["outpaint", "nag"])).toBe(false);
  });

  it("is true when either half of what it needs is gone", () => {
    expect(batchA2vDisabledFor(["chain"])).toBe(true);
    expect(batchA2vDisabledFor(["a2v"])).toBe(true);
  });
});

describe("chainPanelsDisabledFor", () => {
  it("answers four falses for the ordinary case", () => {
    expect(chainPanelsDisabledFor([])).toEqual({
      v2v: false,
      a2v: false,
      endSource: false,
      reference: false,
    });
  });

  it("maps each feature name onto exactly its own panel", () => {
    expect(chainPanelsDisabledFor(["v2v"])).toMatchObject({ v2v: true, a2v: false, endSource: false, reference: false });
    expect(chainPanelsDisabledFor(["a2v"])).toMatchObject({ v2v: false, a2v: true, endSource: false, reference: false });
    expect(chainPanelsDisabledFor(["end_source"])).toMatchObject({
      v2v: false,
      a2v: false,
      endSource: true,
      reference: false,
    });
    expect(chainPanelsDisabledFor(["reference_video"])).toMatchObject({
      v2v: false,
      a2v: false,
      endSource: false,
      reference: true,
    });
  });

  it("greys all four for LTX 2.5's v1 chain scope", () => {
    expect(
      chainPanelsDisabledFor(["retake", "end_source", "v2v", "a2v", "reference_video", "outpaint", "inpaint"]),
    ).toEqual({
      v2v: true,
      a2v: true,
      endSource: true,
      reference: true,
    });
  });

  it("does NOT consult `chain` — that decision is made one level up", () => {
    // A base model that cannot chain at all loses the whole tab through
    // `disabledModesFor`, so folding it in here would duplicate the ruling.
    expect(chainPanelsDisabledFor(["chain"])).toEqual({
      v2v: false,
      a2v: false,
      endSource: false,
      reference: false,
    });
  });

  it("ignores names it has never heard of", () => {
    expect(chainPanelsDisabledFor(["quantum_upscale", "a2v"])).toMatchObject({ a2v: true, v2v: false });
  });
});

describe("editSubTabsDisabledFor", () => {
  // The whole table, one row per input, so the mapping can be read off the test
  // the way `chainPanelsDisabledFor`'s can. `outpaint` -> `outpainting` is the
  // one place the server's feature name and the sub-tab's name differ, and it
  // is the single most likely thing to get wrong here.
  const TABLE: ReadonlyArray<{
    unsupported: readonly string[];
    expected: { retake: boolean; outpainting: boolean; inpainting: boolean; alphaGen: boolean };
    why: string;
  }> = [
    {
      unsupported: [],
      expected: { retake: false, outpainting: false, inpainting: false, alphaGen: false },
      why: "an older backend",
    },
    {
      unsupported: ["retake"],
      expected: { retake: true, outpainting: false, inpainting: false, alphaGen: false },
      why: "Retake only",
    },
    {
      unsupported: ["outpaint"],
      expected: { retake: false, outpainting: true, inpainting: false, alphaGen: false },
      why: "Outpainting only",
    },
    {
      unsupported: ["inpaint"],
      expected: { retake: false, outpainting: false, inpainting: true, alphaGen: false },
      why: "Inpainting only — §3-55's own name",
    },
    {
      unsupported: ["alpha_gen"],
      expected: { retake: false, outpainting: false, inpainting: false, alphaGen: true },
      why: "Alpha Gen only — Alpha Gen 第 2 弾's own name",
    },
    {
      unsupported: ["keep_resident_embeddings", "alpha_gen"],
      expected: { retake: false, outpainting: false, inpainting: false, alphaGen: true },
      why: "LTX 2.3's list TODAY — the alpha-gen IC-LoRA is an LTX 2.5 adapter",
    },
    {
      unsupported: ["retake", "outpaint", "inpaint"],
      expected: { retake: true, outpainting: true, inpainting: true, alphaGen: false },
      why: "three of four — Alpha Gen alone keeps the Edit tab",
    },
    {
      unsupported: ["retake", "outpaint", "inpaint", "alpha_gen"],
      expected: { retake: true, outpainting: true, inpainting: true, alphaGen: true },
      why: "all four — the Edit tab itself is gone at this point",
    },
    {
      unsupported: ["retake", "end_source", "two_stage_hq", "outpaint", "nag", "prune_vaed"],
      expected: { retake: true, outpainting: true, inpainting: false, alphaGen: false },
      why: "LTX 2.5's list as it stood before the Retake/End source 開通",
    },
    {
      unsupported: ["two_stage_hq", "outpaint", "nag", "prune_vaed"],
      expected: { retake: false, outpainting: true, inpainting: false, alphaGen: false },
      why: "…and after it: `retake` and `end_source` had left, `outpaint` had not",
    },
    {
      unsupported: ["two_stage_hq", "prune_vaed", "inpaint"],
      expected: { retake: false, outpainting: false, inpainting: true, alphaGen: false },
      why: "LTX 2.5's list for the fortnight §3-55 put ONE mode name on it — kept as the MECHANISM case now that no engine publishes it",
    },
    {
      unsupported: ["two_stage_hq", "prune_vaed"],
      expected: { retake: false, outpainting: false, inpainting: false, alphaGen: false },
      why: "LTX 2.5's list TODAY — §3-150 built the 2.5 inpaint driver and took `inpaint` off again, so NO sub-tab greys on either engine",
    },
  ];

  for (const { unsupported, expected, why } of TABLE) {
    it(`[${unsupported.join(", ")}] -> retake:${expected.retake} outpainting:${expected.outpainting} inpainting:${expected.inpainting} alphaGen:${expected.alphaGen} (${why})`, () => {
      expect(editSubTabsDisabledFor(unsupported)).toEqual(expected);
    });
  }

  it("keys the OUTPAINTING sub-tab on the server's `outpaint`, never on `outpainting`", () => {
    // The negative half of the row above: a build that matched the sub-tab's own
    // name would pass every positive assertion in a fixture that happened to
    // publish both spellings. The server publishes `outpaint`.
    expect(editSubTabsDisabledFor(["outpainting"])).toEqual({
      retake: false,
      outpainting: false,
      inpainting: false,
      alphaGen: false,
    });
    // §3-55: the same trap one row down — the server publishes `inpaint`.
    expect(editSubTabsDisabledFor(["inpainting"])).toEqual({
      retake: false,
      outpainting: false,
      inpainting: false,
      alphaGen: false,
    });
    // Alpha Gen 第 2 弾: and again — the server publishes `alpha_gen`.
    expect(editSubTabsDisabledFor(["alphaGen"])).toEqual({
      retake: false,
      outpainting: false,
      inpainting: false,
      alphaGen: false,
    });
  });

  it("ignores names it has never heard of", () => {
    expect(editSubTabsDisabledFor(["quantum_upscale", "retake"])).toEqual({
      retake: true,
      outpainting: false,
      inpainting: false,
      alphaGen: false,
    });
  });

  it("does NOT consult anything the Edit TAB is decided by", () => {
    // The mirror of `chainPanelsDisabledFor`'s "does NOT consult `chain`":
    // whether the Edit tab exists at all is `disabledModesFor`'s ruling, made
    // from these same two names one level up. Nothing else may reach in here —
    // `chain` in particular takes the Chained tab, not an Edit sub-tab.
    expect(editSubTabsDisabledFor(["chain", "v2v", "a2v", "reference_video"])).toEqual({
      retake: false,
      outpainting: false,
      inpainting: false,
      alphaGen: false,
    });
  });
});

describe("loraAudioStrengthHiddenFor (台帳 §1-104)", () => {
  it("is false for the ordinary case (both LTX engines publish nothing here)", () => {
    expect(loraAudioStrengthHiddenFor([])).toBe(false);
  });

  it("is true once `lora_audio_strength` is published", () => {
    expect(loraAudioStrengthHiddenFor(["chain", "lora_audio_strength"])).toBe(true);
  });

  it("is NOT closed by `loras` itself — that name still has no UI target", () => {
    expect(loraAudioStrengthHiddenFor(["loras"])).toBe(false);
    expect(targets(["loras"])).toEqual([]);
  });
});
