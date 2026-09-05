import { describe, expect, it } from "vitest";
import type { AppLimits } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { isValidNumFrames } from "../modes/single/paramUtils";
import type { AccelerationSettings } from "./accelerationSettings";
import { ACCELERATION_DEFAULTS } from "./accelerationSettings";
import {
  comfortFramesForBudget,
  resolveComfortRow,
  resolveSingleComfortBudget,
  SINGLE_COMFORT_TOKEN_BUDGET,
} from "./comfortTable";

/** All five toggles at the all-on configuration the `ltx` row requires —
 * `attentionBackend: "sage"` and `vaeMode: "prune_vaed"` are the two that
 * DIFFER from `ACCELERATION_DEFAULTS`.
 *
 * `keepResidentEmbeddings` (台帳 §3-114) stays at its SERVER DEFAULT here, and
 * deliberately: no served row's `requires` map mentions the key, so matching
 * cannot depend on it either way (`matchesRequires` iterates the row's keys).
 * The calibration those rows were measured at is the five above — writing the
 * sixth as `true` would suggest it took part. */
function allOn(overrides: Partial<AccelerationSettings> = {}): AccelerationSettings {
  return {
    attentionBackend: "sage",
    blockSwapPrefetch: true,
    keepResident: true,
    fusedGgufDequantKernel: true,
    vaeMode: "prune_vaed",
    keepResidentEmbeddings: false,
    ...overrides,
  };
}

/** The served limits, with `comfort_budgets` overridable per test. The base is
 * `FALLBACK_APP_CONFIG.limits`, which mirrors the backend's own default table. */
function limitsWith(overrides: Partial<AppLimits> = {}): AppLimits {
  return { ...FALLBACK_APP_CONFIG.limits, ...overrides };
}

/** The served limits with NO table at all — a backend older than 2026-08-31. */
function limitsWithoutTable(overrides: Partial<AppLimits> = {}): AppLimits {
  const limits = limitsWith(overrides);
  delete limits.comfort_budgets;
  return limits;
}

describe("resolveComfortRow", () => {
  describe("the served table (ltx / ltx25)", () => {
    it("matches the ltx row when all five toggles are effectively on", () => {
      const row = resolveComfortRow(limitsWith(), "ltx", allOn(), true);
      expect(row).toEqual({
        singleBudget: 44880,
        chainBudget: 40000,
        spatialFactor: 32,
        temporalFactor: 8,
        rowIndex: 0,
      });
    });

    it("returns null for ltx with the default (all-off) acceleration — the LEGACY table is correct there", () => {
      // Not a bug: LTX 2.3's default configuration deliberately has no row,
      // because its comfort boundary is non-monotone in token count.
      expect(resolveComfortRow(limitsWith(), "ltx", ACCELERATION_DEFAULTS, true)).toBeNull();
    });

    it("returns null for ltx with any single toggle off", () => {
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ attentionBackend: "sdpa" }), true)).toBeNull();
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ blockSwapPrefetch: false }), true)).toBeNull();
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ keepResident: false }), true)).toBeNull();
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ fusedGgufDequantKernel: false }), true)).toBeNull();
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ vaeMode: "default" }), true)).toBeNull();
    });

    it("treats unknown sage availability (null) as satisfying the row, an explicit false as not", () => {
      expect(resolveComfortRow(limitsWith(), "ltx", allOn(), null)?.rowIndex).toBe(0);
      // The server would silently run sdpa for this job.
      expect(resolveComfortRow(limitsWith(), "ltx", allOn(), false)).toBeNull();
    });

    it("reads keep_resident as the EFFECTIVE value it is handed, not a stored choice", () => {
      // `AppShell` folds keepResident down before this ever runs (§1-10); the
      // folded-down object must stop matching the row.
      expect(resolveComfortRow(limitsWith(), "ltx", allOn({ keepResident: false }), true)).toBeNull();
    });

    it("matches the ltx25 row under ANY acceleration configuration (requires: {})", () => {
      for (const acceleration of [ACCELERATION_DEFAULTS, allOn(), allOn({ vaeMode: "default" })]) {
        expect(resolveComfortRow(limitsWith(), "ltx25", acceleration, false)).toEqual({
          singleBudget: 44880,
          chainBudget: 44880,
          spatialFactor: 32,
          temporalFactor: 8,
          rowIndex: 0,
        });
      }
    });

    it("returns null for an engine family the table has no profile for", () => {
      expect(resolveComfortRow(limitsWith(), "wan", allOn(), true)).toBeNull();
    });

    it("returns null for a profile with an empty rows list", () => {
      const limits = limitsWith({
        comfort_budgets: { ltx: { spatial_factor: 32, temporal_factor: 8, rows: [] } },
      });
      expect(resolveComfortRow(limits, "ltx", allOn(), true)).toBeNull();
    });

    // §3-135 / 裁定 J1 (2026-09-05): 画角拡張の線 (`outpaint_budget`) は `rows`
    // の外にある固定線であり、行の照合には一切関与しない。ここが緑であるかぎり
    // 「画角拡張の線を足したら単発・連結のマーカーが動いた」は起こらない。
    it("ignores outpaint_budget entirely — it takes no part in row matching (J1)", () => {
      const rows = [{ requires: { vae_mode: "prune_vaed" }, single_budget: 30000, chain_budget: 31000 }];
      const withoutLine = limitsWith({
        comfort_budgets: { ltx: { spatial_factor: 32, temporal_factor: 8, rows } },
      });
      const withLine = limitsWith({
        comfort_budgets: { ltx: { spatial_factor: 32, temporal_factor: 8, rows, outpaint_budget: 42240 } },
      });

      // 一致する構成: 線があってもなくても解決結果は同じ（余分な鍵も生えない）。
      expect(resolveComfortRow(withLine, "ltx", allOn(), true)).toEqual(
        resolveComfortRow(withoutLine, "ltx", allOn(), true),
      );
      expect(resolveComfortRow(withLine, "ltx", allOn(), true)).toEqual({
        singleBudget: 30000,
        chainBudget: 31000,
        spatialFactor: 32,
        temporalFactor: 8,
        rowIndex: 0,
      });

      // `requires` が一致しない構成: 線があっても行は無いまま。線が行の代役を
      // 務めることはない。
      expect(resolveComfortRow(withLine, "ltx", allOn({ vaeMode: "default" }), true)).toBeNull();

      // 行がまったく無いプロファイルに線だけあっても同じ。
      const lineOnly = limitsWith({
        comfort_budgets: { ltx: { spatial_factor: 32, temporal_factor: 8, rows: [], outpaint_budget: 42240 } },
      });
      expect(resolveComfortRow(lineOnly, "ltx", allOn(), true)).toBeNull();
    });
  });

  describe("row matching order and vocabulary", () => {
    it("takes the FIRST row whose requires all match, not the most specific one", () => {
      const limits = limitsWith({
        comfort_budgets: {
          ltx: {
            spatial_factor: 32,
            temporal_factor: 8,
            rows: [
              { requires: { vae_mode: "prune_vaed" }, single_budget: 10000, chain_budget: 11000 },
              { requires: {}, single_budget: 20000, chain_budget: 21000 },
            ],
          },
        },
      });
      expect(resolveComfortRow(limits, "ltx", allOn(), true)).toMatchObject({
        singleBudget: 10000,
        chainBudget: 11000,
        rowIndex: 0,
      });
      // With prune_vaed off the first row no longer matches, so the
      // unconditional second one does.
      expect(resolveComfortRow(limits, "ltx", allOn({ vaeMode: "default" }), true)).toMatchObject({
        singleBudget: 20000,
        rowIndex: 1,
      });
    });

    it("never matches a row that requires a key this WebUI version does not know", () => {
      const limits = limitsWith({
        comfort_budgets: {
          ltx: {
            spatial_factor: 32,
            temporal_factor: 8,
            rows: [
              { requires: { some_future_toggle: true }, single_budget: 10000, chain_budget: 11000 },
              { requires: {}, single_budget: 20000, chain_budget: 21000 },
            ],
          },
        },
      });
      // Falls through to the unconditional row — the safe side.
      expect(resolveComfortRow(limits, "ltx", allOn(), true)?.rowIndex).toBe(1);
    });
  });

  describe("normalisation of unusable served values", () => {
    function tableWith(row: { single_budget: number; chain_budget: number }, spatial = 32, temporal = 8): AppLimits {
      return limitsWith({
        comfort_budgets: {
          ltx: { spatial_factor: spatial, temporal_factor: temporal, rows: [{ requires: {}, ...row }] },
        },
      });
    }

    it("falls back to the mirrored constants for a budget that is 0, negative or NaN", () => {
      for (const bad of [0, -1, NaN]) {
        const row = resolveComfortRow(tableWith({ single_budget: bad, chain_budget: bad }), "ltx", allOn(), true);
        expect(row?.singleBudget).toBe(44880);
        expect(row?.chainBudget).toBe(40000);
      }
    });

    it("normalises only the bad budget, keeping a good sibling", () => {
      const row = resolveComfortRow(tableWith({ single_budget: 30000, chain_budget: 0 }), "ltx", allOn(), true);
      expect(row?.singleBudget).toBe(30000);
      expect(row?.chainBudget).toBe(40000);
    });

    it("normalises spatial_factor < 1 to 32 and temporal_factor < 1 to 8", () => {
      const row = resolveComfortRow(
        tableWith({ single_budget: 44880, chain_budget: 40000 }, 0, 0),
        "ltx",
        allOn(),
        true,
      );
      expect(row?.spatialFactor).toBe(32);
      expect(row?.temporalFactor).toBe(8);
    });

    it("passes a legitimate non-default factor through untouched (a future engine)", () => {
      const row = resolveComfortRow(
        tableWith({ single_budget: 44880, chain_budget: 40000 }, 16, 4),
        "ltx",
        allOn(),
        true,
      );
      expect(row?.spatialFactor).toBe(16);
      expect(row?.temporalFactor).toBe(4);
    });
  });

  describe("compatibility shim (no served table, or an unknown engine family)", () => {
    it("uses the two legacy scalar keys when the server publishes no table, all five toggles on", () => {
      const row = resolveComfortRow(limitsWithoutTable(), "ltx", allOn(), true);
      expect(row).toEqual({
        singleBudget: 44880,
        chainBudget: 40000,
        spatialFactor: 32,
        temporalFactor: 8,
        rowIndex: -1,
      });
    });

    it("returns null with no served table the moment one toggle is off — exactly the pre-2026-08-31 rule", () => {
      expect(resolveComfortRow(limitsWithoutTable(), "ltx", allOn({ keepResident: false }), true)).toBeNull();
      expect(resolveComfortRow(limitsWithoutTable(), "ltx", ACCELERATION_DEFAULTS, true)).toBeNull();
    });

    it("uses the shim (NOT the served table) while the engine family is still unknown", () => {
      // `GET /models` has not landed yet, or the session is offline. Falling to
      // the table's `ltx`/`ltx25` lookup here would flicker; falling to the
      // legacy table would regress an all-on user's marker to the coarse value.
      expect(resolveComfortRow(limitsWith(), "", allOn(), true)?.rowIndex).toBe(-1);
      expect(resolveComfortRow(limitsWith(), undefined, allOn(), true)?.rowIndex).toBe(-1);
      // ...and the shim's all-on rule still applies with one toggle off.
      expect(resolveComfortRow(limitsWith(), "", allOn({ vaeMode: "default" }), true)).toBeNull();
    });

    it("falls back to the mirrored constants when the legacy scalar keys are unusable too", () => {
      const limits = limitsWithoutTable();
      delete limits.single_comfort_token_budget;
      delete limits.chain_comfort_token_budget;
      expect(resolveComfortRow(limits, "", allOn(), true)).toMatchObject({
        singleBudget: 44880,
        chainBudget: 40000,
      });
    });
  });
});

describe("resolveSingleComfortBudget", () => {
  it("passes through a positive finite published value", () => {
    expect(resolveSingleComfortBudget(50000)).toBe(50000);
    expect(resolveSingleComfortBudget(1)).toBe(1);
  });

  it("falls back to the mirrored constant for every non-usable value", () => {
    expect(resolveSingleComfortBudget(undefined)).toBe(SINGLE_COMFORT_TOKEN_BUDGET);
    expect(resolveSingleComfortBudget(null)).toBe(SINGLE_COMFORT_TOKEN_BUDGET);
    expect(resolveSingleComfortBudget(0)).toBe(SINGLE_COMFORT_TOKEN_BUDGET);
    expect(resolveSingleComfortBudget(-1)).toBe(SINGLE_COMFORT_TOKEN_BUDGET);
    expect(resolveSingleComfortBudget(NaN)).toBe(SINGLE_COMFORT_TOKEN_BUDGET);
  });

  it("mirrors the calibrated 44,880 value", () => {
    expect(SINGLE_COMFORT_TOKEN_BUDGET).toBe(44880);
  });
});

// MIN_NUM_FRAMES=9 / max_num_frames=481, the same bounds
// `FALLBACK_APP_CONFIG.limits` publishes — see `defaultConfig.ts`.
const MIN_FRAMES = 9;
const MAX_FRAMES = 481;
const SF = 32;
const TF = 8;

describe("comfortFramesForBudget", () => {
  // Docs/COMFORT_LIMIT_TABLE.md's calibration table, reproduced independently
  // here — every value below is the closed-form inverse of the token formula at
  // 44,880, not copied from the plan.
  it("matches the calibration table at every anchor resolution", () => {
    expect(comfortFramesForBudget(1920, 1088, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(169);
    expect(comfortFramesForBudget(1280, 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(361);
    expect(comfortFramesForBudget(768, 1280, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(361);
    expect(comfortFramesForBudget(2560, 1472, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(89);
    expect(comfortFramesForBudget(1472, 2560, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(89);
  });

  it("is symmetric under a width/height swap (the token count only cares about area)", () => {
    const pairs: Array<[number, number]> = [
      [1920, 1088],
      [1280, 768],
      [2560, 1472],
      [1344, 768],
    ];
    for (const [w, h] of pairs) {
      expect(comfortFramesForBudget(w, h, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(
        comfortFramesForBudget(h, w, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF),
      );
    }
  });

  it("moves two 8-frame steps per 64px of width", () => {
    expect(comfortFramesForBudget(1344, 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(345);
    expect(comfortFramesForBudget(1408, 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(329);
    expect(comfortFramesForBudget(1216, 704, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(417);
  });

  it("clamps to maxFrames when the raw inverse overshoots it", () => {
    expect(comfortFramesForBudget(320, 320, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(MAX_FRAMES);
    expect(comfortFramesForBudget(512, 320, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(MAX_FRAMES);
  });

  it("clamps to minFrames when the resolution is too large for even one comfortable latent frame", () => {
    // cells = 256*256 = 65,536 > the budget, so the raw inverse's nLatentMax is
    // 0 and the formula alone would go negative — unreachable through any
    // resolution the server actually publishes limits for, but a hand-typed
    // hugely out-of-range value can still reach this function.
    expect(comfortFramesForBudget(8192, 8192, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(MIN_FRAMES);
    expect(comfortFramesForBudget(4096, 4096, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBe(MIN_FRAMES);
  });

  it("returns null when width/height don't reach one full cell (0-division guard)", () => {
    expect(comfortFramesForBudget(0, 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBeNull();
    expect(comfortFramesForBudget(1280, 0, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBeNull();
    // Number("") === 0: a hand-typed blank width/height field reaches here as a
    // bare 0, exactly like the explicit 0 case above.
    expect(comfortFramesForBudget(Number(""), 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBeNull();
    expect(comfortFramesForBudget(-64, 768, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF)).toBeNull();
  });

  it("honours a non-default spatial factor (a finer grid = more cells = fewer frames)", () => {
    // sf=16 quadruples the cell count at the same resolution, so the ceiling
    // drops to roughly a quarter: 1280x768 -> 80*48=3840 cells -> n=11 -> 81.
    expect(comfortFramesForBudget(1280, 768, 44880, MIN_FRAMES, MAX_FRAMES, 16, TF)).toBe(81);
  });

  it("honours a non-default temporal factor", () => {
    // 1920x1088 at sf=32 is 60*34 = 2040 cells, so n = floor(44880/2040) = 22:
    // tf=8 gives the calibrated 169, and tf=4 gives 4*21+1 = 85 — halving the
    // temporal factor halves the frame ceiling, as it must.
    expect(comfortFramesForBudget(1920, 1088, 44880, MIN_FRAMES, MAX_FRAMES, SF, 4)).toBe(85);
    // 2560x1472 is 80*46 = 3680 cells -> n = 12, so tf=4 -> 4*11+1 = 45.
    expect(comfortFramesForBudget(2560, 1472, 44880, MIN_FRAMES, MAX_FRAMES, SF, 4)).toBe(45);
  });

  // Restored from the old `modes/single/spillUtils.test.ts`'s
  // `singleComfortFrames` property sweep (moved here 2026-08-31 along with the
  // function itself) — a representative sweep of resolutions, at the default
  // spatial/temporal factors (32/8), asserting the result always lands on the
  // `8n+1` grid and inside `[minFrames, maxFrames]`.
  it("always returns a value on the 8n+1 grid, within [minFrames, maxFrames], across a representative sweep", () => {
    const sizes: Array<[number, number]> = [
      [256, 128],
      [320, 320],
      [512, 320],
      [640, 448],
      [768, 512],
      [960, 576],
      [1024, 640],
      [1152, 704],
      [1216, 704],
      [1280, 768],
      [1344, 768],
      [1408, 768],
      [1600, 896],
      [1792, 1024],
      [1920, 1088],
      [2048, 1152],
      [2304, 1344],
      [2560, 1472],
      [3072, 1728],
      [4096, 4096],
    ];
    expect(sizes.length).toBe(20);
    for (const [w, h] of sizes) {
      const result = comfortFramesForBudget(w, h, 44880, MIN_FRAMES, MAX_FRAMES, SF, TF);
      expect(result).not.toBeNull();
      expect(isValidNumFrames(result!)).toBe(true);
      expect(result!).toBeGreaterThanOrEqual(MIN_FRAMES);
      expect(result!).toBeLessThanOrEqual(MAX_FRAMES);
    }
  });
});
