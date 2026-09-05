import { describe, expect, it } from "vitest";
import type { AppLimits, EngineComfortProfile } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { resolveOutpaintComfortBudget } from "./outpaintBudget";

/** The served limits, with `comfort_budgets` overridable per test. The base is
 * `FALLBACK_APP_CONFIG.limits`, which mirrors the backend's own default table
 * (`config.py`'s `_default_comfort_budgets`). Same helper shape as
 * `comfortTable.test.ts`. */
function limitsWith(overrides: Partial<AppLimits> = {}): AppLimits {
  return { ...FALLBACK_APP_CONFIG.limits, ...overrides };
}

/** The served limits with NO table at all — a backend older than 2026-08-31. */
function limitsWithoutTable(): AppLimits {
  const limits = limitsWith();
  delete limits.comfort_budgets;
  return limits;
}

/** A one-family table whose `ltx` profile carries exactly the given fields. The
 * `rows` are the real `ltx` rows so it is clear that the outpaint line is read
 * WITHOUT them ever being consulted. */
function tableWith(profile: Partial<EngineComfortProfile>): AppLimits {
  return limitsWith({
    comfort_budgets: {
      ltx: {
        spatial_factor: 32,
        temporal_factor: 8,
        rows: [{ requires: {}, single_budget: 44880, chain_budget: 40000 }],
        ...profile,
      },
    },
  });
}

describe("resolveOutpaintComfortBudget", () => {
  it("reads the served line for each calibrated engine family", () => {
    // 数値の正本はバックエンドの Docs/COMFORT_LIMIT_TABLE.md §9（2026-09-05
    // 全on再較正）。BE の tests/test_comfort_budgets.py が持つ**別々のリテラル**
    // と対になっており、片方だけ動かしても自動では突き合わされない。
    expect(resolveOutpaintComfortBudget(limitsWith(), "ltx")).toBe(42_240);
    expect(resolveOutpaintComfortBudget(limitsWith(), "ltx25")).toBe(46_080);
  });

  it("returns null while the engine family is unknown — no line, so no warning", () => {
    // `GET /models` 未着・オフラインは "" で届く（`activeEngineFamily`）。
    expect(resolveOutpaintComfortBudget(limitsWith(), "")).toBeNull();
    expect(resolveOutpaintComfortBudget(limitsWith(), undefined)).toBeNull();
  });

  it("returns null for a family the served table has no profile for", () => {
    expect(resolveOutpaintComfortBudget(limitsWith(), "wan")).toBeNull();
    // 綴り違いも黙って「線なし」へ。近い名前に寄せる推測はしない。
    expect(resolveOutpaintComfortBudget(limitsWith(), "LTX25")).toBeNull();
  });

  it("returns null for an `Object.prototype` name used as a family name", () => {
    // 素の object index で引いても原型の鍵は穴にならない: `constructor` は
    // 関数であって `.outpaint_budget` を持たないので `undefined` になり、
    // 下の型ガードが null へ落とす。
    expect(resolveOutpaintComfortBudget(limitsWith(), "constructor")).toBeNull();
    expect(resolveOutpaintComfortBudget(limitsWith(), "__proto__")).toBeNull();
    expect(resolveOutpaintComfortBudget(limitsWith(), "toString")).toBeNull();
  });

  it("returns null for a profile that has rows but no outpaint_budget key (an older server)", () => {
    expect(resolveOutpaintComfortBudget(tableWith({}), "ltx")).toBeNull();
  });

  it("returns null for an unusable served value (null, 0, negative, NaN)", () => {
    expect(resolveOutpaintComfortBudget(tableWith({ outpaint_budget: null }), "ltx")).toBeNull();
    expect(resolveOutpaintComfortBudget(tableWith({ outpaint_budget: 0 }), "ltx")).toBeNull();
    expect(resolveOutpaintComfortBudget(tableWith({ outpaint_budget: -1 }), "ltx")).toBeNull();
    expect(resolveOutpaintComfortBudget(tableWith({ outpaint_budget: NaN }), "ltx")).toBeNull();
  });

  it("returns null when the server publishes no comfort_budgets table at all", () => {
    expect(resolveOutpaintComfortBudget(limitsWithoutTable(), "ltx")).toBeNull();
    expect(resolveOutpaintComfortBudget(limitsWithoutTable(), "ltx25")).toBeNull();
  });

  it("passes a legitimate value for a future family straight through", () => {
    const limits = limitsWith({
      comfort_budgets: { wan: { spatial_factor: 16, temporal_factor: 4, rows: [], outpaint_budget: 12_345 } },
    });
    expect(resolveOutpaintComfortBudget(limits, "wan")).toBe(12_345);
  });
});
