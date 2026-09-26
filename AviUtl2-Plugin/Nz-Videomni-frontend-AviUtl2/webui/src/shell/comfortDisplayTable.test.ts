import { describe, expect, it } from "vitest";
import type { AppLimits } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import type { ComfortDisplayColumn } from "./comfortDisplayTable";
import { comfortDisplayTableFor, resolveComfortCell } from "./comfortDisplayTable";

/** The served limits, overridable per test. The base is
 * `FALLBACK_APP_CONFIG.limits`, which mirrors the backend's own defaults —
 * `spill_free_frames` for the legacy column and `comfort_budgets` (single
 * budget 44,880, factors 32/8) for the derived ones. Same helper shape as
 * `comfortTable.test.ts` and `outpaintBudget.test.ts`. */
function limitsWith(overrides: Partial<AppLimits> = {}): AppLimits {
  return { ...FALLBACK_APP_CONFIG.limits, ...overrides };
}

/** A column of each source kind, built here rather than plucked out of the
 * shipped table, so the four branches of `resolveComfortCell` are exercised
 * independently of which columns the LTX table happens to carry. */
const LEGACY_COLUMN: ComfortDisplayColumn = {
  id: "test-legacy",
  labelKey: "comfortColumnLtxDefault",
  source: { kind: "legacy" },
};
const BUDGET_COLUMN: ComfortDisplayColumn = {
  id: "test-budget",
  labelKey: "comfortColumnLtxAllOn",
  source: { kind: "budget", engineFamily: "ltx" },
};
const STATIC_COLUMN: ComfortDisplayColumn = {
  id: "test-static",
  labelKey: "comfortColumnLtx25Q6",
  source: { kind: "static", frames: { "1280x768": 361 } },
};

describe("comfortDisplayTableFor", () => {
  it("gives both LTX families the same table", () => {
    const ltx = comfortDisplayTableFor("ltx");
    const ltx25 = comfortDisplayTableFor("ltx25");
    expect(ltx).not.toBeNull();
    expect(ltx25).toBe(ltx);
  });

  it("has no table for an unknown family, and none for the unknown-engine empty string", () => {
    // `""` is what `activeEngineFamily` reports before `GET /models` lands and
    // while offline — the section must stay away rather than guess a family.
    expect(comfortDisplayTableFor("")).toBeNull();
    expect(comfortDisplayTableFor("foo")).toBeNull();
  });
});

describe("resolveComfortCell", () => {
  it("reads the legacy table for a `legacy` column, and gives `null` where it has no entry", () => {
    // The served `spill_free_frames` value; 896x1152 is deliberately absent
    // from that table, which is the column's own "—".
    expect(resolveComfortCell(limitsWith(), "1280x768", LEGACY_COLUMN)).toBe(
      FALLBACK_APP_CONFIG.limits.spill_free_frames["1280x768"],
    );
    expect(resolveComfortCell(limitsWith(), "896x1152", LEGACY_COLUMN)).toBeNull();
  });

  it("derives a `budget` column from the served row at the calibrated anchors", () => {
    // The closed-form inverse of the token formula at the served 44,880 with
    // factors 32/8 — the same three anchors `comfortTable.test.ts` pins
    // `comfortFramesForBudget` itself with, reached here through the served
    // row rather than through literals.
    const limits = limitsWith();
    expect(limits.comfort_budgets?.ltx?.rows[0]?.single_budget).toBe(44880);
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBe(361);
    expect(resolveComfortCell(limits, "1920x1088", BUDGET_COLUMN)).toBe(169);
    expect(resolveComfortCell(limits, "2560x1472", BUDGET_COLUMN)).toBe(89);
    expect(resolveComfortCell(limits, "896x1152", BUDGET_COLUMN)).toBe(345);
  });

  it("selects the row by its `requires` condition, not by position", () => {
    // A first row under a DIFFERENT condition (sdpa, not the all-on one the
    // column resolves against) must not win just because it is rows[0] — the
    // all-on row further down the list is the one that has to match.
    const limits = limitsWith({
      comfort_budgets: {
        ltx: {
          spatial_factor: 32,
          temporal_factor: 8,
          rows: [
            { requires: { attention_backend: "sdpa" }, single_budget: 10000, chain_budget: 10000 },
            {
              requires: {
                attention_backend: "sage",
                block_swap_prefetch: true,
                keep_resident: true,
                fused_gguf_dequant_kernel: true,
                vae_mode: "prune_vaed",
              },
              single_budget: 44880,
              chain_budget: 40000,
            },
          ],
        },
      },
    });
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBe(361);
  });

  it("clamps a `budget` column to the served frame ceiling", () => {
    // 512x320 inverts to far more frames than the server will accept, so the
    // cell shows the ceiling — the same clamp the Create marker applies.
    expect(resolveComfortCell(limitsWith(), "512x320", BUDGET_COLUMN)).toBe(
      FALLBACK_APP_CONFIG.limits.max_num_frames,
    );
  });

  it("gives `null` for a `budget` column whose family is not in the served table", () => {
    const limits = limitsWith({ comfort_budgets: {} });
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBeNull();
  });

  it("gives `null` for a `budget` column when the family's profile has no rows", () => {
    const limits = limitsWith({
      comfort_budgets: { ltx: { spatial_factor: 32, temporal_factor: 8, rows: [] } },
    });
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBeNull();
  });

  it("reads a `static` column's own points, and gives `null` for anything it does not carry", () => {
    expect(resolveComfortCell(limitsWith(), "1280x768", STATIC_COLUMN)).toBe(361);
    expect(resolveComfortCell(limitsWith(), "2560x1472", STATIC_COLUMN)).toBeNull();
  });
});

describe("the LTX table as it ships", () => {
  it("resolves exactly one cell per column on every row", () => {
    const table = comfortDisplayTableFor("ltx");
    expect(table).not.toBeNull();
    if (!table) return;
    const limits = limitsWith();
    for (const resolution of table.rows) {
      const cells = table.columns.map((column) => resolveComfortCell(limits, resolution, column));
      expect(cells).toHaveLength(table.columns.length);
      // A cell is a frame count or the "—" placeholder — never NaN, which is
      // what a mis-parsed `"<w>x<h>"` row key would produce.
      for (const cell of cells) {
        expect(cell === null || Number.isInteger(cell)).toBe(true);
      }
    }
  });

  it("carries the calibrated `2.5 Q6` points and leaves the rest of that column empty", () => {
    // 数値の正本はバックエンドの `Docs/COMFORT_LIMIT_TABLE.md` §10。
    const table = comfortDisplayTableFor("ltx25");
    const column = table?.columns.find((c) => c.id === "ltx25-q6");
    expect(column).toBeDefined();
    if (!column) return;
    const limits = limitsWith();
    expect(resolveComfortCell(limits, "1280x768", column)).toBe(361);
    expect(resolveComfortCell(limits, "1920x1088", column)).toBe(161);
    expect(resolveComfortCell(limits, "896x1152", column)).toBe(313);
    expect(resolveComfortCell(limits, "512x320", column)).toBeNull();
    expect(resolveComfortCell(limits, "960x576", column)).toBeNull();
    expect(resolveComfortCell(limits, "2560x1472", column)).toBeNull();
  });

  it("carries six columns, including the two fp8 ones", () => {
    const ids = comfortDisplayTableFor("ltx")?.columns.map((c) => c.id) ?? [];
    expect(ids).toHaveLength(6);
    expect(ids).toContain("ltx-fp8-default");
    expect(ids).toContain("ltx25-fp8");
  });

  it("extends the `2.5 fp8` line to every row, reproducing the three measured points", () => {
    // 数値の正本はバックエンドの `Docs/COMFORT_LIMIT_TABLE.md` 第13節。
    const column = comfortDisplayTableFor("ltx25")?.columns.find((c) => c.id === "ltx25-fp8");
    expect(column).toBeDefined();
    if (!column) return;
    const limits = limitsWith();
    expect(resolveComfortCell(limits, "1280x768", column)).toBe(313);
    expect(resolveComfortCell(limits, "1920x1088", column)).toBe(145);
    expect(resolveComfortCell(limits, "896x1152", column)).toBe(297);
    expect(resolveComfortCell(limits, "2560x1472", column)).toBe(73);
    expect(resolveComfortCell(limits, "512x320", column)).toBe(limits.max_num_frames);
    expect(resolveComfortCell(limits, "960x576", column)).toBe(limits.max_num_frames);
  });

  it("carries the single `2.3 fp8 (default)` point and leaves the rest of that column empty", () => {
    const table = comfortDisplayTableFor("ltx");
    const column = table?.columns.find((c) => c.id === "ltx-fp8-default");
    expect(column).toBeDefined();
    if (!table || !column) return;
    const limits = limitsWith();
    expect(resolveComfortCell(limits, "1920x1088", column)).toBe(121);
    const others = table.rows.filter((r) => r !== "1920x1088");
    expect(others).toHaveLength(5);
    for (const resolution of others) {
      expect(resolveComfortCell(limits, resolution, column)).toBeNull();
    }
  });

  it("gives `null` for the `2.5 fp8` column when its family is not in the served table", () => {
    const column = comfortDisplayTableFor("ltx25")?.columns.find((c) => c.id === "ltx25-fp8");
    expect(column).toBeDefined();
    if (!column) return;
    const limits = limitsWith({ comfort_budgets: {} });
    expect(resolveComfortCell(limits, "1280x768", column)).toBeNull();
  });

  it("gives `null` for the `2.5 fp8` column when its family's profile has no rows", () => {
    const column = comfortDisplayTableFor("ltx25")?.columns.find((c) => c.id === "ltx25-fp8");
    expect(column).toBeDefined();
    if (!column) return;
    const limits = limitsWith({
      comfort_budgets: { ltx25: { spatial_factor: 32, temporal_factor: 8, rows: [] } },
    });
    expect(resolveComfortCell(limits, "1280x768", column)).toBeNull();
  });
});
