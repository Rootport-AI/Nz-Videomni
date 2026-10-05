import { describe, expect, it } from "vitest";
import type { AppLimits } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import type { ComfortDisplayColumn } from "./comfortDisplayTable";
import { comfortDisplayTableFor, resolveComfortCell } from "./comfortDisplayTable";
import * as comfortDisplayTableModule from "./comfortDisplayTable";

/** The served limits, overridable per test. The base is
 * `FALLBACK_APP_CONFIG.limits`, which mirrors the backend's own defaults —
 * `spill_free_frames` for the legacy column and `comfort_budgets` (one row per
 * base model × weight class since §1-31, factors 32/8) for the derived ones.
 * Same helper shape as `comfortTable.test.ts` and `outpaintBudget.test.ts`. */
function limitsWith(overrides: Partial<AppLimits> = {}): AppLimits {
  return { ...FALLBACK_APP_CONFIG.limits, ...overrides };
}

/** A column of each source kind, built here rather than plucked out of the
 * shipped table, so both branches of `resolveComfortCell` are exercised
 * independently of which columns the LTX table happens to carry. */
const LEGACY_COLUMN: ComfortDisplayColumn = {
  id: "test-legacy",
  labelKey: "comfortColumnLtxDefault",
  source: { kind: "legacy" },
};
const BUDGET_COLUMN: ComfortDisplayColumn = {
  id: "test-budget",
  labelKey: "comfortColumnLtx4bit",
  source: { kind: "budget", engineFamily: "ltx", weightClass: "4bit" },
};

/** The shipped column with this id (the test fails loudly when it is gone). */
function shippedColumn(id: string): ComfortDisplayColumn {
  const column = comfortDisplayTableFor("ltx")?.columns.find((c) => c.id === id);
  if (!column) throw new Error(`no shipped column ${id}`);
  return column;
}

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

  it("derives a `budget` column from the served row of its family AND weight class", () => {
    // The closed-form inverse of the token formula at the served 4bit line
    // 42,840 with factors 32/8, reached through the served row.
    const limits = limitsWith();
    expect(limits.comfort_budgets?.ltx?.rows[0]?.single_budget).toBe(42840);
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBe(345);
    expect(resolveComfortCell(limits, "1920x1088", BUDGET_COLUMN)).toBe(161);
    expect(resolveComfortCell(limits, "2560x1472", BUDGET_COLUMN)).toBe(81);
  });

  it("selects the row by its `requires` condition, not by position", () => {
    // Earlier rows under a DIFFERENT condition (sdpa, or another weight class)
    // must not win just because they come first — the all-on 4bit row further
    // down the list is the one that has to match.
    const limits = limitsWith({
      comfort_budgets: {
        ltx: {
          spatial_factor: 32,
          temporal_factor: 8,
          rows: [
            { requires: { weight_class: "4bit", attention_backend: "sdpa" }, single_budget: 10000, chain_budget: 10000 },
            { requires: { weight_class: "8bit" }, single_budget: 11000, chain_budget: 11000 },
            {
              requires: {
                weight_class: "4bit",
                attention_backend: "sage",
                block_swap_prefetch: true,
                keep_resident: true,
                fused_gguf_dequant_kernel: true,
                vae_mode: "prune_vaed",
              },
              single_budget: 42840,
              chain_budget: 42240,
            },
          ],
        },
      },
    });
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBe(345);
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

  it("gives `null` for a `budget` column when no row carries its weight class", () => {
    const limits = limitsWith({
      comfort_budgets: {
        ltx: {
          spatial_factor: 32,
          temporal_factor: 8,
          rows: [{ requires: { weight_class: "8bit" }, single_budget: 32640, chain_budget: 32384 }],
        },
      },
    });
    expect(resolveComfortCell(limits, "1280x768", BUDGET_COLUMN)).toBeNull();
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

  it("carries seven columns: the 2.3 default table, then 2 families × 3 weight classes", () => {
    const columns = comfortDisplayTableFor("ltx")?.columns ?? [];
    expect(columns.map((c) => c.id)).toEqual([
      "ltx-default",
      "ltx-4bit",
      "ltx-8bit",
      "ltx-q6k",
      "ltx25-4bit",
      "ltx25-8bit",
      "ltx25-q6k",
    ]);
    expect(columns.map((c) => c.source)).toEqual([
      { kind: "legacy" },
      { kind: "budget", engineFamily: "ltx", weightClass: "4bit" },
      { kind: "budget", engineFamily: "ltx", weightClass: "8bit" },
      { kind: "budget", engineFamily: "ltx", weightClass: "q6k" },
      { kind: "budget", engineFamily: "ltx25", weightClass: "4bit" },
      { kind: "budget", engineFamily: "ltx25", weightClass: "8bit" },
      { kind: "budget", engineFamily: "ltx25", weightClass: "q6k" },
    ]);
  });

  it("no longer exports any hand-written measured points or lines (§1-31: every number is served)", () => {
    // The three constants that used to live here (`LTX25_Q6_FRAMES`,
    // `LTX25_FP8_SINGLE_BUDGET`, `LTX_FP8_DEFAULT_FRAMES`) and the `static` /
    // `fixedBudget` column kinds are gone.
    expect(Object.keys(comfortDisplayTableModule).sort()).toEqual([
      "COMFORT_DISPLAY_TABLES",
      "COMFORT_TABLE_BY_ENGINE_FAMILY",
      "comfortDisplayTableFor",
      "resolveComfortCell",
    ]);
  });

  it("converts each served line at 1280x768 (the owner's visual-check table)", () => {
    const limits = limitsWith();
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx-8bit"))).toBe(265);
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx-4bit"))).toBe(345);
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx-q6k"))).toBe(353);
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx25-8bit"))).toBe(313);
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx25-4bit"))).toBe(377);
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx25-q6k"))).toBe(353);
    // The 2.3 default column is the measured table, not a line.
    expect(resolveComfortCell(limits, "1280x768", shippedColumn("ltx-default"))).toBe(273);
  });

  it("extends every weight-class line to every row (no hand-written gaps any more)", () => {
    const table = comfortDisplayTableFor("ltx");
    if (!table) throw new Error("no table");
    const limits = limitsWith();
    for (const column of table.columns.filter((c) => c.source.kind === "budget")) {
      for (const resolution of table.rows) {
        expect(resolveComfortCell(limits, resolution, column)).not.toBeNull();
      }
    }
    // Two spot checks off the 1280x768 row: 1920x1088 on the 8bit lines.
    expect(resolveComfortCell(limits, "1920x1088", shippedColumn("ltx-8bit"))).toBe(121);
    expect(resolveComfortCell(limits, "1920x1088", shippedColumn("ltx25-8bit"))).toBe(145);
  });

  it("gives `null` for every weight-class column when its family is not in the served table", () => {
    const limits = limitsWith({ comfort_budgets: {} });
    for (const id of ["ltx-4bit", "ltx-8bit", "ltx-q6k", "ltx25-4bit", "ltx25-8bit", "ltx25-q6k"]) {
      expect(resolveComfortCell(limits, "1280x768", shippedColumn(id))).toBeNull();
    }
  });
});
