import { describe, expect, it } from "vitest";
import { DEFAULT_LANG, DICTIONARIES, en, ja } from "./strings";

/** Recursively collects every leaf's dotted path (a leaf being a string or a
 * function — never descends into a function's internals). Object properties
 * recurse; anything else is a leaf. */
function collectPaths(value: unknown, prefix = ""): string[] {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
      collectPaths(child, prefix ? `${prefix}.${key}` : key),
    );
  }
  return [prefix];
}

function leafType(value: unknown): "function" | "string" | "other" {
  if (typeof value === "function") return "function";
  if (typeof value === "string") return "string";
  return "other";
}

describe("i18n dictionaries (en/ja)", () => {
  it("expose exactly the same set of dotted key paths", () => {
    const enPaths = collectPaths(en).sort();
    const jaPaths = collectPaths(ja).sort();
    expect(jaPaths).toEqual(enPaths);
  });

  it("agree on which leaves are plain strings vs. template functions, at every path", () => {
    const enPaths = collectPaths(en);
    for (const path of enPaths) {
      const enLeaf = path.split(".").reduce<unknown>((obj, key) => (obj as Record<string, unknown>)[key], en);
      const jaLeaf = path.split(".").reduce<unknown>((obj, key) => (obj as Record<string, unknown>)[key], ja);
      expect(leafType(jaLeaf), `mismatched leaf type at "${path}"`).toBe(leafType(enLeaf));
    }
  });

  it("has no empty-string leaves in either dictionary", () => {
    function assertNoEmptyStrings(value: unknown, path: string): void {
      if (typeof value === "string") {
        expect(value.length, `empty string at "${path}"`).toBeGreaterThan(0);
        return;
      }
      if (typeof value === "function") return;
      if (value !== null && typeof value === "object") {
        for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
          assertNoEmptyStrings(child, path ? `${path}.${key}` : key);
        }
      }
    }
    assertNoEmptyStrings(en, "en");
    assertNoEmptyStrings(ja, "ja");
  });

  it("function leaves produce non-empty output for both languages", () => {
    expect(en.connection.connected("1.2.3")).toContain("1.2.3");
    expect(ja.connection.connected("1.2.3")).toContain("1.2.3");
    expect(en.chained.clipLabel(0)).toBe("Clip 1");
    expect(ja.chained.clipLabel(0)).toContain("1");
    // X3: the short-clip-vs-SEAM-BLEND reason interpolates the required minimum
    // frame count (n) into both languages.
    expect(en.chained.generateReasons.clipTooShortForOverlap(25)).toContain("25");
    expect(ja.chained.generateReasons.clipTooShortForOverlap(25)).toContain("25");
  });

  // 素材（末尾）v2 (2026-08-15): the predicted-output BREAKDOWN. v2 appends the
  // material after the clips, so the line has to name both the total and the
  // part of it that is the material — in both languages, with all four numbers.
  it("the 予想出力 breakdown line carries the total AND the tail, in both languages", () => {
    // The plan's own worked example: 569 frames at 24fps = 23.7s, of which the
    // last 72 frames (3.0s) are the attached material.
    const enLine = en.chained.outputFramesWithTailLabel(569, 569 / 24, 72, 72 / 24);
    expect(enLine).toContain("569 frames");
    expect(enLine).toContain("23.7");
    expect(enLine).toContain("72 frames");
    expect(enLine).toContain("3.0");
    expect(enLine).toContain("are the material");

    const jaLine = ja.chained.outputFramesWithTailLabel(569, 569 / 24, 72, 72 / 24);
    expect(jaLine).toContain("569 フレーム");
    expect(jaLine).toContain("23.7");
    expect(jaLine).toContain("72 フレーム");
    expect(jaLine).toContain("3.0");
    expect(jaLine).toContain("末尾");
    expect(jaLine).toContain("素材");

    // The band-free form stays exactly what it always was, so a chain without an
    // end source reads unchanged.
    expect(en.chained.outputFramesLabel(497, 497 / 24)).toBe("Predicted output: ≈20.7s (497 frames)");
    expect(ja.chained.outputFramesLabel(497, 497 / 24)).toBe("予想出力: ≈20.7秒（497 フレーム）");
  });

  it("ja translates user-facing copy rather than reusing the English text verbatim", () => {
    // Spot-check a representative sample across namespaces.
    // The base-model dropdown's option LABELS are no longer in these
    // dictionaries at all — they are the server's `display_name`s (§3-97 P7),
    // so what is translatable here is the surrounding copy.
    expect(ja.toolVersion.ariaLabel).not.toBe(en.toolVersion.ariaLabel);
    expect(ja.toolVersion.notInstalled("LTX 2.5", "install-LTX25.bat")).not.toBe(
      en.toolVersion.notInstalled("LTX 2.5", "install-LTX25.bat"),
    );
    // …and the proper noun inside it survives translation untouched, which is
    // what the old `ltx23`/`ltx25` literals used to pin.
    expect(ja.toolVersion.optionNotInstalled("LTX 2.5")).toContain("LTX 2.5");
    expect(en.toolVersion.optionNotInstalled("LTX 2.5")).toContain("LTX 2.5");
    // 2026-08 タブ改称: modes ラベルは en/ja とも "Single"/"Chained"/"Inventory" の
    // 同一表記が仕様（オーナー決定）。翻訳差分アサーションの対象外。
    expect(ja.modes.single).toBe(en.modes.single);
    expect(ja.modes.chained).toBe(en.modes.chained);
    expect(ja.modes.inventory).toBe(en.modes.inventory);
    expect(ja.single.generateButton).not.toBe(en.single.generateButton);
    expect(ja.jobs.status.completed).not.toBe(en.jobs.status.completed);
    expect(ja.settings.title).not.toBe(en.settings.title);
    expect(ja.common.dismissNotification).not.toBe(en.common.dismissNotification);
  });

  it("DICTIONARIES exposes both languages, and DEFAULT_LANG is 'en'", () => {
    expect(DICTIONARIES.en).toBe(en);
    expect(DICTIONARIES.ja).toBe(ja);
    expect(DEFAULT_LANG).toBe("en");
  });
});
