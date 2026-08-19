import { describe, expect, it } from "vitest";
import { deriveGenerationParams } from "./deriveGenerationParams";
import type { GenerationParamLimits } from "./deriveGenerationParams";

/** Mirrors the real server config (Docs/API_REFERENCE.md §3.2 /
 * defaultConfig.ts's FALLBACK_APP_CONFIG.limits): max_height=1088 is
 * deliberately *not* a multiple of 128, which is exactly the case the
 * IC-LoRA floor-clamp test below exercises. */
const LIMITS: GenerationParamLimits = {
  minWidth: 256,
  maxWidth: 1920,
  minHeight: 128,
  maxHeight: 1088,
};

describe("deriveGenerationParams", () => {
  it("prefers the selected media's resolution over the project's when both are known", () => {
    const result = deriveGenerationParams({
      mediaWidth: 800,
      mediaHeight: 450,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.source).toBe("media");
    // 800 -> ceil to 832; 450 -> ceil to 512 (both multiples of 64).
    expect(result.width).toBe(832);
    expect(result.height).toBe(512);
  });

  it("falls back to the project resolution when no media resolution is known", () => {
    const result = deriveGenerationParams({
      mediaWidth: null,
      mediaHeight: null,
      projectWidth: 1280,
      projectHeight: 720,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.source).toBe("project");
    expect(result.width).toBe(1280);
    expect(result.height).toBe(768); // 720 -> ceil to 768
  });

  it("falls back to the project resolution when only one media dimension is usable", () => {
    const result = deriveGenerationParams({
      mediaWidth: 0,
      mediaHeight: 450,
      projectWidth: 1280,
      projectHeight: 720,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.source).toBe("project");
  });

  it("rounds a non-multiple size up to the nearest 64 in general mode", () => {
    const result = deriveGenerationParams({
      mediaWidth: null,
      mediaHeight: null,
      projectWidth: 500,
      projectHeight: 500,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.multiple).toBe(64);
    expect(result.width).toBe(512);
    expect(result.height).toBe(512);
    expect(result.notes.some((n) => n.includes("64"))).toBe(true);
  });

  it("rounds to the nearest 128 for IC-LoRA and notes why", () => {
    const result = deriveGenerationParams({
      mediaWidth: 1000,
      mediaHeight: 500,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: true,
      limits: LIMITS,
    });
    expect(result.multiple).toBe(128);
    expect(result.width).toBe(1024);
    expect(result.height).toBe(512);
    expect(result.clamped).toBe(false);
    expect(result.notes.some((n) => n.includes("IC-LoRA") && n.includes("128"))).toBe(true);
  });

  it("floor-clamps the upper bound to a real multiple (max_height=1088 -> 1024 for IC-LoRA)", () => {
    const result = deriveGenerationParams({
      mediaWidth: 1024,
      mediaHeight: 1080,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: true,
      limits: LIMITS,
    });
    expect(result.height).toBe(1024);
    expect(result.clamped).toBe(true);
    expect(result.notes.some((n) => n.includes("上限") && n.includes("1024"))).toBe(true);
  });

  it("clamps up to the minimum when the rounded value falls below it", () => {
    const result = deriveGenerationParams({
      mediaWidth: 300,
      mediaHeight: 50,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.height).toBe(128);
    expect(result.clamped).toBe(true);
    expect(result.notes.some((n) => n.includes("下限") && n.includes("128"))).toBe(true);
  });

  it("leaves an already-valid multiple unchanged and reports no clamp", () => {
    const result = deriveGenerationParams({
      mediaWidth: 1280,
      mediaHeight: 768,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.width).toBe(1280);
    expect(result.height).toBe(768);
    expect(result.clamped).toBe(false);
  });

  it("always includes a non-empty reasoning note for the chosen source", () => {
    const result = deriveGenerationParams({
      mediaWidth: 800,
      mediaHeight: 450,
      projectWidth: 1920,
      projectHeight: 1080,
      isICLora: false,
      limits: LIMITS,
    });
    expect(result.notes.length).toBeGreaterThan(0);
    expect(result.notes.some((n) => n.includes("選択素材"))).toBe(true);
  });
});
