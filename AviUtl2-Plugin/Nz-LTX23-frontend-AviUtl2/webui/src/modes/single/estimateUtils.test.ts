import { describe, expect, it } from "vitest";
import { estimateGenerationSeconds, formatEstimate } from "./estimateUtils";

describe("estimateGenerationSeconds", () => {
  it("matches its own calibration point (720p, 121 frames -> ~168s)", () => {
    expect(estimateGenerationSeconds(1280, 768, 121)).toBeCloseTo(168, 5);
  });

  it("is linear in num_frames", () => {
    const base = estimateGenerationSeconds(1280, 768, 49);
    const doubled = estimateGenerationSeconds(1280, 768, 98);
    expect(doubled).toBeCloseTo(base * 2, 5);
  });

  it("is linear in pixel area", () => {
    const base = estimateGenerationSeconds(512, 320, 49);
    const doubledArea = estimateGenerationSeconds(1024, 320, 49);
    expect(doubledArea).toBeCloseTo(base * 2, 5);
  });

  it("returns 0 for degenerate input", () => {
    expect(estimateGenerationSeconds(0, 320, 49)).toBe(0);
    expect(estimateGenerationSeconds(512, 320, 0)).toBe(0);
    expect(estimateGenerationSeconds(512, -10, 49)).toBe(0);
  });
});

describe("formatEstimate", () => {
  it("formats sub-minute durations in seconds", () => {
    expect(formatEstimate(1)).toBe("~1s");
    expect(formatEstimate(45)).toBe("~45s");
    expect(formatEstimate(0.2)).toBe("~1s"); // clamped up to at least 1s
  });

  it("formats minute-plus durations in minutes, with one decimal under 10 minutes", () => {
    expect(formatEstimate(168)).toBe("~2.8 min");
    expect(formatEstimate(60)).toBe("~1 min");
  });

  it("rounds to whole minutes at/above 10 minutes", () => {
    expect(formatEstimate(660)).toBe("~11 min");
  });
});
