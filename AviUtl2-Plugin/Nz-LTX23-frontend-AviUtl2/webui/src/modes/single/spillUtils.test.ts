import { describe, expect, it } from "vitest";
import { isValidNumFrames } from "./paramUtils";
import { resolveSingleComfortBudget, resolveSpillFreeFrames, singleComfortFrames, SINGLE_COMFORT_TOKEN_BUDGET } from "./spillUtils";

// Docs/API_REFERENCE.md §3.2's example map.
const SPILL_MAP = { "1280x768": 257, "1920x1088": 153, "2560x1472": 81 };

// The full map as it now ships in config.yaml / FALLBACK_APP_CONFIG, with the
// two low-res 20s-ceiling entries (minimal 512x320, small 960x576) added.
// 2026-07-20: small (960x576) briefly dropped to 457 after the G4 real-device
// gate observed a slight shared-memory uptick at 481f, then restored to 481
// once log review confirmed it was the harmless non-spill baseline (no
// slowdown; RESOLUTION_DURATION_CAPABILITY.md §8.2), same as minimal
// (512x320).
const SPILL_MAP_FULL = {
  "512x320": 481,
  "960x576": 481,
  "1280x768": 257,
  "1920x1088": 153,
  "2560x1472": 81,
};

describe("resolveSpillFreeFrames", () => {
  it("returns the exact value when the resolution is a key", () => {
    expect(resolveSpillFreeFrames(SPILL_MAP, 1280, 768)).toBe(257);
    expect(resolveSpillFreeFrames(SPILL_MAP, 1920, 1088)).toBe(153);
    expect(resolveSpillFreeFrames(SPILL_MAP, 2560, 1472)).toBe(81);
  });

  it("resolves the newly-added low-res 20s-ceiling entries (minimal/small)", () => {
    // 512x320 (minimal) and 960x576 (small) are both exact keys mapping to
    // the 481-frame (20s@24fps) ceiling.
    expect(resolveSpillFreeFrames(SPILL_MAP_FULL, 512, 320)).toBe(481);
    expect(resolveSpillFreeFrames(SPILL_MAP_FULL, 960, 576)).toBe(481);
  });

  it("falls back to the nearest key by pixel area when there is no exact match", () => {
    // 512x320 (area 163,840) is closest to 1280x768 (area 983,040) of the three keys.
    expect(resolveSpillFreeFrames(SPILL_MAP, 512, 320)).toBe(257);
    // 1920x1080 (area 2,073,600) is closest to 1920x1088 (area 2,088,960).
    expect(resolveSpillFreeFrames(SPILL_MAP, 1920, 1080)).toBe(153);
    // 3840x2160 (area 8,294,400) is closest to 2560x1472 (area 3,768,320) of these three.
    expect(resolveSpillFreeFrames(SPILL_MAP, 3840, 2160)).toBe(81);
  });

  it("returns null for an empty map", () => {
    expect(resolveSpillFreeFrames({}, 512, 320)).toBeNull();
  });

  it("ignores malformed keys when computing the nearest area", () => {
    expect(resolveSpillFreeFrames({ ...SPILL_MAP, "not-a-resolution": 999 }, 512, 320)).toBe(257);
  });

  it("returns null when every key is malformed", () => {
    expect(resolveSpillFreeFrames({ foo: 1, bar: 2 }, 512, 320)).toBeNull();
  });
});

// Smart comfort marker (2026-08-18). MIN_NUM_FRAMES=9 / max_num_frames=481,
// the same bounds `FALLBACK_APP_CONFIG.limits` publishes — see
// `defaultConfig.ts`.
const MIN_FRAMES = 9;
const MAX_FRAMES = 481;

describe("singleComfortFrames", () => {
  // Docs/COMFORT_LIMIT_TABLE.md's calibration table, reproduced independently
  // here — every value below is the closed-form inverse of the token formula
  // at SINGLE_COMFORT_TOKEN_BUDGET (44,880), not copied from the plan.
  it("matches the calibration table at every anchor resolution", () => {
    expect(singleComfortFrames(1920, 1088, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(169);
    expect(singleComfortFrames(1280, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(361);
    expect(singleComfortFrames(768, 1280, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(361);
    expect(singleComfortFrames(2560, 1472, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(89);
    expect(singleComfortFrames(1472, 2560, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(89);
    // 512x320: the raw inverse (2,233 frames) is far past the ceiling, so this
    // exercises the upper clamp, not the raw formula.
    expect(singleComfortFrames(512, 320, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(481);
    expect(singleComfortFrames(4096, 4096, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(9);
  });

  it("is symmetric under a width/height swap (the token count only cares about area)", () => {
    const pairs: Array<[number, number]> = [
      [1920, 1088],
      [1280, 768],
      [2560, 1472],
      [1344, 768],
    ];
    for (const [w, h] of pairs) {
      expect(singleComfortFrames(w, h, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(
        singleComfortFrames(h, w, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES),
      );
    }
  });

  it("moves smoothly across intermediate (non-anchor) resolutions, two 8-frame steps per 64px", () => {
    // A 64px width step is two 32px cells, so the marker moves TWO 8-frame
    // steps (16 frames), not one — the old 5-key lookup would instead hold
    // flat at 257 (the nearest-area key, `1280x768`) across all three.
    expect(singleComfortFrames(1280, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(361);
    expect(singleComfortFrames(1344, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(345);
    expect(singleComfortFrames(1408, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(329);
    expect(singleComfortFrames(1216, 704, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(417);
  });

  it("clamps to maxFrames when the raw inverse overshoots it", () => {
    expect(singleComfortFrames(320, 320, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(MAX_FRAMES);
  });

  it("clamps to minFrames when the resolution is too large for even one comfortable latent frame", () => {
    // cells = 256*256 = 65,536 > the budget, so the raw inverse's nLatentMax
    // is 0 and the formula alone would go negative — unreachable through any
    // resolution the server actually publishes limits for, but a hand-typed
    // hugely out-of-range value can still reach this function (setWidth/
    // setHeight's `snap: false` path passes manual entry through unrounded).
    expect(singleComfortFrames(8192, 8192, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(MIN_FRAMES);
  });

  it("returns null when width/height don't reach one full 32px cell (0-division guard)", () => {
    expect(singleComfortFrames(0, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBeNull();
    expect(singleComfortFrames(1280, 0, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBeNull();
    // Number("") === 0: a hand-typed blank width/height field reaches here as
    // a bare 0, exactly like the explicit 0 case above.
    expect(singleComfortFrames(Number(""), 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBeNull();
    expect(singleComfortFrames(-64, 768, SINGLE_COMFORT_TOKEN_BUDGET, MIN_FRAMES, MAX_FRAMES)).toBeNull();
  });

  it("always returns a value on the 8n+1 grid across a representative sweep", () => {
    const budgets = [SINGLE_COMFORT_TOKEN_BUDGET];
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
    for (const budget of budgets) {
      for (const [w, h] of sizes) {
        const result = singleComfortFrames(w, h, budget, MIN_FRAMES, MAX_FRAMES);
        expect(result).not.toBeNull();
        expect(isValidNumFrames(result!)).toBe(true);
      }
    }
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
