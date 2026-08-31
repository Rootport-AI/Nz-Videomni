import { describe, expect, it } from "vitest";
import { resolveSpillFreeFrames } from "./spillUtils";

// A local fixed copy of config.yaml's `limits.spill_free_frames` (no example
// map lives in API_REFERENCE.md any more).
const SPILL_MAP = { "1280x768": 273, "1920x1088": 161, "2560x1472": 81 };

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
  "1280x768": 273,
  "1920x1088": 161,
  "2560x1472": 81,
};

describe("resolveSpillFreeFrames", () => {
  it("returns the exact value when the resolution is a key", () => {
    expect(resolveSpillFreeFrames(SPILL_MAP, 1280, 768)).toBe(273);
    expect(resolveSpillFreeFrames(SPILL_MAP, 1920, 1088)).toBe(161);
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
    expect(resolveSpillFreeFrames(SPILL_MAP, 512, 320)).toBe(273);
    // 1920x1080 (area 2,073,600) is closest to 1920x1088 (area 2,088,960).
    expect(resolveSpillFreeFrames(SPILL_MAP, 1920, 1080)).toBe(161);
    // 3840x2160 (area 8,294,400) is closest to 2560x1472 (area 3,768,320) of these three.
    expect(resolveSpillFreeFrames(SPILL_MAP, 3840, 2160)).toBe(81);
  });

  it("returns null for an empty map", () => {
    expect(resolveSpillFreeFrames({}, 512, 320)).toBeNull();
  });

  it("ignores malformed keys when computing the nearest area", () => {
    expect(resolveSpillFreeFrames({ ...SPILL_MAP, "not-a-resolution": 999 }, 512, 320)).toBe(273);
  });

  it("returns null when every key is malformed", () => {
    expect(resolveSpillFreeFrames({ foo: 1, bar: 2 }, 512, 320)).toBeNull();
  });
});
