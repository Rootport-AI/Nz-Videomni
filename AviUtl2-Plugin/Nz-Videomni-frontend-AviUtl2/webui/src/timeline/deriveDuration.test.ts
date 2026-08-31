import { describe, expect, it } from "vitest";
import { computeTargetNumFrames, floorToFrameGrid } from "./deriveDuration";

// A module-local fixture shaped like `limits.spill_free_frames`, intentionally
// independent of `defaultConfig.ts`'s actual delivered values (it is not meant
// to track them — only to exercise this module's own logic). 481/257/153 are
// all already valid 8n+1 values, as the real backend map's entries are.
const SPILL_MAP = { "512x320": 481, "1280x768": 257, "1920x1088": 153 };

const MIN = 9;
const MAX = 481;

describe("computeTargetNumFrames", () => {
  describe('policy: "untouched"', () => {
    it("always returns null, regardless of the other args", () => {
      expect(
        computeTargetNumFrames({
          policy: "untouched",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 10,
          genFps: 24,
        }),
      ).toBeNull();
      expect(
        computeTargetNumFrames({
          policy: "untouched",
          width: 1280,
          height: 768,
          spillFreeFrames: null,
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBeNull();
    });
  });

  describe('policy: "comfortCeiling"', () => {
    it("resolves the comfort ceiling for each resolution via the exact-key match", () => {
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 512,
          height: 320,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBe(481);
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBe(257);
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 1920,
          height: 1088,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBe(153);
    });

    it("ignores any notion of a 'current' DURATION — the same width/height always yields the same result whether that raises or lowers what was there before", () => {
      // The args shape has no "current numFrames" field at all, so calling
      // with a resolution whose ceiling (481) is far ABOVE a hypothetical
      // prior DURATION (e.g. 257, this same map's 1280x768 entry) still
      // returns 481 — nothing about the call site's history can leak in.
      const afterA1280x768 = computeTargetNumFrames({
        policy: "comfortCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP,
        minNumFrames: MIN,
        maxNumFrames: MAX,
      });
      expect(afterA1280x768).toBe(257);
      const then512x320 = computeTargetNumFrames({
        policy: "comfortCeiling",
        width: 512,
        height: 320,
        spillFreeFrames: SPILL_MAP,
        minNumFrames: MIN,
        maxNumFrames: MAX,
      });
      expect(then512x320).toBe(481);
      // And the reverse order (raise then lower) is just as independent.
      const backTo1280x768 = computeTargetNumFrames({
        policy: "comfortCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP,
        minNumFrames: MIN,
        maxNumFrames: MAX,
      });
      expect(backTo1280x768).toBe(257);
    });

    it("clamps the resolved ceiling into [minNumFrames, maxNumFrames]", () => {
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 512,
          height: 320,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: 257, // narrower than the resolved 481 ceiling
        }),
      ).toBe(257);
    });

    it("returns null when the comfort ceiling can't be resolved (empty map)", () => {
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 512,
          height: 320,
          spillFreeFrames: {},
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBeNull();
      expect(
        computeTargetNumFrames({
          policy: "comfortCeiling",
          width: 512,
          height: 320,
          spillFreeFrames: null,
          minNumFrames: MIN,
          maxNumFrames: MAX,
        }),
      ).toBeNull();
    });
  });

  describe('policy: "materialClampedToCeiling"', () => {
    it("floors materialDurationSec*genFps onto the 8n+1 grid and leaves it as-is when under the ceiling", () => {
      // 10s * 24fps = 240 -> floored to 233 (8n+1), well under the 257 ceiling.
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 10,
          genFps: 24,
        }),
      ).toBe(233);
    });

    it("clamps to the comfort ceiling when the floored material length exceeds it", () => {
      // 30s * 24fps = 720 -> floored to 713, but the 1280x768 ceiling is 257.
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 30,
          genFps: 24,
        }),
      ).toBe(257);
    });

    it("raises the result to minNumFrames when the material is too short, even past what the ceiling would otherwise allow", () => {
      // 0.2s * 24fps = 4.8 -> floor to 4 -> grid-floors to 9, which is below
      // this call's minNumFrames (25), so it's raised to 25 — above both the
      // material-derived 9 AND would-be ceiling comparisons.
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: 25,
          maxNumFrames: MAX,
          materialDurationSec: 0.2,
          genFps: 24,
        }),
      ).toBe(25);
    });

    it("computes off the material length alone when the comfort ceiling can't be resolved", () => {
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: {},
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 10,
          genFps: 24,
        }),
      ).toBe(233);
    });

    it("returns null when materialDurationSec or genFps is missing, or genFps isn't positive", () => {
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          genFps: 24,
        }),
      ).toBeNull();
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 10,
        }),
      ).toBeNull();
      expect(
        computeTargetNumFrames({
          policy: "materialClampedToCeiling",
          width: 1280,
          height: 768,
          spillFreeFrames: SPILL_MAP,
          minNumFrames: MIN,
          maxNumFrames: MAX,
          materialDurationSec: 10,
          genFps: 0,
        }),
      ).toBeNull();
    });
  });
});

describe("floorToFrameGrid", () => {
  it("returns frames unchanged when already on the 8n+1 grid", () => {
    expect(floorToFrameGrid(233)).toBe(233);
    expect(floorToFrameGrid(9)).toBe(9);
    expect(floorToFrameGrid(17)).toBe(17);
  });

  it("floors down to the nearest grid value below", () => {
    expect(floorToFrameGrid(234)).toBe(233);
    expect(floorToFrameGrid(240)).toBe(233);
    expect(floorToFrameGrid(24)).toBe(17);
    expect(floorToFrameGrid(16)).toBe(9);
  });

  it("floors to 9 (n=1) for any input below the first grid value", () => {
    expect(floorToFrameGrid(8)).toBe(9);
    expect(floorToFrameGrid(4)).toBe(9);
    expect(floorToFrameGrid(1)).toBe(9);
    expect(floorToFrameGrid(0)).toBe(9);
  });
});

describe("computeTargetNumFrames — selectedRangeLength (§1-17 Retake)", () => {
  const BASE = {
    policy: "selectedRangeLength" as const,
    width: 1280,
    height: 768,
    // 快適上限の表を渡しても無視されることを見るために、あえて埋めてある。
    spillFreeFrames: { "1280x768": 257 },
    minNumFrames: 9,
    maxNumFrames: 481,
  };

  it("プロジェクトフレームを生成フレームへ換算する（単位変換が本体）", () => {
    // 150 プロジェクトフレーム @30fps = 5.0 秒 -> 24fps で 120 フレーム。
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, projectFps: 30, genFps: 24 }),
    ).toBe(120);
  });

  it("同じ fps 同士なら値は変わらない", () => {
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, projectFps: 30, genFps: 30 }),
    ).toBe(150);
  });

  it("生成 fps のほうが高ければフレーム数は増える", () => {
    // 60 プロジェクトフレーム @24fps = 2.5 秒 -> 60fps で 150 フレーム。
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 60, projectFps: 24, genFps: 60 }),
    ).toBe(150);
  });

  it("8n+1 へは丸めない（撮り直しの窓は別の場所で決まる）", () => {
    // 100 プロジェクトフレーム @25fps = 4.0 秒 -> 25fps で 100。8n+1 なら 97。
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 100, projectFps: 25, genFps: 25 }),
    ).toBe(100);
  });

  it("快適上限では切り下げない（この方針だけは上限表を見ない）", () => {
    // 快適上限 257 を超える長さでも、そのまま返る（max_num_frames までは通す）。
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 300, projectFps: 30, genFps: 30 }),
    ).toBe(300);
  });

  it("[minNumFrames, maxNumFrames] へはクランプする", () => {
    expect(computeTargetNumFrames({ ...BASE, selectedRangeFrames: 1, projectFps: 30, genFps: 1 })).toBe(9);
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 6000, projectFps: 30, genFps: 30 }),
    ).toBe(481);
  });

  it("必要な入力が欠けていれば null（尺には触らない）", () => {
    expect(computeTargetNumFrames({ ...BASE, projectFps: 30, genFps: 24 })).toBeNull();
    expect(computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, genFps: 24 })).toBeNull();
    expect(computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, projectFps: 30 })).toBeNull();
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 0, projectFps: 30, genFps: 24 }),
    ).toBeNull();
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, projectFps: 0, genFps: 24 }),
    ).toBeNull();
    expect(
      computeTargetNumFrames({ ...BASE, selectedRangeFrames: 150, projectFps: 30, genFps: 0 }),
    ).toBeNull();
  });
});

// 2026-08-31: a caller that already knows the SMART (token-derived) comfort
// ceiling for the loaded engine hands it over directly, and this module uses it
// in place of the `spillFreeFrames` lookup. The module itself stays ignorant of
// engines and acceleration settings — it only decides how a ceiling becomes a
// frame count.
describe("comfortCeilingFrames (smart ceiling override)", () => {
  it("wins over the spill_free_frames lookup for the comfortCeiling policy", () => {
    expect(
      computeTargetNumFrames({
        policy: "comfortCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP, // would resolve 257
        minNumFrames: MIN,
        maxNumFrames: MAX,
        comfortCeilingFrames: 361,
      }),
    ).toBe(361);
  });

  it("keeps the legacy lookup when omitted (every pre-existing caller)", () => {
    expect(
      computeTargetNumFrames({
        policy: "comfortCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP,
        minNumFrames: MIN,
        maxNumFrames: MAX,
      }),
    ).toBe(257);
  });

  it("also caps the materialClampedToCeiling policy's material-derived length", () => {
    // 30s @ 24fps floors to 713 raw frames — far above either ceiling, so the
    // clamp is what decides the answer, and it must be the smart one.
    expect(
      computeTargetNumFrames({
        policy: "materialClampedToCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP, // would clamp to 257
        minNumFrames: MIN,
        maxNumFrames: MAX,
        materialDurationSec: 30,
        genFps: 24,
        comfortCeilingFrames: 361,
      }),
    ).toBe(361);
    // A SHORT material is still respected — the ceiling only ever clamps down.
    expect(
      computeTargetNumFrames({
        policy: "materialClampedToCeiling",
        width: 1280,
        height: 768,
        spillFreeFrames: SPILL_MAP,
        minNumFrames: MIN,
        maxNumFrames: MAX,
        materialDurationSec: 2,
        genFps: 24,
        comfortCeilingFrames: 361,
      }),
    ).toBe(41);
  });
});
