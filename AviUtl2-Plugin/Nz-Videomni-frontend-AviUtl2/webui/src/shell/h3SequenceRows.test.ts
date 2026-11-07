import { describe, expect, it } from "vitest";
import type { H3SequenceInput, H3SequenceReference } from "./h3SequenceRows";
import { H3_NO_CONDITIONING, h3ComfortFrames, h3SequenceRows } from "./h3SequenceRows";

// Ledger §1-119: H3's DiT sequence length. The vectors below are the measured
// points of the recalibration (the backend's Docs/Outputs-archive/
// comfort-calib-h3-2026-11-04/RESULTS.md, Docs/COMFORT_LIMIT_TABLE.md §17 /
// §18). "+30" is the calibration prompt's 30 tokens, which this module does
// not count: adding them gives the measured S.

/** The served line (`single_rows_budget`): r1080_362's measured S. */
const BUDGET = 219516;
/** H3's served `num_frames` bounds. */
const MIN_FRAMES = 22;
const MAX_FRAMES = 481;

function input(
  width: number,
  height: number,
  keyframeCount = 0,
  references: readonly H3SequenceReference[] = [],
): H3SequenceInput {
  return { width, height, keyframeCount, references };
}

const video = (seconds: number | null): H3SequenceReference => ({ kind: "video", seconds });
const audio = (seconds: number | null): H3SequenceReference => ({ kind: "audio", seconds });
const image: H3SequenceReference = { kind: "image", seconds: null };
const nineImages = Array.from({ length: 9 }, () => image);

describe("h3SequenceRows — the measured vectors", () => {
  it("s1080_345: 1920x1088, K=1, F=345 (+30 = 213,348)", () => {
    expect(h3SequenceRows(input(1920, 1088, 1), 345)).toEqual({
      video: 208080,
      cond: 2040,
      audio: 1150,
      ref: 0,
      text: 2048,
      total: 213318,
    });
  });

  it("s1440_192: 2560x1472, K=1, F=192 (+30 = 217,798)", () => {
    expect(h3SequenceRows(input(2560, 1472, 1), 192)).toEqual({
      video: 209760,
      cond: 3680,
      audio: 640,
      ref: 0,
      text: 3688,
      total: 217768,
    });
  });

  it("s720_481: 1280x768, K=1, F=481 (+30 = 139,882)", () => {
    expect(h3SequenceRows(input(1280, 768, 1), 481)).toEqual({
      video: 136320,
      cond: 960,
      audio: 1604,
      ref: 0,
      text: 968,
      total: 139852,
    });
  });

  it("r1080_362: 1920x1088, K=0, F=362 (+30 = 219,516, the line)", () => {
    const rows = h3SequenceRows(input(1920, 1088), 362);
    expect(rows).toEqual({ video: 218280, cond: 0, audio: 1206, ref: 0, text: 0, total: 219486 });
    expect(rows!.total + 30).toBe(BUDGET);
  });

  it("v720_345_15s: 1280x768, K=0, a 14.375 s video (Fr = 345), F=345", () => {
    const rows = h3SequenceRows(input(1280, 768, 0, [video(14.375)]), 345);
    expect(rows).toEqual({ video: 97920, cond: 0, audio: 1150, ref: 102816, text: 15261, total: 217147 });
    // Measured S 217,167 with text 15,281: everything but text agrees exactly,
    // and text is not below the measured 15,281 - 30.
    expect(rows!.total - rows!.text).toBe(217167 - 15281);
    expect(rows!.text).toBeGreaterThanOrEqual(15251);
  });

  it("v720_379_15s: the same shape at F=379", () => {
    const rows = h3SequenceRows(input(1280, 768, 0, [video(14.375)]), 379);
    expect(rows).toEqual({ video: 107520, cond: 0, audio: 1264, ref: 102816, text: 15261, total: 226861 });
    expect(rows!.total - rows!.text).toBe(226881 - 15281);
  });

  it("i1080_243_p9: 1920x1088, K=0, nine pictures, F=243 (+30 = 184,512)", () => {
    expect(h3SequenceRows(input(1920, 1088, 0, nineImages), 243)).toEqual({
      video: 146880,
      cond: 0,
      audio: 810,
      ref: 18360,
      text: 18432,
      total: 184482,
    });
  });
});

describe("h3SequenceRows — the counting rules", () => {
  it("counts a 15.0 s video at F=379 as 362 frames (an upper bound)", () => {
    const rows = h3SequenceRows(input(1280, 768, 0, [video(15)]), 379);
    expect(rows?.ref).toBe(107856);
    expect(rows?.text).toBe(16278);
  });

  it("drops the keyframes' text rows once a Reference card rides", () => {
    expect(h3SequenceRows(input(1920, 1088, 1, [image]), 345)?.text).toBe(2048);
    expect(h3SequenceRows(input(1920, 1088, 1, [audio(5)]), 345)?.text).toBe(6);
  });

  it("counts at most two keyframes in the text rows while no Reference card rides", () => {
    const rows = h3SequenceRows(input(1920, 1088, 3), 345);
    expect(rows?.cond).toBe(3 * 2040);
    expect(rows?.text).toBe(2 * (2040 + 8));
  });

  it("counts an audio card at min(seconds, 15), 15 when unknown", () => {
    for (const seconds of [15, null, 20]) {
      const rows = h3SequenceRows(input(1280, 768, 0, [audio(seconds)]), 345);
      expect(rows?.ref).toBe(1200);
      expect(rows?.text).toBe(6);
    }
    expect(h3SequenceRows(input(1280, 768, 0, [audio(2.5)]), 345)?.ref).toBe(200);
  });

  it("counts a video of unknown length as 15 s", () => {
    expect(h3SequenceRows(input(1280, 768, 0, [video(null)]), 379)).toEqual(
      h3SequenceRows(input(1280, 768, 0, [video(15)]), 379),
    );
  });

  it("returns null off the 17n+5 grid and below one cell", () => {
    expect(h3SequenceRows(input(1920, 1088), 344)).toBeNull();
    expect(h3SequenceRows(input(1920, 1088), 361)).toBeNull();
    expect(h3SequenceRows(input(0, 1088), 345)).toBeNull();
    expect(h3SequenceRows(input(1920, 0), 345)).toBeNull();
  });
});

describe("h3ComfortFrames (budget 219,516, frames 22..481)", () => {
  it("1920x1088 K=1 -> 345 (362 would be 223,574)", () => {
    expect(h3ComfortFrames(input(1920, 1088, 1), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(345);
    expect(h3SequenceRows(input(1920, 1088, 1), 362)?.total).toBe(223574);
  });

  it("2560x1472 K=1 -> 192 (209 would total 236,224 = measured U 236,254 - 30)", () => {
    expect(h3ComfortFrames(input(2560, 1472, 1), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(192);
    expect(h3SequenceRows(input(2560, 1472, 1), 209)?.total).toBe(236224);
  });

  it("1280x768, K=1 and K=0 -> 481 (the ceiling)", () => {
    expect(h3ComfortFrames(input(1280, 768, 1), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(481);
    expect(h3ComfortFrames(input(1280, 768, 0), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(481);
  });

  it("1280x768 with a 15 s video -> 345 (362 would be 228,060)", () => {
    expect(h3ComfortFrames(input(1280, 768, 0, [video(15)]), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(345);
    expect(h3SequenceRows(input(1280, 768, 0, [video(15)]), 362)?.total).toBe(228060);
  });

  it("1920x1088 K=0 -> 362", () => {
    expect(h3ComfortFrames(input(1920, 1088), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(362);
    expect(h3ComfortFrames({ width: 1920, height: 1088, ...H3_NO_CONDITIONING }, BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(
      362,
    );
  });

  it("1920x1088 with nine pictures -> 294 (computed, not measured: the measurement stops at 243)", () => {
    expect(h3ComfortFrames(input(1920, 1088, 0, nineImages), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBe(294);
  });

  it("starts from maxFrames, falls to the floor when nothing fits, and returns null below one cell", () => {
    expect(h3ComfortFrames(input(1920, 1088, 0, nineImages), BUDGET, MIN_FRAMES, 294)).toBe(294);
    expect(h3ComfortFrames(input(1920, 1088), 1, MIN_FRAMES, MAX_FRAMES)).toBe(22);
    expect(h3ComfortFrames(input(0, 1088), BUDGET, MIN_FRAMES, MAX_FRAMES)).toBeNull();
  });

  // K=0: with a keyframe, the first Reference card also drops the keyframes'
  // text rows, so S is not monotone in the cards there.
  it("never rises when a Reference card is added to a keyframe-free composition (monotone)", () => {
    const sizes: Array<[number, number]> = [
      [1280, 768],
      [1920, 1088],
      [2560, 1472],
    ];
    const additions: H3SequenceReference[] = [image, video(5), video(15), audio(10)];
    for (const [w, h] of sizes) {
      const base = h3ComfortFrames(input(w, h), BUDGET, MIN_FRAMES, MAX_FRAMES)!;
      for (const added of additions) {
        const withRef = h3ComfortFrames(input(w, h, 0, [added]), BUDGET, MIN_FRAMES, MAX_FRAMES)!;
        expect(withRef).toBeLessThanOrEqual(base);
        const withTwo = h3ComfortFrames(input(w, h, 0, [added, added]), BUDGET, MIN_FRAMES, MAX_FRAMES)!;
        expect(withTwo).toBeLessThanOrEqual(withRef);
      }
    }
  });
});
