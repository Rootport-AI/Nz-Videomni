import { describe, expect, it } from "vitest";
import {
  PLAYBACK_SPEED_NEUTRAL,
  decideSourceTrim,
  formatTrimSeconds,
  playbackStartSec,
  trimQuery,
} from "./sourceTrim";
import type { SourceTrimDecision, TrimSelectionItem } from "./sourceTrim";

/**
 * §1-6 の判定コア。実機調査（2026-08-01）で確定した「再生位置＝素材時間軸の秒」を
 * 前提に、**実測したリボン状態R1〜R4をそのままケースとして固定**している。
 * とくにR4（開始をずらしてもリボン長が変わらない状態＝窓のほうが短い）は
 * `min(spanSec, end - start)` が無いと存在しない映像を切り出してしまうため、
 * 独立したケースで押さえてある。
 */

/** A selection stub — `decideSourceTrim` only ever reads `rate`/`scale`. */
const SEL_30FPS = { rate: 30, scale: 1 };
/** 24 fps: the second project rate the probe was run at, and the one where the
 * source's own frame quantization can make the reported window slightly LONGER
 * than the ribbon. */
const SEL_24FPS = { rate: 24, scale: 1 };
/** 4 fps: 1/fps = 0.25 exactly, so the one-frame boundary arithmetic below is
 * done entirely in exactly-representable binary fractions and cannot hinge on a
 * floating-point crumb. */
const SEL_4FPS = { rate: 4, scale: 1 };

function makeItem(overrides: Partial<TrimSelectionItem> = {}): TrimSelectionItem {
  return {
    layer: 1,
    frameStart: 0,
    frameEnd: 59, // 60 frames inclusive -> 2.0s @30fps
    effectName: "動画ファイル",
    filePath: "C:\\v\\a.mp4",
    objectName: "a",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 10,
    ...overrides,
  };
}

/** An item whose playback window is fully reported — the shape native ships
 * since contract v10. `playbackEndSec` defaults to the end of the file so a
 * caller that only cares about the start does not have to spell it out. */
function withPlayback(overrides: Partial<TrimSelectionItem> = {}): TrimSelectionItem {
  const base = makeItem(overrides);
  return {
    ...base,
    hasPlaybackRange: true,
    playbackStartSec: 0,
    playbackEndSec: base.mediaDurationSec,
    ...overrides,
  };
}

function expectSkip(decision: SourceTrimDecision, reason: string): void {
  expect(decision.trim).toBe(false);
  if (!decision.trim) expect(decision.reason).toBe(reason);
}

describe("decideSourceTrim — the conservative skips", () => {
  it("no item at all -> noItem", () => {
    expectSkip(decideSourceTrim(undefined, SEL_30FPS), "noItem");
  });

  it("mediaDurationSec === 0 (unprobed file) -> unknownMediaDuration", () => {
    expectSkip(decideSourceTrim(withPlayback({ mediaDurationSec: 0 }), SEL_30FPS), "unknownMediaDuration");
  });

  it("rate <= 0 -> unresolvableSpan", () => {
    expectSkip(decideSourceTrim(withPlayback(), { rate: 0, scale: 1 }), "unresolvableSpan");
  });

  it("frameEnd before frameStart (non-positive span) -> unresolvableSpan", () => {
    expectSkip(
      decideSourceTrim(withPlayback({ frameStart: 100, frameEnd: 50 }), SEL_30FPS),
      "unresolvableSpan",
    );
  });

  it("the ribbon exactly matches the file -> spanCoversWholeMedia (no pointless re-encode)", () => {
    // 300 frames @30fps = 10.0s = the file's own duration.
    const item = withPlayback({ frameStart: 0, frameEnd: 299, mediaDurationSec: 10 });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "spanCoversWholeMedia");
  });

  it("the ribbon is LONGER than the file (stretched past the end) -> spanCoversWholeMedia", () => {
    const item = withPlayback({ frameStart: 0, frameEnd: 599, mediaDurationSec: 10 }); // 20s ribbon
    expectSkip(decideSourceTrim(item, SEL_30FPS), "spanCoversWholeMedia");
  });
});

describe("decideSourceTrim — the one-frame rounding boundary (condition 3)", () => {
  // 8 frames @4fps = 2.0s span; one frame = 0.25s. The rule is
  // `spanSec < mediaDurationSec - 1/fps`, biased so anything within one frame of
  // the full duration counts as "the whole file".
  const span2s = { frameStart: 0, frameEnd: 7 };

  it("a difference of EXACTLY one frame stays on the no-trim side", () => {
    const item = withPlayback({ ...span2s, mediaDurationSec: 2.25 }); // 2.0 < 2.25-0.25 == 2.0 -> false
    expectSkip(decideSourceTrim(item, SEL_4FPS), "spanCoversWholeMedia");
  });

  it("a difference of one and a half frames DOES trim", () => {
    const item = withPlayback({ ...span2s, mediaDurationSec: 2.375 }); // 2.0 < 2.375-0.25 == 2.125 -> true
    const decision = decideSourceTrim(item, SEL_4FPS);
    expect(decision.trim).toBe(true);
    if (decision.trim) expect(decision.durationSec).toBeCloseTo(2.0, 9);
  });
});

describe("decideSourceTrim — the material gates", () => {
  it("a reported non-neutral playback speed -> nonNeutralSpeed", () => {
    // 実測: 再生速度の生値は "200.00" で、nativeが100で割って 2.0 で届く。
    const item = withPlayback({ playbackSpeed: 2.0 });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "nonNeutralSpeed");
  });

  it("the neutral speed does NOT skip", () => {
    const item = withPlayback({ playbackSpeed: PLAYBACK_SPEED_NEUTRAL });
    expect(decideSourceTrim(item, SEL_30FPS).trim).toBe(true);
  });

  it("a rounding crumb around 1.0 is still neutral (native divides '100.00' by 100)", () => {
    const item = withPlayback({ playbackSpeed: 100.0 / 100.0 + 1e-12 });
    expect(decideSourceTrim(item, SEL_30FPS).trim).toBe(true);
  });

  it("ループ再生 on -> loopEnabled", () => {
    expectSkip(decideSourceTrim(withPlayback({ loopPlay: true }), SEL_30FPS), "loopEnabled");
  });

  it("ループ再生 off does NOT skip", () => {
    expect(decideSourceTrim(withPlayback({ loopPlay: false }), SEL_30FPS).trim).toBe(true);
  });

  it("2+ sections (中間点) -> multipleSections", () => {
    expectSkip(decideSourceTrim(withPlayback({ sectionCount: 2 }), SEL_30FPS), "multipleSections");
  });

  it("a single reported section does NOT skip", () => {
    expect(decideSourceTrim(withPlayback({ sectionCount: 1 }), SEL_30FPS).trim).toBe(true);
  });

  it("no playback fields at all (an older plugin build) -> unknownPlaybackPosition", () => {
    // A plain pre-v10 SelectionItem: none of the six v10 fields present. The
    // WebUI must fall straight back to the whole-file upload rather than trim
    // from a position it never learned.
    const plain = makeItem({ frameStart: 0, frameEnd: 59, mediaDurationSec: 10 });
    expectSkip(decideSourceTrim(plain, SEL_30FPS), "unknownPlaybackPosition");
  });

  it("hasPlaybackRange:false is also unknownPlaybackPosition (values alone are not enough)", () => {
    const item = makeItem({ hasPlaybackRange: false, playbackStartSec: 3, playbackEndSec: 5 });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "unknownPlaybackPosition");
  });
});

/**
 * 実機で採取したリボン状態をそのまま固定するケース群（正本 §4）。素材は10.700秒。
 */
describe("decideSourceTrim — the real-hardware ribbon states (2026-08-01)", () => {
  const MEDIA = 10.7;

  it("R1 無加工: 再生位置は 0.000,10.700 で span も全体 -> 無トリム（バイト等価の維持）", () => {
    // An untouched ribbon DOES report a range (`0.000,<full>` is written out
    // explicitly, not omitted), so the guarantee rests on condition 3, not on a
    // missing field: the span covers the whole file, so no query is produced and
    // the upload is byte-identical to before §1-6.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 320, // 321 frames @30fps = 10.7s
      mediaDurationSec: MEDIA,
      playbackStartSec: 0,
      playbackEndSec: MEDIA,
    });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "spanCoversWholeMedia");
    expect(trimQuery(decideSourceTrim(item, SEL_30FPS))).toBeUndefined();
  });

  it("R2 頭を2秒削る: window[2.000,10.700] / span 8.7 -> trim{2.0, 8.7}", () => {
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 260, // 261 frames @30fps = 8.7s
      mediaDurationSec: MEDIA,
      playbackStartSec: 2.0,
      playbackEndSec: MEDIA,
    });
    const d = decideSourceTrim(item, SEL_30FPS);
    expect(d.trim).toBe(true);
    if (d.trim) {
      expect(d.startSec).toBeCloseTo(2.0, 6);
      expect(d.durationSec).toBeCloseTo(8.7, 6);
    }
  });

  it("R3 末尾を削る: window[0.000,7.042] / span 7.042 -> trim{0, 7.042}", () => {
    // Trimming the tail moves the SECOND field; the start stays 0.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 210, // 211 frames @30fps = 7.0333…s
      mediaDurationSec: MEDIA,
      playbackStartSec: 0,
      playbackEndSec: 7.042,
    });
    const d = decideSourceTrim(item, SEL_30FPS);
    expect(d.trim).toBe(true);
    if (d.trim) {
      expect(d.startSec).toBe(0);
      // min(span 7.0333…, window 7.042) -> the span wins here by a few ms.
      expect(d.durationSec).toBeCloseTo(211 / 30, 6);
      expect(d.durationSec).toBeLessThanOrEqual(7.042);
    }
  });

  it("R4 はみ出し: span 8.7 だが window[3.333,10.700] は 7.367 -> duration は 7.367 にクランプ", () => {
    // The R4 state: dragging the start rightwards leaves the ribbon's LENGTH
    // untouched, so the ribbon is longer than the window it actually plays (its
    // tail is a frozen last frame). Taking `spanSec` here would cut 8.7s of
    // source that the timeline never shows.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 260, // 261 frames @30fps = 8.7s ribbon
      mediaDurationSec: MEDIA,
      playbackStartSec: 3.333,
      playbackEndSec: MEDIA,
    });
    const d = decideSourceTrim(item, SEL_30FPS);
    expect(d.trim).toBe(true);
    if (d.trim) {
      expect(d.startSec).toBeCloseTo(3.333, 6);
      expect(d.durationSec).toBeCloseTo(10.7 - 3.333, 6); // 7.367
      expect(d.durationSec).toBeLessThan(8.7);
    }
  });

  it("24fps量子化: span 8.667 < window 8.7 -> duration は span 側（8.667）", () => {
    // The other direction of the same `min`: at 24fps the ribbon quantizes to
    // 208 frames = 8.6666…s while the reported window is a full 8.7s. Taking
    // `end - start` would overshoot the ribbon by most of a frame.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 207, // 208 frames @24fps = 8.6666…s
      mediaDurationSec: MEDIA,
      playbackStartSec: 2.0,
      playbackEndSec: MEDIA,
    });
    const d = decideSourceTrim(item, SEL_24FPS);
    expect(d.trim).toBe(true);
    if (d.trim) {
      expect(d.startSec).toBeCloseTo(2.0, 6);
      expect(d.durationSec).toBeCloseTo(208 / 24, 6); // 8.6666…
      expect(d.durationSec).toBeLessThan(8.7);
    }
  });
});

describe("decideSourceTrim — the trim itself", () => {
  it("a clearly-shorter ribbon trims to [playback start, +span)", () => {
    // 60 frames @30fps = 2.0s of a 10s file, starting 3s in.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 59,
      mediaDurationSec: 10,
      playbackStartSec: 3,
      playbackEndSec: 10,
    });
    const decision = decideSourceTrim(item, SEL_30FPS);
    expect(decision.trim).toBe(true);
    if (decision.trim) {
      expect(decision.startSec).toBeCloseTo(3, 9);
      expect(decision.durationSec).toBeCloseTo(2, 9);
    }
  });

  it("clamps the duration so the window never runs past the end of the source", () => {
    // Starts 9s into a 10s file with a 2s ribbon -> only 1s is actually there.
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 59,
      mediaDurationSec: 10,
      playbackStartSec: 9,
      playbackEndSec: 10,
    });
    const decision = decideSourceTrim(item, SEL_30FPS);
    expect(decision.trim).toBe(true);
    if (decision.trim) {
      expect(decision.startSec).toBeCloseTo(9, 9);
      expect(decision.durationSec).toBeCloseTo(1, 9);
      expect(decision.startSec + decision.durationSec).toBeLessThanOrEqual(10);
    }
  });

  it("a start at/after the end of the file falls back to no trim rather than an empty window", () => {
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 59,
      mediaDurationSec: 10,
      playbackStartSec: 10,
      playbackEndSec: 10,
    });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "unresolvableSpan");
  });

  it("a zero-length reported window falls back to no trim", () => {
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 59,
      mediaDurationSec: 10,
      playbackStartSec: 4,
      playbackEndSec: 4,
    });
    expectSkip(decideSourceTrim(item, SEL_30FPS), "unresolvableSpan");
  });

  it("a negative/garbage playback start is floored to 0", () => {
    const item = withPlayback({
      frameStart: 0,
      frameEnd: 59,
      mediaDurationSec: 10,
      playbackStartSec: -5,
      playbackEndSec: 10,
    });
    const decision = decideSourceTrim(item, SEL_30FPS);
    expect(decision.trim).toBe(true);
    if (decision.trim) expect(decision.startSec).toBe(0);
  });
});

describe("playbackStartSec (実機調査で秒確定・2026-08-01)", () => {
  it("reads the reported start straight through as seconds, and 0 when there is none", () => {
    expect(playbackStartSec(makeItem({ playbackStartSec: 2.5 }), SEL_30FPS)).toBe(2.5);
    expect(playbackStartSec(makeItem(), SEL_30FPS)).toBe(0);
  });

  it("does NOT depend on the project fps (the unit is source seconds)", () => {
    const item = makeItem({ playbackStartSec: 2.0 });
    expect(playbackStartSec(item, SEL_30FPS)).toBe(playbackStartSec(item, SEL_24FPS));
  });
});

describe("formatTrimSeconds", () => {
  it("never emits exponential notation for a tiny value", () => {
    expect(formatTrimSeconds(1e-7)).toBe("0.000");
    expect(formatTrimSeconds(0.0000001)).not.toContain("e");
  });

  it("renders millisecond precision", () => {
    expect(formatTrimSeconds(0)).toBe("0.000");
    expect(formatTrimSeconds(2)).toBe("2.000");
    expect(formatTrimSeconds(1.23456)).toBe("1.235");
  });
});

describe("trimQuery", () => {
  it("is undefined for EVERY no-trim decision (so the params key never appears)", () => {
    expect(trimQuery({ trim: false, reason: "unknownPlaybackPosition" })).toBeUndefined();
    expect(trimQuery({ trim: false, reason: "spanCoversWholeMedia" })).toBeUndefined();
    expect(trimQuery({ trim: false, reason: "loopEnabled" })).toBeUndefined();
    expect(trimQuery({ trim: false, reason: "noItem" })).toBeUndefined();
  });

  it("emits both keys as toFixed(3) strings for a trim", () => {
    expect(trimQuery({ trim: true, startSec: 3, durationSec: 2 })).toEqual({
      trim_start_sec: "3.000",
      trim_duration_sec: "2.000",
    });
  });
});
