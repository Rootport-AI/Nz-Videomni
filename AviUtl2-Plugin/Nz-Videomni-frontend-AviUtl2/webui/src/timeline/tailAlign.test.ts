import { describe, expect, it } from "vitest";
import {
  END_SOURCE_CONTEXT_FRAMES,
  END_SOURCE_MIN_FRAMES,
  framesAtGenFps,
  tailAlignedStartFrame,
} from "./tailAlign";

/**
 * 素材（末尾）窓内モード (2026-08-17): the shared anchor/placement arithmetic.
 * Every rule here is consumed from two sides — the right-click reservation
 * (`shell/AppShell.tsx` Step 8) and the Generate-time re-place
 * (`modes/chained/ChainedScreen.tsx`) — so a divergence between the two would
 * show up as a provisional whose ribbon does not match the file that lands in it.
 */
describe("the two end-source constants", () => {
  it("pins the anchor at 8 and the material floor at 9 (anchor + VAE primer)", () => {
    // Both are wire-visible: the anchor is sent as `end_source.context_frames`
    // and the floor decides `endSourceTooShort`. A silent drift in either is a
    // change of what the backend is asked for, so they are pinned literally.
    expect(END_SOURCE_CONTEXT_FRAMES).toBe(8);
    expect(END_SOURCE_MIN_FRAMES).toBe(9);
    expect(END_SOURCE_MIN_FRAMES).toBe(END_SOURCE_CONTEXT_FRAMES + 1);
  });
});

describe("framesAtGenFps", () => {
  it("is floor(seconds × fps) − 1, the one-frame conservative bias", () => {
    expect(framesAtGenFps(5, 24)).toBe(119);
    expect(framesAtGenFps(1, 24)).toBe(23);
    // A hair under a whole frame still floors down.
    expect(framesAtGenFps(5 - 1e-9, 24)).toBe(118);
  });

  it("answers 0 for a degenerate duration or frame rate instead of a negative", () => {
    expect(framesAtGenFps(0, 24)).toBe(0);
    expect(framesAtGenFps(-3, 24)).toBe(0);
    expect(framesAtGenFps(5, 0)).toBe(0);
    expect(framesAtGenFps(Number.NaN, 24)).toBe(0);
  });
});

describe("tailAlignedStartFrame", () => {
  const PROJECT_30 = { projectRate: 30, projectScale: 1 };

  it("shifts the start back by the newly generated part, converted to project frames", () => {
    // 240 generation frames at 24fps = 10s = 300 project frames at 30fps.
    expect(
      tailAlignedStartFrame({ materialFrameStart: 500, leadPixelFrames: 240, genFps: 24, ...PROJECT_30 }),
    ).toBe(200);
  });

  it("is the identity conversion when the two frame rates match", () => {
    expect(
      tailAlignedStartFrame({
        materialFrameStart: 500,
        leadPixelFrames: 240,
        genFps: 24,
        projectRate: 24,
        projectScale: 1,
      }),
    ).toBe(260);
  });

  it("rounds HALF UP, mirroring native's ProjectFramesForPixels verbatim", () => {
    // 1 × 30 / 60 = 0.5 exactly -> 1 (not 0).
    expect(
      tailAlignedStartFrame({ materialFrameStart: 100, leadPixelFrames: 1, genFps: 60, ...PROJECT_30 }),
    ).toBe(99);
    // 3 × 30 / 60 = 1.5 exactly -> 2 (not 1, and not 2 by banker's rounding
    // accident — 2 IS even here, so the next case is the one that separates the
    // two rules).
    expect(
      tailAlignedStartFrame({ materialFrameStart: 100, leadPixelFrames: 3, genFps: 60, ...PROJECT_30 }),
    ).toBe(98);
    // 5 × 30 / 60 = 2.5 exactly -> 3 under half-UP, but 2 under Python's
    // half-to-EVEN. This is the case that proves which rule is implemented.
    expect(
      tailAlignedStartFrame({ materialFrameStart: 100, leadPixelFrames: 5, genFps: 60, ...PROJECT_30 }),
    ).toBe(97);
  });

  it("clamps to frame 0 for material near the head of the timeline", () => {
    expect(
      tailAlignedStartFrame({ materialFrameStart: 10, leadPixelFrames: 497, genFps: 24, ...PROJECT_30 }),
    ).toBe(0);
    expect(
      tailAlignedStartFrame({ materialFrameStart: 0, leadPixelFrames: 100, genFps: 24, ...PROJECT_30 }),
    ).toBe(0);
  });

  it("degrades to HEAD alignment when the frame rates are unresolvable", () => {
    const base = { materialFrameStart: 500, leadPixelFrames: 240 };
    // No project rate/scale (an unopened project, or a snapshot without them).
    expect(tailAlignedStartFrame({ ...base, genFps: 24, projectRate: 30, projectScale: 0 })).toBe(500);
    expect(tailAlignedStartFrame({ ...base, genFps: 24, projectRate: 0, projectScale: 1 })).toBe(500);
    // No generation fps.
    expect(tailAlignedStartFrame({ ...base, genFps: 0, ...PROJECT_30 })).toBe(500);
  });

  it("leaves the material's own start alone when there is nothing ahead of the frozen tail", () => {
    // A degenerate but reachable state mid-edit; head alignment is the honest
    // answer, not a shift by 0-ish garbage.
    expect(
      tailAlignedStartFrame({ materialFrameStart: 42, leadPixelFrames: 0, genFps: 24, ...PROJECT_30 }),
    ).toBe(42);
    expect(
      tailAlignedStartFrame({ materialFrameStart: 42, leadPixelFrames: -10, genFps: 24, ...PROJECT_30 }),
    ).toBe(42);
  });
});
