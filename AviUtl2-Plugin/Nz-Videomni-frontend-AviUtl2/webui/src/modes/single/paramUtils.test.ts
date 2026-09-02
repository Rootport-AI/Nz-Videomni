import { describe, expect, it } from "vitest";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import {
  ceilToMultiple,
  clamp,
  floorToMultiple,
  formatDurationHint,
  FRAME_RATE_FALLBACK,
  FRAME_RATE_MAX,
  FRAME_RATE_MIN,
  framesToSeconds,
  isDimensionOnGrid,
  isNumFramesOnGrid,
  isStepEvent,
  isValidDimension,
  isValidNumFrames,
  isValidPrompt,
  roundToMultiple,
  snapFrameRate,
  snapNumFrames,
} from "./paramUtils";

describe("clamp", () => {
  it("passes values already within range through unchanged", () => {
    expect(clamp(50, 0, 100)).toBe(50);
  });
  it("clamps below min", () => {
    expect(clamp(-5, 0, 100)).toBe(0);
  });
  it("clamps above max", () => {
    expect(clamp(500, 0, 100)).toBe(100);
  });
});

describe("roundToMultiple (width/height slider snapping, 64)", () => {
  it("rounds to the nearest multiple of 64", () => {
    expect(roundToMultiple(500, 64, 256, 1920)).toBe(512);
    expect(roundToMultiple(530, 64, 256, 1920)).toBe(512);
    expect(roundToMultiple(545, 64, 256, 1920)).toBe(576);
  });

  it("clamps to the minimum", () => {
    expect(roundToMultiple(10, 64, 256, 1920)).toBe(256);
  });

  it("clamps to the maximum (config max_width)", () => {
    expect(roundToMultiple(5000, 64, 256, 1920)).toBe(1920);
  });

  it("leaves an already-valid multiple of 64 unchanged", () => {
    expect(roundToMultiple(1280, 64, 256, 1920)).toBe(1280);
  });
});

describe("ceilToMultiple (Get size from AviUtl2)", () => {
  it("rounds up to the next multiple of 64 for a non-multiple size", () => {
    // AviUtl2 edit is 1920x1080 -> width already a multiple of 64, height
    // (1080) is not (1080/64 = 16.875) -> must round UP to 1088, never down.
    expect(ceilToMultiple(1920, 64, 256, 1920)).toBe(1920);
    expect(ceilToMultiple(1080, 64, 128, 1088)).toBe(1088);
  });

  it("never rounds down even when the nearest multiple is below the value", () => {
    expect(ceilToMultiple(1025, 64, 256, 4096)).toBe(1088);
  });

  it("leaves an already-valid multiple of 64 unchanged", () => {
    expect(ceilToMultiple(512, 64, 256, 1920)).toBe(512);
  });

  it("clamps the ceiling result to the configured maximum", () => {
    expect(ceilToMultiple(1930, 64, 256, 1920)).toBe(1920);
  });
});

describe("floorToMultiple (e.g. deriving a resolution seam's upper bound)", () => {
  it("rounds down to the nearest multiple of 64 for a non-multiple size", () => {
    expect(floorToMultiple(1080, 64, 128, 1088)).toBe(1024);
  });

  it("rounds down to the nearest multiple of 128 (IC-LoRA max_height=1088 case)", () => {
    expect(floorToMultiple(1088, 128, 128, 1088)).toBe(1024);
  });

  it("leaves an already-valid multiple unchanged", () => {
    expect(floorToMultiple(512, 64, 256, 1920)).toBe(512);
  });

  it("clamps to the minimum when the floored result falls below it", () => {
    expect(floorToMultiple(10, 64, 256, 1920)).toBe(256);
  });

  it("clamps to the maximum even though flooring alone never rounds up", () => {
    expect(floorToMultiple(5000, 64, 256, 1920)).toBe(1920);
  });
});

describe("snapNumFrames (8n+1, slider snaps in steps of 8)", () => {
  it("snaps an arbitrary value to the nearest 8n+1", () => {
    expect(snapNumFrames(50, 9, 481)).toBe(49);
    // 53 is exactly halfway between 49 and 57; Math.round breaks ties up.
    expect(snapNumFrames(53, 9, 481)).toBe(57);
    expect(snapNumFrames(55, 9, 481)).toBe(57);
  });

  it("clamps below the minimum (9)", () => {
    expect(snapNumFrames(1, 9, 481)).toBe(9);
  });

  it("clamps above the maximum (481, the 20s@24fps cap)", () => {
    expect(snapNumFrames(1000, 9, 481)).toBe(481);
  });

  it("leaves an already-valid 8n+1 value unchanged", () => {
    expect(snapNumFrames(257, 9, 481)).toBe(257);
  });
});

describe("isValidNumFrames / isValidDimension", () => {
  it("accepts valid 8n+1 values and rejects everything else", () => {
    expect(isValidNumFrames(9)).toBe(true);
    expect(isValidNumFrames(49)).toBe(true);
    expect(isValidNumFrames(481)).toBe(true);
    expect(isValidNumFrames(50)).toBe(false);
    expect(isValidNumFrames(0)).toBe(false);
    expect(isValidNumFrames(-7)).toBe(false);
  });

  it("accepts positive multiples of 64 and rejects everything else", () => {
    expect(isValidDimension(512)).toBe(true);
    expect(isValidDimension(1920)).toBe(true);
    expect(isValidDimension(500)).toBe(false);
    expect(isValidDimension(0)).toBe(false);
    expect(isValidDimension(-64)).toBe(false);
  });
});

describe("framesToSeconds / formatDurationHint", () => {
  it("computes seconds from frames and fps", () => {
    expect(framesToSeconds(49, 24)).toBeCloseTo(2.0417, 3);
    expect(framesToSeconds(257, 24)).toBeCloseTo(10.708, 2);
  });

  it("returns 0 for a non-positive frame rate instead of dividing by zero", () => {
    expect(framesToSeconds(49, 0)).toBe(0);
  });

  it("formats the duration hint string", () => {
    expect(formatDurationHint(49, 24)).toBe("49 frames ≈ 2.0s @ 24fps");
  });
});

describe("isValidPrompt", () => {
  it("rejects an empty prompt", () => {
    expect(isValidPrompt("")).toBe(false);
  });
  it("accepts a normal prompt", () => {
    expect(isValidPrompt("a cat riding a skateboard")).toBe(true);
  });
  it("rejects a prompt over 2000 characters", () => {
    expect(isValidPrompt("a".repeat(2001))).toBe(false);
  });
  it("accepts a prompt at exactly the 2000 character boundary", () => {
    expect(isValidPrompt("a".repeat(2000))).toBe(true);
  });
});

describe("isStepEvent (WebView2/Chromium number-input stepper vs. manual typing)", () => {
  it("treats manual text insertion as not a step event", () => {
    expect(isStepEvent({ inputType: "insertText" } as unknown as Event)).toBe(false);
  });
  it("treats manual backspace/delete as not a step event", () => {
    expect(isStepEvent({ inputType: "deleteContentBackward" } as unknown as Event)).toBe(false);
  });
  it("treats an empty-string inputType (stepper arrow/↑↓ key) as a step event", () => {
    expect(isStepEvent({ inputType: "" } as unknown as Event)).toBe(true);
  });
  it("treats a null inputType as a step event", () => {
    expect(isStepEvent({ inputType: null } as unknown as Event)).toBe(true);
  });
  it("treats a plain Event with no inputType (range slider) as a step event", () => {
    expect(isStepEvent({} as unknown as Event)).toBe(true);
  });
});

describe("isNumFramesOnGrid (send-time num_frames validation)", () => {
  it("accepts a valid 8n+1 value within range", () => {
    expect(isNumFramesOnGrid(49, 9, 481)).toBe(true);
    expect(isNumFramesOnGrid(9, 9, 481)).toBe(true);
    expect(isNumFramesOnGrid(481, 9, 481)).toBe(true);
  });

  it("rejects a value that isn't 8n+1", () => {
    expect(isNumFramesOnGrid(50, 9, 481)).toBe(false);
    expect(isNumFramesOnGrid(48, 9, 481)).toBe(false);
  });

  it("rejects a valid 8n+1 value outside [min, max]", () => {
    expect(isNumFramesOnGrid(1, 9, 481)).toBe(false); // below min, even though 8n+1
    expect(isNumFramesOnGrid(489, 9, 481)).toBe(false); // above max, even though 8n+1
  });

  it("rejects a non-integer value", () => {
    expect(isNumFramesOnGrid(49.5, 9, 481)).toBe(false);
  });

  it("respects caller-supplied min/max rather than a hardcoded [9,481] (e.g. Chain's per-clip bounds)", () => {
    expect(isNumFramesOnGrid(9, 9, 481)).toBe(true);
    expect(isNumFramesOnGrid(9, 17, 481)).toBe(false);
  });
});

describe("isDimensionOnGrid (send-time width/height validation)", () => {
  it("accepts a valid multiple of 64 within range", () => {
    expect(isDimensionOnGrid(512, 64, 256, 1920)).toBe(true);
  });
  it("rejects a non-multiple of 64", () => {
    expect(isDimensionOnGrid(500, 64, 256, 1920)).toBe(false);
  });
  it("rejects a multiple of 64 that falls below the minimum", () => {
    expect(isDimensionOnGrid(192, 64, 256, 1920)).toBe(false);
  });
  it("accepts a valid multiple of 128 within range (IC-LoRA height grid)", () => {
    expect(isDimensionOnGrid(1024, 128, 128, 1088)).toBe(true);
  });
  it("rejects a value that is a multiple of 64 but not of 128", () => {
    expect(isDimensionOnGrid(1000, 128, 128, 1088)).toBe(false);
  });
  it("accepts the exact minimum and maximum boundaries", () => {
    expect(isDimensionOnGrid(256, 64, 256, 1920)).toBe(true);
    expect(isDimensionOnGrid(1920, 64, 256, 1920)).toBe(true);
  });
  it("rejects a non-integer value", () => {
    expect(isDimensionOnGrid(512.5, 64, 256, 1920)).toBe(false);
  });
});

// 台帳§3-71/§3-72 (2026-09-02): whole frame rates only. This describe absorbed
// `prefillSeed.test.ts`'s old `snapMaterialFps` block when the two snaps were
// merged into one function — the material-fps cases below are that block,
// verbatim in substance, plus the cases the wider contract added (`null`, and
// the callers that lean on the [1,60] clamp).
describe("snapFrameRate (台帳§3-71/§3-72: generation fps is always a whole number)", () => {
  it("rounds the real-world non-integer rates to their intended integer", () => {
    expect(snapFrameRate(29.97)).toBe(30);
    expect(snapFrameRate(23.976)).toBe(24);
    expect(snapFrameRate(59.94)).toBe(60);
    // The exact rationals an NTSC project's rate/scale produces, not just the
    // rounded decimals a fixture would type.
    expect(snapFrameRate(30000 / 1001)).toBe(30);
    expect(snapFrameRate(24000 / 1001)).toBe(24);
    // An already-integer rate passes through untouched.
    expect(snapFrameRate(30)).toBe(30);
    expect(snapFrameRate(24)).toBe(24);
  });

  it("rounds half away from zero (Math.round), NOT to even", () => {
    // Pinned because the sibling implementations in `mcp_server/`/`gradio_ui/`
    // deliberately avoid Python's `round()` for this exact reason: 24.5 must be
    // 25 on every one of the three, not 24.
    expect(snapFrameRate(24.5)).toBe(25);
    expect(snapFrameRate(25.5)).toBe(26);
  });

  it("clamps to [FRAME_RATE_MIN, FRAME_RATE_MAX] — the range the server enforces", () => {
    // A 120/240 fps material would otherwise seed a value the backend rejects
    // with a 422 before the user has even seen the form.
    expect(snapFrameRate(120)).toBe(60);
    expect(snapFrameRate(240)).toBe(60);
    expect(snapFrameRate(61)).toBe(60);
    // Below the floor: 0.4 rounds to 0, which the clamp lifts to 1.
    expect(snapFrameRate(0.4)).toBe(1);
    expect(snapFrameRate(0.5)).toBe(1);
  });

  it("returns undefined for every 'no usable rate' shape (null / absent / 0 / negative / non-finite)", () => {
    // `undefined` = an older native build that doesn't emit `mediaFps`; `0` = the
    // mediaWidth/mediaHeight "0 means unknown" convention, which is where a
    // non-video object and an unsupported container (mkv/webm) both land — a
    // NORMAL outcome that simply falls through to the next fps tier. `null` is
    // the shape a bridge field takes when native reports "not applicable".
    expect(snapFrameRate(undefined)).toBeUndefined();
    expect(snapFrameRate(null)).toBeUndefined();
    expect(snapFrameRate(0)).toBeUndefined();
    expect(snapFrameRate(-30)).toBeUndefined();
    expect(snapFrameRate(Number.NaN)).toBeUndefined();
    expect(snapFrameRate(Number.POSITIVE_INFINITY)).toBeUndefined();
    expect(snapFrameRate(Number.NEGATIVE_INFINITY)).toBeUndefined();
  });
});

describe("frame-rate constants", () => {
  it("are the [1, 60] range every fps field and every snap shares", () => {
    expect(FRAME_RATE_MIN).toBe(1);
    expect(FRAME_RATE_MAX).toBe(60);
    // The clamp really is expressed in terms of them.
    expect(snapFrameRate(1000)).toBe(FRAME_RATE_MAX);
    expect(snapFrameRate(0.1)).toBe(FRAME_RATE_MIN);
  });

  it("FRAME_RATE_FALLBACK equals the config default fps (the cross-reference in its doc comment)", () => {
    // The two are duplicated on purpose (paramUtils stays dependency-free), so
    // this test is what stops them drifting apart: change `defaultConfig.ts`'s
    // `generation_defaults.frame_rate` without changing the constant and a form
    // whose seed failed would land on a different fps than one with no seed.
    expect(FRAME_RATE_FALLBACK).toBe(FALLBACK_APP_CONFIG.generation_defaults.frame_rate);
  });
});
