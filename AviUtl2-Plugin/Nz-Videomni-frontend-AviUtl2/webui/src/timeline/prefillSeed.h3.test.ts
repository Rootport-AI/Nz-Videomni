import { describe, expect, it } from "vitest";
import type { ResultOf } from "../bridge";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { ACCELERATION_DEFAULTS } from "../shell/accelerationSettings";
import { resolveGenerationConstraints } from "../shell/generationConstraints";
import { resolvePrefillSeed } from "./prefillSeed";

// MiniMax H3 第 2 弾 (2026-10-17): the right-click seeds that land in Create
// follow the loaded base model's rules — the 17n+5 grid, the 32 multiple, the
// fixed 24 fps — and, with no comfort profile for the engine, only snap onto
// the grid (no comfort ceiling, no LTX lookup).

type Selection = ResultOf<"timeline.getSelection">;

function selection(item: Partial<Selection["selected"][number]> = {}, rate = 30): Selection {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 120,
    selected: [
      {
        layer: 1,
        frameStart: 0,
        frameEnd: 120,
        effectName: "動画ファイル",
        filePath: "C:\\v\\a.mp4",
        objectName: "a",
        textContent: null,
        mediaWidth: 1300,
        mediaHeight: 740,
        mediaDurationSec: 0,
        ...item,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate,
    scale: 1,
    sampleRate: 44100,
  };
}

const H3 = resolveGenerationConstraints({
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
});

function seed(intent: string, sel: Selection, withH3: boolean) {
  return resolvePrefillSeed({
    intent,
    selection: sel,
    config: FALLBACK_APP_CONFIG,
    sizePolicy: "material",
    fpsPolicy: "material",
    engineFamily: withH3 ? "h3" : "ltx",
    acceleration: ACCELERATION_DEFAULTS,
    sageAvailable: null,
    weightClass: "",
    ...(withH3 ? { constraints: H3 } : {}),
  });
}

describe("resolvePrefillSeed — MiniMax H3 rules", () => {
  it("#4 image-to-video: size on 32, fps pinned at 24 (no snap toast), DURATION left to the form", () => {
    const s = seed("image-to-video", selection({ mediaFps: 29.97 }), true);
    expect(s.derived.width).toBe(1312);
    expect(s.derived.height).toBe(768);
    expect(s.frameRate).toBe(24);
    expect(s.frameRateSnappedFrom).toBeUndefined();
    // No comfort line for H3 → the comfortCeiling policy resolves nothing and
    // the form keeps its own (grid-snapped) DURATION.
    expect(s.numFrames).toBeUndefined();
  });

  it("the same seed on LTX keeps its old answers (64 grid, material fps, a comfort ceiling)", () => {
    const s = seed("image-to-video", selection({ mediaFps: 29.97 }), false);
    expect(s.derived.width % 64).toBe(0);
    expect(s.frameRate).toBe(30);
    expect(s.numFrames).toBeDefined();
    expect(((s.numFrames ?? 0) - 1) % 8).toBe(0);
  });

  it("#7 audio-to-video: the material length floors onto 17n+5 with no comfort cap", () => {
    // The span (600 project frames at 30 fps = 20 s) at the fixed 24 fps is
    // 480 frames → floored onto 17n+5 = 464. An LTX-style comfort cap at this
    // size would sit far lower (the same span on LTX lands well below it).
    const sel = selection({ effectName: "音声ファイル", frameEnd: 599 });
    const s = seed("audio-to-video", sel, true);
    expect(s.frameRate).toBe(24);
    expect(s.numFrames).toBe(464);
    expect(seed("audio-to-video", sel, false).numFrames ?? 0).toBeLessThan(464);
  });

  // The seed never reads an engine for a Chain intent; snapping it onto the
  // loaded rules is the form's job (useChainForm — see its H3 test).
  it("a Chain-targeted intent ignores the H3 rules", () => {
    const s = seed("image-to-clip-chain", selection(), true);
    expect(s.derived.width % 64).toBe(0);
    expect(s.frameRate).toBe(30);
  });
});
