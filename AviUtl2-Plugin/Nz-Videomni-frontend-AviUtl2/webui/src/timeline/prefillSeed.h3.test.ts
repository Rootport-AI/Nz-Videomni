import { describe, expect, it } from "vitest";
import type { ResultOf } from "../bridge";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { ACCELERATION_DEFAULTS } from "../shell/accelerationSettings";
import { resolveGenerationConstraints } from "../shell/generationConstraints";
import { resolvePrefillSeed, seedKeyframeCount } from "./prefillSeed";

// MiniMax H3 第 2 弾 (2026-10-17): the right-click seeds that land in Create
// follow the loaded base model's rules — the 17n+5 grid, the 32 multiple, the
// fixed 24 fps. Ledger §1-119: H3's comfort profile is the sequence-rows line,
// so the ceiling is `h3ComfortFrames` at the seed's size with the intent's
// keyframe count and no Reference card — never the LTX lookup.

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
  chain: { overlap_frames: 22 },
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

describe("seedKeyframeCount", () => {
  it("counts one keyframe for the intents that open on an image (#4, #10, #6, #12), none otherwise", () => {
    expect(seedKeyframeCount("image-to-video")).toBe(1);
    expect(seedKeyframeCount("image-from-frame")).toBe(1);
    expect(seedKeyframeCount("image-to-clip-chain")).toBe(1);
    expect(seedKeyframeCount("current-frame-to-clip-chain")).toBe(1);
    expect(seedKeyframeCount("extend-video")).toBe(0);
    expect(seedKeyframeCount("audio-to-video")).toBe(0);
    expect(seedKeyframeCount("reference-video")).toBe(0);
  });
});

describe("resolvePrefillSeed — MiniMax H3 rules", () => {
  it("#4 image-to-video: size on 32, fps pinned at 24 (no snap toast), DURATION at the rows line", () => {
    const s = seed("image-to-video", selection({ mediaFps: 29.97 }), true);
    expect(s.derived.width).toBe(1312);
    expect(s.derived.height).toBe(768);
    expect(s.frameRate).toBe(24);
    expect(s.frameRateSnappedFrom).toBeUndefined();
    // 1312x768 with one keyframe fits the rows line up to the ceiling.
    expect(s.numFrames).toBe(481);
  });

  it("#4 image-to-video at 1920x1088 seeds 345, at 2560x1472 seeds 192 (one keyframe)", () => {
    expect(seed("image-to-video", selection({ mediaWidth: 1920, mediaHeight: 1088 }), true).numFrames).toBe(345);
    expect(seed("image-to-video", selection({ mediaWidth: 2560, mediaHeight: 1472 }), true).numFrames).toBe(192);
  });

  it("the same seed on LTX keeps its old answers (64 grid, material fps, a comfort ceiling)", () => {
    const s = seed("image-to-video", selection({ mediaFps: 29.97 }), false);
    expect(s.derived.width % 64).toBe(0);
    expect(s.frameRate).toBe(30);
    expect(s.numFrames).toBeDefined();
    expect(((s.numFrames ?? 0) - 1) % 8).toBe(0);
  });

  it("#7 audio-to-video: the material length floors onto 17n+5 below the rows line", () => {
    // The span (600 project frames at 30 fps = 20 s) at the fixed 24 fps is
    // 480 frames → floored onto 17n+5 = 464. The rows line at 1312x768 (no
    // keyframe) is the 481 ceiling, so it does not cap; an LTX-style comfort
    // cap at this size would sit far lower (the same span on LTX lands well
    // below it).
    const sel = selection({ effectName: "音声ファイル", frameEnd: 599 });
    const s = seed("audio-to-video", sel, true);
    expect(s.frameRate).toBe(24);
    expect(s.numFrames).toBe(464);
    expect(seed("audio-to-video", sel, false).numFrames ?? 0).toBeLessThan(464);
  });

});

// Ledger §1-276: a Chain-targeted seed on an engine that fixes the chain's
// overlap (H3's served `chain`) follows the served rules, and its DURATION is
// clip 0's sequence-rows marker — one keyframe for #6/#12 (the opening image),
// none for #1 (V2V).
describe("resolvePrefillSeed — MiniMax H3 Chain-targeted intents", () => {
  it("a Chain-targeted intent follows the served rules (32 multiple, fps pinned at 24)", () => {
    const s = seed("image-to-clip-chain", selection({ mediaFps: 29.97 }), true);
    expect(s.derived.width).toBe(1312);
    expect(s.derived.height).toBe(768);
    expect(s.frameRate).toBe(24);
    expect(s.frameRateSnappedFrom).toBeUndefined();
    // 1312x768 fits the rows line up to the ceiling.
    expect(s.numFrames).toBe(481);
  });

  it("#6 and #12 at 1920x1088 seed 345 (one keyframe), #1 seeds 362 (none)", () => {
    const big = selection({ mediaWidth: 1920, mediaHeight: 1088 });
    expect(seed("image-to-clip-chain", big, true).numFrames).toBe(345);
    expect(seed("current-frame-to-clip-chain", big, true).numFrames).toBe(345);
    expect(seed("extend-video", big, true).numFrames).toBe(362);
  });

  it("1312x768 seeds 481 for #1 and #12 too", () => {
    expect(seed("extend-video", selection(), true).numFrames).toBe(481);
    expect(seed("current-frame-to-clip-chain", selection(), true).numFrames).toBe(481);
  });

  it("rules that do not fix the overlap keep a Chain seed on LTX's grid", () => {
    const s = resolvePrefillSeed({
      intent: "image-to-clip-chain",
      selection: selection(),
      config: FALLBACK_APP_CONFIG,
      sizePolicy: "material",
      fpsPolicy: "material",
      engineFamily: "ltx",
      acceleration: ACCELERATION_DEFAULTS,
      sageAvailable: null,
      weightClass: "",
      constraints: { ...H3, chain: null },
    });
    expect(s.derived.width % 64).toBe(0);
    expect(s.frameRate).toBe(30);
    expect(((s.numFrames ?? 0) - 1) % 8).toBe(0);
  });

  it("the same Chain seeds on LTX keep their old answers (64 grid, material fps, the legacy table)", () => {
    const big = selection({ mediaWidth: 1920, mediaHeight: 1088 });
    for (const intent of ["image-to-clip-chain", "current-frame-to-clip-chain", "extend-video"]) {
      const withEngine = seed(intent, big, false);
      const bare = resolvePrefillSeed({
        intent,
        selection: big,
        config: FALLBACK_APP_CONFIG,
        sizePolicy: "material",
        fpsPolicy: "material",
      });
      expect(withEngine).toEqual(bare);
      expect(withEngine.derived.width % 64).toBe(0);
      expect(((withEngine.numFrames ?? 0) - 1) % 8).toBe(0);
    }
  });
});
