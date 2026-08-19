import { describe, expect, it } from "vitest";
import type { ResultOf } from "../bridge";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import type { PrefillResolutionPolicy } from "../shell/PrefillPolicyContext";
import {
  DURATION_POLICY_BY_INTENT,
  materialDurationForIntent,
  selectedRangeFramesForIntent,
  resolvePrefillSeed,
  spanDurationSec,
} from "./prefillSeed";

type Selection = ResultOf<"timeline.getSelection">;

// A single-object selection with the fields resolvePrefillSeed reads
// (media resolution/duration + frame span on the item; rate/scale at the top).
function makeSelection(
  item: Partial<Selection["selected"][number]> = {},
  top: Partial<Pick<Selection, "rate" | "scale">> = {},
): Selection {
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
        mediaWidth: 1280,
        mediaHeight: 768,
        mediaDurationSec: 0,
        ...item,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate: 24,
    scale: 1,
    sampleRate: 44100,
    ...top,
  };
}

// A layer selection (no selected object) — for the cursor/no-material intents.
function emptySelection(top: Partial<Pick<Selection, "rate" | "scale">> = {}): Selection {
  return { ...makeSelection(), selected: [], ...top };
}

const cfg = FALLBACK_APP_CONFIG;

// W1: the two axes are independent. Both default to "material" here so the
// existing width/height/fps/DURATION cases keep their material semantics; the
// crossing cases below pass explicit per-axis policies.
function seed(
  intent: string,
  selection: Selection,
  sizePolicy: PrefillResolutionPolicy = "material",
  fpsPolicy: PrefillResolutionPolicy = "material",
) {
  return resolvePrefillSeed({ intent, selection, config: cfg, sizePolicy, fpsPolicy });
}

describe("DURATION_POLICY_BY_INTENT", () => {
  it("maps every right-click generation intent to its DURATION policy (12 intents)", () => {
    expect(DURATION_POLICY_BY_INTENT).toEqual({
      "extend-video": "comfortCeiling",
      "reference-video": "materialClampedToCeiling",
      // 台帳§1-15 W4: a Chain intent, so comfortCeiling — the reference
      // conditions the WHOLE chain by frame number, so its own length says
      // nothing about how long clip 0 should be (unlike its #2 sibling above,
      // whose single generation IS the reference's length).
      "reference-video-chain": "comfortCeiling",
      "video-audio-to-video": "materialClampedToCeiling",
      "image-to-video": "comfortCeiling",
      "image-to-clip-chain": "comfortCeiling",
      "audio-to-video": "materialClampedToCeiling",
      "current-frame-to-clip-chain": "comfortCeiling",
      // 台帳§1-16 長尺A2V: Chain intents, so comfortCeiling — the extracted
      // track's own length re-lays the WHOLE clip list via the audio auto-fit,
      // it does not clamp clip 0 (unlike their single-shot #3/#7 siblings).
      "video-audio-to-long-a2v": "comfortCeiling",
      "audio-to-long-a2v": "comfortCeiling",
      // 素材（末尾）(2026-08-15): a Chain intent, so comfortCeiling. Its material
      // is EMBEDDED in the generation's own tail (the frozen frames) rather than
      // appended to it, so the end material's duration says nothing about how
      // long the clip should be — the same reasoning as the two families above.
      "end-with-this": "comfortCeiling",
      // §1-17 Retake: the third family — neither the resolution nor the
      // material decides the length, the SELECTED RANGE does.
      retake: "selectedRangeLength",
    });
  });

  // 台帳§1-16: the long-a2v intents must NOT inherit #3/#7's material clamp —
  // asserted directly, since the two families are one table row apart.
  it("長尺A2Vの2意図は素材尺でクランプしない（comfortCeilingのまま）", () => {
    for (const intent of ["video-audio-to-long-a2v", "audio-to-long-a2v"]) {
      expect(DURATION_POLICY_BY_INTENT[intent]).toBe("comfortCeiling");
      expect(
        materialDurationForIntent(intent, makeSelection({ frameStart: 0, frameEnd: 120 }, { rate: 24, scale: 1 })),
      ).toBeUndefined();
      // 512x320 の快適上限 481 がそのまま初期尺になる（素材尺 121f には縛られない）。
      expect(seed(intent, makeSelection({ mediaWidth: 512, mediaHeight: 320 })).numFrames).toBe(481);
    }
  });
});

describe("spanDurationSec", () => {
  it("is the inclusive (frameEnd - frameStart + 1) span in seconds", () => {
    // (120 - 10 + 1) frames / (24/1) = 111/24 s.
    const s = makeSelection({ frameStart: 10, frameEnd: 120 }, { rate: 24, scale: 1 });
    expect(spanDurationSec(s.selected[0], s)).toBeCloseTo(111 / 24, 6);
  });

  it("is undefined for a missing item, non-positive span, or unresolvable rate/scale", () => {
    const s = makeSelection({ frameStart: 0, frameEnd: 120 }, { rate: 24, scale: 1 });
    expect(spanDurationSec(undefined, s)).toBeUndefined();
    expect(spanDurationSec(s.selected[0], { rate: 0, scale: 1 })).toBeUndefined();
    expect(spanDurationSec(s.selected[0], { rate: 24, scale: 0 })).toBeUndefined();
    // frameEnd < frameStart -> non-positive span.
    const neg = makeSelection({ frameStart: 10, frameEnd: 5 });
    expect(spanDurationSec(neg.selected[0], neg)).toBeUndefined();
  });
});

describe("resolvePrefillSeed — width/height/fps", () => {
  it("seeds width/height from the material and fps from the selection rate/scale (material/material)", () => {
    const s = seed("image-to-video", makeSelection({ mediaWidth: 800, mediaHeight: 600 }, { rate: 30, scale: 1 }));
    // 800x600 on the general 64 grid -> 832x640; fps = 30/1.
    expect(s.derived.width).toBe(832);
    expect(s.derived.height).toBe(640);
    expect(s.frameRate).toBe(30);
    // `derived` is returned whole (notes etc. survive for the caller).
    expect(Array.isArray(s.derived.notes)).toBe(true);
  });

  it("rounds IC-LoRA (reference-video) width/height to the 128 grid", () => {
    const s = seed("reference-video", makeSelection({ mediaWidth: 1000, mediaHeight: 700 }));
    expect(s.derived.width).toBe(1024);
    expect(s.derived.height).toBe(768);
  });

  // 台帳§1-15 W4: the Chain-targeted IC-LoRA intent uses the SAME 128 grid. It
  // attaches a reference video, which flips `useChainForm`'s grid to 128 the
  // moment the upload turns ready — seeding on 64 would show one size at mount
  // and a re-snapped one a beat later.
  it("rounds IC-LoRA (reference-video-chain) width/height to the 128 grid too", () => {
    const s = seed("reference-video-chain", makeSelection({ mediaWidth: 1000, mediaHeight: 700 }));
    expect(s.derived.width).toBe(1024);
    expect(s.derived.height).toBe(768);
  });

  it("ignores the material size/rate when BOTH axes are `defaults` (config defaults, no fps seed)", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 30, scale: 1 }),
      "defaults",
      "defaults",
    );
    expect(s.derived.width).toBe(cfg.generation_defaults.width); // 1280
    expect(s.derived.height).toBe(cfg.generation_defaults.height); // 768
    expect(s.frameRate).toBeUndefined();
    // Comfort ceiling still resolves off the config-default resolution.
    expect(s.numFrames).toBe(257);
  });

  it("leaves fps unseeded when rate/scale are not resolvable", () => {
    expect(seed("image-to-video", makeSelection({}, { rate: 0, scale: 1 })).frameRate).toBeUndefined();
    expect(seed("image-to-video", makeSelection({}, { rate: 30, scale: 0 })).frameRate).toBeUndefined();
  });

  // --- W1 axis crossings ----------------------------------------------------

  it("size=defaults × fps=material: config-default size, but fps still from the selection", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 30, scale: 1 }),
      "defaults",
      "material",
    );
    expect(s.derived.width).toBe(cfg.generation_defaults.width);
    expect(s.frameRate).toBe(30);
  });

  it("size=material × fps=defaults: material size, but fps kept at the config default (undefined seed)", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 30, scale: 1 }),
      "material",
      "defaults",
    );
    expect(s.derived.width).toBe(512);
    expect(s.derived.height).toBe(320);
    expect(s.frameRate).toBeUndefined();
  });

  it("size=material × fps=project: both seed from the selection (project is overwritten later by the caller)", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 30, scale: 1 }),
      "material",
      "project",
    );
    expect(s.derived.width).toBe(512);
    expect(s.frameRate).toBe(30);
  });

  it("size=project × fps=material: both seed from the selection (project is overwritten later by the caller)", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 30, scale: 1 }),
      "project",
      "material",
    );
    expect(s.derived.width).toBe(512);
    expect(s.frameRate).toBe(30);
  });
});

describe("resolvePrefillSeed — DURATION (comfortCeiling intents)", () => {
  for (const intent of ["extend-video", "image-to-video", "image-to-clip-chain", "current-frame-to-clip-chain"]) {
    it(`${intent}: DURATION is the resolution's comfort ceiling (512x320 -> 481)`, () => {
      const s = seed(intent, makeSelection({ mediaWidth: 512, mediaHeight: 320 }));
      expect(s.derived.width).toBe(512);
      expect(s.numFrames).toBe(481);
    });
  }

  it("comfortCeiling lowers DURATION for a large resolution (1920x1088 -> 153)", () => {
    expect(seed("image-to-video", makeSelection({ mediaWidth: 1920, mediaHeight: 1088 })).numFrames).toBe(153);
  });

  it("current-frame-to-clip-chain has no material, so it uses the project-default resolution's ceiling", () => {
    // Empty selection -> deriveGenerationParams falls back to config defaults
    // (1280x768), whose comfort ceiling is 257.
    expect(seed("current-frame-to-clip-chain", emptySelection()).numFrames).toBe(257);
  });
});

describe("resolvePrefillSeed — DURATION (materialClampedToCeiling intents, W4 span-based)", () => {
  it("#2 reference-video: clamps to the SPAN length when under the ceiling (121-frame span -> 121)", () => {
    // 1280x768 (128-aligned) -> ceiling 257. Span (120-0+1)=121 frames @24fps
    // maps to 121 frames (on the 8n+1 grid), which is < 257 so the span wins.
    const s = seed(
      "reference-video",
      makeSelection({ mediaWidth: 1280, mediaHeight: 768, frameStart: 0, frameEnd: 120, mediaDurationSec: 999 }, { rate: 24, scale: 1 }),
    );
    expect(s.numFrames).toBe(121);
  });

  it("#2 reference-video: clamps to the comfort ceiling when the SPAN is longer (721-frame span -> 153)", () => {
    // 1920x1152 (128 grid) -> nearest-area ceiling 153; a 721-frame span floors
    // well past 153, so it clamps DOWN to 153.
    const s = seed(
      "reference-video",
      makeSelection({ mediaWidth: 1920, mediaHeight: 1152, frameStart: 0, frameEnd: 720, mediaDurationSec: 2 }, { rate: 24, scale: 1 }),
    );
    expect(s.numFrames).toBe(153);
  });

  it("#7 audio-to-video: seeds from the object's timeline SPAN, not the full-file duration", () => {
    // Audio has no media resolution (0) -> derived falls back to the project
    // default 1280x768 (ceiling 257). Span 121 frames @24fps -> 121 (< 257). The
    // full-file mediaDurationSec (999) is deliberately ignored.
    const s = seed(
      "audio-to-video",
      makeSelection({ mediaWidth: 0, mediaHeight: 0, frameStart: 0, frameEnd: 120, mediaDurationSec: 999 }, { rate: 24, scale: 1 }),
    );
    expect(s.derived.width).toBe(cfg.generation_defaults.width);
    expect(s.numFrames).toBe(121);
  });

  it("#3 video-audio-to-video: seeds from the object's timeline SPAN", () => {
    // span = (frameEnd - frameStart + 1) / (rate/scale) = 121/24 s; * genFps(24)
    // = 121 -> already on the 8n+1 grid. media 512x320 -> ceiling 481, no clamp.
    const s = seed(
      "video-audio-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320, frameStart: 0, frameEnd: 120, mediaDurationSec: 999 }, { rate: 24, scale: 1 }),
    );
    expect(s.numFrames).toBe(121);
  });

  it("falls back (undefined DURATION) when the span is unresolvable (rate/scale <= 0)", () => {
    // No rate -> the span can't be resolved: the policy resolves nothing, so
    // numFrames is undefined and the caller keeps the config default.
    expect(seed("reference-video", makeSelection({ frameStart: 0, frameEnd: 120 }, { rate: 0, scale: 1 })).numFrames).toBeUndefined();
    expect(seed("audio-to-video", makeSelection({ mediaWidth: 0, mediaHeight: 0 }, { rate: 24, scale: 0 })).numFrames).toBeUndefined();
    expect(
      seed("video-audio-to-video", makeSelection({ frameStart: 0, frameEnd: 120 }, { rate: 0, scale: 1 })).numFrames,
    ).toBeUndefined();
  });
});

describe("materialDurationForIntent (W4: span-based for #2/#7/#3)", () => {
  it("returns the inclusive timeline span (seconds) for #2/#7/#3", () => {
    // (120 - 10 + 1) frames / (24/1) = 111/24 s.
    for (const intent of ["reference-video", "audio-to-video", "video-audio-to-video"]) {
      expect(
        materialDurationForIntent(intent, makeSelection({ frameStart: 10, frameEnd: 120 }, { rate: 24, scale: 1 })),
      ).toBeCloseTo(111 / 24, 6);
    }
  });

  it("returns undefined for #2/#7/#3 when rate/scale or the span is invalid", () => {
    for (const intent of ["reference-video", "audio-to-video", "video-audio-to-video"]) {
      expect(
        materialDurationForIntent(intent, makeSelection({ frameStart: 0, frameEnd: 120 }, { rate: 0, scale: 1 })),
      ).toBeUndefined();
      expect(materialDurationForIntent(intent, emptySelection())).toBeUndefined();
    }
  });

  it("returns undefined for comfortCeiling intents (no material length needed)", () => {
    expect(materialDurationForIntent("image-to-video", makeSelection({ mediaDurationSec: 5 }))).toBeUndefined();
    expect(materialDurationForIntent("extend-video", makeSelection({ mediaDurationSec: 5 }))).toBeUndefined();
  });
});

describe("selectedRangeFramesForIntent / retake の尺シード（§1-17）", () => {
  it("retake だけが選択範囲の長さを返す（閉区間仮説どおり end - start + 1）", () => {
    const sel = makeSelection({}, { rate: 30, scale: 1 });
    const ranged: Selection = { ...sel, hasRange: true, rangeStart: 60, rangeEnd: 209 };
    expect(selectedRangeFramesForIntent("retake", ranged)).toBe(150);
    // 他の意図は範囲を見ない。
    expect(selectedRangeFramesForIntent("extend-video", ranged)).toBeUndefined();
    expect(selectedRangeFramesForIntent("reference-video", ranged)).toBeUndefined();
  });

  it("範囲が無い/空なら undefined", () => {
    const sel = makeSelection();
    expect(selectedRangeFramesForIntent("retake", { ...sel, hasRange: false })).toBeUndefined();
    expect(
      selectedRangeFramesForIntent("retake", { ...sel, hasRange: true, rangeStart: 20, rangeEnd: 19 }),
    ).toBeUndefined();
  });

  it("materialDurationForIntent は retake に反応しない（担当が違う）", () => {
    const sel = makeSelection({ mediaDurationSec: 30 }, { rate: 30, scale: 1 });
    expect(materialDurationForIntent("retake", sel)).toBeUndefined();
  });

  it("resolvePrefillSeed の retake 尺は、選択範囲を生成 fps へ換算した値", () => {
    // プロジェクト 30fps・150 フレーム = 5.0 秒。fps ポリシーが material なので
    // 生成 fps も 30 になり、150 フレームがそのまま尺になる。
    const sel = makeSelection({}, { rate: 30, scale: 1 });
    const ranged: Selection = { ...sel, hasRange: true, rangeStart: 60, rangeEnd: 209 };
    expect(seed("retake", ranged).numFrames).toBe(150);
    expect(seed("retake", ranged).frameRate).toBe(30);
  });

  it("fps ポリシーが defaults なら、設定の生成 fps へ換算される", () => {
    // プロジェクト 30fps・150 フレーム = 5.0 秒 -> 既定 24fps で 120 フレーム。
    const sel = makeSelection({}, { rate: 30, scale: 1 });
    const ranged: Selection = { ...sel, hasRange: true, rangeStart: 60, rangeEnd: 209 };
    const s = seed("retake", ranged, "material", "defaults");
    expect(s.frameRate).toBeUndefined(); // 設定の既定を使う
    expect(s.numFrames).toBe(120);
  });

  it("範囲が無い retake は尺を決めない（設定の既定へ落ちる）", () => {
    const sel = makeSelection({}, { rate: 30, scale: 1 });
    expect(seed("retake", { ...sel, hasRange: false }).numFrames).toBeUndefined();
  });
});
