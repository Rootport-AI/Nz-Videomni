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
// §3-13: `item` is a Partial of the contract-v11 item, so a case that needs the
// material's own framerate passes `{ mediaFps: 29.97 }` here; leaving it out is
// the "native reported nothing" shape every pre-existing case relies on.
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

// §3-13 (contract v11): native reports the material's RAW framerate and this
// layer decides the integer the generation actually runs at.
// 台帳§3-71/§3-72 (2026-09-02): the snap itself moved to
// `modes/single/paramUtils.ts`'s `snapFrameRate` (one source of truth for every
// fps entry point), so the unit cases that used to live here — the
// `snapMaterialFps` describe — now live in `paramUtils.test.ts`. What stays here
// is what only this module can answer: which TIER a value comes from, and which
// of the two rates derived from `rate/scale` gets snapped.
describe("resolvePrefillSeed — fps snapping (台帳§3-71/§3-72)", () => {
  it("snaps the PROJECT tier too: an NTSC project (30000/1001) seeds 30 and reports the raw rate", () => {
    // This is the case the §3-71/§3-72 fix is really about: before it, only the
    // material tier was snapped, so an NTSC PROJECT's 29.97 went into the
    // request untouched.
    const s = seed("image-to-video", makeSelection({}, { rate: 30000, scale: 1001 }), "material", "project");
    expect(s.frameRate).toBe(30);
    expect(s.frameRateSnappedFrom).toBeCloseTo(29.97, 4);
  });

  it("reports NOTHING when the project is already on a whole frame rate (no false positives)", () => {
    const s = seed("image-to-video", makeSelection({}, { rate: 30, scale: 1 }), "material", "project");
    expect(s.frameRate).toBe(30);
    expect(s.frameRateSnappedFrom).toBeUndefined();
  });

  it("snaps the MATERIAL tier: a 23.976fps clip seeds 24 and reports 23.976", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaFps: 23.976 }, { rate: 30, scale: 1 }),
      "material",
      "material",
    );
    expect(s.frameRate).toBe(24);
    expect(s.frameRateSnappedFrom).toBe(23.976);
  });

  it("falls through to the project tier when the material's own fps is unusable, and snaps THAT", () => {
    // `mediaFps: 0` is native's "unknown" (a non-video object, or an
    // unsupported container) — tier 2 takes over, and it is snapped as well.
    const s = seed(
      "image-to-video",
      makeSelection({ mediaFps: 0 }, { rate: 24000, scale: 1001 }),
      "material",
      "material",
    );
    expect(s.frameRate).toBe(24);
    expect(s.frameRateSnappedFrom).toBeCloseTo(23.976, 4);
  });

  it("clamps a high-frame-rate material to 60 and reports the raw rate", () => {
    const s = seed("image-to-video", makeSelection({ mediaFps: 120 }), "material", "material");
    expect(s.frameRate).toBe(60);
    expect(s.frameRateSnappedFrom).toBe(120);
  });

  it("the `defaults` policy seeds no fps at all, and so reports no snap", () => {
    const s = seed("image-to-video", makeSelection({ mediaFps: 29.97 }, { rate: 30000, scale: 1001 }), "material", "defaults");
    expect(s.frameRate).toBeUndefined();
    expect(s.frameRateSnappedFrom).toBeUndefined();
  });

  it("an unresolvable rate/scale seeds no fps and reports no snap", () => {
    const s = seed("image-to-video", makeSelection({}, { rate: 0, scale: 0 }), "material", "project");
    expect(s.frameRate).toBeUndefined();
    expect(s.frameRateSnappedFrom).toBeUndefined();
  });

  // ⚠ REGRESSION GUARD (台帳§3-71/§3-72, plan §2 item 1): the `selectedRangeLength`
  // policy's `projectFps` must stay RAW. It is the PROJECT's timebase — the thing
  // that turns a selected frame RANGE into real seconds — not a rate anything
  // generates at, so snapping it would mis-measure the selection and Retake would
  // open on the wrong frames. `computeTargetNumFrames` computes
  // `round(selectedRangeFrames / projectFps * genFps)` (no 8n+1 snap on this
  // policy — see `deriveDuration.ts`), so a snapped `projectFps` would make the
  // whole expression collapse to `selectedRangeFrames` itself.
  it("Retake: a 120fps project converts the range at the RAW 120 while generating at the clamped 60", () => {
    // Chosen because the two answers differ by a FACTOR OF TWO rather than the
    // 0.1% an NTSC pair gives: `snapFrameRate(120)` clamps to 60, so a snapped
    // `projectFps` would read 300 project frames as 300 generation frames
    // instead of the correct 150.
    const selection = {
      ...makeSelection({ frameStart: 0, frameEnd: 299 }, { rate: 120, scale: 1 }),
      rangeStart: 0,
      rangeEnd: 299,
    };
    const s = seed("retake", selection, "material", "project");
    expect(s.frameRate).toBe(60);
    // 300 project frames / 120 = 2.5s; x 60fps = 150. (A snapped projectFps
    // would give 300 — twice the length the user selected.)
    expect(s.numFrames).toBe(150);
  });

  it("Retake: an NTSC project keeps the 0.1% the snap would have thrown away", () => {
    // The realistic shape of the same guard. 1000 project frames at 30000/1001
    // is 33.3667s, which is 1001 frames at the seeded 30fps — exactly one frame
    // more than the 1000 a snapped `projectFps` would produce. Needs a config
    // whose `max_num_frames` doesn't clamp the answer away, so this case builds
    // one instead of using `cfg`.
    const longCfg = { ...cfg, limits: { ...cfg.limits, max_num_frames: 4801 } };
    const selection = {
      ...makeSelection({ frameStart: 0, frameEnd: 999 }, { rate: 30000, scale: 1001 }),
      rangeStart: 0,
      rangeEnd: 999,
    };
    const s = resolvePrefillSeed({
      intent: "retake",
      selection,
      config: longCfg,
      sizePolicy: "material",
      fpsPolicy: "project",
    });
    expect(s.frameRate).toBe(30);
    expect(s.numFrames).toBe(1001);
  });
});

describe("resolvePrefillSeed — width/height/fps", () => {
  it("seeds width/height from the material and fps from the selection rate/scale (material/material)", () => {
    // §3-13: this fixture carries no `mediaFps`, so the fps axis's `material`
    // lands on its tier-2 fallback — the selection's own rate/scale. The tier-1
    // (material fps) cases live in their own describe below.
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
    expect(s.numFrames).toBe(273);
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

// §3-13 (contract v11): the fps axis's three tiers, each isolated by a fixture
// where the three candidate values are all different — the material's own fps
// (29.97 -> 30), the selection's project rate (25) and the config default (24).
describe("resolvePrefillSeed — fps の3段フォールバック（§3-13）", () => {
  it("1段目: fps=material は素材自身の fps を整数へスナップして採用する", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320, mediaFps: 29.97 }, { rate: 25, scale: 1 }),
      "material",
      "material",
    );
    // 25 (プロジェクト) でも 24 (設定既定) でもなく、素材の 29.97 -> 30。
    expect(s.frameRate).toBe(30);
  });

  it("2段目: 素材 fps が読めなければプロジェクトの rate/scale へ落ちる（mkv/webm・旧nativeビルド）", () => {
    // 0 は「不明」規約（media_width/height と同じ）。未指定も同じ扱い。
    expect(
      seed(
        "image-to-video",
        makeSelection({ mediaWidth: 512, mediaHeight: 320, mediaFps: 0 }, { rate: 25, scale: 1 }),
        "material",
        "material",
      ).frameRate,
    ).toBe(25);
    expect(
      seed(
        "image-to-video",
        makeSelection({ mediaWidth: 512, mediaHeight: 320 }, { rate: 25, scale: 1 }),
        "material",
        "material",
      ).frameRate,
    ).toBe(25);
  });

  it("3段目: 素材 fps も rate/scale も無ければ undefined（設定の既定 fps を使う）", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320, mediaFps: 0 }, { rate: 0, scale: 1 }),
      "material",
      "material",
    );
    expect(s.frameRate).toBeUndefined();
  });

  it("fps=project は素材 fps を見ない（軸ごとにポリシーが効く）", () => {
    // 同じ 29.97 の素材でも、project ならプロジェクトの 25 のまま。
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320, mediaFps: 29.97 }, { rate: 25, scale: 1 }),
      "material",
      "project",
    );
    expect(s.frameRate).toBe(25);
  });

  it("fps=defaults は素材 fps があっても undefined のまま", () => {
    const s = seed(
      "image-to-video",
      makeSelection({ mediaWidth: 512, mediaHeight: 320, mediaFps: 29.97 }, { rate: 25, scale: 1 }),
      "material",
      "defaults",
    );
    expect(s.frameRate).toBeUndefined();
  });

  it("素材 fps は尺→フレーム数の換算（genFps）にも波及する（意図どおり）", () => {
    // #2 reference-video: 素材尺は 121 プロジェクトフレーム / 24fps = 121/24 秒。
    // genFps 24 なら 121 フレーム、genFps 30 なら floor(151.25)=151 -> 8n+1 で 145。
    const item = { mediaWidth: 1280, mediaHeight: 768, frameStart: 0, frameEnd: 120 };
    expect(seed("reference-video", makeSelection(item, { rate: 24, scale: 1 })).numFrames).toBe(121);
    expect(
      seed("reference-video", makeSelection({ ...item, mediaFps: 29.97 }, { rate: 24, scale: 1 })).numFrames,
    ).toBe(145);
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

  it("comfortCeiling lowers DURATION for a large resolution (1920x1088 -> 161)", () => {
    expect(seed("image-to-video", makeSelection({ mediaWidth: 1920, mediaHeight: 1088 })).numFrames).toBe(161);
  });

  it("current-frame-to-clip-chain has no material, so it uses the project-default resolution's ceiling", () => {
    // Empty selection -> deriveGenerationParams falls back to config defaults
    // (1280x768), whose comfort ceiling is 273.
    expect(seed("current-frame-to-clip-chain", emptySelection()).numFrames).toBe(273);
  });
});

describe("resolvePrefillSeed — DURATION (materialClampedToCeiling intents, W4 span-based)", () => {
  it("#2 reference-video: clamps to the SPAN length when under the ceiling (121-frame span -> 121)", () => {
    // 1280x768 (128-aligned) -> ceiling 273. Span (120-0+1)=121 frames @24fps
    // maps to 121 frames (on the 8n+1 grid), which is < 273 so the span wins.
    const s = seed(
      "reference-video",
      makeSelection({ mediaWidth: 1280, mediaHeight: 768, frameStart: 0, frameEnd: 120, mediaDurationSec: 999 }, { rate: 24, scale: 1 }),
    );
    expect(s.numFrames).toBe(121);
  });

  it("#2 reference-video: clamps to the comfort ceiling when the SPAN is longer (721-frame span -> 161)", () => {
    // 1920x1152 (128 grid) -> nearest-area ceiling 161; a 721-frame span floors
    // well past 161, so it clamps DOWN to 161.
    const s = seed(
      "reference-video",
      makeSelection({ mediaWidth: 1920, mediaHeight: 1152, frameStart: 0, frameEnd: 720, mediaDurationSec: 2 }, { rate: 24, scale: 1 }),
    );
    expect(s.numFrames).toBe(161);
  });

  it("#7 audio-to-video: seeds from the object's timeline SPAN, not the full-file duration", () => {
    // Audio has no media resolution (0) -> derived falls back to the project
    // default 1280x768 (ceiling 273). Span 121 frames @24fps -> 121 (< 273). The
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

// 2026-08-31: when the caller threads the engine family + acceleration settings
// (AppShell and SingleScreen both do), a Single-系 prefill's DURATION seeds to
// the SAME smart ceiling Create's comfort marker will then display. Chain-系
// intents are a different quantity (a per-clip length inside a chain) and stay
// on the legacy table.
describe("resolvePrefillSeed — smart comfort ceiling (2026-08-31)", () => {
  /** All five toggles on: exactly what the served `ltx` row requires. */
  const FULL_ACCELERATION = {
    attentionBackend: "sage",
    blockSwapPrefetch: true,
    keepResident: true,
    fusedGgufDequantKernel: true,
    vaeMode: "prune_vaed",
  } as const;

  function smartSeed(intent: string, selection: Selection, engineFamily = "ltx") {
    return resolvePrefillSeed({
      intent,
      selection,
      config: cfg,
      sizePolicy: "material",
      fpsPolicy: "material",
      engineFamily,
      acceleration: FULL_ACCELERATION,
      sageAvailable: true,
    });
  }

  it("#4 image-to-video seeds the SMART 361 at 1280x768 with all five toggles on (ltx)", () => {
    // Same input as the legacy case above, which seeds 273 — the acceleration
    // trio is the only difference.
    const selection = makeSelection({ mediaWidth: 1280, mediaHeight: 768 });
    expect(seed("image-to-video", selection).numFrames).toBe(273);
    expect(smartSeed("image-to-video", selection).numFrames).toBe(361);
  });

  it("#4 stays on the legacy table for LTX 2.3's default configuration, and goes smart on LTX 2.5", () => {
    const selection = makeSelection({ mediaWidth: 1280, mediaHeight: 768 });
    const defaults = {
      attentionBackend: "sdpa",
      blockSwapPrefetch: true,
      keepResident: false,
      fusedGgufDequantKernel: true,
      vaeMode: "default",
    } as const;
    const withDefaults = (engineFamily: string) =>
      resolvePrefillSeed({
        intent: "image-to-video",
        selection,
        config: cfg,
        sizePolicy: "material",
        fpsPolicy: "material",
        engineFamily,
        acceleration: defaults,
        sageAvailable: true,
      }).numFrames;
    expect(withDefaults("ltx")).toBe(273);
    expect(withDefaults("ltx25")).toBe(361);
  });

  it("Chain-系 intents keep the legacy ceiling even with the trio threaded", () => {
    // `image-to-clip-chain` (#6) targets the Chained screen, so Create's
    // single-shot line must not follow it there.
    const selection = makeSelection({ mediaWidth: 1280, mediaHeight: 768 });
    expect(smartSeed("image-to-clip-chain", selection).numFrames).toBe(273);
    expect(smartSeed("extend-video", selection).numFrames).toBe(273);
  });

  it("#2 reference-video's material clamp moves onto the smart line too", () => {
    // A 721-frame span at 1280x768 is far past either ceiling, so the clamp
    // decides — and it is the smart one.
    const selection = makeSelection(
      { mediaWidth: 1280, mediaHeight: 768, frameStart: 0, frameEnd: 720, mediaDurationSec: 2 },
      { rate: 24, scale: 1 },
    );
    expect(seed("reference-video", selection).numFrames).toBe(273);
    expect(smartSeed("reference-video", selection).numFrames).toBe(361);
  });
});
