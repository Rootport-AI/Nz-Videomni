import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { BaseModelConstraints } from "../../api/types";
import { createMockBridge } from "../../bridge/mockBridge";
import { en, ja } from "../../i18n/strings";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { buildChainReasonMessages } from "./generateReasonMessages";
import { useChainForm } from "./useChainForm";
import type { ChainUnavailablePanels, UseChainFormDeps } from "./useChainForm";

// §1-106: Chained's form under the served constraints. Every pre-existing
// `useChainForm` test omits `constraints` and so pins the LTX side; this file
// pins what changes on MiniMax H3 (17n+5 clips, 32 multiple, 24fps, sampling,
// the server-fixed seam) and that a served LTX block changes nothing.

const H3_BLOCK: BaseModelConstraints = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  chain: { overlap_frames: 22 },
};
const LTX_BLOCK: BaseModelConstraints = {
  num_frames: { step: 8, offset: 1, min: 9, max: 481 },
  dimension_multiple: 64,
  frame_rate: { min: 1, max: 60 },
  sampling: null,
};
const H3 = resolveGenerationConstraints(H3_BLOCK);
const LTX = resolveGenerationConstraints(LTX_BLOCK);

const H3_CLOSED: ChainUnavailablePanels = { v2v: true, startImage: true, a2v: true, endSource: true, reference: true };

function renderForm(constraints: GenerationConstraints | undefined, extra: Partial<UseChainFormDeps> = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return renderHook(
    ({ c }: { c: GenerationConstraints | undefined }) =>
      useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge, constraints: c, ...extra }),
    { initialProps: { c: constraints } },
  );
}

describe("resolveGenerationConstraints — `chain` (§1-106)", () => {
  it("flattens `constraints.chain` and reads an absent block as null", () => {
    expect(H3.chain).toEqual({ overlapFrames: 22 });
    expect(LTX.chain).toBeNull();
    expect(resolveGenerationConstraints(null).chain).toBeNull();
  });
});

describe("useChainForm — MiniMax H3 constraints (§1-106)", () => {
  it("seeds every clip on 17n+5, the size on 32, fps at 24 and the sampling defaults", () => {
    const { result } = renderForm(H3);
    // FALLBACK_APP_CONFIG's default length (345) is already a 17n+5 value.
    for (const clip of result.current.clips) expect((clip.numFrames - 5) % 17).toBe(0);
    expect(result.current.clipFrameLimits).toEqual({ min: 22, max: 481, step: 17 });
    expect(result.current.sizeMultiple).toBe(32);
    expect(result.current.common.frameRate).toBe(24);
    expect(result.current.fixedFrameRate).toBe(24);
    expect(result.current.sampling).toEqual({ numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 });
    expect(result.current.fixedOverlapFrames).toBe(22);
  });

  it("snaps a clip length onto 17n+5 and pins fps against a typed value", () => {
    const { result } = renderForm(H3);
    const id = result.current.clips[0]!.id;
    act(() => result.current.setClipNumFrames(id, 121));
    expect(result.current.clips[0]!.numFrames).toBe(124);
    act(() => result.current.setFrameRate(30));
    expect(result.current.common.frameRate).toBe(24);
    act(() => result.current.setWidth(1000));
    expect(result.current.common.width % 32).toBe(0);
  });

  it("an 8n+1 length typed freely is off H3's grid and blocks Generate", () => {
    const { result } = renderForm(H3);
    act(() => result.current.setClipNumFrames(result.current.clips[0]!.id, 121, false));
    expect(result.current.validityReasons).toContain("clipFramesOffGrid");
  });

  it("output length is Σ − 22 × (n − 1), and the audio note follows the seam", () => {
    const { result } = renderForm(H3);
    const [c0, c1] = result.current.clips;
    act(() => {
      result.current.setClipNumFrames(c0!.id, 124);
      result.current.setClipNumFrames(c1!.id, 124);
    });
    expect(result.current.outputFrames).toBe(226);
    expect(result.current.clipAudioDriftFlags).toEqual([false, false]);
    act(() => result.current.setClipNumFrames(c0!.id, 141));
    expect(result.current.outputFrames).toBe(243);
    expect(result.current.clipAudioDriftFlags).toEqual([false, true]);
    // Never a gate.
    expect(result.current.isValid).toBe(true);
  });

  it("clips after the first must reach 39 (overlap + 17); the first may be 22", () => {
    const { result } = renderForm(H3);
    const [c0, c1] = result.current.clips;
    act(() => {
      result.current.setClipNumFrames(c0!.id, 22);
      result.current.setClipNumFrames(c1!.id, 39);
    });
    expect(result.current.minFramesForOverlap).toBe(39);
    expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
    act(() => result.current.setClipNumFrames(c1!.id, 22));
    expect(result.current.validityReasons).toContain("clipTooShortForOverlap");
  });

  it("the fixed-overlap line replaces the seam-blend one (en and ja)", () => {
    const { result } = renderForm(H3);
    const params = {
      minFramesForOverlap: result.current.minFramesForOverlap,
      minClips: result.current.minClips,
      fixedOverlapFrames: result.current.fixedOverlapFrames,
    };
    expect(buildChainReasonMessages(en, params).clipTooShortForOverlap).toBe(
      en.chained.generateReasons.clipTooShortForFixedOverlap(39),
    );
    expect(buildChainReasonMessages(ja, params).clipTooShortForOverlap).toContain("2本目以降");
    // LTX keeps its sentence.
    expect(buildChainReasonMessages(en, { minFramesForOverlap: 25, minClips: 2 }).clipTooShortForOverlap).toBe(
      en.chained.generateReasons.clipTooShortForOverlap(25),
    );
  });

  it("raises no LTX-geometry reason on a plain H3 chain", () => {
    const { result } = renderForm(H3, { unavailablePanels: H3_CLOSED });
    expect(result.current.validityReasons).toEqual([]);
    expect(result.current.isValid).toBe(true);
  });

  it("the request carries the sampling values and none of the four LTX seam fields", () => {
    const { result } = renderForm(H3);
    act(() => result.current.setSamplingValue("numInferenceSteps", 8.4));
    act(() => result.current.setStage2Window("high_resolution"));
    const req = result.current.buildRequest();
    expect(req.num_inference_steps).toBe(8);
    expect(req.shift_video).toBe(12);
    expect(req.shift_audio).toBe(3);
    expect(req.frame_rate).toBe(24);
    for (const key of ["overlap_frames", "overlap_strength", "chunked_upsample", "stage2_window"]) {
      expect(req).not.toHaveProperty(key);
    }
  });

  it("a served LTX block changes nothing (same request keys as the LTX fallback)", () => {
    const served = renderForm(LTX).result.current.buildRequest();
    const fallback = renderForm(undefined).result.current.buildRequest();
    expect(Object.keys(served)).toEqual(Object.keys(fallback));
    expect(served).toMatchObject({ overlap_frames: 3, overlap_strength: 0.5, chunked_upsample: true });
    expect(served).not.toHaveProperty("num_inference_steps");
  });

  it("a base-model switch re-snaps the clips, the size and fps, and resets the sampling sliders", () => {
    const { result, rerender } = renderForm(undefined);
    const [c0, c1] = result.current.clips;
    act(() => {
      result.current.setClipNumFrames(c0!.id, 121);
      result.current.setClipNumFrames(c1!.id, 49);
      result.current.setFrameRate(30);
    });
    rerender({ c: H3 });
    expect(result.current.clips.map((clip) => clip.numFrames)).toEqual([124, 56]);
    expect(result.current.common.frameRate).toBe(24);
    expect(result.current.sampling).toEqual({ numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 });
    expect(result.current.validityReasons).toEqual([]);
    rerender({ c: LTX });
    expect(result.current.clips.every((clip) => (clip.numFrames - 1) % 8 === 0)).toBe(true);
    expect(result.current.sampling).toBeNull();
    expect(result.current.fixedOverlapFrames).toBeNull();
  });
});

describe("useChainForm — material left in a closed panel (§1-106)", () => {
  it("raises `materialInClosedPanel` only for a CLOSED panel that holds something", async () => {
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const { result, rerender } = renderHook(
      ({ closed }: { closed: ChainUnavailablePanels }) =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge, unavailablePanels: closed }),
      { initialProps: { closed: {} as ChainUnavailablePanels } },
    );
    await act(async () => {
      await result.current.attachSourceByPath("C:\\img\\start.png", "start.png");
    });
    expect(result.current.validityReasons).not.toContain("materialInClosedPanel");
    rerender({ closed: { v2v: true } });
    expect(result.current.validityReasons).not.toContain("materialInClosedPanel");
    rerender({ closed: { startImage: true } });
    expect(result.current.validityReasons).toContain("materialInClosedPanel");
    expect(result.current.isValid).toBe(false);
    // Removing it (the clear button stays live in a closed panel) lifts the block.
    act(() => result.current.clearSource());
    expect(result.current.validityReasons).not.toContain("materialInClosedPanel");
    expect(result.current.isValid).toBe(true);
  });
});

describe("useChainForm — presets and the added-clip slider under the loaded rules (§1-106)", () => {
  it("a preset on a fixed-overlap base model takes the preset's own length on 17n+5 (no LTX comfort ceiling)", () => {
    const config = {
      ...FALLBACK_APP_CONFIG,
      generation_presets: { p: { width: 1280, height: 704, crop_output: null, num_frames: 121 } },
      limits: { ...FALLBACK_APP_CONFIG.limits, spill_free_frames: { "1280x704": 257 } },
    };
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const h3 = renderHook(() => useChainForm(config, "x", { nativeBridge, constraints: H3 })).result;
    act(() => h3.current.applyPreset("p"));
    expect(h3.current.clips.map((clip) => clip.numFrames)).toEqual([124, 124]);
    expect(h3.current.addedClipFrames).toBe(124);
    // LTX keeps reading its comfort ceiling.
    const ltx = renderHook(() => useChainForm(config, "x", { nativeBridge })).result;
    act(() => ltx.current.applyPreset("p"));
    expect(ltx.current.clips.map((clip) => clip.numFrames)).toEqual([257, 257]);
  });

  it("a base-model switch re-snaps the added-clip slider, and a return to LTX puts it back on 8n+1", () => {
    const { result, rerender } = renderForm(undefined);
    expect(result.current.addedClipFrames).toBe(257);
    rerender({ c: H3 });
    expect((result.current.addedClipFrames - 5) % 17).toBe(0);
    expect(result.current.addedClipFrames).toBe(260);
    rerender({ c: LTX });
    expect((result.current.addedClipFrames - 1) % 8).toBe(0);
  });
});

describe("useChainForm — a right-click seed under MiniMax H3's rules (§1-106)", () => {
  it("snaps the seed's size, fps and clip-0 length onto the rules at INIT (not only on a rule change)", () => {
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      useChainForm(
        FALLBACK_APP_CONFIG,
        "x",
        { nativeBridge, constraints: H3 },
        { width: 1920, height: 1080, frameRate: 30, numFrames: 121 },
      ),
    );
    expect(result.current.common.width).toBe(1920);
    expect(result.current.common.height).toBe(1088);
    expect(result.current.common.frameRate).toBe(24);
    expect(result.current.clips[0]!.numFrames).toBe(124);
    expect(result.current.validityReasons).toEqual([]);
  });
});
