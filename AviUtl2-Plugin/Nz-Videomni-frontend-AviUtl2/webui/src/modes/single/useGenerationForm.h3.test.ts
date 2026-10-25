import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AppConfig, BaseModelConstraints } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import { suggestFramesForAudio } from "../chained/chainUtils";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { useGenerationForm } from "./useGenerationForm";
import type { GenerationFormInitial } from "./useGenerationForm";

// MiniMax H3 第 2 弾 (2026-10-17): Create's form under the served constraints.
// Every pre-existing `useGenerationForm` test omits `constraints` and so pins
// the LTX side; this file pins what changes on MiniMax H3 and that a served LTX
// block changes nothing.

const H3_BLOCK: BaseModelConstraints = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
};
const LTX_BLOCK: BaseModelConstraints = {
  num_frames: { step: 8, offset: 1, min: 9, max: 481 },
  dimension_multiple: 64,
  frame_rate: { min: 1, max: 60 },
  sampling: null,
};
const H3 = resolveGenerationConstraints(H3_BLOCK);
const LTX = resolveGenerationConstraints(LTX_BLOCK);

function renderForm(
  constraints: GenerationConstraints | undefined,
  engineFamily: string | undefined,
  initial: GenerationFormInitial = {},
  config: AppConfig = FALLBACK_APP_CONFIG,
) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return renderHook(
    ({ c, family }: { c: GenerationConstraints | undefined; family: string | undefined }) =>
      useGenerationForm(config, "a prompt", { nativeBridge, constraints: c, engineFamily: family }, initial),
    { initialProps: { c: constraints, family: engineFamily } },
  );
}

describe("useGenerationForm — MiniMax H3 constraints", () => {
  it("seeds DURATION on 17n+5, fps at the fixed 24 and the sampling defaults", () => {
    const { result } = renderForm(H3, "h3", { frameRate: 30, numFrames: 121 });
    expect(result.current.values.numFrames).toBe(124);
    expect(result.current.values.frameRate).toBe(24);
    expect(result.current.fixedFrameRate).toBe(24);
    expect(result.current.values.sampling).toEqual({ numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 });
    expect(result.current.limits.minNumFrames).toBe(22);
    expect(result.current.frameGrid).toEqual({ step: 17, offset: 5 });
    expect(result.current.dimensionStep).toBe(32);
  });

  it("snaps DURATION edits onto 17n+5 and gates a hand-typed off-grid value", () => {
    const { result } = renderForm(H3, "h3");
    act(() => result.current.setNumFrames(200));
    expect(result.current.values.numFrames).toBe(192);
    act(() => result.current.setNumFrames(10));
    expect(result.current.values.numFrames).toBe(22);
    act(() => result.current.setNumFrames(121, false));
    expect(result.current.validityReasons).toContain("numFramesOffGrid");
    act(() => result.current.setNumFrames(124, false));
    expect(result.current.validityReasons).not.toContain("numFramesOffGrid");
  });

  it("snaps width/height onto 32 and refuses to move the fixed fps", () => {
    const { result } = renderForm(H3, "h3");
    act(() => result.current.setWidth(1300));
    expect(result.current.values.width).toBe(1312);
    act(() => result.current.setFrameRate(30));
    expect(result.current.values.frameRate).toBe(24);
  });

  it("draws no comfort marker for an engine the served table has no profile for", () => {
    const { result } = renderForm(H3, "h3");
    expect(result.current.spillThresholdFrames).toBeNull();
    expect(result.current.isOverSpillThreshold).toBe(false);
  });

  it("keeps the LTX fallback marker for an LTX row miss (M-6)", () => {
    // `ltx` + the default acceleration + an unknown weight class: the table
    // HAS an `ltx` profile but no row matches, so the legacy lookup applies.
    const { result } = renderForm(LTX, "ltx");
    expect(result.current.spillThresholdFrames).not.toBeNull();
  });

  it("sends the three sampling values on every request, and the setter clamps and rounds", () => {
    const { result } = renderForm(H3, "h3");
    let body = result.current.toGenerateRequest();
    expect(body.num_inference_steps).toBe(20);
    expect(body.shift_video).toBe(12);
    expect(body.shift_audio).toBe(3);
    expect(body.frame_rate).toBe(24);

    act(() => result.current.setSamplingValue("numInferenceSteps", 12.6));
    act(() => result.current.setSamplingValue("shiftVideo", 99));
    act(() => result.current.setSamplingValue("shiftAudio", 0));
    body = result.current.toGenerateRequest();
    expect(body.num_inference_steps).toBe(13);
    expect(body.shift_video).toBe(20);
    expect(body.shift_audio).toBe(1);
  });

  it("LTX (served or not) sends none of the three and keeps its old grid", () => {
    for (const c of [LTX, undefined]) {
      const { result } = renderForm(c, "ltx");
      const body = result.current.toGenerateRequest();
      expect(body).not.toHaveProperty("num_inference_steps");
      expect(body).not.toHaveProperty("shift_video");
      expect(body).not.toHaveProperty("shift_audio");
      expect(result.current.values.sampling).toBeNull();
      expect(result.current.samplingConstraints).toBeNull();
      expect(result.current.fixedFrameRate).toBeNull();
      expect((result.current.values.numFrames - 1) % 8).toBe(0);
      act(() => result.current.setSamplingValue("numInferenceSteps", 30));
      expect(result.current.values.sampling).toBeNull();
    }
  });

  it("follows a base-model switch: re-snaps, pins fps and resets the sliders to the new defaults", () => {
    const { result, rerender } = renderForm(LTX, "ltx", { frameRate: 30, numFrames: 121 });
    expect(result.current.values.numFrames).toBe(121);
    expect(result.current.values.frameRate).toBe(30);

    rerender({ c: H3, family: "h3" });
    expect(result.current.values.numFrames).toBe(124);
    expect(result.current.values.frameRate).toBe(24);
    expect(result.current.values.sampling).toEqual({ numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 });

    act(() => result.current.setSamplingValue("numInferenceSteps", 30));
    // The same rules again (a re-render, not a switch) keep the user's value.
    rerender({ c: H3, family: "h3" });
    expect(result.current.values.sampling?.numInferenceSteps).toBe(30);

    rerender({ c: LTX, family: "ltx" });
    expect((result.current.values.numFrames - 1) % 8).toBe(0);
    expect(result.current.values.sampling).toBeNull();
    expect(result.current.fixedFrameRate).toBeNull();
  });

  it("code review M-1: a reference video + control LoRA chosen on LTX neither block nor ride along on MiniMax H3", () => {
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const controlLora = { name: "ltx-2.3-ic-lora-pose", strength: 1 };
    const { result, rerender } = renderHook(
      ({ closed, c }: { closed: boolean; c: GenerationConstraints }) =>
        useGenerationForm(
          FALLBACK_APP_CONFIG,
          "a prompt",
          { nativeBridge, constraints: c, engineFamily: closed ? "h3" : "ltx", controlLora, referenceVideoUnavailable: closed },
          { carryReferenceVideo: { id: "mock-video-1", fileName: "ref.mp4", filePath: "C:\ref.mp4" } },
        ),
      { initialProps: { closed: false, c: LTX } },
    );
    // On LTX both are live: IC-LoRA mode, and the request carries them.
    expect(result.current.isICLora).toBe(true);
    let body = result.current.toGenerateRequest();
    expect(body.reference_video_id).toBe("mock-video-1");
    expect(body.loras?.map((l) => l.name)).toEqual(["ltx-2.3-ic-lora-pose"]);

    // On MiniMax H3 (reference_video closed) both read as absent.
    rerender({ closed: true, c: H3 });
    expect(result.current.isICLora).toBe(false);
    expect(result.current.validityReasons).not.toContain("referenceNeedsLoras");
    expect(result.current.validityReasons).not.toContain("controlNeedsReference");
    expect(result.current.isValid).toBe(true);
    body = result.current.toGenerateRequest();
    expect(body).not.toHaveProperty("reference_video_id");
    expect(body).not.toHaveProperty("loras");
    // The user's own selection is kept (the slot still holds the video).
    expect(result.current.referenceVideo.state.id).toBe("mock-video-1");

    // Back on LTX the selection comes back.
    rerender({ closed: false, c: LTX });
    expect(result.current.isICLora).toBe(true);
    expect(result.current.toGenerateRequest().reference_video_id).toBe("mock-video-1");
  });

  it("code review M-1: a lone control LoRA (no video) does not block Generate on MiniMax H3", () => {
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a prompt", {
        nativeBridge,
        constraints: H3,
        engineFamily: "h3",
        controlLora: { name: "ltx-2.3-ic-lora-pose", strength: 1 },
        referenceVideoUnavailable: true,
      }),
    );
    expect(result.current.validityReasons).toEqual([]);
    expect(result.current.toGenerateRequest()).not.toHaveProperty("loras");
  });
});

describe("useGenerationForm — single-clip A2V (constraints.source_audio)", () => {
  const H3_A2V = resolveGenerationConstraints({
    ...H3_BLOCK,
    source_audio: { strength: { min: 0, max: 1, step: 0.05, default: 0.5 } },
  });

  function renderA2v(constraints: GenerationConstraints, engineFamily: string) {
    const fs = createMockFs({
      folders: {
        "C:\\Users\\mock\\Downloads": [{ name: "dropped.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 3.886 }],
      },
    });
    const nativeBridge = createMockBridge({ delayMs: 0, fs });
    return renderHook(
      ({ c, family }: { c: GenerationConstraints; family: string }) =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a prompt", { nativeBridge, constraints: c, engineFamily: family }),
      { initialProps: { c: constraints, family: engineFamily } },
    );
  }

  it("strength starts at the served default, is clamped by the setter, and resets on an engine switch", () => {
    const { result, rerender } = renderA2v(H3_A2V, "h3");
    expect(result.current.sourceAudioStrength).toBe(0.5);
    act(() => result.current.setSourceAudioStrength(0.4));
    expect(result.current.sourceAudioStrength).toBe(0.4);
    act(() => result.current.setSourceAudioStrength(7));
    expect(result.current.sourceAudioStrength).toBe(1);
    act(() => result.current.setSourceAudioStrength(-3));
    expect(result.current.sourceAudioStrength).toBe(0);

    act(() => result.current.setSourceAudioStrength(0.4));
    rerender({ c: LTX, family: "ltx" });
    expect(result.current.singleSourceAudio).toBeNull();
    // Without the constraint the setter is a no-op.
    act(() => result.current.setSourceAudioStrength(0.2));
    rerender({ c: H3_A2V, family: "h3" });
    expect(result.current.sourceAudioStrength).toBe(0.5);
  });

  it("audio-fit on a wav import uses the H3 grid (3.886 s -> 90) when the constraint exists, the chain logic when not", async () => {
    const h3 = renderA2v(H3_A2V, "h3");
    await act(async () => {
      await h3.result.current.attachSourceAudioByPath("C:\\Users\\mock\\Downloads\\dropped.wav", "dropped.wav");
    });
    await waitFor(() => expect(h3.result.current.values.numFrames).toBe(90));
    expect(h3.result.current.audioPrecheck?.ok).toBe(true);

    const ltx = renderA2v(LTX, "ltx");
    await act(async () => {
      await ltx.result.current.attachSourceAudioByPath("C:\\Users\\mock\\Downloads\\dropped.wav", "dropped.wav");
    });
    const expected = suggestFramesForAudio(3.886, ltx.result.current.values.frameRate);
    await waitFor(() => expect(ltx.result.current.values.numFrames).toBe(expected));
  });
});
