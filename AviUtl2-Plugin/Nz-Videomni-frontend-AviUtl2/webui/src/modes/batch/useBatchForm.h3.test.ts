import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useKeyframes } from "../single/useKeyframes";
import { __resetRunLockForTests } from "../../shell/runLock";
import { __resetBatchRuntimeForTests } from "./runtime";
import { useBatchForm } from "./useBatchForm";
import type { BatchEngineRules, BatchGenerationValues } from "./useBatchForm";

// §1-126: Batch A2V on MiniMax H3 (`engine` given — `constraints.singleSourceAudio`
// published). Rows sized on 17n+5, the start gates on H3's grid and its 32
// multiple, and every a2v row sent to `/generate` with its `source_audio`.
// `useBatchForm.test.ts` (no `engine`) pins the LTX side unchanged.

const WAV_DIR = "C:\\voice\\h3";
const IMG_DIR = "C:\\pics\\h3";

const H3_ENGINE: BatchEngineRules = {
  frameGrid: { step: 17, offset: 5 },
  minNumFrames: 22,
  maxNumFrames: 481,
  dimensionMultiple: 32,
  omitImageStrength: true,
  omitLoraAudioStrength: true,
  sampling: { numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 },
  sourceAudioStrength: 0.65,
};

/** 481 is 17n+5, and 512/320 sit on both 32 and 64. */
const GEN: BatchGenerationValues = { width: 512, height: 320, cropOutput: null, frameRate: 24, seed: 1, numFrames: 481 };

function captureSubmitBridge(base: NativeBridge): {
  wrapped: NativeBridge;
  submits: Array<{ path: string; body: Record<string, unknown> }>;
} {
  const submits: Array<{ path: string; body: Record<string, unknown> }> = [];
  const wrapped: NativeBridge = {
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.request") {
        const p = params as { method?: string; path?: string; body?: Record<string, unknown> };
        if (p.method === "POST" && (p.path === "/api/v1/generate" || p.path === "/api/v1/generate/chain")) {
          submits.push({ path: p.path, body: p.body ?? {} });
          return {
            status: 500,
            body: { error: { code: "MOCK_STOP", message: "captured by test" } },
          } as unknown as ResultOf<M>;
        }
      }
      return base.request(method, params);
    },
    requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
    on: (event, handler) => base.on(event, handler),
  };
  return { wrapped, submits };
}

function render(bridge: NativeBridge, opts: { gen?: BatchGenerationValues; engine?: BatchEngineRules | undefined } = {}) {
  const gen = opts.gen ?? GEN;
  const engine = "engine" in opts ? opts.engine : H3_ENGINE;
  return renderHook(() => {
    const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
    return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, {
      nativeBridge: bridge,
      ...(engine ? { engine } : {}),
    });
  });
}

function audioFs(durations: Record<string, number>) {
  return createMockFs({
    folders: {
      [WAV_DIR]: Object.entries(durations).map(([name, durationSec], i) => ({
        name,
        sizeBytes: 100,
        mtimeMs: 1000 + i,
        durationSec,
      })),
      [IMG_DIR]: [{ name: "cover.png", sizeBytes: 10, mtimeMs: 1 }],
    },
  });
}

async function scanAudio(result: { current: ReturnType<typeof useBatchForm> }) {
  await act(async () => {
    await result.current.setWavDir(WAV_DIR);
  });
  await act(async () => {
    await result.current.setImgDir(IMG_DIR);
  });
  await act(async () => {
    await result.current.scan();
  });
}

beforeEach(() => {
  __resetRunLockForTests();
  __resetBatchRuntimeForTests();
});

describe("useBatchForm — §1-126 MiniMax H3", () => {
  it("sizes a2v rows on 17n+5 (the largest value the audio covers) and Skips past the cap", async () => {
    // 5 s = 120 frames -> 107; 2 s = 48 -> 39; 30 s = 720 > 481 -> over-cap.
    const bridge = createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 5, "b.wav": 2, "c.wav": 30 }) });
    const { result } = render(bridge);
    await scanAudio(result);
    const byWav = new Map(result.current.rows.map((r) => [r.wav, r]));
    expect(byWav.get("a.wav")).toMatchObject({ stat: "Waiting", frames: 107 });
    expect(byWav.get("b.wav")).toMatchObject({ stat: "Waiting", frames: 39 });
    expect(byWav.get("c.wav")).toMatchObject({ stat: "Skip", skipReason: "over-cap" });
    for (const row of result.current.rows) if (row.stat === "Waiting") expect((row.frames - 5) % 17).toBe(0);
  });

  it("the over-cap line is min(DURATION, the engine's ceiling)", async () => {
    // 15.5 s = 372 frames -> 362 on the grid: under 481, over a DURATION of 345.
    const bridge = createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 15.5 }) });
    const low = render(bridge, { gen: { ...GEN, numFrames: 345 } });
    await scanAudio(low.result);
    expect(low.result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "over-cap" });

    // An engine ceiling under the DURATION wins: capped at 409 here.
    const high = render(createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 19.9 }) }), {
      engine: { ...H3_ENGINE, maxNumFrames: 409 },
    });
    await scanAudio(high.result);
    // 19.9 s = 477 frames -> 464 on the grid: under DURATION 481, over the 409 ceiling.
    expect(high.result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "over-cap" });
  });

  it("start: an a2v row goes to /generate with source_audio and Create's omissions and sampling values", async () => {
    const base = createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 5 }) });
    const { wrapped, submits } = captureSubmitBridge(base);
    const { result } = render(wrapped);
    await scanAudio(result);
    act(() => {
      result.current.updateRowImage(0, "cover.png");
    });
    await act(async () => {
      await result.current.setOutDir("C:\\out\\h3");
    });
    expect(result.current.canStart).toBe(true);
    await act(async () => {
      result.current.start();
    });
    await act(async () => {
      for (let i = 0; i < 50 && submits.length === 0; i += 1) await new Promise((r) => setTimeout(r, 5));
    });
    expect(submits).toHaveLength(1);
    const { path, body } = submits[0]!;
    expect(path).toBe("/api/v1/generate");
    expect(body.num_frames).toBe(107);
    expect(body.source_audio).toEqual({ audio_id: expect.stringMatching(/^mock-audio-/), strength: 0.65 });
    expect(body.conditioning_images).toEqual([{ image_id: expect.stringMatching(/^mock-image-/), frame_idx: 0 }]);
    expect(body).toMatchObject({ num_inference_steps: 20, shift_video: 12, shift_audio: 3 });
    expect(body).not.toHaveProperty("clips");
  });

  it("the size gate uses the engine's multiple (32): 544 passes on H3 and fails on LTX", async () => {
    const gen = { ...GEN, width: 544 };
    const h3 = render(createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 5 }) }), { gen });
    expect(h3.result.current.resolutionValid).toBe(true);
    const ltx = render(createMockBridge({ delayMs: 0, fs: audioFs({ "a.wav": 5 }) }), { gen, engine: undefined });
    expect(ltx.result.current.resolutionValid).toBe(false);
  });

  it("i2v: the DURATION gate is on 17n+5 (124 passes, 121 does not)", async () => {
    const imageFs = createMockFs({ folders: { [IMG_DIR]: [{ name: "cover.png", sizeBytes: 10, mtimeMs: 1 }] } });
    for (const [numFrames, offGrid] of [
      [124, false],
      [121, true],
    ] as const) {
      const { result } = render(createMockBridge({ delayMs: 0, fs: imageFs }), { gen: { ...GEN, numFrames } });
      await act(async () => {
        await result.current.setImgDir(IMG_DIR);
      });
      await act(async () => {
        await result.current.scan();
      });
      expect(result.current.mode).toBe("i2v");
      expect(result.current.numFramesOffGrid, String(numFrames)).toBe(offGrid);
    }
  });
});
