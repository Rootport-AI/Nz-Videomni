import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BaseModelConstraints } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { H3_NO_CONDITIONING, h3ComfortFrames, h3SequenceRows } from "../../shell/h3SequenceRows";
import { publishInpaintPartialFilter, publishInpaintTarget, resetInpaintSlots } from "../../timeline/inpaintSlots";
import type { SelectionItem, TimelineSelection } from "../../timeline/menuSelection";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useInpaintForm } from "./useInpaintForm";
import type { UseInpaintFormDeps } from "./useInpaintForm";

/**
 * 台帳 §1-124 弾 2: MiniMax H3 の Inpainting（フォームの hook）。
 *
 * 配信に `constraints.inpaint` があるとき、フォームは H3 の形になる: キャンバスは 32 の
 * 倍数、fps は 24 固定、フレーム数は 17n+5・[22, 345]（24fps の枚数）、窓・マスクは
 * プロジェクトの fps に換算した長さ（切り上げ）、席は `{F, 24}`、要求は
 * `inpaint.source_video_id`＋`audio_strength`＋`blend_dilation_stage1` で、
 * `reference_video_id`・固定 LoRA・`blend_dilation_stage2` は送らない。
 * LTX の形は `useInpaintForm.test.tsx` が従来どおり固定している。
 */

const FILE = "C:\\videos\\shot.mp4";
const MASK_FILE = "C:\\Users\\mock\\AppData\\Local\\NzVideomni\\masks\\mask_100_361_0001.mp4";

const H3_BLOCK: BaseModelConstraints = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  keyframes: { step: 1, offset: 0 },
  inpaint: {
    audio_strength: { min: 0, max: 1, step: 0.05, default: 0.5 },
    blend_dilation_stage1: { min: 0, max: 15, step: 1, default: 1 },
  },
};
const H3 = resolveGenerationConstraints(H3_BLOCK);
/** 既定が違う H3 の配信（追従のテスト用）。 */
const H3_OTHER = resolveGenerationConstraints({
  ...H3_BLOCK,
  inpaint: {
    audio_strength: { min: 0, max: 1, step: 0.05, default: 0.25 },
    blend_dilation_stage1: { min: 0, max: 10, step: 1, default: 3 },
  },
});

/** 素材 40 秒・リボンは 0〜599（600 フレーム = 20.0 秒 @30fps）・再生窓は素材の
 * 2.0〜22.0 秒（`useInpaintForm.test.tsx` と同じ）。 */
function makeItem(overrides: Partial<SelectionItem> = {}): SelectionItem {
  return {
    layer: 3,
    frameStart: 0,
    frameEnd: 599,
    effectName: "動画ファイル",
    filePath: FILE,
    objectName: "shot",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 40,
    hasPlaybackRange: true,
    playbackStartSec: 2,
    playbackEndSec: 22,
    ...overrides,
  };
}

function makeSelection(item: SelectionItem = makeItem()): TimelineSelection {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [item],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

function createBridge() {
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "getEditInfo") {
      return {
        width: 1920,
        height: 1080,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
        frame: 0,
        layer: 0,
        frameMax: 600,
        layerMax: 100,
      };
    }
    if (method === "backend.uploadFile") {
      const filePath = (params as { filePath?: string }).filePath;
      if (filePath === MASK_FILE) return { status: 200, body: { video_id: "mask-1" } };
      return { status: 200, body: { video_id: "vid-1", trimmed: true } };
    }
    if (method === "timeline.renderMaskVideo") {
      return { filePath: MASK_FILE, width: 1920, height: 1080, frameCount: 262 };
    }
    throw new Error(`unexpected bridge call: ${method}`);
  });
  const on = vi.fn(() => () => {});
  const bridge = { request, requestWithFiles: vi.fn(), on } as unknown as NativeBridge;
  return { bridge, request };
}

interface Props {
  constraints: GenerationConstraints | undefined;
}

function renderForm(
  prompt = "a cat",
  extra: Partial<UseInpaintFormDeps> = {},
  constraints: GenerationConstraints | null = H3,
) {
  const harness = createBridge();
  const view = renderHook(
    (props: Props) =>
      useInpaintForm({
        prompt,
        nativeBridge: harness.bridge,
        ...extra,
        ...(props.constraints ? { constraints: props.constraints } : {}),
      }),
    { initialProps: { constraints: constraints ?? undefined } },
  );
  return { ...view, ...harness };
}

/** 部分フィルタ 100..340（241 フレーム @30fps）と対象動画を両方置く。 */
function seedBothSlots(item: SelectionItem = makeItem()) {
  act(() => {
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    publishInpaintTarget({ item, selection: makeSelection(item) });
  });
}

beforeEach(() => {
  resetInpaintSlots();
});
afterEach(() => {
  resetInpaintSlots();
  vi.restoreAllMocks();
});

describe("useInpaintForm（H3）— 形の切り替え", () => {
  it("`constraints.inpaint` があれば h3、キャンバスは 32 の倍数（1920×1080 → 1920×1088）、fps は 24 固定", async () => {
    seedBothSlots();
    const { result } = renderForm();
    expect(result.current.h3).toBe(true);
    expect(result.current.canvasWidth).toBe(1920);
    expect(result.current.canvasHeight).toBe(1088);
    expect(result.current.frameRate).toBe(24);
    expect(result.current.fixedFrameRate).toBe(24);
    expect(result.current.frameGrid).toEqual({ step: 17, offset: 5 });
    expect(result.current.minFrames).toBe(22);
    expect(result.current.maxFrames).toBe(345);
    await waitFor(() => expect(result.current.isValid).toBe(true));
  });

  it("配信が無ければ（LTX）h3 ではなく、従来の 128・プロジェクト fps のまま", () => {
    seedBothSlots();
    const { result } = renderForm("a cat", {}, null);
    expect(result.current.h3).toBe(false);
    expect(result.current.canvasHeight).toBe(1152);
    expect(result.current.frameRate).toBe(30);
    expect(result.current.fixedFrameRate).toBeNull();
    expect(result.current.audioStrengthRange).toBeNull();
    expect(result.current.sampling).toBeNull();
  });

  it("対象動画の寸法が不明なら 0 のままで mediaInfoUnknown が止める", async () => {
    seedBothSlots(makeItem({ mediaWidth: 0, mediaHeight: 0 }));
    const { result } = renderForm();
    expect(result.current.canvasWidth).toBe(0);
    expect(result.current.canvasKnown).toBe(false);
    await waitFor(() => expect(result.current.validityReasons).toContain("mediaInfoUnknown"));
  });
});

describe("useInpaintForm（H3）— フレーム数と窓", () => {
  it("初期値は部分フィルタの長さを 24fps に換算して 17n+5 へ切り上げた値（241@30fps → 193 → 209）", () => {
    seedBothSlots();
    const { result } = renderForm();
    expect(result.current.numFrames).toBe(209);
  });

  it("部分フィルタが無いうちの値も格子に乗っている", () => {
    const { result } = renderForm();
    expect((result.current.numFrames - 5) % 17).toBe(0);
    expect(result.current.numFrames).toBeGreaterThanOrEqual(22);
    expect(result.current.numFrames).toBeLessThanOrEqual(345);
  });

  it("確定時に 17n+5 へ乗せ、[22, 345] にクランプする（手打ち中は素通し）", () => {
    seedBothSlots();
    const { result } = renderForm();
    act(() => result.current.setNumFrames(100, false));
    expect(result.current.numFrames).toBe(100);
    act(() => result.current.commitNumFrames(100));
    expect(result.current.numFrames).toBe(107);
    act(() => result.current.commitNumFrames(9999));
    expect(result.current.numFrames).toBe(345);
    act(() => result.current.commitNumFrames(1));
    expect(result.current.numFrames).toBe(22);
    act(() => result.current.setNumFrames(60, true));
    expect(result.current.numFrames).toBe(56);
  });

  it("30fps のプロジェクトでは窓の長さが ceil(F·30/24) になり、windowNotCovered が出ない（R1）", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    // F = 209 → ceil(261.25) = 262 フレーム（プロジェクト）。
    expect(result.current.window).toMatchObject({ windowStart: 100, windowEnd: 361, frames: 262, coversFilter: true });
    expect(result.current.mapping?.ok && result.current.mapping.frameCount).toBe(262);
    expect(result.current.validityReasons).not.toContain("windowNotCovered");

    act(() => result.current.commitNumFrames(56));
    // ceil(56·30/24) = 70。部分フィルタより短いが、止めない（注意文だけ）。
    expect(result.current.window).toMatchObject({ windowStart: 100, windowEnd: 169, frames: 70, coversFilter: false });
    expect(result.current.validityReasons).not.toContain("windowNotCovered");
  });

  it("換算した窓がリボンに収まらなければ windowNotCovered", async () => {
    // リボン 100..360（261 フレーム）には 262 が収まらない。
    seedBothSlots(makeItem({ frameStart: 100, frameEnd: 360 }));
    const { result } = renderForm();
    await waitFor(() => expect(result.current.uploadStatus).toBe("ready"));
    expect(result.current.window).toBeNull();
    expect(result.current.validityReasons).toContain("windowNotCovered");
  });

  it("マスクは換算した窓の絶対フレームで頼む", async () => {
    seedBothSlots();
    const { result, request } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    await act(async () => {
      await result.current.renderAndUploadMask();
    });
    const render = request.mock.calls.find(([m]) => m === "timeline.renderMaskVideo");
    expect(render?.[1]).toEqual({ layer: 5, frameStart: 100, frameEnd: 340, windowStart: 100, windowEnd: 361 });
  });

  it("席の予約とジョブの紐づけは {F, 24}（M1。プロジェクトのフレーム数ではない）", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(result.current.placement).toEqual({
      layer: 3,
      frameStart: 100,
      frameEnd: 361,
      numFrames: 209,
      genFps: 24,
    });
  });
});

describe("useInpaintForm（H3）— buildRequest", () => {
  it("H3 の形: inpaint の 5 鍵、32 のキャンバス、24fps、サンプリング、タグ由来の LoRA だけ", async () => {
    seedBothSlots();
    const { result } = renderForm("a quiet street <lora:style-a:0.7:0.3>", { loraAudioHidden: true });
    await waitFor(() => expect(result.current.isValid).toBe(true));
    const body = result.current.buildRequest("mask-1");
    expect(body).toMatchObject({
      prompt: "a quiet street",
      width: 1920,
      height: 1088,
      num_frames: 209,
      frame_rate: 24,
      num_inference_steps: 20,
      shift_video: 12,
      shift_audio: 3,
    });
    // 完全一致: stage2 や余計な鍵が増えればここで落ちる。窓の頭は LTX と同じ
    // 「アップロード後ファイルの時間軸」（100/30 秒）。
    expect(body.inpaint).toEqual({
      mask_video_id: "mask-1",
      source_video_id: "vid-1",
      window_start_sec: 3.333,
      audio_strength: 0.5,
      blend_dilation_stage1: 1,
    });
    // 固定の in-outpainting は無く、音声の強度は落ちる。
    expect(body.loras).toEqual([{ name: "style-a", strength: 0.7 }]);
    expect(body).not.toHaveProperty("reference_video_id");
  });

  it("タグが無ければ loras の鍵ごと省く", async () => {
    seedBothSlots();
    const { result } = renderForm("");
    await waitFor(() => expect(result.current.isValid).toBe(true));
    const body = result.current.buildRequest("mask-1");
    expect(body).not.toHaveProperty("loras");
    expect(body.prompt).toBe("");
  });

  it("動かした値（なじませ・音声・サンプリング）が載り、stage2 は送らない", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    act(() => {
      result.current.setBlendStage1(5);
      result.current.setAudioStrength(0.8);
      result.current.setSamplingValue("numInferenceSteps", 30);
    });
    const body = result.current.buildRequest("mask-1");
    expect(body.inpaint).toMatchObject({ blend_dilation_stage1: 5, audio_strength: 0.8 });
    expect(body.inpaint).not.toHaveProperty("blend_dilation_stage2");
    expect(body.num_inference_steps).toBe(30);
  });
});

describe("useInpaintForm（H3）— 既定と配信への追従", () => {
  it("既定はなじませ 1・音声 0.5（配信の既定）。範囲は配信値でクランプする", () => {
    const { result } = renderForm();
    expect(result.current.blendStage1).toBe(1);
    expect(result.current.blendRange).toEqual({ min: 0, max: 15, step: 1 });
    expect(result.current.audioStrength).toBe(0.5);
    expect(result.current.audioStrengthRange).toMatchObject({ min: 0, max: 1, step: 0.05, default: 0.5 });
    expect(result.current.sampling).toEqual({ numInferenceSteps: 20, shiftVideo: 12, shiftAudio: 3 });
    act(() => result.current.setBlendStage1(99));
    expect(result.current.blendStage1).toBe(15);
    act(() => result.current.setAudioStrength(2));
    expect(result.current.audioStrength).toBe(1);
  });

  it("❌ でいまの配信の既定へ戻る", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    act(() => {
      result.current.setBlendStage1(7);
      result.current.setAudioStrength(0.1);
    });
    act(() => result.current.clearAll());
    expect(result.current.blendStage1).toBe(1);
    expect(result.current.audioStrength).toBe(0.5);
  });

  it("LTX → H3 → 別の H3 の配信へ切り替わると、既定とフレーム数を引き直す", () => {
    seedBothSlots();
    const { result, rerender } = renderForm("a cat", {}, null);
    expect(result.current.h3).toBe(false);
    expect(result.current.blendStage1).toBe(5);
    expect(result.current.numFrames).toBe(241);

    rerender({ constraints: H3 });
    expect(result.current.h3).toBe(true);
    expect(result.current.blendStage1).toBe(1);
    expect(result.current.audioStrength).toBe(0.5);
    expect(result.current.numFrames).toBe(209);

    act(() => result.current.setBlendStage1(9));
    rerender({ constraints: H3_OTHER });
    expect(result.current.blendStage1).toBe(3);
    expect(result.current.blendRange).toEqual({ min: 0, max: 10, step: 1 });
    expect(result.current.audioStrength).toBe(0.25);

    rerender({ constraints: undefined });
    expect(result.current.h3).toBe(false);
    expect(result.current.blendStage1).toBe(5);
    expect(result.current.audioStrength).toBeNull();
    expect(result.current.sampling).toBeNull();
    expect(result.current.numFrames).toBe(241);
  });
});

describe("useInpaintForm（H3）— 快適上限", () => {
  it("総行数の線（single_rows_budget）で目盛りと処理量を出す（Outpainting の H3 と同じ部品）", () => {
    seedBothSlots();
    const { result } = renderForm("a cat", { config: FALLBACK_APP_CONFIG, engineFamily: "h3" });
    const budget = FALLBACK_APP_CONFIG.limits.comfort_budgets?.h3?.single_rows_budget as number;
    const shape = { width: 1920, height: 1088, ...H3_NO_CONDITIONING };
    expect(result.current.comfortFrames).toBe(h3ComfortFrames(shape, budget, 22, 345));
    const rows = h3SequenceRows(shape, result.current.numFrames);
    expect(result.current.comfortTokens).toBe(rows?.total);
    expect(result.current.isOverComfortBudget).toBe((rows?.total ?? 0) > budget);
  });
});
