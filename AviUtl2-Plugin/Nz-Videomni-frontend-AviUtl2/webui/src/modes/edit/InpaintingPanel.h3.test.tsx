import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BaseModelConstraints } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { publishInpaintPartialFilter, publishInpaintTarget, resetInpaintSlots } from "../../timeline/inpaintSlots";
import type { SelectionItem, TimelineSelection } from "../../timeline/menuSelection";
import { InpaintingPanel } from "./InpaintingPanel";
import { featherWidthPx } from "./outpaintGeometry";
import { useInpaintForm } from "./useInpaintForm";

/**
 * 台帳 §1-124 弾 2: Inpainting パネルの H3 の描き分け。
 *
 * H3（`constraints.inpaint` あり）では、のりしろ 2 欄（Stage-1／Stage-2）と「+18px」の
 * 注意文の代わりに、Outpainting の H3 と同じマスクブラー（なじませ段数）1 本・音声の
 * Strength・サンプリング欄・24fps 固定の注記が出る。LTX では従来どおり。
 */

// シード欄の ♻ が `useJobsContext` を読む（`OutpaintingPanel.h3.test.tsx` と同じ差し替え）。
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({
    jobs: [],
    serverBusy: false,
    cancellingIds: new Set<string>(),
    deletingIds: new Set<string>(),
    cancelJob: () => Promise.resolve(),
    deleteJob: () => Promise.resolve(),
  }),
}));

const H3: GenerationConstraints = resolveGenerationConstraints({
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
} satisfies BaseModelConstraints);

const FILE = "C:\\videos\\shot.mp4";

function makeItem(): SelectionItem {
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
  };
}

function makeSelection(item: SelectionItem): TimelineSelection {
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

function createBridge(): NativeBridge {
  const request = vi.fn(async (method: string): Promise<unknown> => {
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
    if (method === "backend.uploadFile") return { status: 200, body: { video_id: "vid-1", trimmed: true } };
    throw new Error(`unexpected bridge call: ${method}`);
  });
  return { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
}

function Harness({ bridge, constraints }: { bridge: NativeBridge; constraints: GenerationConstraints | undefined }) {
  const form = useInpaintForm({ nativeBridge: bridge, ...(constraints ? { constraints } : {}) });
  return <InpaintingPanel form={form} disabled={false} />;
}

async function renderPanel(constraints: GenerationConstraints | undefined) {
  act(() => {
    const item = makeItem();
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    publishInpaintTarget({ item, selection: makeSelection(item) });
  });
  render(
    <LanguageProvider>
      <Harness bridge={createBridge()} constraints={constraints} />
    </LanguageProvider>,
  );
  // 対象動画のアップロードが決着するまで待つ（act の警告を残さない）。
  await waitFor(() => expect(screen.queryByRole("status", { name: en.edit.inpainting.uploading })).toBeNull());
}

const t = en.edit.inpainting;
const tOut = en.edit.outpainting;

beforeEach(() => {
  resetInpaintSlots();
});
afterEach(() => {
  resetInpaintSlots();
});

describe("InpaintingPanel（H3）", () => {
  it("なじませ 1 本・音声の Strength・サンプリング・24fps の注記が出て、のりしろ 2 欄と「+18px」の注意文は出ない", async () => {
    await renderPanel(H3);
    const blur = screen.getByRole("slider", { name: tOut.blurLabel }) as HTMLInputElement;
    expect(blur.value).toBe("1");
    expect(blur.max).toBe("15");
    // 1 段 × 長辺 1920 ÷ 64 = 30px（定数項 18 は付かない）。
    expect(screen.getByText(t.h3BlendNote(featherWidthPx(1, 1920)))).toBeTruthy();
    expect(screen.getByText(tOut.audioStrengthLabel)).toBeTruthy();
    expect(screen.getByText(tOut.audioStrengthHint)).toBeTruthy();
    const strength = document.querySelector<HTMLInputElement>('input[type="range"][max="1"]');
    expect(strength?.value).toBe("0.5");
    expect(screen.getByText(en.single.sampling.steps)).toBeTruthy();
    expect(screen.getByText(t.h3FixedFpsNote(24))).toBeTruthy();

    expect(screen.queryByText(t.blend.stage1Label)).toBeNull();
    expect(screen.queryByText(t.blend.stage2Label)).toBeNull();
    expect(screen.queryByText(t.blend.heading)).toBeNull();
    expect(screen.queryByText(t.blend.note(featherWidthPx(2, 1920) + 18))).toBeNull();
  });

  it("フレーム数のスライダーは 17 刻み・[22, 345]", async () => {
    await renderPanel(H3);
    const frames = document.querySelector<HTMLInputElement>('.inpaint-panel input[type="range"][min="22"]');
    expect(frames?.step).toBe("17");
    expect(frames?.max).toBe("345");
    expect(frames?.value).toBe("209");
  });
});

describe("InpaintingPanel（LTX）", () => {
  it("のりしろ 2 欄と「+18px」の注意文が出て、H3 の欄は出ない", async () => {
    await renderPanel(undefined);
    expect(screen.getByText(t.blend.heading)).toBeTruthy();
    expect(screen.getByText(t.blend.stage1Label)).toBeTruthy();
    expect(screen.getByText(t.blend.stage2Label)).toBeTruthy();
    // Stage-2 の既定 2 段 × 1920 ÷ 64 + 18 = 78px（キャンバス 1920×1152 の長辺）。
    expect(screen.getByText(t.blend.note(featherWidthPx(2, 1920) + 18))).toBeTruthy();

    expect(screen.queryByRole("slider", { name: tOut.blurLabel })).toBeNull();
    expect(screen.queryByText(tOut.audioStrengthLabel)).toBeNull();
    expect(screen.queryByText(en.single.sampling.steps)).toBeNull();
    expect(screen.queryByText(t.h3FixedFpsNote(24))).toBeNull();
    const frames = document.querySelector<HTMLInputElement>('.inpaint-panel input[type="range"][min="9"]');
    expect(frames?.step).toBe("8");
  });
});
