import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { BaseModelConstraints } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { en } from "../../i18n/strings";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { ToastProvider } from "../../shell/ToastContext";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { H3_NO_CONDITIONING, h3ComfortFrames, h3SequenceRows } from "../../shell/h3SequenceRows";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { EditScreen } from "./EditScreen";

/**
 * §1-125 MiniMax H3 の Outpainting（台帳 `Docs/PENDING_TASKS.md` §1-125）。
 *
 * 配信の `constraints.outpaint` があるとき、Outpainting 画面は IC-LoRA を使わない
 * H3 の形になる: キャンバスの格子は 32、尺は 17n+5、fps は固定 24、音声の Strength と
 * サンプリングのスライダーが出て、要求は `outpaint.source_video_id`＋`audio_strength`＋
 * `blend_dilation_stage1`（常に送る）で、`reference_video_id` とピン留めの LoRA は
 * 送らない。LTX の形は `OutpaintingPanel.test.tsx` が従来どおり固定している。
 */

// The seed field's ♻ and the ledger read `useJobsContext`; the same stub as
// `OutpaintingPanel.test.tsx`.
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
  outpaint: {
    audio_strength: { min: 0, max: 1, step: 0.05, default: 0.5 },
    blend_dilation_stage1: { min: 0, max: 15, step: 1, default: 1 },
  },
};
const H3 = resolveGenerationConstraints(H3_BLOCK);

const OUTPAINT_TAB_INTENT: GenerationPrefill = {
  intent: "outpaint",
  targetMode: "edit",
  selection: {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [],
    cursorFrame: 0,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  },
};

interface BridgeOptions {
  media?: { durationSec: number; width: number; height: number };
  /** `GET /loras` rows. Defaults to NONE — H3 does not need `in-outpainting`. */
  loras?: Array<{ name: string; kind: "style" | "control" }>;
  generateResponse?: { status: number; body: unknown };
}

function createBridge(options: BridgeOptions = {}) {
  const media = options.media ?? { durationSec: 8, width: 1280, height: 720 };
  const loras = options.loras ?? [];
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "ui.pickFile") return { filePath: "C:\\videos\\clip.mp4", fileName: "clip.mp4" };
    if (method === "backend.uploadFile") return { status: 200, body: { video_id: "vid-1" } };
    if (method === "fs.probeMediaInfo") return media;
    if (method === "backend.request") {
      const { method: httpMethod, path } = params as { method: string; path: string };
      if (httpMethod === "GET" && path.endsWith("/loras")) {
        return {
          status: 200,
          body: { loras: loras.map((l) => ({ ...l, has_thumbnail: false, exists: true, source: "config" })) },
        };
      }
      if (httpMethod === "POST" && path.endsWith("/generate")) {
        return (
          options.generateResponse ?? {
            status: 202,
            body: { job_id: "job-1", status: "queued", created_at: "2026-11-10T00:00:00Z" },
          }
        );
      }
    }
    throw new Error(`unexpected bridge call: ${method}`);
  });
  const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
  return { bridge, request };
}

function screenTree(
  bridge: NativeBridge,
  constraints: GenerationConstraints | undefined,
  prompt = "a wide street",
  engineFamily = "h3",
) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <div role="tabpanel">
          <EditScreen
            nativeBridge={bridge}
            initialIntent={OUTPAINT_TAB_INTENT}
            prompt={prompt}
            onJobSubmitted={vi.fn()}
            engineFamily={engineFamily}
            {...(constraints ? { constraints } : {})}
            loraAudioHidden
          />
        </div>
      </ToastProvider>
    </LanguageProvider>
  );
}

function renderH3(options: BridgeOptions = {}, prompt?: string) {
  const { bridge, request } = createBridge(options);
  const view = render(screenTree(bridge, H3, prompt));
  return { bridge, request, view, panel: within(screen.getByRole("tabpanel")) };
}

type Panel = ReturnType<typeof renderH3>["panel"];

const outpaintPanel = () => document.querySelector<HTMLElement>(".outpaint-panel")!;

function setPads(panel: Panel, pads: Partial<Record<"left" | "right" | "top" | "bottom", number>>) {
  for (const [side, value] of Object.entries(pads)) {
    fireEvent.change(panel.getByRole("slider", { name: new RegExp(`^${side}$`, "i") }), {
      target: { value: String(value) },
    });
  }
}

async function attach(panel: Panel, user: ReturnType<typeof userEvent.setup>) {
  await user.click(panel.getByRole("button", { name: /choose source video/i }));
  await waitFor(() => expect(panel.getByRole("slider", { name: /^left$/i })).toBeInTheDocument());
}

function generateBody(request: ReturnType<typeof vi.fn>): Record<string, unknown> | undefined {
  const call = request.mock.calls.find(
    ([method, params]) =>
      method === "backend.request" &&
      (params as { method: string; path: string }).method === "POST" &&
      (params as { path: string }).path.endsWith("/generate"),
  ) as [string, { body: Record<string, unknown> }] | undefined;
  return call?.[1].body;
}

async function pressGenerate(panel: Panel, user: ReturnType<typeof userEvent.setup>) {
  const button = panel.getByRole("button", { name: /^generate$/i });
  await waitFor(() => expect(button).toBeEnabled());
  await user.click(button);
}

describe("EditScreen — Outpainting（MiniMax H3・§1-125）", () => {
  it("Strength・サンプリング・固定 fps が出て、尺は 17n+5、マスクブラーは配信の既定 1、格子の説明は 32", async () => {
    const user = userEvent.setup();
    const { panel } = renderH3();
    const p = within(outpaintPanel());
    expect(p.getByText(en.edit.outpainting.audioStrengthLabel)).toBeTruthy();
    expect(p.getByText(en.edit.outpainting.audioStrengthHint)).toBeTruthy();
    const strength = outpaintPanel().querySelector<HTMLInputElement>('input[type="range"][max="1"]');
    expect(strength?.value).toBe("0.5");
    expect(strength?.step).toBe("0.05");
    expect(p.getByText(en.single.sampling.steps)).toBeTruthy();
    const fps = p.getByRole("spinbutton", { name: en.single.duration.fps }) as HTMLInputElement;
    expect(fps).toBeDisabled();
    expect(fps.value).toBe("24");
    const duration = p.getByRole("slider", { name: /duration/i });
    expect(duration).toHaveAttribute("step", "17");
    expect(duration).toHaveAttribute("min", "22");

    await attach(panel, user);
    expect(p.getByText(en.edit.outpainting.gridNote(32))).toBeTruthy();
    expect((p.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("1");
    // 緑の助言の無い H3 の説明（1280 の長辺・1 段 = 20px）。
    expect(p.getByText(en.edit.outpainting.featherNote(20))).toBeTruthy();
    expect(p.queryByText(/green of the mask/i)).toBeNull();
  });

  it("ピン留めの LoRA が無くても止めない（`loraMissing` は H3 では出ない）", async () => {
    const user = userEvent.setup();
    const { panel } = renderH3({ loras: [] });
    await attach(panel, user);
    setPads(panel, { left: 32, right: 32, top: 24, bottom: 24 });
    await waitFor(() => expect(panel.getByRole("button", { name: /^generate$/i })).toBeEnabled());
    expect(screen.queryByText(/control adapter is not installed/i)).toBeNull();
  });

  it("格子の理由は 32 を名指す（幅と高さで別の行）", async () => {
    const user = userEvent.setup();
    const { panel } = renderH3();
    await attach(panel, user);
    setPads(panel, { left: 1, top: 1 });
    expect(screen.getByText(en.edit.outpainting.generateReasons.canvasWidthOffGrid(32))).toBeTruthy();
    expect(screen.getByText(en.edit.outpainting.generateReasons.canvasHeightOffGrid(32))).toBeTruthy();
    expect(screen.getByText("The width must be a multiple of 32.")).toBeTruthy();
    // 32 に乗れば消える（LTX の 128 には乗っていない 1312 でも通る）。
    setPads(panel, { left: 32, top: 48 });
    expect(screen.queryByText("The width must be a multiple of 32.")).toBeNull();
    expect(screen.queryByText("The height must be a multiple of 32.")).toBeNull();
  });

  it("Generate で H3 の形の body が飛ぶ（source_video_id・audio_strength・stage1 あり、参照動画・ピン留め・stage2・freeze なし）", async () => {
    const user = userEvent.setup();
    const { panel, request } = renderH3({}, "a wide street <lora:Pixar_Toon:0.8:0.3>");
    await attach(panel, user);
    setPads(panel, { left: 12, right: 20, bottom: 48 });
    await pressGenerate(panel, user);

    await waitFor(() => expect(generateBody(request)).toBeDefined());
    const body = generateBody(request)!;
    expect(body).toMatchObject({
      prompt: "a wide street",
      width: 1312,
      height: 768,
      // 8 秒 × 24fps = 192 枚（17n+5 に乗る）。既定の 345 が素材の長さで頭打ち。
      num_frames: 192,
      frame_rate: 24,
      num_inference_steps: 20,
      shift_video: 12,
      shift_audio: 3,
    });
    expect(body.outpaint).toEqual({
      pad_left: 12,
      pad_right: 20,
      pad_top: 0,
      pad_bottom: 48,
      source_video_id: "vid-1",
      audio_strength: 0.5,
      blend_dilation_stage1: 1,
    });
    // LoRA はプロンプトのタグから（音声の強度は落とす）。ピン留めの in-outpainting は無い。
    expect(body.loras).toEqual([{ name: "Pixar_Toon", strength: 0.8 }]);
    expect(body).not.toHaveProperty("reference_video_id");
  });

  it("マスクブラーが 5 でも `blend_dilation_stage1` を省かない（LTX の「既定なら省く」は H3 に無い）", async () => {
    const user = userEvent.setup();
    const { panel, request } = renderH3();
    await attach(panel, user);
    setPads(panel, { left: 32, right: 32, top: 24, bottom: 24 });
    fireEvent.change(panel.getByRole("slider", { name: /mask blur/i }), { target: { value: "5" } });
    fireEvent.change(outpaintPanel().querySelector<HTMLInputElement>('input[type="range"][max="1"]')!, {
      target: { value: "0.8" },
    });
    await pressGenerate(panel, user);

    await waitFor(() => expect(generateBody(request)).toBeDefined());
    const outpaint = generateBody(request)!.outpaint as Record<string, unknown>;
    expect(outpaint.blend_dilation_stage1).toBe(5);
    expect(outpaint.audio_strength).toBe(0.8);
    expect(outpaint).not.toHaveProperty("blend_dilation_stage2");
    expect(outpaint).not.toHaveProperty("freeze_source_audio");
  });

  it("配信に追従する: LTX→H3 で既定が入り、H3 の配信が変わると触った値も新しい既定へ戻る", async () => {
    const user = userEvent.setup();
    const { bridge } = createBridge();
    const { rerender } = render(screenTree(bridge, undefined));
    const panel = within(screen.getByRole("tabpanel"));
    await attach(panel, user);
    // LTX の形: マスクブラーは 5、Strength は無い。
    expect((panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("5");
    expect(within(outpaintPanel()).queryByText(en.edit.outpainting.audioStrengthLabel)).toBeNull();

    rerender(screenTree(bridge, H3));
    await waitFor(() =>
      expect((panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("1"),
    );
    const strength = () => outpaintPanel().querySelector<HTMLInputElement>('input[type="range"][max="1"]')!;
    expect(strength().value).toBe("0.5");

    fireEvent.change(panel.getByRole("slider", { name: /mask blur/i }), { target: { value: "7" } });
    fireEvent.change(strength(), { target: { value: "0.9" } });
    expect((panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("7");

    const changed = resolveGenerationConstraints({
      ...H3_BLOCK,
      outpaint: {
        audio_strength: { min: 0, max: 1, step: 0.05, default: 0.25 },
        blend_dilation_stage1: { min: 0, max: 15, step: 1, default: 3 },
      },
    });
    rerender(screenTree(bridge, changed));
    await waitFor(() =>
      expect((panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("3"),
    );
    expect(strength().value).toBe("0.25");

    // H3→LTX で LTX の既定 5 に戻り、Strength は消える。
    rerender(screenTree(bridge, undefined));
    await waitFor(() =>
      expect((panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement).value).toBe("5"),
    );
    expect(within(outpaintPanel()).queryByText(en.edit.outpainting.audioStrengthLabel)).toBeNull();
  });

  it("422 OUTPAINT_VIDEO_NO_AUDIO では赤いカードの代わりに黄色の注意を出す", async () => {
    const user = userEvent.setup();
    const { panel } = renderH3({
      generateResponse: {
        status: 422,
        body: { error: { code: "OUTPAINT_VIDEO_NO_AUDIO", message: "no audio", detail: "…" } },
      },
    });
    await attach(panel, user);
    setPads(panel, { left: 32, right: 32, top: 24, bottom: 24 });
    await pressGenerate(panel, user);

    const note = await screen.findByText(en.edit.outpainting.noAudioStrengthWarning);
    expect(note).toHaveClass("warning-banner", "warning-banner-mild");
    expect(document.querySelector(".error-code")).toBeNull();
  });

  it("ほかの 422（奇数寸法の素材）は理由のコードを赤いカードで見せる", async () => {
    const user = userEvent.setup();
    const { panel } = renderH3({
      generateResponse: {
        status: 422,
        body: { error: { code: "OUTPAINT_SOURCE_ODD_SIZE", message: "source width and height must be even" } },
      },
    });
    await attach(panel, user);
    setPads(panel, { left: 32, right: 32, top: 24, bottom: 24 });
    await pressGenerate(panel, user);

    await waitFor(() => expect(document.querySelector(".error-code")?.textContent).toBe("OUTPAINT_SOURCE_ODD_SIZE"));
    expect(screen.queryByText(en.edit.outpainting.noAudioStrengthWarning)).toBeNull();
  });

  it("快適上限は総行数の線で目盛りと警告を出し、`spill_free_frames` の目盛りと「2〜4 倍」の文言は出さない", async () => {
    const user = userEvent.setup();
    // 2560×1440・30 秒: 尺の上限は 481。既定 345 は線の内側、481 は外側。
    const { panel } = renderH3({ media: { durationSec: 30, width: 2560, height: 1440 } });
    await attach(panel, user);
    const budget = FALLBACK_APP_CONFIG.limits.comfort_budgets?.h3?.single_rows_budget as number;
    const shape = { width: 2560, height: 1440, ...H3_NO_CONDITIONING };
    const expected = h3ComfortFrames(shape, budget, 22, 481);
    expect(expected).not.toBeNull();

    const duration = panel.getByRole("slider", { name: /duration/i });
    expect(duration).toHaveAttribute("list", "outpaint-spill-tick");
    expect(document.getElementById("outpaint-spill-tick")?.querySelector("option")?.getAttribute("value")).toBe(
      String(expected),
    );

    const over = (frames: number) => (h3SequenceRows(shape, frames)?.total ?? 0) > budget;
    fireEvent.change(duration, { target: { value: "481" } });
    expect(over(481)).toBe(true);
    const rows481 = h3SequenceRows(shape, 481)!.total;
    expect(panel.getByText(en.edit.outpainting.comfortWarning(rows481))).toBeTruthy();
    expect(panel.queryByText(/may slow down/i)).toBeNull();

    fireEvent.change(duration, { target: { value: String(expected) } });
    expect(panel.queryByText(/units of work/i)).toBeNull();
  });
});
