import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { ToastProvider } from "../../shell/ToastContext";
import { Toasts } from "../../shell/Toasts";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import type { SelectionItem } from "../../timeline/menuSelection";
import { EditScreen } from "./EditScreen";

/**
 * Alpha Gen 第 2 弾の通し（画面側）: サブタブの並び・灰色・右クリックでの着地・
 * 表示（fps と開始位置）・Generate で `POST /generate/alpha` に届くまで。
 * フォームの規則そのものは `useAlphaGenForm.test.tsx` が持つ。
 */

const jobsStub = vi.hoisted(() => ({ serverBusy: false }));

vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({
    jobs: [],
    serverBusy: jobsStub.serverBusy,
    cancellingIds: new Set<string>(),
    deletingIds: new Set<string>(),
    cancelJob: () => Promise.resolve(),
    deleteJob: () => Promise.resolve(),
  }),
}));

const t = en.edit.alphaGen;
const FILE = "C:\\videos\\dance.mp4";

function makeItem(overrides: Partial<SelectionItem> = {}): SelectionItem {
  return {
    layer: 2,
    frameStart: 0,
    frameEnd: 299,
    effectName: "動画ファイル",
    filePath: FILE,
    objectName: "dance",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 10,
    hasPlaybackRange: true,
    playbackStartSec: 2.5,
    playbackEndSec: 10,
    ...overrides,
  };
}

function alphaGenIntent(item: SelectionItem = makeItem()): GenerationPrefill {
  return {
    intent: "alpha-gen",
    targetMode: "edit",
    selection: {
      hasRange: false,
      rangeStart: 0,
      rangeEnd: 0,
      selected: [item],
      cursorFrame: 0,
      cursorLayer: 0,
      rate: 30,
      scale: 1,
      sampleRate: 44100,
    },
  };
}

const MEDIA: MockBridgeOptions = {
  probeMediaInfoDurationSec: 10,
  probeMediaInfoWidth: 1920,
  probeMediaInfoHeight: 1080,
  probeMediaInfoFps: 29.97,
};

function renderEdit(
  options: {
    intent?: GenerationPrefill;
    bridge?: ReturnType<typeof createMockBridge>;
    subTabsDisabled?: { retake: boolean; outpainting: boolean; inpainting: boolean; alphaGen: boolean };
  } = {},
) {
  const bridge = options.bridge ?? createMockBridge({ delayMs: 0, ...MEDIA });
  const onJobSubmitted = vi.fn();
  render(
    <LanguageProvider>
      <ToastProvider>
        <EditScreen
          {...(options.intent ? { initialIntent: options.intent } : {})}
          nativeBridge={bridge}
          onJobSubmitted={onJobSubmitted}
          {...(options.subTabsDisabled ? { subTabsDisabled: options.subTabsDisabled } : {})}
        />
        <Toasts />
      </ToastProvider>
    </LanguageProvider>,
  );
  return { bridge, onJobSubmitted };
}

const alphaTab = () => screen.getByRole("tab", { name: "Alpha Gen" });
const generateButton = () => screen.getByRole("button", { name: t.generateButton });

describe("EditScreen — Alpha Gen サブタブ", () => {
  it("末尾に並び、手動で開ける（入力動画が無いうちは押せない理由が出る）", async () => {
    const user = userEvent.setup();
    renderEdit();
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Retake",
      "Outpainting",
      "Inpainting",
      "Alpha Gen",
    ]);
    await user.click(alphaTab());
    expect(alphaTab()).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: t.heading })).toBeVisible();
    expect(generateButton()).toBeDisabled();
    expect(screen.getByText(t.reasons.sourceMissing)).toBeInTheDocument();
  });

  it("灰色（LTX 2.3）なら押せず、LTX 2.5 専用である旨をツールチップに出す", () => {
    renderEdit({ subTabsDisabled: { retake: false, outpainting: false, inpainting: false, alphaGen: true } });
    expect(alphaTab()).toBeDisabled();
    expect(alphaTab()).toHaveAttribute("title", en.edit.unavailableOnBaseModel.alphaGen);
    expect(en.edit.unavailableOnBaseModel.alphaGen).toMatch(/LTX 2\.5/);
  });

  it("灰色のときに右クリックが届いても Alpha Gen へは入らない", () => {
    renderEdit({
      intent: alphaGenIntent(),
      subTabsDisabled: { retake: false, outpainting: false, inpainting: false, alphaGen: true },
    });
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
  });

  it("Alpha Gen 表示中に alpha_gen が灰色になる（2.3 へ切り替え）とサブタブが Retake へ移る", () => {
    const bridge = createMockBridge({ delayMs: 0, ...MEDIA });
    const live = { retake: false, outpainting: false, inpainting: false, alphaGen: false };
    const tree = (disabled: typeof live) => (
      <LanguageProvider>
        <ToastProvider>
          <EditScreen initialIntent={alphaGenIntent()} nativeBridge={bridge} subTabsDisabled={disabled} />
        </ToastProvider>
      </LanguageProvider>
    );
    const { rerender } = render(tree(live));
    expect(alphaTab()).toHaveAttribute("aria-selected", "true");
    rerender(tree({ ...live, alphaGen: true }));
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: t.heading, hidden: true })).not.toBeVisible();
    expect(screen.queryByRole("button", { name: t.generateButton })).not.toBeInTheDocument();
  });

  it("右クリックで着地し、ファイル名・fps（生値）・再生位置からの開始を出す", async () => {
    renderEdit({ intent: alphaGenIntent() });
    expect(alphaTab()).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("dance.mp4")).toBeInTheDocument();
    expect(await screen.findByText(`${t.fpsLabel}: 29.97`)).toBeInTheDocument();
    expect(screen.getByText(t.startFromPlayback("2.5"))).toBeInTheDocument();
    // 軽量モードのチェックは押せる（警告は出しても無効化しない）。
    expect(screen.getByRole("checkbox", { name: t.lightModeLabel })).not.toBeDisabled();
  });

  it("線の下の説明は軽量モードのチェックで切り替わる（OFF は目安・ON は縮小）", async () => {
    const user = userEvent.setup();
    // LTX 2.5・4bit の行（組み込みの既定）: 1920x1080 → 原寸の予算で 81 コマの線。
    const bridge = createMockBridge({ delayMs: 0, ...MEDIA });
    render(
      <LanguageProvider>
        <ToastProvider>
          <EditScreen initialIntent={alphaGenIntent()} nativeBridge={bridge} engineFamily="ltx25" weightClass="4bit" />
        </ToastProvider>
      </LanguageProvider>,
    );
    expect(await screen.findByText(t.comfortHintFull)).toBeInTheDocument();
    expect(screen.queryByText(t.comfortHintLight)).not.toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: t.lightModeLabel }));
    expect(screen.getByText(t.comfortHintLight)).toBeInTheDocument();
    expect(screen.queryByText(t.comfortHintFull)).not.toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: t.lightModeLabel }));
    expect(screen.getByText(t.comfortHintFull)).toBeInTheDocument();
  });

  it("fps が読めない素材は「不明」と出す", async () => {
    renderEdit({
      intent: alphaGenIntent(),
      bridge: createMockBridge({ delayMs: 0, ...MEDIA, probeMediaInfoFps: 0 }),
    });
    expect(await screen.findByText(`${t.fpsLabel}: ${t.fpsUnknown}`)).toBeInTheDocument();
  });

  it("Generate で窓を上げて POST /generate/alpha に届く（LTX 2.5）", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, ltx25Install: "full", ...MEDIA });
    const loaded = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/pipeline/load",
      body: { base_model: "LTX25" },
    });
    expect(loaded.status).toBe(200);
    const spy = vi.spyOn(bridge, "request");
    const { onJobSubmitted } = renderEdit({ intent: alphaGenIntent(), bridge });
    await screen.findByText(`${t.fpsLabel}: 29.97`);

    await user.click(generateButton());
    await waitFor(() => expect(onJobSubmitted).toHaveBeenCalledTimes(1));

    const upload = spy.mock.calls.find(([m]) => m === "backend.uploadFile");
    expect(upload?.[1]).toMatchObject({
      kind: "video",
      filePath: FILE,
      query: { trim_start_sec: "2.500", max_frames: "82" },
    });
    const post = spy.mock.calls.find(
      ([m, p]) => m === "backend.request" && (p as { path: string }).path === "/api/v1/generate/alpha",
    );
    expect(post).toBeDefined();
    expect((post![1] as { body: Record<string, unknown> }).body).toEqual({
      reference_video_id: expect.any(String),
      // The mock answers `trimmed: true` for a trim query, so the window starts
      // at the head of the uploaded file.
      window_start_sec: 0,
      num_frames: 81,
      seed: -1,
      light_mode: false,
    });
  });

  it("LTX 2.3 のサーバーが断ったら（422）エラーカードに出す", async () => {
    const user = userEvent.setup();
    renderEdit({ intent: alphaGenIntent() });
    await screen.findByText(`${t.fpsLabel}: 29.97`);
    await user.click(generateButton());
    expect(await screen.findByText("FEATURE_UNSUPPORTED")).toBeInTheDocument();
  });
});
