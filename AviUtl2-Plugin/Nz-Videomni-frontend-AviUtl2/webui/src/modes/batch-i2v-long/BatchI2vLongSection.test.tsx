import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockBridge } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { __resetRunLockForTests } from "../../shell/runLock";
import { BatchI2vLongSection } from "./BatchI2vLongSection";
import type { ChainSnapshotSource } from "./chainSnapshot";
import { __resetI2vLongRuntimeForTests } from "./runtime";

const IMG_DIR = "C:\\shots\\ep01";

interface ChainFakeState {
  prompt?: string;
  seed?: number;
  mode?: "scratch" | "v2v";
  clips?: Array<{ prompt: string; numFrames: number }>;
  isValid?: boolean;
  validityReasons?: string[];
  /** §1-16 長尺A2V. */
  hasSourceAudio?: boolean;
}

function makeChain(state: ChainFakeState = {}): ChainSnapshotSource {
  const prompt = state.prompt ?? "a cat walking";
  const seed = state.seed ?? -1;
  const mode = state.mode ?? "scratch";
  const clips = state.clips ?? [
    { prompt: "", numFrames: 121 },
    { prompt: "", numFrames: 121 },
  ];
  const validityReasons = state.validityReasons ?? [];
  return {
    buildRequest: (): GenerateChainRequest => ({
      prompt,
      width: 768,
      height: 512,
      frame_rate: 24,
      seed,
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: clips.map((c) => ({ ...(c.prompt ? { prompt: c.prompt } : {}), num_frames: c.numFrames })),
      ...(mode === "v2v" ? { source_video: { video_id: "v", context_frames: 41 } } : {}),
      ...(state.hasSourceAudio ? { source_audio: { audio_id: "a" } } : {}),
      chunked_upsample: true,
    }),
    mode,
    hasSourceAudio: state.hasSourceAudio ?? false,
    clips: clips.map((c) => ({ prompt: c.prompt })),
    isValid: state.isValid ?? validityReasons.length === 0,
    validityReasons,
    outputFrames: 239,
    outputSeconds: 9.96,
    minClips: 2,
    minFramesForOverlap: 25,
  };
}

function renderSection(
  bridge: NativeBridge,
  opts: { chain?: ChainSnapshotSource; serverBusy?: boolean } = {},
) {
  return render(
    <LanguageProvider>
      <BatchI2vLongSection
        config={FALLBACK_APP_CONFIG}
        chain={opts.chain ?? makeChain()}
        serverBusy={opts.serverBusy ?? false}
        nativeBridge={bridge}
      />
    </LanguageProvider>,
  );
}

/** Opens the collapsed `<details>` so its body is queryable. jsdom does not
 * implement the summary-click toggle, so the `open` attribute is set directly
 * (the "closed by default" assertion below is what actually pins the default). */
function open(container: HTMLElement): HTMLElement {
  const details = container.querySelector("details.batch-section") as HTMLDetailsElement;
  act(() => {
    details.open = true;
  });
  return details;
}

function imgFolder(names: string[]) {
  return createMockFs({
    folders: { [IMG_DIR]: names.map((name) => ({ name, sizeBytes: 1024, mtimeMs: 0 })) },
  });
}

/** Wraps a mock bridge so every `POST /generate/chain` answers 500 — a started
 * run then fails its row immediately instead of polling a job for seconds. */
function failChainSubmit(base: NativeBridge): NativeBridge {
  return {
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.request") {
        const p = params as { method?: string; path?: string };
        if (p.method === "POST" && p.path === "/api/v1/generate/chain") {
          return { status: 500, body: { error: { code: "MOCK_STOP", message: "stopped by test" } } } as unknown as ResultOf<M>;
        }
      }
      return base.request(method, params);
    },
    requestWithFiles: (m, p, f) => base.requestWithFiles(m, p, f),
    on: (e, h) => base.on(e, h),
  };
}

describe("BatchI2vLongSection", () => {
  beforeEach(() => {
    __resetI2vLongRuntimeForTests();
    __resetRunLockForTests();
  });

  it("既定で折りたたまれている", () => {
    const { container } = renderSection(createMockBridge({ delayMs: 0 }));
    const details = container.querySelector("details.batch-section") as HTMLDetailsElement;
    expect(details).not.toBeNull();
    expect(details.open).toBe(false);
    expect(screen.getByText(en.batchI2vLong.heading)).toBeInTheDocument();
  });

  // 2026-07-30 オーナー実機フィードバック: 長い説明文（notice / chainSettingsNote）は
  // 撤去し、設定の実効値が見える1行サマリだけを残す。
  it("1枚あたりの尺サマリだけを出す（長い説明文は無い）", () => {
    const { container } = renderSection(createMockBridge({ delayMs: 0 }));
    open(container);

    expect(screen.getByText(en.batchI2vLong.chainSummary(2, 239, 9.96))).toBeInTheDocument();
    expect(container.querySelector(".batch-notice")).toBeNull();
  });

  it("ブロック理由をバッチ側・Chain側とも1行ずつ個別に展開する", () => {
    const chain = makeChain({ isValid: false, validityReasons: ["dimensionsOffGrid", "clipFramesOffGrid"] });
    const { container } = renderSection(createMockBridge({ delayMs: 0 }), { chain, serverBusy: true });
    open(container);

    const note = screen.getByRole("note");
    // バッチ自身の理由。
    expect(within(note).getByText(en.batchI2vLong.blockReasons.imgDirMissing)).toBeInTheDocument();
    expect(within(note).getByText(en.batchI2vLong.blockReasons.outDirMissing)).toBeInTheDocument();
    expect(within(note).getByText(en.batchI2vLong.blockReasons.noRows)).toBeInTheDocument();
    expect(within(note).getByText(en.batchI2vLong.blockReasons.jobActive)).toBeInTheDocument();
    // Chain側の理由も、Chain画面と同じ文言のまま個別に出る。
    expect(within(note).getByText(en.single.generateReasons.dimensionsOffGrid)).toBeInTheDocument();
    expect(within(note).getByText(en.chained.generateReasons.clipFramesOffGrid)).toBeInTheDocument();

    expect(screen.getByRole("button", { name: en.batchI2vLong.startButton })).toBeDisabled();
  });

  it("元動画が付いているときはsourceVideoAttachedを出し、Chain側のsource系理由は重ねない", () => {
    const chain = makeChain({ mode: "v2v", isValid: false, validityReasons: ["sourceNotReady"] });
    const { container } = renderSection(createMockBridge({ delayMs: 0 }), { chain });
    open(container);

    const note = screen.getByRole("note");
    expect(within(note).getByText(en.batchI2vLong.blockReasons.sourceVideoAttached)).toBeInTheDocument();
    expect(within(note).queryByText(en.chained.sourceVideo.contextFramesError)).toBeNull();
  });

  it("音声が付いているときはsourceAudioAttachedを出し、Chain側のaudio系理由は重ねない", () => {
    const chain = makeChain({
      hasSourceAudio: true,
      isValid: false,
      validityReasons: ["audioNotReady", "chainLayoutInvalid"],
    });
    const { container } = renderSection(createMockBridge({ delayMs: 0 }), { chain });
    open(container);

    const note = screen.getByRole("note");
    expect(within(note).getByText(en.batchI2vLong.blockReasons.sourceAudioAttached)).toBeInTheDocument();
    expect(within(note).queryByText(en.chained.generateReasons.audioNotReady)).toBeNull();
    expect(within(note).queryByText(en.chained.generateReasons.chainLayoutInvalid)).toBeNull();
  });

  it("clip0に個別プロンプトがあるときだけ強警告を出す", () => {
    const withOverride = makeChain({
      clips: [
        { prompt: "clip0 override", numFrames: 121 },
        { prompt: "", numFrames: 121 },
      ],
    });
    const { container, unmount } = renderSection(createMockBridge({ delayMs: 0 }), { chain: withOverride });
    open(container);
    expect(screen.getByText(en.batchI2vLong.notes.clip0PromptOverride)).toBeInTheDocument();
    expect(screen.queryByText(en.batchI2vLong.notes.clipPromptOverride)).toBeNull();
    unmount();

    const laterOnly = makeChain({
      clips: [
        { prompt: "", numFrames: 121 },
        { prompt: "clip1 override", numFrames: 121 },
      ],
    });
    const second = renderSection(createMockBridge({ delayMs: 0 }), { chain: laterOnly });
    open(second.container);
    expect(screen.queryByText(en.batchI2vLong.notes.clip0PromptOverride)).toBeNull();
    expect(screen.getByText(en.batchI2vLong.notes.clipPromptOverride)).toBeInTheDocument();
  });

  it("シード固定の注意はseed>=0のときだけ出る", () => {
    const { container, unmount } = renderSection(createMockBridge({ delayMs: 0 }), { chain: makeChain({ seed: -1 }) });
    open(container);
    expect(screen.queryByText(en.batchI2vLong.notes.seedFixed)).toBeNull();
    unmount();

    const second = renderSection(createMockBridge({ delayMs: 0 }), { chain: makeChain({ seed: 7 }) });
    open(second.container);
    expect(screen.getByText(en.batchI2vLong.notes.seedFixed)).toBeInTheDocument();
  });

  it("add/replaceラジオを切り替えられる", async () => {
    const user = userEvent.setup();
    const { container } = renderSection(createMockBridge({ delayMs: 0 }));
    open(container);

    const add = screen.getByRole("radio", { name: en.batchI2vLong.promptMode.add });
    const replace = screen.getByRole("radio", { name: en.batchI2vLong.promptMode.replace });
    expect(add).toBeChecked();

    await user.click(replace);
    expect(replace).toBeChecked();
    expect(add).not.toBeChecked();
  });

  // 2026-07-30: バッチ全体のプロンプト欄は撤去（行ごとの欄へ移動）。
  it("バッチ全体のプロンプト用textareaは存在しない", () => {
    const { container } = renderSection(createMockBridge({ delayMs: 0 }));
    open(container);

    expect(container.querySelector("textarea")).toBeNull();
  });

  it("スキャンした行にはプロンプト入力欄（2000字上限）と📝が出る", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({
      delayMs: 0,
      fs: imgFolder(["a.png", "b.png"]),
      pickFolderPath: IMG_DIR,
    });
    const { container } = renderSection(bridge);
    open(container);

    await user.click(screen.getByRole("button", { name: en.batchI2vLong.imgDir.button }));
    await user.click(screen.getByRole("button", { name: en.batchI2vLong.scanButton }));
    await waitFor(() => {
      expect(screen.getByText("a.png")).toBeInTheDocument();
    });

    const inputs = screen.getAllByLabelText(en.batchI2vLong.table.promptInputLabel) as HTMLInputElement[];
    expect(inputs).toHaveLength(2);
    expect(inputs[0]!.maxLength).toBe(2000);

    // 行ごとに独立して入力できる。
    await user.type(inputs[1]!, "on the beach");
    expect(inputs[1]).toHaveValue("on the beach");
    expect(inputs[0]).toHaveValue("");

    // 📝でChain画面の共通プロンプトが流し込まれる（フェイクChainのprompt）。
    await user.click(screen.getAllByRole("button", { name: en.batchI2vLong.table.copyChainPromptButton })[0]!);
    expect(inputs[0]).toHaveValue("a cat walking");
  });

  // プロンプトが空の行は開始をブロックし、その行番号が理由に出る。
  it("行プロンプトも共通プロンプトも空ならpromptEmptyを行番号つきで出す", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({
      delayMs: 0,
      fs: imgFolder(["a.png", "b.png"]),
      pickFolderPath: IMG_DIR,
    });
    const { container } = renderSection(bridge, { chain: makeChain({ prompt: "  " }) });
    open(container);

    await user.click(screen.getByRole("button", { name: en.batchI2vLong.imgDir.button }));
    await user.click(screen.getByRole("button", { name: en.batchI2vLong.scanButton }));
    await waitFor(() => {
      expect(screen.getByText("a.png")).toBeInTheDocument();
    });

    const note = screen.getByRole("note");
    expect(within(note).getByText(en.batchI2vLong.promptRowIssues.empty("#1, #2"))).toBeInTheDocument();
    expect(screen.getByRole("button", { name: en.batchI2vLong.startButton })).toBeDisabled();
  });

  it("📁→スキャン→開始で走行中は全操作がdisabledになり、停止ボタンが出る", async () => {
    const user = userEvent.setup();
    // `holdUploads` で1枚目の画像アップロードを止め、走行中の状態を
    // 実タイマー競合なしに観測する。解放後は chain POST が500を返すので
    // その行だけFailedになって即座に走行が終わる。
    const base: MockBridge = createMockBridge({
      delayMs: 0,
      fs: imgFolder(["a.png", "b.png"]),
      pickFolderPath: IMG_DIR,
      holdUploads: true,
    });
    const { container } = renderSection(failChainSubmit(base));
    open(container);

    await user.click(screen.getByRole("button", { name: en.batchI2vLong.imgDir.button }));
    await waitFor(() => {
      expect(screen.getByLabelText(en.batchI2vLong.imgDir.label)).toHaveValue(IMG_DIR);
    });

    await user.click(screen.getByRole("button", { name: en.batchI2vLong.scanButton }));
    await waitFor(() => {
      expect(screen.getByText("a.png")).toBeInTheDocument();
    });

    const startButton = screen.getByRole("button", { name: en.batchI2vLong.startButton });
    expect(startButton).toBeEnabled();
    await user.click(startButton);

    // 走行中: 開始は「実行中…」へ変わり、フォルダ・スキャン・🔁が全部無効。
    expect(screen.getByRole("button", { name: en.batchI2vLong.runningButton })).toBeDisabled();
    expect(screen.getByRole("button", { name: en.batchI2vLong.stopButton })).toBeEnabled();
    expect(screen.getByRole("button", { name: en.batchI2vLong.scanButton })).toBeDisabled();
    expect(screen.getByLabelText(en.batchI2vLong.imgDir.label)).toBeDisabled();
    expect(screen.getByLabelText(en.batchI2vLong.outDir.label)).toBeDisabled();
    for (const input of screen.getAllByLabelText(en.batchI2vLong.table.promptInputLabel)) {
      expect(input).toBeDisabled();
    }
    for (const button of screen.getAllByRole("button", { name: en.batchI2vLong.table.copyChainPromptButton })) {
      expect(button).toBeDisabled();
    }
    for (const button of screen.getAllByRole("button", { name: en.batchI2vLong.table.resetButton })) {
      expect(button).toBeDisabled();
    }

    // 後片付け: アップロードを解放して走行を終わらせる（module-levelの
    // ランタイムに走行が残ったまま次テストへ漏れないように）。
    await act(async () => {
      base.releaseUploads();
    });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: en.batchI2vLong.startButton })).toBeInTheDocument();
    }, { timeout: 5_000 });
  }, 15_000);
});
