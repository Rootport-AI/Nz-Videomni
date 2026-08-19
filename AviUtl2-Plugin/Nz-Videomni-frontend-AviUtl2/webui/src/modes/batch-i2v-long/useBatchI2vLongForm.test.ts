import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../../api/client";
import type { AppConfig, GenerateChainRequest, LoraEntry } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFsFileEntry } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  __resetRunLockForTests,
  acquireRunLock,
  releaseRunLock,
} from "../../shell/runLock";
import type { ChainSnapshotSource } from "./chainSnapshot";
import { __resetI2vLongRuntimeForTests } from "./runtime";
import { useBatchI2vLongForm } from "./useBatchI2vLongForm";
import type { UseBatchI2vLongFormResult } from "./useBatchI2vLongForm";

const IMG_DIR = "C:\\shots\\ep01";
const AUTO_OUT_DIR = "C:\\shots\\ep01_i2vlong_out";

// ---------------------------------------------------------------------------
// The Chain form fake. This is the whole point of `ChainSnapshotSource`: the
// batch never imports `useChainForm`, so its tests only need this ~20-line
// object. Every `makeChain()` call produces a FRESH `buildRequest` identity,
// which is exactly what a real Chain-form edit does — that is what makes the
// template memo recompute on rerender.
// ---------------------------------------------------------------------------

interface ChainFakeState {
  prompt?: string;
  seed?: number;
  mode?: "scratch" | "v2v";
  clips?: Array<{ prompt: string; numFrames: number; startImage?: boolean }>;
  isValid?: boolean;
  validityReasons?: string[];
  loras?: Array<{ name: string; strength: number }>;
  /** §1-16 長尺A2V: an audio track attached to the Chain form. */
  hasSourceAudio?: boolean;
  /** §1-15 参照動画: a reference video attached to the Chain form. */
  hasReferenceVideo?: boolean;
  /** 素材（末尾）v2: end material attached to the Chain form. */
  hasEndSource?: boolean;
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
  const isValid = state.isValid ?? validityReasons.length === 0;

  const buildRequest = (): GenerateChainRequest => ({
    prompt,
    width: 768,
    height: 512,
    frame_rate: 24,
    seed,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: clips.map((clip) => ({
      ...(clip.prompt ? { prompt: clip.prompt } : {}),
      num_frames: clip.numFrames,
      ...(clip.startImage ? { conditioning_images: [{ image_id: "chain-start", frame_idx: 0, strength: 1 }] } : {}),
    })),
    ...(mode === "v2v" ? { source_video: { video_id: "vid-1", context_frames: 41 } } : {}),
    ...(state.hasSourceAudio ? { source_audio: { audio_id: "aud-1" } } : {}),
    ...(state.loras ? { loras: state.loras } : {}),
    ...(state.hasReferenceVideo
      ? { reference_video_id: "ref-1", conditioning_attention_strength: 0.8, reference_video_strength: 0.9 }
      : {}),
    ...(state.hasEndSource ? { end_source: { video_id: "end-1", context_frames: 136, strength: 1.0 } } : {}),
    chunked_upsample: true,
  });

  return {
    buildRequest,
    mode,
    hasSourceAudio: state.hasSourceAudio ?? false,
    hasReferenceVideo: state.hasReferenceVideo ?? false,
    hasEndSource: state.hasEndSource ?? false,
    clips: clips.map((clip) => ({ prompt: clip.prompt })),
    isValid,
    validityReasons,
    outputFrames: 239,
    outputSeconds: 9.96,
    minClips: 2,
    minFramesForOverlap: 25,
  };
}

/** A `useLoras`-shaped client stub. Only `getLoras` is ever reached from this
 * hook, so the rest of `ApiClient` is deliberately absent — a real client would
 * drag the whole bridge into every one of these tests for nothing. */
function lorasClient(names: string[]): ApiClient {
  const loras: LoraEntry[] = names.map((name) => ({
    name,
    kind: "style",
    has_thumbnail: false,
    exists: true,
    source: "scan",
  }));
  return { getLoras: async () => ({ loras }) } as unknown as ApiClient;
}

/** A client whose `GET /loras` never succeeds — the "cannot tell" case, where
 * the `unknownLoraTag` guard must PASS rather than block a whole batch on a
 * transient failure. */
function failingLorasClient(): ApiClient {
  return {
    getLoras: async () => {
      throw new Error("offline");
    },
  } as unknown as ApiClient;
}

interface RenderOpts {
  chain?: ChainSnapshotSource;
  hasActiveJob?: boolean;
  config?: AppConfig;
  apiClient?: ApiClient;
}

function renderForm(bridge: NativeBridge, opts: RenderOpts = {}) {
  const initialProps = {
    chain: opts.chain ?? makeChain(),
    hasActiveJob: opts.hasActiveJob ?? false,
  };
  const config = opts.config ?? FALLBACK_APP_CONFIG;
  const apiClient = opts.apiClient ?? lorasClient([]);
  return renderHook(
    (props: { chain: ChainSnapshotSource; hasActiveJob: boolean }) =>
      useBatchI2vLongForm(config, props.chain, props.hasActiveJob, {
        nativeBridge: bridge,
        apiClient,
        // Test-only: the production 1s job poll would make every
        // run-to-completion test take tens of seconds.
        pollIntervalMs: 4,
        jobBusyBackoffMs: 4,
      }),
    { initialProps },
  );
}

function imgFolder(entries: MockFsFileEntry[]) {
  return createMockFs({ folders: { [IMG_DIR]: entries } });
}

function png(name: string, sizeBytes = 1024): MockFsFileEntry {
  return { name, sizeBytes, mtimeMs: 0 };
}

/** Drives the hook to "image folder picked + scanned", the state every guard
 * test starts from. */
async function scanned(result: { current: UseBatchI2vLongFormResult }) {
  await act(async () => {
    await result.current.pickImgDir();
  });
  await act(async () => {
    await result.current.scan();
  });
}

describe("useBatchI2vLongForm", () => {
  beforeEach(() => {
    __resetI2vLongRuntimeForTests();
    __resetRunLockForTests();
  });

  // --- folders -------------------------------------------------------------

  it("pickImgDirは画像フォルダを設定し、出力フォルダを兄弟の_i2vlong_outへ自動導出する", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]), pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);

    await act(async () => {
      await result.current.pickImgDir();
    });

    expect(result.current.imgDir).toBe(IMG_DIR);
    expect(result.current.outDir).toBe(AUTO_OUT_DIR);
    expect(result.current.outDirIsAuto).toBe(true);
  });

  it("出力フォルダを手動指定すると自動導出が止まり、以後は画像フォルダを変えても追随しない", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]), pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);

    await act(async () => {
      await result.current.setOutDir("D:\\mine");
    });
    expect(result.current.outDirIsAuto).toBe(false);

    await act(async () => {
      await result.current.pickImgDir();
    });
    expect(result.current.imgDir).toBe(IMG_DIR);
    expect(result.current.outDir).toBe("D:\\mine");
  });

  it("setImgDirは存在プローブに失敗するとscanErrorへ出す", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const failing: NativeBridge = {
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "fs.listFiles") throw new Error("FOLDER_NOT_FOUND");
        return bridge.request(method, params);
      },
      requestWithFiles: (m, p, f) => bridge.requestWithFiles(m, p, f),
      on: (e, h) => bridge.on(e, h),
    };
    const { result } = renderForm(failing);

    await act(async () => {
      await result.current.setImgDir("C:\\nope");
    });

    expect(result.current.imgDir).toBe("C:\\nope");
    expect(result.current.scanError).toContain("FOLDER_NOT_FOUND");
  });

  it("pickImgDirのCANCELLEDは無視される（フォルダもエラーも変わらない）", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]), failPickFolder: "CANCELLED" });
    const { result } = renderForm(bridge);

    await act(async () => {
      await result.current.pickImgDir();
    });

    expect(result.current.imgDir).toBeNull();
    expect(result.current.scanError).toBeNull();
  });

  // --- scan ----------------------------------------------------------------

  it("scanはconfig由来の拡張子をfs.listFilesへ渡し、名前昇順で行を作る", async () => {
    const fs = imgFolder([png("b.png"), png("a10.png"), png("a2.png"), png("notes.txt")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderForm(bridge);

    await scanned(result);

    const listCall = spy.mock.calls.find(
      (call) => call[0] === "fs.listFiles" && (call[1] as { extensions?: string[] }).extensions !== undefined,
    );
    expect(listCall).toBeDefined();
    expect((listCall![1] as { extensions?: string[] }).extensions).toEqual(
      FALLBACK_APP_CONFIG.upload.allowed_image_extensions,
    );
    expect(result.current.rows.map((r) => r.image)).toEqual(["a2.png", "a10.png", "b.png"]);
    expect(result.current.rows.map((r) => r.queue)).toEqual([1, 2, 3]);
  });

  it("max_image_size_mbを超える画像はscan時点でFailedになる（伝搬の確認）", async () => {
    const config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      upload: { ...FALLBACK_APP_CONFIG.upload, max_image_size_mb: 1 },
    };
    const fs = imgFolder([png("small.png", 512), png("huge.png", 2 * 1024 * 1024)]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { config });

    await scanned(result);

    expect(result.current.rows.find((r) => r.image === "small.png")?.stat).toBe("Waiting");
    const huge = result.current.rows.find((r) => r.image === "huge.png");
    expect(huge?.stat).toBe("Failed");
    expect(huge?.error).toContain("too large");
  });

  it("resetRowToWaitingはDone/FailedのみWaitingへ戻す", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);

    // Waiting 行は 🔁 の対象外（何も起きない）。
    act(() => result.current.resetRowToWaiting(1));
    expect(result.current.rows[0]?.stat).toBe("Waiting");

    // Failed（サイズ超過など）→ Waiting は通る。
    const failedFs = imgFolder([png("big.png", 999_999_999)]);
    const bridge2 = createMockBridge({ delayMs: 0, fs: failedFs, pickFolderPath: IMG_DIR });
    const second = renderForm(bridge2, { config: { ...FALLBACK_APP_CONFIG, upload: { ...FALLBACK_APP_CONFIG.upload, max_image_size_mb: 1 } } });
    await scanned(second.result);
    expect(second.result.current.rows[0]?.stat).toBe("Failed");

    act(() => second.result.current.resetRowToWaiting(1));
    expect(second.result.current.rows[0]?.stat).toBe("Waiting");
    expect(second.result.current.rows[0]?.error).toBe("");
  });

  // --- block reasons, one by one -------------------------------------------

  it("画像フォルダ未選択でimgDirMissing/outDirMissing/noRows", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const { result } = renderForm(bridge);

    expect(result.current.blockReasons).toContain("imgDirMissing");
    expect(result.current.blockReasons).toContain("outDirMissing");
    expect(result.current.blockReasons).toContain("noRows");
    expect(result.current.canStart).toBe(false);
  });

  // `noRunnableRows`（全行Done）は、完走まで走らせる下のテストがピン留めする。

  it("chain.mode==='v2v'でsourceVideoAttached、かつchain側のsource系理由は抑制される", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      mode: "v2v",
      isValid: false,
      validityReasons: ["sourceNotReady", "minClips"],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).toContain("sourceVideoAttached");
    expect(result.current.chainBlockReasons).toEqual(["minClips"]);
  });

  // §1-16 長尺A2V: 動画側とまったく同型。バッチは`source_audio`を剥がして送るので、
  // 「音声を外して」の1行だけを見せ、chain側の音声理由は重ねない。
  it("chain.hasSourceAudioでsourceAudioAttached、かつchain側のaudio系理由は抑制される", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      hasSourceAudio: true,
      isValid: false,
      validityReasons: ["audioNotReady", "chainLayoutInvalid", "minClips"],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).toContain("sourceAudioAttached");
    expect(result.current.chainBlockReasons).toEqual(["minClips"]);
    expect(result.current.canStart).toBe(false);
  });

  it("音声が付いていなければsourceAudioAttachedは出ず、chain側のaudio系理由もそのまま出る", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ isValid: false, validityReasons: ["audioNotReady"] });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).not.toContain("sourceAudioAttached");
    expect(result.current.chainBlockReasons).toEqual(["audioNotReady"]);
  });

  // §1-15 参照動画: 音声とまったく同型。バッチは`reference_video_id`と強度2種を
  // 剥がして送るので、「参照動画を外して」の1行だけを見せ、chain側の参照理由は
  // 重ねない（ただし128グリッドの理由は参照を外しても残りうるので抑制しない）。
  it("chain.hasReferenceVideoでreferenceVideoAttached、かつchain側の参照理由は抑制される", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      hasReferenceVideo: true,
      isValid: false,
      validityReasons: ["referenceNeedsLoras", "depthChainUnsupported", "minClips"],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).toContain("referenceVideoAttached");
    expect(result.current.chainBlockReasons).toEqual(["minClips"]);
    expect(result.current.canStart).toBe(false);
  });

  it("参照が付いていなければreferenceVideoAttachedは出ず、chain側の参照理由もそのまま出る", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ isValid: false, validityReasons: ["referenceNeedsLoras"] });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).not.toContain("referenceVideoAttached");
    expect(result.current.chainBlockReasons).toEqual(["referenceNeedsLoras"]);
  });

  it("参照が付いていても128グリッドの理由は抑制しない（参照を外しても直らないため）", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      hasReferenceVideo: true,
      isValid: false,
      validityReasons: ["referenceDimensionsOffGrid"],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.chainBlockReasons).toEqual(["referenceDimensionsOffGrid"]);
  });

  // 素材（末尾）: 音声・参照とまったく同型。バッチは `end_source` を剥がして
  // 送るので、「素材（末尾）を外して」の1行だけを見せ、chain側の末尾理由は重ねない。
  // 逆順Chained (2026-08-18) で増えた `endSourceAudioOverlapBudget` も同じ扱い
  // （素材を外せば消える理由なので、`CHAIN_END_SOURCE_REASON_CODES` に入っている）。
  it("chain.hasEndSourceでendSourceAttached、かつchain側の末尾理由は抑制される", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      hasEndSource: true,
      isValid: false,
      validityReasons: ["endSourceTooShort", "endSourceAudioOverlapBudget", "endSourceNeedsOverlap", "minClips"],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).toContain("endSourceAttached");
    expect(result.current.chainBlockReasons).toEqual(["minClips"]);
    expect(result.current.canStart).toBe(false);
  });

  it("素材（末尾）が付いていなければendSourceAttachedは出ず、chain側の末尾理由もそのまま出る", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ isValid: false, validityReasons: ["endSourceTooShort"] });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).not.toContain("endSourceAttached");
    expect(result.current.chainBlockReasons).toEqual(["endSourceTooShort"]);
  });

  it("素材（末尾）付きのテンプレートから end_source が剥がれている", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ hasEndSource: true });
    const { result } = renderForm(bridge, { chain });

    expect("end_source" in result.current.templatePreview).toBe(false);
  });

  it("clipsが2本未満でclipsTooFew", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ clips: [{ prompt: "", numFrames: 121 }] });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.blockReasons).toContain("clipsTooFew");
  });

  it("chainが不正ならvalidityReasonsを個別にそのまま公開する", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ isValid: false, validityReasons: ["dimensionsOffGrid", "clipFramesOffGrid"] });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.chainBlockReasons).toEqual(["dimensionsOffGrid", "clipFramesOffGrid"]);
    expect(result.current.canStart).toBe(false);
  });

  // --- prompt guards (2026-07-30 行ごとプロンプト) ---------------------------

  it("スキャン前は共通プロンプトだけでpromptEmptyを判定する", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "   " }) });

    expect(result.current.blockReasons).toContain("promptEmpty");
    expect(result.current.promptEmptyQueues).toEqual([]);
  });

  it("共通プロンプトが空なら、行プロンプトが空の実行対象行だけがpromptEmptyになる", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "   " }) });
    await scanned(result);

    expect(result.current.blockReasons).toContain("promptEmpty");
    expect(result.current.promptEmptyQueues).toEqual([1, 2]);

    // 1行だけ埋めても、もう片方が空なら開始できない。
    act(() => result.current.setRowPromptLocal(0, "a dog"));
    expect(result.current.promptEmptyQueues).toEqual([2]);
    expect(result.current.blockReasons).toContain("promptEmpty");

    act(() => result.current.setRowPromptLocal(1, "a cat"));
    expect(result.current.promptEmptyQueues).toEqual([]);
    expect(result.current.blockReasons).not.toContain("promptEmpty");
  });

  it("共通プロンプトがあれば、行プロンプトが空でもpromptEmptyにはならない", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "a cat walking" }) });
    await scanned(result);

    expect(result.current.blockReasons).not.toContain("promptEmpty");
    // replaceでも、行が空なら共通プロンプトがそのまま使われる。
    act(() => result.current.setPromptMode("replace"));
    expect(result.current.blockReasons).not.toContain("promptEmpty");
  });

  it("promptTooLongは合成後2000字で通り、2001字で立つ（境界・行単位）", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "base" }) });
    await scanned(result);

    act(() => result.current.setPromptMode("replace"));

    act(() => result.current.setRowPromptLocal(1, "x".repeat(2000)));
    expect(result.current.blockReasons).not.toContain("promptTooLong");

    act(() => result.current.setRowPromptLocal(1, "x".repeat(2001)));
    expect(result.current.blockReasons).toContain("promptTooLong");
    // 長すぎるのは2行目だけ。
    expect(result.current.promptTooLongQueues).toEqual([2]);
  });

  it("addモードでは共通プロンプトと行プロンプトの合計長で判定する", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "base" }) });
    await scanned(result);

    // "base" + 空白1文字 + 1995 = 2000ちょうど。
    act(() => result.current.setRowPromptLocal(0, "x".repeat(1995)));
    expect(result.current.blockReasons).not.toContain("promptTooLong");

    act(() => result.current.setRowPromptLocal(0, "x".repeat(1996)));
    expect(result.current.blockReasons).toContain("promptTooLong");
  });

  // --- row prompt editing ---------------------------------------------------

  it("行プロンプトは行単位で編集でき、再スキャンで空に戻る", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);

    expect(result.current.rows.map((r) => r.prompt)).toEqual(["", ""]);

    act(() => result.current.setRowPromptLocal(1, "on the beach"));
    expect(result.current.rows.map((r) => r.prompt)).toEqual(["", "on the beach"]);

    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows.map((r) => r.prompt)).toEqual(["", ""]);
  });

  it("📝は共通プロンプト（LoRAタグ除去後）をその行へ流し込む", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "a cat walking" }) });
    await scanned(result);

    act(() => result.current.copyChainPromptToRow(1));
    expect(result.current.rows.map((r) => r.prompt)).toEqual(["", "a cat walking"]);
  });

  it("既知LoRA名に無い名前があればunknownLoraTag", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ loras: [{ name: "Ghost", strength: 1 }] });
    const { result } = renderForm(bridge, { chain, apiClient: lorasClient(["Pixar_Toon"]) });

    await waitFor(() => {
      expect(result.current.blockReasons).toContain("unknownLoraTag");
    });
  });

  it("既知LoRA名に含まれていればunknownLoraTagは立たない", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ loras: [{ name: "Pixar_Toon", strength: 1 }] });
    const { result } = renderForm(bridge, { chain, apiClient: lorasClient(["Pixar_Toon"]) });

    await waitFor(() => {
      expect(result.current.blockReasons).not.toContain("unknownLoraTag");
    });
  });

  it("LoRA一覧が取得できないときはunknownLoraTagを立てない（保守的に通す）", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({ loras: [{ name: "Ghost", strength: 1 }] });
    const { result } = renderForm(bridge, { chain, apiClient: failingLorasClient() });

    // 失敗が確定するまで待ってから、それでもガードが立たないことを見る。
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.blockReasons).not.toContain("unknownLoraTag");
  });

  it("hasActiveJobでjobActive", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const { result } = renderForm(bridge, { hasActiveJob: true });

    expect(result.current.blockReasons).toContain("jobActive");
  });

  // 共有ランロック（`shell/runLock.ts`）がバッチA2V側に握られている間は開始できない。
  // 理由を出さないと、押しても何も起きないボタンになる（runtime.run が取得に失敗して
  // 黙って idle のまま戻る）。
  it("バッチA2Vがランロックを保持している間はlockedByOtherで開始できない", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);
    // 満たせば開始できる状態から始める（ロック以外の理由が混ざらないこと）。
    expect(result.current.blockReasons).toEqual([]);

    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(token).not.toBeNull();
    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.blockReasons).toEqual(["lockedByOther"]);
    expect(result.current.canStart).toBe(false);

    // A2V側が解放すれば即座に開始できるようになる（購読が効いていること）。
    await act(async () => {
      releaseRunLock(token);
      await Promise.resolve();
    });
    expect(result.current.blockReasons).toEqual([]);
    expect(result.current.canStart).toBe(true);
  });

  it("自分（i2v-long）が握っているロックはlockedByOtherにしない", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);

    acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    await act(async () => {
      await Promise.resolve();
    });

    // 走行中の自分自身は `runner.state !== "idle"` 側で既に止まっているので、
    // ここで lockedByOther を出すと二重表示かつ意味的に誤りになる。
    expect(result.current.blockReasons).not.toContain("lockedByOther");
  });

  it("全ガードを満たせばcanStartがtrueになる", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);

    expect(result.current.blockReasons).toEqual([]);
    expect(result.current.chainBlockReasons).toEqual([]);
    expect(result.current.canStart).toBe(true);
  });

  // --- non-blocking notes ---------------------------------------------------

  it("非ブロック注意フラグ（seed固定・clip0個別prompt・他clip個別prompt・開始フレーム置換）", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      seed: 42,
      clips: [
        { prompt: "clip0 override", numFrames: 121, startImage: true },
        { prompt: "clip1 override", numFrames: 121 },
      ],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.notes).toEqual({
      seedFixed: true,
      clip0PromptOverride: true,
      otherClipPromptOverride: true,
      startFrameIgnored: true,
    });
    // どれもブロックはしない。
    expect(result.current.blockReasons).not.toContain("promptEmpty");
  });

  it("seed=-1・個別prompt無し・開始フレーム無しなら注意フラグは全部false", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const { result } = renderForm(bridge);

    expect(result.current.notes).toEqual({
      seedFixed: false,
      clip0PromptOverride: false,
      otherClipPromptOverride: false,
      startFrameIgnored: false,
    });
  });

  // --- template preview -----------------------------------------------------

  // 2026-07-30: テンプレートのpromptは常にChain画面の共通プロンプトそのまま
  // （行ごとの合成はランナー側）。promptModeはテンプレートを変えない。
  it("templatePreview.promptは共通プロンプトのままで、promptModeでは変わらない", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const { result } = renderForm(bridge, { chain: makeChain({ prompt: "base" }) });

    expect(result.current.templatePreview.prompt).toBe("base");

    act(() => result.current.setPromptMode("replace"));
    expect(result.current.templatePreview.prompt).toBe("base");
  });

  it("templatePreviewはsource_videoとclip0の開始フレームを落とす", () => {
    const bridge = createMockBridge({ delayMs: 0, fs: imgFolder([]) });
    const chain = makeChain({
      mode: "v2v",
      clips: [
        { prompt: "", numFrames: 121, startImage: true },
        { prompt: "", numFrames: 121 },
      ],
    });
    const { result } = renderForm(bridge, { chain });

    expect(result.current.templatePreview.source_video).toBeUndefined();
    expect(result.current.templatePreview.clips[0]?.conditioning_images).toBeUndefined();
  });

  // --- snapshot freeze ------------------------------------------------------

  it("start後にchainフェイクを変えても、ランナーへ渡ったtemplateは変わらない", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR, runningPollCount: 30 });
    const chainBodies: Array<Record<string, unknown>> = [];
    const capturing: NativeBridge = {
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "backend.request") {
          const p = params as { method?: string; path?: string; body?: Record<string, unknown> };
          if (p.method === "POST" && p.path === "/api/v1/generate/chain") {
            chainBodies.push(p.body ?? {});
            return { status: 500, body: { error: { code: "MOCK_STOP", message: "captured" } } } as unknown as ResultOf<M>;
          }
        }
        return bridge.request(method, params);
      },
      requestWithFiles: (m, p, f) => bridge.requestWithFiles(m, p, f),
      on: (e, h) => bridge.on(e, h),
    };

    const { result, rerender } = renderForm(capturing, { chain: makeChain({ prompt: "before" }) });
    await scanned(result);

    const startedTemplate = result.current.templatePreview;
    act(() => result.current.start());

    // 走行中にChain画面が編集された、を模す。
    rerender({ chain: makeChain({ prompt: "AFTER EDIT" }), hasActiveJob: false });
    expect(result.current.templatePreview.prompt).toBe("AFTER EDIT");
    expect(result.current.templatePreview).not.toBe(startedTemplate);

    await waitFor(() => {
      expect(chainBodies).toHaveLength(1);
    });
    // 送信されたのは開始時点のスナップショット。
    expect(chainBodies[0]?.prompt).toBe("before");
  });

  // 行ごとプロンプトが実際に送信ボディへ載ることの通しピン（2026-07-30）。
  it("start()は行ごとのプロンプトを合成して送る（modeは開始時点で凍結される）", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR });
    const chainBodies: Array<Record<string, unknown>> = [];
    const capturing: NativeBridge = {
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "backend.request") {
          const p = params as { method?: string; path?: string; body?: Record<string, unknown> };
          if (p.method === "POST" && p.path === "/api/v1/generate/chain") {
            chainBodies.push(p.body ?? {});
            return { status: 500, body: { error: { code: "MOCK_STOP", message: "captured" } } } as unknown as ResultOf<M>;
          }
        }
        return bridge.request(method, params);
      },
      requestWithFiles: (m, p, f) => bridge.requestWithFiles(m, p, f),
      on: (e, h) => bridge.on(e, h),
    };

    const { result } = renderForm(capturing, { chain: makeChain({ prompt: "a cat walking" }) });
    await scanned(result);

    act(() => result.current.setRowPromptLocal(0, "in the rain"));
    act(() => result.current.start());

    // 走行開始後にラジオを切り替えても、走行中のバッチには効かない。
    act(() => result.current.setPromptMode("replace"));

    await waitFor(() => {
      expect(chainBodies).toHaveLength(2);
    }, { timeout: 5_000 });
    expect(chainBodies.map((b) => b.prompt)).toEqual(["a cat walking in the rain", "a cat walking"]);

    await waitFor(() => {
      expect(result.current.runnerState).toBe("idle");
    }, { timeout: 5_000 });
  }, 10_000);

  it("start()は開始時点のフォルダ・テンプレ・行をランナーへ値渡しし、実行中はcanStartが落ちる", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR, runningPollCount: 30 });
    const { result } = renderForm(bridge);
    await scanned(result);

    act(() => result.current.start());

    expect(result.current.runnerState).toBe("running");
    expect(result.current.canStart).toBe(false);
    expect(result.current.canScan).toBe(false);

    act(() => result.current.stop());
    expect(result.current.runnerState).toBe("stopping");

    await waitFor(() => {
      expect(result.current.runnerState).toBe("idle");
    }, { timeout: 5_000 });
  }, 10_000);

  // 実機報告(2026-07-30): 1行目だけ⏳Generatingにならず、完了した瞬間に✅Doneへ
  // 飛ぶ。`holdUploads`で1行目のアップロードを止めておくと、「Generatingを
  // flushした直後」の状態をタイマー競争なしで観測できる。
  it("start()直後に1行目が⏳Generatingになる（2行目以降だけでなく）", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: IMG_DIR, runningPollCount: 1, holdUploads: true });
    const { result } = renderForm(bridge);
    await scanned(result);

    act(() => result.current.start());

    expect(result.current.runnerState).toBe("running");
    expect(result.current.rows[0]?.stat).toBe("Generating");
    expect(result.current.rows[1]?.stat).toBe("Waiting");
    expect(result.current.summary.generating).toBe(1);
    expect(result.current.currentRow?.image).toBe("a.png");

    bridge.releaseUploads();
    await waitFor(() => {
      expect(result.current.runnerState).toBe("idle");
    }, { timeout: 8_000 });
  }, 15_000);

  it("実行が完走すると全行Doneになり、noRunnableRowsでブロックされる", async () => {
    const fs = imgFolder([png("a.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, runningPollCount: 1, pickFolderPath: IMG_DIR });
    const { result } = renderForm(bridge);
    await scanned(result);

    act(() => result.current.start());
    await waitFor(() => {
      expect(result.current.runnerState).toBe("idle");
    }, { timeout: 5_000 });

    expect(result.current.rows.every((r) => r.stat === "Done")).toBe(true);
    expect(result.current.summary.done).toBe(1);
    expect(result.current.blockReasons).toContain("noRunnableRows");
    expect(result.current.blockReasons).not.toContain("noRows");
  }, 10_000);

  it("リマウントしても走行中バッチのフォルダ・行・状態へ再接続する", async () => {
    const fs = imgFolder([png("a.png"), png("b.png")]);
    const bridge = createMockBridge({ delayMs: 0, fs, runningPollCount: 3, pickFolderPath: IMG_DIR });
    const first = renderForm(bridge);
    await scanned(first.result);

    act(() => first.result.current.start());
    expect(first.result.current.runnerState).toBe("running");

    first.unmount();
    const second = renderForm(bridge);

    expect(second.result.current.runnerState).toBe("running");
    expect(second.result.current.imgDir).toBe(IMG_DIR);
    expect(second.result.current.outDir).toBe(AUTO_OUT_DIR);
    expect(second.result.current.outDirIsAuto).toBe(false);
    expect(second.result.current.rows).toHaveLength(2);

    await waitFor(() => {
      expect(second.result.current.runnerState).toBe("idle");
    }, { timeout: 8_000 });
  }, 15_000);
});
