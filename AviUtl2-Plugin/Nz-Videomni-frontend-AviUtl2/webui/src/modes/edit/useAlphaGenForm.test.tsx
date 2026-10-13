import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../../api/client";
import type { NativeBridge } from "../../bridge";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import type { SelectionItem, TimelineSelection } from "../../timeline/menuSelection";
import { useGenerationSubmit } from "../single/useGeneration";
import { useAlphaGenForm } from "./useAlphaGenForm";
import type { UseAlphaGenFormDeps } from "./useAlphaGenForm";

/**
 * Alpha Gen 第 2 弾のフォーム。押さえたいのは:
 *  - 入口 3 つの開始秒（右クリックは再生位置・ドロップと 📁 は冒頭）
 *  - fps は表示のための生値で、0 は「不明」（秒数も線も出さない）
 *  - 快適上限の線（軽量モードで予算が変わる・行が無ければ出ない・9 未満は出ない）
 *  - 寸法の注意 2 種
 *  - 生成時のアップロードの query 2 通りと `window_start_sec`
 *  - サーバーの実測で切り下げたらトーストを 1 回
 *  - 本文は素通しの 5 鍵だけ・`seed: -1`・`frame_rate` 無し
 *  - アップロード中に画面が作り直されたら送らない
 */

const FILE = "C:\\videos\\dance.mp4";

interface MediaInfo {
  durationSec: number;
  width: number;
  height: number;
  fps?: number;
}

interface BridgeOptions {
  media?: MediaInfo;
  /** `backend.uploadFile`'s body. Default: the window was cut and measured. */
  upload?: Record<string, unknown>;
  /** Hold the upload until the returned `release` is called. */
  holdUpload?: boolean;
  pick?: { filePath: string; fileName: string };
}

function createBridge(options: BridgeOptions = {}) {
  const media = options.media ?? { durationSec: 10, width: 1920, height: 1080, fps: 30 };
  let release: () => void = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "fs.probeMediaInfo") return media;
    if (method === "ui.pickFile") return options.pick ?? { filePath: "C:\\videos\\picked.mp4", fileName: "picked.mp4" };
    if (method === "backend.uploadFile") {
      if (options.holdUpload) await held;
      return { status: 200, body: options.upload ?? { video_id: "vid-1", trimmed: true, frame_count: 82, fps: 30 } };
    }
    if (method === "backend.request") {
      const { path } = params as { path: string };
      if (path.endsWith("/generate/alpha")) {
        return { status: 202, body: { job_id: "job-1", status: "queued", created_at: "2026-10-13T00:00:00Z" } };
      }
    }
    throw new Error(`unexpected bridge call: ${method}`);
  });
  const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
  return { bridge, request, release: () => release() };
}

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

function intent(item: SelectionItem = makeItem(), name = "alpha-gen"): GenerationPrefill {
  const selection: TimelineSelection = {
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
  return { intent: name, targetMode: "edit", selection };
}

function renderForm(deps: Partial<UseAlphaGenFormDeps> = {}, options: BridgeOptions = {}) {
  const { bridge, request, release } = createBridge(options);
  const view = renderHook(() => useAlphaGenForm({ nativeBridge: bridge, ...deps }));
  return { ...view, request, release };
}

/** The `backend.uploadFile` params of the (only) upload. */
function uploadParams(request: ReturnType<typeof vi.fn>): { filePath: string; query?: Record<string, string> } {
  const call = request.mock.calls.find((c) => c[0] === "backend.uploadFile");
  if (!call) throw new Error("no upload");
  return call[1] as { filePath: string; query?: Record<string, string> };
}

async function waitMeasured(result: { current: ReturnType<typeof useAlphaGenForm> }) {
  await waitFor(() => expect(result.current.mediaInfo.width).toBeGreaterThan(0));
}

describe("useAlphaGenForm — 入口と開始秒", () => {
  it("右クリックは再生位置から（右クリックの intent が alpha-gen のときだけ）", async () => {
    const { result, request } = renderForm({ initialIntent: intent() });
    expect(result.current.source).toEqual({
      filePath: FILE,
      fileName: "dance.mp4",
      startSec: 2.5,
      origin: "rightClick",
    });
    // 入口ではアップロードしない（生成ボタンの時点で窓だけを上げる）。
    await waitMeasured(result);
    expect(request.mock.calls.some((c) => c[0] === "backend.uploadFile")).toBe(false);
  });

  it("再生位置が報告されていなければ冒頭から", () => {
    const { result } = renderForm({ initialIntent: intent(makeItem({ hasPlaybackRange: false })) });
    expect(result.current.source?.startSec).toBe(0);
  });

  it("他のサブタブ行きの右クリックには反応しない", () => {
    const { result } = renderForm({ initialIntent: intent(makeItem(), "outpaint") });
    expect(result.current.source).toBeNull();
    expect(result.current.validityReasons).toEqual(["sourceMissing"]);
  });

  it("ドロップと 📁 は冒頭から", async () => {
    const { result } = renderForm({ initialIntent: intent() });
    act(() => result.current.attachDropped("C:\\videos\\drop.mp4", "drop.mp4"));
    expect(result.current.source).toMatchObject({ startSec: 0, origin: "drop", fileName: "drop.mp4" });
    await act(async () => {
      await result.current.pick();
    });
    expect(result.current.source).toMatchObject({ startSec: 0, origin: "pick", fileName: "picked.mp4" });
  });
});

describe("useAlphaGenForm — fps と秒数", () => {
  it("fps は生値のまま持ち、秒数はそこから出す", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 10, width: 1920, height: 1080, fps: 29.97 } },
    );
    await waitMeasured(result);
    expect(result.current.mediaInfo.fps).toBe(29.97);
    expect(result.current.seconds).toBeCloseTo(81 / 29.97, 6);
  });

  it("fps が 0（不明）なら秒数は出さず、生成は止めない", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 0, width: 1920, height: 1080, fps: 0 } },
    );
    await waitMeasured(result);
    expect(result.current.seconds).toBeNull();
    expect(result.current.isValid).toBe(true);
  });

  it("古いプラグイン（fps が無い）は 0 として読む", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 10, width: 1920, height: 1080 } },
    );
    await waitMeasured(result);
    expect(result.current.mediaInfo.fps).toBe(0);
  });

  it("使えるコマが 9 未満なら sourceTooShort", async () => {
    // 再生位置 2.5 秒・長さ 2.7 秒 @30fps → 残り 6 コマ。
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 2.7, width: 1920, height: 1080, fps: 30 } },
    );
    await waitFor(() => expect(result.current.validityReasons).toEqual(["sourceTooShort"]));
  });
});

describe("useAlphaGenForm — 快適上限の線", () => {
  // 1920x1080 → キャンバス 1920x1088 → 60x34 = 2040 マス。
  const LTX25_4BIT = { engineFamily: "ltx25", weightClass: "4bit" } as const;

  it("原寸の予算（alpha_gen_budget）で引き、軽量モードでは単発の予算に変わる", async () => {
    const { result } = renderForm({ initialIntent: intent(), ...LTX25_4BIT });
    await waitMeasured(result);
    // 23460 / 2040 = 11 潜在フレーム → 81 コマ。
    expect(result.current.comfortFrames).toBe(81);
    act(() => result.current.setLightMode(true));
    // 46920 / 2040 = 23 → 177 → 145 で頭打ち。
    expect(result.current.comfortFrames).toBe(145);
  });

  it("行が無い（重みの種別が不明）なら線を出さない", async () => {
    const { result } = renderForm({ initialIntent: intent(), engineFamily: "ltx25", weightClass: "" });
    await waitMeasured(result);
    expect(result.current.comfortFrames).toBeNull();
  });

  it("行に alpha_gen_budget が無い（LTX 2.3 の行）なら線を出さない", async () => {
    const { result } = renderForm({ initialIntent: intent(), engineFamily: "ltx", weightClass: "8bit" });
    await waitMeasured(result);
    expect(result.current.comfortFrames).toBeNull();
  });

  it("頭打ち前の値が 9 未満（9 コマでも縮小される）なら線を出さない", async () => {
    const { result } = renderForm(
      { initialIntent: intent(), ...LTX25_4BIT },
      { media: { durationSec: 10, width: 4096, height: 4096, fps: 30 } },
    );
    await waitMeasured(result);
    expect(result.current.comfortFrames).toBeNull();
  });
});

describe("useAlphaGenForm — 寸法の注意", () => {
  it("1920*1088 を超えると注意", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 10, width: 2560, height: 1440, fps: 30 } },
    );
    await waitMeasured(result);
    expect(result.current.aboveOfficialMax).toBe(true);
    expect(result.current.belowLightFloor).toBe(false);
  });

  it("軽量モードで 1280*768 未満なら警告（チェックそのものは止めない）", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { media: { durationSec: 10, width: 960, height: 540, fps: 30 } },
    );
    await waitMeasured(result);
    expect(result.current.belowLightFloor).toBe(false);
    act(() => result.current.setLightMode(true));
    expect(result.current.lightMode).toBe(true);
    expect(result.current.belowLightFloor).toBe(true);
    expect(result.current.aboveOfficialMax).toBe(false);
  });
});

describe("useAlphaGenForm — prepareRequest", () => {
  it("窓の時間が決まるときは窓だけを上げる（trim＋max_frames）", async () => {
    const { result, request } = renderForm({ initialIntent: intent() });
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(uploadParams(request)).toEqual({
      kind: "video",
      filePath: FILE,
      query: { trim_start_sec: "2.500", trim_duration_sec: "2.733", max_frames: "82" },
    });
    // 切り出せたので、上げたファイルの頭が窓の頭。
    expect(body).toEqual({
      reference_video_id: "vid-1",
      window_start_sec: 0,
      num_frames: 81,
      seed: -1,
      light_mode: false,
    });
  });

  it("開始が 0 なら max_frames だけ（開始の無い trim_duration_sec はサーバーが無視するので送らない）", async () => {
    const { result, request } = renderForm({ initialIntent: intent(makeItem({ playbackStartSec: 0 })) });
    await waitMeasured(result);
    await act(async () => {
      await result.current.prepareRequest();
    });
    expect(uploadParams(request).query).toEqual({ max_frames: "82" });
  });

  it("開始 S>0・トリムが効いた・実測が要求より少ない → window_start_sec=0 で実測から切り下げる", async () => {
    const onAdjusted = vi.fn();
    const { result } = renderForm(
      { initialIntent: intent(), onAdjusted },
      { upload: { video_id: "vid-6", trimmed: true, frame_count: 60, fps: 30 } },
    );
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    // snapRetakeWindowFrames(60) = 57.
    expect(body).toMatchObject({ reference_video_id: "vid-6", window_start_sec: 0, num_frames: 57 });
    expect(onAdjusted).toHaveBeenCalledTimes(1);
  });

  it("アップロード後の実測が 9 コマ未満なら null で送らず、理由に sourceTooShort", async () => {
    const { bridge, request } = createBridge({
      upload: { video_id: "vid-7", trimmed: true, frame_count: 5, fps: 30 },
    });
    const client = createApiClient(bridge);
    const view = renderHook(() => {
      const form = useAlphaGenForm({ nativeBridge: bridge, initialIntent: intent() });
      const submit = useGenerationSubmit({ apiClient: client });
      return { form, submit };
    });
    await waitFor(() => expect(view.result.current.form.mediaInfo.width).toBeGreaterThan(0));
    let body: Awaited<ReturnType<typeof view.result.current.form.prepareRequest>> = null;
    await act(async () => {
      body = await view.result.current.form.prepareRequest();
    });
    expect(body).toBeNull();
    expect(view.result.current.form.validityReasons).toEqual(["sourceTooShort"]);

    // Through the submit path too: nothing reaches /generate/alpha.
    act(() => view.result.current.submit.submitAlpha(view.result.current.form.prepareRequest));
    await waitFor(() => expect(view.result.current.submit.submitState.phase).toBe("idle"));
    expect(
      request.mock.calls.some(
        (c) => c[0] === "backend.request" && (c[1] as { path: string }).path.endsWith("/generate/alpha"),
      ),
    ).toBe(false);
  });

  it("窓の時間が決まらない（fps 不明）なら query なしで丸ごと上げ、window_start_sec に再生位置", async () => {
    const { result, request } = renderForm(
      { initialIntent: intent() },
      {
        media: { durationSec: 10, width: 1920, height: 1080, fps: 0 },
        upload: { video_id: "vid-2", trimmed: false, frame_count: null, fps: null },
      },
    );
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(uploadParams(request)).toEqual({ kind: "video", filePath: FILE });
    expect(body).toMatchObject({ reference_video_id: "vid-2", window_start_sec: 2.5, num_frames: 81 });
  });

  it("トリムが効かなかった（trimmed=false）ら元のファイルの上で再生位置から切る", async () => {
    const { result } = renderForm(
      { initialIntent: intent() },
      { upload: { video_id: "vid-3", trimmed: false, frame_count: null, fps: null } },
    );
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(body).toMatchObject({ window_start_sec: 2.5, num_frames: 81 });
  });

  it("サーバーの実測が短ければ切り下げ、トーストを 1 回だけ頼む", async () => {
    const onAdjusted = vi.fn();
    const { result } = renderForm(
      { initialIntent: intent(makeItem({ playbackStartSec: 0 })), onAdjusted },
      { upload: { video_id: "vid-4", trimmed: false, frame_count: 50, fps: 30 } },
    );
    await waitMeasured(result);
    act(() => result.current.setNumFrames(121, true));
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(body).toMatchObject({ num_frames: 49, window_start_sec: 0 });
    expect(result.current.numFrames).toBe(49);
    expect(onAdjusted).toHaveBeenCalledTimes(1);
  });

  it("長さから見て足りないときもアップロード前に切り下げる（トースト 1 回）", async () => {
    const onAdjusted = vi.fn();
    // 3 秒 @30fps・冒頭から → 90 コマ → 8n+1 で 89。
    const { result, request } = renderForm(
      { initialIntent: intent(makeItem({ playbackStartSec: 0 })), onAdjusted },
      {
        media: { durationSec: 3, width: 1920, height: 1080, fps: 30 },
        upload: { video_id: "vid-5", trimmed: false, frame_count: 90, fps: 30 },
      },
    );
    await waitMeasured(result);
    act(() => result.current.setNumFrames(121, true));
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(uploadParams(request).query).toMatchObject({ max_frames: "90" });
    expect(body).toMatchObject({ num_frames: 89 });
    expect(onAdjusted).toHaveBeenCalledTimes(1);
  });

  it("切り下げが無ければトーストは頼まない", async () => {
    const onAdjusted = vi.fn();
    const { result } = renderForm({ initialIntent: intent(), onAdjusted });
    await waitMeasured(result);
    await act(async () => {
      await result.current.prepareRequest();
    });
    expect(onAdjusted).not.toHaveBeenCalled();
  });

  it("素通しは 5 鍵だけ（vae_mode / embed_mp4_metadata は送らない）・seed は -1・frame_rate 無し", async () => {
    const acceleration: AccelerationSettings = {
      ...ACCELERATION_DEFAULTS,
      attentionBackend: "sage",
      blockSwapPrefetch: false,
      keepResident: false,
      fusedGgufDequantKernel: false,
      vaeMode: "prune_vaed",
      keepResidentEmbeddings: true,
      embedMp4Metadata: false,
    };
    const { result } = renderForm({ initialIntent: intent(), acceleration });
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(body).toEqual({
      reference_video_id: "vid-1",
      window_start_sec: 0,
      num_frames: 81,
      seed: -1,
      light_mode: false,
      attention_backend: "sage",
      block_swap_prefetch: false,
      keep_resident_embeddings: true,
      fused_gguf_dequant_kernel: false,
    });
    expect(body).not.toHaveProperty("frame_rate");
  });

  it("アップロードに失敗したら null と uploadError", async () => {
    const { result } = renderForm({ initialIntent: intent() }, { upload: { error: { code: "UPLOAD_FAILED" } } });
    await waitMeasured(result);
    let body: Awaited<ReturnType<typeof result.current.prepareRequest>> = null;
    await act(async () => {
      body = await result.current.prepareRequest();
    });
    expect(body).toBeNull();
    expect(result.current.uploadError).toBe(true);
  });
});

describe("useAlphaGenForm × submitAlpha — 作り直された画面は送らない", () => {
  it("アップロード中にアンマウントされたら /generate/alpha を叩かない", async () => {
    const { bridge, request, release } = createBridge({ holdUpload: true });
    const client = createApiClient(bridge);
    const view = renderHook(() => {
      const form = useAlphaGenForm({ nativeBridge: bridge, initialIntent: intent() });
      const submit = useGenerationSubmit({ apiClient: client });
      return { form, submit };
    });
    await waitFor(() => expect(view.result.current.form.mediaInfo.width).toBeGreaterThan(0));

    act(() => view.result.current.submit.submitAlpha(view.result.current.form.prepareRequest));
    await waitFor(() => expect(view.result.current.form.uploading).toBe(true));
    // The right-click remount: this instance goes away mid-upload.
    view.unmount();
    release();
    // Let the held upload settle and anything that would follow it run.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(request.mock.calls.some((c) => c[0] === "backend.uploadFile")).toBe(true);
    expect(
      request.mock.calls.some(
        (c) => c[0] === "backend.request" && (c[1] as { path: string }).path.endsWith("/generate/alpha"),
      ),
    ).toBe(false);
  });

  it("作り直されなければ送る（同じ組み立てで /generate/alpha に届く）", async () => {
    const { bridge, request } = createBridge();
    const client = createApiClient(bridge);
    const onSubmitted = vi.fn();
    const view = renderHook(() => {
      const form = useAlphaGenForm({ nativeBridge: bridge, initialIntent: intent() });
      const submit = useGenerationSubmit({ apiClient: client, onSubmitted });
      return { form, submit };
    });
    await waitFor(() => expect(view.result.current.form.mediaInfo.width).toBeGreaterThan(0));
    act(() => view.result.current.submit.submitAlpha(view.result.current.form.prepareRequest));
    await waitFor(() => expect(onSubmitted).toHaveBeenCalledWith("job-1"));
    const call = request.mock.calls.find(
      (c) => c[0] === "backend.request" && (c[1] as { path: string }).path.endsWith("/generate/alpha"),
    );
    expect(call).toBeDefined();
    expect((call![1] as { body: unknown }).body).toMatchObject({ reference_video_id: "vid-1", num_frames: 81 });
    await waitFor(() => expect(view.result.current.submit.submitState.phase).toBe("idle"));
  });
});
