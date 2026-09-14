import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BridgeError, TIMELINE_MASK_PROGRESS_EVENT } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import {
  getInpaintSlots,
  publishInpaintPartialFilter,
  publishInpaintTarget,
  resetInpaintSlots,
} from "../../timeline/inpaintSlots";
import type { SelectionItem, TimelineSelection } from "../../timeline/menuSelection";
import { useInpaintForm } from "./useInpaintForm";

/**
 * 台帳 §3-55 Inpainting のフォーム。ここで押さえたいのは 5 点:
 *  - 入力は**保管庫**から読む（`initialIntent` を一切見ない＝再マウントに強い）
 *  - 対象動画のアップロードは対象 1 つにつき 1 回きり
 *  - `window_start_sec` が**アップロード後ファイルの時間軸**であること
 *    （トリムした分を引く。ここを間違えると別の場所を描き替える）
 *  - マスクは購読を張ってから描き、トリム無しで上げる
 *  - 解像度不一致・トリム失敗・窓が置けない、をそれぞれ止める
 */

const FILE = "C:\\videos\\shot.mp4";
const MASK_FILE = "C:\\Users\\mock\\AppData\\Local\\NzVideomni\\masks\\mask_100_121_0001.mp4";

/** 素材 40 秒・リボンは 0〜599（600 フレーム = 20.0 秒 @30fps）・再生窓は素材の
 * 2.0〜22.0 秒。リボンがファイル全長を占めていないので §1-6 のトリムが立ち、
 * 切り出し開始は 2.0 秒になる。 */
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

interface BridgeOptions {
  upload?: { status: number; body: Record<string, unknown> };
  /** マスクアップロードだけ別の応答にしたいとき（既定は `upload` と共通）。 */
  maskUpload?: { status: number; body: Record<string, unknown> };
  editInfo?: { width: number; height: number; rate: number; scale: number };
  /** `timeline.renderMaskVideo` を失敗させるコード。 */
  maskError?: string;
  /** ハンドラの中から発火する進捗（本番と同じ「promise が pending の間」）。 */
  maskProgress?: Array<{ frame: number; index: number; total: number }>;
  /** 与えると `timeline.renderMaskVideo` は進捗を出したあとこれを待つ。飛行中の
   * 画面（進捗表示・`maskPhase`）を観測するための足場。 */
  hold?: Promise<void>;
}

function createBridge(options: BridgeOptions = {}) {
  const upload = options.upload ?? { status: 200, body: { video_id: "vid-1", trimmed: true } };
  const handlers = new Map<string, Set<(data: unknown) => void>>();
  let uploadCount = 0;
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "getEditInfo") {
      const info = options.editInfo ?? { width: 1920, height: 1080, rate: 30, scale: 1, midpoints: 3 };
      return { ...info, sampleRate: 44100, frame: 0, layer: 0, frameMax: 600, layerMax: 100 };
    }
    if (method === "backend.uploadFile") {
      uploadCount += 1;
      const filePath = (params as { filePath?: string }).filePath;
      if (filePath === MASK_FILE) {
        return options.maskUpload ?? { status: 200, body: { video_id: "mask-1" } };
      }
      return upload;
    }
    if (method === "timeline.renderMaskVideo") {
      if (options.maskError) throw new BridgeError(options.maskError, "forced");
      for (const push of options.maskProgress ?? []) {
        for (const handler of handlers.get(TIMELINE_MASK_PROGRESS_EVENT) ?? []) handler(push);
      }
      if (options.hold) await options.hold;
      return { filePath: MASK_FILE, width: 1920, height: 1080, frameCount: 121 };
    }
    throw new Error(`unexpected bridge call: ${method}`);
  });
  const on = vi.fn((event: string, handler: (data: unknown) => void) => {
    let set = handlers.get(event);
    if (!set) {
      set = new Set();
      handlers.set(event, set);
    }
    set.add(handler);
    return () => set?.delete(handler);
  });
  const bridge = { request, requestWithFiles: vi.fn(), on } as unknown as NativeBridge;
  return { bridge, request, on, uploads: () => uploadCount };
}

function renderForm(
  options: BridgeOptions = {},
  prompt = "a cat",
  acceleration?: AccelerationSettings,
  submitting = false,
) {
  const harness = createBridge(options);
  const view = renderHook(() =>
    useInpaintForm({
      prompt,
      nativeBridge: harness.bridge,
      submitting,
      ...(acceleration ? { acceleration } : {}),
    }),
  );
  return { ...view, ...harness };
}

/** 部分フィルタ 100..340（241 フレーム）と対象動画を両方置いた既定の状態。 */
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

describe("useInpaintForm — 保管庫から読む", () => {
  it("何も来ていなければ両方の理由が立ち、生成できない", () => {
    const { result } = renderForm();
    expect(result.current.validityReasons).toContain("partialFilterMissing");
    expect(result.current.validityReasons).toContain("targetMissing");
    expect(result.current.isValid).toBe(false);
  });

  it("再マウントしても保管庫の中身は残る（右クリック 2 回が成立する理由）", async () => {
    seedBothSlots();
    const { result, unmount } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    unmount();

    // 新しいマウント＝2 回目の右クリック後の世界。値は保管庫にあるので健在。
    const second = renderForm();
    await waitFor(() => expect(second.result.current.isValid).toBe(true));
    expect(second.result.current.targetVideoId).toBe("vid-1");
    // …そして**アップロードし直さない**（保管庫に id が残っているため）。
    expect(second.uploads()).toBe(0);
  });
});

describe("useInpaintForm — アップロード", () => {
  it("対象 1 つにつき 1 回だけ、トリム付きで上げる", async () => {
    seedBothSlots();
    const { result, request } = renderForm();
    await waitFor(() => expect(result.current.uploadStatus).toBe("ready"));

    const calls = request.mock.calls.filter(([m]) => m === "backend.uploadFile");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.[1]).toMatchObject({
      kind: "video",
      filePath: FILE,
      // §1-6: 再生窓 2.0〜22.0 秒のトリム。
      query: { trim_start_sec: "2.000", trim_duration_sec: "20.000" },
    });
  });

  it("トリムを頼んだのに適用されなければ止める", async () => {
    seedBothSlots();
    const { result } = renderForm({ upload: { status: 200, body: { video_id: "vid-1", trimmed: false } } });
    await waitFor(() => expect(result.current.trimFailed).toBe(true));
    expect(result.current.validityReasons).toContain("sourceTrimFailed");
    expect(result.current.isValid).toBe(false);
  });

  it("アップロードが失敗すれば sourceUploadFailed", async () => {
    seedBothSlots();
    const { result } = renderForm({ upload: { status: 500, body: { error: { code: "BOOM" } } } });
    await waitFor(() => expect(result.current.uploadStatus).toBe("error"));
    expect(result.current.validityReasons).toContain("sourceUploadFailed");
  });
});

describe("useInpaintForm — フレーム数と窓", () => {
  it("部分フィルタが来たらその長さを 8n+1 へ切り上げた値になる", async () => {
    seedBothSlots();
    const { result } = renderForm();
    // 100..340 は 241 フレーム。
    expect(result.current.numFrames).toBe(241);
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(result.current.window).toMatchObject({ windowStart: 100, windowEnd: 340, coversFilter: true });
  });

  it("新しい部分フィルタが来たら再シードする（対象の差し替えでは戻らない）", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));

    act(() => result.current.commitNumFrames(121));
    expect(result.current.numFrames).toBe(121);

    // 対象だけ差し替え → 打ち直した 121 のまま。
    act(() => {
      publishInpaintTarget({ item: makeItem({ layer: 4 }), selection: makeSelection(makeItem({ layer: 4 })) });
    });
    expect(result.current.numFrames).toBe(121);

    // 新しい部分フィルタ（0..8 = 9 フレーム） → 再シード。
    act(() => {
      publishInpaintPartialFilter({ layer: 5, frameStart: 0, frameEnd: 8, rate: 30, scale: 1, midpoints: 3 });
    });
    await waitFor(() => expect(result.current.numFrames).toBe(9));
  });

  it("確定時に 8n+1 の格子へ乗せ、[9,481] にクランプする", () => {
    seedBothSlots();
    const { result } = renderForm();
    act(() => result.current.setNumFrames(100, false)); // 手打ち中は素通し
    expect(result.current.numFrames).toBe(100);
    act(() => result.current.commitNumFrames(100));
    expect(result.current.numFrames).toBe(97);
    act(() => result.current.commitNumFrames(9999));
    expect(result.current.numFrames).toBe(481);
  });

  it("窓が対象動画に収まらなければ windowNotCovered で止める", async () => {
    // リボンを 100..180（81 フレーム）に縮めると、既定の 241 は置けない。
    const item = makeItem({ frameStart: 100, frameEnd: 180 });
    seedBothSlots(item);
    const { result } = renderForm();
    await waitFor(() => expect(result.current.uploadStatus).toBe("ready"));
    expect(result.current.window).toBeNull();
    expect(result.current.validityReasons).toContain("windowNotCovered");
    expect(result.current.isValid).toBe(false);
  });
});

describe("useInpaintForm — 解像度ゲート（オーナー裁定 D5）", () => {
  it("プロジェクトと対象動画が一致すれば通る", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(result.current.validityReasons).not.toContain("resolutionMismatch");
    // 送るのは 128 の倍数へ切り上げたキャンバス（1080 → 1152）。
    expect(result.current.canvasWidth).toBe(1920);
    expect(result.current.canvasHeight).toBe(1152);
  });

  it("一致しなければ止める（伸縮はしない）", async () => {
    seedBothSlots(makeItem({ mediaWidth: 1280, mediaHeight: 768 }));
    const { result } = renderForm();
    await waitFor(() => expect(result.current.validityReasons).toContain("resolutionMismatch"));
    expect(result.current.isValid).toBe(false);
  });

  it("対象動画の寸法が不明なら mediaInfoUnknown", async () => {
    seedBothSlots(makeItem({ mediaWidth: 0, mediaHeight: 0 }));
    const { result } = renderForm();
    await waitFor(() => expect(result.current.validityReasons).toContain("mediaInfoUnknown"));
    expect(result.current.validityReasons).not.toContain("resolutionMismatch");
  });

  it("部分フィルタがまだ無いうちは解像度の話をしない（F2・実機ゲート G3）", async () => {
    // 比べる相手（＝部分フィルタの解像度＝プロジェクト解像度）が揃っていない
    // 段階で「一致させてください」と言うのは早すぎる。まず言うべきは
    // `partialFilterMissing` の 1 行だけ。
    const mismatched = makeItem({ mediaWidth: 1280, mediaHeight: 768 });
    act(() => {
      publishInpaintTarget({ item: mismatched, selection: makeSelection(mismatched) });
    });
    const first = renderForm();
    await waitFor(() => expect(first.result.current.validityReasons).toContain("partialFilterMissing"));
    expect(first.result.current.validityReasons).not.toContain("resolutionMismatch");
    expect(first.result.current.validityReasons).not.toContain("mediaInfoUnknown");
    // 対象は届いているので、その理由は立たない。
    expect(first.result.current.validityReasons).not.toContain("targetMissing");
    first.unmount();

    // 寸法が読めない対象でも同じ —— 部分フィルタが来るまでは黙っている。
    resetInpaintSlots();
    const unknown = makeItem({ mediaWidth: 0, mediaHeight: 0 });
    act(() => {
      publishInpaintTarget({ item: unknown, selection: makeSelection(unknown) });
    });
    const second = renderForm();
    await waitFor(() => expect(second.result.current.validityReasons).toContain("partialFilterMissing"));
    expect(second.result.current.validityReasons).not.toContain("mediaInfoUnknown");

    // 部分フィルタが届いた瞬間に、はじめて解像度の理由が立つ。
    act(() => {
      publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    });
    await waitFor(() => expect(second.result.current.validityReasons).toContain("mediaInfoUnknown"));
  });
});

describe("useInpaintForm — マスク", () => {
  it("購読を先に張るので、promise が pending の間の進捗を取りこぼさない", async () => {
    // 本番では進捗は promise が pending の間に流れてくる。あとから購読を
    // 張る実装だと、ハンドラから発火するこの形のテストでは 1 本も届かない。
    seedBothSlots();
    let release = (): void => {};
    const hold = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { result } = renderForm({
      hold,
      maskProgress: [
        { frame: 100, index: 1, total: 241 },
        { frame: 220, index: 121, total: 241 },
      ],
    });
    await waitFor(() => expect(result.current.isValid).toBe(true));

    let pending: Promise<string | null> | null = null;
    await act(async () => {
      pending = result.current.renderAndUploadMask();
    });
    // 飛行中: 直近の進捗が画面に出ており、Generate は押せない。
    expect(result.current.maskPhase).toBe("rendering");
    expect(result.current.maskProgress).toEqual({ index: 121, total: 241 });
    expect(result.current.validityReasons).toContain("maskRendering");
    expect(result.current.isValid).toBe(false);

    let maskId: string | null = null;
    await act(async () => {
      release();
      maskId = await pending;
    });
    expect(maskId).toBe("mask-1");
    // 終わったら片付いている。
    expect(result.current.maskPhase).toBe("idle");
    expect(result.current.maskProgress).toBeNull();
  });

  it("窓の絶対フレームでマスクを頼み、トリム無しで上げる", async () => {
    seedBothSlots();
    const { result, request } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    await act(async () => {
      await result.current.renderAndUploadMask();
    });

    const render = request.mock.calls.find(([m]) => m === "timeline.renderMaskVideo");
    expect(render?.[1]).toEqual({
      layer: 5,
      frameStart: 100,
      frameEnd: 340,
      windowStart: 100,
      windowEnd: 340,
    });
    const maskUpload = request.mock.calls.find(
      ([m, p]) => m === "backend.uploadFile" && (p as { filePath?: string }).filePath === MASK_FILE,
    );
    // マスクは窓そのものなので、切り出す範囲が無い＝`query` を付けない。
    expect(maskUpload?.[1]).toEqual({ kind: "video", filePath: MASK_FILE });
  });

  it("MASK_SEED_INVALID はコードのまま運ばれ、id は返らない", async () => {
    seedBothSlots();
    const { result } = renderForm({ maskError: "MASK_SEED_INVALID" });
    await waitFor(() => expect(result.current.isValid).toBe(true));
    let maskId: string | null = "not-null";
    await act(async () => {
      maskId = await result.current.renderAndUploadMask();
    });
    expect(maskId).toBeNull();
    expect(result.current.maskErrorCode).toBe("MASK_SEED_INVALID");
    expect(result.current.maskPhase).toBe("idle");
  });

  it("publishes the busy flag AppShell refuses right-clicks on (敵対的レビュー m1)", async () => {
    seedBothSlots();
    let release = (): void => {};
    const hold = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { result } = renderForm({ hold });
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(getInpaintSlots().busy).toBe(false);

    let pending: Promise<string | null> | null = null;
    await act(async () => {
      pending = result.current.renderAndUploadMask();
    });
    expect(getInpaintSlots().busy).toBe(true);

    await act(async () => {
      release();
      await pending;
    });
    expect(getInpaintSlots().busy).toBe(false);
  });

  it("counts a submit in flight as busy too", () => {
    // マスクの描画だけでなく、生成要求の送信中も断る対象（`EditScreen` が
    // `useGenerationSubmit` の phase をそのまま渡す）。
    seedBothSlots();
    renderForm({}, "a cat", undefined, true);
    expect(getInpaintSlots().busy).toBe(true);
  });

  it("MASK_BUSY も同じ経路で戻る", async () => {
    seedBothSlots();
    const { result } = renderForm({ maskError: "MASK_BUSY" });
    await waitFor(() => expect(result.current.isValid).toBe(true));
    await act(async () => {
      await result.current.renderAndUploadMask();
    });
    expect(result.current.maskErrorCode).toBe("MASK_BUSY");
  });
});

describe("useInpaintForm — buildRequest", () => {
  it("キャンバス寸・窓長・プロジェクト fps・LoRA・inpaint ブロックを載せる", async () => {
    seedBothSlots();
    const { result } = renderForm({}, "a quiet street <lora:style-a:0.7>");
    await waitFor(() => expect(result.current.isValid).toBe(true));

    const body = result.current.buildRequest("mask-1");
    expect(body).toMatchObject({
      prompt: "a quiet street",
      width: 1920,
      height: 1152,
      num_frames: 241,
      frame_rate: 30,
      reference_video_id: "vid-1",
    });
    // 制御系 LoRA は固定で先頭、タグ由来はその後ろ。
    expect(body.loras?.[0]).toEqual({ name: "in-outpainting", strength: 1 });
    expect(body.loras?.[1]).toMatchObject({ name: "style-a", strength: 0.7 });
    // **アップロード後ファイルの時間軸**。窓はプロジェクトの 100 フレーム目
    // ＝素材の 2.0 + 100/30 秒だが、トリムで 2.0 秒を切り落としているので
    // 送る値は 100/30 = 3.333 秒。
    // のりしろは既定のままでも**常に**載る（省略してサーバー既定に落ちる経路は
    // 作らない）。完全一致で見ているので、余計なキーが増えればここで落ちる。
    expect(body.inpaint).toEqual({
      mask_video_id: "mask-1",
      window_start_sec: 3.333,
      blend_dilation_stage1: 5,
      blend_dilation_stage2: 2,
    });
  });

  it("プロンプト空欄でも生成でき、空文字で送る", async () => {
    seedBothSlots();
    const { result } = renderForm({}, "");
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(result.current.buildRequest("mask-1").prompt).toBe("");
  });

  it("加速設定が既定のままならキーを 1 つも載せない", async () => {
    seedBothSlots();
    const { result } = renderForm({}, "a cat", ACCELERATION_DEFAULTS);
    await waitFor(() => expect(result.current.isValid).toBe(true));
    const body = result.current.buildRequest("mask-1");
    expect(body).not.toHaveProperty("attention_backend");
    expect(body).not.toHaveProperty("keep_resident");
  });
});

describe("useInpaintForm — マスク周囲ののりしろ", () => {
  it("既定は 5 / 2（公式ワークフローと同じ値）", () => {
    const { result } = renderForm();
    expect(result.current.blendStage1).toBe(5);
    expect(result.current.blendStage2).toBe(2);
  });

  it("整数へ丸めて 0〜15 へクランプする（サーバーの ge=0, le=15 と同じ）", () => {
    const { result } = renderForm();
    act(() => result.current.setBlendStage1(-1));
    expect(result.current.blendStage1).toBe(0);
    act(() => result.current.setBlendStage1(16));
    expect(result.current.blendStage1).toBe(15);
    act(() => result.current.setBlendStage1(2.6));
    expect(result.current.blendStage1).toBe(3);

    act(() => result.current.setBlendStage2(-1));
    expect(result.current.blendStage2).toBe(0);
    act(() => result.current.setBlendStage2(16));
    expect(result.current.blendStage2).toBe(15);
    act(() => result.current.setBlendStage2(2.6));
    expect(result.current.blendStage2).toBe(3);
  });

  it("空欄などの数値でない入力は値を動かさない", () => {
    // `Number("")` は 0 だが `Number("abc")` は NaN。丸めの前に弾かないと
    // 画面の数字が NaN になる。
    const { result } = renderForm();
    act(() => result.current.setBlendStage1(Number.NaN));
    expect(result.current.blendStage1).toBe(5);
  });

  it("動かした値が `buildRequest` に載る", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    act(() => {
      result.current.setBlendStage1(0);
      result.current.setBlendStage2(7);
    });
    const body = result.current.buildRequest("mask-1");
    expect(body.inpaint).toMatchObject({ blend_dilation_stage1: 0, blend_dilation_stage2: 7 });
  });

  it("❌ で既定へ戻る（シードと同じ扱い）", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    act(() => {
      result.current.setBlendStage1(12);
      result.current.setBlendStage2(9);
    });
    act(() => result.current.clearAll());
    expect(result.current.blendStage1).toBe(5);
    expect(result.current.blendStage2).toBe(2);
  });
});

describe("useInpaintForm — placement と clearAll", () => {
  it("席は対象レイヤー × 窓に置く", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    expect(result.current.placement).toEqual({
      layer: 3, // 部分フィルタのレイヤー(5)ではなく、対象動画のレイヤー
      frameStart: 100,
      frameEnd: 340,
      numFrames: 241,
      genFps: 30,
    });
  });

  it("❌ は両方のスロットを捨てる", async () => {
    seedBothSlots();
    const { result } = renderForm();
    await waitFor(() => expect(result.current.isValid).toBe(true));
    act(() => result.current.clearAll());
    expect(result.current.slots.partialFilter).toBeNull();
    expect(result.current.slots.target).toBeNull();
    expect(result.current.isValid).toBe(false);
  });
});
