import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import type { TimelineSelection } from "../../timeline/menuSelection";
import { resetProvisionalReservation } from "../../timeline/provisionalReservation";
import {
  RETAKE_GLUE_HEAD_FRAMES,
  clearRetakeCarryOver,
  sameRetakeSelection,
  useRetakeForm,
} from "./useRetakeForm";

/**
 * §1-17 Retake のフォーム。ここで押さえたいのは 4 点:
 *  - スナップショットは右クリック 1 回分で固定され、以後タイムラインを追わない
 *  - 窓は**保存されない**（要求値だけが状態で、8n+1 と上下限は毎回導出される）
 *  - `window_start_sec` が**アップロード後ファイルの時間軸**であること
 *    （トリムした分を引く。ここを間違えると別の場所を撮り直す）
 *  - trimFailed が必ずブロックすること
 */

const FILE = "C:\\videos\\take1.mp4";

/** 素材 20 秒・リボンは 0〜299（300 フレーム = 10.0 秒 @30fps）・再生窓は素材の
 * 2.0〜12.0 秒。つまり §1-6 のトリムが立ち、切り出し開始は 2.0 秒。 */
function makeSelection(overrides: Partial<TimelineSelection> = {}): TimelineSelection {
  return {
    hasRange: true,
    rangeStart: 60,
    rangeEnd: 209, // 150 フレーム = 5.0 秒 @30fps
    selected: [
      {
        layer: 3,
        frameStart: 0,
        frameEnd: 299,
        effectName: "動画ファイル",
        filePath: FILE,
        objectName: "take1",
        textContent: null,
        mediaWidth: 1280,
        mediaHeight: 768,
        mediaDurationSec: 20,
        hasPlaybackRange: true,
        playbackStartSec: 2,
        playbackEndSec: 12,
      },
    ],
    cursorFrame: 60,
    cursorLayer: 3,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
    ...overrides,
  };
}

function makeIntent(selection: TimelineSelection = makeSelection(), intent = "retake"): GenerationPrefill {
  return { intent, targetMode: "edit", selection };
}

interface BridgeOptions {
  /** `backend.uploadFile` の応答。既定は「トリムが効いた」正常系。 */
  upload?: { status: number; body: Record<string, unknown> };
  /** `timeline.getSelection` の応答（`checkStale` 用）。 */
  fresh?: TimelineSelection;
}

function createBridge(options: BridgeOptions = {}) {
  const upload = options.upload ?? { status: 200, body: { video_id: "vid-1", trimmed: true } };
  const request = vi.fn(async (method: string): Promise<unknown> => {
    if (method === "backend.uploadFile") return upload;
    if (method === "timeline.getSelection") return options.fresh ?? makeSelection();
    throw new Error(`unexpected bridge call: ${method}`);
  });
  return { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
}

function renderForm(intent: GenerationPrefill | undefined, options: BridgeOptions = {}, prompt = "a cat") {
  const nativeBridge = createBridge(options);
  const view = renderHook(() =>
    useRetakeForm({ prompt, nativeBridge, ...(intent ? { initialIntent: intent } : {}) }),
  );
  return { ...view, nativeBridge };
}

/** アップロードが終わって video_id が入るまで待つ。 */
async function waitReady(result: { current: ReturnType<typeof useRetakeForm> }) {
  await waitFor(() => expect(result.current.source.state.status).toBe("ready"));
}

describe("useRetakeForm — スナップショット", () => {
  it("右クリックが無ければ案内状態（スナップショット無し・生成不可）", () => {
    const { result } = renderForm(undefined);
    expect(result.current.snapshot).toBeNull();
    expect(result.current.isValid).toBe(false);
  });

  it("Retake 以外の右クリックには反応しない（隣のパネル用ファイルを二重に上げない）", () => {
    const { result, nativeBridge } = renderForm(makeIntent(makeSelection(), "outpaint"));
    expect(result.current.snapshot).toBeNull();
    expect(nativeBridge.request).not.toHaveBeenCalled();
  });

  it("範囲が選ばれていない Retake も案内状態", () => {
    const { result } = renderForm(makeIntent(makeSelection({ hasRange: false })));
    expect(result.current.snapshot).toBeNull();
  });

  it("スナップショットは選択オブジェクトの素性とプロジェクト fps を持つ", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.snapshot).toMatchObject({
      layer: 3,
      frameStart: 0,
      filePath: FILE,
      fileName: "take1.mp4",
      projectFps: 30,
    });
  });

  it("マウント時に素材を 1 回だけ、トリム付きでアップロードする", async () => {
    const { result, nativeBridge } = renderForm(makeIntent());
    await waitReady(result);
    const uploads = (nativeBridge.request as ReturnType<typeof vi.fn>).mock.calls.filter(
      (call) => call[0] === "backend.uploadFile",
    );
    expect(uploads).toHaveLength(1);
    // 再生窓 2.0〜12.0 秒 -> trim_start_sec=2.000 / trim_duration_sec=10.000
    expect(uploads[0]?.[1]).toMatchObject({
      query: { trim_start_sec: "2.000", trim_duration_sec: "10.000" },
    });
  });
});

describe("useRetakeForm — 窓", () => {
  it("初期の窓は選択範囲を 8n+1 へ切り下げたもの", async () => {
    // 範囲 150 プロジェクトフレーム @30fps = 5.0 秒 -> 既定生成 24fps で 120 ->
    // 8n+1 へ切り下げて 113。
    const { result } = renderForm(makeIntent());
    expect(result.current.frameRate).toBe(24);
    expect(result.current.window?.ok).toBe(true);
    expect(result.current.windowFrames).toBe(113);
  });

  it("窓の頭は選択範囲の頭（素材秒 4.0 = 2.0 + 60/30）", () => {
    const { result } = renderForm(makeIntent());
    const window = result.current.window;
    expect(window?.ok && window.startSec).toBe(4);
    // RangeBand 座標では「使える素材の先頭（2.0 秒）から 2.0 秒」= 48 フレーム @24fps。
    expect(result.current.windowStartFrame).toBe(48);
  });

  it("RangeBand の帯の全幅は、実際に再生されている 10 秒ぶん", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.materialFrames).toBe(240); // 10.0 秒 × 24fps
  });

  it("setWindow は要求値を保存するだけで、8n+1 と上限は導出側が効かせる", () => {
    const { result } = renderForm(makeIntent());
    act(() => result.current.setWindow({ startFrame: 0, frames: 200 }));
    // 200 -> 上限 169 でクランプ
    expect(result.current.windowFrames).toBe(169);
    act(() => result.current.setWindow({ startFrame: 0, frames: 100 }));
    // 100 -> 8n+1 へ切り下げて 97
    expect(result.current.windowFrames).toBe(97);
    act(() => result.current.setWindow({ startFrame: 0, frames: 20 }));
    // 下限 73 まで伸ばす
    expect(result.current.windowFrames).toBe(73);
  });

  it("窓は保存されないので、fps を変えると同じ要求から作り直される", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.windowFrames).toBe(113);
    act(() => result.current.setFrameRate(30));
    // 同じ 5.0 秒の要求が 30fps では 150 -> 8n+1 で 145。
    expect(result.current.windowFrames).toBe(145);
  });

  it("窓の上下限は config から読む", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.minWindowFrames).toBe(73);
    expect(result.current.maxWindowFrames).toBe(169);
  });

  // §1-17 ⑤: 窓は stage-2 のタイル1枚で精錬されるので、タイルが短くなれば
  // 撮り直せる最長区間も縮む（8·vTile − 7 = 8·19 − 7 = 145）。
  it("Stage-2 が潜在19フレームなら窓の上限は 145 へ縮む", () => {
    const { result } = renderForm(makeIntent());
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.maxWindowFrames).toBe(145);
  });

  it("上限を超えている窓は、Stage-2 を切り替えた時点で新しい上限へ詰め直される", () => {
    const { result } = renderForm(makeIntent());
    act(() => result.current.setWindow({ startFrame: 0, frames: 200 }));
    expect(result.current.windowFrames).toBe(169);
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.windowFrames).toBe(145);
    // 戻せば元の上限まで復帰する（要求値は保存されているので、切り替えで
    // ユーザーの指定が失われることはない）。
    act(() => result.current.setStage2Window("standard"));
    expect(result.current.windowFrames).toBe(169);
  });

  // §1-19 (2026-08-11): 型レベルの露出防止ガード。バックエンドは
  // `stage2_window` に第三のプリセット `"full_length"` を持つが、Retake との
  // 併用はサーバが 422 で弾く上、`Stage2Window` を 3 値に広げると
  // `retakeMaxWindowPx` の上限計算が 169 → 481 へ化けてしまう。だから
  // `Stage2Window`（`shell/tokenBudget.ts`）は意図的に 2 値のまま —
  // 万一そこへ `"full_length"` が混ざれば、下の `@ts-expect-error` が
  // 「未使用」になって typecheck が落ちる。
  it("setStage2Window は 'full_length' を型として受理しない（§1-19の露出防止）", () => {
    const { result } = renderForm(makeIntent());
    // 型チェックのためだけの参照 —— 実際には呼ばない（"full_length" は
    // `STAGE2_WINDOW_PRESETS` に存在しないキーなので、実行すればフックが
    // クラッシュする。ここで確かめたいのは型が弾くことだけ）。
    const rejectedByTypeSystem = () => {
      // @ts-expect-error "full_length"はUIで選べる窓ではない（§1-19）
      result.current.setStage2Window("full_length");
    };
    expect(typeof rejectedByTypeSystem).toBe("function");
  });

  it("のりしろは実測の既定（25/24）を表示用に持つ", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.glueHeadFrames).toBe(RETAKE_GLUE_HEAD_FRAMES);
    expect(result.current.glueHeadFrames).toBe(25);
    expect(result.current.glueTailFrames).toBe(24);
  });
});

describe("useRetakeForm — ゲート", () => {
  it("読み込みが終われば生成できる", async () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.validityReasons).toContain("sourceUploading");
    await waitReady(result);
    expect(result.current.validityReasons).toEqual([]);
    expect(result.current.isValid).toBe(true);
  });

  it("trimFailed は必ず弾く（別の場所を撮り直す事故の防止）", async () => {
    const { result } = renderForm(makeIntent(), {
      // トリムを頼んだのに `trimmed` が返ってこない = 全長が上がっている。
      upload: { status: 200, body: { video_id: "vid-1" } },
    });
    await waitReady(result);
    expect(result.current.source.state.trimFailed).toBe(true);
    expect(result.current.validityReasons).toContain("sourceTrimFailed");
    expect(result.current.isValid).toBe(false);
  });

  it("写像できない範囲は rangeUnusable", async () => {
    // ループ再生 = タイムライン時間と素材時間が 1:1 でない。
    const selection = makeSelection();
    const item = { ...selection.selected[0]!, loopPlay: true };
    const { result } = renderForm(makeIntent({ ...selection, selected: [item] }));
    await waitReady(result);
    expect(result.current.validityReasons).toContain("rangeUnusable");
  });

  it("素材が最短の窓より短ければ outOfMaterial", async () => {
    // 再生窓 2.0〜4.0 秒 = 2.0 秒しかない（最短窓 73f @24fps = 3.04 秒）。
    const selection = makeSelection({ rangeStart: 0, rangeEnd: 59 });
    const item = { ...selection.selected[0]!, frameEnd: 59, playbackEndSec: 4 };
    const { result } = renderForm(makeIntent({ ...selection, selected: [item] }));
    await waitReady(result);
    expect(result.current.validityReasons).toContain("outOfMaterial");
    expect(result.current.isValid).toBe(false);
  });
});

describe("useRetakeForm — 解像度（④ 幅・高さの編集）", () => {
  it("初期値は素材の実寸ベースで、ユーザーが触るまで導出値に追随する", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.width).toBe(1280);
    expect(result.current.height).toBe(768);
  });

  it("setWidth/setHeight は snap=true で 64 の倍数へ丸め、snap=false は素通し", () => {
    const { result } = renderForm(makeIntent());
    act(() => result.current.setWidth(1000));
    expect(result.current.width).toBe(1024); // 1000 -> 最寄りの 64 の倍数
    act(() => result.current.setHeight(700, false));
    expect(result.current.height).toBe(700); // 手打ちはそのまま
    expect(result.current.validityReasons).toContain("dimensionsOffGrid");
    expect(result.current.isValid).toBe(false);
    act(() => result.current.setHeight(704));
    expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
  });

  it("buildRequest は実効値（override があればそれ）を送る", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    expect(result.current.buildRequest()).toMatchObject({ width: 1280, height: 768 });
    act(() => result.current.setWidth(1024));
    expect(result.current.buildRequest()).toMatchObject({ width: 1024, height: 768 });
  });

  it("素材の実寸が不明（0）なら設定の既定へ落ちる", () => {
    const selection = makeSelection();
    const item = { ...selection.selected[0]!, mediaWidth: 0, mediaHeight: 0 };
    const { result } = renderForm(makeIntent({ ...selection, selected: [item] }));
    expect(result.current.snapshot).toMatchObject({ mediaWidth: 0, mediaHeight: 0 });
    expect(result.current.width).toBeGreaterThan(0);
    expect(result.current.height).toBeGreaterThan(0);
  });
});

describe("useRetakeForm — buildRequest", () => {
  it("契約どおりの body を作る", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    act(() => result.current.setSeed(1234));
    const body = result.current.buildRequest();

    expect(body).toMatchObject({
      prompt: "a cat",
      width: 1280,
      height: 768,
      frame_rate: 24,
      seed: 1234,
      overlap_frames: 3,
      overlap_strength: 0.5,
      // 窓長の単一ソースは clips[0].num_frames。
      clips: [{ num_frames: 113 }],
      retake: { video_id: "vid-1", regenerate_audio: true },
    });
    // 糊代も stage2_window も送らない（サーバ既定に追随させる）。
    expect(body.retake).not.toHaveProperty("head_px");
    expect(body.retake).not.toHaveProperty("tail_px");
    expect(body).not.toHaveProperty("stage2_window");
    expect(body).not.toHaveProperty("source_video");
    // `chunked_upsample` も載せない（窓 ≤ 169f は一括で足りる）。
    expect(body).not.toHaveProperty("chunked_upsample");
  });

  it("stage2_window は既定（standard）では省き、潜在19フレームのときだけ載せる", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    expect(result.current.buildRequest()).not.toHaveProperty("stage2_window");
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.buildRequest()).toMatchObject({ stage2_window: "high_resolution" });
    act(() => result.current.setStage2Window("standard"));
    expect(result.current.buildRequest()).not.toHaveProperty("stage2_window");
  });

  it("window_start_sec はトリム後ファイルの時間軸（素材秒 − 切り出し開始秒）", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    // 素材秒 4.0、切り出し開始 2.0 -> アップロードしたファイルの中では 2.0 秒。
    expect(result.current.buildRequest().retake?.window_start_sec).toBe(2);
  });

  it("トリムが立たない素材では素材秒がそのまま window_start_sec になる", async () => {
    // リボンが素材全長を占め、再生開始 0 の素直なケース。
    const selection = makeSelection({ rangeStart: 60, rangeEnd: 209 });
    const item = {
      ...selection.selected[0]!,
      mediaDurationSec: 10,
      playbackStartSec: 0,
      playbackEndSec: 10,
      frameEnd: 299, // 300 フレーム @30fps = 10.0 秒 = 全長
    };
    const { result } = renderForm(makeIntent({ ...selection, selected: [item] }), {
      upload: { status: 200, body: { video_id: "vid-2" } },
    });
    await waitReady(result);
    expect(result.current.source.state.trimFailed).toBe(false); // トリクエリを送っていない
    expect(result.current.buildRequest().retake?.window_start_sec).toBe(2); // 60/30
  });

  it("音声トグルが regenerate_audio に載る（既定は映像と音声）", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    expect(result.current.regenerateAudio).toBe(true);
    act(() => result.current.setRegenerateAudio(false));
    expect(result.current.buildRequest().retake?.regenerate_audio).toBe(false);
  });

  // §3-62: 以前は「タグを外して `loras` は送らない」契約だったが、バックエンドは
  // Retake（1 クリップのチェーンジョブ）でも `loras` を素通しで受けるため、
  // Create/Chain と同じ扱いへ揃えた。
  it("プロンプトの LoRA タグは指示文から外し、loras[] として送る（Create/Chain と同じ扱い）", async () => {
    const { result } = renderForm(makeIntent(), {}, "a cat <lora:Pixar_Toon:0.8>");
    await waitReady(result);
    const body = result.current.buildRequest();
    expect(body.prompt).toBe("a cat");
    expect(body.loras).toEqual([{ name: "Pixar_Toon", strength: 0.8 }]);
  });

  it("タグが無ければ loras キー自体を載せない", async () => {
    // `renderForm` の既定プロンプト "a cat" にはタグが無い。
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    const body = result.current.buildRequest();
    expect(body.prompt).toBe("a cat");
    expect(body).not.toHaveProperty("loras");
  });

  // 制御系（IC-LoRA）を手打ちされてもクライアントは止めない。Retake には制御系の
  // 選択パネルが無く、参照動画も付けられないので、この組み合わせはバックエンドの
  // 既存 422（`LORA_REQUIRES_REFERENCE`）が断る。ここにだけ別のゲートを増やさない
  // ことがこの pin の主旨。
  it("制御系 LoRA の名前を手打ちしてもクライアント側では止めず、そのまま loras[] に載せる", async () => {
    const { result } = renderForm(makeIntent(), {}, "a cat <lora:in-outpainting:1.0>");
    await waitReady(result);
    const body = result.current.buildRequest();
    expect(body.prompt).toBe("a cat");
    expect(body.loras).toEqual([{ name: "in-outpainting", strength: 1 }]);
    expect(result.current.isValid).toBe(true);
  });
});

describe("useRetakeForm — placement（予約の打ち直し先）", () => {
  it("確定した窓をプロジェクトフレームへ戻した位置を返す", () => {
    const { result } = renderForm(makeIntent());
    // 窓の頭 = 素材 4.0 秒 = リボン頭から 2.0 秒 = プロジェクト 60 フレーム。
    // 窓長 113f @24fps = 4.7083 秒 -> @30fps で 141 プロジェクトフレーム。
    expect(result.current.placement).toEqual({
      layer: 3,
      frameStart: 60,
      frameEnd: 200,
      numFrames: 113,
      genFps: 24,
    });
  });

  it("窓を動かすと打ち直し先も動く", () => {
    const { result } = renderForm(makeIntent());
    act(() => result.current.setWindow({ startFrame: 0, frames: 73 }));
    // 使える素材の先頭（素材 2.0 秒）= リボン頭 = プロジェクト 0 フレーム。
    expect(result.current.placement?.frameStart).toBe(0);
    expect(result.current.placement?.numFrames).toBe(73);
  });

  // ③ 読み出し行はこの写像を**ドラッグ中のプレビュー値**に対しても呼ぶので、
  // 確定値（placement）と同じ関数から出ていることが要点。
  it("toProjectRange は placement と同じ写像で、任意の窓に対して答える", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.toProjectRange(result.current.windowStartFrame, result.current.windowFrames)).toEqual({
      frameStart: result.current.placement!.frameStart,
      frameEnd: result.current.placement!.frameEnd,
    });
    // 素材頭から 0 フレーム・73 フレームぶん = プロジェクト 0〜90（73/24 秒 × 30fps）。
    expect(result.current.toProjectRange(0, 73)).toEqual({ frameStart: 0, frameEnd: 90 });
  });

  it("素材カード用の秒数は「実際に再生されている長さ」（ファイル全長 20 秒ではない）", () => {
    const { result } = renderForm(makeIntent());
    expect(result.current.materialSeconds).toBe(10);
    expect(result.current.materialFrames).toBe(240);
  });
});

describe("useRetakeForm — checkStale", () => {
  it("タイムラインが変わっていなければ注意文は出ない", async () => {
    const { result } = renderForm(makeIntent());
    await act(async () => {
      await result.current.checkStale();
    });
    expect(result.current.stale).toBe(false);
  });

  it("範囲が動いていたら注意文が立つ（ただし生成はブロックしない）", async () => {
    const { result } = renderForm(makeIntent(), { fresh: makeSelection({ rangeStart: 90, rangeEnd: 239 }) });
    await waitReady(result);
    await act(async () => {
      await result.current.checkStale();
    });
    expect(result.current.stale).toBe(true);
    expect(result.current.isValid).toBe(true); // ← ゲートには影響しない
  });

  it("`timeline.getSelection` が 1 回しか呼ばれない", async () => {
    const { result, nativeBridge } = renderForm(makeIntent());
    await act(async () => {
      await result.current.checkStale();
    });
    const calls = (nativeBridge.request as ReturnType<typeof vi.fn>).mock.calls.filter(
      (call) => call[0] === "timeline.getSelection",
    );
    expect(calls).toHaveLength(1);
  });

  it("取得に失敗しても静かに何もしない", async () => {
    const failing = {
      request: vi.fn(async (method: string) => {
        if (method === "backend.uploadFile") return { status: 200, body: { video_id: "v", trimmed: true } };
        throw new Error("boom");
      }),
      requestWithFiles: vi.fn(),
      on: vi.fn(() => () => {}),
    } as unknown as NativeBridge;
    const { result } = renderHook(() => useRetakeForm({ nativeBridge: failing, initialIntent: makeIntent() }));
    await act(async () => {
      await result.current.checkStale();
    });
    expect(result.current.stale).toBe(false);
  });
});

/**
 * ❌（片付け）と 🔁（別の場所を撮り直す）。要点は 3 つ:
 *  - ❌ はすべてを既定へ戻す（設定も、🔁 の持ち越しも）
 *  - 🔁 は設定を残して「素材待ち」にし、**次の 1 マウントにだけ**持ち越す
 *  - 持ち越しに**幅・高さは入らない**（新しい素材の実寸に追随させるため・
 *    オーナー確定①）
 *
 * 持ち越しはモジュール状態なので、各テストの後で必ず捨てる —— 残すと、次の
 * テストが「既定から始まる」前提を静かに失う。
 */
describe("useRetakeForm — ❌ / 🔁", () => {
  afterEach(() => {
    clearRetakeCarryOver();
    resetProvisionalReservation();
  });

  /** 素材 640×384 の別オブジェクト（🔁 のあとで送る「次の素材」）。 */
  function smallIntent(): GenerationPrefill {
    const selection = makeSelection();
    const item = { ...selection.selected[0]!, mediaWidth: 640, mediaHeight: 384 };
    return makeIntent({ ...selection, selected: [item] });
  }

  it("❌ は素材も区間も設定も、すべて既定へ戻す", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    act(() => {
      result.current.setFrameRate(30);
      result.current.setSeed(1234);
      result.current.setRegenerateAudio(false);
      result.current.setStage2Window("high_resolution");
      result.current.setWidth(1024);
    });

    act(() => result.current.clearAll());

    expect(result.current.snapshot).toBeNull();
    expect(result.current.mapping).toBeNull();
    expect(result.current.awaitingSource).toBe(false); // 素材待ちではなく案内状態
    expect(result.current.source.state.status).toBe("idle");
    expect(result.current.frameRate).toBe(24);
    expect(result.current.seed).toBe(-1);
    expect(result.current.regenerateAudio).toBe(true);
    expect(result.current.stage2Window).toBe("standard");
    // 素材が無いので、幅・高さは設定の既定（1280×768）へ戻る。
    expect(result.current.width).toBe(1280);
    expect(result.current.isValid).toBe(false);
  });

  it("🔁 は素材と区間だけを捨て、設定は値のまま「素材待ち」になる", async () => {
    const { result } = renderForm(makeIntent());
    await waitReady(result);
    act(() => {
      result.current.setFrameRate(30);
      result.current.setSeed(1234);
      result.current.setStage2Window("high_resolution");
    });

    act(() => result.current.resetSource());

    expect(result.current.snapshot).toBeNull();
    expect(result.current.awaitingSource).toBe(true);
    expect(result.current.source.state.status).toBe("idle");
    expect(result.current.frameRate).toBe(30);
    expect(result.current.seed).toBe(1234);
    expect(result.current.stage2Window).toBe("high_resolution");
    // 待機中の表示凍結: 素材が消えても、いま出ていた実効値のまま動かない。
    expect(result.current.width).toBe(1280);
    expect(result.current.height).toBe(768);
  });

  it("🔁 のあと次の右クリックで設定が戻る（ただし幅・高さは新しい素材の実寸）", async () => {
    const first = renderForm(makeIntent());
    await waitReady(first.result);
    act(() => {
      first.result.current.setFrameRate(30);
      first.result.current.setSeed(1234);
      first.result.current.setRegenerateAudio(false);
      first.result.current.setStage2Window("high_resolution");
      // 幅も動かしておく —— これが**持ち越されない**ことが要点。
      first.result.current.setWidth(1024);
    });
    act(() => first.result.current.resetSource());
    first.unmount();

    // 次の右クリック = 新しいマウント（`AppShell` が remount させる）。
    const second = renderForm(smallIntent());
    expect(second.result.current.frameRate).toBe(30);
    expect(second.result.current.seed).toBe(1234);
    expect(second.result.current.regenerateAudio).toBe(false);
    expect(second.result.current.stage2Window).toBe("high_resolution");
    // 幅・高さは持ち越さない: 新しい素材の実寸（640×384）から導かれる。
    expect(second.result.current.width).toBe(640);
    expect(second.result.current.height).toBe(384);
  });

  it("持ち越しは一度きり（次の次の右クリックは既定から始まる）", async () => {
    const first = renderForm(makeIntent());
    await waitReady(first.result);
    act(() => first.result.current.setFrameRate(30));
    act(() => first.result.current.resetSource());
    first.unmount();

    const second = renderForm(makeIntent());
    expect(second.result.current.frameRate).toBe(30); // 1 回目は戻る
    second.unmount();

    const third = renderForm(makeIntent());
    expect(third.result.current.frameRate).toBe(24); // 2 回目はもう戻らない
  });

  it("❌ は持ち越しも捨てる（🔁 のあとで気が変わったとき）", async () => {
    const first = renderForm(makeIntent());
    await waitReady(first.result);
    act(() => first.result.current.setFrameRate(30));
    act(() => first.result.current.resetSource());
    act(() => first.result.current.clearAll());
    first.unmount();

    const second = renderForm(makeIntent());
    expect(second.result.current.frameRate).toBe(24);
  });

  it("Outpainting の右クリックは持ち越しを消費しない（隣のパネルは無関係）", async () => {
    const first = renderForm(makeIntent());
    await waitReady(first.result);
    act(() => first.result.current.setFrameRate(30));
    act(() => first.result.current.resetSource());
    first.unmount();

    // 同じ `initialIntent` は両サブパネルへ届くので、Outpainting の右クリックで
    // ここが持ち越しを食べてしまうと、次の Retake が既定から始まってしまう。
    const outpaint = renderForm(makeIntent(makeSelection(), "outpaint"));
    expect(outpaint.result.current.snapshot).toBeNull();
    expect(outpaint.result.current.frameRate).toBe(24);
    outpaint.unmount();

    const retake = renderForm(makeIntent());
    expect(retake.result.current.frameRate).toBe(30);
  });
});

describe("sameRetakeSelection", () => {
  it("撮り直す場所を決めている値が同じなら同じ", () => {
    const a = makeSelection();
    const b = makeSelection({ cursorFrame: 999, sampleRate: 48000 });
    expect(sameRetakeSelection(a, b)).toBe(true);
  });

  it("範囲・オブジェクトの素性が変われば違う", () => {
    const base = makeSelection();
    expect(sameRetakeSelection(makeSelection({ rangeStart: 61 }), base)).toBe(false);
    expect(sameRetakeSelection(makeSelection({ rangeEnd: 210 }), base)).toBe(false);
    expect(sameRetakeSelection(makeSelection({ hasRange: false }), base)).toBe(false);
    const moved = makeSelection();
    moved.selected = [{ ...base.selected[0]!, frameStart: 5 }];
    expect(sameRetakeSelection(moved, base)).toBe(false);
    const other = makeSelection();
    other.selected = [{ ...base.selected[0]!, filePath: "C:\\videos\\other.mp4" }];
    expect(sameRetakeSelection(other, base)).toBe(false);
  });

  it("選択が空になったら違う", () => {
    const empty = makeSelection();
    empty.selected = [];
    expect(sameRetakeSelection(empty, makeSelection())).toBe(false);
  });
});
