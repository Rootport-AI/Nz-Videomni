import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import type { TimelineSelection } from "../../timeline/menuSelection";
import {
  getReservationPhase,
  publishRetakeReservation,
  releaseIfSettled,
  reservePlacement,
  rollbackReservedPlacement,
} from "../../timeline/provisionalReservation";
import { EditScreen } from "./EditScreen";
import { clearRetakeCarryOver } from "./useRetakeForm";

/**
 * §1-17 Retake の通し。右クリック（`retakeRange`）→ Retake サブタブが開く →
 * 素材が自動で読み込まれる → Generate → `POST /generate/chain` に契約どおりの
 * body が飛ぶ、までを 1 本のブリッジ越しに見る。`request.mock.calls` がその回の
 * 完全な記録になるので、送っていないこと（糊代・stage2_window）も同じ場所で
 * 確かめられる。
 *
 * 予約席（`timeline/provisionalReservation.ts`）はモジュールレベルの状態なので、
 * 各テストの後で idle へ戻す。戻さないと、次のテストの `bindToJob` が
 * 「席が空いていない」で無言の no-op になり、緑のまま何も検証しなくなる。
 */

/** ジョブ台帳のスタブ。`hasActiveJob` だけはテストごとに切り替えたい（busy
 * ガード）ので、可変の 1 箱越しに読ませる。`vi.hoisted` なのは `vi.mock` の
 * ファクトリが巻き上げられて先に走るため。 */
const jobsStub = vi.hoisted(() => ({ hasActiveJob: false }));

vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({
    jobs: [],
    hasActiveJob: jobsStub.hasActiveJob,
    cancellingIds: new Set<string>(),
    deletingIds: new Set<string>(),
    cancelJob: () => Promise.resolve(),
    deleteJob: () => Promise.resolve(),
  }),
}));

const FILE = "C:\\videos\\take1.mp4";

/** 素材 20 秒・リボン 0〜299（10.0 秒 @30fps）・再生窓は素材 2.0〜12.0 秒。
 * 選択範囲は 60〜209（150 フレーム = 5.0 秒）。 */
function makeSelection(overrides: Partial<TimelineSelection> = {}): TimelineSelection {
  return {
    hasRange: true,
    rangeStart: 60,
    rangeEnd: 209,
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

const RETAKE_INTENT: GenerationPrefill = {
  intent: "retake",
  targetMode: "edit",
  selection: makeSelection(),
};

interface BridgeOptions {
  /** `POST /generate/chain` の応答。既定は 202 受理。 */
  chainResponse?: { status: number; body: unknown };
}

function createBridge(options: BridgeOptions = {}) {
  const chainResponse = options.chainResponse ?? {
    status: 202,
    body: { job_id: "job-1", status: "queued", created_at: "2026-08-10T00:00:00Z", num_clips: 1 },
  };
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "backend.uploadFile") return { status: 200, body: { video_id: "vid-1", trimmed: true } };
    if (method === "timeline.getSelection") return makeSelection();
    if (method === "timeline.insertProvisional") {
      return { inserted: true, layer: 9, frame: 60, objectName: "x", placedLayer: 9, placedFrame: 60, usedFallback: false };
    }
    if (method === "timeline.updateProvisionalReservation") {
      return { updated: true, placedLayer: 9, placedFrame: 60, usedFallback: false };
    }
    if (method === "timeline.deleteProvisionalByJob") return { deleted: true };
    if (method === "backend.request") {
      const { method: httpMethod, path } = params as { method: string; path: string };
      if (httpMethod === "GET" && path.endsWith("/config")) return { status: 200, body: CONFIG_BODY };
      if (httpMethod === "GET" && path.endsWith("/loras")) return { status: 200, body: { loras: [] } };
      if (httpMethod === "POST" && path.endsWith("/generate/chain")) return chainResponse;
    }
    throw new Error(`unexpected bridge call: ${method} ${JSON.stringify(params)}`);
  });
  return { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
}

/** `GET /config` の最小形。窓の上下限がここから読まれることを見るために、
 * わざと F1 の既定と同じ 73/169 を返している（違う値を返す専用テストは別途）。 */
const CONFIG_BODY = {
  generation_presets: {},
  generation_defaults: { width: 1280, height: 768, crop_output: null, num_frames: 257, frame_rate: 24, seed: -1 },
  limits: {
    max_width: 4096,
    max_height: 4096,
    max_num_frames: 481,
    max_conditioning_images: 5,
    conditioning_frame_idx_multiple: 8,
    conditioning_keyframe_grid_offset: 1,
    phase1_max_concurrent_jobs: 1,
    spill_free_frames: { "1280x768": 257 },
    v2v_context_frames_default: 73,
    v2v_context_frames_min: 25,
    v2v_context_frames_max: 145,
    retake_window_min_frames: 73,
    retake_window_max_frames: 169,
  },
  upload: {
    max_image_size_mb: 20,
    allowed_image_extensions: [".png"],
    max_video_size_mb: 200,
    allowed_video_extensions: [".mp4"],
    max_audio_size_mb: 50,
    allowed_audio_extensions: [".wav"],
  },
};

/** `null` = 右クリック無し（手動でタブを開いただけ）。既定値ではなく明示的な
 * `null` にしてあるのは、`undefined` を渡すと JS の既定引数が効いてしまい、
 * 「右クリック無し」のつもりのテストが実は Retake 意図つきで走るため。 */
function renderRetake(intent: GenerationPrefill | null = RETAKE_INTENT, options: BridgeOptions = {}) {
  const nativeBridge = createBridge(options);
  const onJobSubmitted = vi.fn();
  const view = render(
    <LanguageProvider>
      <EditScreen
        {...(intent ? { initialIntent: intent } : {})}
        prompt="a cat"
        nativeBridge={nativeBridge}
        onJobSubmitted={onJobSubmitted}
      />
    </LanguageProvider>,
  );
  const request = nativeBridge.request as ReturnType<typeof vi.fn>;
  return { ...view, nativeBridge, request, onJobSubmitted };
}

/** その回に飛んだ `POST /generate/chain` の body。 */
function chainBody(request: ReturnType<typeof vi.fn>): Record<string, unknown> | undefined {
  const call = request.mock.calls.find(
    (c) =>
      c[0] === "backend.request" &&
      (c[1] as { path?: string }).path?.endsWith("/generate/chain") === true,
  );
  return call ? ((call[1] as { body: Record<string, unknown> }).body) : undefined;
}

function bridgeCalls(request: ReturnType<typeof vi.fn>, method: string) {
  return request.mock.calls.filter((c) => c[0] === method);
}

afterEach(async () => {
  // 席を idle へ戻す（reserved なら rollback、bind 済みなら release）。
  releaseIfSettled("job-1");
  publishRetakeReservation(null);
  await rollbackReservedPlacement(createBridge());
  // 🔁 の持ち越しもモジュール状態なので、残すと次のテストへ漏れる。
  clearRetakeCarryOver();
  jobsStub.hasActiveJob = false;
});

describe("EditScreen — Retake の通し", () => {
  it("retakeRange の右クリックで Retake サブタブが開き、素材が自動で読み込まれる", async () => {
    const { request } = renderRetake();
    const tabs = screen.getAllByRole("tab");
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");

    await waitFor(() => expect(bridgeCalls(request, "backend.uploadFile").length).toBeGreaterThan(0));
    // タイムライン上で一部だけを使っているので、その範囲だけを切り出して上げる。
    // NOTE: 件数は 1 と決め打たない —— `EditScreen` は両サブパネルを常時マウント
    // するので、同じ `initialIntent` を見る `useOutpaintForm` の自動読み込みも
    // 同時に走り、同じファイルがもう一度上がる（`useOutpaintForm.ts` は本作業の
    // 立入禁止領域なので、ここでは事実として受け入れて記録するに留める）。
    for (const call of bridgeCalls(request, "backend.uploadFile")) {
      expect(call[1]).toMatchObject({ query: { trim_start_sec: "2.000", trim_duration_sec: "10.000" } });
    }
    // 素材名が Retake パネル**の中**に出ている（同じ右クリックを見ている
    // Outpainting パネルにも同名が出るので、必ずパネル内へ絞る）。
    const panel = document.querySelector<HTMLElement>(".retake-panel")!;
    expect(await within(panel).findByText(/take1\.mp4/)).toBeTruthy();
    // 素材を**差し替える**導線はこのパネルには無い。あるのは片付けの 2 つ
    // （❌＝この撮り直しごと捨てる／🔁＝素材と区間だけ捨てて設定は残す）と、
    // シードの 2 つだけ。全数一致で見るので、増えたら必ずここが落ちる。
    expect(within(panel).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual([
      en.edit.retake.clearButton,
      en.edit.retake.resetSourceButton,
      "Randomize seed",
      "Reuse last seed",
    ]);
  });

  it("区間バーと注意文が出る", async () => {
    renderRetake();
    expect(screen.getByRole("group", { name: "Section to retake" })).toBeTruthy();
    // 主表示はタイムライン上の位置（1 始まり）。窓の頭はプロジェクト 60 フレーム、
    // 尻尾は 200 フレーム（`useRetakeForm` の placement テストと同じ算術）なので
    // 表示は 61〜201。末尾には 8n+1 の断り書きが付く（2026-08-10 オーナー指示）。
    expect(screen.getByTestId("range-band-placement").textContent).toBe(
      "Redoing frames 61 to 201 of the timeline. (Window lengths are always a multiple of 8, plus 1.)",
    );
    // 副表示は窓 113 フレーム（5.0 秒の選択を 24fps で 120 -> 8n+1 で 113）= 4.71 秒。
    expect(screen.getByTestId("range-band-readout").textContent).toBe("(a 113-frame window, 4.71s)");
    // 3 本の注意はいつも出る。
    expect(screen.getByText(/becomes a different take/)).toBeTruthy();
    expect(screen.getByText(/decoded and re-encoded once/)).toBeTruthy();
    expect(screen.getByText(/invent sound for it/)).toBeTruthy();
    // のりしろは説明のみ（つまみは無い）。「変更できない」の一文は 2026-08-10 に削除。
    const glue = screen.getByText(/25 frames at the front and 24 at the back/);
    expect(glue.textContent).toBe("The join keeps 25 frames at the front and 24 at the back.");
  });

  it("素材カードに秒・フレーム数・fps換算・解像度が出る", async () => {
    renderRetake();
    const panel = document.querySelector<HTMLElement>(".retake-panel")!;
    // 実際に再生されている 10.00 秒（ファイル全長の 20 秒ではない）を、パネルの
    // FPS 欄（既定 24）で 240 フレームへ換算。解像度は素材の実寸。
    expect(
      within(panel).getByText("take1.mp4 — 10.00s, 240 frames (at 24fps), 1280 x 768"),
    ).toBeTruthy();
  });

  it("素材の解像度が不明（0）なら解像度は出さない", async () => {
    const selection = makeSelection();
    const item = { ...selection.selected[0]!, mediaWidth: 0, mediaHeight: 0 };
    renderRetake({ intent: "retake", targetMode: "edit", selection: { ...selection, selected: [item] } });
    const panel = document.querySelector<HTMLElement>(".retake-panel")!;
    expect(within(panel).getByText("take1.mp4 — 10.00s, 240 frames (at 24fps)")).toBeTruthy();
  });

  it("Generate で契約どおりの body が飛ぶ", async () => {
    const user = userEvent.setup();
    const { request } = renderRetake();
    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await waitFor(() => expect(chainBody(request)).toBeDefined());
    const body = chainBody(request)!;
    expect(body).toMatchObject({
      prompt: "a cat",
      width: 1280,
      height: 768,
      frame_rate: 24,
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: [{ num_frames: 113 }],
      retake: {
        video_id: "vid-1",
        // 素材秒 4.0 − 切り出し開始 2.0 = アップロードしたファイルの中で 2.0 秒。
        window_start_sec: 2,
        regenerate_audio: true,
      },
    });
    // 送らないもの（サーバ既定に追随させる / 排他）。
    expect(body).not.toHaveProperty("stage2_window");
    expect(body).not.toHaveProperty("source_video");
    expect(body).not.toHaveProperty("source_audio");
    expect(body).not.toHaveProperty("reference_video_id");
    expect(body.retake).not.toHaveProperty("head_px");
    expect(body.retake).not.toHaveProperty("tail_px");
  });

  it("Generate 押下時に checkStale → 予約の打ち直し → 送信 の順で走る", async () => {
    const user = userEvent.setup();
    const { request } = renderRetake();
    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await waitFor(() => expect(chainBody(request)).toBeDefined());

    const order = request.mock.calls.map((c) =>
      c[0] === "backend.request" ? `backend.request ${(c[1] as { path: string }).path}` : (c[0] as string),
    );
    const stale = order.indexOf("timeline.getSelection");
    const place = order.findIndex((m) => m === "timeline.insertProvisional" || m === "timeline.updateProvisionalReservation");
    const submit = order.findIndex((m) => m.endsWith("/generate/chain"));
    expect(stale).toBeGreaterThanOrEqual(0);
    expect(place).toBeGreaterThan(stale);
    expect(submit).toBeGreaterThan(place);

    // 打ち直しは**確定した窓**の頭と長さで行う（選択範囲の長さではない）。
    const placeCall = request.mock.calls[place]![1] as Record<string, unknown>;
    expect(placeCall).toMatchObject({ numFrames: 113, genFps: 24, materialFrameStart: 60 });
  });

  it("受理されたら予約をジョブへ渡す（bindToJob）", async () => {
    const user = userEvent.setup();
    const { request, onJobSubmitted } = renderRetake();
    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await waitFor(() => expect(onJobSubmitted).toHaveBeenCalledWith("job-1"));
    await waitFor(() => {
      const binds = bridgeCalls(request, "timeline.updateProvisionalReservation").filter(
        (c) => (c[1] as { newJobId?: string }).newJobId === "job-1",
      );
      expect(binds).toHaveLength(1);
    });
  });

  it("同期的な失敗（422）では予約を片付ける（rollback）", async () => {
    const user = userEvent.setup();
    const { request } = renderRetake(RETAKE_INTENT, {
      chainResponse: { status: 422, body: { error: { code: "VALIDATION_ERROR", message: "nope" } } },
    });
    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await waitFor(() => expect(bridgeCalls(request, "timeline.deleteProvisionalByJob")).toHaveLength(1));
    // 席が残ったままにならない = 以後の生成がブロックされない。
    expect(bridgeCalls(request, "timeline.updateProvisionalReservation")).toHaveLength(0);
  });

  it("手動でタブを開いただけなら案内文だけで、生成ボタンは出ない", () => {
    renderRetake(null);
    expect(screen.getByText(/Right-click the object on the timeline/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Redo the selected range" })).toBeNull();
  });

  // ── ① busy ガード（オーナー目視 2026-08-10） ─────────────────────────────
  it("サーバが busy の間は Generate が押せず、ラベルが busy 表記になる", async () => {
    jobsStub.hasActiveJob = true;
    renderRetake();
    const button = await screen.findByRole("button", { name: "Busy…" });
    expect(button).toBeDisabled();
    // 撮り直しの通常ラベルは出ていない（= ボタンが 2 つある訳ではない）。
    expect(screen.queryByRole("button", { name: "Redo the selected range" })).toBeNull();
  });

  // ── ④ 幅・高さ ────────────────────────────────────────────────────────
  it("幅・高さは編集でき、送信値もそれに従う（「AviUtl2から取得」は出さない）", async () => {
    const user = userEvent.setup();
    const { request } = renderRetake();
    const panel = document.querySelector<HTMLElement>(".retake-panel")!;
    expect(within(panel).queryByRole("button", { name: /AviUtl2/ })).toBeNull();

    // スライダーは 2 本（幅・高さ）。素材実寸そのままが初期値。
    const sliders = within(panel).getAllByRole("slider").filter((el) => el.tagName === "INPUT");
    expect(sliders.map((el) => (el as HTMLInputElement).value)).toEqual(["1280", "768"]);
    const [widthRange] = sliders as HTMLInputElement[];
    fireEvent.change(widthRange!, { target: { value: "1024" } });

    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await waitFor(() => expect(chainBody(request)).toBeDefined());
    expect(chainBody(request)).toMatchObject({ width: 1024, height: 768 });
  });

  // ── ⑤ Stage-2 クリップ長 ──────────────────────────────────────────────
  it("Stage-2 クリップ長は Chained と同じ 2 択で、潜在19フレームなら stage2_window を送る", async () => {
    const user = userEvent.setup();
    const { request } = renderRetake();
    const panel = document.querySelector<HTMLElement>(".retake-panel")!;
    const select = within(panel).getByRole("combobox") as HTMLSelectElement;
    // 文言は `strings.chained.stage2Window` の直読み（Retake 専用キーは作らない）。
    expect([...select.options].map((o) => o.textContent)).toEqual([
      en.chained.stage2Window.standardOption,
      en.chained.stage2Window.highResolutionOption,
    ]);
    expect(select.value).toBe("standard");

    await user.selectOptions(select, "high_resolution");
    const button = await screen.findByRole("button", { name: "Redo the selected range" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await waitFor(() => expect(chainBody(request)).toBeDefined());
    expect(chainBody(request)).toMatchObject({ stage2_window: "high_resolution" });
  });
});

/**
 * 素材カードの ❌（この撮り直しごと片付ける）と 🔁（素材と区間だけ捨てて、設定は
 * 残したまま次の右クリックを待つ）。フック単体の検証は `useRetakeForm.test.tsx`
 * 側にあるので、ここで見るのは**画面がどう変わるか**（案内文・生成ボタン・
 * ⏳予約リボン・並び順）だけ。
 */
describe("EditScreen — Retake の ❌ / 🔁", () => {
  const panelOf = () => document.querySelector<HTMLElement>(".retake-panel")!;

  it("❌ を押すと案内状態へ戻り、生成ボタンも消える", async () => {
    const user = userEvent.setup();
    renderRetake();
    await user.click(within(panelOf()).getByRole("button", { name: en.edit.retake.clearButton }));

    expect(within(panelOf()).getByText(en.edit.retake.idle)).toBeTruthy();
    // 素材待ちの案内ではない（＝設定欄も残っていない）。
    expect(within(panelOf()).queryByText(en.edit.retake.awaitingSource)).toBeNull();
    expect(within(panelOf()).queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("button", { name: "Redo the selected range" })).toBeNull();
  });

  it("🔁 を押すと素材待ちになり、設定は入力値のまま残る（生成ボタンは消える）", async () => {
    const user = userEvent.setup();
    renderRetake();
    // 既定から動かしておく（残っていることを見るため）。
    const fps = within(panelOf()).getByLabelText(en.single.duration.fps);
    fireEvent.change(fps, { target: { value: "30" } });
    await user.selectOptions(within(panelOf()).getByRole("combobox"), "high_resolution");

    await user.click(within(panelOf()).getByRole("button", { name: en.edit.retake.resetSourceButton }));

    expect(within(panelOf()).getByText(en.edit.retake.awaitingSource)).toBeTruthy();
    expect((within(panelOf()).getByLabelText(en.single.duration.fps) as HTMLInputElement).value).toBe("30");
    expect((within(panelOf()).getByRole("combobox") as HTMLSelectElement).value).toBe("high_resolution");
    // 押せるものが無いので生成群は消える（`EditScreen` の `snapshot !== null` ゲート）。
    expect(screen.queryByRole("button", { name: "Redo the selected range" })).toBeNull();
    // 捨てる素材はもう無いので、素材待ちでは 🔁 を出さない（❌ だけ）。
    expect(within(panelOf()).queryByRole("button", { name: en.edit.retake.resetSourceButton })).toBeNull();
    expect(within(panelOf()).getByRole("button", { name: en.edit.retake.clearButton })).toBeTruthy();
  });

  it("❌ は自分が置いた ⏳予約リボンを片付ける", async () => {
    const user = userEvent.setup();
    const { nativeBridge, request } = renderRetake();
    // 席を **本当に** reserved にしてから押す —— idle のままだと rollback は
    // no-op で、何も検証していない緑になる。publish は `AppShell` の系統D が
    // やっていることの写し（❌ が「自分の席か」を見分ける唯一の手がかり）。
    const reserved = await reservePlacement(nativeBridge, {
      placement: "D",
      material: { layer: 3, frameStart: 60, frameEnd: 209 },
      numFrames: 113,
      genFps: 24,
      displayText: "x",
    });
    publishRetakeReservation(reserved.pendingId);

    await user.click(within(panelOf()).getByRole("button", { name: en.edit.retake.clearButton }));

    await waitFor(() => {
      const deletes = bridgeCalls(request, "timeline.deleteProvisionalByJob");
      expect(deletes.map((c) => (c[1] as { jobId: string }).jobId)).toEqual([reserved.pendingId]);
    });
    expect(getReservationPhase()).toBe("idle");
  });

  it("席が他所へ移っていたら ❌ は予約に触らない（他タブのリボンを消さない）", async () => {
    const user = userEvent.setup();
    const { nativeBridge, request } = renderRetake();
    const mine = await reservePlacement(nativeBridge, {
      placement: "D",
      material: { layer: 3, frameStart: 60, frameEnd: 209 },
      numFrames: 113,
      genFps: 24,
      displayText: "x",
    });
    publishRetakeReservation(mine.pendingId);
    // Create 側の右クリック相当: 同じ 1 席が**移動**し、新しい pending id が振られる
    // （publish 値は古いままなので、以後一致しない）。
    const moved = await reservePlacement(nativeBridge, {
      placement: "C",
      cursor: { layer: 0, frame: 0 },
      numFrames: 97,
      genFps: 24,
      displayText: "y",
    });
    expect(moved.moved).toBe(true);
    expect(moved.pendingId).not.toBe(mine.pendingId);

    await user.click(within(panelOf()).getByRole("button", { name: en.edit.retake.clearButton }));

    expect(bridgeCalls(request, "timeline.deleteProvisionalByJob")).toHaveLength(0);
    expect(getReservationPhase()).toBe("reserved");
  });

  it("設定欄の並びは 対象 → 幅 → 高さ → fps → シード → Stage-2（注意はその後ろ）", () => {
    renderRetake();
    const labels = [...panelOf().querySelectorAll(".field-label")].map((el) => el.textContent);
    expect(labels).toEqual([
      en.edit.retake.sourceHeading,
      en.edit.retake.audioHeading,
      en.edit.retake.audioBoth,
      en.edit.retake.audioVideoOnly,
      en.single.size.width,
      en.single.size.height,
      en.single.duration.fps,
      en.single.seed.label,
      en.chained.stage2Window.label,
      en.edit.retake.noticesHeading,
    ]);
  });
});
