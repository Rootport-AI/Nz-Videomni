import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { ToastProvider } from "../../shell/ToastContext";
import { Toasts } from "../../shell/Toasts";
import { en } from "../../i18n/strings";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import {
  publishInpaintPartialFilter,
  publishInpaintTarget,
  resetInpaintSlots,
} from "../../timeline/inpaintSlots";
import type { SelectionItem, TimelineSelection } from "../../timeline/menuSelection";
import {
  publishRetakeReservation,
  releaseIfSettled,
  rollbackReservedPlacement,
} from "../../timeline/provisionalReservation";
import { EditScreen } from "./EditScreen";

/**
 * 台帳 §3-55 Inpainting の通し（画面側）。右クリック 2 種でサブタブが開き →
 * Generate を押すと「マスクを描く → 上げる → 席を取る → `POST /generate`」の
 * 順で動く、までを 1 本のブリッジ越しに見る。`request.mock.calls` がその回の
 * 完全な記録になるので、**順序**も、送っていないことも同じ場所で確かめられる。
 *
 * 窓の算術そのものは `inpaintWindow.test.ts`、フックの規則は
 * `useInpaintForm.test.tsx` が持つ。ここはその 2 つが画面として繋がっていることを見る。
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

const FILE = "C:\\videos\\shot.mp4";
const MASK_FILE = "C:\\masks\\mask_100_241_0001.mp4";

/** 素材 40 秒・リボン 0〜599（20.0 秒 @30fps）・再生窓は素材 2.0〜22.0 秒。 */
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

function intentFor(intent: string): GenerationPrefill {
  return { intent, targetMode: "edit", selection: makeSelection() };
}

interface BridgeOptions {
  /** `POST /generate` の応答。既定は 202 受理。 */
  generateResponse?: { status: number; body: unknown };
  /** マスク描画を失敗させるコード。 */
  maskError?: string;
}

function createBridge(options: BridgeOptions = {}) {
  const generateResponse = options.generateResponse ?? {
    status: 202,
    body: { job_id: "job-1", status: "queued", created_at: "2026-09-14T00:00:00Z" },
  };
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
      if (options.maskError) {
        const { BridgeError } = await import("../../bridge");
        throw new BridgeError(options.maskError, "forced");
      }
      return { filePath: MASK_FILE, width: 1920, height: 1080, frameCount: 241 };
    }
    if (method === "timeline.getSelection") return makeSelection();
    if (method === "timeline.insertProvisional") {
      return {
        inserted: true,
        layer: 9,
        frame: 100,
        objectName: "x",
        placedLayer: 9,
        placedFrame: 100,
        usedFallback: false,
      };
    }
    if (method === "timeline.updateProvisionalReservation") {
      return { updated: true, placedLayer: 9, placedFrame: 100, usedFallback: false };
    }
    if (method === "timeline.deleteProvisionalByJob") return { deleted: true };
    if (method === "backend.request") {
      const { method: httpMethod, path } = params as { method: string; path: string };
      if (httpMethod === "GET" && path.endsWith("/config")) return { status: 200, body: CONFIG_BODY };
      if (httpMethod === "GET" && path.endsWith("/loras")) return { status: 200, body: { loras: [] } };
      if (httpMethod === "POST" && path.endsWith("/generate")) return generateResponse;
    }
    throw new Error(`unexpected bridge call: ${method} ${JSON.stringify(params)}`);
  });
  return { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
}

const CONFIG_BODY = {
  generation_presets: {},
  generation_defaults: { width: 1920, height: 1088, crop_output: null, num_frames: 121, frame_rate: 30, seed: -1 },
  limits: {
    max_width: 4096,
    max_height: 4096,
    max_num_frames: 481,
    max_conditioning_images: 10,
    conditioning_frame_idx_multiple: 8,
    conditioning_keyframe_grid_offset: 1,
    phase1_max_concurrent_jobs: 1,
    spill_free_frames: {},
    v2v_context_frames_default: 73,
    v2v_context_frames_min: 25,
    v2v_context_frames_max: 145,
    retake_window_min_frames: 73,
    retake_window_max_frames: 169,
    comfort_budgets: { ltx: { outpaint_budget: 42240 } },
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

function renderInpaint(
  intent: GenerationPrefill | null = intentFor("inpaint-mask"),
  options: BridgeOptions = {},
  subTabsDisabled?: { retake: boolean; outpainting: boolean; inpainting: boolean },
) {
  const nativeBridge = createBridge(options);
  const onJobSubmitted = vi.fn();
  // F3: マスクの失敗はパネルの 1 行ではなくトーストで出る。本番の入れ子
  // （`AppShell`）と同じく `ToastProvider` で包み、スタックそのもの
  // （`shell/Toasts.tsx`）も一緒に出して、実際に読める文字列として確かめる。
  const view = render(
    <LanguageProvider>
      <ToastProvider>
        <EditScreen
          {...(intent ? { initialIntent: intent } : {})}
          prompt="a quiet street"
          nativeBridge={nativeBridge}
          onJobSubmitted={onJobSubmitted}
          engineFamily="ltx"
          {...(subTabsDisabled ? { subTabsDisabled } : {})}
        />
        <Toasts />
      </ToastProvider>
    </LanguageProvider>,
  );
  const request = nativeBridge.request as ReturnType<typeof vi.fn>;
  return { ...view, nativeBridge, request, onJobSubmitted };
}

/** その回に飛んだ `POST /generate` の body。 */
function generateBody(request: ReturnType<typeof vi.fn>): Record<string, unknown> | undefined {
  const call = request.mock.calls.find(
    (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path?.endsWith("/generate") === true,
  );
  return call ? (call[1] as { body: Record<string, unknown> }).body : undefined;
}

/** 両方のスロットを埋める（右クリック 2 回ぶん）。 */
function seedBothSlots(item: SelectionItem = makeItem()) {
  publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
  publishInpaintTarget({ item, selection: makeSelection(item) });
}

const panelEl = () => document.querySelector<HTMLElement>(".inpaint-panel")!;

beforeEach(() => {
  resetInpaintSlots();
});

afterEach(async () => {
  releaseIfSettled("job-1");
  publishRetakeReservation(null);
  await rollbackReservedPlacement(createBridge());
  resetInpaintSlots();
  jobsStub.serverBusy = false;
});

describe("EditScreen — Inpainting サブタブの昇格と振り分け", () => {
  it("inpaint-mask の右クリックで Inpainting サブタブが開く", () => {
    seedBothSlots();
    renderInpaint(intentFor("inpaint-mask"));
    expect(screen.getByRole("tab", { name: "Inpainting" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: /^inpainting/i, hidden: true })).toBeVisible();
  });

  it("inpaint-target の右クリックでも同じサブタブが開く", () => {
    seedBothSlots();
    renderInpaint(intentFor("inpaint-target"));
    expect(screen.getByRole("tab", { name: "Inpainting" })).toHaveAttribute("aria-selected", "true");
  });

  it("灰色なら並び順で最初に生きているサブタブへ逃げる", () => {
    // D11: LTX 2.5 では Inpainting が灰色。押せないタブの下に生成群を出さない。
    seedBothSlots();
    renderInpaint(intentFor("inpaint-mask"), {}, { retake: false, outpainting: false, inpainting: true });
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: /^inpainting/i, hidden: true })).not.toBeVisible();
  });

  it("右クリックがまだ 1 つも無くても案内文つきで開ける", () => {
    renderInpaint(intentFor("inpaint-mask"));
    const panel = panelEl();
    expect(within(panel).getByText(en.edit.inpainting.idle)).toBeTruthy();
    expect(within(panel).getByText(en.edit.inpainting.partialFilterNone)).toBeTruthy();
    expect(within(panel).getByText(en.edit.inpainting.targetNone)).toBeTruthy();
  });
});

describe("EditScreen — Inpainting パネルの中身", () => {
  it("部分フィルタはレイヤー・フレーム範囲・中間点の枚数だけを 1 始まりで出す（マスクは見せない）", async () => {
    seedBothSlots();
    renderInpaint();
    const panel = panelEl();
    // 設計正本 §3.1 の 3 つ: レイヤー 5 → 6、100..340 → 101..341、中間点 3 個。
    expect(within(panel).getByText(en.edit.inpainting.partialFilterReadout(6, 101, 341, 3))).toBeTruthy();
    // マスク動画の**ファイル名も置き場所も**、どこにも出ない（オーナー裁定
    // D7）。「mask」という語自体は説明文に出るので、出てはならないのは
    // ファイルの実体を指す文字列の方。
    await waitFor(() => expect(within(panel).getByText(/shot\.mp4/)).toBeTruthy());
    expect(panel.textContent).not.toContain("mask_");
    expect(panel.textContent).not.toContain(".mp4\\");
    expect(panel.textContent).not.toContain("masks");
  });

  it("対象動画カードには片付けの ❌ とシードの 2 つしかボタンが無い", async () => {
    seedBothSlots();
    renderInpaint();
    const panel = panelEl();
    await waitFor(() => expect(within(panel).getByText(/shot\.mp4/)).toBeTruthy());
    // 素材を差し替える導線はこのパネルに無い（窓はその場所に対して測った量）。
    expect(within(panel).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual([
      en.edit.inpainting.clearButton,
      "Randomize seed",
      "Reuse last seed",
    ]);
  });

  it("マスク周囲ののりしろは有効な欄で、シードの後・時間軸のモックの前に出る", () => {
    seedBothSlots();
    renderInpaint();
    const panel = panelEl();
    const blend = within(panel).getByRole("group", { name: en.edit.inpainting.blend.heading });

    // 下のモックと同じ見た目でも、こちらは押せる（既定は 5 / 2）。
    const inputs = within(blend).getAllByRole("spinbutton") as HTMLInputElement[];
    expect(inputs).toHaveLength(2);
    for (const input of inputs) expect(input).not.toBeDisabled();
    expect(inputs[0]?.value).toBe("5");
    expect(inputs[1]?.value).toBe("2");

    // 並びはオーナー指定どおり「シード → マスク周囲ののりしろ → 時間軸の
    // のりしろ（モック）」。索引ではなく DOM の前後で見る。
    const seedButton = within(panel).getByRole("button", { name: en.single.seed.randomTooltip });
    const glue = within(panel).getByRole("group", { name: en.edit.inpainting.glueHeading });
    expect(seedButton.compareDocumentPosition(blend) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(blend.compareDocumentPosition(glue) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("対象動画があれば注記に実寸が出る（1280×768・Stage-2＝2 なら 58px）", () => {
    // 帯幅は Stage-2 で決まる（`VERIFICATION_LOG.md` §105.3 の実測）:
    // 2 × 1280 ÷ 64 ＝ 40、＋18px ＝ 58px。
    seedBothSlots(makeItem({ mediaWidth: 1280, mediaHeight: 768 }));
    renderInpaint();
    const blend = within(panelEl()).getByRole("group", { name: en.edit.inpainting.blend.heading });
    const note = within(blend).getByText(en.edit.inpainting.blend.note(58));
    expect(note.textContent).toContain("58px");
  });

  it("対象動画が無ければ数字抜きの注記を出す", () => {
    // 長辺が測れていないのに px を出すと、0px という嘘の数字になる。
    renderInpaint();
    const blend = within(panelEl()).getByRole("group", { name: en.edit.inpainting.blend.heading });
    expect(within(blend).getByText(en.edit.inpainting.blend.noteUnknown)).toBeTruthy();
  });

  it("のりしろのモックは全部 disabled で、既定は「なし」", () => {
    seedBothSlots();
    renderInpaint();
    const panel = panelEl();
    const glue = within(panel).getByRole("group", { name: en.edit.inpainting.glueHeading });
    const inputs = within(glue).getAllByRole("radio") as HTMLInputElement[];
    expect(inputs).toHaveLength(2);
    for (const input of inputs) expect(input).toBeDisabled();
    // 既定は「なし」＝2 本目。
    expect(inputs[0]?.checked).toBe(false);
    expect(inputs[1]?.checked).toBe(true);
    // 数値欄も含めて、この塊で押せるものは 1 つも無い。
    const number = within(glue).getByRole("spinbutton");
    expect(number).toBeDisabled();
    expect(within(glue).getByText(en.edit.inpainting.glueNote)).toBeTruthy();
  });

  it("窓が部分フィルタより短ければ注意文を出す（ブロックはしない）", async () => {
    seedBothSlots();
    const { request } = renderInpaint();
    await waitFor(() => expect(request.mock.calls.some(([m]) => m === "backend.uploadFile")).toBe(true));
    const panel = panelEl();
    // 既定（241）は覆っているので出ていない。
    expect(within(panel).queryByText(en.edit.inpainting.windowNotCoveredNote)).toBeNull();

    const user = userEvent.setup();
    const number = within(panel).getAllByRole("spinbutton")[0] as HTMLInputElement;
    await user.clear(number);
    await user.type(number, "121");
    number.blur();
    await waitFor(() =>
      expect(within(panel).getByText(en.edit.inpainting.windowNotCoveredNote)).toBeTruthy(),
    );
    // 注意文であってブロックではない —— Generate はまだ押せる。
    expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).not.toBeDisabled();
  });

  it("解像度が一致しなければ実数入りの理由でゲートする（オーナー裁定 D5）", async () => {
    seedBothSlots(makeItem({ mediaWidth: 1280, mediaHeight: 768 }));
    renderInpaint();
    await waitFor(() =>
      expect(
        screen.getByText(en.edit.inpainting.generateReasons.resolutionMismatch(1920, 1080, 1280, 768)),
      ).toBeTruthy(),
    );
    expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).toBeDisabled();
  });
});

describe("EditScreen — Inpainting の Generate", () => {
  it("マスク描画 → アップロード → 席 → 送信 の順に動く", async () => {
    seedBothSlots();
    const { request, onJobSubmitted } = renderInpaint();
    await waitFor(() => expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).not.toBeDisabled());

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: en.edit.inpainting.generateButton }));
    await waitFor(() => expect(onJobSubmitted).toHaveBeenCalledWith("job-1"));

    const order = request.mock.calls
      .map(([method, params]) => {
        if (method === "backend.uploadFile") {
          return (params as { filePath?: string }).filePath === MASK_FILE ? "uploadMask" : "uploadTarget";
        }
        if (method === "backend.request") {
          const path = (params as { path: string }).path;
          return path.endsWith("/generate") ? "generate" : null;
        }
        if (method === "timeline.renderMaskVideo") return "renderMask";
        if (method === "timeline.insertProvisional") return "reserve";
        return null;
      })
      .filter((name) => name !== null);
    expect(order).toEqual(["uploadTarget", "renderMask", "uploadMask", "reserve", "generate"]);

    // 席は**対象動画のレイヤー**×窓に置く。
    const reserve = request.mock.calls.find(([m]) => m === "timeline.insertProvisional");
    expect(reserve?.[1]).toMatchObject({ materialLayer: 3, materialFrameStart: 100, materialFrameEnd: 340 });

    const body = generateBody(request);
    expect(body).toMatchObject({
      prompt: "a quiet street",
      width: 1920,
      height: 1152,
      num_frames: 241,
      frame_rate: 30,
      reference_video_id: "vid-1",
      inpaint: { mask_video_id: "mask-1", window_start_sec: 3.333 },
    });
    const loras = (body?.loras ?? []) as Array<{ name: string; strength: number }>;
    expect(loras[0]).toEqual({ name: "in-outpainting", strength: 1 });
    // 画角拡張のブロックは載らない（別の機能）。
    expect(body).not.toHaveProperty("outpaint");
  });

  it("マスクが作れなければ席も取らず送信もしない", async () => {
    seedBothSlots();
    const { request, onJobSubmitted } = renderInpaint(intentFor("inpaint-mask"), {
      maskError: "MASK_SEED_INVALID",
    });
    await waitFor(() => expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).not.toBeDisabled());

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: en.edit.inpainting.generateButton }));

    // 「もう一度右クリックしてください」が**トースト**で出て、そこで止まる
    // （F3。実機ゲート G9 でパネルの 1 行は見落とされた）。
    await waitFor(() =>
      expect(within(document.querySelector<HTMLElement>(".toast-stack")!).getByText(en.edit.inpainting.seedGone))
        .toBeTruthy(),
    );
    // パネル側には残っていない（案内は 1 か所だけ）。
    expect(within(panelEl()).queryByText(en.edit.inpainting.seedGone)).toBeNull();
    expect(request.mock.calls.some(([m]) => m === "timeline.insertProvisional")).toBe(false);
    expect(generateBody(request)).toBeUndefined();
    expect(onJobSubmitted).not.toHaveBeenCalled();
  });

  it("送信が失敗したら席を巻き戻す", async () => {
    seedBothSlots();
    const { request } = renderInpaint(intentFor("inpaint-mask"), {
      generateResponse: { status: 422, body: { error: { code: "INPAINT_SOURCE_MISMATCH", message: "nope" } } },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).not.toBeDisabled());

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: en.edit.inpainting.generateButton }));

    await waitFor(() => expect(screen.getByText("INPAINT_SOURCE_MISMATCH")).toBeTruthy());
    // 未バインドの席を idle へ戻す（放置すると以後の生成が永久にブロックされる）。
    await waitFor(() =>
      expect(request.mock.calls.some(([m]) => m === "timeline.deleteProvisionalByJob")).toBe(true),
    );
  });

  it("サーバーが塞がっていればボタンだけ止める", async () => {
    jobsStub.serverBusy = true;
    seedBothSlots();
    renderInpaint();
    const button = screen.getByRole("button", { name: en.single.busyButton });
    expect(button).toBeDisabled();
    // フォームは触れるまま（Create/Chain/Retake と同じ作法）。
    expect(within(panelEl()).getAllByRole("spinbutton")[0]).not.toBeDisabled();
  });

  it("puts the comfort-ceiling tick on the frames slider (敵対的レビュー M1)", async () => {
    // 快適上限は「助言」だが、**目盛りが無ければ助言にならない**。線は配信値
    // 42,240 を 1920×1152 のキャンバスで割り戻した値で、`DurationField` の
    // `<datalist>` 1 点として出る。
    seedBothSlots();
    renderInpaint();
    const panel = panelEl();
    await waitFor(() => expect(panel.querySelector("datalist option")).not.toBeNull());
    const option = panel.querySelector<HTMLOptionElement>("datalist option")!;
    // floor(1920/32) * floor(1152/32) = 60*36 = 2160 セル → floor(42240/2160) = 19
    // 潜在枚 → 8*(19-1)+1 = 145 フレーム。
    expect(option.value).toBe("145");
    // スライダーがその datalist を参照していなければ、目盛りは描かれない。
    const slider = within(panel).getAllByRole("slider")[0] as HTMLInputElement;
    expect(slider.getAttribute("list")).toBe(panel.querySelector("datalist")!.id);
  });

  it("shows the three prompt notes the spike settled on, in order", () => {
    // F4: 「マスクの外に既にある物を書くと複製される恐れ」の 1 本は、実機ゲート
    // 後のオーナー裁定で落とした（en/ja 両方の文言ごと削除）。
    seedBothSlots();
    renderInpaint();
    const notes = within(panelEl()).getByRole("list");
    expect(within(notes).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      en.edit.inpainting.promptNotes.coverWholeSubject,
      en.edit.inpainting.promptNotes.writePositively,
      en.edit.inpainting.promptNotes.blurIgnored,
    ]);
  });

  it("reports a target with no backing file as a load failure, not as 'loading' (敵対的レビュー m3)", async () => {
    // `filePath: null` の対象は待っても ready にならない。以前は「読み込み中」の
    // まま固まり、理由の行も `sourceUploading` で止まっていた。
    seedBothSlots(makeItem({ filePath: null }));
    const { request } = renderInpaint();
    await waitFor(() =>
      expect(screen.getByText(en.edit.inpainting.generateReasons.sourceUploadFailed)).toBeTruthy(),
    );
    expect(screen.queryByText(en.edit.inpainting.generateReasons.sourceUploading)).toBeNull();
    // 上げようがないので、アップロードは 1 回も走らない。
    expect(request.mock.calls.some(([m]) => m === "backend.uploadFile")).toBe(false);
  });
});

