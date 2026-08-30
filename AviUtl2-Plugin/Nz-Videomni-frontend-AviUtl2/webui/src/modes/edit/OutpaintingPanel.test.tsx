import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { EditScreen } from "./EditScreen";

// The seed field reads the job ledger via `useJobsContext` for its ♻ seed-reuse
// button; stub it exactly the way `GenerationForm.accordion.test.tsx` and
// `CommonGenerationFields.test.tsx` already do, so this unit test needs no
// `JobsProvider`. The four extra members are the ones `jobs/JobLedger` reads —
// the generation column mounts it (see `renderPanel` below). An empty `jobs`
// array makes the ledger render its "no jobs yet" line, so the cancel/delete
// halves are never actually invoked; they exist so the destructuring finds them.
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

const SOURCE_PATH = "C:\\videos\\clip.mp4";
const SOURCE_NAME = "clip.mp4";

interface BridgeOptions {
  /** `fs.probeMediaInfo` output. 1265x720 is the design notes' worked example
   * (§5-B): neither side is a multiple of 128, so the canvas starts off the
   * grid and the pads have to be dialled in before Generate opens. */
  media?: { durationSec: number; width: number; height: number };
  /** 2 本目の素材（差し替えを試すテスト用）。指定すると 2 回目以降の 📁 が別の
   * `video_id` を返し、`fs.probeMediaInfo` もこの寸法を答える —— 同じ id では
   * probe が再実行されない（`probedIdRef`）ので、id ごと変える必要がある。 */
  swapMedia?: { durationSec: number; width: number; height: number };
  /** Rows `GET /loras` reports. Defaults to the pinned control adapter present. */
  loras?: Array<{ name: string; kind: "style" | "control" }>;
}

/** A hand-built `NativeBridge` covering exactly the calls this panel makes: the
 * file picker, the upload, the media probe, and the two backend routes
 * (`GET /loras` through `useLoras`, `POST /generate` on submit). Everything the
 * panel does goes through this one object, so `request.mock.calls` is a complete
 * transcript of its behaviour. */
function createPanelBridge(options: BridgeOptions = {}) {
  const media = options.media ?? { durationSec: 8, width: 1265, height: 720 };
  const loras = options.loras ?? [
    { name: "Pixar_Toon", kind: "style" as const },
    { name: "in-outpainting", kind: "control" as const },
  ];
  // 何回目の 📁 か。`swapMedia` が指定されたときだけ意味を持つ。
  let picks = 0;
  const swapped = () => options.swapMedia !== undefined && picks > 1;
  const request = vi.fn(async (method: string, params: unknown): Promise<unknown> => {
    if (method === "ui.pickFile") {
      picks += 1;
      return { filePath: SOURCE_PATH, fileName: SOURCE_NAME };
    }
    if (method === "backend.uploadFile") return { status: 200, body: { video_id: swapped() ? "vid-2" : "vid-1" } };
    if (method === "fs.probeMediaInfo") return swapped() ? options.swapMedia : media;
    if (method === "backend.request") {
      const { method: httpMethod, path } = params as { method: string; path: string };
      if (httpMethod === "GET" && path.endsWith("/loras")) {
        return {
          status: 200,
          body: { loras: loras.map((l) => ({ ...l, has_thumbnail: false, exists: true, source: "config" })) },
        };
      }
      if (httpMethod === "POST" && path.endsWith("/generate")) {
        return { status: 202, body: { job_id: "job-1", status: "queued", created_at: "2026-08-09T00:00:00Z" } };
      }
    }
    throw new Error(`unexpected bridge call: ${method}`);
  });
  const bridge = {
    request,
    requestWithFiles: vi.fn(),
    on: vi.fn(() => () => {}),
  } as unknown as NativeBridge;
  return { bridge, request };
}

interface RenderOptions {
  initialIntent?: GenerationPrefill;
  /** The SHARED prompt (`AppShell`'s `PromptBar`). The panel has no prompt box
   * of its own — production threads this in through `EditScreen`. */
  prompt?: string;
  onJobSubmitted?: (jobId: string) => void;
}

/** A routed payload that carries NO material — just enough to put `EditScreen`
 * on the Outpainting sub-tab. `selected: []` means the mount-time auto-load
 * finds no file path and returns without uploading, so every test below still
 * attaches (or deliberately does not attach) its own source. */
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

/** Renders the panel through `EditScreen`, its production host.
 *
 * 2026-08-09 (2カラム化): the Generate button, the submit-error row and the
 * "why can't I press Generate?" note moved OUT of `OutpaintingPanel` into
 * `EditScreen`'s `.generation-column`, and the form state moved up with them
 * (the screen owns `useOutpaintForm`, exactly as `SingleScreen` owns
 * `useGenerationForm`). The panel is therefore no longer meaningfully
 * renderable on its own — driving it through the screen is what keeps these
 * tests covering the whole Outpainting flow rather than half of it.
 *
 * Production wraps every mode screen in a `role="tabpanel"` div
 * (`shell/AppShell.tsx`); Edit's own SUB-panels deliberately carry no such role
 * (see `EditSubTabs`'s doc comment). Mirroring the shell's wrapper here keeps
 * the app-wide `within(screen.getByRole("tabpanel"))` scoping convention usable
 * in this file too. */
function renderPanel(bridge: NativeBridge, options: RenderOptions = {}) {
  render(
    <LanguageProvider>
      <div role="tabpanel">
        <EditScreen
          nativeBridge={bridge}
          initialIntent={options.initialIntent ?? OUTPAINT_TAB_INTENT}
          prompt={options.prompt}
          onJobSubmitted={options.onJobSubmitted}
        />
      </div>
    </LanguageProvider>,
  );
  return within(screen.getByRole("tabpanel"));
}

/** The prompt every submitting test uses. Any non-empty string clears the
 * `promptEmpty` gate — the point is that it arrives as a PROP. */
const PROMPT = "a wide city street";

type Panel = ReturnType<typeof renderPanel>;

/** A range input's current value, as the DOM reports it. Read directly rather
 * than through `toHaveValue`, which normalizes differently per input type. */
function sliderValue(panel: Panel, name: RegExp): string {
  return (panel.getByRole("slider", { name }) as HTMLInputElement).value;
}

/** 上下左右のパッド。テストが流し込む値の型。 */
type PadValues = { left: number; right: number; top: number; bottom: number };

/** The pads most tests below want: the amounts that take the 1265x720 fixture to
 * a legal 1280x768 canvas.
 *
 * 2026-08-11 まではこれが「アライメントの下限」として自動で入っていた値だった。
 * パッドが 0 起点になった今、同じキャンバスを得るには自分で入れる必要がある——
 * だから `attachSource` の既定にしてある。下流の期待値（1280x768、ブレンド帯
 * 100px、送信 body の pad 4 値）はそれで無傷のまま、テストの主題を「格子合わせ」
 * ではなく本来の題目に保てる。 */
const DEFAULT_PADS: PadValues = { left: 7, right: 8, top: 24, bottom: 24 };

/** Types the four pads in through the sliders. `fireEvent.change` mirrors a
 * drag; the value arrives at `setPad` exactly as the browser would send it.
 *
 * **センタリングが切であることが前提**（§5-D）: 入のあいだは 1 辺を打つたびに対辺が
 * 同値で上書きされるため、4 辺を順に流すこのヘルパーでは後に流した辺が勝ってしまう。
 * 入の側のテストは辺ごとに単発の `fireEvent.change` を打つこと。 */
function setPads(panel: Panel, pads: Partial<PadValues>) {
  for (const [side, value] of Object.entries(pads)) {
    fireEvent.change(panel.getByRole("slider", { name: new RegExp(`^${side}$`, "i") }), {
      target: { value: String(value) },
    });
  }
}

/** Picks the source video through the 📁 button, waits for the probe to land —
 * the pad sliders only exist once the source has been measured — and dials in
 * `pads`. Pass `null` to leave every pad at its 0 starting point. */
async function attachSource(
  panel: Panel,
  user: ReturnType<typeof userEvent.setup>,
  pads: PadValues | null = DEFAULT_PADS,
) {
  await user.click(panel.getByRole("button", { name: /choose source video/i }));
  await waitFor(() => expect(panel.getByRole("slider", { name: /^left$/i })).toBeInTheDocument());
  if (pads) setPads(panel, pads);
}

describe("OutpaintingPanel", () => {
  it("blocks Generate and explains why while nothing is attached", async () => {
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);

    expect(panel.getByRole("button", { name: /^generate$/i })).toBeDisabled();
    const note = panel.getByRole("note");
    expect(within(note).getByText(/fill the main prompt/i)).toBeInTheDocument();
    expect(within(note).getByText(/attach the video you want to extend/i)).toBeInTheDocument();

    // The preview is mounted from the start — as an EMPTY FRAME with no copy in
    // it (owner, 2026-08-09: the old "the preview appears once…" line is gone),
    // so the panel does not jump when a source finally lands.
    expect(panel.queryByText(/the preview appears once a source video has been read/i)).not.toBeInTheDocument();
    expect(panel.queryByRole("img")).not.toBeInTheDocument();

    // No pad slider yet — there is no measured source to size the pads against.
    expect(panel.queryByRole("slider", { name: /^left$/i })).not.toBeInTheDocument();
    // The pinned adapter IS installed in this fixture, so once `GET /loras`
    // lands its line must disappear from the reasons.
    await waitFor(() =>
      expect(within(panel.getByRole("note")).queryByText(/in-outpainting/)).not.toBeInTheDocument(),
    );
  });

  it("reports the missing control adapter when GET /loras does not list it", async () => {
    const { bridge } = createPanelBridge({ loras: [{ name: "Pixar_Toon", kind: "style" }] });
    const panel = renderPanel(bridge);

    expect(await within(panel.getByRole("note")).findByText(/in-outpainting/)).toBeInTheDocument();
  });

  it("shows the four pad sliders, starting at 0 and stepping one pixel, once a source is measured", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    // 2026-08-11: 自動で足される分は無い。どの辺も 0 から始まる。
    expect(sliderValue(panel, /^left$/i)).toBe("0");
    expect(sliderValue(panel, /^right$/i)).toBe("0");
    expect(sliderValue(panel, /^top$/i)).toBe("0");
    expect(sliderValue(panel, /^bottom$/i)).toBe("0");
    // 0 起点・1px 刻み —— 矢印キー 1 回が 1px であることをこの 2 つが固定する。
    expect(panel.getByRole("slider", { name: /^left$/i })).toHaveAttribute("min", "0");
    expect(panel.getByRole("slider", { name: /^left$/i })).toHaveAttribute("step", "1");
    // 第2弾: スライダーの上限は定数 220（入力補助としての可動域）。元動画の寸法
    // にも対辺のパッドにも依存しない —— それが依存していた頃は可動域が約 2800 に
    // なり、マウスでは決して指定できない値が生まれていた。
    expect(panel.getByRole("slider", { name: /^left$/i })).toHaveAttribute("max", "220");
    // 正確な値・大きな値の入り口は数値ボックスで、上限は API のパッド上限 4096。
    expect(panel.getByRole("spinbutton", { name: /^left$/i })).toHaveAttribute("max", "4096");
    expect(panel.getByRole("spinbutton", { name: /^left$/i })).toHaveAttribute("min", "0");
    // したがってキャンバスは元動画そのままの寸法から始まる（128 の倍数でなくても
    // よい。生成が止まるだけで、表示は嘘をつかない）。「Generating at」を含めて
    // 探すのは、素材カードの読み出し（"1265 x 720 px, 8.0s"）と区別するため。
    expect(panel.getByText(/Generating at 1265 x 720 px/)).toBeInTheDocument();
    // 必須の説明文（128の倍数 / ブレンド帯）が出ていること。ブレンド帯の幅は
    // このキャンバス（長辺 1265）と既定の 5 段から 99px。
    expect(panel.getByText(/must both be multiples of 128 pixels/i)).toBeInTheDocument();
    expect(panel.getByText(/about 99 px inward from the edge of the extension/i)).toBeInTheDocument();
    // The preview reads out both sizes for assistive technology.
    expect(panel.getByRole("img", { name: /1265 by 720 pixels/i })).toBeInTheDocument();
  });

  it("keeps a value that is NOT a multiple of 128 exactly as it was set", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    // 「上10px・下20px」——本改修の動機そのもの。どこにも丸められない。
    setPads(panel, { top: 10, bottom: 20 });
    expect(sliderValue(panel, /^top$/i)).toBe("10");
    expect(sliderValue(panel, /^bottom$/i)).toBe("20");
    setPads(panel, { left: 7, right: 1 });
    expect(sliderValue(panel, /^left$/i)).toBe("7");
    expect(sliderValue(panel, /^right$/i)).toBe("1");
    // 720 + 10 + 20 = 750、1265 + 7 + 1 = 1273 —— どちらも 128 の倍数ではないが、
    // 表示はその通りの数字を出す。
    expect(panel.getByText(/1273 x 750 px/)).toBeInTheDocument();
  });

  it("has no alignment radios any more", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user);

    // 初期配置（左詰め/中央/右詰め・上詰め/中央/下詰め）は 2026-08-11 に撤去。
    // Retake の映像/音声ラジオは同時マウントだが `hidden` の中なので、`byRole`
    // の既定（アクセシブルな要素だけ）には現れない。
    expect(panel.queryByRole("radio")).not.toBeInTheDocument();
  });

  it("takes a typed pad value from the number box, unsnapped, and clamps only at the ends", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    const leftBox = panel.getByRole("spinbutton", { name: /^left$/i });
    fireEvent.change(leftBox, { target: { value: "7" } });
    // 打った値がそのまま。スライダーにも同じ値が乗る（同じ 1 つの状態）。
    expect((leftBox as HTMLInputElement).value).toBe("7");
    expect(sliderValue(panel, /^left$/i)).toBe("7");

    // 範囲外だけが端で止まる。上限は定数 4096（API のパッド上限）—— 第2弾で対辺
    // 連動を廃止したので、元動画の幅も対辺の値もここには効かない。
    fireEvent.change(leftBox, { target: { value: "99999" } });
    expect((leftBox as HTMLInputElement).value).toBe("4096");
    fireEvent.change(leftBox, { target: { value: "-40" } });
    expect((leftBox as HTMLInputElement).value).toBe("0");
  });

  it("pins the slider thumb at its own max while the number box keeps the real value", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    // 220 を超える値はボックスからしか入らない（`setPads` はスライダー経由なので
    // 220 に丸められてしまい、この挙動の検証にならない）。
    const leftBox = panel.getByRole("spinbutton", { name: /^left$/i }) as HTMLInputElement;
    fireEvent.change(leftBox, { target: { value: "500" } });

    // state は 500 のまま。ボックスは実値を出し続ける。
    expect(leftBox.value).toBe("500");
    // スライダーのつまみは自分の上限（220）に貼りついて見える —— `value > max` の
    // range 入力を表示上 max へ丸める、HTML 本来の挙動をそのまま使っている。
    expect(sliderValue(panel, /^left$/i)).toBe("220");
    // 送られるのは state の 500 のほうであることを、キャンバス表示で確かめる
    // （1265 + 500 = 1765）。つまみの見た目に引きずられて値が変わってはいない。
    expect(panel.getByText(/Generating at 1765 x 720 px/)).toBeInTheDocument();
  });

  it("blocks Generate with SEPARATE width and height lines, each clearing on its own", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user, null);

    // 1265x720 —— 幅も高さも 128 の倍数ではない。2 行が別々に出る（1 行にまとめ
    // ないのが仕様: どちらを直せばよいか目で追えるようにするため）。
    const note = () => panel.getByRole("note");
    expect(within(note()).getByText(/the width must be a multiple of 128/i)).toBeInTheDocument();
    expect(within(note()).getByText(/the height must be a multiple of 128/i)).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^generate$/i })).toBeDisabled();

    // 高さだけ合わせる（720 + 24 + 24 = 768）と、高さの行だけが消える。
    setPads(panel, { top: 24, bottom: 24 });
    expect(within(note()).queryByText(/the height must be a multiple of 128/i)).not.toBeInTheDocument();
    expect(within(note()).getByText(/the width must be a multiple of 128/i)).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^generate$/i })).toBeDisabled();

    // 幅も合わせる（1265 + 7 + 8 = 1280）と両方消え、生成できるようになる。
    setPads(panel, { left: 7, right: 8 });
    // 全部解消すると理由の note ごと消えるので、パネル全体から探す。
    expect(panel.queryByText(/must be a multiple of 128/i)).not.toBeInTheDocument();
    await waitFor(() => expect(panel.getByRole("button", { name: /^generate$/i })).toBeEnabled());
  });

  it("spells out the 4096 ceiling in the reason note when the pads push past it", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user, null);

    // 1265 + 3000 = 4265 —— 4096 超。数値ボックスから入れる（220 を超える値は
    // スライダー経由では入らないため）。
    fireEvent.change(panel.getByRole("spinbutton", { name: /^left$/i }), { target: { value: "3000" } });

    // 文言まで見るのは `GenerateReasonsNote` が「表に無いコードは黙って読み飛ばす」
    // 部品だからである。`editReasonMessages.ts` の配線が将来外れると、理由コード
    // 自体は出ているのに行だけが消える —— コード名だけを見るテストではその退行を
    // 検出できない。
    const note = panel.getByRole("note");
    expect(within(note).getByText("The width can be at most 4096 pixels.")).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^generate$/i })).toBeDisabled();

    // 減らせば消える（高さ側は最初から巻き込まれていない）。
    fireEvent.change(panel.getByRole("spinbutton", { name: /^left$/i }), { target: { value: "15" } });
    expect(panel.queryByText("The width can be at most 4096 pixels.")).not.toBeInTheDocument();
    expect(panel.queryByText("The height can be at most 4096 pixels.")).not.toBeInTheDocument();
  });

  it("caps the duration slider at the source video's own length", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);

    // Before a source is attached the ceiling is just the config maximum.
    expect(panel.getByRole("slider", { name: /duration/i })).toHaveAttribute("max", "481");

    await attachSource(panel, user);

    // 8s * 24fps = 192 frames -> the largest 8n+1 at or below that is 185, and
    // the previously-stored 257 is capped down to it (derived, never stored).
    expect(panel.getByRole("slider", { name: /duration/i })).toHaveAttribute("max", "185");
    expect(sliderValue(panel, /duration/i)).toBe("185");
  });

  // 快適上限マーカー（オーナー指示 第2バッチ、2026-08-09）。Create の DURATION と
  // 同じ `spillUtils.resolveSpillFreeFrames` を、**拡張後キャンバス** の寸法で
  // 引く —— 元動画の寸法で引くと必ず甘い値になるため、そこがこのテストの主眼。
  it("ticks the comfortable ceiling on the duration slider, resolved from the EXTENDED canvas", async () => {
    const user = userEvent.setup();
    // 30 秒あるので尺の上限は元動画長ではなく設定側の 481 になり、快適上限
    // (257) を跨いだ状態を作れる。
    const { bridge } = createPanelBridge({ media: { durationSec: 30, width: 1265, height: 720 } });
    const panel = renderPanel(bridge);

    // 素材が無いうちは目盛りも警告も出ない（0x0 のキャンバスに快適上限は無い）。
    expect(panel.getByRole("slider", { name: /duration/i })).not.toHaveAttribute("list");

    // 既定パッド（7/8/24/24）で 1280x768 のキャンバスにする —— パッド 0 のままだと
    // キャンバスが元動画と同じ寸法になり、「元動画の寸法で引いていないこと」を
    // 示すというこのテストの主眼が消えてしまう。
    await attachSource(panel, user);

    const duration = panel.getByRole("slider", { name: /duration/i });
    expect(duration).toHaveAttribute("list", "outpaint-spill-tick");
    // 1265x720 -> キャンバス 1280x768。その行の値は 257 で、元動画寸法
    // (1265x720、面積的には 960x576 より 1280x768 に近い) ではなくキャンバスの
    // 行が引かれていることを、値そのものが示す。
    const tick = document.getElementById("outpaint-spill-tick");
    expect(tick?.querySelector("option")?.getAttribute("value")).toBe("257");

    // 既定値は 361（2026-08-19、賢い快適上限マーカーの算出値に引き上げ）で、
    // この行の快適上限 257 を既に超えているため、警告は最初から出る。
    expect(panel.getByText(/may slow down/i)).toBeInTheDocument();
    // ちょうど上限 (257) に下げれば超えていない扱いになり、警告は消える。
    fireEvent.change(duration, { target: { value: "257" } });
    expect(panel.queryByText(/may slow down/i)).not.toBeInTheDocument();
    fireEvent.change(duration, { target: { value: "265" } });
    expect(panel.getByText(/may slow down/i)).toBeInTheDocument();
    // 警告であってブロックではない。
    expect(panel.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
  });

  it("sends the extended canvas, the four pads, the pinned LoRA and the reference video", async () => {
    const user = userEvent.setup();
    const { bridge, request } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user);

    const generate = panel.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(generate).toBeEnabled());
    await user.click(generate);

    const isGenerate = ([method, params]: [string, unknown]) =>
      method === "backend.request" && (params as { path: string }).path.endsWith("/generate");
    await waitFor(() => expect(request.mock.calls.some((c) => isGenerate(c as [string, unknown]))).toBe(true));
    const call = request.mock.calls.find((c) => isGenerate(c as [string, unknown])) as
      | [string, { body: Record<string, unknown> }]
      | undefined;
    if (!call) throw new Error("no POST /generate call was made");
    const body = call[1].body;

    // 拡張後キャンバス（元 1265x720 + 上下左右の pad）。両方 128 の倍数。
    expect(body.width).toBe(1280);
    expect(body.height).toBe(768);
    expect((body.width as number) % 128).toBe(0);
    expect((body.height as number) % 128).toBe(0);
    expect(body.prompt).toBe(PROMPT);
    // 既定のマスクブラー（5 / 2）ではキーごと省略する —— このフィールドを
    // 知らないサーバーへも通るようにするため。`toEqual` がそれを固定している。
    expect(body.outpaint).toEqual({
      pad_left: 7,
      pad_right: 8,
      pad_top: 24,
      pad_bottom: 24,
      freeze_source_audio: true,
    });
    // 保持領域が元動画の寸法と一致すること（サーバーが ffprobe で照合する条件）。
    expect((body.width as number) - 7 - 8).toBe(1265);
    expect((body.height as number) - 24 - 24).toBe(720);
    expect(body.loras).toEqual([{ name: "in-outpainting", strength: 1.0 }]);
    expect(body.reference_video_id).toBe("vid-1");
    expect(body.num_frames).toBe(185);
    expect(((body.num_frames as number) - 1) % 8).toBe(0);
    // Outpainting は conditioning_images / crop_output と排他 —— どちらも送らない。
    expect(body.conditioning_images).toBeUndefined();
    expect(body.crop_output).toBeUndefined();
  });

  it("uploads the right-clicked material on mount, trimmed to the ribbon's range", async () => {
    const { bridge, request } = createPanelBridge();
    renderPanel(bridge, {
      initialIntent: {
        intent: "outpaint",
        targetMode: "edit",
        selection: {
          hasRange: true,
          rangeStart: 0,
          rangeEnd: 60,
          selected: [
            {
              layer: 1,
              frameStart: 0,
              frameEnd: 59,
              effectName: "動画ファイル",
              filePath: SOURCE_PATH,
              objectName: "clip",
              textContent: null,
              mediaWidth: 1265,
              mediaHeight: 720,
              mediaDurationSec: 30,
              playbackStartSec: 2,
              playbackEndSec: 4,
              hasPlaybackRange: true,
              playbackSpeed: 1,
              loopPlay: false,
              sectionCount: 1,
            },
          ],
          cursorFrame: 0,
          cursorLayer: 1,
          rate: 30,
          scale: 1,
          sampleRate: 44100,
        },
      },
    });

    await waitFor(() => {
      const upload = request.mock.calls.find(([method]) => method === "backend.uploadFile");
      expect(upload).toBeDefined();
      expect(upload?.[1]).toEqual({
        kind: "video",
        filePath: SOURCE_PATH,
        // 再生開始 2 秒、リボンの長さ 60 フレーム＝2 秒（30fps）。
        query: { trim_start_sec: "2.000", trim_duration_sec: "2.000" },
      });
    });
    // The picker was never opened — the material came straight from the route.
    expect(request.mock.calls.some(([method]) => method === "ui.pickFile")).toBe(false);
  });
});

// --- センタリング（設計方針書 §5-D、2026-08-12） ------------------------------
// 明示的に入にしたときだけ働く「値のミラー」である。§5-C で廃止した対辺連動（片辺
// の値が対辺の**上限**を動かす・常時暗黙）とは別物で、上限の定数 2 本は不変。
//
// 入の側のテストは共有ヘルパー `setPads` を**使わない**。あれは 4 辺を順に流すので
// 入のあいだは後の辺が対辺を上書きしてしまい、しかも `DEFAULT_PADS` は左右がほぼ
// 対称なのでミラーの故障を素通りさせる。全辺 0 から入り、辺ごとに単発の
// `fireEvent.change` を打って「打った辺」と「対辺」の両方を見る。
describe("OutpaintingPanel centering", () => {
  /** チェックボックス本体。素材が測れるまで存在しない（スライダー 4 本と同じ
   * `known` ゲートの中にある）。 */
  function centeringBox(panel: Panel): HTMLInputElement {
    return panel.getByRole("checkbox", { name: /keep centered/i }) as HTMLInputElement;
  }

  /** 1 辺だけをスライダーで動かす。 */
  function dragPad(panel: Panel, side: keyof PadValues, value: number) {
    fireEvent.change(panel.getByRole("slider", { name: new RegExp(`^${side}$`, "i") }), {
      target: { value: String(value) },
    });
  }

  /** 数値入力ボックスの表示値。 */
  function boxValue(panel: Panel, name: RegExp): string {
    return (panel.getByRole("spinbutton", { name }) as HTMLInputElement).value;
  }

  it("素材を測ってから現れ、既定は切", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);

    // 素材が無いうちはスライダー 4 本と同じく存在しない。
    expect(panel.queryByRole("checkbox", { name: /keep centered/i })).not.toBeInTheDocument();

    await attachSource(panel, user, null);
    expect(centeringBox(panel)).toBeInTheDocument();
    expect(centeringBox(panel)).not.toBeChecked();
  });

  // オーナー決定 (2026-08-12): 当初案「スライダー群の下」から「見出しの直下・
  // スライダーの上」へ配置を変更した。DOM 順そのものを固定する回帰網が無いと
  // この配置は静かに崩れうるので、見出し -> チェックボックス -> 最初のスライダー
  // (top) の順を `compareDocumentPosition` で確かめる。
  it("見出し・チェックボックス・最初のスライダーがこの順で並ぶ", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    const heading = panel.getByText("How much to add");
    const box = centeringBox(panel);
    const topSlider = panel.getByRole("slider", { name: /^top$/i });

    expect(Boolean(heading.compareDocumentPosition(box) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    expect(Boolean(box.compareDocumentPosition(topSlider) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  });

  it("切のあいだは対辺に何も起きない（既存挙動の非回帰）", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    dragPad(panel, "top", 10);
    expect(sliderValue(panel, /^top$/i)).toBe("10");
    expect(sliderValue(panel, /^bottom$/i)).toBe("0");

    dragPad(panel, "left", 7);
    expect(sliderValue(panel, /^left$/i)).toBe("7");
    expect(sliderValue(panel, /^right$/i)).toBe("0");
  });

  it("切→入で各軸を等分する（端数は切り捨て、1px 消えるのは許容仕様）", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);

    // 切のうちに非対称な値を作る（上 10・下 21 = 合計 31）。
    dragPad(panel, "top", 10);
    dragPad(panel, "bottom", 21);
    expect(panel.getByText(/Generating at 1265 x 751 px/)).toBeInTheDocument();

    await user.click(centeringBox(panel));

    expect(centeringBox(panel)).toBeChecked();
    expect(sliderValue(panel, /^top$/i)).toBe("15");
    expect(sliderValue(panel, /^bottom$/i)).toBe("15");
    // 合計 31 -> 30。キャンバスが 1px 縮むことまで見えていてよい（許容仕様）。
    expect(panel.getByText(/Generating at 1265 x 750 px/)).toBeInTheDocument();
  });

  it("入のあいだはスライダーの値が対辺へ入る（軸をまたいで漏れない）", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);
    await user.click(centeringBox(panel));

    dragPad(panel, "top", 33);
    expect(sliderValue(panel, /^top$/i)).toBe("33");
    expect(sliderValue(panel, /^bottom$/i)).toBe("33");
    // 横は無傷 —— 縦を動かしても横軸には触れない。
    expect(sliderValue(panel, /^left$/i)).toBe("0");
    expect(sliderValue(panel, /^right$/i)).toBe("0");

    dragPad(panel, "left", 12);
    expect(sliderValue(panel, /^left$/i)).toBe("12");
    expect(sliderValue(panel, /^right$/i)).toBe("12");
    expect(sliderValue(panel, /^top$/i)).toBe("33");
    expect(sliderValue(panel, /^bottom$/i)).toBe("33");
  });

  it("入のあいだは数値ボックスからの入力も対辺へ入る（対辺のスライダー・ボックス両方）", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);
    await user.click(centeringBox(panel));

    fireEvent.change(panel.getByRole("spinbutton", { name: /^right$/i }), { target: { value: "55" } });

    expect(boxValue(panel, /^right$/i)).toBe("55");
    expect(boxValue(panel, /^left$/i)).toBe("55");
    // 対辺は「2 つの入力部品が同じ 1 つの状態を見ている」ので、スライダーも動く。
    expect(sliderValue(panel, /^left$/i)).toBe("55");
    expect(sliderValue(panel, /^right$/i)).toBe("55");
  });

  // `setPad` は `clampPad` をミラーより先に 1 回だけ適用する（`useOutpaintForm.ts`
  // の `next = clampPad(value)` を両辺へ書く）。ここが崩れて「先にミラー、後で
  // 個別クランプ」のような順に変わると、対辺だけ 4096 を超えたまま残るような
  // 退行が起こりうるので、打った辺・対辺の両方が上限で止まることを見る。
  it("入のあいだ、範囲外の入力は打った辺・対辺の両方が上限 4096 で止まる", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);
    await user.click(centeringBox(panel));

    fireEvent.change(panel.getByRole("spinbutton", { name: /^right$/i }), { target: { value: "9999" } });

    expect(boxValue(panel, /^right$/i)).toBe("4096");
    expect(boxValue(panel, /^left$/i)).toBe("4096");
  });

  it("入→切では値をリセットせず、以後は片辺だけが動く", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user, null);
    await user.click(centeringBox(panel));

    dragPad(panel, "top", 40);
    await user.click(centeringBox(panel));

    // 揃えるのを手伝う機能であって、揃った値を取り消す機能ではない。
    expect(centeringBox(panel)).not.toBeChecked();
    expect(sliderValue(panel, /^top$/i)).toBe("40");
    expect(sliderValue(panel, /^bottom$/i)).toBe("40");

    dragPad(panel, "top", 12);
    expect(sliderValue(panel, /^top$/i)).toBe("12");
    expect(sliderValue(panel, /^bottom$/i)).toBe("40");
  });

  it("チェック状態は生成リクエストに一切入らない（純粋な入力補助）", async () => {
    const user = userEvent.setup();
    // 幅も高さも偶数の素材にする —— 入のあいだ 1 軸の増分は必ず偶数なので、
    // 1265 のような奇数幅では 128 の格子に絶対に乗らず、submit まで進めない。
    const { bridge, request } = createPanelBridge({ media: { durationSec: 8, width: 1024, height: 704 } });
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user, null);
    await user.click(centeringBox(panel));

    // 1024 + 64×2 = 1152、704 + 32×2 = 768 —— どちらも 128 の倍数。
    dragPad(panel, "left", 64);
    dragPad(panel, "top", 32);

    const generate = panel.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(generate).toBeEnabled());
    await user.click(generate);

    const isGenerate = ([method, params]: [string, unknown]) =>
      method === "backend.request" && (params as { path: string }).path.endsWith("/generate");
    await waitFor(() => expect(request.mock.calls.some((c) => isGenerate(c as [string, unknown]))).toBe(true));
    const call = request.mock.calls.find((c) => isGenerate(c as [string, unknown])) as
      | [string, { body: Record<string, unknown> }]
      | undefined;
    if (!call) throw new Error("no POST /generate call was made");
    const body = call[1].body;

    // トップレベルにそれらしいキーが 1 つも無いこと。送られるのはミラーの**結果**
    // であるパッド 4 値だけである。
    expect(Object.keys(body).filter((key) => /cent(er|re|ring)/i.test(key))).toEqual([]);
    expect(body.width).toBe(1152);
    expect(body.height).toBe(768);
    expect(body.outpaint).toEqual({
      pad_left: 64,
      pad_right: 64,
      pad_top: 32,
      pad_bottom: 32,
      freeze_source_audio: true,
    });
  });

  it("等分で全辺 0 になれば padsZero が戻り、素材を差し替えてもチェックは入のまま", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge({ swapMedia: { durationSec: 8, width: 640, height: 480 } });
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user, null);

    // 上 1・下 0 —— 合計 1 なので等分すると 0/0 になる。特例は設けない（オーナー
    // 許容仕様）ので、四辺 0 に戻って padsZero が復活する。
    dragPad(panel, "top", 1);
    await user.click(centeringBox(panel));

    expect(sliderValue(panel, /^top$/i)).toBe("0");
    expect(sliderValue(panel, /^bottom$/i)).toBe("0");
    expect(
      within(panel.getByRole("note")).getByText(/add at least some amount on one of the four sides/i),
    ).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^generate$/i })).toBeDisabled();

    // 素材差し替え: パッドは 0 に戻る（§5-B の 5）。ここは入のまま値を入れてから
    // 差し替えて、確かにリセットが起きたことを見る。
    dragPad(panel, "top", 30);
    expect(sliderValue(panel, /^bottom$/i)).toBe("30");

    await user.click(panel.getByRole("button", { name: /change source video/i }));
    await waitFor(() => expect(panel.getByText(/Generating at 640 x 480 px/)).toBeInTheDocument());
    expect(sliderValue(panel, /^top$/i)).toBe("0");
    expect(sliderValue(panel, /^bottom$/i)).toBe("0");
    // チェックは維持される —— `ZERO_PADS` は対称なので不整合が生じない。
    expect(centeringBox(panel)).toBeChecked();
  });
});

// --- 共有プロンプト（2026-08-09 一本化） ------------------------------------
// The panel used to own a prompt textarea of its own. It now reads the SHARED
// `PromptBar` (`AppShell` -> `EditScreen` -> here), exactly like Create/Chain,
// so there is one prompt in the app and nothing to keep in sync.
describe("OutpaintingPanel prompt", () => {
  it("has no prompt box of its own", () => {
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    expect(panel.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("clears the promptEmpty gate from the prop alone, with nothing typed", async () => {
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await waitFor(() =>
      expect(within(panel.getByRole("note")).queryByText(/fill the main prompt/i)).not.toBeInTheDocument(),
    );
  });

  it("blocks again when the shared prompt is only whitespace", () => {
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: "   " });
    expect(within(panel.getByRole("note")).getByText(/fill the main prompt/i)).toBeInTheDocument();
  });
});

// --- マスクブラー ------------------------------------------------------------
describe("OutpaintingPanel mask blur", () => {
  it("shows the dilation in full-resolution pixels for THIS canvas", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user);

    const blur = panel.getByRole("slider", { name: /mask blur/i }) as HTMLInputElement;
    // 内部はサーバーと同じ膨張段数 0-15。
    expect(blur).toHaveAttribute("min", "0");
    expect(blur).toHaveAttribute("max", "15");
    expect(blur).toHaveAttribute("step", "1");
    expect(blur.value).toBe("5");
    // 1280x768 のキャンバス（長辺 1280）で 5 段 = 100px。1920px なら 150px に
    // なる値で、これが旧固定値 150 の正体。
    expect(panel.getByText(/about 100 px inward from the edge of the extension/i)).toBeInTheDocument();
    // プレビューの凡例も同じ 1 つの値から出ている。
    expect(panel.getByText(/about 100 px inside the edge/i)).toBeInTheDocument();

    fireEvent.change(blur, { target: { value: "10" } });
    expect(panel.getByText(/about 200 px inward from the edge of the extension/i)).toBeInTheDocument();
    expect(panel.getByText(/about 200 px inside the edge/i)).toBeInTheDocument();
  });

  it("says 0 px explicitly at the bottom of the slider", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user);

    fireEvent.change(panel.getByRole("slider", { name: /mask blur/i }), { target: { value: "0" } });
    expect(panel.getByText(/0 px \(no dilation\)/i)).toBeInTheDocument();
    // 帯が無いので、プレビューの破線の凡例も消える。
    expect(panel.queryByText(/inside the edge/i)).not.toBeInTheDocument();
  });

  // 2026-08-09 (オーナー指示): this advice used to be a SECOND line of its own
  // (`blurGreenNote`) under the blend-band note. It is now a parenthetical
  // inside that one sentence, and the separate key is gone — so the assertion
  // is that both halves arrive as ONE paragraph.
  it("points at the mask blur, in the same sentence, when the mask's green is left showing", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const panel = renderPanel(bridge);
    await attachSource(panel, user);
    expect(
      panel.getByText(
        /about 100 px inward from the edge of the extension.*if the green of the mask is left showing in the added area, raise the mask blur/i,
      ),
    ).toBeInTheDocument();
  });

  it("warns (without blocking) once the two opposing bands would meet", async () => {
    const user = userEvent.setup();
    // 1265x300 -> キャンバス 1280x384。1 段 = 20px なので、8 段（160px）で
    // 上下の帯が 320px となり、元動画の短辺 300px を覆い切る。この素材は縦の
    // 不足分が 84px なので、上下 42px ずつを自分で入れて 384 にする。
    const { bridge } = createPanelBridge({ media: { durationSec: 8, width: 1265, height: 300 } });
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user, { left: 7, right: 8, top: 42, bottom: 42 });

    const blur = panel.getByRole("slider", { name: /mask blur/i });
    expect(panel.queryByText(/almost the whole of the original video/i)).not.toBeInTheDocument();

    fireEvent.change(blur, { target: { value: "8" } });
    expect(panel.getByText(/almost the whole of the original video/i)).toBeInTheDocument();
    // 警告であってブロックではない —— Generate は押せるまま。
    await waitFor(() => expect(panel.getByRole("button", { name: /^generate$/i })).toBeEnabled());
  });

  it("sends both dilation stages, 5:2 linked, once the value is moved off the default", async () => {
    const user = userEvent.setup();
    const { bridge, request } = createPanelBridge();
    const panel = renderPanel(bridge, { prompt: PROMPT });
    await attachSource(panel, user);

    fireEvent.change(panel.getByRole("slider", { name: /mask blur/i }), { target: { value: "10" } });

    const generate = panel.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(generate).toBeEnabled());
    await user.click(generate);

    const isGenerate = ([method, params]: [string, unknown]) =>
      method === "backend.request" && (params as { path: string }).path.endsWith("/generate");
    await waitFor(() => expect(request.mock.calls.some((c) => isGenerate(c as [string, unknown]))).toBe(true));
    const call = request.mock.calls.find((c) => isGenerate(c as [string, unknown])) as
      | [string, { body: Record<string, unknown> }]
      | undefined;
    if (!call) throw new Error("no POST /generate call was made");

    expect(call[1].body.outpaint).toEqual({
      pad_left: 7,
      pad_right: 8,
      pad_top: 24,
      pad_bottom: 24,
      blend_dilation_stage1: 10,
      blend_dilation_stage2: 4,
      freeze_source_audio: true,
    });
  });
});

// --- ジョブの行き先 ----------------------------------------------------------
describe("OutpaintingPanel job reporting", () => {
  it("reports the accepted job_id up to the shell (which highlights it in the ledger)", async () => {
    const user = userEvent.setup();
    const { bridge } = createPanelBridge();
    const onJobSubmitted = vi.fn();
    const panel = renderPanel(bridge, { prompt: PROMPT, onJobSubmitted });
    await attachSource(panel, user);

    const generate = panel.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(generate).toBeEnabled());
    await user.click(generate);

    await waitFor(() => expect(onJobSubmitted).toHaveBeenCalledWith("job-1"));
  });
});
