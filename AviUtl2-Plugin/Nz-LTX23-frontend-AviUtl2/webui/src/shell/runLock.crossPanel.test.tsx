/**
 * バッチA2V ↔ バッチi2v-long の「同時に走らせない」配線の通しテスト
 * （2026-07-31 オーナー実機報告のリグレッション）。
 *
 * これまでのロックのテストは、どちらの機能も「テストが手で
 * `acquireRunLock(相手)` してから片方のフックを見る」形しかなかった。実機で
 * 問題になったのはそこではなく、**実際に片方のパネルのStartを押して走らせた
 * 状態で、もう片方のパネルのStartを見る**という2画面ぶんの経路と、走行中に
 * 画面がリマウントされた／ページが再読み込みされた後の状態である。ここでは
 * 実物のコンポーネントを2つ並べて、両方向＋その2つの事故ケースを通しでピンする。
 *
 * The cross-panel wiring for "only one batch runs at a time". Every
 * pre-existing lock test drives ONE panel and fakes the other side by calling
 * `acquireRunLock` by hand; what actually broke in the field is the two-panel
 * path — a REAL run started from one panel's Start button must disable the
 * OTHER panel's Start and explain why — plus the two ways that state can be
 * lost: a mid-run remount of the screen the panel sits on, and a page reload
 * that leaves the job running server-side while every browser-local lock is
 * gone.
 *
 * The Stop button (owner symptom 1: "Stop is nowhere to be found while
 * running") is asserted in the same place, since it is the same "does the panel
 * know a run is in flight?" wiring.
 */
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../api/types";
import { createMockBridge, createMockFs } from "../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../bridge";
import { LanguageProvider } from "../i18n/LanguageContext";
import { en } from "../i18n/strings";
import { BatchSection } from "../modes/batch/BatchSection";
import type { BatchGenerationValues } from "../modes/batch/useBatchForm";
import { BatchI2vLongSection } from "../modes/batch-i2v-long/BatchI2vLongSection";
import type { ChainSnapshotSource } from "../modes/batch-i2v-long/chainSnapshot";
import { __resetI2vLongRuntimeForTests } from "../modes/batch-i2v-long/runtime";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { useKeyframes } from "../modes/single/useKeyframes";
import { __resetRunLockForTests, getRunLockOwner } from "./runLock";

const WAV_DIR = "C:\\voice\\ep01";
const IMG_DIR = "C:\\shots\\ep01";

const GEN_VALUES: BatchGenerationValues = { width: 512, height: 320, frameRate: 24, seed: -1, numFrames: 481 };

function makeChain(): ChainSnapshotSource {
  return {
    buildRequest: (): GenerateChainRequest => ({
      prompt: "a cat walking",
      width: 768,
      height: 512,
      frame_rate: 24,
      seed: -1,
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: [{ num_frames: 121 }, { num_frames: 121 }],
      chunked_upsample: true,
    }),
    mode: "scratch",
    clips: [{ prompt: "" }, { prompt: "" }],
    isValid: true,
    validityReasons: [],
    outputFrames: 239,
    outputSeconds: 9.96,
    minClips: 2,
    minFramesForOverlap: 25,
  };
}

/**
 * Holds every `POST /generate/chain` open until the test releases it, then
 * answers 500 so the row fails and the run unwinds. Both runners submit to the
 * same path, so ONE gate parks whichever batch is running in `running` —
 * deterministically, and without touching `backend.uploadFile` (which the
 * KEYFRAMES capture the A2V panel needs also goes through).
 */
function gateChainSubmit(base: NativeBridge): { wrapped: NativeBridge; release: () => void } {
  let open!: () => void;
  const gate = new Promise<void>((resolve) => {
    open = resolve;
  });
  const wrapped: NativeBridge = {
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.request") {
        const p = params as { method?: string; path?: string };
        if (p.method === "POST" && p.path === "/api/v1/generate/chain") {
          await gate;
          return {
            status: 500,
            body: { error: { code: "MOCK_STOP", message: "gated by test" } },
          } as unknown as ResultOf<M>;
        }
      }
      return base.request(method, params);
    },
    requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
    on: (event, handler) => base.on(event, handler),
  };
  return { wrapped, release: () => open() };
}

interface PanelsProps {
  bridge: NativeBridge;
  /** Bumping either key remounts just that panel — what `AppShell`'s
   * `remountTokens` does to the whole Create/Chain screen when a right-click
   * routes an intent at it. */
  a2vKey?: number;
  i2vKey?: number;
  /** `JobsContext.hasActiveJob`, as both screens pass it down. */
  hasActiveJob?: boolean;
}

function Panels({ bridge, a2vKey = 0, i2vKey = 0, hasActiveJob = false }: PanelsProps) {
  // The Create screen owns the KEYFRAMES panel Batch A2V borrows its `Shared`
  // image from — mirrored here so the A2V panel can actually reach `canStart`.
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
  return (
    <>
      <button type="button" onClick={() => void keyframes.addFromCapture()}>
        add-keyframe
      </button>
      <div data-testid="a2v-panel">
        <BatchSection
          key={`a2v-${a2vKey}`}
          config={FALLBACK_APP_CONFIG}
          prompt="a cat walking"
          generationValues={GEN_VALUES}
          icLoraActive={false}
          keyframes={keyframes}
          nativeBridge={bridge}
          hasActiveJob={hasActiveJob}
        />
      </div>
      <div data-testid="i2v-panel">
        <BatchI2vLongSection
          key={`i2v-${i2vKey}`}
          config={FALLBACK_APP_CONFIG}
          chain={makeChain()}
          hasActiveJob={hasActiveJob}
          nativeBridge={bridge}
        />
      </div>
    </>
  );
}

/** jsdom doesn't implement the `<summary>` click toggle, so both accordions are
 * opened by setting `open` directly. Re-run after a remount (a fresh `<details>`
 * starts closed again). */
function openBoth(container: HTMLElement): void {
  act(() => {
    for (const details of container.querySelectorAll("details.batch-section")) {
      (details as HTMLDetailsElement).open = true;
    }
  });
}

function a2vPanel() {
  return screen.getByTestId("a2v-panel");
}
function i2vPanel() {
  return screen.getByTestId("i2v-panel");
}

/** Types a folder path into a hand-typable folder row and commits it on blur. */
async function typeFolder(
  user: ReturnType<typeof userEvent.setup>,
  panel: HTMLElement,
  label: string,
  value: string,
): Promise<void> {
  const input = within(panel).getByLabelText(label);
  await user.click(input);
  await user.type(input, value);
  await user.tab();
  await waitFor(() => expect(input).toHaveValue(value));
}

function foldersFs() {
  return createMockFs({
    folders: {
      [WAV_DIR]: [{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }],
      [IMG_DIR]: [{ name: "a.png", sizeBytes: 1024, mtimeMs: 0 }],
    },
  });
}

describe("バッチA2V ↔ バッチi2v-long の相互排他（実走行・両方向）", () => {
  beforeEach(() => {
    __resetI2vLongRuntimeForTests();
    __resetRunLockForTests();
  });

  /** Renders both panels and brings each to the point where its own Start
   * button is enabled (folder committed, folder scanned, shared keyframe
   * ready). Returns the render handle and the gated-submit release. */
  async function readyBothPanels(opts: { hasActiveJob?: boolean } = {}) {
    const user = userEvent.setup();
    const base = createMockBridge({ delayMs: 0, fs: foldersFs() });
    const { wrapped, release } = gateChainSubmit(base);
    let props: PanelsProps = { bridge: wrapped, ...(opts.hasActiveJob ? { hasActiveJob: true } : {}) };
    const view = render(
      <LanguageProvider>
        <Panels {...props} />
      </LanguageProvider>,
    );
    /** Re-renders both panels with `patch` merged in — bumping `a2vKey`/`i2vKey`
     * remounts that panel, exactly like `AppShell`'s `remountTokens` remounts a
     * whole mode screen. */
    const remount = (patch: Partial<PanelsProps>) => {
      props = { ...props, ...patch };
      view.rerender(
        <LanguageProvider>
          <Panels {...props} />
        </LanguageProvider>,
      );
    };
    openBoth(view.container);

    // --- Batch A2V ---
    await typeFolder(user, a2vPanel(), en.batch.wavDir.label, WAV_DIR);
    await user.click(within(a2vPanel()).getByRole("button", { name: en.batch.scanButton }));
    await waitFor(() => expect(within(a2vPanel()).getByText("a.wav")).toBeInTheDocument());
    // A freshly scanned row's image column is the `Shared` sentinel, which
    // needs one ready image in the (Create-owned) KEYFRAMES panel.
    await user.click(screen.getByRole("button", { name: "add-keyframe" }));

    // --- Batch i2v-long ---
    await typeFolder(user, i2vPanel(), en.batchI2vLong.imgDir.label, IMG_DIR);
    await user.click(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.scanButton }));
    await waitFor(() => expect(within(i2vPanel()).getByText("a.png")).toBeInTheDocument());

    return { user, view, release, remount, bridge: wrapped };
  }

  it("i2v-long実行中: 自分にStopが出て、バッチA2VのStartは無効＋案内文が出る", async () => {
    const { user, release } = await readyBothPanels();

    // 走行前: どちらのパネルにもStopは無い（実行中だけ出る、という設計）。
    expect(within(i2vPanel()).queryByRole("button", { name: en.batchI2vLong.stopButton })).toBeNull();
    expect(within(a2vPanel()).queryByRole("button", { name: en.batch.stopButton })).toBeNull();
    await waitFor(() => {
      expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeEnabled();
      expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton })).toBeEnabled();
    });

    await user.click(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton }));

    // 症状1: 走行中はStopが（押せる状態で）出る。
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.stopButton })).toBeEnabled();
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.runningButton })).toBeDisabled();

    // 症状2: 相手（バッチA2V）のStartが無効化され、理由が読める。
    await waitFor(() => {
      expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeDisabled();
    });
    expect(within(a2vPanel()).getByText(en.batch.lockedByOther)).toBeInTheDocument();

    // Stopは飾りではなく実際に効く（押すと「停止処理中…」へ移り、Stop自身は
    // 二度押しできなくなる）。ロックは中止要求だけでは返さない——いま生成中の
    // 1枚がバックエンドの唯一のジョブ枠を占め続けているため。
    await user.click(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.stopButton }));
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.stoppingButton })).toBeDisabled();
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.stopButton })).toBeDisabled();
    expect(getRunLockOwner()).toBe("batch-i2v-long");
    expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeDisabled();

    // 走行が終われば相手のStartは自動で戻る。
    await act(async () => {
      release();
    });
    await waitFor(() => {
      expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton })).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeEnabled();
    });
    expect(within(a2vPanel()).queryByText(en.batch.lockedByOther)).toBeNull();
    expect(getRunLockOwner()).toBeNull();
  }, 20_000);

  it("バッチA2V実行中: 自分にStopが出て、i2v-longのStartは無効＋理由ノートに案内が出る", async () => {
    const { user, release } = await readyBothPanels();

    await user.click(within(a2vPanel()).getByRole("button", { name: en.batch.startButton }));

    expect(within(a2vPanel()).getByRole("button", { name: en.batch.stopButton })).toBeEnabled();
    expect(within(a2vPanel()).getByRole("button", { name: en.batch.runningButton })).toBeDisabled();

    await waitFor(() => {
      expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton })).toBeDisabled();
    });
    const note = within(i2vPanel()).getByRole("note");
    expect(within(note).getByText(en.batchI2vLong.blockReasons.lockedByOther)).toBeInTheDocument();

    await act(async () => {
      release();
    });
    await waitFor(() => {
      expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton })).toBeEnabled();
    });
    expect(getRunLockOwner()).toBeNull();
  }, 20_000);

  // Chain画面は右クリックのintentルーティングで`key`リマウントされる
  // （`AppShell`の`remountTokens`）。i2v-longはランナーをモジュールレベルの
  // シングルトン（`runtime.ts`）に置いているので、リマウント後も走行中の
  // バッチに再接続でき、Stopもロックも保たれる。
  it("i2v-long走行中にChain側パネルがリマウントされても、Stopは出たままでロックも保たれる", async () => {
    const { user, view, release, remount } = await readyBothPanels();

    await user.click(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton }));
    expect(getRunLockOwner()).toBe("batch-i2v-long");

    remount({ i2vKey: 1 });
    openBoth(view.container);

    // リマウント後も走行中として復帰する（`runtime.ts`のシングルトンに再接続）。
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.stopButton })).toBeEnabled();
    expect(getRunLockOwner()).toBe("batch-i2v-long");
    expect(within(a2vPanel()).getByRole("button", { name: en.batch.startButton })).toBeDisabled();

    await act(async () => {
      release();
    });
    await waitFor(() => expect(getRunLockOwner()).toBeNull());
  }, 20_000);

  // バッチA2Vのランナーは`useRef`所有のままなので、Create画面が走行中に
  // リマウントされると走行そのものは孤児化する（既知の限界）。修正前はそれに
  // 加えて**共有ロックが永久に取り残され**、i2v-longが二度と開始できなくなって
  // いた。いまはロックの返却が実行Promise側にぶら下がっているので、孤児化した
  // 走行が終わればロックは必ず戻る。その間はStartが`hasActiveJob`で無効化され、
  // 案内文も出る（黙って何も起きないボタンにしない）。
  it("A2V走行中にCreate側パネルがリマウントされても、ロックは取り残されず、その間のStartは無効＋案内", async () => {
    const { user, view, release, remount } = await readyBothPanels();

    await user.click(within(a2vPanel()).getByRole("button", { name: en.batch.startButton }));
    expect(getRunLockOwner()).toBe("batch-a2v");

    // 走行中にCreate画面がリマウントされる（右クリック経由のintentルーティング）。
    // ジョブ台帳から見れば、孤児化した走行のジョブはまだ動いている。
    remount({ a2vKey: 1, hasActiveJob: true });
    openBoth(view.container);

    const start = within(a2vPanel()).getByRole("button", { name: en.batch.startButton });
    expect(start).toBeDisabled();
    expect(within(a2vPanel()).getByText(en.batch.jobActive)).toBeInTheDocument();

    // 孤児化した走行が終われば、ロックは戻る（修正前はここが`batch-a2v`のまま）。
    await act(async () => {
      release();
    });
    await waitFor(() => expect(getRunLockOwner()).toBeNull());
  }, 20_000);

  // ページ再読み込み（プラグインのパネルを閉じて開き直す等）を跨ぐと、共有
  // ロックもランナーもブラウザ側の揮発状態なので消える。一方でバックエンドの
  // ジョブは動き続ける。ロックだけに頼っていたバッチA2Vは、この状態でStartが
  // 押せてしまい、押しても何も起きない（あるいは409を撃つ）ままだった。
  it("再読み込み後のようにロックが無くてもジョブが動いていれば、バッチA2VのStartは無効＋案内", async () => {
    const { user } = await readyBothPanels({ hasActiveJob: true });

    expect(getRunLockOwner()).toBeNull();
    const start = within(a2vPanel()).getByRole("button", { name: en.batch.startButton });
    expect(start).toBeDisabled();
    expect(within(a2vPanel()).getByText(en.batch.jobActive)).toBeInTheDocument();

    // i2v-long側は元から`jobActive`の理由行を持っている（同じ状態で同じ扱い）。
    const note = within(i2vPanel()).getByRole("note");
    expect(within(note).getByText(en.batchI2vLong.blockReasons.jobActive)).toBeInTheDocument();
    expect(within(i2vPanel()).getByRole("button", { name: en.batchI2vLong.startButton })).toBeDisabled();

    // 無効なので押しても何も起きない（ジョブスロットへの投入は無し）。
    await user.click(start);
    expect(getRunLockOwner()).toBeNull();
  }, 20_000);
});
