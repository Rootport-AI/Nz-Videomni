import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import { createMockBridge } from "./bridge/mockBridge";
import type { MockBridgeOptions } from "./bridge/mockBridge";
import type { ResultOf } from "./bridge";
import { en } from "./i18n/strings";
import { AppShell } from "./shell/AppShell";
import { getInpaintSlots, resetInpaintSlots, setInpaintBusy } from "./timeline/inpaintSlots";
import { getReservationState, resetProvisionalReservation } from "./timeline/provisionalReservation";

// 台帳 §3-55 Inpainting: 右クリック #22/#23 の通し。ここでしか証明できないのは
// 「右クリックを**2 回**受けて 1 つのパネルが揃う」こと —— native イベント →
// ルーティング → §4 ガード → 保管庫 → タブ切り替え → 再マウント、を 2 回ぶん
// 通して、**2 回目が 1 回目を消さない**ところまで。
//
// パネルの中身そのものは `modes/edit/EditScreen.inpaint.test.tsx`、窓の算術は
// `modes/edit/inpaintWindow.test.ts` が持つ。

/** 選択された部分フィルタ（レイヤー 5・100〜340 フレーム）。
 *
 * `filePath: null` は手抜きではない: 部分フィルタには裏のファイルが無いので
 * native も本当に null を返し、`resolveMenuSelection` が
 * `timeline.getSelection` を引き直す。だからこの同じ値を mock bridge の
 * `selection` にも渡して、引き直しが同じオブジェクトを答えるようにしてある
 * （`App.trackRoute.test.tsx` と同じ作法）。 */
function partialFilterSelection(): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [
      {
        layer: 5,
        frameStart: 100,
        frameEnd: 340,
        effectName: "部分フィルタ",
        filePath: null,
        objectName: "mosaic",
        textContent: null,
        mediaWidth: 0,
        mediaHeight: 0,
        mediaDurationSec: 0,
      },
    ],
    cursorFrame: 250,
    cursorLayer: 9,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

/** 対象動画（レイヤー 3・0〜599 フレーム）。`filePath` が入っているので
 * 引き直しは起きない —— つまり上の部分フィルタ用の `selection` 固定値に
 * 上書きされることもない。 */
function videoSelection(
  overrides: Partial<ResultOf<"timeline.getSelection">["selected"][number]> = {},
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [
      {
        layer: 3,
        frameStart: 0,
        frameEnd: 599,
        effectName: "動画ファイル",
        filePath: "C:\\v\\shot.mp4",
        objectName: "shot",
        textContent: null,
        mediaWidth: 1920,
        mediaHeight: 1080,
        mediaDurationSec: 40,
        hasPlaybackRange: true,
        playbackStartSec: 2,
        playbackEndSec: 22,
        ...overrides,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

const panel = () => screen.getByRole("tabpanel");
const inpaintPanel = () => document.querySelector<HTMLElement>(".inpaint-panel")!;

async function renderShell(bridgeOptions: MockBridgeOptions = {}) {
  const bridge = createMockBridge({ delayMs: 0, selection: partialFilterSelection(), ...bridgeOptions });
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^(generate|busy…)$/i }, { timeout: 5_000 });
  // スパイは起動が落ち着いてから張る —— シェルの起動時の問い合わせ（config /
  // status / jobs）を数えたくないので、右クリック以降だけを記録する。
  const spy = vi.spyOn(bridge, "request");
  return { bridge, spy };
}

/** その回に飛んだ `method` の呼び出しだけ。 */
function callsOf(spy: { mock: { calls: unknown[][] } }, method: string): unknown[][] {
  return spy.mock.calls.filter((call) => call[0] === method);
}

function emit(
  bridge: ReturnType<typeof createMockBridge>,
  action: string,
  selection: ResultOf<"timeline.getSelection">,
) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action, selection });
  });
}

/** Edit 画面が見えている状態になるまで待つ。 */
async function waitForInpaintPanel(): Promise<HTMLElement> {
  await screen.findByRole("tab", { name: "Inpainting" }, { timeout: 5_000 });
  await waitFor(() =>
    expect(screen.getByRole("tab", { name: "Inpainting" })).toHaveAttribute("aria-selected", "true"),
  );
  return panel();
}

/** 部分フィルタカードが埋まったか。 */
function hasPartialFilter(): boolean {
  return inpaintPanel().textContent?.includes("Layer 6, frames 101-341") === true;
}
/** 対象動画カードが埋まったか。 */
function hasTarget(): boolean {
  return inpaintPanel().textContent?.includes("shot.mp4") === true;
}

/** Inpainting パネルのフレーム数欄（数値側）。`DurationField` はスライダーと
 * 数値欄の対なので、`spinbutton` の 1 つ目がフレーム数・2 つ目がシード。 */
function framesInput(): HTMLInputElement {
  return within(inpaintPanel()).getAllByRole("spinbutton")[0] as HTMLInputElement;
}
function seedInput(): HTMLInputElement {
  return within(inpaintPanel()).getAllByRole("spinbutton")[1] as HTMLInputElement;
}

describe("App / 台帳 §3-55 Inpainting right-click routing", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    resetInpaintSlots();
  });
  afterEach(() => {
    window.localStorage.clear();
    resetInpaintSlots();
  });

  it(
    "部分フィルタ → 動画 の順で両方のカードが揃う",
    async () => {
      const { bridge } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(hasPartialFilter()).toBe(true));

      emit(bridge, "inpaintVideo", videoSelection());
      await waitFor(() => expect(hasTarget()).toBe(true));
      // 2 回目が 1 回目を消していない —— 保管庫がモジュール側にあるからで、
      // これが `remountTokens.edit` を挟んでも成立することがこの機能の前提。
      expect(hasPartialFilter()).toBe(true);
    },
    20_000,
  );

  it(
    "動画 → 部分フィルタ の逆順でも同じ（順序不問）",
    async () => {
      const { bridge } = await renderShell();
      emit(bridge, "inpaintVideo", videoSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(hasTarget()).toBe(true));

      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitFor(() => expect(hasPartialFilter()).toBe(true));
      expect(hasTarget()).toBe(true);
    },
    20_000,
  );

  it(
    "部分フィルタを撮り直しても、対象動画は上げ直さない",
    async () => {
      const { bridge, spy } = await renderShell();
      emit(bridge, "inpaintVideo", videoSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(hasTarget()).toBe(true));

      expect(callsOf(spy, "backend.uploadFile")).toHaveLength(1);

      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitFor(() => expect(hasPartialFilter()).toBe(true));
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitFor(() => expect(hasPartialFilter()).toBe(true));

      // 部分フィルタは対象スロットに触らないので、上げ直しは起きない。
      expect(callsOf(spy, "backend.uploadFile")).toHaveLength(1);
    },
    20_000,
  );

  it(
    "種別が違えば §4 の案内文で断られ、タブも動かない",
    async () => {
      const { bridge } = await renderShell();
      // 動画に「マスクに使う」は通らない（`requiredKind: "partialFilter"`）。
      emit(bridge, "inpaintPartialFilter", videoSelection());
      // §4 の案内文はシェルの `NoteArea`（`role="status"`）に出る。
      await waitFor(() => expect(document.querySelector(".note-area")).not.toBeNull(), { timeout: 5_000 });
      // Create のままで、Edit へは行っていない。
      expect(within(panel()).queryByRole("tab", { name: "Inpainting" })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "解像度が一致しなければ実数入りの理由でゲートする（オーナー裁定 D5）",
    async () => {
      const { bridge } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      // プロジェクトは 1920×1080（mock の `getEditInfo`）、素材は 1280×768。
      emit(bridge, "inpaintVideo", videoSelection({ mediaWidth: 1280, mediaHeight: 768 }));
      await waitFor(() => expect(hasTarget()).toBe(true));

      await screen.findByText(
        en.edit.inpainting.generateReasons.resolutionMismatch(1920, 1080, 1280, 768),
        undefined,
        { timeout: 5_000 },
      );
      expect(screen.getByRole("button", { name: en.edit.inpainting.generateButton })).toBeDisabled();
    },
    20_000,
  );

  it(
    "Generate で席が対象レイヤー × 窓に置かれる",
    async () => {
      const { bridge, spy } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      emit(bridge, "inpaintVideo", videoSelection());
      await waitFor(() => expect(hasTarget()).toBe(true));

      const user = userEvent.setup();
      // mock の `/generate` はプロンプト 1 文字以上を要求する（実サーバーと同じ）。
      // Inpainting 側は空欄でも止めない仕様なので、ここは mock を通すため。
      await user.type(
        screen.getByPlaceholderText(/describe the video you want to generate/i),
        "a quiet street",
      );

      const button = screen.getByRole("button", { name: en.edit.inpainting.generateButton });
      await waitFor(() => expect(button).not.toBeDisabled());
      await user.click(button);

      await waitFor(() => expect(getReservationState().phase).not.toBe("idle"), { timeout: 5_000 });
      const reserve = callsOf(spy, "timeline.insertProvisional")[0];
      expect(reserve?.[1]).toMatchObject({
        // 部分フィルタのレイヤー(5)ではなく、**対象動画**のレイヤー(3)。
        materialLayer: 3,
        materialFrameStart: 100,
        materialFrameEnd: 340,
      });
    },
    20_000,
  );

  it(
    "飛行中にもう一度右クリックしても、対象のアップロードは 1 回きり",
    async () => {
      // 敵対的レビュー M3 の回帰: 以前は右クリックのたびに Edit 画面を作り直して
      // いたので、飛行中の 2 回目が同じファイルをもう一度上げていた。いまは
      // 「始めた」の記録が保管庫にあり、画面も作り直さない。
      const { bridge, spy } = await renderShell({ holdUploads: true });
      emit(bridge, "inpaintVideo", videoSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(callsOf(spy, "backend.uploadFile")).toHaveLength(1));

      // まだ上がりきっていない状態で、同じ動画をもう一度／部分フィルタも 1 回。
      emit(bridge, "inpaintVideo", videoSelection());
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitFor(() => expect(hasPartialFilter()).toBe(true));

      bridge.releaseUploads();
      await waitFor(() => expect(hasTarget()).toBe(true));
      expect(callsOf(spy, "backend.uploadFile")).toHaveLength(1);
    },
    20_000,
  );

  it(
    "打ち直したフレーム数とシードは、2 回目の動画の右クリックで消えない",
    async () => {
      // 敵対的レビュー M2 の回帰: 再マウントしていた頃は、ここが毎回既定へ戻った。
      const { bridge } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(hasPartialFilter()).toBe(true));
      emit(bridge, "inpaintVideo", videoSelection());
      await waitFor(() => expect(hasTarget()).toBe(true));

      const user = userEvent.setup();
      await user.clear(framesInput());
      await user.type(framesInput(), "121");
      framesInput().blur();
      await user.clear(seedInput());
      await user.type(seedInput(), "4242");
      await waitFor(() => expect(framesInput().value).toBe("121"));
      expect(seedInput().value).toBe("4242");

      // 2 回目の動画の右クリック（別のレイヤー＝本当の差し替え）。
      emit(bridge, "inpaintVideo", videoSelection({ layer: 7 }));
      await waitFor(() => expect(hasTarget()).toBe(true));

      // フォームの状態は生きている。
      expect(framesInput().value).toBe("121");
      expect(seedInput().value).toBe("4242");
    },
    20_000,
  );

  it(
    "2 回目の部分フィルタの右クリックはフレーム数を再シードする",
    async () => {
      // ここだけは**戻る**のが仕様: フレーム数の既定は部分フィルタの長さで、
      // 新しい部分フィルタが来たらその長さに合わせ直す（オーナー裁定 D3）。
      const { bridge } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      await waitFor(() => expect(framesInput().value).toBe("241"));

      const user = userEvent.setup();
      await user.clear(framesInput());
      await user.type(framesInput(), "121");
      framesInput().blur();
      await waitFor(() => expect(framesInput().value).toBe("121"));

      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitFor(() => expect(framesInput().value).toBe("241"));
    },
    20_000,
  );

  it(
    "実行中の右クリックは案内文で断り、スロットを書き換えない（敵対的レビュー m1）",
    async () => {
      // マスクを描いている／送っている最中に入力を差し替えると、マスクと要求が
      // 別の場所を指したまま生成が走る。`AppShell` は保管庫の `busy` を見て
      // §4 ガードと同じ形で断る —— 案内文だけ、何も書き換えない。
      const { bridge } = await renderShell();
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();
      emit(bridge, "inpaintVideo", videoSelection());
      await waitFor(() => expect(hasTarget()).toBe(true));

      act(() => setInpaintBusy(true));
      const before = getInpaintSlots();

      emit(bridge, "inpaintVideo", videoSelection({ layer: 7 }));
      await screen.findByText(en.edit.inpainting.busyRightClick, undefined, { timeout: 5_000 });
      // 対象は差し替わっていない（レイヤー 3 のまま）。
      expect(getInpaintSlots().target).toBe(before.target);
      expect(getInpaintSlots().subTabRequest).toBe(before.subTabRequest);
    },
    20_000,
  );

  it(
    "実行中は追尾の右クリックも同じ案内文で断る（native のスロットは共有）",
    async () => {
      const { bridge } = await renderShell();
      // 追尾の可否は `/status` のポーリング由来なので、それが着くまで待つ ——
      // 待たないと `trackingUnavailable` の方で断られ、別の理由を見てしまう。
      await screen.findByText(/server online/i, undefined, { timeout: 5_000 });
      emit(bridge, "inpaintPartialFilter", partialFilterSelection());
      await waitForInpaintPanel();

      act(() => setInpaintBusy(true));
      emit(bridge, "trackObject", partialFilterSelection());

      await screen.findByText(en.edit.inpainting.busyRightClick, undefined, { timeout: 5_000 });
      // Toolbox へは行っていない（Edit のまま）。
      expect(within(panel()).getByRole("tab", { name: "Inpainting" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
    },
    20_000,
  );
});


