import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { RangeBand, type RangeBandProps } from "./RangeBand";

/**
 * §1-17 Retake の区間バー。
 *
 * トラックは左 0 / 幅 400px に固定し、素材も 400 フレームにしてあるので
 * **1 フレーム = 1px**。clientX がそのままフレーム番号になり、ドラッグの
 * 期待値が読んで分かる。
 */
const MATERIAL_FRAMES = 400;
const GEN_FPS = 24;

function renderBand(overrides: Partial<RangeBandProps> = {}) {
  const onChange = vi.fn();
  const props: RangeBandProps = {
    materialFrames: MATERIAL_FRAMES,
    startFrame: 100,
    frames: 145,
    genFps: GEN_FPS,
    glueHeadFrames: 25,
    glueTailFrames: 24,
    onChange,
    ...overrides,
  };
  const utils = render(
    <LanguageProvider>
      <RangeBand {...props} />
    </LanguageProvider>,
  );
  const track = utils.getByTestId("range-band-track");
  vi.spyOn(track, "getBoundingClientRect").mockReturnValue({
    left: 0,
    right: 400,
    top: 0,
    bottom: 18,
    width: 400,
    height: 18,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
  return { ...utils, onChange, track };
}

/** jsdom は `setPointerCapture`/`releasePointerCapture` を持たないので、
 * 呼ばれたかどうかを見るためにスパイを生やす（本体側は `?.` 越しに呼ぶ）。 */
function spyPointerCapture(el: HTMLElement) {
  const setPointerCapture = vi.fn();
  const releasePointerCapture = vi.fn();
  Object.assign(el, { setPointerCapture, releasePointerCapture });
  return { setPointerCapture, releasePointerCapture };
}

describe("RangeBand — 4 層の描画", () => {
  it("窓・糊代 2 本・ハンドル 2 本を素材全長に対する比率で置く", () => {
    const { getByTestId } = renderBand();
    const window = getByTestId("range-band-window");
    expect(window.style.left).toBe("25%"); // 100/400
    expect(window.style.width).toBe("36.25%"); // 145/400

    const head = getByTestId("range-band-glue-head");
    expect(head.style.left).toBe("25%");
    expect(head.style.width).toBe("6.25%"); // 25/400

    const tail = getByTestId("range-band-glue-tail");
    expect(tail.style.left).toBe("55.25%"); // (245-24)/400
    expect(tail.style.width).toBe("6%"); // 24/400

    expect(getByTestId("range-band-handle-start").style.left).toBe("25%");
    expect(getByTestId("range-band-handle-end").style.left).toBe("61.25%"); // 245/400
  });

  it("糊代が 0 のときは縞を描かない", () => {
    const { queryByTestId } = renderBand({ glueHeadFrames: 0, glueTailFrames: 0 });
    expect(queryByTestId("range-band-glue-head")).toBeNull();
    expect(queryByTestId("range-band-glue-tail")).toBeNull();
  });

  it("数値読み出しはフレーム数と秒を 1 行で出す", () => {
    const { getByTestId } = renderBand();
    // 145 / 24fps = 6.0416… -> 6.04 秒（オーナー向けの日本語は
    // 「窓 145フレーム（6.04秒）」。テストの既定言語は en）。
    expect(getByTestId("range-band-readout").textContent).toBe(en.edit.retake.rangeBand.readout(145, "6.04"));
  });

  it("上限に届いていないときは注意行を出さない", () => {
    expect(renderBand({ frames: 145 }).queryByTestId("range-band-max-note")).toBeNull();
  });

  it("上限に達したら注意行を出す", () => {
    expect(renderBand({ frames: 169 }).getByTestId("range-band-max-note").textContent).toBe(
      en.edit.retake.rangeBand.maxNote(169),
    );
  });
});

describe("RangeBand — aria", () => {
  it("両ハンドルが role=slider と現在値・読み上げ文を持つ", () => {
    const { getByTestId } = renderBand();
    const start = getByTestId("range-band-handle-start");
    const end = getByTestId("range-band-handle-end");

    expect(start.getAttribute("role")).toBe("slider");
    expect(end.getAttribute("role")).toBe("slider");
    expect(start.getAttribute("aria-label")).toBe(en.edit.retake.rangeBand.startHandle);
    expect(end.getAttribute("aria-label")).toBe(en.edit.retake.rangeBand.endHandle);

    expect(start.getAttribute("aria-valuenow")).toBe("100");
    expect(start.getAttribute("aria-valuetext")).toBe(en.edit.retake.rangeBand.startValueText(100, "4.17"));
    expect(end.getAttribute("aria-valuenow")).toBe("245");
    expect(end.getAttribute("aria-valuetext")).toBe(en.edit.retake.rangeBand.endValueText(245, "10.21"));
  });

  it("aria の可動域は窓長の下限・上限から出る", () => {
    const { getByTestId } = renderBand();
    const start = getByTestId("range-band-handle-start");
    // 終端 245 を固定したまま頭を動かせる範囲: [245-169, 245-73]
    expect(start.getAttribute("aria-valuemin")).toBe("76");
    expect(start.getAttribute("aria-valuemax")).toBe("172");

    const end = getByTestId("range-band-handle-end");
    // 頭 100 を固定したまま尻尾を動かせる範囲: [100+73, min(400, 100+169)]
    expect(end.getAttribute("aria-valuemin")).toBe("173");
    expect(end.getAttribute("aria-valuemax")).toBe("269");
  });

  it("バー全体が group ラベルを持つ", () => {
    const { getByRole } = renderBand();
    expect(getByRole("group", { name: en.edit.retake.rangeBand.label })).toBeTruthy();
  });
});

describe("RangeBand — ポインタ操作", () => {
  it("窓をつかんで動かすとキャプチャを取り、離すときに解放して確定する", () => {
    const { getByTestId, onChange } = renderBand();
    const window = getByTestId("range-band-window");
    const capture = spyPointerCapture(window);

    fireEvent.pointerDown(window, { pointerId: 3, clientX: 150 }); // 窓頭から +50 の位置をつかむ
    expect(capture.setPointerCapture).toHaveBeenCalledWith(3);

    fireEvent.pointerMove(window, { pointerId: 3, clientX: 200 }); // 頭は 150 へ
    expect(getByTestId("range-band-window").style.left).toBe("37.5%"); // 150/400
    expect(onChange).not.toHaveBeenCalled(); // ドラッグ中は上げない

    fireEvent.pointerUp(window, { pointerId: 3, clientX: 200 });
    expect(capture.releasePointerCapture).toHaveBeenCalledWith(3);
    expect(onChange).toHaveBeenCalledWith({ startFrame: 150, frames: 145 });
  });

  it("pointercancel はキャプチャを解放し、確定せずに元へ戻す", () => {
    const { getByTestId, onChange } = renderBand();
    const window = getByTestId("range-band-window");
    const capture = spyPointerCapture(window);

    fireEvent.pointerDown(window, { pointerId: 3, clientX: 150 });
    fireEvent.pointerMove(window, { pointerId: 3, clientX: 250 });
    expect(getByTestId("range-band-window").style.left).not.toBe("25%");

    fireEvent.pointerCancel(window, { pointerId: 3 });
    expect(capture.releasePointerCapture).toHaveBeenCalledWith(3);
    expect(onChange).not.toHaveBeenCalled();
    expect(getByTestId("range-band-window").style.left).toBe("25%"); // 元の 100/400
  });

  it("lostpointercapture も同じくドラッグを畳む", () => {
    const { getByTestId, onChange } = renderBand();
    const window = getByTestId("range-band-window");
    spyPointerCapture(window);

    fireEvent.pointerDown(window, { pointerId: 3, clientX: 150 });
    fireEvent.pointerMove(window, { pointerId: 3, clientX: 250 });
    fireEvent.lostPointerCapture(window, { pointerId: 3 });

    expect(onChange).not.toHaveBeenCalled();
    expect(getByTestId("range-band-window").style.left).toBe("25%");
  });

  it("窓の平行移動は素材の末尾で止まる（長さは変わらない）", () => {
    const { getByTestId, onChange } = renderBand();
    const window = getByTestId("range-band-window");
    spyPointerCapture(window);

    fireEvent.pointerDown(window, { pointerId: 1, clientX: 150 });
    fireEvent.pointerMove(window, { pointerId: 1, clientX: 900 });
    fireEvent.pointerUp(window, { pointerId: 1, clientX: 900 });

    expect(onChange).toHaveBeenCalledWith({ startFrame: MATERIAL_FRAMES - 145, frames: 145 });
  });

  it("終了側ハンドルを外へ引っぱっても頭は動かず、上限で止まる", () => {
    const { getByTestId, onChange } = renderBand();
    const end = getByTestId("range-band-handle-end");
    spyPointerCapture(end);

    fireEvent.pointerDown(end, { pointerId: 2, clientX: 245 });
    fireEvent.pointerMove(end, { pointerId: 2, clientX: 900 });
    fireEvent.pointerUp(end, { pointerId: 2, clientX: 900 });

    expect(onChange).toHaveBeenCalledWith({ startFrame: 100, frames: 169 });
  });

  it("開始側ハンドルを外へ引っぱっても終端は動かず、上限で止まる", () => {
    const { getByTestId, onChange } = renderBand();
    const start = getByTestId("range-band-handle-start");
    spyPointerCapture(start);

    fireEvent.pointerDown(start, { pointerId: 2, clientX: 100 });
    fireEvent.pointerMove(start, { pointerId: 2, clientX: -500 });
    fireEvent.pointerUp(start, { pointerId: 2, clientX: -500 });

    // 終端 245 のまま長さが上限 169 -> 頭は 76
    expect(onChange).toHaveBeenCalledWith({ startFrame: 76, frames: 169 });
  });
});

describe("RangeBand — キーボード", () => {
  it("矢印は 1 フレーム、Shift つきは 8 フレーム動かす（終了側）", () => {
    const { getByTestId, onChange } = renderBand();
    const end = getByTestId("range-band-handle-end");

    fireEvent.keyDown(end, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith({ startFrame: 100, frames: 146 });

    fireEvent.keyDown(end, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenLastCalledWith({ startFrame: 100, frames: 144 });

    fireEvent.keyDown(end, { key: "ArrowRight", shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith({ startFrame: 100, frames: 153 });
  });

  it("開始側は終端を固定したまま頭だけ動かす", () => {
    const { getByTestId, onChange } = renderBand();
    const start = getByTestId("range-band-handle-start");

    fireEvent.keyDown(start, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenLastCalledWith({ startFrame: 99, frames: 146 });

    fireEvent.keyDown(start, { key: "ArrowRight", shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith({ startFrame: 108, frames: 137 });
  });

  it("上限に達したハンドルはキーでも動かない（何も上げない）", () => {
    const { getByTestId, onChange } = renderBand({ frames: 169 });
    fireEvent.keyDown(getByTestId("range-band-handle-end"), { key: "ArrowRight" });
    fireEvent.keyDown(getByTestId("range-band-handle-start"), { key: "ArrowLeft" });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("下限に達したハンドルもキーでは縮まない", () => {
    const { getByTestId, onChange } = renderBand({ frames: 73 });
    fireEvent.keyDown(getByTestId("range-band-handle-end"), { key: "ArrowLeft" });
    fireEvent.keyDown(getByTestId("range-band-handle-start"), { key: "ArrowRight" });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("素材の頭で開始側ハンドルが止まる", () => {
    const { getByTestId, onChange } = renderBand({ startFrame: 0, frames: 169 });
    fireEvent.keyDown(getByTestId("range-band-handle-start"), { key: "ArrowLeft" });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("矢印以外のキーは無視する", () => {
    const { getByTestId, onChange } = renderBand();
    fireEvent.keyDown(getByTestId("range-band-handle-end"), { key: "Enter" });
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("RangeBand — 糊代帯は非対話", () => {
  it("aria から隠れていて、フォーカスも取らない", () => {
    const { getByTestId } = renderBand();
    for (const id of ["range-band-glue-head", "range-band-glue-tail"]) {
      const glue = getByTestId(id);
      expect(glue.getAttribute("aria-hidden")).toBe("true");
      expect(glue.hasAttribute("tabindex")).toBe(false);
      expect(glue.getAttribute("role")).toBeNull();
    }
  });

  it("糊代の上でポインタを押してもドラッグが始まらない（窓へも伝わらない）", () => {
    const { getByTestId, onChange } = renderBand();
    const glue = getByTestId("range-band-glue-head");

    fireEvent.pointerDown(glue, { pointerId: 5, clientX: 110 });
    fireEvent.pointerMove(glue, { pointerId: 5, clientX: 300 });
    fireEvent.pointerUp(glue, { pointerId: 5, clientX: 300 });

    expect(onChange).not.toHaveBeenCalled();
    // 窓はドラッグ状態にも入っていない。
    expect(getByTestId("range-band-window").className).not.toContain("dragging");
    expect(getByTestId("range-band-window").style.left).toBe("25%");
  });
});

describe("RangeBand — disabled", () => {
  it("ポインタもキーも受け付けず、ハンドルはタブ順から外れる", () => {
    const { getByTestId, onChange } = renderBand({ disabled: true });
    const window = getByTestId("range-band-window");
    const end = getByTestId("range-band-handle-end");
    expect(end.tabIndex).toBe(-1);
    expect(end.getAttribute("aria-disabled")).toBe("true");

    fireEvent.pointerDown(window, { pointerId: 1, clientX: 150 });
    fireEvent.pointerMove(window, { pointerId: 1, clientX: 250 });
    fireEvent.pointerUp(window, { pointerId: 1, clientX: 250 });
    fireEvent.keyDown(end, { key: "ArrowRight" });

    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("RangeBand — 素材長が未知でも壊れない", () => {
  it("materialFrames が 0 でも NaN を描かない", () => {
    const { getByTestId } = renderBand({ materialFrames: 0 });
    expect(getByTestId("range-band-window").style.left).toBe("0%");
    expect(getByTestId("range-band-window").style.width).toBe("0%");
    expect(getByTestId("range-band-readout").textContent).not.toContain("NaN");
  });

  it("genFps が 0 でも秒表示が NaN にならない", () => {
    const { getByTestId } = renderBand({ genFps: 0 });
    expect(getByTestId("range-band-readout").textContent).toBe(
      en.edit.retake.rangeBand.readout(145, "0.00"),
    );
  });
});
