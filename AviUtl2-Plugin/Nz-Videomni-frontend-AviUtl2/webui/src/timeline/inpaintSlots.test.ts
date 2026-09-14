import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearInpaintSlots,
  getInpaintSlots,
  inpaintTargetKey,
  markInpaintUploadStarted,
  midpointsFromSectionCount,
  setInpaintBusy,
  publishInpaintPartialFilter,
  publishInpaintTarget,
  resetInpaintSlots,
  setInpaintTargetUpload,
  subscribeInpaintSlots,
} from "./inpaintSlots";
import type { SelectionItem, TimelineSelection } from "./menuSelection";

// 台帳 §3-55: この保管庫が存在する理由は「右クリック 2 回で成立する唯一の機能」
// だから。下の 6 本は、そのために必要なことだけを押さえている。

function item(overrides: Partial<SelectionItem> = {}): SelectionItem {
  return {
    layer: 3,
    frameStart: 0,
    frameEnd: 240,
    effectName: "動画ファイル",
    filePath: "C:\\v\\a.mp4",
    objectName: "a",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 8,
    ...overrides,
  };
}

function selection(): TimelineSelection {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [item()],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

beforeEach(() => {
  resetInpaintSlots();
});

describe("midpointsFromSectionCount", () => {
  it("turns 中間点で区切られた区間の数 into the number of 中間点", () => {
    // native の契約: 答えられないときは 1（＝区間 1 つ＝中間点 0 個）。
    expect(midpointsFromSectionCount(1)).toBe(0);
    expect(midpointsFromSectionCount(3)).toBe(2);
    expect(midpointsFromSectionCount(undefined)).toBe(0);
    expect(midpointsFromSectionCount(0)).toBe(0);
    expect(midpointsFromSectionCount(Number.NaN)).toBe(0);
  });
});

describe("timeline/inpaintSlots", () => {
  it("starts empty", () => {
    expect(getInpaintSlots()).toEqual({
      partialFilter: null,
      target: null,
      version: 0,
      targetVersion: 0,
      subTabRequest: 0,
      uploadStartedFor: null,
      busy: false,
    });
  });

  it("keeps BOTH slots whichever right-click arrives first (実機ゲート G2)", () => {
    // これが保管庫の存在理由そのもの: 片方の publish がもう片方を消したら、
    // 2 回目の右クリックで 1 回目が消える。
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    publishInpaintTarget({ item: item(), selection: selection() });
    expect(getInpaintSlots().partialFilter).not.toBeNull();
    expect(getInpaintSlots().target).not.toBeNull();

    resetInpaintSlots();
    publishInpaintTarget({ item: item(), selection: selection() });
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    expect(getInpaintSlots().partialFilter).not.toBeNull();
    expect(getInpaintSlots().target).not.toBeNull();
  });

  it("bumps `version` for the partial filter ONLY", () => {
    // フレーム数欄の再シードはこの番号で駆動される。対象動画の差し替えでも
    // 増やすと、打ち直したフレーム数が既定へ戻る事故になる。
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    const after = getInpaintSlots().version;
    expect(after).toBe(1);

    publishInpaintTarget({ item: item(), selection: selection() });
    setInpaintTargetUpload("vid-1", false);
    expect(getInpaintSlots().version).toBe(after);

    publishInpaintPartialFilter({ layer: 6, frameStart: 0, frameEnd: 8, rate: 30, scale: 1, midpoints: 3 });
    expect(getInpaintSlots().version).toBe(after + 1);
  });

  it("drops the upload id when the TARGET is replaced", () => {
    // 前の素材の id を持ち越すと、別のファイルを描き替えることになる。
    publishInpaintTarget({ item: item(), selection: selection() });
    setInpaintTargetUpload("vid-1", true);
    expect(getInpaintSlots().target).toMatchObject({ videoId: "vid-1", trimFailed: true });

    publishInpaintTarget({ item: item({ layer: 9 }), selection: selection() });
    expect(getInpaintSlots().target).toMatchObject({ videoId: null, trimFailed: false });
  });

  it("ORs `trimFailed` rather than overwriting it, and ignores a write with no target", () => {
    publishInpaintTarget({ item: item(), selection: selection() });
    setInpaintTargetUpload("vid-1", true);
    setInpaintTargetUpload("vid-1", false);
    // 一度立った「トリムできなかった」は、その素材を使う限り消えない事実。
    expect(getInpaintSlots().target?.trimFailed).toBe(true);

    clearInpaintSlots();
    // ❌ のあとに飛行中のアップロードが着地しても、何も復活させない。
    setInpaintTargetUpload("vid-2", false);
    expect(getInpaintSlots().target).toBeNull();
  });

  it("notifies subscribers on every change and stops after unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeInpaintSlots(listener);

    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    publishInpaintTarget({ item: item(), selection: selection() });
    expect(listener).toHaveBeenCalledTimes(2);

    // `useSyncExternalStore` の要件: 変化が無い間はスナップショットの参照が
    // 変わらない（変化したときは必ず変わる）。
    const snapshot = getInpaintSlots();
    expect(getInpaintSlots()).toBe(snapshot);
    setInpaintTargetUpload("vid-1", false);
    expect(getInpaintSlots()).not.toBe(snapshot);

    unsubscribe();
    clearInpaintSlots();
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it("bumps `subTabRequest` on BOTH publishes (敵対的レビュー M2)", () => {
    // サブタブを寄せる合図はこのカウンタだけ。画面を作り直さないので、
    // どちらの右クリックでも増えなければ、2 回目で Inpainting に来ない。
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    expect(getInpaintSlots().subTabRequest).toBe(1);
    publishInpaintTarget({ item: item(), selection: selection() });
    expect(getInpaintSlots().subTabRequest).toBe(2);
    // 同じ対象をもう一度右クリック —— 上げ直しはしないが、合図だけは出す。
    publishInpaintTarget({ item: item(), selection: selection() });
    expect(getInpaintSlots().subTabRequest).toBe(3);
  });

  it("keeps the upload when the SAME target is right-clicked again", () => {
    // 敵対的レビュー M3: 飛行中のアップロードを掛け直す理由が無い。鍵が同じ
    // ままなので、`useInpaintForm` の効果は 2 回目を早期 return する。
    publishInpaintTarget({ item: item(), selection: selection() });
    const key = inpaintTargetKey(getInpaintSlots().target, getInpaintSlots().targetVersion);
    setInpaintTargetUpload("vid-1", false);

    publishInpaintTarget({ item: item(), selection: selection() });
    expect(getInpaintSlots().target?.videoId).toBe("vid-1");
    expect(inpaintTargetKey(getInpaintSlots().target, getInpaintSlots().targetVersion)).toBe(key);

    // 別のオブジェクトなら鍵が変わり、id も捨てる。
    publishInpaintTarget({ item: item({ layer: 9 }), selection: selection() });
    expect(getInpaintSlots().target?.videoId).toBeNull();
    expect(inpaintTargetKey(getInpaintSlots().target, getInpaintSlots().targetVersion)).not.toBe(key);
  });

  it("remembers which target an upload was STARTED for, and forgets it on ❌", () => {
    publishInpaintTarget({ item: item(), selection: selection() });
    const key = inpaintTargetKey(getInpaintSlots().target, getInpaintSlots().targetVersion)!;
    markInpaintUploadStarted(key);
    expect(getInpaintSlots().uploadStartedFor).toBe(key);
    // 同じ鍵の再記録はスナップショットを動かさない（描画ループを作らない）。
    const snapshot = getInpaintSlots();
    markInpaintUploadStarted(key);
    expect(getInpaintSlots()).toBe(snapshot);

    clearInpaintSlots();
    expect(getInpaintSlots().uploadStartedFor).toBeNull();
  });

  it("carries the busy flag `AppShell` refuses right-clicks on", () => {
    expect(getInpaintSlots().busy).toBe(false);
    setInpaintBusy(true);
    expect(getInpaintSlots().busy).toBe(true);
    const snapshot = getInpaintSlots();
    setInpaintBusy(true);
    expect(getInpaintSlots()).toBe(snapshot); // 変化が無ければ何もしない
    setInpaintBusy(false);
    expect(getInpaintSlots().busy).toBe(false);
  });

  it("resets WITHOUT dropping its subscribers (敵対的レビュー m2)", () => {
    // リスナーを捨てると、マウント済みの `useSyncExternalStore` が二度と更新を
    // 受け取れない購読者になり、リセット後のテストが静かに壊れる。
    const listener = vi.fn();
    subscribeInpaintSlots(listener);
    publishInpaintPartialFilter({ layer: 5, frameStart: 100, frameEnd: 340, rate: 30, scale: 1, midpoints: 3 });
    expect(listener).toHaveBeenCalledTimes(1);

    resetInpaintSlots();
    // リセットそのものも通知される。
    expect(listener).toHaveBeenCalledTimes(2);
    expect(getInpaintSlots().partialFilter).toBeNull();

    // …そして購読は生きている。
    publishInpaintTarget({ item: item(), selection: selection() });
    expect(listener).toHaveBeenCalledTimes(3);
  });
});

