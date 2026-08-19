import { describe, expect, it } from "vitest";
import {
  RETAKE_WINDOW_MAX_PX,
  RETAKE_WINDOW_MIN_PX,
  mapRangeToMaterialSec,
  normalizeSelectionRange,
  resolveRetakeWindow,
  snapRetakeWindowFrames,
} from "./retakeWindow";
import type { RetakeRangeResult, RetakeWindowResult } from "./retakeWindow";
import type { TrimSelectionItem } from "./sourceTrim";

/**
 * §1-17 Retake の窓決定コア。
 *
 * ここで最も重要なのは 2 本:
 *  - 「選択範囲は閉区間」という**未確定の仮説**を pin した
 *    `normalizeSelectionRange` のテスト（F0 が実機で確定したら、直すのは
 *    その 1 関数とこのテストの期待値だけ）、
 *  - 全長リボン（`spanCoversWholeMedia`）の条件付き許可 —— 再生開始が 0 で
 *    ないものを許すと**別の場所を無言で撮り直す**ため、許可と拒否の両方向を
 *    独立したケースで押さえてある。
 */

/** `decideSourceTrim` と同じく、選択スナップショットは rate/scale しか使わない。 */
const SEL_30FPS = { rate: 30, scale: 1 };
/** 1/fps = 0.25 ちょうど。境界の算術が二進で厳密に表せるので、浮動小数の
 * ごみが結論を左右しないケース用。 */
const SEL_4FPS = { rate: 4, scale: 1 };

/** 素材 10 秒・リボンは 0〜59（60 フレーム = 2.0 秒 @30fps）。全長より十分
 * 短いので、既定では §1-6 のトリムが立つ（＝写像の起点が再生開始秒になる）。 */
function makeItem(overrides: Partial<TrimSelectionItem> = {}): TrimSelectionItem {
  return {
    layer: 1,
    frameStart: 0,
    frameEnd: 59,
    effectName: "動画ファイル",
    filePath: "C:\\v\\a.mp4",
    objectName: "a",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 10,
    hasPlaybackRange: true,
    playbackStartSec: 0,
    playbackEndSec: 10,
    ...overrides,
  };
}

function expectRangeSkip(result: RetakeRangeResult, reason: string): void {
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.reason).toBe(reason);
}

function expectWindowSkip(result: RetakeWindowResult, reason: string): void {
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.reason).toBe(reason);
}

describe("normalizeSelectionRange — 閉区間仮説の隔離点（F0 未確定）", () => {
  /**
   * ⚠️ このテストは**仮説の pin** であって、実機で確認した事実ではない。
   *
   * 仮説: `timeline.getSelection` の `rangeStart`/`rangeEnd` は閉区間
   * （`rangeEnd` を含む）。根拠は同じスナップショット内の
   * `frameStart`/`frameEnd` の既存慣行だけで、範囲側の規約は TS/native とも
   * 未文書（bridge.cpp は SDK 値を無変換で流すだけ）。
   *
   * 実機確定後にここが半開だと判明した場合、直すのは
   * **`timeline/selectionRange.ts` の `normalizeSelectionRange` 本体 1 行**
   * （`rangeEnd + 1`）と、この describe 内の期待値だけ。`retakeWindow` は
   * それを再 export しているだけで、`prefillSeed` の右クリック尺シードも
   * 同じ葉を見ているので、両方とも自動で追随する。
   */
  it("閉区間として +1 し、半開 [start, endEx) を返す", () => {
    expect(normalizeSelectionRange({ rangeStart: 10, rangeEnd: 20 })).toEqual({
      startFrame: 10,
      endFrameEx: 21,
    });
  });

  it("1 フレームだけの範囲（start === end）は長さ 1 になる", () => {
    expect(normalizeSelectionRange({ rangeStart: 7, rangeEnd: 7 })).toEqual({
      startFrame: 7,
      endFrameEx: 8,
    });
  });
});

describe("mapRangeToMaterialSec — 範囲固有の拒否", () => {
  it("オブジェクトが無い -> noItem", () => {
    expectRangeSkip(mapRangeToMaterialSec(undefined, SEL_30FPS, { rangeStart: 0, rangeEnd: 10 }), "noItem");
  });

  it("逆転した範囲 -> emptyRange", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem(), SEL_30FPS, { rangeStart: 20, rangeEnd: 10 }),
      "emptyRange",
    );
  });

  it("閉区間仮説のもとで空になる範囲（end = start - 1）-> emptyRange", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem(), SEL_30FPS, { rangeStart: 20, rangeEnd: 19 }),
      "emptyRange",
    );
  });

  it("非有限な範囲 -> emptyRange", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem(), SEL_30FPS, { rangeStart: Number.NaN, rangeEnd: 10 }),
      "emptyRange",
    );
  });

  it("リボンより手前だけの範囲 -> rangeOutsideObject", () => {
    const item = makeItem({ frameStart: 100, frameEnd: 200 });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 0, rangeEnd: 99 }),
      "rangeOutsideObject",
    );
  });

  it("リボンより後ろだけの範囲 -> rangeOutsideObject", () => {
    const item = makeItem({ frameStart: 100, frameEnd: 200 });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 201, rangeEnd: 300 }),
      "rangeOutsideObject",
    );
  });

  it("末尾がリボン頭に 1 フレームだけ掛かる範囲は交差する（境界の内側）", () => {
    const item = makeItem({ frameStart: 100, frameEnd: 200 });
    const result = mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 0, rangeEnd: 100 });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.startFrame).toBe(100);
      expect(result.endFrameEx).toBe(101);
      expect(result.frameCount).toBe(1);
    }
  });
});

describe("mapRangeToMaterialSec — 交差クランプ", () => {
  it("両端がはみ出した範囲はリボンの内側へ詰められる", () => {
    const item = makeItem({ frameStart: 100, frameEnd: 199 });
    const result = mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 50, rangeEnd: 500 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startFrame).toBe(100);
    expect(result.endFrameEx).toBe(200); // frameEnd 199 は閉区間 -> 200 が排他端
    expect(result.frameCount).toBe(100);
    expect(result.durationSec).toBeCloseTo(100 / 30, 9);
  });

  it("リボンの内側に収まる範囲はそのまま通る", () => {
    const item = makeItem({ frameStart: 100, frameEnd: 199 });
    const result = mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 120, rangeEnd: 149 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startFrame).toBe(120);
    expect(result.endFrameEx).toBe(150);
    expect(result.frameCount).toBe(30);
  });
});

describe("mapRangeToMaterialSec — 素材秒への写像", () => {
  it("再生開始が 0 のとき、リボン頭からのフレーム差がそのまま素材秒になる", () => {
    // 4fps・リボン 100〜115（16 フレーム = 4.0 秒。素材 10 秒より十分短いので
    // §1-6 のトリムが立つ経路）・範囲頭 108 -> (108-100)/4 = 2.0 秒
    const item = makeItem({ frameStart: 100, frameEnd: 115, playbackStartSec: 0, playbackEndSec: 4 });
    const result = mapRangeToMaterialSec(item, SEL_4FPS, { rangeStart: 108, rangeEnd: 115 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBe(2);
    expect(result.trimOffsetSec).toBe(0);
    // 使える素材はファイル全長 10 秒ではなく、このリボンが実際に再生している
    // 0.0〜4.0 秒。
    expect(result.materialStartSec).toBe(0);
    expect(result.materialDurationSec).toBe(4);
    expect(result.projectFps).toBe(4);
  });

  it("再生開始が 0 でないとき、その分だけ素材秒が後ろへずれる", () => {
    // 再生開始 2.5 秒 + (108-100)/4 = 2.0 秒 -> 4.5 秒
    const item = makeItem({
      frameStart: 100,
      frameEnd: 115, // 16 フレーム = 4.0 秒 @4fps
      playbackStartSec: 2.5,
      playbackEndSec: 6.5,
    });
    const result = mapRangeToMaterialSec(item, SEL_4FPS, { rangeStart: 108, rangeEnd: 115 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBe(4.5);
    // §1-6 のトリム付きアップロードでは、切り出しファイル内の窓開始は
    // startSec - trimOffsetSec = 2.0 秒になる（F3 が使う）。
    expect(result.trimOffsetSec).toBe(2.5);
    // 使える素材は 2.5〜6.5 秒。頭より前（0〜2.5 秒）は画面に出ていない。
    expect(result.materialStartSec).toBe(2.5);
    expect(result.materialDurationSec).toBe(6.5);
  });

  it("起点秒はミリ秒へ丸められ、浮動小数のごみが残らない", () => {
    // 0.1 + 1/30*7 は素のままだと ...0000000001 級のごみが乗る組み合わせ。
    const item = makeItem({ frameStart: 0, frameEnd: 59, playbackStartSec: 0.1, playbackEndSec: 2 });
    const result = mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 7, rangeEnd: 20 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    // 0.1 + 7/30 = 0.33333... -> "0.333"
    expect(result.startSec).toBe(0.333);
  });
});

describe("mapRangeToMaterialSec — sourceTrim 由来の拒否 7 種", () => {
  it("mediaDurationSec 不明 -> unknownMediaDuration", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem({ mediaDurationSec: 0 }), SEL_30FPS, { rangeStart: 0, rangeEnd: 30 }),
      "unknownMediaDuration",
    );
  });

  it("rate/scale が解けない -> unresolvableSpan", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem(), { rate: 0, scale: 1 }, { rangeStart: 0, rangeEnd: 30 }),
      "unresolvableSpan",
    );
  });

  it("再生速度が 1.0 でない -> nonNeutralSpeed", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem({ playbackSpeed: 2 }), SEL_30FPS, { rangeStart: 0, rangeEnd: 30 }),
      "nonNeutralSpeed",
    );
  });

  it("ループ再生が有効 -> loopEnabled", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem({ loopPlay: true }), SEL_30FPS, { rangeStart: 0, rangeEnd: 30 }),
      "loopEnabled",
    );
  });

  it("中間点で 2 区間以上 -> multipleSections", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(makeItem({ sectionCount: 2 }), SEL_30FPS, { rangeStart: 0, rangeEnd: 30 }),
      "multipleSections",
    );
  });

  it("再生位置が読めていない（全長リボンではない）-> unknownPlaybackPosition", () => {
    const item = makeItem({ hasPlaybackRange: false });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 0, rangeEnd: 30 }),
      "unknownPlaybackPosition",
    );
  });

  it("オブジェクトが無い -> noItem（7 種の 1 つ）", () => {
    expectRangeSkip(mapRangeToMaterialSec(undefined, SEL_30FPS, { rangeStart: 0, rangeEnd: 1 }), "noItem");
  });
});

describe("mapRangeToMaterialSec — spanCoversWholeMedia の条件付き許可", () => {
  /** 素材 2 秒ちょうどをリボンが丸ごと占めている状態（60 フレーム @30fps）。 */
  function fullLength(overrides: Partial<TrimSelectionItem> = {}): TrimSelectionItem {
    return makeItem({ frameStart: 0, frameEnd: 59, mediaDurationSec: 2, playbackEndSec: 2, ...overrides });
  }

  it("再生開始 0 の素直な全長リボンは通り、起点 0 で写像される", () => {
    const result = mapRangeToMaterialSec(fullLength({ playbackStartSec: 0 }), SEL_30FPS, {
      rangeStart: 30,
      rangeEnd: 59,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBe(1); // 30/30
    expect(result.trimOffsetSec).toBe(0);
  });

  it("再生位置を読めていない全長リボンも通る（全長を占める以上、起点は 0 しかない）", () => {
    // v10 の 3 フィールドを **キーごと持たない** 古い native の形。
    const { playbackStartSec: _s, playbackEndSec: _e, hasPlaybackRange: _h, ...noPlayback } = fullLength();
    const result = mapRangeToMaterialSec(noPlayback, SEL_30FPS, { rangeStart: 30, rangeEnd: 59 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBe(1);
  });

  /**
   * `decideSourceTrim` は `spanCoversWholeMedia` を再生速度・ループ・中間点より
   * **先に**判定して即 return するので、全長リボンではその 3 つのチェックに
   * 到達しない（`sourceTrim.ts` 176-207 行）。`mapRangeToMaterialSec` は同じ
   * しきい値で自前にやり直すことになっており、以下 3 本がその pin。
   * 3 本とも「リボン長 = 素材全長」を満たすフィクスチャでなければ意味がない
   * （満たさないと `decideSourceTrim` 側で普通に弾かれてしまい、やり直しの
   * コードを一度も通らない）ので、`fullLength()` を使っている。
   */
  it("全長リボン × ループ再生 -> loopEnabled（早期 return に隠れる穴）", () => {
    // 0〜0.5 秒を 4 回ループして 2 秒のリボンを埋めている状態。素直に写すと
    // 1.5 秒地点＝実際は素材 0.0 秒、を撮り直してしまう。
    const item = fullLength({ loopPlay: true, playbackStartSec: 0, playbackEndSec: 0.5 });
    expectRangeSkip(mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 30, rangeEnd: 59 }), "loopEnabled");
  });

  it("全長リボン × 再生速度 0.5 -> nonNeutralSpeed（早期 return に隠れる穴）", () => {
    const item = fullLength({ playbackSpeed: 0.5, playbackStartSec: 0, playbackEndSec: 1 });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 30, rangeEnd: 59 }),
      "nonNeutralSpeed",
    );
  });

  it("全長リボン × 中間点 2 区間 -> multipleSections（早期 return に隠れる穴）", () => {
    const item = fullLength({ sectionCount: 2 });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 30, rangeEnd: 59 }),
      "multipleSections",
    );
  });

  it("再生開始が 0 でない全長リボンは弾く（別の場所を撮り直す無言の故障の防止）", () => {
    // R4 の実測状態: 頭を右へ引っぱってもリボン長は変わらず、尾は静止フレーム。
    // 長さは全長のままなのに再生窓は 0.5 秒目から始まっている。
    const item = fullLength({ hasPlaybackRange: true, playbackStartSec: 0.5, playbackEndSec: 2 });
    expectRangeSkip(
      mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 30, rangeEnd: 59 }),
      "spanCoversWholeMedia",
    );
  });

  it("再生開始が報告されているが 0 の全長リボンは通る（境界のもう一方）", () => {
    const item = fullLength({ hasPlaybackRange: true, playbackStartSec: 0 });
    expect(mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 0, rangeEnd: 59 }).ok).toBe(true);
  });
});

describe("mapRangeToMaterialSec — 静止フレームの尻尾（R4）", () => {
  /** 10 秒のリボンだが、実際に再生しているのは素材の 0〜8 秒だけ。残り 2 秒は
   * 8 秒目のコマが止まって見えている（`sourceTrim.ts` の R4 所見）。 */
  function frozenTail(): TrimSelectionItem {
    return makeItem({
      frameStart: 0,
      frameEnd: 299, // 300 フレーム = 10.0 秒 @30fps
      mediaDurationSec: 20,
      playbackStartSec: 0,
      playbackEndSec: 8,
    });
  }

  it("交差は静止フレームの手前で止まる", () => {
    const result = mapRangeToMaterialSec(frozenTail(), SEL_30FPS, { rangeStart: 200, rangeEnd: 299 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.endFrameEx).toBe(240); // 8.0 秒 × 30fps
    expect(result.frameCount).toBe(40);
    // 使える素材の終端もファイル全長 20 秒ではなく 8 秒。
    expect(result.materialDurationSec).toBe(8);
  });

  it("静止フレームの上だけを選んだら rangeOutsideObject", () => {
    expectRangeSkip(
      mapRangeToMaterialSec(frozenTail(), SEL_30FPS, { rangeStart: 250, rangeEnd: 299 }),
      "rangeOutsideObject",
    );
  });
});

describe("snapRetakeWindowFrames", () => {
  it("8n+1 へ切り下げる", () => {
    expect(snapRetakeWindowFrames(73)).toBe(73);
    expect(snapRetakeWindowFrames(80)).toBe(73);
    expect(snapRetakeWindowFrames(81)).toBe(81);
    expect(snapRetakeWindowFrames(169)).toBe(169);
    expect(snapRetakeWindowFrames(176)).toBe(169);
  });

  it("1 未満・非有限は 1 に落ちる", () => {
    expect(snapRetakeWindowFrames(0)).toBe(1);
    expect(snapRetakeWindowFrames(-40)).toBe(1);
    expect(snapRetakeWindowFrames(Number.NaN)).toBe(1);
  });
});

describe("resolveRetakeWindow", () => {
  const GEN_FPS = 24;

  it("格子に乗る長さはそのまま通る（クランプも詰めものも無し）", () => {
    const result = resolveRetakeWindow({
      startSec: 1,
      durationSec: 145 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 30,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBe(145);
    expect(result.startSec).toBe(1);
    expect(result.durationSec).toBeCloseTo(145 / 24, 9);
    expect(result).toMatchObject({ clamped: false, padded: false, shifted: false });
  });

  it("格子に乗らない長さは切り下げられる", () => {
    const result = resolveRetakeWindow({
      startSec: 0,
      durationSec: 152 / GEN_FPS, // 145 と 153 の間
      genFps: GEN_FPS,
      materialDurationSec: 30,
    });
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.frames).toBe(145);
  });

  it("上限超過は頭アンカーで切り詰める（開始は動かさない）", () => {
    const result = resolveRetakeWindow({
      startSec: 5,
      durationSec: 400 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 60,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBe(RETAKE_WINDOW_MAX_PX);
    expect(result.startSec).toBe(5); // 頭は動かない
    expect(result).toMatchObject({ clamped: true, padded: false, shifted: false });
  });

  it("下限未満は下限まで伸ばす（padded）", () => {
    const result = resolveRetakeWindow({
      startSec: 1,
      durationSec: 40 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 30,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBe(RETAKE_WINDOW_MIN_PX);
    expect(result.startSec).toBe(1);
    expect(result).toMatchObject({ clamped: false, padded: true, shifted: false });
  });

  it("素材の終端に当たったら開始を手前へずらす（shifted）", () => {
    // 素材 10 秒・窓 73f @24fps = 3.041666… 秒 -> 開始の上限は 6.958333… 秒
    const result = resolveRetakeWindow({
      startSec: 9,
      durationSec: 40 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 10,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBe(RETAKE_WINDOW_MIN_PX);
    expect(result.startSec).toBeCloseTo(10 - 73 / 24, 9);
    expect(result).toMatchObject({ padded: true, shifted: true });
  });

  it("素材が窓の下限より短ければ outOfMaterial（リボンの実尺不足の事前検出）", () => {
    // 73f @24fps = 3.04 秒 > 素材 2.0 秒
    expectWindowSkip(
      resolveRetakeWindow({ startSec: 0, durationSec: 2, genFps: GEN_FPS, materialDurationSec: 2 }),
      "outOfMaterial",
    );
  });

  it("素材の全長が不明（0 以下・非有限）でも outOfMaterial", () => {
    expectWindowSkip(
      resolveRetakeWindow({ startSec: 0, durationSec: 5, genFps: GEN_FPS, materialDurationSec: 0 }),
      "outOfMaterial",
    );
    expectWindowSkip(
      resolveRetakeWindow({
        startSec: 0,
        durationSec: 5,
        genFps: GEN_FPS,
        materialDurationSec: Number.NaN,
      }),
      "outOfMaterial",
    );
  });

  it("生成 fps が解けない -> unresolvableFps", () => {
    expectWindowSkip(
      resolveRetakeWindow({ startSec: 0, durationSec: 5, genFps: 0, materialDurationSec: 30 }),
      "unresolvableFps",
    );
  });

  it("負の開始秒は 0 に寄せる", () => {
    const result = resolveRetakeWindow({
      startSec: -3,
      durationSec: 145 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 30,
    });
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.startSec).toBe(0);
  });

  it("materialStartSec より手前へは詰めない（頭がトリムされたリボン）", () => {
    // 使える素材は 2.5〜10.0 秒。窓 73f @24fps = 3.0417 秒なので開始の上限は
    // 6.9583 秒。9 秒を頼まれても 6.9583 秒までしか下がらない。
    const result = resolveRetakeWindow({
      startSec: 9,
      durationSec: 40 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 10,
      materialStartSec: 2.5,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBeCloseTo(10 - 73 / 24, 9);
    expect(result.shifted).toBe(true);
  });

  it("materialStartSec より手前を頼まれたら頭で止める（shifted）", () => {
    const result = resolveRetakeWindow({
      startSec: 1,
      durationSec: 145 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 20,
      materialStartSec: 2.5,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.startSec).toBe(2.5);
    expect(result.shifted).toBe(true);
  });

  it("使える素材（始端〜終端）が窓より短ければ outOfMaterial", () => {
    // 2.5〜5.0 秒 = 2.5 秒しかないのに窓の下限は 73f @24fps = 3.04 秒。
    expectWindowSkip(
      resolveRetakeWindow({
        startSec: 2.6,
        durationSec: 2,
        genFps: GEN_FPS,
        materialDurationSec: 5,
        materialStartSec: 2.5,
      }),
      "outOfMaterial",
    );
  });

  it("下限・上限は引数で上書きできる（将来の AppConfig.limits 追随）", () => {
    const result = resolveRetakeWindow({
      startSec: 0,
      durationSec: 400 / GEN_FPS,
      genFps: GEN_FPS,
      materialDurationSec: 60,
      minFrames: 41,
      maxFrames: 97,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBe(97);
    expect(result.clamped).toBe(true);
  });
});

describe("mapRangeToMaterialSec -> resolveRetakeWindow の受け渡し", () => {
  it("写像の出力をそのまま食わせて窓が決まる", () => {
    const item = makeItem({ frameStart: 0, frameEnd: 299, mediaDurationSec: 30, playbackEndSec: 30 });
    const mapped = mapRangeToMaterialSec(item, SEL_30FPS, { rangeStart: 60, rangeEnd: 209 });
    expect(mapped.ok).toBe(true);
    if (!mapped.ok) return;

    const window = resolveRetakeWindow({
      startSec: mapped.startSec,
      durationSec: mapped.durationSec,
      genFps: 24,
      materialDurationSec: mapped.materialDurationSec,
      materialStartSec: mapped.materialStartSec,
    });
    expect(window.ok).toBe(true);
    if (!window.ok) return;
    // 150 プロジェクトフレーム @30fps = 5.0 秒 -> 24fps で 120 -> 切り下げて 113
    expect(window.frames).toBe(113);
    expect(window.startSec).toBe(2); // 60/30
  });
});
