import { describe, expect, it } from "vitest";
import {
  BLEND_DILATION_MAX,
  BLEND_DILATION_MIN,
  canvasSize,
  centerPads,
  clampPad,
  COMFORT_TOKEN_BUDGET,
  comfortTokenEstimate,
  DEFAULT_BLEND_DILATION_STAGE1,
  DEFAULT_BLEND_DILATION_STAGE2,
  featherWidthPx,
  isOverComfortBudget,
  latentFrameCount,
  MAX_CANVAS_SIDE,
  MAX_PAD,
  maxNumFrames,
  MIN_INNER_SIDE,
  OUTPAINT_COMFORT_TOKEN_BUDGETS,
  outpaintReasons,
  resolveOutpaintComfortBudget,
  stage2FromStage1,
  totalPad,
} from "./outpaintGeometry";
import type { OutpaintReasonsInput } from "./outpaintGeometry";

describe("canvasSize", () => {
  it("元寸法 + 上下左右の pad", () => {
    expect(canvasSize(1265, 720, { left: 7, right: 8, top: 24, bottom: 24 })).toEqual({ width: 1280, height: 768 });
  });
});

describe("pad の範囲クランプ（0 起点・1px 刻み・スナップしない）", () => {
  // 2026-08-11 オーナー決定の中核。ここが緑である限り「指定した値と違う値が
  // 出てくる」ことは起こらない。
  it("128 の倍数でない値もそのまま通る（自動スナップをしない）", () => {
    expect(clampPad(7)).toBe(7);
    expect(clampPad(10)).toBe(10);
    expect(clampPad(20)).toBe(20);
    expect(clampPad(127)).toBe(127);
    expect(clampPad(129)).toBe(129);
  });

  it("0 は通る（下限は 0、アライメントの下限はもう無い）", () => {
    expect(clampPad(0)).toBe(0);
  });

  it("負値は 0、上限は定数 4096（対辺には依存しない）", () => {
    expect(clampPad(-1)).toBe(0);
    expect(clampPad(-500)).toBe(0);
    // 第2弾: 上限は API のパッド上限そのもの。元動画の大きさも対辺の値も
    // 関係しない —— 4096 を超えるキャンバスは理由コードが止める。
    expect(clampPad(99_999)).toBe(MAX_PAD);
    expect(clampPad(MAX_PAD)).toBe(MAX_PAD);
    expect(clampPad(MAX_PAD + 1)).toBe(MAX_PAD);
  });

  it("非数（空欄の数値入力ボックス）は 0、小数は四捨五入", () => {
    expect(clampPad(Number.NaN)).toBe(0);
    expect(clampPad(Number.POSITIVE_INFINITY)).toBe(0);
    expect(clampPad(10.4)).toBe(10);
    expect(clampPad(10.6)).toBe(11);
  });

  it("totalPad は 4 辺の合計", () => {
    expect(totalPad({ left: 1, right: 2, top: 3, bottom: 4 })).toBe(10);
    expect(totalPad({ left: 0, right: 0, top: 0, bottom: 0 })).toBe(0);
  });
});

describe("centerPads（センタリングを入にした瞬間の等分・2026-08-12）", () => {
  it("軸ごとに合計を半分ずつ両辺へ入れる", () => {
    expect(centerPads({ left: 10, right: 20, top: 0, bottom: 0 })).toEqual({
      left: 15,
      right: 15,
      top: 0,
      bottom: 0,
    });
  });

  it("奇数の合計は切り捨てる（1px 消えるのは許容済み）", () => {
    // 10 + 21 = 31 -> 15/15。合計は 30 になる。
    expect(centerPads({ left: 10, right: 21, top: 0, bottom: 0 })).toEqual({
      left: 15,
      right: 15,
      top: 0,
      bottom: 0,
    });
  });

  it("片軸の合計が 1 なら 0/0 になる（特例は設けない）", () => {
    // 上 1・下 0 -> 0/0。四辺すべて 0 になれば padsZero が生成を止める。
    expect(centerPads({ left: 0, right: 0, top: 1, bottom: 0 })).toEqual({
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
    });
  });

  it("縦と横は独立に計算する（軸をまたいで混ざらない）", () => {
    expect(centerPads({ left: 4, right: 6, top: 30, bottom: 40 })).toEqual({
      left: 5,
      right: 5,
      top: 35,
      bottom: 35,
    });
  });

  it("全辺 0 は 0 のまま", () => {
    expect(centerPads({ left: 0, right: 0, top: 0, bottom: 0 })).toEqual({
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
    });
  });

  it("冪等 —— 2 回通しても値は動かない（StrictMode の updater 二重呼び出しに安全）", () => {
    const once = centerPads({ left: 7, right: 8, top: 24, bottom: 25 });
    expect(centerPads(once)).toEqual(once);
  });

  it("引数を書き換えない（純関数）", () => {
    const input = { left: 10, right: 20, top: 24, bottom: 24 };
    centerPads(input);
    expect(input).toEqual({ left: 10, right: 20, top: 24, bottom: 24 });
  });
});

describe("maxNumFrames", () => {
  it("durationSec*fps がちょうど 8n+1 のときはその値のまま", () => {
    // 5s * 24fps = 120 -> 直下の 8n+1 は 113
    expect(maxNumFrames(5, 24)).toBe(113);
    // ちょうど 8n+1 になる組み合わせ: 4.7083...s は使わず、整数で作る
    expect(maxNumFrames(2, 12.5)).toBe(25); // 25 = 8*3+1
    expect(maxNumFrames(10, 24.1)).toBe(241); // floor(241) = 241 = 8*30+1
  });

  it("8n+1 でない積は切り捨てて 8n+1 に丸める", () => {
    expect(maxNumFrames(1, 24)).toBe(17); // 24 -> 17
    expect(maxNumFrames(3, 30)).toBe(89); // 90 -> 89
    expect(maxNumFrames(12.3, 30)).toBe(369); // 369 = 8*46+1
  });

  it("9 未満になる場合は下限 9", () => {
    expect(maxNumFrames(0.1, 24)).toBe(9); // 2 frames
    expect(maxNumFrames(0.33, 24)).toBe(9); // 7 frames
    expect(maxNumFrames(8 / 24, 24)).toBe(9); // ちょうど 8 frames
    expect(maxNumFrames(9 / 24, 24)).toBe(9); // ちょうど 9 frames = 8*1+1
  });

  it("不明な尺・fps（0 や負値）は下限 9", () => {
    expect(maxNumFrames(0, 24)).toBe(9);
    expect(maxNumFrames(10, 0)).toBe(9);
    expect(maxNumFrames(-5, 24)).toBe(9);
    expect(maxNumFrames(Number.NaN, 24)).toBe(9);
  });

  it("結果は必ず 8n+1", () => {
    for (const sec of [0.5, 1, 2.7, 5, 13.33, 20]) {
      for (const fps of [8, 24, 30, 60]) {
        expect((maxNumFrames(sec, fps) - 1) % 8).toBe(0);
      }
    }
  });
});

describe("快適上限（警告のみ・ブロックではない）", () => {
  it("潜在フレーム数は (num_frames - 1) / 8 + 1", () => {
    expect(latentFrameCount(9)).toBe(2);
    expect(latentFrameCount(97)).toBe(13);
    expect(latentFrameCount(1)).toBe(1);
  });

  it("推定トークン数は ⌊幅/32⌋*⌊高さ/32⌋*潜在フレーム数", () => {
    // 1280x768 = 40*24 = 960 マス。潜在 13 コマで 12480。
    expect(comfortTokenEstimate(1280, 768, 97)).toBe(12_480);
    // 1920x1088 = 60*34 = 2040 マス。潜在 21 コマで 42840。
    expect(comfortTokenEstimate(1920, 1088, 161)).toBe(42_840);
  });

  it("128 の格子から外れた寸法では各軸を切り捨てる（マスを増やさない端数）", () => {
    // 1265x720 -> ⌊1265/32⌋=39, ⌊720/32⌋=22 で 858 マス。切り捨てない旧式なら
    // 39.53*22.5 = 889.45 マス相当になっていた。生成できるのは 128 の倍数の
    // キャンバスだけなので、この差が出るのは編集途中の表示のみ。
    expect(comfortTokenEstimate(1265, 720, 97)).toBe(858 * 13);
  });

  it("エンジン系統ごとに予算を引く。表に無い系統・未確定はフォールバックの 40000", () => {
    expect(OUTPAINT_COMFORT_TOKEN_BUDGETS.ltx).toBe(42_240);
    expect(OUTPAINT_COMFORT_TOKEN_BUDGETS.ltx25).toBe(44_880);
    expect(COMFORT_TOKEN_BUDGET).toBe(40_000);

    expect(resolveOutpaintComfortBudget("ltx")).toBe(42_240);
    expect(resolveOutpaintComfortBudget("ltx25")).toBe(44_880);
    // `GET /models` 未着・オフラインは "" で届く（`activeEngineFamily`）。
    expect(resolveOutpaintComfortBudget(undefined)).toBe(COMFORT_TOKEN_BUDGET);
    expect(resolveOutpaintComfortBudget("")).toBe(COMFORT_TOKEN_BUDGET);
    // 将来のエンジンや綴り違いも黙って従来値へ。予算 0 は出さない。
    expect(resolveOutpaintComfortBudget("ltx3")).toBe(COMFORT_TOKEN_BUDGET);
    expect(resolveOutpaintComfortBudget("LTX25")).toBe(COMFORT_TOKEN_BUDGET);
    // `Object.prototype` の名前を系統名として渡しても表の穴にはならない。
    expect(resolveOutpaintComfortBudget("constructor")).toBe(COMFORT_TOKEN_BUDGET);
  });

  it("同じ幾何でも系統が違えば警告の出方が変わる（判定は予算引数だけを見る）", () => {
    // 1920x1088 / 161 コマ = 42840 トークン。
    expect(isOverComfortBudget(1920, 1088, 161, resolveOutpaintComfortBudget("ltx25"))).toBe(false);
    expect(isOverComfortBudget(1920, 1088, 161, resolveOutpaintComfortBudget("ltx"))).toBe(true);
    expect(isOverComfortBudget(1920, 1088, 161, resolveOutpaintComfortBudget(undefined))).toBe(true);

    // 1920x1088 / 153 コマ = 40800 トークン —— ltx の予算には収まるが、系統が
    // 分からないときの 40000 は超える。
    expect(isOverComfortBudget(1920, 1088, 153, resolveOutpaintComfortBudget("ltx"))).toBe(false);
    expect(isOverComfortBudget(1920, 1088, 153, resolveOutpaintComfortBudget(undefined))).toBe(true);

    // どの系統でも余裕のある寸法。
    expect(isOverComfortBudget(1280, 768, 97, resolveOutpaintComfortBudget("ltx"))).toBe(false);
    expect(isOverComfortBudget(1280, 768, 97, resolveOutpaintComfortBudget("ltx25"))).toBe(false);
    expect(isOverComfortBudget(1280, 768, 97, resolveOutpaintComfortBudget(undefined))).toBe(false);
  });

  it("ちょうど予算どおりは超過ではない（境界は `>`）", () => {
    // 1920x1088 = 2040 マス、潜在 22 コマ（169 フレーム）でちょうど 44880。
    expect(comfortTokenEstimate(1920, 1088, 169)).toBe(44_880);
    expect(isOverComfortBudget(1920, 1088, 169, 44_880)).toBe(false);
    expect(isOverComfortBudget(1920, 1088, 169, 44_879)).toBe(true);
  });
});

describe("マスクブラー（膨張段数 <-> 実寸ピクセル）", () => {
  it("実寸は 段数 × キャンバス長辺 ÷ 64", () => {
    // 既定の 5 段・1920px 長辺 = 150px —— `OutpaintPreview` に直書きされていた
    // 旧定数と一致する（この式はその定数の一般化なので、ここがずれたら
    // プレビューの破線も凡例の数字もずれる）。
    expect(featherWidthPx(5, 1920)).toBe(150);
    expect(featherWidthPx(5, 1280)).toBe(100);
    expect(featherWidthPx(1, 1280)).toBe(20);
    expect(featherWidthPx(15, 1280)).toBe(300);
  });

  it("長辺は縦長キャンバスでも「長い方の辺」", () => {
    // 768x1280 と 1280x768 は同じ長辺なので同じ帯幅になる。
    expect(featherWidthPx(5, Math.max(768, 1280))).toBe(featherWidthPx(5, Math.max(1280, 768)));
  });

  it("0 段・不明な寸法・不正値はすべて 0（帯を描かない）", () => {
    expect(featherWidthPx(0, 1920)).toBe(0);
    expect(featherWidthPx(5, 0)).toBe(0);
    expect(featherWidthPx(-1, 1920)).toBe(0);
    expect(featherWidthPx(Number.NaN, 1920)).toBe(0);
    expect(featherWidthPx(5, Number.NaN)).toBe(0);
  });

  it("stage 2 は公式の 5:2 で stage 1 に連動する", () => {
    expect(stage2FromStage1(DEFAULT_BLEND_DILATION_STAGE1)).toBe(DEFAULT_BLEND_DILATION_STAGE2);
    expect(stage2FromStage1(10)).toBe(4);
    expect(stage2FromStage1(15)).toBe(6);
  });

  it("stage 1 が 0 のときだけ stage 2 も 0、それ以外は下限 1", () => {
    expect(stage2FromStage1(0)).toBe(0);
    expect(stage2FromStage1(1)).toBe(1); // round(0.4)=0 だが下限で 1
    expect(stage2FromStage1(2)).toBe(1);
    expect(stage2FromStage1(-3)).toBe(0);
    expect(stage2FromStage1(Number.NaN)).toBe(0);
  });

  it("stage 2 は常に stage 1 以下 —— 見えるなじみ幅は stage 1 が決める", () => {
    for (let r = BLEND_DILATION_MIN; r <= BLEND_DILATION_MAX; r += 1) {
      expect(stage2FromStage1(r)).toBeLessThanOrEqual(r);
      expect(stage2FromStage1(r)).toBeLessThanOrEqual(BLEND_DILATION_MAX);
    }
  });
});

describe("outpaintReasons", () => {
  const READY: OutpaintReasonsInput = {
    prompt: "a wide shot of a city",
    sourceStatus: "ready",
    sourceVideoId: "vid-1",
    sourceTrimFailed: false,
    sourceWidth: 1265,
    sourceHeight: 720,
    sourceDurationSec: 8,
    pads: { left: 7, right: 8, top: 24, bottom: 24 },
    numFrames: 97,
    maxNumFrames: 185,
    hasOutpaintLora: true,
  };

  it("すべて揃っていれば空配列（＝生成可）", () => {
    expect(outpaintReasons(READY)).toEqual([]);
  });

  it("プロンプトが空白だけなら promptEmpty", () => {
    expect(outpaintReasons({ ...READY, prompt: "   " })).toContain("promptEmpty");
  });

  it("ソース未設定・アップロード中・失敗をそれぞれ区別する", () => {
    expect(outpaintReasons({ ...READY, sourceStatus: "idle", sourceVideoId: null })).toContain("sourceMissing");
    expect(outpaintReasons({ ...READY, sourceStatus: "uploading", sourceVideoId: null })).toContain("sourceUploading");
    expect(outpaintReasons({ ...READY, sourceStatus: "error", sourceVideoId: null })).toContain("sourceUploadFailed");
  });

  it("トリム失敗は生成ブロック", () => {
    expect(outpaintReasons({ ...READY, sourceTrimFailed: true })).toContain("sourceTrimFailed");
  });

  it("probe が 0 を返したら mediaInfoUnknown（寸法系の他の理由は出さない）", () => {
    const reasons = outpaintReasons({ ...READY, sourceWidth: 0, sourceHeight: 0, sourceDurationSec: 0 });
    expect(reasons).toContain("mediaInfoUnknown");
    expect(reasons).not.toContain("innerTooSmall");
    expect(reasons).not.toContain("padsZero");
  });

  it("拡張量が全部 0 なら padsZero", () => {
    const reasons = outpaintReasons({
      ...READY,
      sourceWidth: 1280,
      sourceHeight: 768,
      pads: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    expect(reasons).toEqual(["padsZero"]);
  });

  it(`元動画が ${MIN_INNER_SIDE}px 未満なら innerTooSmall`, () => {
    expect(outpaintReasons({ ...READY, sourceHeight: 200, pads: { left: 7, right: 8, top: 28, bottom: 28 } })).toContain(
      "innerTooSmall",
    );
    expect(outpaintReasons({ ...READY, sourceWidth: 255, sourceHeight: 256, pads: { left: 0, right: 1, top: 0, bottom: 0 } })).toContain(
      "innerTooSmall",
    );
    expect(outpaintReasons({ ...READY, sourceWidth: 256, sourceHeight: 256, pads: { left: 0, right: 128, top: 0, bottom: 0 } })).not.toContain(
      "innerTooSmall",
    );
  });

  it("num_frames が 8n+1 でない・上限超え・下限割れなら numFramesOffGrid", () => {
    expect(outpaintReasons({ ...READY, numFrames: 100 })).toEqual(["numFramesOffGrid"]);
    // 上限は元動画の長さから毎回導出される値（state に持たない）。
    expect(outpaintReasons({ ...READY, numFrames: 193, maxNumFrames: 185 })).toEqual(["numFramesOffGrid"]);
    expect(outpaintReasons({ ...READY, numFrames: 1 })).toEqual(["numFramesOffGrid"]);
    expect(outpaintReasons({ ...READY, numFrames: 185 })).toEqual([]);
  });

  it("in-outpainting が無ければ loraMissing", () => {
    expect(outpaintReasons({ ...READY, hasOutpaintLora: false })).toEqual(["loraMissing"]);
  });

  it("トークン超過の理由コードは存在しない（警告バナー側の担当）", () => {
    const reasons = outpaintReasons({ ...READY, pads: { left: 7 + 128 * 8, right: 8 + 128 * 8, top: 24, bottom: 24 } });
    expect(reasons).toEqual([]);
  });

  // --- 128 の格子（2026-08-11・パッド 1px 刻み化の中核） --------------------
  // 値の自動スナップをやめた代わりに、格子から外れたキャンバスはここで止める。
  // 幅と高さを別コードにしてあるのは仕様（片方だけ直すと片方の行だけ消える）。

  it("幅だけ 128 の倍数でなければ canvasWidthOffGrid だけが出る", () => {
    // 1265 + 0 + 0 = 1265（外れ）／720 + 24 + 24 = 768（合っている）
    const reasons = outpaintReasons({ ...READY, pads: { left: 0, right: 0, top: 24, bottom: 24 } });
    expect(reasons).toEqual(["canvasWidthOffGrid"]);
  });

  it("高さだけ 128 の倍数でなければ canvasHeightOffGrid だけが出る", () => {
    // 1265 + 7 + 8 = 1280（合っている）／720 + 0 + 0 = 720（外れ）
    const reasons = outpaintReasons({ ...READY, pads: { left: 7, right: 8, top: 0, bottom: 0 } });
    expect(reasons).toEqual(["canvasHeightOffGrid"]);
  });

  it("両方外れていれば 2 行とも出る（幅 → 高さの順）", () => {
    const reasons = outpaintReasons({ ...READY, pads: { left: 1, right: 0, top: 1, bottom: 0 } });
    expect(reasons).toEqual(["canvasWidthOffGrid", "canvasHeightOffGrid"]);
  });

  it("境界: ちょうど 128 の倍数なら通り、1px ずれれば止まる", () => {
    const ok = outpaintReasons({ ...READY, sourceWidth: 1272, pads: { left: 0, right: 8, top: 24, bottom: 24 } });
    expect(ok).toEqual([]); // 1272 + 8 = 1280
    const off = outpaintReasons({ ...READY, sourceWidth: 1273, pads: { left: 0, right: 8, top: 24, bottom: 24 } });
    expect(off).toEqual(["canvasWidthOffGrid"]); // 1273 + 8 = 1281
  });

  it("寸法不明のあいだは格子コードを出さない（mediaInfoUnknown が先に止める）", () => {
    const reasons = outpaintReasons({ ...READY, sourceWidth: 0, sourceHeight: 0, sourceDurationSec: 0 });
    expect(reasons).toContain("mediaInfoUnknown");
    expect(reasons).not.toContain("canvasWidthOffGrid");
    expect(reasons).not.toContain("canvasHeightOffGrid");
    expect(reasons).not.toContain("canvasWidthTooLarge");
    expect(reasons).not.toContain("canvasHeightTooLarge");
  });

  it("padsZero と格子コードは同時に出る（抑制しない）", () => {
    const reasons = outpaintReasons({ ...READY, pads: { left: 0, right: 0, top: 0, bottom: 0 } });
    expect(reasons).toEqual(["padsZero", "canvasWidthOffGrid", "canvasHeightOffGrid"]);
  });

  // --- 4096 の上限（2026-08-11 第2弾） -------------------------------------
  // 対辺連動をやめ、入力部品では 4096 超を止めなくなったので、ここが唯一の関門に
  // なった。格子と同じく幅・高さは別コードで、拡張後キャンバスを測る。

  it("パッドを盛って幅が 4096 を超えたら canvasWidthTooLarge", () => {
    // 1265 + 2000 + 959 = 4224（=128×33 なので格子は合っている＝TooLarge 単独で
    // 出る）。対辺連動を廃止したことで初めて到達できるようになった経路の回帰。
    const reasons = outpaintReasons({ ...READY, pads: { left: 2000, right: 959, top: 24, bottom: 24 } });
    expect(reasons).toEqual(["canvasWidthTooLarge"]);
  });

  it("高さが 4096 を超えたら canvasHeightTooLarge（幅は巻き込まない）", () => {
    // 720 + 2000 + 1504 = 4224（=128×33）／幅は 1265 + 7 + 8 = 1280 で無傷。
    const reasons = outpaintReasons({ ...READY, pads: { left: 7, right: 8, top: 2000, bottom: 1504 } });
    expect(reasons).toEqual(["canvasHeightTooLarge"]);
  });

  it("4096 超かつ 128 非倍数なら、同じ軸で 2 行とも出る", () => {
    // 1265 + 3000 = 4265 —— 4096 超であり、128 の倍数でもない。
    const reasons = outpaintReasons({ ...READY, pads: { left: 3000, right: 0, top: 24, bottom: 24 } });
    expect(reasons).toEqual(["canvasWidthTooLarge", "canvasWidthOffGrid"]);
  });

  it("境界: ちょうど 4096 は合法（TooLarge も OffGrid も出ない）", () => {
    // 4096 は 128 の倍数でもあるので、境界がどちらの行も呼ばないことを 1 本で
    // 確かめられる。
    const reasons = outpaintReasons({
      ...READY,
      pads: { left: MAX_CANVAS_SIDE - 1265, right: 0, top: 24, bottom: 24 },
    });
    expect(reasons).toEqual([]);
  });

  it("元動画自体が 4096 超なら、パッド 0 でも TooLarge が出る（padsZero と併存）", () => {
    // 5120x2816 はどちらも 128 の倍数なので、格子検査だけでは通ってしまう。この
    // 行はパッドを動かしても消えないが、そのままでよい（オーナー決定 2026-08-11:
    // 例外ルールを増やさない。キャンバス寸法の表示と合わせれば原因は読み取れる）。
    const reasons = outpaintReasons({
      ...READY,
      sourceWidth: 5120,
      sourceHeight: 2816,
      pads: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    expect(reasons).toEqual(["padsZero", "canvasWidthTooLarge"]);
  });
});
