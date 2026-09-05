/**
 * Outpainting（動画キャンバス拡張）の寸法計算コア。React にも i18n にも bridge にも
 * 依存しない純関数だけを置く。
 *
 * The pure geometry core behind the Edit tab's Outpainting panel (正本
 * `Docs/OUTPAINTING_DESIGN_NOTES.md` §4). Everything the panel needs to answer
 * "what canvas will this produce, and may I press Generate?" lives here so it
 * can be unit-tested without rendering anything.
 *
 * ## Why the pads start at 0 and move ONE PIXEL at a time
 *
 * A reference-video (`reference_video_id`) request requires width/height to be
 * multiples of 128 (設計書 §3-7 — the existing API enforces it regardless of
 * adapter kind), and a source video is very often NOT a multiple of 128
 * (1265×720, say). The original design (D4 "アライメント方式") answered that by
 * giving every slider a FLOOR — the unavoidable remainder, distributed by a
 * 詰め方向 the user picked — and a 128px ladder above it, so the canvas could
 * never leave the grid.
 *
 * オーナー決定 (2026-08-11) でその方式は撤回された: 128px 刻みでは「縦 738px の
 * 動画に上 10px・下 20px を足して 768px にする」という実用的な使い方が表現でき
 * ないため。今の規則は 3 つだけである。
 *
 *  1. どのスライダーも 0 から始まり、1 ピクセルずつ動く。
 *  2. 値の自動スナップ・自動補正は一切しない（{@link clampPad} は範囲に収める
 *     だけ）。
 *  3. キャンバス側の制約を破った値は黙って直さず、**生成のブロック理由**として
 *     出す（{@link outpaintReasons} の `canvasWidthOffGrid` /
 *     `canvasHeightOffGrid` が 128 の格子、`canvasWidthTooLarge` /
 *     `canvasHeightTooLarge` が 4096 の上限）。利用者は行が 1 本ずつ消えるのを
 *     見ながら自分で合わせる。
 *
 * ## スライダーと数値ボックスの役割分担（2026-08-11 第2弾）
 *
 * 上限の持ち方は 2 つに分かれている。{@link PAD_SLIDER_MAX} がスライダーの、
 * {@link MAX_PAD} が値そのものの上限である。第1弾では「4096 − 元動画の辺 − 対辺の
 * パッド」を両方の上限にしていたが、それはやめた —— 可動域が約 2800 に達すると
 * トラック上のマウス 1px が値 13 相当になり、マウスでは決して指定できない値が
 * 生まれていたし、対辺連動（片方の値が他方の上限を動かす）は「値を自動調整せず
 * メッセージで案内する」という本機能の思想と矛盾していた。
 *
 * したがってキャンバス側の制約（128 の格子・4096 の上限）は**すべて理由コード**で
 * 担保する。上限を入力部品に持たせて到達不能にする方式は取らない。
 */

/** The canvas grid every generated width/height must sit on (設計書 §3-7). */
export const CANVAS_MULTIPLE = 128;

/** The smallest KEPT (original video) side the server accepts, in pixels. The
 * kept region is the source video itself, so this is a property of the input
 * file, not of the pads — a 200px-tall clip simply cannot be outpainted. */
export const MIN_INNER_SIDE = 256;

/** Upper bound for either canvas side. Mirrors `limits.max_width` /
 * `limits.max_height` (`modes/single/defaultConfig.ts`), i.e. the 4096 ceiling
 * the rest of the app already uses. Since 2026-08-11 (第2弾) it is a BLOCK
 * REASON threshold (`canvasWidthTooLarge` / `canvasHeightTooLarge`), not an
 * input-widget ceiling — nothing stops the user from dialling past it. */
export const MAX_CANVAS_SIDE = 4096;

/** どのスライダーも 0〜この値だけを受け持つ（オーナー決定 2026-08-11 第2弾）。
 *
 * The pad SLIDER's ceiling — a plain constant, not derived from the source
 * video or from the opposite pad. Its size comes from the pad grid's own column
 * minimum (`EditScreen.css` の `.outpaint-pads` は `minmax(220px, 1fr)`): at
 * roughly that many pixels of track, one pixel of mouse travel is worth roughly
 * one unit of value, which is the property that makes the slider usable at all.
 * It is an approximation on purpose — at the narrowest layout the real track is
 * shorter than the column (the thumb takes room), so a mouse pixel is worth a
 * little over one unit, and on a wide window the column stretches and it is
 * worth less. 220 is a chosen working range, not an exact 1:1 mapping.
 *
 * The slider is therefore an INPUT AID covering the common small amounts; the
 * number box beside it is where an exact or large value goes ({@link MAX_PAD}). */
export const PAD_SLIDER_MAX = 220;

/** 1 辺のパッドとして送れる最大値。API の `OutpaintSpec` の `pad_* le=4096` と
 * 対応する（`api/models.py`）。
 *
 * Numerically the same as {@link MAX_CANVAS_SIDE} but a different quantity — one
 * pad's ceiling versus a canvas side's ceiling — so it is its own constant. The
 * number box carries this as its `max`, and {@link clampPad} enforces it. */
export const MAX_PAD = 4096;

/** 上下左右に足すピクセル数。すべて 0 以上。 */
export interface Pads {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Every pad at zero — the "nothing known yet / nothing added" pads. Frozen so
 * a caller can hand it around without defensive copying. */
export const ZERO_PADS: Pads = Object.freeze({ left: 0, right: 0, top: 0, bottom: 0 });

/** 現在の pad から拡張後キャンバスの寸法を出す。 */
export function canvasSize(srcW: number, srcH: number, pads: Pads): { width: number; height: number } {
  return {
    width: srcW + pads.left + pads.right,
    height: srcH + pads.top + pads.bottom,
  };
}

/**
 * Brings a raw pad value into `[0, MAX_PAD]` and makes it a whole number.
 *
 * Deliberately NOT a snap (2026-08-11): a value already inside the range comes
 * back untouched, 128 の倍数でなくても構わない。格子から外れたキャンバスは
 * {@link outpaintReasons} の `canvasWidthOffGrid` / `canvasHeightOffGrid` が
 * 生成を止めるので、ここで黙って別の値へ書き換える必要がない。
 *
 * 上限は引数ではなく定数 {@link MAX_PAD}（第2弾で対辺連動を廃止したため、呼び出し
 * 側が上限を計算して渡す理由が無くなった）。4096 を超えるキャンバスもここでは
 * 止めず、`canvasWidthTooLarge` / `canvasHeightTooLarge` が担当する。
 *
 * The `Number.isFinite` guard is pure defence, not a live path: the panel reads
 * its inputs with `Number(e.target.value)`, and `Number("")` is `0`, so an
 * emptied 数値入力ボックス arrives here as a plain zero rather than as `NaN`. It
 * stays as the last line of defence for the day that changes (a switch to
 * `valueAsNumber`, say, which DOES report `NaN` for an empty box) — a `NaN` pad
 * would poison the canvas readout and every gate downstream.
 */
export function clampPad(raw: number): number {
  if (!Number.isFinite(raw)) return 0;
  return Math.min(MAX_PAD, Math.max(0, Math.round(raw)));
}

/** The sum of all four pads — 0 exactly when nothing at all would be drawn. */
export function totalPad(pads: Pads): number {
  return pads.left + pads.right + pads.top + pads.bottom;
}

/**
 * 「センタリング」を入にした瞬間の等分。軸ごとに合計を 2 で割り（端数切捨て）、
 * 両辺へ同じ値を入れる（オーナー決定 2026-08-12）。
 *
 * 切り捨てなので拡張後キャンバスが指定より大きくなることはなく、値は常に元の
 * 両端の間に収まるので {@link clampPad} を掛け直す必要もない（パッドは常に
 * `[0, MAX_PAD]` の整数である）。左 10・右 21 なら 15/15 —— 合計 31 が 30 になり
 * 1 ピクセル消えるが、これは許容済みの仕様である。片軸の合計が 1（上 1・下 0 など）
 * のときは 0/0 になり、四辺すべてが 0 になれば `padsZero` が生成を止める。ここでも
 * 特例は設けない。
 *
 * 純関数かつ冪等 —— 2 回通しても値は動かないので、React の StrictMode が updater を
 * 二度呼んでも結果は同じである。引数は書き換えず、常に新しい {@link Pads} を返す。
 *
 * 対辺ミラー（片辺を動かすと対辺に同値が入る）のほうは**入力の操作規則**なので
 * ここには置かず、`useOutpaintForm` の `setPad` が担う。
 */
export function centerPads(pads: Pads): Pads {
  const vertical = Math.floor((pads.top + pads.bottom) / 2);
  const horizontal = Math.floor((pads.left + pads.right) / 2);
  return { top: vertical, bottom: vertical, left: horizontal, right: horizontal };
}

/** The smallest legal `num_frames` (8n+1, n>=1). Mirrors
 * `modes/single/defaultConfig.ts`'s `MIN_NUM_FRAMES`; duplicated as a local
 * constant to keep this module dependency-free. */
export const MIN_NUM_FRAMES = 9;

/**
 * `num_frames` の上限。`durationSec * fps` を切り捨ててから 8n+1 に丸める
 * （下限 9）。
 *
 * D5 (設計書 §4-4): the duration slider's ceiling follows the SOURCE video, so
 * a generation longer than its guide can never even be dialled in. `durationSec`
 * is a best-effort probe (`fs.probeMediaInfo`), hence the deliberate rounding
 * DOWN at both steps — an unknown/zero duration or fps degrades to the 9-frame
 * floor rather than inventing headroom.
 */
export function maxNumFrames(durationSec: number, fps: number): number {
  if (!Number.isFinite(durationSec) || !Number.isFinite(fps) || durationSec <= 0 || fps <= 0) {
    return MIN_NUM_FRAMES;
  }
  const raw = Math.floor(durationSec * fps);
  if (raw < MIN_NUM_FRAMES) return MIN_NUM_FRAMES;
  return Math.floor((raw - 1) / 8) * 8 + 1;
}

/** 潜在フレーム数（VAE が時間方向に 8 分の 1 へ圧縮した後のコマ数）。
 * `(num_frames - 1) / 8 + 1`. */
export function latentFrameCount(numFrames: number): number {
  if (!Number.isFinite(numFrames) || numFrames <= 1) return 1;
  return Math.floor((numFrames - 1) / 8) + 1;
}

/** 快適上限の推定トークン数: `⌊幅/32⌋ × ⌊高さ/32⌋ × 潜在フレーム数`。
 *
 * 各軸を先に切り捨てるのは `shell/comfortTable.ts` の `comfortFramesForBudget`
 * と同じ形にするため —— 32×32 の画素ブロック1つが潜在1マスなので、端数の画素は
 * マスを増やさない。128 の倍数に載ったキャンバス（生成できる唯一の形）では
 * 切り捨てが効かないので数値は変わらず、値が動くのは 128 格子から外れた**編集
 * 途中**の表示だけである。 */
export function comfortTokenEstimate(width: number, height: number, numFrames: number): number {
  return Math.floor(width / 32) * Math.floor(height / 32) * latentFrameCount(numFrames);
}

/** True once {@link comfortTokenEstimate} passes `budget`. A WARNING only —
 * generation is never blocked on it.
 *
 * 予算は引数で受ける。エンジン系統ごとに違う値であり、その解決は
 * `shell/outpaintBudget.ts` の `resolveOutpaintComfortBudget`（サーバー配信の
 * `limits.comfort_budgets` を引く）が担う —— このモジュールは import ゼロを
 * 保つため、系統名を知らないままでいる。線が無い（同解決器が `null`）ときは
 * 呼び手がこの関数を呼ばず、警告も出さない。 */
export function isOverComfortBudget(width: number, height: number, numFrames: number, budget: number): boolean {
  return comfortTokenEstimate(width, height, numFrames) > budget;
}

/* --- マスクブラー（なじみ幅） ---------------------------------------------
 *
 * The blend between the kept region and the newly drawn area is a Laplacian
 * pyramid composite whose MASK is first shrunk so its long side is 64px,
 * dilated by `r` steps there, and interpolated back up. The band the user
 * actually sees is therefore `r × キャンバス長辺 ÷ 64` full-resolution pixels —
 * a value that depends on the canvas, which is exactly why the old fixed
 * "150px" copy (correct only at 1920px) was wrong everywhere else.
 *
 * 正本はバックエンドの `api/models.py` の `OutpaintSpec`（`0..15`、既定 5 / 2）
 * と `engine/pipeline/outpaint_pipeline.py`。
 */

/** マスクブラー段階の下限／上限（サーバーの `ge=0, le=15` と同じ）。 */
export const BLEND_DILATION_MIN = 0;
export const BLEND_DILATION_MAX = 15;

/** 公式ワークフロー（node 5266 / 5226）と同じ既定値。UI もここから始める。 */
export const DEFAULT_BLEND_DILATION_STAGE1 = 5;
export const DEFAULT_BLEND_DILATION_STAGE2 = 2;

/**
 * 膨張段数 `r` を実寸のなじみ幅（ピクセル）へ直す。`r × 長辺 ÷ 64`。
 *
 * 1920px のキャンバス・既定の 5 段なら 150px —— これまで `OutpaintPreview` に
 * 直書きされていた定数と一致する（つまりこの式はその定数の一般化）。stage 2 は
 * 常に stage 1 以下の幅にしかならない（{@link stage2FromStage1} が 5:2 で連動
 * するため）ので、「見えるなじみ幅」は stage 1 だけで決まる。
 *
 * 不正値・0 以下はすべて 0（＝膨張なし）へ落とす。
 */
export function featherWidthPx(r: number, canvasLongSide: number): number {
  if (!Number.isFinite(r) || !Number.isFinite(canvasLongSide)) return 0;
  if (r <= 0 || canvasLongSide <= 0) return 0;
  return Math.round((r * canvasLongSide) / 64);
}

/**
 * stage 2 の段数を stage 1 から導く。公式の既定値 5 / 2 の比をそのまま保つ
 * （`round(r × 2 ÷ 5)`）。
 *
 * `r === 0` だけは特例で 0 —— 「膨張なし」を選んだのに stage 2 だけ 1 段
 * 膨らむのでは、UI の 0px 表示が嘘になるため。それ以外は 1 を下限にする
 * （比の計算だけだと r=1,2 で 0 になり、stage 2 の合成が段差を残す）。
 */
export function stage2FromStage1(r: number): number {
  if (!Number.isFinite(r) || r <= 0) return 0;
  return Math.max(1, Math.round((r * 2) / 5));
}

/** The upload slot's status, mirroring `modes/chained/useSourceUpload.ts`'s
 * `SourceUploadStatus` structurally (re-declared rather than imported so this
 * module keeps zero imports). */
export type OutpaintSourceStatus = "idle" | "uploading" | "ready" | "error";

/** Everything {@link outpaintReasons} needs. All measurements are raw probe
 * output: `0` means "unknown", never "zero pixels". */
export interface OutpaintReasonsInput {
  /** The panel's own prompt text (untrimmed — the check trims). */
  prompt: string;
  sourceStatus: OutpaintSourceStatus;
  /** `video_id` from `POST /upload/video`, or `null` until one exists. */
  sourceVideoId: string | null;
  /** §1-6: a range trim was requested but the server did not report it. Using
   * the id anyway would generate from the wrong part of the footage. */
  sourceTrimFailed: boolean;
  /** `fs.probeMediaInfo` output for the attached file (`0` = unknown). */
  sourceWidth: number;
  sourceHeight: number;
  sourceDurationSec: number;
  /** The CURRENT pads — absolute pixel amounts, 0 起点・1px 刻み。 */
  pads: Pads;
  /** The `num_frames` about to be sent — free-typed values pass through the
   * field unsnapped (mirroring Create), so the grid is checked here at
   * submit-gate time rather than corrected mid-edit. */
  numFrames: number;
  /** The ceiling {@link maxNumFrames} derived from the CURRENT source video and
   * frame rate (§4-4). Never stored in state by the caller — re-derived every
   * render, so swapping/removing the source can't leave a stale clamp behind. */
  maxNumFrames: number;
  /** Is the `in-outpainting` control LoRA present in `GET /loras`? */
  hasOutpaintLora: boolean;
}

/**
 * 生成可否の理由コード配列。空配列 = 生成可。
 *
 * Shaped exactly like Create/Chain's `validityReasons` so the existing
 * `modes/single/GenerateReasonsNote.tsx` can render it verbatim, with
 * `editReasonMessages.ts` supplying the code→sentence map.
 *
 * Deliberately ABSENT: any token-budget code. 快適上限 (§4-5) is a non-blocking
 * warning banner, not a gate.
 *
 * `canvasWidthOffGrid` と `canvasHeightOffGrid` を 1 本にまとめないこと（オーナー
 * 決定 2026-08-11）。`modes/single/GenerateReasonsNote.tsx` は**解決済みの文言**で
 * de-dup するため、共通文言にすると幅だけ直した段階でも行が残り、次にどちらを
 * 動かせばよいのか分からなくなる。文言が別であること自体が仕様である。
 */
export function outpaintReasons(input: OutpaintReasonsInput): string[] {
  const reasons: string[] = [];

  if (input.prompt.trim().length === 0) reasons.push("promptEmpty");

  const frames = input.numFrames;
  if (
    !Number.isInteger(frames) ||
    (frames - 1) % 8 !== 0 ||
    frames < MIN_NUM_FRAMES ||
    frames > input.maxNumFrames
  ) {
    reasons.push("numFramesOffGrid");
  }

  const attached = input.sourceStatus === "ready" && input.sourceVideoId !== null;
  if (input.sourceStatus === "uploading") {
    reasons.push("sourceUploading");
  } else if (input.sourceStatus === "error") {
    reasons.push("sourceUploadFailed");
  } else if (!attached) {
    // idle, or a "ready" that somehow carries no id — both mean "there is no
    // usable source video", which is the same instruction to the user.
    reasons.push("sourceMissing");
  }

  if (attached) {
    if (input.sourceTrimFailed) reasons.push("sourceTrimFailed");

    const measured = input.sourceWidth > 0 && input.sourceHeight > 0 && input.sourceDurationSec > 0;
    if (!measured) {
      // Without the source's own size there is no canvas to check against the
      // 128 grid (nor a ceiling for the pads), and without its length there is
      // no ceiling for num_frames — the whole panel would be guessing.
      reasons.push("mediaInfoUnknown");
    } else {
      if (input.sourceWidth < MIN_INNER_SIDE || input.sourceHeight < MIN_INNER_SIDE) {
        reasons.push("innerTooSmall");
      }
      if (totalPad(input.pads) === 0) reasons.push("padsZero");

      // 拡張後キャンバスそのものの検査。パッドは自動補正しない方針なので、
      // 「128 の倍数から外れている」はここでしか止められない。`padsZero` との
      // 同時表示は抑制しない —— 3 行とも事実であり、最初の 1 手から幅と高さの
      // 両方を案内できるほうが調整が速い（オーナー決定 2026-08-11）。
      const canvas = canvasSize(input.sourceWidth, input.sourceHeight, input.pads);
      // 4096 の上限も 128 の格子と同じ方式（拡張後キャンバスを測ってメッセージで
      // 止める）へ統一した（第2弾）。入力部品の上限では止めないので、超過の経路は
      // 2 つある —— パッドを盛りすぎた場合と、元動画自体が 4096 を超えている場合。
      // 拡張後キャンバスで測れば、どちらも同じ 1 つの機構で拾える。
      //
      // ちょうど 4096 は合法なので `>` で測る（4096 は 128 の倍数でもあるため、
      // 境界で OffGrid が道連れに出ることもない）。4096 超かつ 128 非倍数のときに
      // 同じ軸で 2 行出るのは抑制しない —— どちらも本当に直す必要がある。
      if (canvas.width > MAX_CANVAS_SIDE) reasons.push("canvasWidthTooLarge");
      if (canvas.height > MAX_CANVAS_SIDE) reasons.push("canvasHeightTooLarge");
      if (canvas.width % CANVAS_MULTIPLE !== 0) reasons.push("canvasWidthOffGrid");
      if (canvas.height % CANVAS_MULTIPLE !== 0) reasons.push("canvasHeightOffGrid");
    }
  }

  if (!input.hasOutpaintLora) reasons.push("loraMissing");

  return reasons;
}
