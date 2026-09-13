/**
 * Inpainting の「時間の窓」を決める純関数（台帳 §3-55、オーナー裁定 D3）。
 *
 * 窓の定義はたったひとつ:「部分フィルタの開始フレームを起点に、フレーム数欄の
 * 値だけ」。それが素材リボンの尻尾からはみ出すときだけ頭側へずらし、それでも
 * 収まらなければ**生成不可**（`null`）。窓を勝手に縮めることはしない —— 長さは
 * フレーム数欄が持つ唯一の正であって、ここが黙って別の長さへ書き換えると、
 * 画面の数字と生成される長さが食い違う。
 *
 * ## `modes/edit/` に置く理由（敵対的レビュー m4）
 *
 * 快適上限の式（`comfortTokenEstimate`）と同じ層に置いて、層の逆転を避ける。
 * `timeline/` は「タイムラインの座標と素材の秒」を扱う層で、ここが決めるのは
 * Inpainting というモードの窓の規則 —— タイムライン一般の話ではない。
 *
 * ## import ゼロ
 *
 * `outpaintGeometry.ts` と同じく、このモジュールは何も import しない。窓の
 * 規則はこの1ファイルを読めば全部わかる、という状態を保つため。
 */

/** フレーム数欄の下限（Single と同じ 8n+1 格子の最小値）。 */
export const INPAINT_FRAMES_MIN = 9;

/**
 * フレーム数欄の上限。サーバーの `num_frames: Field(49, ge=9, le=481)` と同じ
 * 481 で、**これが硬い上限**（オーナー裁定 D12）。快適上限マーカーは助言専用で、
 * 生成を止めることは無い。
 */
export const INPAINT_FRAMES_MAX = 481;

/** 8n+1 の格子（Single・Chained・Retake と共通）。 */
const FRAME_GRID = 8;

/**
 * `frames` を 8n+1 の格子へ**切り上げる**。
 *
 * 既存の `timeline/retakeWindow.ts` の `snapRetakeWindowFrames` は切り**下げ**
 * なので流用できない（D3 は「部分フィルタの長さを 8n+1 へ切り上げた値」を既定に
 * すると言っている——切り下げると、部分フィルタの尻尾がマスクの外に出る）。
 *
 * 下限 {@link INPAINT_FRAMES_MIN} でクランプする。上限はここではクランプ
 * **しない** —— 既定値を作る {@link defaultInpaintFrames} だけが 481 で頭を
 * 打つ話で、この関数自体は格子の写像に徹する。
 */
export function roundUpTo8n1(frames: number): number {
  if (!Number.isFinite(frames)) return INPAINT_FRAMES_MIN;
  if (frames <= INPAINT_FRAMES_MIN) return INPAINT_FRAMES_MIN;
  const steps = Math.ceil((frames - 1) / FRAME_GRID);
  return steps * FRAME_GRID + 1;
}

/**
 * 部分フィルタが来たときに、フレーム数欄へ入れる既定値。
 *
 * 「部分フィルタの長さ（閉区間なので `end − start + 1`）を 8n+1 へ切り上げ、
 * 481 でクランプ」。481 に当たった場合は部分フィルタより窓が短くなるので、
 * {@link placeInpaintWindow} の `coversFilter` が `false` になり、パネルは
 * 「窓の外のマスクは描き替えられない」注意文を出す。
 */
export function defaultInpaintFrames(filterStart: number, filterEnd: number): number {
  const length = filterEnd - filterStart + 1;
  return Math.min(INPAINT_FRAMES_MAX, roundUpTo8n1(length));
}

export interface PlaceInpaintWindowArgs {
  /** 部分フィルタの開始フレーム（プロジェクト絶対フレーム）。 */
  filterStart: number;
  /** 部分フィルタの終了フレーム（絶対・**閉区間**）。 */
  filterEnd: number;
  /** 対象動画リボンの開始フレーム（絶対）。 */
  ribbonStart: number;
  /** 対象動画リボンの終了フレーム（絶対・**閉区間**）。 */
  ribbonEnd: number;
  /** フレーム数欄の値（窓の長さ）。 */
  frames: number;
}

export interface InpaintWindow {
  /** 窓の開始フレーム（絶対）。 */
  windowStart: number;
  /** 窓の終了フレーム（絶対・閉区間）。 */
  windowEnd: number;
  /** 窓の長さ（＝要求された `frames` そのもの。縮めない）。 */
  frames: number;
  /** 尻尾がリボンからはみ出したので頭側へずらしたか。 */
  shiftedHead: boolean;
  /** 部分フィルタ全体が窓の内側に収まっているか。`false` なら「窓の外の
   * マスクは描き替えられない」注意文を出す（ブロックはしない・D3）。 */
  coversFilter: boolean;
}

/**
 * 窓を置く。置けなければ `null`。
 *
 * 手順:
 *  1. 入力の健全性（有限な整数・リボンと部分フィルタの向き）。壊れていれば `null`
 *  2. 部分フィルタ自体がリボンの外（交差しない）なら `null` —— 対象動画の上に
 *     載っていない部分フィルタからはマスクを作れない
 *  3. 窓がリボンより長ければ `null`（収まらない）
 *  4. 起点は `filterStart`。尻尾が `ribbonEnd` を越えるなら
 *     `ribbonEnd − frames + 1` まで頭側へずらす
 *  5. 頭がリボンの頭を割るなら `ribbonStart` へ寄せる（3 を通っているので
 *     このとき尻尾は必ずリボン内）
 *
 * `coversFilter` は「部分フィルタの両端が窓の内側」。窓を短くした（D3 の自由な
 * フレーム数欄）ときと、頭側へずらして部分フィルタの尻尾が外へ出たときの
 * **両方**でここが `false` になる。
 */
export function placeInpaintWindow(args: PlaceInpaintWindowArgs): InpaintWindow | null {
  const { filterStart, filterEnd, ribbonStart, ribbonEnd, frames } = args;
  if (![filterStart, filterEnd, ribbonStart, ribbonEnd, frames].every(Number.isFinite)) return null;
  if (frames < 1) return null;
  if (filterEnd < filterStart) return null;
  if (ribbonEnd < ribbonStart) return null;
  // 部分フィルタがリボンと一切重なっていない。どこを描き替えるのか決まらない。
  if (filterEnd < ribbonStart || filterStart > ribbonEnd) return null;

  const ribbonFrames = ribbonEnd - ribbonStart + 1;
  if (frames > ribbonFrames) return null;

  let windowStart = filterStart;
  let shiftedHead = false;
  if (windowStart + frames - 1 > ribbonEnd) {
    windowStart = ribbonEnd - frames + 1;
    shiftedHead = true;
  }
  if (windowStart < ribbonStart) {
    // 3 のガードを通っているので、ここへ来るのは「部分フィルタがリボンの頭より
    // 前から始まっている」場合だけ。寄せても尻尾は必ずリボン内に収まる。
    windowStart = ribbonStart;
    shiftedHead = false;
  }
  const windowEnd = windowStart + frames - 1;

  return {
    windowStart,
    windowEnd,
    frames,
    shiftedHead,
    coversFilter: filterStart >= windowStart && filterEnd <= windowEnd,
  };
}

/**
 * 素材の実寸を 128 の倍数へ**切り上げた**キャンバス寸。
 *
 * `POST /generate` の `width`/`height` に載る値で、余白（右・下）はサーバーが
 * 素材の実寸から導出する（敵対的レビュー C1: 正本は素材ファイルの実寸ひとつ）。
 * 快適上限マーカーもこのキャンバス寸で評価する。
 *
 * 0 以下（＝素材寸が不明）は 0 のまま返す —— ここで既定値へ落とすと、寸法を
 * 読めていないことがゲート（`mediaInfoUnknown`）まで届かなくなる。
 */
export function roundUpTo128(px: number): number {
  if (!Number.isFinite(px) || px <= 0) return 0;
  return Math.ceil(px / 128) * 128;
}
