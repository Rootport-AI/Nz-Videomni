/**
 * §1-17 Retake（選択範囲の撮り直し）の**窓**を決める判定コア。
 *
 * 「タイムライン上で選んだ時間範囲」を、元の動画ファイル自身の時間軸へ写し、
 * さらに Retake が実際に生成できる形（長さ 8n+1・[73, 169] フレーム）へ整える
 * までを担当する。ここで決まった窓が、そのまま
 *  - バックエンドへ渡す `window_start_sec` / 窓長、
 *  - 画面の RangeBand（`modes/edit/RangeBand.tsx`）が描く帯、
 *  - 生成後に置き直す仮オブジェクトの尺
 * の共通の出どころになる。
 *
 * ## 純関数であること（`sourceTrim.ts` と同じ規律）
 *
 * React も i18n も bridge も import しない。理由は sourceTrim.ts と同じで、
 * この判定が「右クリックのガード」「Retake パネル」「送信直前の再検証」という
 * 別々の場所から呼ばれ、三者が一字一句同じ答えを出す必要があるため。
 *
 * ## 窓の定義（実装計画 §1）
 *
 * 窓 ＝ 選択範囲を 8n+1 へスナップし [73, 169] へクランプしたもの。
 * 糊代（のりしろ）は**窓の内側**にあり、リボン全長とは無関係。
 * 73 と 169 の出どころは backend の `chain_math` 側の幾何で、169 は
 * 「stage-2 のタイル1枚に収まる最大の画素フレーム数」（`outputs/retake_spike/
 * T1_RESULTS.md` C1 の実測）、73 は自由中間潜在が成立する最小窓。
 *
 * 上限 169 は Stage-2 のクリップ長が既定（潜在22フレーム）のときの値で、
 * ユーザーが潜在19フレームを選ぶと 145 へ縮む（{@link retakeMaxWindowPx}）。
 * 実際に効かせる上限は呼び出し側が `maxFrames` で渡す。
 */

import type { TimelineSelection } from "./menuSelection";
import { normalizeSelectionRange } from "./selectionRange";
import type { TimelineFrameRange } from "./selectionRange";
import {
  PLAYBACK_SPEED_EPSILON,
  PLAYBACK_SPEED_NEUTRAL,
  decideSourceTrim,
  formatTrimSeconds,
} from "./sourceTrim";
import type { SourceTrimSkipReason, TrimSelectionItem } from "./sourceTrim";

/**
 * 窓の最小・最大フレーム数（生成 fps 上の画素フレーム）。
 *
 * 将来 `AppConfig.limits` の `retake_window_min_frames` /
 * `retake_window_max_frames` で上書きできるようにする予定（実装計画 B4 が
 * サーバ側 `LimitsConfig` に 2 定数だけを公開する）。そのときに備えて、
 * {@link resolveRetakeWindow} と `RangeBand` は**この定数を直接読まず**、
 * 引数・props の既定値としてだけ受け取る形にしてある —— 差し替え点は
 * 「呼び出し側が既定値を渡さない」1 箇所に閉じている。
 */
export const RETAKE_WINDOW_MIN_PX = 73;
export const RETAKE_WINDOW_MAX_PX = 169;

/** 窓長の格子（8n+1）の刻み幅。 */
export const RETAKE_WINDOW_GRID = 8;

/**
 * stage-2 のタイル1枚（`vTile` 潜在フレーム）に収まる最大の画素フレーム数
 * ＝ `px_from_v_latent(vTile)` = `8·vTile − 7`。backend の
 * `chain_math.retake_max_window_px(v_tile)` のミラー。
 *
 * Retake は窓ぜんぶを stage-2 の**1タイル**で精錬するので、これがそのまま
 * 窓長の上限になる。`standard`（vTile 22）で 169 = {@link RETAKE_WINDOW_MAX_PX}、
 * `high_resolution`（vTile 19）で 145。つまり Stage-2 のクリップ長を短い方へ
 * 切り替えると、撮り直せる最長区間もその分だけ縮む。
 *
 * `shell/tokenBudget.ts` の `STAGE2_WINDOW_PRESETS` を**ここから import しない**
 * のは、このモジュールが依存ゼロの判定コアだから（冒頭 doc の「純関数であること」）。
 * `vTile` は呼び出し側（`modes/edit/useRetakeForm.ts`）が渡す。
 */
export function retakeMaxWindowPx(vTile: number): number {
  return (vTile - 1) * RETAKE_WINDOW_GRID + 1;
}

/**
 * 開閉区間の規約（＝未確定の閉区間仮説）は依存ゼロの葉
 * `timeline/selectionRange.ts` が持っている。ここから再 export しているのは
 * 後方互換のため —— この 3 つを `retakeWindow` から import している呼び出し側は
 * そのままでよい。**仮説を直す場所は葉のほうの 1 行**。
 *
 * 葉へ出した理由は循環 import の回避（`prefillSeed` も同じ規約を要るため）。
 * 詳細は `selectionRange.ts` の冒頭コメント。
 */
export { normalizeSelectionRange, selectionRangeFrames } from "./selectionRange";
export type { HalfOpenFrameRange, TimelineFrameRange } from "./selectionRange";

/**
 * 素材秒への写像を諦めた理由。
 *
 * `sourceTrim.ts` の 8 種をそのまま引き継ぎ、範囲固有の 2 種を足したもの。
 * 引き継ぎ側の意味は「タイムライン時間 ↔ 素材時間 の 1:1 写像が保証できない」
 * で統一されている。ただし `spanCoversWholeMedia` だけは意味が変わるので
 * {@link mapRangeToMaterialSec} の説明を読むこと。
 */
export type RetakeRangeSkipReason =
  | SourceTrimSkipReason
  /** 範囲が空（または逆転している）。 */
  | "emptyRange"
  /** 範囲とオブジェクトのリボンが交差しない。 */
  | "rangeOutsideObject";

export interface RetakeRangeMapping {
  /** 窓の開始（**元ファイル自身の時間軸**の秒）。 */
  startSec: number;
  /** 交差後の範囲の長さ（秒）。 */
  durationSec: number;
  /** 交差後の範囲の長さ（プロジェクトフレーム）。 */
  frameCount: number;
  /** 交差クランプ後のプロジェクトフレーム区間 `[startFrame, endFrameEx)`。
   * 仮オブジェクトの置き直し位置に使う。 */
  startFrame: number;
  endFrameEx: number;
  /** `selection.rate / selection.scale`。 */
  projectFps: number;
  /**
   * アップロードが §1-6 のトリム付きで行われる場合の切り出し開始秒
   * （`decideSourceTrim` の `startSec`）。トリムしないときは 0。
   *
   * F3 が送信値を組み立てるときに使う: サーバが受け取る `video_id` が
   * **トリム済み**ファイルを指すなら、その中での窓開始は
   * `startSec - trimOffsetSec` になる。元ファイル直切りなら `startSec` のまま。
   */
  trimOffsetSec: number;
  /**
   * 使える素材の**始端**（素材の時間軸の秒）＝このリボンが実際に再生を始める位置。
   * {@link resolveRetakeWindow} の `materialStartSec` へそのまま渡す。
   */
  materialStartSec: number;
  /**
   * 使える素材の**終端**（素材の時間軸の秒）＝このリボンが実際に再生を終える位置。
   * {@link resolveRetakeWindow} の `materialDurationSec` へそのまま渡す。
   *
   * ファイル全長ではなく**実際に再生されている窓の終わり**なのが要点。
   * AviUtl2 では「頭を右へ引っぱるとリボンの長さは変わらず尾が静止フレームになる」
   * （`sourceTrim.ts` の R4 所見）ため、リボンの尻尾がファイルの続きを再生している
   * とは限らない。ファイル全長を上限にすると、**画面に一度も出ていない映像**を
   * 撮り直しの糊代や詰めものに使ってしまう。
   */
  materialDurationSec: number;
}

export type RetakeRangeResult =
  | { ok: false; reason: RetakeRangeSkipReason }
  | ({ ok: true } & RetakeRangeMapping);

/**
 * 選択範囲を、素材ファイル自身の時間軸上の `[startSec, +durationSec)` へ写す。
 *
 * 手順:
 *  1. `item` がある（無ければ `noItem`）
 *  2. 範囲を半開へ正規化（{@link normalizeSelectionRange}）。空/逆転は `emptyRange`
 *  3. `decideSourceTrim` で写像の健全性を確認（下記）。ここが先なのは、
 *     交差の相手（＝実際に再生されている区間）がこの判定の副産物だから
 *  4. リボンの**再生されている区間**との交差クランプ:
 *     `s = max(rangeStart, objStart)` / `e = min(rangeEndEx, objPlayedEndEx)`。
 *     交差しなければ `rangeOutsideObject`
 *  5. `sourceSec = playbackStart + (windowStartFrame - item.frameStart) / projectFps`
 *
 * ## 3 の「健全性」——`spanCoversWholeMedia` の扱いが最重要
 *
 * `decideSourceTrim` は「アップロード時に切り詰めてよいか」を答える関数なので、
 * その `trim: false` の理由 8 種のうち **`spanCoversWholeMedia` だけは
 * Retake にとって障害ではない**。「リボンがファイル全長を占めている」は
 * 「切り詰める必要がない」の意味であって、写像が壊れているわけではないからだ。
 *
 * ただし**無条件では許せない**。全長リボンには 2 種類ある:
 *  - 再生開始が 0 の素直な全長リボン → 起点 0 で写せる。許可。
 *  - 再生開始が 0 でない全長リボン → AviUtl2 では「頭を右へ引っぱるとリボンの
 *    長さはそのままで尾が静止フレームになる」（`sourceTrim.ts` の R4 所見）ため、
 *    長さが全長のままでも再生窓は 2 秒目から、という状態が実在する。ここで
 *    起点 0 と決めつけると、**ユーザーが指した場所とは別の場所を無言で撮り直す**。
 *    最悪の故障なので弾く。
 *
 * よって除外の追加条件は `hasPlaybackRange !== true || (playbackStartSec ?? 0) === 0`。
 * `hasPlaybackRange !== true`（再生位置を読めなかった古い native 等）を許すのは、
 * リボンが全長を占めている以上、非 0 の再生開始はそもそもファイルに収まらない
 * ——「読めない」と「長さが全長」が同時に成り立つなら起点は 0 しかない —— から。
 *
 * この条件を満たさない全長リボンは `spanCoversWholeMedia` の理由で弾かれる。
 * つまり**この理由が呼び出し側へ返るのは危険な方のケースだけ**であり、
 * 文言は「再生位置がずらされているので撮り直せない」の意味で書いてよい。
 *
 * ### 全長リボンでは残り 3 種を**自分でやり直す**（敵対的レビュー指摘 2026-08-09）
 *
 * `decideSourceTrim` は `spanCoversWholeMedia` を**再生速度・ループ・中間点より
 * 先に**判定して即 return する（`sourceTrim.ts` の 176-207 行）。つまり全長リボンでは
 * その 3 つのチェックに**到達しない**。上の除外を素通しにすると、たとえば
 * 「0〜0.5 秒をループ再生して 2 秒のリボンを埋めている全長オブジェクト」が
 * 無条件に通り、1.5 秒地点を選んだつもりが素材の 1.5 秒（実際に映っているのは
 * 0.0 秒）を撮り直す —— この関数が防ごうとしている故障そのものが別の口から入る。
 * そこで {@link fullLengthRibbonSkipReason} が同じしきい値・同じ順序で 3 つを
 * やり直す。
 *
 * 残る 7 種（`noItem` / `unknownMediaDuration` / `unresolvableSpan` /
 * `nonNeutralSpeed` / `loopEnabled` / `multipleSections` /
 * `unknownPlaybackPosition`）はそのまま弾く。
 *
 * ## 交差クランプは「実際に再生されている範囲」まで
 *
 * 交差の相手はリボンの全区間ではなく、リボンのうち**実際に素材を再生している
 * 区間**（R4 の静止フレーム尻尾を除いた部分）。`decideSourceTrim` が
 * `min(spanSec, playbackEnd - start, ファイル残り)` として既に計算している値を
 * そのまま使う。静止フレームの上を選ぶと、そこに対応する素材秒はもう存在しない
 * ——「その位置に映っているのは直前のコマの停止画」なので、素直に写すと
 * 存在しない続きを撮り直してしまう。
 */
export function mapRangeToMaterialSec(
  item: TrimSelectionItem | undefined,
  selection: Pick<TimelineSelection, "rate" | "scale">,
  range: TimelineFrameRange,
): RetakeRangeResult {
  if (!item) return { ok: false, reason: "noItem" };

  const { startFrame: rangeStart, endFrameEx: rangeEndEx } = normalizeSelectionRange(range);
  if (!Number.isFinite(rangeStart) || !Number.isFinite(rangeEndEx) || rangeEndEx <= rangeStart) {
    return { ok: false, reason: "emptyRange" };
  }

  const decision = decideSourceTrim(item, selection);
  if (!decision.trim) {
    if (decision.reason !== "spanCoversWholeMedia") return { ok: false, reason: decision.reason };
    const masked = fullLengthRibbonSkipReason(item);
    if (masked !== undefined) return { ok: false, reason: masked };
  }

  // ここまで来ていれば `decideSourceTrim` が rate/scale と mediaDurationSec を
  // 検証済み（どちらが欠けても `unresolvableSpan` / `unknownMediaDuration` で
  // 上で返っている）。
  const projectFps = selection.rate / selection.scale;

  // 起点秒。トリムが立つときの `decision.startSec` は sourceTrim が
  // 「再生位置は秒である」という単位宣言をしている唯一の場所
  // （`playbackStartSec()`）を 0 下限で正規化した値そのもの。ここで
  // `playbackStartSec()` を呼び直さないのは、同じ正規化（有限性チェックと
  // 0 下限）を二重に書かないため —— 単位の宣言点は sourceTrim 側 1 箇所に保つ。
  // トリムが立たない = 上の「素直な全長リボン」だけなので起点は 0。
  const offsetSec = decision.trim ? decision.startSec : 0;
  // 実際に再生されている長さ。トリムが立つときは sourceTrim が既に
  // `min(spanSec, 再生窓, ファイル残り)` として出しているのでそれを使い、
  // 全長リボンのときは再生窓の終わり（報告があれば）かファイル全長。
  const playedSec = decision.trim ? decision.durationSec : fullLengthPlayedSec(item);
  if (!(playedSec > 0)) return { ok: false, reason: "unresolvableSpan" };

  // 交差クランプ。`objEndEx` の `+ 1` は `frameEnd` が閉区間という**確立済み**の
  // 事実（`prefillSeed.spanDurationSec`）に基づくもので、範囲側の仮説とは
  // 別物 —— F0 が確定しても、この行は動かさない。`objPlayedEndEx` の方は
  // 「静止フレームの尻尾を切り落とした、本当に素材が流れている終端」。
  const objStart = item.frameStart;
  const objEndEx = item.frameEnd + 1;
  const objPlayedEndEx = Math.min(objEndEx, objStart + Math.round(playedSec * projectFps));
  const startFrame = Math.max(rangeStart, objStart);
  const endFrameEx = Math.min(rangeEndEx, objPlayedEndEx);
  if (endFrameEx <= startFrame) return { ok: false, reason: "rangeOutsideObject" };

  const rawStartSec = offsetSec + (startFrame - objStart) / projectFps;
  // ミリ秒へ丸める。`offsetSec`（native が 3 桁で報告した秒）と
  // `フレーム差 / fps`（有理数）という出どころの違う 2 つの和なので、素のままだと
  // 1e-16 級のごみが乗る。丸め幅 1ms は最短フレーム（60fps で 16.6ms）の
  // 1/16 以下なので、どのフレームを指すかは変わらない。`formatTrimSeconds` を
  // 使うのは、同じ値がやがて同じ書式でサーバへ渡るため（指数表記も同時に殺せる）。
  const startSec = Number(formatTrimSeconds(rawStartSec));

  const frameCount = endFrameEx - startFrame;
  return {
    ok: true,
    startSec,
    durationSec: frameCount / projectFps,
    frameCount,
    startFrame,
    endFrameEx,
    projectFps,
    trimOffsetSec: decision.trim ? decision.startSec : 0,
    materialStartSec: offsetSec,
    materialDurationSec: offsetSec + playedSec,
  };
}

/**
 * 全長リボン（`decideSourceTrim` が `spanCoversWholeMedia` で早期 return した
 * ケース）について、その early return が**追い越してしまった**残りのチェックを
 * やり直す。弾くべきなら理由を、通してよければ `undefined` を返す。
 *
 * 順序としきい値は `decideSourceTrim`（`sourceTrim.ts` 183-207 行）と同一。
 * 定数（`PLAYBACK_SPEED_NEUTRAL` / `PLAYBACK_SPEED_EPSILON`）も import して
 * 使っており、値をここへ写してはいない —— 片方だけ直る事故を避けるため。
 */
function fullLengthRibbonSkipReason(item: TrimSelectionItem): SourceTrimSkipReason | undefined {
  if (
    item.playbackSpeed !== undefined &&
    Math.abs(item.playbackSpeed - PLAYBACK_SPEED_NEUTRAL) > PLAYBACK_SPEED_EPSILON
  ) {
    return "nonNeutralSpeed";
  }
  if (item.loopPlay === true) return "loopEnabled";
  if (item.sectionCount !== undefined && item.sectionCount >= 2) return "multipleSections";
  // 再生開始が 0 でない全長リボン（R4）。ここだけは `spanCoversWholeMedia` の
  // 名前のまま返す —— 呼び出し側にこの理由が届くのはこのケースだけなので、
  // 文言は「再生位置がずらされている」の意味で書いてよい。
  if (item.hasPlaybackRange === true && (item.playbackStartSec ?? 0) !== 0) {
    return "spanCoversWholeMedia";
  }
  return undefined;
}

/**
 * 全長リボンが実際に再生している長さ（秒）。再生窓の終わりが報告されていれば
 * それ（ファイル全長で頭打ち）、無ければファイル全長。起点が 0 であることは
 * {@link fullLengthRibbonSkipReason} が保証済み。
 */
function fullLengthPlayedSec(item: TrimSelectionItem): number {
  const rawEnd = item.playbackEndSec;
  if (item.hasPlaybackRange === true && rawEnd !== undefined && Number.isFinite(rawEnd)) {
    return Math.min(rawEnd, item.mediaDurationSec);
  }
  return item.mediaDurationSec;
}

/**
 * 窓長を 8n+1 の格子へ**切り下げ**る。
 *
 * 切り上げでなく切り下げなのは、窓が選択範囲からはみ出さないようにするため
 * （足りない側は {@link resolveRetakeWindow} が `padded` として明示的に伸ばす）。
 * 1 未満は 1 に落ちる。
 */
export function snapRetakeWindowFrames(frames: number): number {
  if (!Number.isFinite(frames)) return 1;
  const steps = Math.floor((frames - 1) / RETAKE_WINDOW_GRID);
  return steps <= 0 ? 1 : steps * RETAKE_WINDOW_GRID + 1;
}

export interface ResolveRetakeWindowArgs {
  /** 窓の開始（素材の時間軸の秒）。{@link mapRangeToMaterialSec} の `startSec`。 */
  startSec: number;
  /** ユーザーが選んだ長さ（秒）。 */
  durationSec: number;
  /** 生成 fps。窓長のフレーム数はこの fps 上で数える。 */
  genFps: number;
  /**
   * `startSec` と**同じ時間軸**で測った、使える素材の終端（秒）。
   * {@link mapRangeToMaterialSec} の同名フィールドをそのまま渡す。
   *
   * 「窓がここを越えたら素材が足りない」の 1 本の線。これが
   * **リボンの実トリム後尺が窓に足りないケースの事前検出**そのもの。
   */
  materialDurationSec: number;
  /**
   * `startSec` と同じ時間軸で測った、使える素材の**始端**（秒）。既定 0。
   * {@link mapRangeToMaterialSec} の `materialStartSec` をそのまま渡す。
   *
   * 窓を素材の頭側へずらすときの下限。0 のままにすると、頭がトリムされた
   * リボン（再生開始が 2 秒目のオブジェクト）で詰めものが**リボンより前**の
   * 映像 —— 画面に出ていない部分 —— へ食い込む。
   */
  materialStartSec?: number;
  /** 窓長の下限。既定 {@link RETAKE_WINDOW_MIN_PX}。 */
  minFrames?: number;
  /** 窓長の上限。既定 {@link RETAKE_WINDOW_MAX_PX}。 */
  maxFrames?: number;
}

/** 窓を作れなかった理由。 */
export type RetakeWindowSkipReason =
  /** 生成 fps が 0 以下／非有限。 */
  | "unresolvableFps"
  /** 素材が窓の下限より短い（あるいは全長不明）。伸ばす先が無い。 */
  | "outOfMaterial";

export type RetakeWindowResult =
  | { ok: false; reason: RetakeWindowSkipReason }
  | {
      ok: true;
      /** 確定した窓の開始（素材の時間軸の秒）。 */
      startSec: number;
      /** 確定した窓の長さ（秒）＝ `frames / genFps`。 */
      durationSec: number;
      /** 確定した窓の長さ（生成フレーム）。必ず 8n+1 かつ `[min, max]`。 */
      frames: number;
      /** 上限で切り詰めた（選んだ範囲より窓が**短い**）。 */
      clamped: boolean;
      /** 下限まで伸ばした（選んだ範囲より窓が**長い** = 選んでいない所も
       * 撮り直される）。 */
      padded: boolean;
      /** 素材の終端に当たって開始を手前へずらした。 */
      shifted: boolean;
    };

/**
 * 選択範囲（秒）から、実際に生成できる窓を決める。
 *
 * 規則（実装計画 §4）:
 *  1. 長さを生成フレームへ直し、8n+1 へ**切り下げ**
 *  2. `[minFrames, maxFrames]` へクランプ。上限側は**頭アンカー** ——
 *     開始はそのままで尾だけ切る（`clamped`）
 *  3. 下限に満たなければ下限まで伸ばす（`padded`）
 *  4. 尾が素材の終端を越えるなら、越えた分だけ開始を手前へずらす（`shifted`）。
 *     ずらす先は `materialStartSec` まで。ずらしても入らない = 使える素材が
 *     窓より短い → `outOfMaterial`
 *
 * 開始秒は**量子化しない**。生成 fps の格子へ丸めると継ぎ目が最大で半フレーム
 * ずれるが、窓の長さだけがフレーム数で決まればよく、切り出し位置は秒のままで
 * 困らないため（サーバも `window_start_sec` を秒で受ける）。
 */
export function resolveRetakeWindow(args: ResolveRetakeWindowArgs): RetakeWindowResult {
  const {
    startSec,
    durationSec,
    genFps,
    materialDurationSec,
    materialStartSec = 0,
    minFrames = RETAKE_WINDOW_MIN_PX,
    maxFrames = RETAKE_WINDOW_MAX_PX,
  } = args;

  if (!Number.isFinite(genFps) || genFps <= 0) return { ok: false, reason: "unresolvableFps" };
  const lowerSec = Number.isFinite(materialStartSec) ? Math.max(0, materialStartSec) : 0;
  if (!Number.isFinite(materialDurationSec) || materialDurationSec <= lowerSec) {
    return { ok: false, reason: "outOfMaterial" };
  }

  const snapped = snapRetakeWindowFrames(Math.round(durationSec * genFps));
  const clamped = snapped > maxFrames;
  const padded = snapped < minFrames;
  const frames = clamped ? maxFrames : padded ? minFrames : snapped;

  const windowSec = frames / genFps;
  const maxStartSec = materialDurationSec - windowSec;
  // 使える素材そのものが窓より短い。頭へ寄せても尾へ寄せても入らない。
  if (maxStartSec < lowerSec) return { ok: false, reason: "outOfMaterial" };

  const requestedStartSec = Number.isFinite(startSec) ? startSec : lowerSec;
  const resolvedStartSec = Math.min(Math.max(lowerSec, requestedStartSec), maxStartSec);

  return {
    ok: true,
    startSec: resolvedStartSec,
    durationSec: windowSec,
    frames,
    clamped,
    padded,
    // 「言われた場所から動かした」かどうか。素材の頭側で止めた場合も含む。
    shifted: resolvedStartSec !== requestedStartSec,
  };
}
