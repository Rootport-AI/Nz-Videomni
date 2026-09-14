/**
 * Edit タブ（Outpainting）の「なぜ生成ボタンが押せないのか」文言マップ。
 *
 * Deliberately the same shape as `modes/chained/generateReasonMessages.ts`'s
 * `buildChainReasonMessages(strings, opts) -> Record<string, string>`: the
 * renderer on the other end is the very same component
 * (`modes/single/GenerateReasonsNote.tsx`), which takes a list of codes plus a
 * code→sentence map and skips any code it has no entry for. Keeping the two
 * builders structurally identical means a reader who has seen one already knows
 * this one.
 *
 * Kept dependency-light for the same reason as Chain's: the i18n dictionary type
 * and nothing else. No React, no hooks — the caller passes the `useStrings()`
 * result in.
 *
 * NOTE — there is deliberately NO token-budget code here. 快適上限 (設計方針書
 * §4-5) is surfaced as a non-blocking warning banner inside the panel, not as a
 * Generate gate, and the shared token-budget module is another workstream's
 * territory.
 */
import type { Strings } from "../../i18n/strings";
import { MIN_INNER_SIDE } from "./outpaintGeometry";

/** The one runtime value two of the lines interpolate. Taken as a parameter
 * object (rather than read off the form hook's result) so this module has no
 * dependency on `useOutpaintForm`'s own type. */
export interface OutpaintReasonMessageParams {
  /** The control LoRA the panel pins (`lora/controlLoras.OUTPAINT_LORA_NAME`,
   * `"in-outpainting"`), named in the `loraMissing` line so the user knows
   * exactly which file to install. */
  loraName: string;
  /** The smallest kept side the server accepts. Defaults to
   * {@link MIN_INNER_SIDE} — the only value production ever passes. */
  minInnerSide?: number;
}

/**
 * Builds the `code -> line` map `GenerateReasonsNote` consumes for
 * `outpaintGeometry.outpaintReasons`' output.
 *
 * `promptEmpty` reuses Create's shared line (identical instruction, identical
 * wording); everything else is Outpainting-specific copy under
 * `strings.edit.outpainting.generateReasons`.
 *
 * The return type is the loose `Record<string, string>` the note expects, not a
 * `Record<OutpaintReasonCode, string>` — the note takes a plain map and a caller
 * may merge extra codes in.
 */
export function buildOutpaintReasonMessages(
  strings: Strings,
  { loraName, minInnerSide = MIN_INNER_SIDE }: OutpaintReasonMessageParams,
): Record<string, string> {
  const t = strings.edit.outpainting.generateReasons;
  return {
    promptEmpty: strings.single.generateReasons.promptEmpty,
    // Free-typed frame counts pass through the field unsnapped exactly as they
    // do on Create, so the same line applies verbatim.
    numFramesOffGrid: strings.single.generateReasons.numFramesOffGrid,
    sourceMissing: t.sourceMissing,
    sourceUploading: t.sourceUploading,
    sourceUploadFailed: t.sourceUploadFailed,
    // §1-6: the trim was asked for but not applied, so the stored upload is the
    // WHOLE file — generating would extend the wrong part of the footage.
    sourceTrimFailed: t.sourceTrimFailed,
    mediaInfoUnknown: t.mediaInfoUnknown,
    padsZero: t.padsZero,
    // 2026-08-11: パッドが 1px 刻みになったので、拡張後キャンバスが 128 の格子から
    // 外れうる。幅と高さで別の行にすること —— `GenerateReasonsNote` は解決済みの
    // 文言で de-dup するため、共通の 1 文にすると片方だけ直した段階でも行が残る。
    // 他タブの `dimensionsOffGrid`（幅・高さ 1 行まとめ）を借りないのはこのため。
    canvasWidthOffGrid: t.canvasWidthOffGrid,
    canvasHeightOffGrid: t.canvasHeightOffGrid,
    // 第2弾（2026-08-11）: 4096 の上限も 128 の格子と同じくメッセージで担保する
    // ようになった（スライダー上限は定数 220、数値ボックスは 4096 固定で、どちらも
    // キャンバス寸法を見ていないため）。ここも幅・高さで別の行にする。
    canvasWidthTooLarge: t.canvasWidthTooLarge,
    canvasHeightTooLarge: t.canvasHeightTooLarge,
    innerTooSmall: t.innerTooSmall(minInnerSide),
    loraMissing: t.loraMissing(loraName),
  };
}

/**
 * §1-17 Retake 版。`buildOutpaintReasonMessages` と同じ形（strings を受けて
 * `code -> 一文` の表を返すだけ）で、同じ `GenerateReasonsNote` が描く。
 *
 * Outpainting と違って `promptEmpty` は入っていない —— 撮り直しは「同じものを
 * もう一度」も正当な使い方なので、プロンプト空欄はブロック理由にしない
 * （`useRetakeForm` の `validityReasons` にもそのコードは存在しない）。
 */
export function buildRetakeReasonMessages(strings: Strings): Record<string, string> {
  const t = strings.edit.retake.generateReasons;
  return {
    sourceMissing: t.sourceMissing,
    sourceUploading: t.sourceUploading,
    sourceUploadFailed: t.sourceUploadFailed,
    sourceTrimFailed: t.sourceTrimFailed,
    rangeUnusable: t.rangeUnusable,
    outOfMaterial: t.outOfMaterial,
    // 幅・高さを編集可能にした（2026-08-10）ので、Create/Chain と同じ格子ゲートが
    // Retake にも要る。文言も同じもの（同じ指示・同じ言い回し）を借りる。
    dimensionsOffGrid: strings.single.generateReasons.dimensionsOffGrid,
  };
}

/** {@link buildInpaintReasonMessages} が `resolutionMismatch` の 1 行へ差し込む
 * 実測値（オーナー裁定 D5 の文面には 4 つの数字が入る）。
 *
 * 引数オブジェクトで受けるのは `buildOutpaintReasonMessages` と同じ理由 ——
 * このモジュールが `useInpaintForm` の型に依存しないようにするため。素材寸が
 * 読めていないときは `0` が来るが、そのときは `mediaInfoUnknown` が先に立つので
 * この行は出ない（`GenerateReasonsNote` は理由コードに載った行しか描かない）。 */
export interface InpaintReasonMessageParams {
  /** 部分フィルタの解像度＝プロジェクトの解像度（`getEditInfo`）。 */
  filterWidth: number;
  filterHeight: number;
  /** 対象動画の実寸（選択スナップショットの `mediaWidth`/`mediaHeight`）。 */
  targetWidth: number;
  targetHeight: number;
}

/**
 * 台帳 §3-55 Inpainting 版。上の 2 つとまったく同じ形（strings と実測値を受けて
 * `code -> 一文` の表を返すだけ）で、同じ `GenerateReasonsNote` が描く。
 *
 * `promptEmpty` は入っていない —— プロンプト空欄は正当な使い方（物体を消すだけ
 * なら書くことが無い）で、監督が置いた既定のとおりブロック理由にしない。
 * `useInpaintForm` の `validityReasons` にもそのコードは存在しない。
 */
export function buildInpaintReasonMessages(
  strings: Strings,
  { filterWidth, filterHeight, targetWidth, targetHeight }: InpaintReasonMessageParams,
): Record<string, string> {
  const t = strings.edit.inpainting.generateReasons;
  return {
    partialFilterMissing: t.partialFilterMissing,
    targetMissing: t.targetMissing,
    sourceUploading: t.sourceUploading,
    sourceUploadFailed: t.sourceUploadFailed,
    sourceTrimFailed: t.sourceTrimFailed,
    mediaInfoUnknown: t.mediaInfoUnknown,
    resolutionMismatch: t.resolutionMismatch(filterWidth, filterHeight, targetWidth, targetHeight),
    windowNotCovered: t.windowNotCovered,
    maskRendering: t.maskRendering,
  };
}
