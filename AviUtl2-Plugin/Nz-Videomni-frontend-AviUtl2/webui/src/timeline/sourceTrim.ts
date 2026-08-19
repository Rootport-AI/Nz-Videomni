/**
 * §1-6 V2Vリボン範囲トリムの判定コア。「タイムライン上のリボンが元動画ファイルの
 * 一部しか占めていないとき、その範囲だけを切り出してアップロードしてよいか」を
 * 保守的に判定する純関数群。
 *
 * The decision core for §1-6 "V2V ribbon range trim": given one selected
 * timeline object and the project selection it came from, decide whether the
 * WebUI may ask the backend to cut the uploaded source video down to the ribbon's
 * own range (`/upload/video?trim_start_sec=…&trim_duration_sec=…`) instead of
 * sending the whole backing file.
 *
 * Everything here is PURE and free of React/i18n/bridge imports, for two
 * reasons:
 *  - it is called from three very different places — the right-click guard
 *    (`menuSelection.guardMenuSelection`), the Chain screen's auto-load path
 *    (`modes/chained/ChainedScreen.tsx`) and the upload itself
 *    (`modes/chained/useSourceUpload.ts`) — and all three must agree exactly;
 *  - #2 `referenceVideo` (IC-LoRA reference video) has the same "the ribbon is
 *    a window into a longer file" problem and is expected to reuse these very
 *    functions once its own trim is enabled. Nothing here is V2V-specific: the
 *    inputs are an object + a selection, the output is a start/duration pair.
 *
 * ## The conservative bias
 *
 * Every uncertainty resolves to NO TRIM, i.e. the pre-§1-6 behavior of uploading
 * the whole file. A wrongly-skipped trim costs the user nothing they did not
 * already have; a wrongly-APPLIED trim would silently generate from the wrong
 * part of the video. Hence: unknown media duration, an unresolvable span, a span
 * that already covers the whole file (within one project frame), a non-neutral
 * playback speed, loop playback, a multi-section (中間点) object, and an
 * unreported playback range all return `{trim: false, reason}` with the reason
 * preserved for diagnostics/tests.
 *
 * ## The unit (real-hardware finding, 2026-08-01)
 *
 * AviUtl2's 再生位置 item is a 4-field CSV — `開始,終了,再生範囲,0`, e.g.
 * `"2.000,10.700,再生範囲,0"` — whose first two fields are the start and end of
 * the played window in SECONDS on the SOURCE's own time axis, with 3 decimals.
 * The unit does NOT depend on the project fps (the same 2-second head trim reads
 * `2.000` at both 30fps and 24fps), and an untouched ribbon writes
 * `0.000,<full duration>` explicitly rather than omitting the item. Native parses
 * those two numbers (`ParsePlaybackRange`) and reports them as
 * `playbackStartSec`/`playbackEndSec` + the `hasPlaybackRange` flag; this module
 * consumes them directly. See Docs/V2V_RIBBON_TRIM_WORKORDER.md §4.
 */

import type { SelectionItem, TimelineSelection } from "./menuSelection";
import { spanDurationSec } from "./prefillSeed";

/**
 * A selected timeline object carrying the playback fields the §1-6 trim decision
 * needs. Now a plain re-export of the bridge contract's own selection item: the
 * six v10 fields moved into `bridge/types.ts` when the real-hardware probe landed
 * (2026-08-01), exactly as this alias's predecessor predicted. It is kept as a
 * name because the three call sites read better saying "a trim selection item",
 * and because it marks which fields a caller is expected to have populated.
 */
export type TrimSelectionItem = SelectionItem;

/**
 * The neutral 再生速度. A speed other than this makes the mapping from timeline
 * seconds to source seconds non-identity, so the ribbon's span no longer equals
 * the source range to cut — we skip rather than guess.
 */
export const PLAYBACK_SPEED_NEUTRAL = 1.0;

/** How far from {@link PLAYBACK_SPEED_NEUTRAL} still counts as neutral. Native
 * divides AviUtl2's percentage string (`"100.00"`) by 100, so the value that
 * reaches us is a computed double rather than a literal `1`; comparing exactly
 * would let a rounding crumb masquerade as a speed change and needlessly skip
 * every trim. The tolerance is far tighter than any speed a user can dial in
 * (the item has 2 decimal places, i.e. steps of 0.0001 on this scale). */
export const PLAYBACK_SPEED_EPSILON = 1e-6;

/** Why {@link decideSourceTrim} declined to trim. Every one of these means
 * "upload the whole file, exactly like before §1-6". */
export type SourceTrimSkipReason =
  /** No selected object at all. */
  | "noItem"
  /** `mediaDurationSec <= 0` — native could not probe the backing file, so
   * there is nothing to compare the span against. */
  | "unknownMediaDuration"
  /** The ribbon's own length is unresolvable: project `rate`/`scale` <= 0, or a
   * non-positive frame span. */
  | "unresolvableSpan"
  /** The ribbon already spans (essentially) the whole file — trimming would be
   * a no-op re-encode. Also the "measurement error" bucket: the comparison is
   * biased so that anything within one project frame of the full duration
   * lands here. */
  | "spanCoversWholeMedia"
  /** 再生速度 ≠ 1.0: timeline time no longer maps 1:1 onto source time. */
  | "nonNeutralSpeed"
  /** ループ再生 is on: the ribbon replays its window, so a single cut out of the
   * source cannot reproduce what is on the timeline. */
  | "loopEnabled"
  /** The object has 2+ sections (中間点), so one start/duration pair cannot
   * describe which part of the source it plays. */
  | "multipleSections"
  /** No playback range was reported — an older plugin build, a non-video object,
   * or an object whose 再生位置 item native could not read/parse. Kept under its
   * original name (it predates the probe) so no caller has to churn. */
  | "unknownPlaybackPosition";

/** The decision: either "upload the whole file" (with the reason why) or "cut
 * `[startSec, startSec + durationSec)` out of the source first". */
export type SourceTrimDecision =
  | { trim: false; reason: SourceTrimSkipReason }
  | { trim: true; startSec: number; durationSec: number };

/**
 * The source-time offset (seconds) the ribbon starts playing from.
 *
 * 実機調査で秒確定（2026-08-01）: AviUtl2's 再生位置 reports its window's start
 * directly in SECONDS on the source's own time axis, so this is an identity read
 * of `playbackStartSec` — no fps conversion, at either end. The function is kept
 * (rather than inlined) because it is the single documented place the unit is
 * asserted, and `selection` stays in the signature so a future object type whose
 * offset IS frame-based has somewhere to convert.
 *
 * A missing value reads as 0; callers only reach here after
 * {@link decideSourceTrim} has confirmed `hasPlaybackRange`.
 */
export function playbackStartSec(
  item: TrimSelectionItem,
  _selection: Pick<TimelineSelection, "rate" | "scale">,
): number {
  return item.playbackStartSec ?? 0;
}

/**
 * Decide whether the source video for this selected object may be trimmed to the
 * ribbon's range before upload (正本 §3.2). Pure.
 *
 * The checks run in this order, each falling through to the next only when it
 * cannot decide "no":
 *  1. there is an object at all;
 *  2. its backing file's duration is known (`mediaDurationSec > 0`);
 *  3. its ribbon span is resolvable ({@link spanDurationSec});
 *  4. the span is MEANINGFULLY shorter than the file — `spanSec <
 *     mediaDurationSec - 1/projectFps`. The one-frame slack is deliberately on
 *     the NO-TRIM side: an object placed at full length can report a span that
 *     differs from `mediaDurationSec` by a rounding crumb, and re-encoding it
 *     would be pure loss;
 *  5. 再生速度 is neutral (or unreported);
 *  6. ループ再生 is off (or unreported);
 *  7. the object is single-section (or the section count is unreported);
 *  8. a playback RANGE was actually reported.
 *
 * Only then does it return `trim: true`. The window is
 * `[playbackStartSec, playbackStartSec + durationSec)` where
 * `durationSec = min(spanSec, playbackEndSec - playbackStartSec)`, then clamped
 * to the end of the source. The `min` is not belt-and-braces — BOTH sides of it
 * are load-bearing on real hardware (see the R4 finding, 正本 §4):
 *  - the ribbon can be LONGER than the window AviUtl2 actually plays (dragging
 *    the start rightwards leaves the ribbon's length untouched, so its tail is a
 *    frozen last frame). Taking `spanSec` there would cut footage that never
 *    appears on the timeline;
 *  - the window can be a hair longer than the ribbon (the source's frames are
 *    quantized to the project fps, so `end - start` can exceed the span by up to
 *    ~1 frame at 24fps). Taking `end - start` there would overshoot.
 * The smaller of the two is right in both directions.
 */
export function decideSourceTrim(
  item: TrimSelectionItem | undefined,
  selection: Pick<TimelineSelection, "rate" | "scale">,
): SourceTrimDecision {
  if (!item) return { trim: false, reason: "noItem" };

  const mediaDurationSec = item.mediaDurationSec;
  if (!(mediaDurationSec > 0)) return { trim: false, reason: "unknownMediaDuration" };

  const spanSec = spanDurationSec(item, selection);
  if (spanSec === undefined) return { trim: false, reason: "unresolvableSpan" };

  // Condition 3, written so that measurement error falls on the no-trim side:
  // the span must be shorter than the file by MORE than one project frame.
  const projectFps = selection.rate / selection.scale;
  const oneFrameSec = 1 / projectFps;
  if (!(spanSec < mediaDurationSec - oneFrameSec)) {
    return { trim: false, reason: "spanCoversWholeMedia" };
  }

  // A reported, non-neutral 再生速度 breaks the 1:1 timeline<->source mapping.
  // An UNREPORTED speed (older native, or an un-probed build) is not a skip — the overwhelming
  // majority of objects are at 1.0, and the full-file fallback is still one step
  // away at `unknownPlaybackPosition` anyway.
  // Native normalizes 再生速度 to a 1.0 scale (the raw item is a percentage
  // string), so compare against 1.0 with a tolerance rather than exactly: a
  // value that went through "100.00" / 100 need not be bit-identical to 1.
  if (
    item.playbackSpeed !== undefined &&
    Math.abs(item.playbackSpeed - PLAYBACK_SPEED_NEUTRAL) > PLAYBACK_SPEED_EPSILON
  ) {
    return { trim: false, reason: "nonNeutralSpeed" };
  }

  if (item.loopPlay === true) {
    return { trim: false, reason: "loopEnabled" };
  }

  if (item.sectionCount !== undefined && item.sectionCount >= 2) {
    return { trim: false, reason: "multipleSections" };
  }

  if (item.hasPlaybackRange !== true) {
    return { trim: false, reason: "unknownPlaybackPosition" };
  }

  const rawStart = playbackStartSec(item, selection);
  const startSec = Number.isFinite(rawStart) && rawStart > 0 ? rawStart : 0;
  // Degenerate: the reported start is at/after the end of the file. Nothing
  // sensible to cut — fall back to the whole file rather than invent a window.
  if (startSec >= mediaDurationSec) return { trim: false, reason: "unresolvableSpan" };

  // The played window as AviUtl2 reports it. A non-finite/absent end degrades to
  // the ribbon's own span, i.e. the pre-R4 behavior.
  const rawEnd = item.playbackEndSec;
  const windowSec =
    rawEnd !== undefined && Number.isFinite(rawEnd) ? rawEnd - startSec : spanSec;
  const durationSec = Math.min(spanSec, windowSec, mediaDurationSec - startSec);
  if (!(durationSec > 0)) return { trim: false, reason: "unresolvableSpan" };

  return { trim: true, startSec, durationSec };
}

/**
 * Format a seconds value for the `/upload/video` query string. `toFixed(3)` is
 * chosen over `String(sec)` specifically to kill EXPONENTIAL notation: a tiny
 * value like `1e-7` would otherwise serialize as `"1e-7"`, which the backend's
 * float parse may or may not accept — `"0.000"` always does. Millisecond
 * resolution is well below the server's own frame-level rounding
 * (`round(start_sec * src_fps)`), so nothing is lost.
 */
export function formatTrimSeconds(sec: number): string {
  return sec.toFixed(3);
}

/**
 * The query parameters to hand `backend.uploadFile` for this decision, or
 * `undefined` when no trim applies.
 *
 * `undefined` (not `{}`) is load-bearing: the caller spreads it as
 * `...(query ? { query } : {})`, so a no-trim upload's request params are
 * byte-identical to the pre-§1-6 ones — no `query` key is even present. That
 * identity is machine-checked by `ChainedScreen.prefill.test.tsx`'s strict
 * `toHaveBeenCalledWith` assertion.
 */
export function trimQuery(decision: SourceTrimDecision): Record<string, string> | undefined {
  if (!decision.trim) return undefined;
  return {
    trim_start_sec: formatTrimSeconds(decision.startSec),
    trim_duration_sec: formatTrimSeconds(decision.durationSec),
  };
}
