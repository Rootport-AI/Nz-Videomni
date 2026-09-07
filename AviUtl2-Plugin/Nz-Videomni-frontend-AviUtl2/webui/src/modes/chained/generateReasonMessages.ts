/**
 * Chainの「なぜGenerateが押せないのか」文言マップ。元は`ChainedScreen.tsx`内の
 * ローカル定数だったものを、そのままの内容で純関数として切り出した。
 *
 * The Chain screen's `ChainValidityReason` -> localized imperative line map,
 * lifted verbatim out of `ChainedScreen.tsx` (where it used to be an inline
 * `const generateReasonMessages` next to the `return`). Behaviour is unchanged
 * — same codes, same strings, same order.
 *
 * Why it moved: Batch i2v-long (§1-7, `modes/batch-i2v-long/`) blocks Start
 * while the Chain form itself is invalid, and expands `chain.validityReasons`
 * ONE BY ONE rather than showing an opaque "the chain settings are invalid"
 * (which would leave the user with no way to find out what to fix). Both
 * screens must show the same sentence for the same code, so the map has to be
 * reachable from outside `ChainedScreen`'s render body.
 *
 * Kept dependency-light on purpose: the i18n dictionary type plus one pure
 * constant from `chainUtils`. No React, no hooks — the caller passes the
 * `useStrings()` result in.
 */
import type { Strings } from "../../i18n/strings";
import { MAX_CHAIN_TOTAL_FRAMES } from "./chainUtils";

/** The two form-derived numbers two of the lines interpolate. Taken as a
 * parameter object rather than read off a `UseChainFormResult` so this module
 * has no dependency on the Chain form's own type — Batch i2v-long only ever
 * sees the structural `ChainSnapshotSource`, which carries both fields for
 * exactly this call. */
export interface ChainReasonMessageParams {
  /** `UseChainFormResult.minFramesForOverlap`. */
  minFramesForOverlap: number;
  /** `UseChainFormResult.minClips`. */
  minClips: number;
  /** §1-14/§3-57: `UseChainFormResult.stage2MaxContextFrames` — the carry-over
   * ceiling of the CURRENTLY selected finishing-pass step. Optional so Batch
   * i2v-long's structural `ChainSnapshotSource` callers keep compiling without
   * it; omitted -> the standard step's 161, which never binds (the server caps
   * `context_frames` at 145 anyway), so the line simply never fires. */
  stage2MaxContextFrames?: number;
  // 素材（末尾）v2 (2026-08-15): `stage2MaxEndContextFrames` and
  // `endContextRequiredLastClipFrames` were removed from this params object
  // together with the two gates that interpolated them. Neither number exists on
  // the form any more — the band is derived from the material, so there is no
  // window ceiling to name and no last clip to lengthen.
}

/**
 * Builds the `code -> line` map `GenerateReasonsNote` consumes for Chain's
 * `validityReasons`.
 *
 * `promptEmpty`/`dimensionsOffGrid`/`cropInvalid`/`nagNegativeEmpty` and (§1-15)
 * the three reference lines Create words identically reuse Create's shared
 * copy; `minClips`/`totalFramesExceeded`/`sourceNotReady` reuse Chain's
 * existing field-adjacent banner copy (those banners stay where they are — this
 * note is an additional at-a-glance summary next to the button); the rest have
 * Chain-specific `generateReasons` copy of their own.
 *
 * The return type is the loose `Record<string, string>` `GenerateReasonsNote`
 * expects (it skips codes it has no entry for), not a `Record<
 * ChainValidityReason, string>` — Batch i2v-long merges this map with its own
 * block-reason codes before handing it over.
 */
export function buildChainReasonMessages(
  strings: Strings,
  { minFramesForOverlap, minClips, stage2MaxContextFrames = 161 }: ChainReasonMessageParams,
): Record<string, string> {
  return {
    promptEmpty: strings.single.generateReasons.promptEmpty,
    dimensionsOffGrid: strings.single.generateReasons.dimensionsOffGrid,
    cropInvalid: strings.single.generateReasons.cropInvalid,
    clipFramesOffGrid: strings.chained.generateReasons.clipFramesOffGrid,
    clipTooShortForOverlap: strings.chained.generateReasons.clipTooShortForOverlap(minFramesForOverlap),
    minClips: strings.chained.minClipsError(minClips),
    totalFramesExceeded: strings.chained.totalFramesExceeded(MAX_CHAIN_TOTAL_FRAMES),
    sourceUploading: strings.chained.generateReasons.uploadInFlight,
    sourceNotReady: strings.chained.sourceVideo.contextFramesError,
    // §1-6 (source trim): both lines describe the SOURCE VIDEO's material, and
    // both name a concrete way out (shorten the carry-over / lengthen the
    // range; update the plugin / re-pick the file).
    sourceVideoTooShortForContext: strings.chained.generateReasons.sourceVideoTooShortForContext,
    sourceTrimFailed: strings.chained.generateReasons.sourceTrimFailed,
    // §1-14/§3-57: also a source-video line (it only fires with one attached),
    // so it joins CHAIN_SOURCE_REASON_CODES below.
    contextFramesTooLongForWindow:
      strings.chained.generateReasons.contextFramesTooLongForWindow(stage2MaxContextFrames),
    // §1-16 長尺A2V: one line per audio code. `audioUploading` reuses Create's
    // sentence verbatim (same upload, same wording) rather than duplicating it
    // into `chain.generateReasons`; every other line is Chain-specific because
    // it names a Chain-only remedy (the ↔️ auto-fit button, the Single screen).
    audioConflictsWithSourceVideo: strings.chained.generateReasons.audioConflictsWithSourceVideo,
    audioUploading: strings.single.generateReasons.audioUploading,
    audioNotReady: strings.chained.generateReasons.audioNotReady,
    audioTooShort: strings.chained.generateReasons.audioTooShort,
    audioTooShortForChain: strings.chained.generateReasons.audioTooShortForChain,
    audioTooLong: strings.chained.generateReasons.audioTooLong,
    chainLayoutInvalid: strings.chained.generateReasons.chainLayoutInvalid,
    // §1-15 参照動画. The three lines that say exactly what Create's say — same
    // rule, same remedy, same upload — reuse Create's copy verbatim rather than
    // duplicating it; the rest are Chain-specific because they name a
    // Chain-only constraint (the 128 grid with a reference attached, the
    // V2V clash, the multi-clip depth limit).
    referenceNeedsLoras: strings.single.generateReasons.referenceNeedsLoras,
    controlLoraNeedsReference: strings.single.generateReasons.attachReferenceVideo,
    referenceUploading: strings.single.generateReasons.referenceUploading,
    referenceNotReady: strings.chained.generateReasons.referenceNotReady,
    // §1-15 W4 (2026-08-11): the reference slot's own trim failure. Chain-
    // specific copy rather than Create's: it sits next to `referenceNotReady`
    // and is worded as its twin ("attach it again, because …"), which is the
    // shape the rest of this block reads in.
    referenceTrimFailed: strings.chained.generateReasons.referenceTrimFailed,
    referenceConflictsWithSourceVideo: strings.chained.generateReasons.referenceConflictsWithSourceVideo,
    referenceDimensionsOffGrid: strings.chained.generateReasons.referenceDimensionsOffGrid,
    depthChainUnsupported: strings.chained.generateReasons.depthChainUnsupported,
    // 素材（末尾）: one line per end-source code, in the same order
    // `useChainForm` pushes them (the two conflicts first — with a second
    // slot filled nothing else about the end source matters until it's
    // removed — then the upload's own state, then the geometry rules).
    endSourceConflictsWithAudio: strings.chained.generateReasons.endSourceConflictsWithAudio,
    endSourceConflictsWithReference: strings.chained.generateReasons.endSourceConflictsWithReference,
    endSourceUploading: strings.chained.generateReasons.endSourceUploading,
    endSourceNotReady: strings.chained.generateReasons.endSourceNotReady,
    endSourceTrimFailed: strings.chained.generateReasons.endSourceTrimFailed,
    // The v1 geometry lines became these — two about the material's own
    // length, then the rules about the CHAIN the anchor is frozen into (the
    // seam-blend width, and — 逆順Chained, 2026-08-18 — its 2+-clip audio-
    // budget counterpart).
    endSourceTooShort: strings.chained.generateReasons.endSourceTooShort,
    endSourceLengthUnknown: strings.chained.generateReasons.endSourceLengthUnknown,
    endSourceNeedsOverlap: strings.chained.generateReasons.endSourceNeedsOverlap,
    endSourceAudioOverlapBudget: strings.chained.generateReasons.endSourceAudioOverlapBudget,
    // NAG (2026-07-28): shares Create's copy — `useChainForm`'s own
    // `validityReasons` pushes this code identically to `useGenerationForm`'s.
    nagNegativeEmpty: strings.single.generateReasons.nagNegativeEmpty,
  };
}

/**
 * The subset of {@link buildChainReasonMessages}' codes that describe the
 * SOURCE VIDEO slot. Batch i2v-long suppresses these while it is already
 * showing its own `sourceVideoAttached` block line ("clear the source video"),
 * so the user is not told twice — once vaguely, once precisely — about the same
 * attachment. Exported (rather than inlined in the batch) so a future
 * source-related `ChainValidityReason` is added in one obvious place.
 */
export const CHAIN_SOURCE_REASON_CODES: readonly string[] = [
  "sourceUploading",
  "sourceNotReady",
  // §1-6: both can only ever be raised while a source video IS attached, which
  // is precisely the state the batch's own line is already telling the user to
  // undo.
  "sourceVideoTooShortForContext",
  "sourceTrimFailed",
  // §1-14/§3-57: same rule — only raisable with a source video attached.
  "contextFramesTooLongForWindow",
];

/**
 * §1-16 長尺A2V: the subset of {@link buildChainReasonMessages}' codes that
 * describe the SOURCE AUDIO slot — the exact audio-side counterpart of
 * {@link CHAIN_SOURCE_REASON_CODES}, and there for the same consumer.
 *
 * Batch i2v-long has no audio slot of its own: it strips `source_audio` from the
 * payload it builds and blocks Start with its own line while one is attached, so
 * expanding the Chain form's audio reasons underneath that would tell the user
 * to fix things about a track the batch is not going to send. Suppressing this
 * whole set keeps the batch's own line the single explanation.
 */
export const CHAIN_AUDIO_REASON_CODES: readonly string[] = [
  "audioConflictsWithSourceVideo",
  "audioUploading",
  "audioNotReady",
  "audioTooLong",
  "audioTooShortForChain",
  "audioTooShort",
  "chainLayoutInvalid",
];

/**
 * §1-15 参照動画: the reference-video counterpart of
 * {@link CHAIN_AUDIO_REASON_CODES}, for the same consumer and the same reason —
 * Batch i2v-long strips `reference_video_id` (and the two strengths) from every
 * row it sends and blocks Start with its own `referenceVideoAttached` line, so
 * expanding the Chain form's reference gates underneath that would ask the user
 * to fix things about material the batch is not going to send.
 *
 * `referenceDimensionsOffGrid` is deliberately NOT in the set: an off-grid
 * width/height is a property of the CHAIN, not of the attachment, and it stays
 * wrong after the reference is removed (the 64 grid still rejects, say, 700) —
 * so it must keep being reported.
 */
/**
 * 素材（末尾）v2 (2026-08-15): the END-source counterpart of
 * {@link CHAIN_REFERENCE_REASON_CODES}, for the same consumer and the same
 * reason — Batch i2v-long strips `end_source` from every row it sends and blocks
 * Start with its own `endSourceAttached` line, so expanding the Chain form's
 * end-source gates underneath that would ask the user to fix material the batch
 * is not going to send.
 *
 * `endSourceNeedsOverlap` and `endSourceAudioOverlapBudget` ARE in the set,
 * unlike the reference block's `referenceDimensionsOffGrid`: both fire only
 * while an end source is attached (a 1-frame seam blend and a particular clip
 * layout are perfectly ordinary on their own), so removing the material really
 * does clear them.
 */
export const CHAIN_END_SOURCE_REASON_CODES: readonly string[] = [
  "endSourceConflictsWithAudio",
  "endSourceConflictsWithReference",
  "endSourceUploading",
  "endSourceNotReady",
  "endSourceTrimFailed",
  "endSourceTooShort",
  "endSourceLengthUnknown",
  "endSourceNeedsOverlap",
  "endSourceAudioOverlapBudget",
];

export const CHAIN_REFERENCE_REASON_CODES: readonly string[] = [
  "referenceConflictsWithSourceVideo",
  "referenceUploading",
  "referenceNotReady",
  // §1-15 W4: only raisable while a reference IS attached — exactly the state
  // the batch's own line already tells the user to undo.
  "referenceTrimFailed",
  "referenceNeedsLoras",
  "controlLoraNeedsReference",
  "depthChainUnsupported",
];
