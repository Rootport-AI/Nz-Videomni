/**
 * バッチi2v-long（§1-7）がChainフォームから読む最小面。`modes/chained/`への
 * importはこの型経由で回避する。
 *
 * The minimal surface Batch i2v-long reads from the Chain form. Every import
 * from `modes/chained/` is avoided by going through this structural type — the
 * batch feature never names `useChainForm`/`chainUtils` at all, so the Chain
 * screen stays completely unmodified (it just happens to satisfy this shape)
 * and this module's tests can be written against a ~10-line fake.
 *
 * `UseChainFormResult` (`modes/chained/useChainForm.ts`) is structurally
 * assignable to this type; that assignability is pinned by a compile-time
 * assertion in `buildI2vLongPayload.test.ts` so a future Chain refactor that
 * renames/retypes any of these fields fails the type check instead of
 * silently breaking the batch at runtime.
 */
import type { GenerateChainRequest } from "../../api/types";

export interface ChainSnapshotSource {
  /** The Chain form's complete `POST /generate/chain` body for the CURRENT
   * form state — the template every batch row is derived from
   * (`buildI2vLongPayload.buildI2vLongTemplate`). A `useCallback`, so its
   * identity is stable enough to be a `useMemo` dependency. */
  buildRequest: () => GenerateChainRequest;
  /** `"v2v"` once a source video is attached. Batch i2v-long blocks on this
   * (`blockReasons.sourceVideoAttached`): `buildRequest` drops clip 0's
   * `conditioning_images` in V2V mode, so a row's image would be silently
   * ignored. */
  mode: "scratch" | "v2v";
  /** §1-16 長尺A2V: `true` once an audio track is attached to the Chain form.
   * Batch i2v-long blocks on this too (`blockReasons.sourceAudioAttached`):
   * `buildI2vLongTemplate` strips `source_audio`, so the batch would otherwise
   * run a whole overnight job silently ignoring the track the user attached.
   * Optional so the ~10-line fakes in this feature's own tests (written before
   * the audio slot existed) keep compiling — an omitted value reads as "no
   * audio", which is what a chain without the field means. */
  hasSourceAudio?: boolean;
  /** §1-15 参照動画: `true` once a reference video is attached to the Chain form.
   * Batch i2v-long blocks on this for the same reason as the two slots above
   * (`blockReasons.referenceVideoAttached`): `buildI2vLongTemplate` strips
   * `reference_video_id` and both strengths, so the batch would otherwise run a
   * whole overnight job silently ignoring the reference the user attached.
   * Optional for the same reason as {@link hasSourceAudio} — an omitted value
   * reads as "no reference video". */
  hasReferenceVideo?: boolean;
  /** 素材（末尾）v2 (2026-08-15): `true` once end material is attached to the
   * Chain form. Batch i2v-long blocks on this for the same reason as the two
   * slots above (`blockReasons.endSourceAttached`): `buildI2vLongTemplate` strips
   * `end_source`, so the batch would otherwise run a whole overnight job silently
   * ignoring the material the user attached — and every row would come out
   * shorter than the Chain screen's own predicted output said. Optional for the
   * same reason as {@link hasSourceAudio}. */
  hasEndSource?: boolean;
  /** The clip list — read only for its length (`clips < 2` would 422, since a
   * source-less chain needs at least 2 clips) and for detecting per-clip
   * prompt overrides, which batch prompts deliberately do NOT rewrite. */
  clips: readonly { readonly prompt: string }[];
  /** The Chain form's own Generate gate; batch Start requires it too. */
  isValid: boolean;
  /** The structured breakdown behind `isValid`, expanded one-by-one in the
   * batch's block-reason list (an opaque "chain settings are invalid" would
   * leave the user with no way to find out why Start is disabled). */
  validityReasons: readonly string[];
  /** Predicted output length per image, for the `chainSummary` readout. */
  outputFrames: number;
  outputSeconds: number;
  /** 第3弾 addition. The two numbers `modes/chained/generateReasonMessages.ts`
   * interpolates into the `minClips` / `clipTooShortForOverlap` lines. The
   * batch expands {@link validityReasons} using the CHAIN SCREEN'S OWN wording,
   * so it needs the Chain form's own current values here — deriving them again
   * on the batch side would be exactly the kind of re-judgment this type exists
   * to avoid, and would print a wrong frame count the moment the overlap
   * slider moves. Both are already public on `UseChainFormResult`, so the
   * compile-time assignability pin in `buildI2vLongPayload.test.ts` still
   * holds. */
  minClips: number;
  minFramesForOverlap: number;
}
