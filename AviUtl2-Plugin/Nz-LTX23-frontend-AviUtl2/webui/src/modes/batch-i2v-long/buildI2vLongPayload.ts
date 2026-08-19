/**
 * バッチi2v-long（§1-7）のペイロード組み立て。Chain画面の`buildRequest()`が
 * 返す1本分の完全な`GenerateChainRequest`をテンプレートとして受け取り、
 * 行ごとに変わる「clip 0の画像1点」と「その行のプロンプト」だけを差し替える。
 *
 * Batch i2v-long's payload assembly, built on the template-snapshot design:
 * the Chain form's own `buildRequest()` produces one complete, already-valid
 * `POST /generate/chain` body, and this module (a) turns it into a row-neutral
 * TEMPLATE and (b) stamps each row's image onto clip 0 and each row's own
 * prompt onto the chain-wide `prompt`.
 *
 * This deliberately does NOT re-implement `chainUtils.buildChainRequest`'s
 * calling convention the way Batch A2V's `buildA2vChainPayload` mirrors the
 * Python reference: a chain here has a user-chosen clip count, per-clip frame
 * counts and overlap parameters, so a second hand-maintained builder would be
 * a re-invention with a permanent drift risk. Taking the Chain screen's own
 * output as the template means every future Chain field (crop, NAG/VSF,
 * chunked upsample, LoRAs, …) flows through untouched for free.
 *
 * Everything here is pure and copy-on-write: `base`/`template` are NEVER
 * mutated, and shared sub-objects (`loras`, `crop_output`, clips 1..n) are
 * passed through by reference — so nothing in this module may mutate them
 * either.
 */
import type { GenerateChainRequest } from "../../api/types";

/** How a ROW's own prompt combines with the Chain form's shared prompt. Same
 * two-value vocabulary — and the same one-radio-pair-for-the-whole-batch
 * placement — as Batch A2V's row prompt mode
 * (`modes/batch/buildA2vChainPayload.ts`). */
export type PromptMode = "add" | "replace";

/**
 * Combines the Chain form's shared prompt with ONE ROW's own prompt. Local
 * implementation of the exact rule Batch A2V uses — `composeRowPrompt`
 * (`modes/batch/buildA2vChainPayload.ts`:43-49), itself a mirror of
 * `gradio_ui.batch.compose_prompt`:
 * - an empty/whitespace-only row prompt -> the base prompt verbatim,
 *   regardless of `mode`;
 * - `mode === "replace"` -> the row prompt verbatim;
 * - `mode === "add"` -> `` `${base} ${row}` `` with both ends trimmed.
 *
 * Duplicated rather than imported so this feature keeps its A2V dependency at
 * exactly zero modules (the two panels live on different screens and are
 * expected to diverge); the shared behavior is pinned by both files' tests
 * asserting the same three-branch table.
 *
 * Like A2V's version, the row prompt is passed through unexamined: a
 * `<lora:...>` tag typed into it stays plain text and is never parsed — the
 * Chain form already stripped its own prompt's tags into `loras[]` before
 * `buildRequest()` ever produced `base`.
 *
 * 2026-07-30: the second argument used to be the panel's single batch-wide
 * prompt textarea. That control is gone (owner feedback: a prompt that applies
 * to every image is just the shared prompt written twice) — the text now comes
 * from `I2vLongRow.prompt`, one per image.
 */
export function composeBatchPrompt(basePrompt: string, rowPrompt: string, mode: PromptMode): string {
  const base = basePrompt ?? "";
  const row = rowPrompt ?? "";
  if (row.trim().length === 0) return base;
  if (mode === "replace") return row;
  return `${base} ${row}`.trim();
}

export interface BuildI2vLongTemplateParams {
  /** One complete chain body straight from the Chain form's `buildRequest()`
   * (`ChainSnapshotSource.buildRequest`). Never mutated. */
  base: GenerateChainRequest;
}

/** Returns `base` without its `source_video` key (the same object when the key
 * was already absent, so the common scratch-mode path allocates nothing
 * extra). Key ORDER of everything else is preserved by the spread. */
function withoutSourceVideo(base: GenerateChainRequest): GenerateChainRequest {
  if (!("source_video" in base)) return base;
  const next = { ...base };
  delete next.source_video;
  return next;
}

/** Returns `base` without its `source_audio` key (§1-16 長尺A2V; same
 * allocation-free shortcut as {@link withoutSourceVideo} when the key is
 * absent, which is every batch that never touched Chain's audio card). */
function withoutSourceAudio(base: GenerateChainRequest): GenerateChainRequest {
  if (!("source_audio" in base)) return base;
  const next = { ...base };
  delete next.source_audio;
  return next;
}

/** Returns `base` without its §1-15 reference-video keys — `reference_video_id`
 * and the two strengths that only ever ride WITH it
 * (`conditioning_attention_strength`/`reference_video_strength`). Same
 * allocation-free shortcut as {@link withoutSourceVideo} when nothing is
 * attached, which is every batch that never touched Chain's reference card.
 *
 * All three go together deliberately: leaving the strengths behind would send
 * adapter parameters for an adapter that is no longer there, which the server
 * rejects (both fields require `reference_video_id`). */
function withoutReferenceVideo(base: GenerateChainRequest): GenerateChainRequest {
  if (!("reference_video_id" in base)) return base;
  const next = { ...base };
  delete next.reference_video_id;
  delete next.conditioning_attention_strength;
  delete next.reference_video_strength;
  return next;
}

/** Returns `base` without its 素材（末尾）`end_source` key (v2, 2026-08-15). Same
 * allocation-free shortcut as {@link withoutSourceVideo} when nothing is
 * attached, which is every batch that never touched Chain's end card.
 *
 * Unlike the reference block there is only ONE key to drop: the band length
 * lives INSIDE `end_source` (`context_frames`), so nothing is left behind. */
function withoutEndSource(base: GenerateChainRequest): GenerateChainRequest {
  if (!("end_source" in base)) return base;
  const next = { ...base };
  delete next.end_source;
  return next;
}

/** Returns `clip` without its `conditioning_images` key (the same object when
 * already absent). */
function withoutConditioningImages(clip: GenerateChainRequest["clips"][number]): GenerateChainRequest["clips"][number] {
  if (!("conditioning_images" in clip)) return clip;
  const next = { ...clip };
  delete next.conditioning_images;
  return next;
}

/**
 * Turns one Chain-form request into the batch's row-neutral TEMPLATE:
 *
 * 1. `prompt` is left exactly as the Chain form emitted it (already
 *    `<lora:>`-stripped). Each row's own prompt is combined with it later, per
 *    row, by {@link buildRowPayload} — nothing prompt-related is decided here.
 * 2. `source_video` is dropped. Defence in depth: Start is already blocked
 *    while `chain.mode === "v2v"`, but a template that still carried a source
 *    would make every row a V2V continuation of the same video and silently
 *    ignore the row's image.
 * 3. `source_audio` is dropped (§1-16 長尺A2V). Same defence in depth: Start is
 *    already blocked while an audio track is attached to the Chain form, but a
 *    template that still carried one would drive EVERY row's chain with the
 *    same track — and, since audio and a source image are not mutually
 *    exclusive, would do it silently.
 * 4. `reference_video_id` and its two strengths are dropped (§1-15 参照動画).
 *    Same defence in depth once more: Start is already blocked while a
 *    reference video is attached to the Chain form, but a template that still
 *    carried one would condition EVERY row's chain on the same reference — and,
 *    since a reference and a start image are not mutually exclusive, would do it
 *    silently.
 * 5. `end_source` is dropped (素材（末尾）v2). The same defence in depth a fourth
 *    time: Start is already blocked while end material is attached, but a
 *    template that still carried it would make EVERY row end with the same
 *    material — and, since an end source and a start image are not mutually
 *    exclusive, would do it silently. v2 makes this stricter than a cosmetic
 *    concern: the band lengthens every row's output, so a stray `end_source`
 *    would also make every delivered file longer than the panel's own summary
 *    says.
 * 6. `clips[0].conditioning_images` is emptied — whatever start frame the
 *    Chain screen had is replaced per row by {@link buildRowPayload}.
 * 7. `clips[1..]` pass through completely untouched (per-clip prompt,
 *    `num_frames`, everything) — by reference, so they must not be mutated.
 * 8. Every other field (width/height/crop/fps/seed/overlap/loras/
 *    chunked_upsample/NAG/VSF/…) passes through verbatim.
 *
 * `base` is never modified: a fresh object and a fresh `clips` array come
 * back, and clip 0 is only cloned when it actually had an image.
 */
export function buildI2vLongTemplate({ base }: BuildI2vLongTemplateParams): GenerateChainRequest {
  const withoutSource = withoutEndSource(withoutReferenceVideo(withoutSourceAudio(withoutSourceVideo(base))));
  const clips = base.clips.map((clip, index) => (index === 0 ? withoutConditioningImages(clip) : clip));

  // `clips` already exists on `withoutSource`, so re-assigning it here
  // overwrites in place and leaves the key ORDER of the whole request exactly
  // as `buildChainRequest` emitted it.
  return { ...withoutSource, clips };
}

export interface BuildRowPayloadParams {
  /** This row's image, already uploaded and resolved to an `image_id` by the
   * runner (this module does no I/O). */
  imageId: string;
  /** The row's own prompt column (`I2vLongRow.prompt`); `""` for a row the
   * user never typed into. */
  rowPrompt: string;
  /** The panel's single add/replace radio — one mode for the whole batch, one
   * text per row (Batch A2V's arrangement exactly). */
  promptMode: PromptMode;
}

/**
 * The per-row step, in one pass:
 *
 * - the chain-wide `prompt` becomes `composeBatchPrompt(template.prompt,
 *   rowPrompt, promptMode)` — so every image can carry its own text;
 * - clip 0 — and only clip 0 (Docs/API_REFERENCE.md §5.2: 「clip 0 のみ画像を
 *   持てる」; the server 422s a `conditioning_images` on any later clip) —
 *   carries this row's image as its single frame-0 keyframe at full strength.
 *
 * 案A（驚き最小）is unchanged by the 2026-07-30 row-prompt redesign: the
 * composition touches the chain-wide prompt ONLY, never a clip's own `prompt`
 * override. A user who typed a per-clip prompt on the Chain screen sees it
 * respected verbatim — whereas applying `replace` across every clip would wipe
 * out visible on-screen text. **Future switch point**: moving to 案B (compose
 * into every clip) is three lines inside the `clips.map` below — `prompt:
 * clip.prompt === undefined ? undefined : composeBatchPrompt(clip.prompt,
 * rowPrompt, promptMode)` — and nothing else in this file or its callers
 * changes.
 *
 * Copy-on-write: `template` and every clip in it are left untouched, so the
 * SAME template is reused for every row of a run and repeated calls are fully
 * independent of each other. Clips 1..n, `loras`, `crop_output` etc. stay
 * shared by reference — none of them is ever mutated here. `prompt`/`clips`
 * already exist on `template`, so the request's key ORDER is preserved.
 */
export function buildRowPayload(
  template: GenerateChainRequest,
  { imageId, rowPrompt, promptMode }: BuildRowPayloadParams,
): GenerateChainRequest {
  const clips = template.clips.map((clip, index) =>
    index === 0 ? { ...clip, conditioning_images: [{ image_id: imageId, frame_idx: 0, strength: 1.0 }] } : clip,
  );
  return { ...template, prompt: composeBatchPrompt(template.prompt, rowPrompt, promptMode), clips };
}
