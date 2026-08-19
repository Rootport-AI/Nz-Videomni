/**
 * Batch A2V — assembling the `POST /generate/chain` body for a single
 * manifest row. Pure TypeScript mirror of `gradio_ui/handlers.py`'s
 * `build_a2v_chain_payload()` (the request-shape source of truth,
 * `Nz-LTX23-backend`, lines ~339-396), plus the two small pieces of per-row
 * assembly that live in `gradio_ui/batch.py` alongside it
 * (`compose_prompt` and `BatchRunner._build_conditioning`) — bundled into
 * this one file because this sprint allocates a single file for "turn a
 * BatchRow into an API payload" and every piece here is small and pure.
 *
 * Deliberately does NOT reuse `modes/chained/chainUtils.ts`'s
 * `buildChainRequest`: A2V's payload freezes several fields
 * `GenerateChainRequest`/`buildChainRequest` never set at all
 * (`num_inference_steps`, `guidance_scale`, `pipeline`, an always-present
 * `crop_output` key even when `null`) and is architecturally a single-clip,
 * single-purpose shape — routing it through the general chain builder and
 * patching the result afterward would be more code than this self-contained
 * mirror. It also sidesteps this sprint's parallel-edit rule (`chainUtils.ts`
 * is being edited concurrently by another agent).
 *
 * §1-19 (2026-08-11): both callers (Single's A2V path, `useGenerationForm.ts`,
 * and this builder's own Batch a2v caller, `batchRunner.ts`) now always send
 * `stage2_window: "full_length"` — see `A2vChainPayload.stage2_window`'s doc
 * comment below for why it is a fixed literal here rather than a
 * `BuildA2vChainPayloadParams` field. The Gradio backend's own mirror of this
 * function, `build_a2v_chain_payload()` (`gradio_ui/handlers.py`,
 * `Nz-LTX23-backend`), carries the same fixed key.
 */
import type { ChainClip, ConditioningImage, CropOutput, LoraSpec, SourceAudioSpec } from "../../api/types";
import { nagRequestFields } from "../../shell/nagSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { accelerationRequestFields } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { IMAGE_SHARED } from "./manifestMerge";

export type PromptMode = "add" | "replace";

/**
 * Combines the batch's common (Generate-tab) prompt with a row's individual
 * prompt. Mirrors `gradio_ui.batch.compose_prompt` exactly:
 * - an empty/whitespace-only row prompt -> the common prompt verbatim,
 *   regardless of `mode`;
 * - `mode === "replace"` -> the row prompt verbatim;
 * - `mode === "add"` -> `` `${common} ${row}` `` with both ends trimmed (an
 *   empty common prompt yields just the row prompt).
 *
 * The row prompt is passed through completely unexamined: any `<lora:...>`
 * tag written into it is left as plain text, never parsed (spec §8 — a
 * batch row's LoRAs can only come from the common/shared prompt, which the
 * caller runs through the normal prompt-tag parser BEFORE it ever reaches
 * this function).
 */
export function composeRowPrompt(commonPrompt: string, rowPrompt: string, mode: PromptMode): string {
  const common = commonPrompt ?? "";
  const row = rowPrompt ?? "";
  if (row.trim().length === 0) return common;
  if (mode === "replace") return row;
  return `${common} ${row}`.trim();
}

export interface ConditioningResolutionInput {
  /** The manifest row's `image` column value (`IMAGE_SHARED` or a bare
   * filename; `Docs/BATCH_A2V_CSV_SPEC.md` §3.1). */
  image: string;
  /** The Generate tab's common i2v keyframe(s), already resolved to
   * `image_id`s — used verbatim when `image === IMAGE_SHARED`. */
  sharedImages: ConditioningImage[];
  /** The row's own image, already uploaded/resolved to an `image_id` by the
   * caller (this module does no I/O) — used when `image` names a specific
   * file. `null`/`undefined` when not yet resolved. */
  rowImageId?: string | null;
}

/**
 * Resolves a row's `conditioning_images` list. Mirrors
 * `gradio_ui.batch.BatchRunner._build_conditioning`:
 * - `image === IMAGE_SHARED` -> the shared keyframes, as-is;
 * - a non-empty, non-Shared `image` -> a single frame-0/strength-1.0
 *   keyframe using the row's own (caller-resolved) `image_id` — `[]` if
 *   that id isn't available yet;
 * - an empty `image` -> `[]` (matches the Python function's own literal
 *   `elif row.image:` guard; in practice `image` is never empty by the time
 *   it reaches this function, since `parseManifest` already normalizes an
 *   empty CSV cell to `IMAGE_SHARED` — this branch exists purely so the
 *   mirror is faithful to the reference implementation's behavior at this
 *   exact layer).
 */
export function resolveConditioningImages(input: ConditioningResolutionInput): ConditioningImage[] {
  if (input.image === IMAGE_SHARED) {
    return input.sharedImages;
  }
  if (input.image) {
    return input.rowImageId ? [{ image_id: input.rowImageId, frame_idx: 0, strength: 1.0 }] : [];
  }
  return [];
}

/** The A2V `POST /generate/chain` body's parameters. Not
 * `GenerateChainRequest` (`api/types.ts`) — that type deliberately omits
 * `num_inference_steps`/`guidance_scale`/`pipeline` (see its file doc
 * comment: "UIに出してはいけない機能") because the interactive Chain screen
 * lets the server default them, whereas Batch A2V's distilled fast path
 * FREEZES them, exactly like the Generate tab's own A2V branch
 * (`gradio_ui.handlers.make_generate_handler`). */
export interface BuildA2vChainPayloadParams {
  audioId: string;
  numFrames: number;
  /** Already composed (see `composeRowPrompt`) — this function does no
   * prompt composition of its own, mirroring `build_a2v_chain_payload`'s
   * `prompt` kwarg (the caller, `batch.py`'s `_process_row`, also
   * pre-composes before calling in). */
  prompt: string;
  /** Pre-NAG negative-prompt param, kept for the single/batch A2V callers
   * that today pass `""`/omit it entirely — see the NAG doc comment below
   * for how it interacts with `nag`. */
  negativePrompt?: string | null;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings
   * (`shell/nagSettings.ts`). When `nag.enabled`, `nagRequestFields(nag)`
   * OVERRIDES `negativePrompt` above (same-key reassignment, placed
   * immediately after `negative_prompt` in the payload literal so key order —
   * and therefore the OFF-time `JSON.stringify` byte-identity — is
   * unaffected). `undefined`/disabled leaves `negativePrompt`'s own value in
   * place and adds no `nag_*` keys at all. */
  nag?: NagSettings;
  /** Acceleration (2026-07-31, backend §43): the Settings panel's shared
   * attention-backend choice (`shell/accelerationSettings.ts`). Adds the
   * single `attention_backend` key only while the user has moved off the
   * server default; `undefined`, or the default backend, adds nothing at all
   * — so an unaffected payload stays byte-identical (`JSON.stringify` key
   * order included) to before this feature existed. */
  acceleration?: AccelerationSettings;
  width: number;
  height: number;
  cropOutput?: CropOutput | null;
  frameRate: number;
  seed: number;
  /** Already resolved (see `resolveConditioningImages`) — omitted or empty
   * omits the clip's `conditioning_images` key entirely. */
  conditioningImages?: ConditioningImage[];
  loras?: LoraSpec[];
  useAdapter?: boolean;
  referenceVideoId?: string | null;
  controlAdherence?: number;
  referenceStrength?: number;
  /** Opt into temporal-chunk upsampling, mirroring
   * `GenerateChainRequest.chunked_upsample` (`modes/chained/chainUtils.ts`'s
   * `buildChainRequest`). **Not present** in the current
   * `gradio_ui.handlers.build_a2v_chain_payload` — Batch A2V's Python
   * runner never sends this field at all today (a real gap in the reference
   * implementation, called out in this sprint's report rather than silently
   * carried over) — but required here and always sent explicitly, matching
   * every other `chunked_upsample` call site in this codebase: omitting it
   * silently reverts to the slow one-pass upsample path. */
  chunkedUpsample: boolean;
}

export interface A2vChainPayload {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  crop_output: CropOutput | null;
  frame_rate: number;
  num_inference_steps: 8;
  guidance_scale: 1.0;
  seed: number;
  pipeline: "distilled";
  overlap_frames: 3;
  overlap_strength: 0.5;
  clips: [ChainClip];
  source_audio: SourceAudioSpec;
  chunked_upsample: boolean;
  /** §1-19 (2026-08-11): frozen at `"full_length"`, same style as
   * `pipeline: "distilled"` above — not a `BuildA2vChainPayloadParams` field,
   * because every caller of this builder assembles exactly 1 clip and giving
   * that a knob would just be a UI-selectable stage-2 window smuggled back in
   * through the params object, defeating the "no new UI" constraint this
   * feature shipped under. Mirrors `chain_math.STAGE2_WINDOW_PRESETS["full_length"]
   * = (61, 61)` (`Nz-LTX23-backend`) — one stage-2 tile spanning the whole
   * clip (61 video-latent frames = 481 pixel frames), so there is never more
   * than one tile and therefore never a seam. */
  stage2_window: "full_length";
  loras?: LoraSpec[];
  reference_video_id?: string | null;
  conditioning_attention_strength?: number;
  reference_video_strength?: number;
  // NAG (2026-07-28)/VSF (2026-07-29): only the 6 fields NOT already covered
  // by the required `negative_prompt: string` above — see
  // `BuildA2vChainPayloadParams.nag`'s doc comment for how the two interact.
  nag_enabled?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  neg_method?: "nag" | "vsf";
  vsf_scale?: number;
  // Acceleration (2026-07-31 / 2026-08-01 / 2026-08-02 / 2026-08-04 /
  // 2026-08-05): the five real fields — see
  // `BuildA2vChainPayloadParams.acceleration`. Every field
  // `accelerationRequestFields` can emit must appear here, or the spread below
  // would not type-check.
  attention_backend?: "sdpa" | "sage";
  block_swap_prefetch?: boolean;
  keep_resident?: boolean;
  fused_gguf_dequant_kernel?: boolean;
  vae_mode?: "default" | "prune_vaed";
}

/**
 * Assembles the A2V `POST /generate/chain` body (案A: a single `ChainClip`
 * carrying `num_frames` + any keyframe `conditioning_images`). Byte-for-byte
 * mirror of `gradio_ui.handlers.build_a2v_chain_payload` (frozen distilled
 * quality contract, `overlap_frames=3`/`overlap_strength=0.5`,
 * `source_audio.audio_id`), plus `chunked_upsample` (see its JSDoc above).
 * `loras`/`reference_video_id`/the two adapter-strength fields are each
 * included only when applicable, exactly like the Python function, so a
 * token-free, adapter-free request stays minimal — matching
 * `handlers.py`'s comment that a no-lora/no-adapter request must stay
 * byte-identical to before those features existed.
 */
export function buildA2vChainPayload(params: BuildA2vChainPayloadParams): A2vChainPayload {
  const clipEntry: ChainClip = { num_frames: Math.trunc(params.numFrames) };
  if (params.conditioningImages && params.conditioningImages.length > 0) {
    clipEntry.conditioning_images = params.conditioningImages;
  }

  const payload: A2vChainPayload = {
    prompt: params.prompt,
    negative_prompt: params.negativePrompt || "",
    // NAG (2026-07-28)/VSF (2026-07-29): a same-key reassignment of
    // `negative_prompt` (plus the 6 `nag_*`/`neg_method`/`vsf_scale` keys)
    // the instant it appears right after the key's own
    // first assignment above — JS object-literal key order is unaffected by
    // reassigning an EXISTING key, only by inserting a genuinely new one, so
    // this keeps the OFF-time (`nag` unset/disabled, `{}` returned) payload
    // byte-identical (`JSON.stringify` order included) to before NAG existed.
    ...nagRequestFields(params.nag),
    width: Math.trunc(params.width),
    height: Math.trunc(params.height),
    crop_output: params.cropOutput ?? null,
    frame_rate: params.frameRate,
    num_inference_steps: 8,
    guidance_scale: 1.0,
    seed: Math.trunc(params.seed),
    pipeline: "distilled",
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [clipEntry],
    source_audio: { audio_id: params.audioId },
    chunked_upsample: params.chunkedUpsample,
    // §1-19 (2026-08-11): fixed literal, always sent — see
    // `A2vChainPayload.stage2_window`'s doc comment above for why this is
    // not parameterized.
    stage2_window: "full_length",
    // Acceleration (2026-07-31): LAST, and conditional — an unset/default
    // choice spreads `{}` and leaves this payload exactly as it was before
    // the feature existed, key order included.
    ...accelerationRequestFields(params.acceleration),
  };

  if (params.loras && params.loras.length > 0) {
    payload.loras = params.loras;
  }
  if (params.useAdapter) {
    payload.reference_video_id = params.referenceVideoId ?? null;
    const controlAdherence = params.controlAdherence ?? 1.0;
    const referenceStrength = params.referenceStrength ?? 1.0;
    if (controlAdherence < 1.0) {
      payload.conditioning_attention_strength = controlAdherence;
    }
    if (referenceStrength < 1.0) {
      payload.reference_video_strength = referenceStrength;
    }
  }

  return payload;
}
