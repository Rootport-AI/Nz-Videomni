import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge as defaultBridge } from "../../bridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import type { AppConfig, ConditioningImage, CropOutput, GenerateRequest } from "../../api/types";
import { combineLoras } from "../../lora/controlLoras";
import type { ControlLoraSelection } from "../../lora/controlLoras";
import { clampLoraStrength, parseLoraPrompt } from "../../lora/loraTags";
import { buildA2vChainPayload } from "../batch/buildA2vChainPayload";
import type { A2vChainPayload } from "../batch/buildA2vChainPayload";
import {
  audioLengthPrecheck,
  clampCropOutput,
  isCropOutputValid,
  MAX_REFERENCE_STRENGTH,
  MIN_REFERENCE_STRENGTH,
  suggestFramesForAudio,
} from "../chained/chainUtils";
import type { AudioLengthPrecheckResult } from "../chained/chainUtils";
import { useSourceUpload } from "../chained/useSourceUpload";
import type { UseSourceUploadResult } from "../chained/useSourceUpload";
import { isNagNegativeEmpty, NAG_OFF, nagRequestFields } from "../../shell/nagSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { accelerationRequestFields, ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { comfortFramesForBudget, resolveComfortRow } from "../../shell/comfortTable";
import { MIN_HEIGHT, MIN_NUM_FRAMES, MIN_WIDTH } from "./defaultConfig";
import { estimateGenerationSeconds, formatEstimate } from "./estimateUtils";
import {
  ceilToMultiple,
  clamp,
  floorToMultiple,
  formatDurationHint,
  FRAME_RATE_FALLBACK,
  FRAME_RATE_MIN,
  isDimensionOnGrid,
  isNumFramesOnGrid,
  isValidPrompt,
  roundToMultiple,
  snapFrameRate,
  snapNumFrames,
} from "./paramUtils";
import { resolveSpillFreeFrames } from "./spillUtils";

/** The width/height rounding multiple for the IC-LoRA (`referenceVideo`) flow:
 * `/generate`'s reference-video conditioning requires both dimensions to be
 * multiples of 128, vs. the general-mode 64 (Docs/TIMELINE_ALPHA_REQUIREMENTS.md
 * §2). Kept identical to `deriveGenerationParams`'s `multiple` for IC-LoRA. */
const IC_LORA_MULTIPLE = 128;
const GENERAL_MULTIPLE = 64;

/** W7: the hook-owned reasons the Create Generate button is currently
 * un-submittable, one code per failing gate — the structured form of `isValid`
 * (which is exactly `validityReasons.length === 0`). `SingleScreen` composes
 * these with its own screen-level codes (keyframe/audio upload in flight,
 * reference not ready, overflow keyframes) before handing the union to
 * `GenerateReasonsNote`. Each code has an imperative "do this to enable
 * Generate" line in `strings.single.generateReasons`. */
export type GenerationValidityReason =
  | "promptEmpty"
  | "dimensionsOffGrid"
  | "numFramesOffGrid"
  | "audioTooShort"
  | "referenceNeedsLoras"
  | "controlNeedsReference"
  | "cropInvalid"
  | "nagNegativeEmpty"
  /** §1-6 拡張 (2026-08-01): the IC-LoRA reference video's ribbon-range cut was
   * requested but the upload response did not confirm it, so the stored material
   * is the WHOLE file. Same gate (and same wording) as Chain's
   * `sourceTrimFailed` — see `useSourceUpload`'s `trimFailed`. */
  | "referenceTrimFailed";

export interface FormValues {
  width: number;
  height: number;
  numFrames: number;
  frameRate: number;
  seed: number;
}

export interface FormLimits {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  minNumFrames: number;
  maxNumFrames: number;
}

export interface UseGenerationFormDeps {
  nativeBridge?: NativeBridge | undefined;
  /** IC-LoRA UI redesign (第5波): the panel's current selection, owned by
   * `shell/AppShell.tsx` (not this hook) so it survives a tab switch away
   * from Create and back — see `AppShell`'s own doc comment for why. `null`
   * (and the paired no-op `setControlLora`) when omitted, which is what
   * every pre-existing unit test that doesn't thread AppShell's state gets:
   * the form then behaves as "no control LoRA selected, ever" — a control
   * LoRA can only enter the picture through these two options. */
  controlLora?: ControlLoraSelection | null | undefined;
  setControlLora?: ((value: ControlLoraSelection | null) => void) | undefined;
  /** Selectable control-LoRA names (`lora/controlLoras.ts`'s
   * `resolveControlLoraNames`, computed by the caller from `GET /loras` +
   * `config.model.ic_loras`) — passed straight through to this hook's result
   * so `GenerationForm` can build the dropdown off `form.controlLoraNames`
   * instead of needing its own separate wiring. Falls back to
   * `config.model.ic_loras`'s own keys when omitted (pre-existing N2
   * fallback), so a test that only cares about the gate/merge behavior
   * doesn't also have to fabricate a `GET /loras` response. */
  controlLoraNames?: ReadonlySet<string> | undefined;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings,
   * owned by `shell/AppShell.tsx` (single `NagSettings` object props-passed
   * down, D1) — the same "caller owns the state, this hook just reads it"
   * shape as `controlLora` above. Defaults to the frozen `NAG_OFF` sentinel
   * when omitted (every pre-existing unit test), which keeps
   * `nagRequestFields` returning `{}` — every existing request stays
   * byte-identical to before NAG existed. */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31, backend §43): the Settings panel's shared
   * attention-backend choice, owned by `shell/AppShell.tsx` (single
   * `AccelerationSettings` object props-passed down, D1) — same "caller owns
   * the state" shape as `nag` above. Defaults to the frozen
   * `ACCELERATION_DEFAULTS` sentinel when omitted (every pre-existing unit
   * test), which keeps `accelerationRequestFields` returning `{}` — every
   * existing request stays byte-identical to before this feature existed. */
  acceleration?: AccelerationSettings | undefined;
  /** Smart comfort marker (2026-08-18): whether the server reports
   * SageAttention as installed (`shell/accelerationSettings.sageAvailability`),
   * owned by `shell/AppShell.tsx` off the same `/status` poll it already reads
   * `blockSwapPrefetchAvailability` from — same "caller owns the state" shape
   * as `acceleration` above. Defaults to `null` ("unknown") when omitted
   * (every pre-existing unit test), which
   * `accelerationSettings.effectiveAccelerationFields` treats as effectively on
   * (see that function's own doc comment) — so an omitted dep never itself
   * blocks the smart marker; only an explicit `false` does. */
  sageAvailable?: boolean | null | undefined;
  /** Smart comfort marker (2026-08-31): the LOADED base model's engine family
   * (`BaseModelBlock.engine_family` — `"ltx"`, `"ltx25"`, …), owned by
   * `shell/AppShell.tsx` via `useBaseModels().activeEngineFamily` — same
   * "caller owns the state" shape as `sageAvailable` above. It is the key into
   * the served `config.limits.comfort_budgets` table, so the comfort marker
   * follows the engine instead of a hard-coded rule.
   *
   * Omitted (every pre-existing unit test), `undefined` or `""` (before `GET
   * /models` lands, or offline) all mean "engine unknown", which
   * `shell/comfortTable.ts`'s `resolveComfortRow` answers with its
   * compatibility shim — i.e. exactly the pre-2026-08-31 behaviour, with no
   * startup flicker. */
  engineFamily?: string | undefined;
}

function noopSetControlLora(): void {
  // Default when the caller doesn't thread AppShell's setter through (every
  // pre-existing unit test) — the form has no way to ever set `controlLora`
  // away from `null` in that case, so silently doing nothing is correct, not
  // just a placeholder.
}

/** One-shot initial overrides seeded from a right-click `GenerationPrefill`
 * (`SingleScreen` derives width/height via `deriveGenerationParams`). Applied
 * only in the form's lazy `useState` initializer, so a later re-render never
 * re-applies them. Omitted entirely for the normal (tab-opened) form, which
 * keeps seeding from `config.generation_defaults`. */
export interface GenerationFormInitial {
  width?: number;
  height?: number;
  /** W7: one-shot initial frame rate seeded from a right-click prefill (the
   * `material` policy's selection rate/scale, else omitted to keep the config
   * default). Applied only in the lazy `useState` initializer, like
   * `width`/`height`. Omitted for the normal (tab-opened) form and the
   * `defaults` policy, both of which keep `config.generation_defaults.frame_rate`. */
  frameRate?: number;
  /** W8: one-shot initial DURATION (`num_frames`) seeded from a right-click
   * prefill's DURATION policy (`deriveDuration.computeTargetNumFrames`) — the
   * resolution's comfort ceiling for #4 image-to-video, or the reference
   * material's own length clamped to that ceiling for #2 reference-video.
   * Applied only in the lazy `useState` initializer (re-snapped onto the 8n+1
   * grid), like `width`/`height`. Omitted for the normal (tab-opened) form and
   * any prefill whose policy leaves DURATION untouched (#3/#7 A2V, whose length
   * comes from the wav auto-adjust below instead), both of which keep
   * `config.generation_defaults.num_frames`. */
  numFrames?: number;
  /** W4 (#7 trim追従): the span-derived DURATION cap for an audio-to-video
   * right-click prefill. #7 uploads the WHOLE audio file, so the wav-duration
   * auto-adjust below would otherwise re-seed `numFrames` off the full file
   * length, overriding the trimmed (ribbon) span the prefill seeded. When set,
   * the wav suggestion is capped at this value — BUT only while the currently
   * attached audio is still the prefill-attached one; swapping the audio lifts
   * the cap and restores the ordinary full-wav follow. Omitted for every
   * non-#7 path (normal form, #3 whose extracted wav is already span-length). */
  a2vSeedMaxFrames?: number;
  /** true for the IC-LoRA (`referenceVideo`) flow: pins the rounding multiple
   * to 128 and switches the max bounds onto that grid from mount. */
  isICLora?: boolean;
  /** W5 (反対スロット保持): a reference video carried across a #3/#7 (A2V)
   * remount — the opposite slot the new A2V material does not replace. Seeds the
   * `referenceVideo` slot's ready state (no re-upload) AND, crucially, forces the
   * form's width/height grid onto the 128 multiple from mount (see the derived
   * `seedICLora`): without that, the 64-grid seed would fail the IC-LoRA 128
   * gate and permanently block Generate (回帰対策1). */
  carryReferenceVideo?: { id: string; fileName: string; filePath: string };
  /** W5: the carried reference video's `conditioning_attention_strength`
   * (`null` when the user hadn't opted it in on the previous form). Seeds the
   * matching state so the strength survives the remount alongside the video. */
  carryConditioningAttentionStrength?: number | null;
  /** W5: the carried reference video's `reference_video_strength` (see
   * `carryConditioningAttentionStrength`). */
  carryReferenceVideoStrength?: number | null;
  /** W5 (反対スロット保持): a source audio carried across a #2 (IC-LoRA) remount —
   * the opposite slot the new reference material does not replace. Seeds the
   * `sourceAudio` slot's ready state (no re-upload); `lastAudioFilePathRef` is
   * primed from its `filePath` so the mount-time wav probe still fires and the
   * DURATION auto-adjusts to the audio (回帰対策2 — keeps the A2V DURATION
   * priority). NOT paired with `a2vSeedMaxFrames`: a #2-carried audio is not a
   * prefill material, so it follows the full wav length (no span cap). */
  carryAudio?: { id: string; fileName: string; filePath: string };
}

export interface UseGenerationFormResult {
  values: FormValues;
  limits: FormLimits;
  presets: AppConfig["generation_presets"];
  durationHint: string;
  promptError: string | null;
  isValid: boolean;
  /** W7: the structured breakdown behind `isValid` — one code per failing gate,
   * empty exactly when `isValid` is true (`isValid === (validityReasons.length
   * === 0)`). Consumed by `SingleScreen`'s Generate-reasons note; see
   * {@link GenerationValidityReason}. */
  validityReasons: GenerationValidityReason[];
  gettingSize: boolean;
  getSizeError: string | null;
  /** The comfortable `num_frames` ceiling for the current width/height.
   *
   * Smart comfort marker (2026-08-31, was 2026-08-18): when the SERVED table
   * `config.limits.comfort_budgets` has a row for the loaded engine family +
   * the current acceleration configuration (`shell/comfortTable.ts`'s
   * `resolveComfortRow`), this is the resolution-exact ceiling
   * `comfortFramesForBudget` derives from that row's token budget — it moves
   * smoothly with every 64px step instead of jumping between five measured
   * points. When no row applies (LTX 2.3's DEFAULT configuration
   * deliberately has none), or the smart derivation itself returns `null` (a
   * hand-typed 0/blank width), this falls back to the coarse 5-key
   * nearest-area lookup (Docs/API_REFERENCE.md §3.2 `spill_free_frames`, see
   * `spillUtils.resolveSpillFreeFrames`). `null` only when even THAT lookup
   * can't resolve (empty/malformed config map). See {@link isComfortMarkerSmart}
   * for which branch produced the current value — it drives which warning
   * copy is shown past the threshold.
   *
   * This same value is the ceiling the A2V wav auto-adjust clamps its
   * suggestion at, so the marker and the auto-adjusted DURATION can never
   * disagree. */
  spillThresholdFrames: number | null;
  /** True once `values.numFrames` exceeds `spillThresholdFrames`. Formula is
   * unchanged by the smart marker (2026-08-18) — only where
   * `spillThresholdFrames` itself comes from changed. */
  isOverSpillThreshold: boolean;
  /** Smart comfort marker (2026-08-18): true while `spillThresholdFrames` is
   * the SMART per-resolution ceiling rather than the coarse `spill_free_frames`
   * fallback (see that field's own doc comment for the exact gate). Drives
   * which warning copy `GenerationForm` shows once `isOverSpillThreshold` is
   * true — `strings.single.comfortWarningSmart` here, `strings.single.spillWarning`
   * otherwise — since the two warnings claim different (and not both always
   * true) things about how much generation slows down. */
  isComfortMarkerSmart: boolean;
  /** Rough estimate (see `estimateUtils.ts`) in seconds, for the current
   * width/height/numFrames. */
  estimateSeconds: number;
  /** Ready-to-render "Est. ~X min" label for the Generate button area. */
  estimateLabel: string;
  /** True while the form is in IC-LoRA (`referenceVideo`) mode — either seeded
   * from the routed intent or once a reference video has been uploaded. Drives
   * the 128-multiple snapping and the reference-video UI block. */
  isICLora: boolean;
  /** The IC-LoRA reference video upload slot (reuses Chain's `useSourceUpload`,
   * `kind:"video"`). A separate effect re-snaps width/height onto the 128 grid
   * once the upload becomes ready (its id turns non-null); its `state.id`
   * becomes `reference_video_id`. The section is always rendered now (the
   * exclusivity is expressed by disabling the picker), not just in IC-LoRA
   * mode. */
  referenceVideo: UseSourceUploadResult;
  /** Contract v9: the ready reference video's measured duration in seconds
   * (`fs.probeMediaInfo`), for the IC-LoRA source card's `12.3s` readout.
   * `null` until a ready reference video's duration has been probed and came
   * back `> 0`, and `null` again whenever the reference video is not ready
   * (idle/uploading/error) or the probe returned `0`/failed. */
  referenceVideoDurationSec: number | null;
  /** N1: opt-in output crop (`GenerateRequest.crop_output`). `null` = "not
   * set" (omitted from the request). See `CommonGenerationFields.CropOutputField`
   * and `chainUtils.clampCropOutput`/`isCropOutputValid` for the 32-pixel-grid
   * constraint (LTX's VAE space-compression factor) folded into `isValid`
   * below. Mirrors Chain's `useChainForm.cropOutput`. */
  cropOutput: CropOutput | null;
  /** Sets `cropOutput`, snapping onto the 32-pixel grid and clamping to
   * `[32, values.width/values.height]` — never the config max — via
   * `chainUtils.clampCropOutput`. `null` clears it (omitted from the
   * request). */
  setCropOutput: (value: CropOutput | null) => void;
  /** 0.0-1.0 or `null` ("not set" — the field is omitted from the request).
   * Only ever sent alongside a ready reference video AND a non-empty
   * `loras[]` (N3's loras gate — see {@link isReferenceValid}). Mirrors
   * Chain's `useChainForm.conditioningAttentionStrength`. */
  conditioningAttentionStrength: number | null;
  setConditioningAttentionStrength: (value: number | null) => void;
  /** 0.0-1.0 or `null` ("not set"). See {@link conditioningAttentionStrength}. */
  referenceVideoStrength: number | null;
  setReferenceVideoStrength: (value: number | null) => void;
  /** IC-LoRA UI redesign (第5波): the panel's current control-LoRA selection
   * (`null` = "none"), threaded through from `UseGenerationFormDeps` — see
   * that option's doc comment for why AppShell, not this hook, owns the
   * actual state. */
  controlLora: ControlLoraSelection | null;
  setControlLora: (value: ControlLoraSelection | null) => void;
  /** Sets `controlLora`'s strength only, clamped to
   * `[LORA_STRENGTH_MIN, LORA_STRENGTH_MAX]` (`lora/loraTags.ts`) — the
   * panel's weight slider. A no-op while `controlLora` is `null` (the slider
   * isn't rendered in that state, but this stays a total function rather
   * than throwing). */
  setControlLoraStrength: (value: number) => void;
  /** Selectable control-LoRA names for the panel's dropdown — pass-through of
   * `UseGenerationFormDeps.controlLoraNames` (see its doc comment for the
   * `GET /loras` / `config.model.ic_loras` fallback). */
  controlLoraNames: ReadonlySet<string>;
  /** N3: true once the reference video is ready (`referenceVideo.state.id`
   * set) but the merged `loras[]` (control selection + prompt's STYLE
   * `<lora:...>` tags — see {@link combineLoras}) is still empty. The server
   * 422s `reference_video_id`/the two strength fields whenever `loras` is
   * empty. Drives the "add a LoRA" warning and blocks `isValid`. */
  referenceVideoNeedsLoras: boolean;
  /** True once a control LoRA is selected in the panel (`controlLora !==
   * null`) but no reference video is attached yet — the inverse of
   * {@link referenceVideoNeedsLoras}. Drives the "this control LoRA requires
   * a reference video" warning and blocks `isValid`. */
  controlLoraNeedsReferenceVideo: boolean;
  /** `!referenceVideoNeedsLoras && !controlLoraNeedsReferenceVideo` — folded
   * into `isValid` below; also surfaced standalone so `GenerationForm` can
   * render the specific warning rather than a generic "can't submit". */
  isReferenceValid: boolean;
  /** Group3 item11: the optional A2V source-audio upload slot (reuses Chain's
   * `useSourceUpload`, `kind:"audio"`), always rendered Gradio-style above the
   * size fields. Attaching a wav flips the whole form into A2V mode
   * (`isA2v`) — the request is then assembled by {@link buildA2vRequest} and
   * submitted to `POST /generate/chain` instead of `/generate`. Combinable
   * with the IC-LoRA reference video (`isICLora`): with both attached, the
   * A2V payload also carries the reference-video fields (reference-controlled
   * A2V). */
  sourceAudio: UseSourceUploadResult;
  /** Contract v7 (drag-and-drop): attaches a dropped audio file, mirroring
   * `sourceAudio.uploadPath` but ALSO priming `lastAudioFilePathRef` first, so
   * the wav-duration auto-adjust probe below (which reads that ref, not
   * `sourceAudio.state`) still fires for a drop the same as it does for a
   * `ui.pickFile`-driven pick. Use this instead of `sourceAudio.uploadPath`
   * directly for any non-`ui.pickFile` attach path. */
  attachSourceAudioByPath: (filePath: string, fileName: string) => Promise<void>;
  /** True once a source audio is attached (`status === "ready"`) — i.e. audio
   * is the generation's driving source — the flag that routes submission
   * through {@link buildA2vRequest}. Independent of `isICLora`: both can hold
   * at once, in which case the A2V payload also carries the reference-video
   * fields (reference-controlled A2V). */
  isA2v: boolean;
  /** A2V-only (`null` otherwise, or before the attached wav's duration has
   * been measured): whether the audio's measured duration VAE-encodes to
   * enough audio-latent frames for `values.numFrames`
   * (`chainUtils.audioLengthPrecheck`). Folded into `isValid`; also surfaced
   * so the form can show the exact "audio too short" seconds. */
  audioPrecheck: AudioLengthPrecheckResult | null;
  /** A2V source card's `12.3s` duration readout: the attached wav's measured
   * duration in seconds (`fs.probeAudioDuration`), or `null` before a `.wav`
   * has been successfully probed (and `null` again whenever the attachment
   * changes/clears). This is the same `audioDurationSec` the numFrames
   * auto-adjust/`audioPrecheck` already derive from, surfaced for display. */
  audioDurationSec: number | null;
  /** Fires once per successful A2V wav attach whose duration this hook could
   * measure and auto-adjust `values.numFrames` from — `id` increments per
   * event so `SingleScreen` can key a toast effect off it. `null` before any
   * wav has been probed. Mirrors Chain's `audioFramesAdjustedEvent`. */
  audioFramesAdjustedEvent: { id: number; frames: number; durationSec: number } | null;
  /** Sets width. `snap` (default true) rounds `value` onto the active grid
   * (64, or 128 in IC-LoRA mode) and clamps to the active bounds — used by the
   * stepper arrows, ↑↓ keys, slider, presets, and the getSize/reference-video
   * flows. `snap === false` (free keyboard typing) passes the value straight
   * through with only a NaN guard (Gradio-faithful: no rounding, no clamping);
   * the grid gate lives in `isValid` so an off-grid manual entry blocks submit. */
  setWidth: (value: number, snap?: boolean) => void;
  /** Sets height. See `setWidth` for the `snap` semantics. */
  setHeight: (value: number, snap?: boolean) => void;
  /** Sets `values.numFrames`. `snap` (default true) rounds `value` onto the
   * 8n+1 grid and clamps to `[limits.minNumFrames, limits.maxNumFrames]` —
   * used by the stepper arrows, ↑↓ keys, and slider. `snap === false` (free
   * keyboard typing) passes the value straight through with only a NaN
   * guard; the grid gate lives in `isValid` so an off-grid manual entry
   * blocks submit. Mirrors `setWidth`/`setHeight`'s `snap` semantics. */
  setNumFrames: (value: number, snap?: boolean) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
  applyPreset: (name: string) => void;
  getSizeFromAviUtl2: () => Promise<void>;
  toGenerateRequest: () => GenerateRequest;
  /** Assembles the A2V `POST /generate/chain` body (distilled fast path,
   * frozen `num_inference_steps=8`/`guidance_scale=1.0`/`pipeline="distilled"`/
   * `overlap_frames=3`/`overlap_strength=0.5`) from the current form state,
   * reusing Batch's `buildA2vChainPayload`. `<lora:...>` tags in the prompt
   * are stripped and moved into `loras` (the merged control-panel selection +
   * STYLE tags); the passed `conditioningImages` (Create's I2V keyframes) ride
   * on the single clip. When a reference video and at least one loRA are both
   * present it ALSO sets `useAdapter`/`reference_video_id` and the two adapter
   * strengths (reference-controlled A2V); a lora-less reference video carries
   * no adapter fields. Only meaningful while `isA2v`. */
  buildA2vRequest: (conditioningImages: ConditioningImage[]) => A2vChainPayload;
}

/** Owns Create-form state (everything *except* the prompt, which M3 lifted
 * to the shared `PromptBar` above the mode tabs so it survives switching to
 * Chain and back — Mock/AVIUTL2_DESIGN_BRIEF.md §11 — and is passed in here
 * as `prompt`), seeded from `/config`'s `generation_defaults` and bounded by
 * its `limits`. Every setter snaps to the frozen constraints
 * (Docs/API_REFERENCE.md §5.1: width/height multiples of 64, num_frames
 * 8n+1) so the form can never hold an invalid combination. */
export function useGenerationForm(
  config: AppConfig,
  prompt: string,
  deps: UseGenerationFormDeps = {},
  initial: GenerationFormInitial = {},
): UseGenerationFormResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;

  // IC-LoRA UI redesign (第5波): the panel's selection state is owned by the
  // caller (AppShell in production — see `UseGenerationFormDeps`'s doc
  // comment) so it survives a tab switch away from Create; every
  // pre-existing unit test that omits these options gets `null` and a no-op
  // setter, i.e. "no control LoRA, ever" (this hook never introduces its own
  // fallback `useState` for it — there is exactly one owner).
  const controlLora = deps.controlLora ?? null;
  const setControlLora = deps.setControlLora ?? noopSetControlLora;
  const controlLoraNames = deps.controlLoraNames ?? new Set(Object.keys(config.model?.ic_loras ?? {}));

  // NAG (2026-07-28): mirrors controlLora's optional-dep pattern (D3) — the
  // caller (`AppShell`) owns the actual `NagSettings` state; every
  // pre-existing test that omits `deps.nag` gets the frozen `NAG_OFF`
  // sentinel, which keeps `nagRequestFields`/`isNagNegativeEmpty` behaving
  // exactly as if NAG didn't exist.
  const nag = deps.nag ?? NAG_OFF;
  // Acceleration (2026-07-31): same optional-dep shape as `nag` — the caller
  // (`AppShell`) owns the state; omitting it yields the all-defaults sentinel,
  // and therefore no request fields at all.
  const acceleration = deps.acceleration ?? ACCELERATION_DEFAULTS;
  // Smart comfort marker (2026-08-18): same optional-dep shape again —
  // `null` ("unknown") is the correct default, not just a filler value, since
  // `effectiveAccelerationFields` treats `null` as effectively on (see its own
  // doc comment).
  const sageAvailable = deps.sageAvailable ?? null;
  // Smart comfort marker (2026-08-31): the engine family keying the served
  // comfort table. `undefined` (omitted, or `GET /models` not in yet) is the
  // "engine unknown" case `resolveComfortRow` answers with its compatibility
  // shim, so an omitted dep reproduces the pre-table behaviour exactly.
  const engineFamily = deps.engineFamily;

  const setControlLoraStrength = useCallback(
    (raw: number) => {
      if (!controlLora) return;
      setControlLora({ name: controlLora.name, strength: clampLoraStrength(raw) });
    },
    [controlLora, setControlLora],
  );

  // W5 回帰対策1: a carried reference video (#3/#7 A2V remount) must ALSO seed
  // the form in IC-LoRA mode, exactly as a routed `reference-video` intent does —
  // otherwise the 64-grid width/height seed fails the 128 gate the reference
  // video imposes and Generate is permanently blocked (`dimensionsOffGrid`).
  // Y1: this is now ONLY the mount-time `values` initializer's "seed on the 128
  // grid?" flag (evaluated once). The active-mode determination (`icLoraActive`
  // below) no longer reads it — it is fully dynamic off the live reference
  // video / control LoRA — so a routed #2 whose reference video is later
  // removed correctly falls back out of IC-LoRA mode instead of sticking. The
  // 128-aligned seed left behind stays valid on the relaxed 64 grid (128 is a
  // multiple of 64), so no re-snap is needed when the mode relaxes.
  const seedICLora = (initial.isICLora ?? false) || initial.carryReferenceVideo !== undefined;

  // W4 (#7 trim追従): the span-derived DURATION cap for an audio-to-video
  // prefill, frozen at mount like every other one-shot `initial.*`. `undefined`
  // for every non-#7 path (no cap). `a2vSeedCapPathRef` captures the FIRST
  // audio path attached while the cap is set — the prefill-attached audio — so
  // the wav auto-adjust below can tell "still the prefill audio" (apply the cap)
  // from "user swapped the audio" (lift the cap) by comparing paths.
  const [a2vSeedMaxFrames] = useState(() => initial.a2vSeedMaxFrames);
  const a2vSeedCapPathRef = useRef<string | null>(null);

  const baseLimits: FormLimits = useMemo(
    () => ({
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
      minNumFrames: MIN_NUM_FRAMES,
      maxNumFrames: config.limits.max_num_frames,
    }),
    [config.limits.max_width, config.limits.max_height, config.limits.max_num_frames],
  );

  const referenceVideo = useSourceUpload("video", {
    nativeBridge,
    // W5: seed the carried reference video's ready state (#3/#7 remount) — the
    // id is still valid server-side, so no re-upload.
    ...(initial.carryReferenceVideo ? { initial: initial.carryReferenceVideo } : {}),
  });
  const referenceVideoId = referenceVideo.state.id;

  // Reference-video (control IC-LoRA) strength overrides — `null` ("not set")
  // until the user opts in via the UI's enable checkbox, matching the
  // server's own "omitted = use its own default" semantics. Mirrors Chain's
  // `useChainForm.conditioningAttentionStrength`/`referenceVideoStrength`.
  // W5: the two strengths seed from the carried reference video (#3/#7 remount)
  // so an opted-in strength survives alongside the video; `undefined` (no carry)
  // and `null` (carried but not opted in) both start at `null` ("not set").
  const [conditioningAttentionStrength, setConditioningAttentionStrengthState] = useState<number | null>(
    initial.carryConditioningAttentionStrength ?? null,
  );
  const setConditioningAttentionStrength = useCallback(
    (raw: number | null) =>
      setConditioningAttentionStrengthState(raw == null ? null : clamp(raw, MIN_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH)),
    [],
  );
  const [referenceVideoStrength, setReferenceVideoStrengthState] = useState<number | null>(
    initial.carryReferenceVideoStrength ?? null,
  );
  const setReferenceVideoStrength = useCallback(
    (raw: number | null) =>
      setReferenceVideoStrengthState(raw == null ? null : clamp(raw, MIN_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH)),
    [],
  );

  // Group3 item11 (A2V move to Create): a wav-picker snoop wrapper around
  // `nativeBridge`, identical in intent to Chain's `audioProbingBridge`
  // (`modes/chained/useChainForm.ts`): `useSourceUpload`'s `SourceUploadState`
  // only ever exposes `fileName`, never the local `filePath` that
  // `ui.pickFile` resolved with — and `useSourceUpload` is out of this file's
  // scope. This thin pass-through forwards every call unchanged while
  // capturing the audio pick's `filePath` into `lastAudioFilePathRef`, so the
  // wav-duration probe below has a path to hand `fs.probeAudioDuration`.
  // W5 回帰対策2: prime the ref with the carried audio's path (#2 remount) so the
  // mount-time wav probe below — which reads THIS ref, not `sourceAudio.state` —
  // still fires for a carried-over audio and re-derives the DURATION from it
  // (keeps the A2V DURATION priority across the remount).
  const lastAudioFilePathRef = useRef<string | null>(initial.carryAudio?.filePath ?? null);
  const audioProbingBridge = useMemo<NativeBridge>(
    () => ({
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        const result = await nativeBridge.request(method, params);
        if (method === "ui.pickFile" && (params as ParamsOf<"ui.pickFile">).kind === "audio") {
          lastAudioFilePathRef.current = (result as ResultOf<"ui.pickFile">).filePath;
        }
        return result;
      },
      requestWithFiles(method, params, files) {
        return nativeBridge.requestWithFiles(method, params, files);
      },
      on(event, handler) {
        return nativeBridge.on(event, handler);
      },
    }),
    [nativeBridge],
  );
  const sourceAudio = useSourceUpload("audio", {
    nativeBridge: audioProbingBridge,
    // W5: seed the carried source audio's ready state (#2 remount).
    ...(initial.carryAudio ? { initial: initial.carryAudio } : {}),
  });
  const audioReady = sourceAudio.state.status === "ready";

  // Contract v7 (drag-and-drop): D&D never goes through `ui.pickFile`, so
  // `audioProbingBridge`'s own snoop (above) never fires for it -- priming
  // `lastAudioFilePathRef` here directly is what keeps the wav-duration
  // auto-adjust probe working for a dropped file too. Forgetting this call
  // order (ref set BEFORE uploadPath, not after) is the most likely
  // regression: the probe effect reads the ref synchronously off of
  // `sourceAudio.state.id` turning non-null, so it must already be current by
  // the time that happens.
  const attachSourceAudioByPath = useCallback(
    (filePath: string, fileName: string) => {
      lastAudioFilePathRef.current = filePath;
      // W4 (#7 trim追従): remember the FIRST audio attached under an active
      // span cap as the prefill audio (the #7 auto-load attaches exactly once at
      // mount, before any user action). Later attaches — a drag-and-drop swap —
      // leave this untouched, so their path differs from the cap path and the
      // wav auto-adjust lifts the cap. `undefined` cap → nothing to capture.
      if (a2vSeedMaxFrames !== undefined && a2vSeedCapPathRef.current === null) {
        a2vSeedCapPathRef.current = filePath;
      }
      return sourceAudio.uploadPath(filePath, fileName);
    },
    [sourceAudio, a2vSeedMaxFrames],
  );

  // Audio (`isA2v`) and the reference video (`icLoraActive`) are combinable,
  // not mutually exclusive: `isA2v` means "audio is the generation's driving
  // source" and holds whenever a wav is ready, regardless of any reference
  // video. When both hold, submission is an audio-driven single-clip chain
  // that ALSO carries the IC-LoRA reference-video fields (a reference-controlled
  // A2V — see `buildA2vRequest`). `referenceActive`/`icLoraActive` still express
  // the 128-grid dominance: an attached reference video (or a selected control
  // LoRA, which mandates one) forces width/height onto the 128 grid regardless
  // of any audio.
  //
  // Y1 (参照動画を外すとGenerate復帰不能になるバグの根治・案A): this determination is
  // now FULLY DYNAMIC off the live reference video / control LoRA rather than a
  // mount-frozen intent flag. Previously a routed #2 (reference-video intent)
  // froze IC-LoRA mode on at mount, so removing the reference video (❌/🔁, id →
  // null) left `isICLora` stuck true and `referenceNotReady`/`generateDisabled`
  // permanently blocked Generate. Reading the live `referenceVideoId` /
  // `controlLora` instead lets the mode relax the instant the reference video
  // goes away. The two strength overrides are deliberately NOT part of this — a
  // 🔁 replace that clears the id must drop IC-LoRA mode even if a strength is
  // still opted in (a 🔁-orphaned strength would otherwise re-stick the mode).
  const referenceActive = referenceVideoId !== null || controlLora !== null;
  const isA2v = audioReady;
  // IC-LoRA is active once a reference video is attached, or a control LoRA is
  // selected (which requires one) — either way width/height must live on the
  // 128 grid.
  const icLoraActive = referenceActive;

  // The active rounding multiple + width/height bounds. In IC-LoRA mode the
  // config maxima are floored onto the 128 grid first (1088 -> 1024), so a
  // rounded value can never slip past the real server ceiling.
  const active = useMemo(() => {
    if (!icLoraActive) {
      return {
        multiple: GENERAL_MULTIPLE,
        minWidth: baseLimits.minWidth,
        maxWidth: baseLimits.maxWidth,
        minHeight: baseLimits.minHeight,
        maxHeight: baseLimits.maxHeight,
      };
    }
    const maxWidth = floorToMultiple(baseLimits.maxWidth, IC_LORA_MULTIPLE, 0, baseLimits.maxWidth);
    const maxHeight = floorToMultiple(baseLimits.maxHeight, IC_LORA_MULTIPLE, 0, baseLimits.maxHeight);
    return {
      multiple: IC_LORA_MULTIPLE,
      minWidth: ceilToMultiple(baseLimits.minWidth, IC_LORA_MULTIPLE, baseLimits.minWidth, maxWidth),
      maxWidth,
      minHeight: ceilToMultiple(baseLimits.minHeight, IC_LORA_MULTIPLE, baseLimits.minHeight, maxHeight),
      maxHeight,
    };
  }, [icLoraActive, baseLimits]);

  // The limits exposed to the UI (SizeFields' input bounds) reflect the active
  // width/height grid; num_frames bounds are unaffected by IC-LoRA.
  const limits: FormLimits = useMemo(
    () => ({
      ...baseLimits,
      minWidth: active.minWidth,
      maxWidth: active.maxWidth,
      minHeight: active.minHeight,
      maxHeight: active.maxHeight,
    }),
    [baseLimits, active],
  );

  const [values, setValues] = useState<FormValues>(() => {
    // Seed on the correct grid from the start: an IC-LoRA route already handed
    // us 128-aligned width/height via `deriveGenerationParams`, but re-round
    // here defensively so the initial state is always self-consistent.
    const mult = seedICLora ? IC_LORA_MULTIPLE : GENERAL_MULTIPLE;
    const maxW = seedICLora
      ? floorToMultiple(config.limits.max_width, IC_LORA_MULTIPLE, 0, config.limits.max_width)
      : config.limits.max_width;
    const maxH = seedICLora
      ? floorToMultiple(config.limits.max_height, IC_LORA_MULTIPLE, 0, config.limits.max_height)
      : config.limits.max_height;
    const minW = seedICLora ? ceilToMultiple(MIN_WIDTH, IC_LORA_MULTIPLE, MIN_WIDTH, maxW) : MIN_WIDTH;
    const minH = seedICLora ? ceilToMultiple(MIN_HEIGHT, IC_LORA_MULTIPLE, MIN_HEIGHT, maxH) : MIN_HEIGHT;
    return {
      width: roundToMultiple(initial.width ?? config.generation_defaults.width, mult, minW, maxW),
      height: roundToMultiple(initial.height ?? config.generation_defaults.height, mult, minH, maxH),
      // W8: a right-click prefill can seed DURATION (comfort ceiling / material
      // length); otherwise the config default, unchanged. Re-snapped so the
      // initial state is always a valid 8n+1 value in range.
      numFrames: snapNumFrames(initial.numFrames ?? config.generation_defaults.num_frames, MIN_NUM_FRAMES, config.limits.max_num_frames),
      // W7: a right-click prefill can seed fps (material policy: selection
      // rate/scale); otherwise the config default.
      // 台帳§3-71/§3-72: snapped to a whole frame rate here too, so a non-integer
      // CONFIG default can't slip past every entry point. The fallback is the
      // constant, NOT `?? config…` — chaining back to the config would let the
      // raw value through exactly when the snap rejected it.
      frameRate: snapFrameRate(initial.frameRate ?? config.generation_defaults.frame_rate) ?? FRAME_RATE_FALLBACK,
      seed: config.generation_defaults.seed,
    };
  });

  // Smart comfort marker (2026-08-31, was 2026-08-18): which comfort row the
  // SERVED table (`config.limits.comfort_budgets`) gives this engine family +
  // acceleration configuration, or `null` when none applies — see
  // `shell/comfortTable.ts`'s `resolveComfortRow` for the compatibility shim,
  // the sage 3-value handling, and why `null` is a normal outcome (LTX 2.3's
  // default configuration deliberately has no row).
  //
  // `acceleration` must already be the EFFECTIVE object, and it is: this
  // hook's `acceleration` local is `deps.acceleration`, which `AppShell`
  // already ran through `effectiveAcceleration` before handing it down.
  //
  // ⚠ Computed HERE — above `gettingSize`, below `values` — rather than beside
  // the rest of the derived values further down, because the A2V wav
  // auto-adjust effect reads `spillThresholdFrames` and would otherwise see it
  // in its temporal dead zone.
  const comfortRow = useMemo(
    () => resolveComfortRow(config.limits, engineFamily, acceleration, sageAvailable),
    [config.limits, engineFamily, acceleration, sageAvailable],
  );
  const smartComfortFrames = comfortRow
    ? comfortFramesForBudget(
        values.width,
        values.height,
        comfortRow.singleBudget,
        limits.minNumFrames,
        limits.maxNumFrames,
        comfortRow.spatialFactor,
        comfortRow.temporalFactor,
      )
    : null;
  // `??`, not a ternary on `comfortRow`: a matched row whose smart derivation
  // still yields `null` (a hand-typed 0/blank width) falls back to the legacy
  // lookup exactly as it did before, rather than blanking the marker.
  const spillThresholdFrames =
    smartComfortFrames ?? resolveSpillFreeFrames(config.limits.spill_free_frames, values.width, values.height);
  const isComfortMarkerSmart = smartComfortFrames !== null;
  const isOverSpillThreshold = spillThresholdFrames !== null && values.numFrames > spillThresholdFrames;

  const [gettingSize, setGettingSize] = useState(false);
  const [getSizeError, setGetSizeError] = useState<string | null>(null);

  // N1: opt-in output crop — `null` ("not set") until the user opts in via
  // the UI's enable checkbox. The setter re-clamps to the CURRENT
  // width/height every call (not just at the moment the checkbox was
  // ticked), since `values.width`/`values.height` can change independently
  // afterward; a crop left stale by a later width/height shrink is instead
  // caught by `isValid` (see `chainUtils.isCropOutputValid`'s doc comment).
  const [cropOutput, setCropOutputState] = useState<CropOutput | null>(config.generation_defaults.crop_output ?? null);
  const setCropOutput = useCallback(
    (raw: CropOutput | null) => setCropOutputState(raw === null ? null : clampCropOutput(raw, values.width, values.height)),
    [values.width, values.height],
  );

  const setWidth = useCallback(
    (raw: number, snap = true) =>
      setValues((prev) => {
        if (snap) return { ...prev, width: roundToMultiple(raw, active.multiple, active.minWidth, active.maxWidth) };
        // Free manual entry: pass through untouched (no rounding, no clamp —
        // Gradio-faithful). Reject only non-finite input, keeping the prior value.
        if (!Number.isFinite(raw)) return prev;
        return { ...prev, width: raw };
      }),
    [active],
  );

  const setHeight = useCallback(
    (raw: number, snap = true) =>
      setValues((prev) => {
        if (snap) return { ...prev, height: roundToMultiple(raw, active.multiple, active.minHeight, active.maxHeight) };
        if (!Number.isFinite(raw)) return prev;
        return { ...prev, height: raw };
      }),
    [active],
  );

  const setNumFrames = useCallback(
    (raw: number, snap = true) =>
      setValues((prev) => {
        if (snap) return { ...prev, numFrames: snapNumFrames(raw, limits.minNumFrames, limits.maxNumFrames) };
        // Free manual entry: pass through untouched (no snapping, no clamp —
        // Gradio-faithful), mirroring `setWidth`/`setHeight`. Reject only
        // non-finite input, keeping the prior value.
        if (!Number.isFinite(raw)) return prev;
        return { ...prev, numFrames: raw };
      }),
    [limits.minNumFrames, limits.maxNumFrames],
  );

  // 台帳§3-71/§3-72: rounds as well as clamping now — the generation only ever
  // runs at whole frame rates (see `paramUtils.snapFrameRate`; the range is the
  // server's, the integer is our UI policy). `?? FRAME_RATE_MIN` keeps the
  // pre-existing "an emptied field becomes 1fps" behaviour: an
  // `<input type="number">` reports a cleared (or half-typed, e.g. `"29."`)
  // box as `""`, which the change handler passes on as `Number("") === 0`,
  // and 0 landed on 1 under the old `Math.min(60, Math.max(1, raw))` too.
  const setFrameRate = useCallback(
    (raw: number) => setValues((prev) => ({ ...prev, frameRate: snapFrameRate(raw) ?? FRAME_RATE_MIN })),
    [],
  );

  const setSeed = useCallback((raw: number) => setValues((prev) => ({ ...prev, seed: Math.trunc(raw) })), []);

  const applyPreset = useCallback(
    (name: string) => {
      const preset = config.generation_presets[name];
      if (!preset) return;
      // Compute the actually-applied (rounded) width/height locally so the
      // crop clamp below uses the SAME values this call is about to set —
      // not the stale `values.width/height` still in the closure. This
      // matters most in IC-LoRA mode, where `active`'s 128-grid rounding can
      // shrink the generation size well below the preset's own (64-grid)
      // width/height; clamping against the preset's raw values there could
      // leave `cropOutput` bigger than the generation size it's ceilinged
      // to, permanently blocking Generate.
      const newWidth = roundToMultiple(preset.width, active.multiple, active.minWidth, active.maxWidth);
      const newHeight = roundToMultiple(preset.height, active.multiple, active.minHeight, active.maxHeight);
      setValues((prev) => ({
        ...prev,
        width: newWidth,
        height: newHeight,
        numFrames: snapNumFrames(preset.num_frames, limits.minNumFrames, limits.maxNumFrames),
      }));
      // N1: mirror the preset's own crop_output (Gradio-faithful) — reflect
      // it when present, or turn crop OFF (back to `null`) when the preset
      // doesn't define one, rather than leaving a stale prior crop in place.
      setCropOutputState(preset.crop_output ? clampCropOutput(preset.crop_output, newWidth, newHeight) : null);
    },
    [config.generation_presets, active, limits.minNumFrames, limits.maxNumFrames],
  );

  const getSizeFromAviUtl2 = useCallback(async () => {
    setGettingSize(true);
    setGetSizeError(null);
    try {
      const editInfo = await nativeBridge.request("getEditInfo", {});
      setValues((prev) => ({
        ...prev,
        width: ceilToMultiple(editInfo.width, active.multiple, active.minWidth, active.maxWidth),
        height: ceilToMultiple(editInfo.height, active.multiple, active.minHeight, active.maxHeight),
      }));
    } catch (err) {
      setGetSizeError(err instanceof Error ? err.message : String(err));
    } finally {
      setGettingSize(false);
    }
  }, [nativeBridge, active]);

  // Re-snap width/height onto the 128 grid IC-LoRA needs, but ONLY once a
  // reference video actually becomes ready — the reference-video id
  // transitions from null to non-null. The previous approach re-snapped
  // unconditionally right after `referenceVideo.pick()` resolved, which also
  // fired on a CANCELLED/failed pick (leaving the form still on the 64 grid),
  // wrongly rounding the untouched width/height onto the 128 multiple. Keying
  // off the id transition instead means a cancel/error (id stays null) never
  // re-snaps; and by the time this effect runs, `icLoraActive` has already
  // flipped true, so `active` here holds the 128-grid bounds. `useSourceUpload`
  // (frozen, in chain/) returns nothing from `pick()`, so the id transition is
  // the only reliable "the choose succeeded" signal available here.
  const prevReferenceVideoIdRef = useRef<string | null>(referenceVideoId);
  useEffect(() => {
    const becameReady = prevReferenceVideoIdRef.current === null && referenceVideoId !== null;
    prevReferenceVideoIdRef.current = referenceVideoId;
    if (!becameReady) return;
    setValues((prev) => ({
      ...prev,
      width: roundToMultiple(prev.width, IC_LORA_MULTIPLE, active.minWidth, active.maxWidth),
      height: roundToMultiple(prev.height, IC_LORA_MULTIPLE, active.minHeight, active.maxHeight),
    }));
  }, [referenceVideoId, active]);

  // Contract v9: the ready reference video's measured duration, for the
  // IC-LoRA source card's `12.3s` readout. Same shape as the A2V wav-duration
  // probe below (keyed off the upload's id turning non-null, guarded against
  // re-probing the same id, cancelled/settled teardown), but reads the path
  // straight off `referenceVideo.state.filePath` (now that `useSourceUpload`
  // exposes it) and hits `fs.probeMediaInfo`. `.catch()` is mandatory: without
  // it a rejected probe becomes an unhandled rejection that fails the test
  // run — and unlike the audio probe, `fs.probeMediaInfo` DOES reject on a
  // BAD_REQUEST (empty path), so this isn't merely defensive.
  const [referenceVideoDurationSec, setReferenceVideoDurationSec] = useState<number | null>(null);
  const probedReferenceIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (referenceVideo.state.status !== "ready") {
      probedReferenceIdRef.current = null;
      setReferenceVideoDurationSec(null);
      return;
    }
    const id = referenceVideo.state.id;
    if (!id || probedReferenceIdRef.current === id) return;
    probedReferenceIdRef.current = id;
    const filePath = referenceVideo.state.filePath;
    if (!filePath) return;

    let cancelled = false;
    let settled = false;
    void nativeBridge
      .request("fs.probeMediaInfo", { filePath })
      .then((info) => {
        settled = true;
        if (cancelled) return;
        // Only a positive duration is meaningful (a still image / unknown
        // duration reports 0); anything else leaves the readout hidden.
        setReferenceVideoDurationSec(info.durationSec > 0 ? info.durationSec : null);
      })
      .catch(() => {
        settled = true;
        if (cancelled) return;
        setReferenceVideoDurationSec(null);
      });
    return () => {
      cancelled = true;
      // Same interrupted-probe re-claim as the audio effect: if this run was
      // torn down before the request settled, drop the id claim so the next
      // run for the same id re-probes instead of silently never probing.
      if (!settled) probedReferenceIdRef.current = null;
    };
  }, [referenceVideo.state.status, referenceVideo.state.id, referenceVideo.state.filePath, nativeBridge]);

  // A2V wav-duration auto-adjust (Group3 item11), mirroring Chain's
  // `useChainForm` but driving the single scalar `values.numFrames` instead of
  // `clips[0]`. `audioDurationSec` is the measured seconds for whatever
  // `sourceAudio` currently holds — `null` until a `.wav` is successfully
  // probed (and re-null'd whenever the attachment changes/clears). Drives both
  // the numFrames auto-adjust and `audioPrecheck`.
  const [audioDurationSec, setAudioDurationSec] = useState<number | null>(null);
  const [audioFramesAdjustedEvent, setAudioFramesAdjustedEvent] = useState<{
    id: number;
    frames: number;
    durationSec: number;
  } | null>(null);
  const audioAdjustCounterRef = useRef(0);
  // Guards against re-probing the SAME successful upload more than once (this
  // effect's deps include `values.frameRate`, which can change after a probe
  // already ran for the current id).
  const probedAudioIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (sourceAudio.state.status !== "ready") {
      probedAudioIdRef.current = null;
      setAudioDurationSec(null);
      return;
    }
    const id = sourceAudio.state.id;
    if (!id || probedAudioIdRef.current === id) return;
    probedAudioIdRef.current = id;
    const filePath = lastAudioFilePathRef.current;
    if (!filePath) return;

    let cancelled = false;
    let settled = false;
    void nativeBridge.request("fs.probeAudioDuration", { filePath }).then((probe) => {
      settled = true;
      if (cancelled) return;
      // Non-wav/unreadable: no-op — the server's 422 is the fallback authority.
      if (!probe.isWav || probe.durationSec <= 0) {
        setAudioDurationSec(null);
        return;
      }
      setAudioDurationSec(probe.durationSec);
      let suggested = snapNumFrames(
        suggestFramesForAudio(probe.durationSec, values.frameRate),
        limits.minNumFrames,
        limits.maxNumFrames,
      );
      // W8: cap the wav-derived suggestion at the current resolution's comfort
      // ceiling so a long wav never auto-adjusts DURATION past the point
      // generation starts spilling. Applies to EVERY A2V attach path (pick /
      // drag-and-drop / right-click #3/#7), an owner-approved intentional
      // behavior change. A null ceiling (unresolvable map) leaves the
      // suggestion untouched.
      //
      // 2026-08-31: this is the SAME `spillThresholdFrames` the comfort marker
      // draws, not an independent `spill_free_frames` lookup — so on a smart
      // engine the auto-adjusted DURATION lands exactly on the marker instead
      // of at the coarser legacy value. (E4 of the 2026-08-31 calibration
      // measured A2V against T2V at identical geometry and found every metric
      // within ±0.3%, so Create's single-shot line applies to A2V unchanged.)
      // Both the smart ceiling and the `spill_free_frames` entries are valid
      // 8n+1 values, so clamping down to one keeps `suggested` on-grid.
      if (spillThresholdFrames !== null && suggested > spillThresholdFrames) suggested = spillThresholdFrames;
      // W4 (#7 trim追従): while the attached audio is STILL the prefill-attached
      // one (path matches the captured cap path), additionally cap the wav
      // suggestion at the span-derived DURATION so #7 follows the trimmed ribbon
      // rather than the full uploaded file. Both `a2vSeedMaxFrames` and the span
      // seed are valid 8n+1 values, so `min` keeps `suggested` on-grid. Swapping
      // the audio makes the paths differ, lifting the cap (ordinary full-wav
      // follow resumes).
      if (
        a2vSeedMaxFrames !== undefined &&
        a2vSeedCapPathRef.current !== null &&
        lastAudioFilePathRef.current === a2vSeedCapPathRef.current &&
        suggested > a2vSeedMaxFrames
      ) {
        suggested = a2vSeedMaxFrames;
      }
      setValues((prev) => ({ ...prev, numFrames: suggested }));
      audioAdjustCounterRef.current += 1;
      setAudioFramesAdjustedEvent({ id: audioAdjustCounterRef.current, frames: suggested, durationSec: probe.durationSec });
    });
    return () => {
      cancelled = true;
      // If the request never resolved before this effect was torn down
      // (unmount, or a dep like `frameRate` changing mid-flight), undo the
      // `probedAudioIdRef` claim so the next effect run for this same id
      // re-probes instead of silently never probing at all — otherwise the
      // guard above (`probedAudioIdRef.current === id`) would treat the
      // interrupted attempt as done and Generate's audio-length gate would
      // never see a duration for this upload.
      if (!settled) probedAudioIdRef.current = null;
    };
  }, [sourceAudio.state.status, sourceAudio.state.id, nativeBridge, values.frameRate, values.width, values.height, spillThresholdFrames, limits.minNumFrames, limits.maxNumFrames, a2vSeedMaxFrames]);

  // A2V-only: gates on whether the attached wav's MEASURED duration is long
  // enough for `values.numFrames` (`chainUtils.audioLengthPrecheck`). `null`
  // outside A2V, or before any wav duration was measured — folded into
  // `isValid` below and surfaced standalone for the "audio too short" warning.
  const audioPrecheck = useMemo<AudioLengthPrecheckResult | null>(
    () => (isA2v && audioDurationSec != null ? audioLengthPrecheck(audioDurationSec, values.numFrames, values.frameRate) : null),
    [isA2v, audioDurationSec, values.numFrames, values.frameRate],
  );

  const promptError = isValidPrompt(prompt) ? null : "Prompt must be 1-2000 characters.";
  // Gate submission on the width/height grid too: free-typed values are passed
  // through unsnapped (see `setWidth`/`setHeight`), so an off-grid or
  // out-of-range dimension must block Generate here. In IC-LoRA mode
  // `active.multiple` is 128, so the gate demands the 128 grid.
  const dimensionsOnGrid =
    isDimensionOnGrid(values.width, active.multiple, active.minWidth, active.maxWidth) &&
    isDimensionOnGrid(values.height, active.multiple, active.minHeight, active.maxHeight);

  // Same free-typed-manual-entry gate as width/height, now that `setNumFrames`
  // also accepts `snap=false` (`DurationField`'s number input): a hand-typed
  // non-8n+1 or out-of-range value must block Generate. Uses the SAME
  // min/max source `setNumFrames`'s snap path already uses
  // (`limits.minNumFrames/maxNumFrames`, unaffected by IC-LoRA).
  const numFramesOnGrid = isNumFramesOnGrid(values.numFrames, limits.minNumFrames, limits.maxNumFrames);

  // `<lora:name:strength>` tags found in the prompt — parsed once here
  // (rather than only at request-build time) so the reference-video gates
  // below can evaluate live, before the user ever clicks Generate. Mirrors
  // Chain's `useChainForm.parsedPrompt`; reused by `toGenerateRequest`/
  // `buildA2vRequest` instead of re-running `parseLoraPrompt`.
  const parsedPrompt = useMemo(() => parseLoraPrompt(prompt), [prompt]);

  // IC-LoRA UI redesign (第5波): the final `loras[]` the request would carry —
  // the panel's `controlLora` selection first, the prompt's STYLE tags after,
  // deduped by name (`combineLoras`, `lora/controlLoras.ts`). Computed here
  // (not only in `toGenerateRequest`) so the reference-video gates below can
  // react live, same reasoning as `parsedPrompt` above.
  const mergedLoras = useMemo(() => combineLoras(controlLora, parsedPrompt.loras), [controlLora, parsedPrompt]);

  // N3/N2: cross-field validity for the reference-video (control IC-LoRA)
  // path — mirrors `chainUtils.isReferenceChainValid`'s "reference_video_id
  // requires a loras[] entry" rule (N3), plus its converse ("a control LoRA
  // requires a reference video"). `referenceVideoId` is non-null only once
  // the upload is `"ready"` (`useSourceUpload`), so this only ever fires for
  // an attached reference video, not a merely-routed IC-LoRA intent.
  // `referenceVideoNeedsLoras` checks the MERGED array (not just the
  // prompt's own tags) so a control-LoRA-only selection already satisfies the
  // "needs at least one loRA" rule without also requiring a STYLE tag.
  const referenceVideoNeedsLoras = referenceVideoId !== null && mergedLoras.length === 0;
  const controlLoraNeedsReferenceVideo = controlLora !== null && referenceVideoId === null;
  const isReferenceValid = !referenceVideoNeedsLoras && !controlLoraNeedsReferenceVideo;

  // W7: build the same gate set as `isValid` used to be, but as a reason-code
  // list so the UI can tell the user exactly what to fix. Each sub-boolean
  // expression is UNCHANGED from the old AND-chain (regression-safe); a failing
  // gate just pushes its code. `isValid` is then simply "no reasons".
  //  - A2V's `(!isA2v || audioPrecheck.ok)` term is pushed in its negated
  //    (De Morgan) form `isA2v && !audioPrecheck.ok` — logically identical.
  //  - `isReferenceValid` is split back into its two constituent gates
  //    (`referenceVideoNeedsLoras` / `controlLoraNeedsReferenceVideo`) so each
  //    gets its own specific instruction.
  // N1: a crop, if enabled, must still be a valid 32-pixel-grid crop of the
  // CURRENT width/height (see `chainUtils.isCropOutputValid`'s doc comment —
  // this is what catches a crop left stale by a later width/height shrink).
  const validityReasons: GenerationValidityReason[] = [];
  if (promptError !== null) validityReasons.push("promptEmpty");
  if (!dimensionsOnGrid) validityReasons.push("dimensionsOffGrid");
  if (!numFramesOnGrid) validityReasons.push("numFramesOffGrid");
  if (isA2v && !(audioPrecheck?.ok ?? false)) validityReasons.push("audioTooShort");
  if (referenceVideoNeedsLoras) validityReasons.push("referenceNeedsLoras");
  if (controlLoraNeedsReferenceVideo) validityReasons.push("controlNeedsReference");
  if (!isCropOutputValid(cropOutput, values.width, values.height)) validityReasons.push("cropInvalid");
  // NAG (2026-07-28): enabled + blank negative-prompt body blocks Generate
  // client-side (the server 422s the same combination) — see
  // `nagSettings.isNagNegativeEmpty`'s own doc comment.
  if (isNagNegativeEmpty(nag)) validityReasons.push("nagNegativeEmpty");
  // §1-6 拡張 (2026-08-01): a reference-video trim we asked for but did not get.
  // The upload itself succeeded (`ready`, with a usable id), but it holds the
  // WHOLE file rather than the ribbon's range — generating would condition on
  // footage the user never selected, so block instead. The flag is only ever set
  // when a trim was actually requested (`useSourceUpload`), i.e. for a routed #2
  // right-click; the manual 📁 pick never arms it.
  if (referenceVideo.state.trimFailed) validityReasons.push("referenceTrimFailed");
  const isValid = validityReasons.length === 0;

  const durationHint = formatDurationHint(values.numFrames, values.frameRate);

  const estimateSeconds = useMemo(
    () => estimateGenerationSeconds(values.width, values.height, values.numFrames),
    [values.width, values.height, values.numFrames],
  );
  const estimateLabel = `Est. ${formatEstimate(estimateSeconds)}`;

  const toGenerateRequest = useCallback((): GenerateRequest => {
    // M5/第5波: the prompt is the source of truth for STYLE
    // `<lora:name:strength>` tags (Docs/API_REFERENCE.md §5.3) — split it
    // into the body the backend renders and the tags' own `LoraSpec[]`
    // right here, then merge in the panel's `controlLora` selection via the
    // SAME `mergedLoras` memo the gates above already computed (control
    // first, Gradio-faithful — see `combineLoras`'s doc comment).
    const { strippedPrompt } = parsedPrompt;
    const loras = mergedLoras;
    // N3: `reference_video_id`/the two strength fields require a non-empty
    // `loras[]` server-side (422 otherwise) — only ever sent together, and
    // only once both a ready reference video AND at least one loRA (control
    // or style) are present. A ready-but-lora-less reference video is
    // submitted as a plain T2V/I2V request instead of silently 422ing (see
    // `isReferenceValid`, which blocks Generate in this state before
    // submission is even possible — this omission is the belt to that
    // suspenders).
    const sendReference = referenceVideoId !== null && loras.length > 0;
    return {
      prompt: strippedPrompt,
      width: values.width,
      height: values.height,
      // N1: only sent while enabled (non-null) — `isValid` already gates
      // submission on `isCropOutputValid`, so a stale/off-grid crop never
      // reaches this point.
      ...(cropOutput ? { crop_output: cropOutput } : {}),
      num_frames: values.numFrames,
      frame_rate: values.frameRate,
      seed: values.seed,
      // NAG (2026-07-28)/VSF (2026-07-29): the additive `negative_prompt`/
      // `nag_*`/`neg_method`/`vsf_scale` set of 7, built by the single shared
      // contract function — `{}` (no keys at all)
      // whenever NAG is off, so this stays byte-identical to before NAG
      // existed (see `nagSettings.nagRequestFields`'s doc comment).
      ...nagRequestFields(nag),
      // Acceleration (2026-07-31): `attention_backend`, and only while the user
      // has moved off the server default — `{}` otherwise, same additive rule
      // as NAG above (see `accelerationRequestFields`).
      ...accelerationRequestFields(acceleration),
      ...(loras.length > 0 ? { loras } : {}),
      ...(sendReference ? { reference_video_id: referenceVideoId } : {}),
      // `!= null` (not truthiness): `0.0` is a valid, distinct strength value
      // that must still be sent, not treated as "unset".
      ...(sendReference && conditioningAttentionStrength != null
        ? { conditioning_attention_strength: conditioningAttentionStrength }
        : {}),
      ...(sendReference && referenceVideoStrength != null ? { reference_video_strength: referenceVideoStrength } : {}),
    };
  }, [parsedPrompt, mergedLoras, values, cropOutput, referenceVideoId, conditioningAttentionStrength, referenceVideoStrength, nag, acceleration]);

  const buildA2vRequest = useCallback(
    (conditioningImages: ConditioningImage[]): A2vChainPayload => {
      // `strippedPrompt` still comes from the prompt's tag split, but `loras`
      // must be the SAME `mergedLoras` (control-panel selection + STYLE tags)
      // `toGenerateRequest` uses — otherwise a control LoRA selected alongside
      // a reference video drops out of the A2V payload and the server 422s the
      // reference fields (N3: `reference_video_id` requires a non-empty
      // `loras[]`).
      const { strippedPrompt } = parsedPrompt;
      const loras = mergedLoras;
      // N3 gate, same rule as `toGenerateRequest`'s `sendReference`: the
      // reference-video fields are only ever carried alongside a ready
      // reference video AND at least one loRA. Without a loRA the adapter
      // fields are omitted entirely (a plain audio-driven chain).
      const sendReference = referenceVideoId !== null && loras.length > 0;
      return buildA2vChainPayload({
        audioId: sourceAudio.state.id!,
        numFrames: values.numFrames,
        prompt: strippedPrompt,
        width: values.width,
        height: values.height,
        // N1: the crop field is rendered unconditionally in `GenerationForm`
        // (right after `SizeFields`), so a crop the user set still applies
        // in A2V mode — the CropOutputField/isValid gate above is what keeps
        // it well-formed regardless of which submission path fires.
        ...(cropOutput ? { cropOutput } : {}),
        frameRate: values.frameRate,
        seed: values.seed,
        chunkedUpsample: true,
        // NAG (2026-07-28): `buildA2vChainPayload` folds this straight into
        // `nagRequestFields`, overriding `negative_prompt` only while
        // `nag.enabled` — see that function's own doc comment.
        nag,
        // Acceleration (2026-07-31): `buildA2vChainPayload` folds this through
        // the same `accelerationRequestFields` contract — nothing is added to
        // the payload while the server default is selected.
        acceleration,
        ...(conditioningImages.length > 0 ? { conditioningImages } : {}),
        ...(loras.length > 0 ? { loras } : {}),
        // Reference-controlled A2V: when a reference video and a loRA are both
        // present, ride the same adapter fields `toGenerateRequest` builds
        // from `referenceVideoId`/the two strengths. A `null` strength is left
        // off so `buildA2vChainPayload` falls back to its own "unset = 1.0,
        // omit the field" default, matching the server's omitted-means-default
        // semantics.
        ...(sendReference
          ? {
              useAdapter: true,
              referenceVideoId,
              ...(conditioningAttentionStrength != null ? { controlAdherence: conditioningAttentionStrength } : {}),
              ...(referenceVideoStrength != null ? { referenceStrength: referenceVideoStrength } : {}),
            }
          : {}),
      });
    },
    [
      parsedPrompt,
      mergedLoras,
      values,
      cropOutput,
      sourceAudio.state.id,
      referenceVideoId,
      conditioningAttentionStrength,
      referenceVideoStrength,
      nag,
      acceleration,
    ],
  );

  return {
    values,
    limits,
    presets: config.generation_presets,
    durationHint,
    promptError,
    isValid,
    validityReasons,
    gettingSize,
    getSizeError,
    spillThresholdFrames,
    isOverSpillThreshold,
    isComfortMarkerSmart,
    estimateSeconds,
    estimateLabel,
    isICLora: icLoraActive,
    referenceVideo,
    referenceVideoDurationSec,
    cropOutput,
    setCropOutput,
    conditioningAttentionStrength,
    setConditioningAttentionStrength,
    referenceVideoStrength,
    setReferenceVideoStrength,
    controlLora,
    setControlLora,
    setControlLoraStrength,
    controlLoraNames,
    referenceVideoNeedsLoras,
    controlLoraNeedsReferenceVideo,
    isReferenceValid,
    sourceAudio,
    attachSourceAudioByPath,
    isA2v,
    audioPrecheck,
    audioDurationSec,
    audioFramesAdjustedEvent,
    setWidth,
    setHeight,
    setNumFrames,
    setFrameRate,
    setSeed,
    applyPreset,
    getSizeFromAviUtl2,
    toGenerateRequest,
    buildA2vRequest,
  };
}
