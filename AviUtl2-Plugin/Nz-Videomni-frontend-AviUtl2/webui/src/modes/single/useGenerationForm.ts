import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
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
import { comfortFramesForBudget, comfortProfileMissing, resolveComfortRow } from "../../shell/comfortTable";
import { fixedFrameRate, frameLimitsFor, LTX_GENERATION_CONSTRAINTS } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import type { SamplingConstraints } from "../../api/types";
import { clampSamplingValue, samplingDefaults, samplingRequestFields } from "./samplingValues";
import type { SamplingValues } from "./samplingValues";
import { singleAudioPrecheck, suggestFramesForSingleAudio } from "./audioFit";
import { MIN_HEIGHT, MIN_WIDTH } from "./defaultConfig";
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
 * multiples of 128, vs. the general-mode multiple (Docs/TIMELINE_ALPHA_REQUIREMENTS.md
 * §2). Kept identical to `deriveGenerationParams`'s `multiple` for IC-LoRA. */
const IC_LORA_MULTIPLE = 128;
// The general-mode multiple is the loaded base model's
// `constraints.dimensionMultiple`, read as `generalMultiple` inside the hook.

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
  /** The three sampling sliders' values, or `null` while the loaded base model
   * has no sampling knobs (LTX — the ordinary case). Seeded from, and reset
   * to, `constraints.sampling.*.default` whenever the base model's rules
   * change. */
  sampling: SamplingValues | null;
}

// The sampling values' type and rules live in `samplingValues.ts` (shared with
// Chained, §1-106); re-exported so existing importers keep their path.
export type { SamplingValues } from "./samplingValues";

/** The single-clip A2V Strength's server default, or 1.0 (full strength)
 * for an engine that takes no source audio on `/generate`. */
function sourceAudioStrengthDefault(rules: GenerationConstraints): number {
  return rules.singleSourceAudio?.strength.default ?? 1.0;
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
  /** §1-31: the LOADED transformer's weight class
   * (`useBaseModels().activeWeightClass` — `"4bit"`, `"8bit"`, `"q6k"`),
   * matched against the served rows' `requires.weight_class` beside
   * {@link engineFamily}. Omitted/`""` = unknown: no `weight_class` joins the
   * match, so a typed row misses and the marker falls back to the legacy
   * table. */
  weightClass?: string | undefined;
  /** The LOADED base model's numeric rules
   * (`useBaseModels().activeConstraints`) — the frame grid and its floor, the
   * size multiple, the fps range (a fixed rate disables the field) and the
   * sampling sliders. Omitted = the LTX rules (`LTX_GENERATION_CONSTRAINTS`).
   * When it changes (a base-model switch) the
   * form re-snaps DURATION and the size onto the new grid, pins a fixed fps,
   * and resets the sampling sliders to the new defaults. */
  constraints?: GenerationConstraints | undefined;
  /** The loaded base model's engine
   * cannot take a reference video (`reference_video` — Create HIDES the
   * IC-LoRA accordion, with the control-LoRA picker inside it). While true the
   * form reads the reference video AND the control LoRA as absent — the same
   * shape as NAG's `effectiveNag`: nothing of them rides on a request, none of
   * their Generate gates fire, and the 128 grid does not apply. The user's
   * selections are not cleared, so a base model that takes them gets them
   * back. Style `<lora:…>` tags in the prompt are untouched (the server's 422
   * covers them). Omitted = available. */
  referenceVideoUnavailable?: boolean | undefined;
}

function noopSetControlLora(): void {
  // Default when the caller doesn't thread AppShell's setter through (every
  // pre-existing unit test) — the form has no way to ever set `controlLora`
  // away from `null` in that case, so silently doing nothing is correct, not
  // just a placeholder.
}

export interface UseGenerationFormResult {
  values: FormValues;
  limits: FormLimits;
  presets: AppConfig["generation_presets"];
  durationHint: string;
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
   * deliberately has none for the 4bit and Q6_K weight classes; the 8bit row
   * matches on weight class alone), or the smart derivation itself returns `null` (a
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
   * set" (omitted from the request). See `CommonGenerationFields.CropOutputField`:
   * any integer at least 32 and no larger than the CURRENT generation
   * width/height — there is no grid. The judgement lives in exactly one
   * place, `chainUtils.isCropOutputValid`, folded into `isValid` below.
   * Mirrors Chain's `useChainForm.cropOutput`. */
  cropOutput: CropOutput | null;
  /** Stores `cropOutput` verbatim — what the user typed is never altered
   * (free entry, 2026-09-16), so an emptied input arrives here as `NaN`. The
   * range is judged by `chainUtils.isCropOutputValid` (folded into `isValid`),
   * not by this setter; the only surviving clamp is `applyPreset`'s, for the
   * value the PROGRAM supplies. `null` clears it (omitted from the request). */
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
  /** Contract v7 (drag-and-drop): attaches an audio file by path — the
   * non-`ui.pickFile` attach path (drag-and-drop and the #7 right-click
   * auto-load). A thin wrapper over `sourceAudio.uploadPath` that drops the
   * upload's outcome value; the wav-duration auto-adjust probe reads
   * `sourceAudio.state.filePath`, which `uploadPath` itself sets. */
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
  /** Single-clip A2V's Strength slider (`constraints.source_audio`), or
   * `null` when A2V goes through `/generate/chain` (LTX). Its presence is
   * what switches the A2V submit to `/generate`. */
  singleSourceAudio: GenerationConstraints["singleSourceAudio"];
  /** The Strength `source_audio.strength` carries on the `/generate` route
   * (0.0 = same as no source audio). Starts at, and goes back to on a
   * base-model switch, the served default (1.0 without one). */
  sourceAudioStrength: number;
  /** Clamped into the served range; ignored without one. */
  setSourceAudioStrength: (value: number) => void;
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
   * (`constraints.dimensionMultiple`, or 128 in IC-LoRA mode) and clamps to the active bounds — used by the
   * stepper arrows, ↑↓ keys, slider, presets, and the getSize/reference-video
   * flows. `snap === false` (free keyboard typing) passes the value straight
   * through with only a NaN guard (Gradio-faithful: no rounding, no clamping);
   * the grid gate lives in `isValid` so an off-grid manual entry blocks submit. */
  setWidth: (value: number, snap?: boolean) => void;
  /** Sets height. See `setWidth` for the `snap` semantics. */
  setHeight: (value: number, snap?: boolean) => void;
  /** Sets `values.numFrames`. `snap` (default true) rounds `value` onto the
   * frame grid in force (`frameGrid`) and clamps to `[limits.minNumFrames, limits.maxNumFrames]` —
   * used by the stepper arrows, ↑↓ keys, and slider. `snap === false` (free
   * keyboard typing) passes the value straight through with only a NaN
   * guard; the grid gate lives in `isValid` so an off-grid manual entry
   * blocks submit. Mirrors `setWidth`/`setHeight`'s `snap` semantics. */
  setNumFrames: (value: number, snap?: boolean) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
  /** Sets one sampling slider. The value is clamped into
   * the served range (and the step count rounded to a whole number). A no-op
   * while `values.sampling` is `null`. */
  setSamplingValue: (key: keyof SamplingValues, value: number) => void;
  /** The served sampling ranges (`constraints.sampling`), or `null` — what
   * `GenerationForm` draws the three sliders from. */
  samplingConstraints: SamplingConstraints | null;
  /** The fixed frame rate the loaded engine runs at, or `null` when the user
   * chooses (`constraints.frame_rate.min === max`). Disables the fps field. */
  fixedFrameRate: number | null;
  /** The `num_frames` lattice in force (`constraints.frameGrid`) — the
   * DURATION slider's step and the preset/guard snap read it. */
  frameGrid: GenerationConstraints["frameGrid"];
  /** The general-mode width/height multiple in force (the size sliders' step). */
  dimensionStep: number;
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
 * its `limits`. Every setter snaps to the loaded base model's rules (the
 * `constraints` dep — the LTX rules when omitted; 128-multiple sizes in
 * IC-LoRA mode) so the form can never hold an invalid combination. */
export function useGenerationForm(
  config: AppConfig,
  prompt: string,
  deps: UseGenerationFormDeps = {},
): UseGenerationFormResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;

  // IC-LoRA UI redesign (第5波): the panel's selection state is owned by the
  // caller (AppShell in production — see `UseGenerationFormDeps`'s doc
  // comment) so it survives a tab switch away from Create; every
  // pre-existing unit test that omits these options gets `null` and a no-op
  // setter, i.e. "no control LoRA, ever" (this hook never introduces its own
  // fallback `useState` for it — there is exactly one owner).
  // While the reference-video feature is closed, the control
  // LoRA reads as "none" (see `referenceVideoUnavailable`). The ONE place it
  // is folded; every gate and request builder below reads this local.
  const referenceVideoUnavailable = deps.referenceVideoUnavailable ?? false;
  const controlLora = referenceVideoUnavailable ? null : (deps.controlLora ?? null);
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
  // §1-31: the weight class joins the match beside the engine family.
  const weightClass = deps.weightClass;
  // The loaded base model's numeric rules. Omitted = LTX.
  const rules = deps.constraints ?? LTX_GENERATION_CONSTRAINTS;
  const frameGrid = rules.frameGrid;
  const generalMultiple = rules.dimensionMultiple;
  const fixedFps = fixedFrameRate(rules);

  const setControlLoraStrength = useCallback(
    (raw: number) => {
      if (!controlLora) return;
      setControlLora({ name: controlLora.name, strength: clampLoraStrength(raw) });
    },
    [controlLora, setControlLora],
  );

  const baseLimits: FormLimits = useMemo(
    () => ({
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
      // The engine's own floor and ceiling, capped by the config's maximum
      // (`frameLimitsFor`).
      ...frameLimitsFor(rules, config.limits.max_num_frames),
    }),
    [config.limits.max_width, config.limits.max_height, config.limits.max_num_frames, rules],
  );

  const referenceVideo = useSourceUpload("video", { nativeBridge });
  // The same fold for the reference video itself.
  const referenceVideoId = referenceVideoUnavailable ? null : referenceVideo.state.id;

  // Reference-video (control IC-LoRA) strength overrides — `null` ("not set")
  // until the user opts in via the UI's enable checkbox, matching the
  // server's own "omitted = use its own default" semantics. Mirrors Chain's
  // `useChainForm.conditioningAttentionStrength`/`referenceVideoStrength`.
  const [conditioningAttentionStrength, setConditioningAttentionStrengthState] = useState<number | null>(null);
  const setConditioningAttentionStrength = useCallback(
    (raw: number | null) =>
      setConditioningAttentionStrengthState(raw == null ? null : clamp(raw, MIN_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH)),
    [],
  );
  const [referenceVideoStrength, setReferenceVideoStrengthState] = useState<number | null>(null);
  const setReferenceVideoStrength = useCallback(
    (raw: number | null) =>
      setReferenceVideoStrengthState(raw == null ? null : clamp(raw, MIN_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH)),
    [],
  );

  // Group3 item11 (A2V move to Create): the optional A2V source-audio slot.
  //
  // §3-36 (2026-09-04): the local `filePath` the wav-duration probe below needs
  // now comes straight off `sourceAudio.state.filePath`. `useSourceUpload` sets
  // that field on EVERY transition — a `ui.pickFile` pick, a direct
  // `uploadPath` (drag-and-drop / #7) and the H3 Reference `adopt` alike —
  // so the plain `nativeBridge` is passed through here as it is everywhere
  // else. The old wav-picker snoop wrapper (`audioProbingBridge` capturing the
  // pick's path into a `lastAudioFilePathRef`) dated from contract v9, before
  // `SourceUploadState` carried `filePath` at all; it was the last consumer of
  // that mechanism, so it is gone and this file has one source of truth for
  // the audio path — matching the reference-video slot right below.
  const sourceAudio = useSourceUpload("audio", {
    nativeBridge,
    // 台帳 §1-113: the 📁 dialog also offers video files (only their sound is
    // used); the upload stays `kind: "audio"`.
    pickKind: "audioOrVideo",
  });
  const audioReady = sourceAudio.state.status === "ready";

  // Contract v7 (drag-and-drop): the non-`ui.pickFile` attach path — D&D, and
  // the #7 right-click auto-load (the wav-duration probe reads
  // `sourceAudio.state.filePath`, which `uploadPath` sets for every caller).
  const attachSourceAudioByPath = useCallback(
    (filePath: string, fileName: string) => {
      // `uploadPath` resolves with the upload's outcome since Alpha Gen 第 2 弾;
      // this attach path only ever needed the settle, so the value is dropped.
      return sourceAudio.uploadPath(filePath, fileName).then(() => undefined);
    },
    [sourceAudio],
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
        multiple: generalMultiple,
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
  }, [icLoraActive, baseLimits, generalMultiple]);

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

  // Seeded from the config defaults on the general grid. A right-click never
  // seeds this initializer any more (台帳 §1-114): `SingleScreen` applies its
  // `prefillMaterial` seed through the ordinary setters below.
  const [values, setValues] = useState<FormValues>(() => {
    return {
      width: roundToMultiple(config.generation_defaults.width, generalMultiple, MIN_WIDTH, config.limits.max_width),
      height: roundToMultiple(config.generation_defaults.height, generalMultiple, MIN_HEIGHT, config.limits.max_height),
      // Re-snapped so the initial state is always on the frame grid in force
      // and in range.
      numFrames: snapNumFrames(
        config.generation_defaults.num_frames,
        baseLimits.minNumFrames,
        baseLimits.maxNumFrames,
        frameGrid,
      ),
      // 台帳§3-71/§3-72: snapped to a whole frame rate here too, so a non-integer
      // CONFIG default can't slip past every entry point. The fallback is the
      // constant, NOT `?? config…` — chaining back to the config would let the
      // raw value through exactly when the snap rejected it.
      // An engine fixed at one rate takes it, whatever the config said.
      frameRate: fixedFps ?? snapFrameRate(config.generation_defaults.frame_rate) ?? FRAME_RATE_FALLBACK,
      seed: config.generation_defaults.seed,
      sampling: samplingDefaults(rules.sampling),
    };
  });

  // Single-clip A2V Strength (the `/generate` route). Kept beside `values`
  // rather than in it: it rides only on that route's `source_audio`.
  const [sourceAudioStrength, setSourceAudioStrengthRaw] = useState(() => sourceAudioStrengthDefault(rules));
  const singleSourceAudio = rules.singleSourceAudio;
  const setSourceAudioStrength = useCallback(
    (raw: number) => {
      if (!singleSourceAudio || !Number.isFinite(raw)) return;
      setSourceAudioStrengthRaw(clamp(raw, singleSourceAudio.strength.min, singleSourceAudio.strength.max));
    },
    [singleSourceAudio],
  );

  // Follow a change of the loaded base model's rules (a
  // base-model switch while the form is mounted). Skips the first render —
  // the lazy initializer above already used these rules — and fires only on a
  // real change: `useBaseModels` keeps the object stable while its content is.
  // DURATION and the size are re-snapped onto the new grid (a value valid on
  // the old one would otherwise block Generate), a fixed fps is pinned, and the
  // sampling sliders and the A2V Strength go back to the new base model's
  // defaults.
  const appliedRulesRef = useRef(rules);
  useEffect(() => {
    if (appliedRulesRef.current === rules) return;
    appliedRulesRef.current = rules;
    setValues((prev) => ({
      ...prev,
      width: roundToMultiple(prev.width, active.multiple, active.minWidth, active.maxWidth),
      height: roundToMultiple(prev.height, active.multiple, active.minHeight, active.maxHeight),
      numFrames: snapNumFrames(prev.numFrames, baseLimits.minNumFrames, baseLimits.maxNumFrames, frameGrid),
      frameRate: fixedFps ?? prev.frameRate,
      sampling: samplingDefaults(rules.sampling),
    }));
    setSourceAudioStrengthRaw(sourceAudioStrengthDefault(rules));
  }, [rules, active, baseLimits, frameGrid, fixedFps]);

  // Smart comfort marker (2026-08-31, was 2026-08-18): which comfort row the
  // SERVED table (`config.limits.comfort_budgets`) gives this engine family +
  // acceleration configuration, or `null` when none applies — see
  // `shell/comfortTable.ts`'s `resolveComfortRow` for the compatibility shim,
  // the sage 3-value handling, and why `null` is a normal outcome (LTX 2.3's
  // default configuration deliberately has no row for 4bit and Q6_K; the 8bit
  // row matches on weight class alone).
  //
  // `acceleration` must already be the EFFECTIVE object, and it is: this
  // hook's `acceleration` local is `deps.acceleration`, which `AppShell`
  // already ran through `effectiveAcceleration` before handing it down.
  //
  // ⚠ Computed HERE — above `gettingSize`, below `values` — rather than beside
  // the rest of the derived values further down, because the A2V wav
  // auto-adjust effect reads `spillThresholdFrames` and would otherwise see it
  // in its temporal dead zone.
  //
  // An engine the served table has NO profile for has no
  // comfort line at all — `noComfortLine` drops the marker, its warning and the
  // A2V cap rather than falling back to the LTX `spill_free_frames` lookup
  // (`comfortTable.comfortProfileMissing` says why only that case does).
  const noComfortLine = comfortProfileMissing(config.limits, engineFamily);
  const comfortRow = useMemo(
    () => resolveComfortRow(config.limits, engineFamily, acceleration, sageAvailable, weightClass),
    [config.limits, engineFamily, acceleration, sageAvailable, weightClass],
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
  const spillThresholdFrames = noComfortLine
    ? null
    : (smartComfortFrames ?? resolveSpillFreeFrames(config.limits.spill_free_frames, values.width, values.height));
  const isComfortMarkerSmart = smartComfortFrames !== null;
  const isOverSpillThreshold = spillThresholdFrames !== null && values.numFrames > spillThresholdFrames;

  const [gettingSize, setGettingSize] = useState(false);
  const [getSizeError, setGetSizeError] = useState<string | null>(null);

  // N1: opt-in output crop — `null` ("not set") until the user opts in via
  // the UI's enable checkbox. Free entry (2026-09-16): the setter stores what
  // it is handed, unchanged (an emptied input arrives as `NaN`). Out-of-range,
  // blank or non-integer crops — including one left stale by a later
  // width/height shrink — are caught by `isValid` (see
  // `chainUtils.isCropOutputValid`'s doc comment), the single judgement point.
  // The only clamp left is `applyPreset`'s, which supplies a PROGRAM value.
  const [cropOutput, setCropOutput] = useState<CropOutput | null>(config.generation_defaults.crop_output ?? null);

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
        if (snap) return { ...prev, numFrames: snapNumFrames(raw, limits.minNumFrames, limits.maxNumFrames, frameGrid) };
        // Free manual entry: pass through untouched (no snapping, no clamp —
        // Gradio-faithful), mirroring `setWidth`/`setHeight`. Reject only
        // non-finite input, keeping the prior value.
        if (!Number.isFinite(raw)) return prev;
        return { ...prev, numFrames: raw };
      }),
    [limits.minNumFrames, limits.maxNumFrames, frameGrid],
  );

  // 台帳§3-71/§3-72: rounds as well as clamping now — the generation only ever
  // runs at whole frame rates (see `paramUtils.snapFrameRate`; the range is the
  // server's, the integer is our UI policy). `?? FRAME_RATE_MIN` keeps the
  // pre-existing "an emptied field becomes 1fps" behaviour: an
  // `<input type="number">` reports a cleared (or half-typed, e.g. `"29."`)
  // box as `""`, which the change handler passes on as `Number("") === 0`,
  // and 0 landed on 1 under the old `Math.min(60, Math.max(1, raw))` too.
  // A fixed rate cannot be moved (the field is disabled,
  // but the `project` prefill overwrite still calls this setter).
  const setFrameRate = useCallback(
    (raw: number) => setValues((prev) => ({ ...prev, frameRate: fixedFps ?? snapFrameRate(raw) ?? FRAME_RATE_MIN })),
    [fixedFps],
  );

  // One setter for the three sampling sliders, clamped into
  // the served range. The step count is a whole number on the wire.
  const samplingConstraints = rules.sampling;
  const setSamplingValue = useCallback(
    (key: keyof SamplingValues, raw: number) => {
      const value = clampSamplingValue(samplingConstraints, key, raw);
      if (value === null) return;
      setValues((prev) => (prev.sampling ? { ...prev, sampling: { ...prev.sampling, [key]: value } } : prev));
    },
    [samplingConstraints],
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
        numFrames: snapNumFrames(preset.num_frames, limits.minNumFrames, limits.maxNumFrames, frameGrid),
      }));
      // N1: mirror the preset's own crop_output (Gradio-faithful) — reflect
      // it when present, or turn crop OFF (back to `null`) when the preset
      // doesn't define one, rather than leaving a stale prior crop in place.
      setCropOutput(preset.crop_output ? clampCropOutput(preset.crop_output, newWidth, newHeight) : null);
    },
    [config.generation_presets, active, limits.minNumFrames, limits.maxNumFrames, frameGrid],
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

  // The wav's frame suggestion, by route: `/generate` (the engine publishes
  // `constraints.source_audio`) fits plain time onto the engine's own grid
  // (`./audioFit`); the chain route keeps the LTX audio-latent arithmetic
  // (`chainUtils.suggestFramesForAudio`).
  const suggestFramesForAudioOnRoute = useCallback(
    (durationSec: number): number => {
      if (singleSourceAudio) {
        return suggestFramesForSingleAudio(durationSec, values.frameRate, {
          ...frameGrid,
          min: limits.minNumFrames,
          max: limits.maxNumFrames,
        });
      }
      return snapNumFrames(
        suggestFramesForAudio(durationSec, values.frameRate),
        limits.minNumFrames,
        limits.maxNumFrames,
        frameGrid,
      );
    },
    [singleSourceAudio, values.frameRate, frameGrid, limits.minNumFrames, limits.maxNumFrames],
  );

  useEffect(() => {
    if (sourceAudio.state.status !== "ready") {
      probedAudioIdRef.current = null;
      setAudioDurationSec(null);
      return;
    }
    const id = sourceAudio.state.id;
    if (!id || probedAudioIdRef.current === id) return;
    probedAudioIdRef.current = id;
    // §3-36: read the path straight off the upload slot (same as the
    // reference-video probe above). `useSourceUpload` keeps `filePath` in step
    // with `id` on every transition, so a `ready` id always has its path.
    const filePath = sourceAudio.state.filePath;
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
      let suggested = suggestFramesForAudioOnRoute(probe.durationSec);
      // W8: cap the wav-derived suggestion at the current resolution's comfort
      // ceiling so a long wav never auto-adjusts DURATION past the point
      // generation starts spilling. Applies to EVERY A2V attach path (pick /
      // drag-and-drop / right-click #7), an owner-approved intentional
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
      // No separate ribbon cap (§1-113): the #7 right-click wav is rendered
      // from exactly the ribbon range (`timeline/extractObjectAudio.ts`), so
      // its measured length already bounds this suggestion.
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
  }, [sourceAudio.state.status, sourceAudio.state.id, sourceAudio.state.filePath, nativeBridge, values.width, values.height, spillThresholdFrames, suggestFramesForAudioOnRoute]);

  // A2V-only: gates on whether the attached wav's MEASURED duration is long
  // enough for `values.numFrames` (`chainUtils.audioLengthPrecheck`). `null`
  // outside A2V, or before any wav duration was measured — folded into
  // `isValid` below and surfaced standalone for the "audio too short" warning.
  // On the `/generate` route the need is plain time (`./audioFit`).
  const audioPrecheck = useMemo<AudioLengthPrecheckResult | null>(() => {
    if (!isA2v || audioDurationSec == null) return null;
    if (singleSourceAudio) {
      return {
        ok: singleAudioPrecheck(audioDurationSec, values.numFrames, values.frameRate),
        requiredSeconds: values.numFrames / values.frameRate,
      };
    }
    return audioLengthPrecheck(audioDurationSec, values.numFrames, values.frameRate);
  }, [isA2v, audioDurationSec, values.numFrames, values.frameRate, singleSourceAudio]);

  // Gate submission on the width/height grid too: free-typed values are passed
  // through unsnapped (see `setWidth`/`setHeight`), so an off-grid or
  // out-of-range dimension must block Generate here. In IC-LoRA mode
  // `active.multiple` is 128, so the gate demands the 128 grid.
  const dimensionsOnGrid =
    isDimensionOnGrid(values.width, active.multiple, active.minWidth, active.maxWidth) &&
    isDimensionOnGrid(values.height, active.multiple, active.minHeight, active.maxHeight);

  // Same free-typed-manual-entry gate as width/height, now that `setNumFrames`
  // also accepts `snap=false` (`DurationField`'s number input): a hand-typed
  // off-grid (`frameGrid`) or out-of-range value must block Generate. Uses the SAME
  // min/max source `setNumFrames`'s snap path already uses
  // (`limits.minNumFrames/maxNumFrames`, unaffected by IC-LoRA).
  const numFramesOnGrid = isNumFramesOnGrid(values.numFrames, limits.minNumFrames, limits.maxNumFrames, frameGrid);

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
  // N1: a crop, if enabled, must still be a valid in-range crop of the
  // CURRENT width/height (see `chainUtils.isCropOutputValid`'s doc comment —
  // this is what catches a crop left stale by a later width/height shrink).
  const validityReasons: GenerationValidityReason[] = [];
  if (!isValidPrompt(prompt)) validityReasons.push("promptEmpty");
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
  if (!referenceVideoUnavailable && referenceVideo.state.trimFailed) validityReasons.push("referenceTrimFailed");
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
      // The three sampling values, ALWAYS while
      // the loaded base model publishes `constraints.sampling` (one rule — the
      // server would write its defaults back anyway) and never otherwise, so
      // a request to a base model without them carries none of the three keys.
      ...samplingRequestFields(values.sampling),
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
    singleSourceAudio,
    sourceAudioStrength,
    setSourceAudioStrength,
    audioDurationSec,
    audioFramesAdjustedEvent,
    setWidth,
    setHeight,
    setNumFrames,
    setFrameRate,
    setSeed,
    setSamplingValue,
    samplingConstraints,
    fixedFrameRate: fixedFps,
    frameGrid,
    dimensionStep: generalMultiple,
    applyPreset,
    getSizeFromAviUtl2,
    toGenerateRequest,
    buildA2vRequest,
  };
}
