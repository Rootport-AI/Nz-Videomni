import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { AppConfig, CropOutput, GenerateChainRequest } from "../../api/types";
import { combineLoras } from "../../lora/controlLoras";
import type { ControlLoraSelection } from "../../lora/controlLoras";
import { clampLoraStrength, parseLoraPrompt } from "../../lora/loraTags";
import { isNagNegativeEmpty, NAG_OFF } from "../../shell/nagSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import {
  STAGE2_WINDOW_DEFAULT,
  STAGE2_WINDOW_PRESETS,
  chainWindowBudgetMarkers,
  isChainWindowOverBudget,
  resolveChainComfortBudget,
  stage2MaxContextFrames,
} from "../../shell/tokenBudget";
import type { ChainWindowBudgetMarkers, Stage2Window } from "../../shell/tokenBudget";
import { resolveComfortRow } from "../../shell/comfortTable";
import { MIN_HEIGHT, MIN_WIDTH } from "../single/defaultConfig";
import { estimateGenerationSeconds, formatEstimate } from "../single/estimateUtils";
import {
  ceilToMultiple,
  clamp,
  floorToMultiple,
  isDimensionOnGrid,
  isNumFramesOnGrid,
  isValidPrompt,
  roundToMultiple,
} from "../single/paramUtils";
import { useKeyframes } from "../single/useKeyframes";
import type { UseKeyframesResult } from "../single/useKeyframes";
import {
  ADDED_CLIP_FRAMES_DEFAULT,
  AUDIO_LATENTS_PER_SEC,
  audioLatentsAvailable,
  audioLatentsRequired,
  audioSegmentWindows,
  buildChainRequest,
  chainLayoutError,
  clampCropOutput,
  clampEndSourceStrength,
  clampOverlapFrames,
  clampOverlapStrength,
  computeOutputFrames,
  computeTotalFrames,
  DEFAULT_END_SOURCE_STRENGTH,
  DEFAULT_OVERLAP_FRAMES,
  DEFAULT_OVERLAP_STRENGTH,
  endSourceAudioOverlapOk,
  endSourceLastClipHasFreeLatents,
  isClipCountValid,
  isClipIntact,
  isContextFramesValid,
  isCropOutputValid,
  isTotalFramesValid,
  MAX_CHAIN_TOTAL_FRAMES,
  MAX_CLIP_NUM_FRAMES,
  MAX_CLIPS,
  MAX_REFERENCE_STRENGTH,
  maxChainAudioLatents,
  MIN_CLIP_NUM_FRAMES,
  MIN_CLIPS_NO_SOURCE,
  MIN_CLIPS_WITH_SOURCE,
  minClipsForChain,
  MIN_REFERENCE_STRENGTH,
  pxFromVLatent,
  recommendedClipFrames,
  snapClipNumFrames,
  snapContextFrames,
  vTailLatents,
} from "./chainUtils";
import type { AudioSegmentWindow, ChainClipInput, EndSourceInput } from "./chainUtils";
import { END_SOURCE_CONTEXT_FRAMES, END_SOURCE_MIN_FRAMES, framesAtGenFps } from "../../timeline/tailAlign";
import { AUDIO_FIT_SAFETY_MARGIN_LATENTS, isAudioTooShortForChain, planAudioFit } from "./audioFit";
import type { AudioFitOutcome } from "./audioFit";
import { routeSourceByExtension } from "./sourceRouting";
import { trimQuery } from "../../timeline/sourceTrim";
import type { SourceTrimDecision } from "../../timeline/sourceTrim";
import { useSourceUpload } from "./useSourceUpload";
import type { SourceUploadStatus, UseSourceUploadResult } from "./useSourceUpload";

function errorCodeOf(err: unknown): string {
  if (err instanceof BridgeError) return err.code;
  return "UNKNOWN";
}

export interface ChainCommonValues {
  width: number;
  height: number;
  frameRate: number;
  seed: number;
}

export interface ChainCommonLimits {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
}

export interface UseChainFormDeps {
  nativeBridge?: NativeBridge | undefined;
  /** Control-LoRA names (`lora/controlLoras.ts`'s `resolveControlLoraNames`,
   * computed by `AppShell`) — the dropdown's options for Chain's own
   * control-LoRA selection, and the name set the
   * {@link UseChainFormResult.controlLoraNeedsReference} gate matches the
   * MERGED `loras[]` against (so a HAND-TYPED `<lora:name:strength>` tag whose
   * name is a real control LoRA is caught exactly like a panel selection).
   * Omitted (every pre-existing unit test) => the empty set, i.e. "no name is
   * ever a control LoRA": the dropdown is empty and the gate never fires.
   *
   * §1-15 (2026-08-11): a control LoRA is no longer FORBIDDEN here. Chain now
   * accepts one long reference video for the whole chain, so the old
   * `hasControlLoraTag` hard block is gone and this set feeds Single's exact
   * pair of cross-field gates instead. */
  controlLoraNames?: ReadonlySet<string> | undefined;
  /** §1-15: the control adapters whose frame preprocess is `"depth"`
   * (`lora/controlLoras.ts`'s `resolveDepthLoraNames`, computed by `AppShell`
   * off `GET /loras`). A depth adapter cannot run on a MULTI-CLIP chain — the
   * server 422s it — so {@link UseChainFormResult.validityReasons} pre-empts
   * that with `depthChainUnsupported`. Omitted, or an older server that does
   * not publish `preprocess` at all, => the empty set and NO gate: an unknown
   * preprocess is never guessed at, the server stays the arbiter. */
  depthLoraNames?: ReadonlySet<string> | undefined;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings,
   * owned by `shell/AppShell.tsx` (single `NagSettings` object, D1) — same
   * optional-dep shape as `controlLoraNames` above. Defaults to the frozen
   * `NAG_OFF` sentinel when omitted (every pre-existing unit test), keeping
   * `buildRequest`'s `nagRequestFields` call at `{}`. */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31): the Settings panel's shared
   * attention-backend choice, owned by `shell/AppShell.tsx` (single
   * `AccelerationSettings` object, D1) — same optional shape as `nag`
   * above, defaulting to the frozen `ACCELERATION_DEFAULTS` sentinel when
   * omitted, which keeps `buildRequest`'s `accelerationRequestFields` call
   * at `{}`. */
  acceleration?: AccelerationSettings | undefined;
  /** Smart comfort marker (2026-08-31): whether the server reports
   * SageAttention as installed (`shell/accelerationSettings.sageAvailability`),
   * owned by `shell/AppShell.tsx` — same optional-dep shape as `acceleration`
   * above. `null` ("unknown", the default when omitted) never demotes sage. */
  sageAvailable?: boolean | null | undefined;
  /** Smart comfort marker (2026-08-31): the LOADED base model's engine family
   * (`useBaseModels().activeEngineFamily`), the key into the served
   * `config.limits.comfort_budgets` table. This hook takes the matched row's
   * CHAIN budget for the stage-2 window guide line and its warning; with no
   * matching row it keeps using `config.limits.chain_comfort_token_budget`,
   * i.e. today's value.
   *
   * Omitted, `undefined` or `""` all mean "engine unknown" →
   * `shell/comfortTable.ts`'s compatibility shim, so every pre-existing unit
   * test keeps its current budget. */
  engineFamily?: string | undefined;
}

const EMPTY_CONTROL_LORA_NAMES: ReadonlySet<string> = new Set();

/** §1-15: the width/height rounding multiple while a REFERENCE VIDEO (or a
 * control LoRA, which requires one) is active — the reference conditioning
 * needs both dimensions on the 128 grid, vs. 64 for a plain chain. Mirrors
 * Create's `IC_LORA_MULTIPLE`/`GENERAL_MULTIPLE` pair
 * (`modes/single/useGenerationForm.ts`), which is the same constraint on the
 * same server field. */
const REFERENCE_MULTIPLE = 128;
const GENERAL_MULTIPLE = 64;

/** §1-15: the reference upload's frame cap, sent as `POST /upload/video`'s
 * `max_frames` query. Exactly `MAX_CHAIN_TOTAL_FRAMES` (11544 = 24 x 481) —
 * the most pixel frames any chain can consume, so nothing usable is ever cut,
 * and a multi-hour file dropped in by accident does not become a multi-hour
 * upload. A STRING because the query is a `Record<string, string>` passed
 * straight through to the URL. */
const REFERENCE_MAX_FRAMES_QUERY: Record<string, string> = { max_frames: String(MAX_CHAIN_TOTAL_FRAMES) };

/** 素材（末尾）: the END slot's own `max_frames` upload cap, in SOURCE frames.
 *
 * The chain only ever reads the material's first `END_SOURCE_CONTEXT_FRAMES + 1`
 * frames (9 at the generation rate), so the cap is pure generosity rather than a
 * derived ceiling — it exists to keep a two-hour file out of `uploads/`, not to
 * bound what is read. 685 is kept UNCHANGED from the v2 derivation (137 x 5)
 * deliberately: lowering it would buy nothing measurable and would newly refuse
 * a very-high-frame-rate material whose first 9 generation-rate frames sit
 * further into the file than a tighter cap allows. It is deliberately NOT the
 * reference slot's 11544 — an end source is a tail, not a whole timeline, and
 * `uploads/` is a staging area (`STORAGE_POLICY.md`), not a library. */
const END_SOURCE_MAX_UPLOAD_FRAMES = 685;

/** {@link END_SOURCE_MAX_UPLOAD_FRAMES} as the upload query object. A STRING for
 * the same reason `REFERENCE_MAX_FRAMES_QUERY` is one: the query is a
 * `Record<string, string>` passed straight through to the URL. */
const END_SOURCE_MAX_FRAMES_QUERY: Record<string, string> = { max_frames: String(END_SOURCE_MAX_UPLOAD_FRAMES) };

/** What {@link resolveEndSourceContext} is asked about — one attach, described
 * from the three places the form learns about it. */
interface EndSourceContextInput {
  /** Which half of the end slot holds material, `null` for an empty slot. */
  kind: "image" | "video" | null;
  /** The SERVER's own measurement off the upload response, `null` when it
   * reported none. */
  serverFrames: number | null;
  serverFps: number | null;
  /** The length the ATTACH knew (the right-click route's ribbon window / probed
   * duration), `null` for a picked or dropped file. The fallback only. */
  attachDurationSec: number | null;
  /** The generation frame rate the chain will be produced at. */
  genFps: number;
}

/**
 * 素材（末尾）窓内モード: the anchor to send (`end_source.context_frames`) and,
 * when there is none, WHY.
 *
 * The anchor itself is not derived from anything — it is the constant
 * {@link END_SOURCE_CONTEXT_FRAMES}. All this decides is whether the material
 * can supply it at all, which is a question only a VIDEO can fail:
 *
 * 1. nothing attached → no anchor, no issue (silent, exactly as `contextFrames`
 *    is silent without a source video);
 * 2. an IMAGE → the anchor, no measurement involved — a still is frozen for as
 *    many frames as asked;
 * 3. a VIDEO the server measured (`frames` AND `fps`) → convert to the
 *    generation rate and compare against `END_SOURCE_MIN_FRAMES`. The conversion
 *    is the IDENTITY when the two rates match, and `floor(frames × genFps /
 *    srcFps) − 1` when they do not — the `− 1` being the same one-frame
 *    conservative bias every other length gate on this screen uses, since the
 *    server's own resample rounds in a way the WebUI cannot reproduce exactly;
 * 4. a VIDEO the server did NOT measure, but whose duration the ATTACH knew →
 *    the same comparison off `framesAtGenFps`;
 * 5. anything else → `"unknown"`, i.e. a Generate block. Deliberately not an
 *    assumption: sending an anchor the material cannot supply is a 422 the user
 *    only meets after waiting.
 *
 * Case 3 falling through to case 4 when `fps` alone is missing is intentional —
 * a frame count without its rate is not a length. Pure, and never throws, so it
 * is safe to call on every render while the user is still mid-edit.
 */
function resolveEndSourceContext({
  kind,
  serverFrames,
  serverFps,
  attachDurationSec,
  genFps,
}: EndSourceContextInput): { frames: number | null; issue: "tooShort" | "unknown" | null } {
  if (kind === null) return { frames: null, issue: null };
  if (kind === "image") return { frames: END_SOURCE_CONTEXT_FRAMES, issue: null };

  let effective: number | null = null;
  if (serverFrames !== null && serverFps !== null && genFps > 0) {
    effective = serverFps === genFps ? serverFrames : Math.floor((serverFrames * genFps) / serverFps) - 1;
  } else if (attachDurationSec !== null && attachDurationSec > 0) {
    effective = framesAtGenFps(attachDurationSec, genFps);
  }

  if (effective === null) return { frames: null, issue: "unknown" };
  if (effective < END_SOURCE_MIN_FRAMES) return { frames: null, issue: "tooShort" };
  return { frames: END_SOURCE_CONTEXT_FRAMES, issue: null };
}

/** One-shot initial overrides seeded from a right-click `GenerationPrefill`
 * (`ChainedScreen` derives width/height via `deriveGenerationParams`, IC-LoRA
 * off). Applied only in the `common` lazy `useState` initializer; omitted for
 * the normal tab-opened form, which keeps seeding from
 * `config.generation_defaults`. The GRID the values arrive on is the caller's
 * decision (`timeline/prefillSeed.ts`): §1-15 W4's `reference-video-chain`
 * prefill — the one that does attach a reference video — seeds on §1-15's 128
 * grid so the size does not visibly jump when the upload turns ready and the
 * re-snap fires; every other prefill seeds on the general 64 grid. */
export interface ChainFormInitial {
  width?: number;
  height?: number;
  /** W7: one-shot initial frame rate seeded from a right-click prefill (the
   * `material` policy's selection rate/scale, else omitted to keep the config
   * default). Applied only in the `common` lazy `useState` initializer, like
   * `width`/`height`. */
  frameRate?: number;
  /** W8: one-shot initial DURATION seeded from a right-click prefill's DURATION
   * policy (`deriveDuration.computeTargetNumFrames`) — the comfort ceiling for
   * every Chain prefill (#1 extend-video / #6 image-to-clip-chain / #12
   * current-frame-to-clip-chain). Applied only in the initial `clips` lazy
   * `useState` initializer, and only to CLIP 0 (the chain's opening clip / the
   * long-i2v origin); the from-scratch padding clip keeps the config default.
   * Re-snapped onto the clip 8n+1 grid. Omitted for the normal tab-opened form. */
  numFrames?: number;
  /** X4: force the initial clip list to a SINGLE clip instead of the
   * from-scratch 2-clip minimum. Set only for the #1 right-click `extend-video`
   * prefill (`ChainedScreen`'s `initialCommon`) — that flow attaches a source
   * video (V2V), whose clip floor is 1, but the source is still `idle` at mount
   * (the async upload hasn't started), so the initial state would otherwise pad
   * to 2 and leave a stray second clip card even after the source turns ready
   * (the ready-transition fit only relaxes the floor, it never shrinks). Manual
   * source attachment is unaffected (this stays omitted for the tab-opened
   * form). Applied only in the initial `clips` lazy `useState` initializer. */
  forceSingleClip?: boolean;
}

/** The Chain screen's auto-detected mode (no user-selectable submode any
 * more): `"v2v"` once a source video is attached (V2V continuation),
 * `"scratch"` otherwise (from-scratch clip concatenation). Derived purely from
 * `sourceVideo`'s attach state — see `hasSourceVideo`. */
export type ChainMode = "v2v" | "scratch";

/** W7: the reasons the Chain Generate button is un-submittable, one code per
 * failing gate — the structured form of `isValid` (`isValid ===
 * (validityReasons.length === 0)`). Each code has an imperative instruction in
 * the UI: `promptEmpty`/`dimensionsOffGrid`/`cropInvalid` reuse Create's
 * `strings.single.generateReasons`; `minClips`/`totalFramesExceeded`/
 * `sourceNotReady` reuse Chain's existing field-adjacent
 * banners; `clipFramesOffGrid`/`sourceUploading` are new
 * `strings.chained.generateReasons` lines. R3: the three overlapping source gates
 * (`sourceReady`/`noUploadInFlight`/`sourceNotBusy`) are normalized to a SINGLE
 * source code — `sourceUploading` while an upload is in flight, else
 * `sourceNotReady` — rather than three duplicate lines. */
export type ChainValidityReason =
  | "promptEmpty"
  | "dimensionsOffGrid"
  | "clipFramesOffGrid"
  | "clipTooShortForOverlap"
  | "minClips"
  | "totalFramesExceeded"
  | "sourceUploading"
  | "sourceNotReady"
  /** §1-6 ゲートB: the material that will actually be uploaded (the trimmed
   * ribbon range, or the whole file when no trim applies) is too short to
   * supply `contextFrames` — the server would 422 `SOURCE_VIDEO_TOO_SHORT`. */
  | "sourceVideoTooShortForContext"
  /** §1-6: a range trim was requested for the attached source but the server
   * did not report it applied, so the id points at the WHOLE file. Generating
   * would silently continue from the wrong part of the footage. */
  | "sourceTrimFailed"
  /** §1-14/§3-57: the chosen stage-2 window cannot hold this V2V
   * `contextFrames` — the frozen head would fill the whole first window and
   * leave nothing to generate, which the server 422s
   * (`api/models.py`'s `stage2_max_context_px` cross-validation). Only
   * reachable on `"high_resolution"`, whose ceiling (137) is below the
   * server's published `v2v_context_frames_max` (145). */
  | "contextFramesTooLongForWindow"
  /** §1-16 (long-form A2V): a source VIDEO and a source AUDIO are attached at
   * the same time. `api/models.py` rejects the pair outright (V2V continues
   * existing footage, A2V drives the whole timeline from a track — there is no
   * defined meaning for both), so this blocks before the 422. `buildRequest`
   * additionally drops the audio defensively. */
  | "audioConflictsWithSourceVideo"
  /** §1-16: the audio upload is still in flight. */
  | "audioUploading"
  /** §1-16: the audio upload failed — the slot holds no usable `audio_id`. */
  | "audioNotReady"
  /** §1-16: the attached track VAE-encodes to more audio-latent frames than
   * even a maximal chain (24 x 481) could consume. Normally unreachable — the
   * attach-time hard error (`audioAttachError`) detaches such a file
   * immediately — this is the DEFENSIVE re-check for the case where the frame
   * rate or the seam overlap is changed AFTER a borderline attach. */
  | "audioTooLong"
  /** §1-16 (owner spec 9): the track cannot even cover the shortest two-clip
   * chain that could exist right now, so no amount of auto-fitting helps and
   * the user wants the Single screen instead. */
  | "audioTooShortForChain"
  /** §1-16 (owner spec 7): the current clip list needs more audio-latent frames
   * than the track supplies (the server's `SOURCE_AUDIO_TOO_SHORT` 422). The
   * remedy is the ↔️ auto-fit button, or shorter clips. */
  | "audioTooShort"
  /** §1-16 (judgement point 2): the clip lengths do not tile onto the audio
   * timeline at this frame rate — `chain_math.compute_chain_layout` would raise,
   * which the server surfaces as a 422. Mirrored by `chainUtils.chainLayoutError`
   * and raised ONLY while an audio track is attached: the same constraint exists
   * for a plain chain, but widening the gate there is a separate ledger item. */
  | "chainLayoutInvalid"
  /** §1-15: a reference video is attached but the merged `loras[]` (Chain's own
   * control selection + the prompt's STYLE tags) is empty — the server 422s
   * `reference_video_id` without one. Create's `referenceNeedsLoras` twin. */
  | "referenceNeedsLoras"
  /** §1-15: the merged `loras[]` carries a CONTROL adapter (panel selection OR
   * a hand-typed tag whose name is in `deps.controlLoraNames`) but no reference
   * video is attached — the converse 422. */
  | "controlLoraNeedsReference"
  /** §1-15: the reference-video upload is still in flight. */
  | "referenceUploading"
  /** §1-15: the reference-video upload failed — the slot holds no usable
   * `video_id`. */
  | "referenceNotReady"
  /** §1-15 W4 (2026-08-11): a range trim was requested for the attached
   * REFERENCE video but the server did not report it applied, so the id points
   * at the whole file. The exact reference-slot twin of `sourceTrimFailed`
   * above (and of Create's `referenceTrimFailed`): generating would condition
   * on footage the user never selected, silently. Only ever raisable through
   * the right-click `reference-video-chain` route — the 📁 pick and the
   * drag-and-drop attach send no trim keys, so `useSourceUpload` never arms the
   * flag for them. */
  | "referenceTrimFailed"
  /** §1-15: a V2V source video and a reference video are attached at the same
   * time. `api/models.py` rejects the pair outright (the source continues
   * existing footage, the reference conditions new footage), so this blocks
   * before the 422; `buildRequest` additionally drops the reference
   * defensively — the same belt-and-braces the audio slot gets. */
  | "referenceConflictsWithSourceVideo"
  /** §1-15: with a reference video active, width/height must be multiples of
   * 128 (the reference conditioning's own grid), not the general 64. Raised
   * INSTEAD of `dimensionsOffGrid` in that state so the note names the real
   * constraint. Normally unreachable through the sliders — they snap to 128
   * while the reference is active — this catches free-typed entry and a value
   * left behind by a size change made before the attach. */
  | "referenceDimensionsOffGrid"
  /** §1-15 (owner decision 2026-08-11): a DEPTH-preprocess control adapter on a
   * chain of more than one clip. The depth estimator is a whole-video
   * sequential pass, which a chain-length reference cannot afford, so v1 ships
   * this as an explicit server 422 — pre-empted here. Only ever raised when the
   * server actually published the adapter's `preprocess` (see
   * `UseChainFormDeps.depthLoraNames`). */
  | "depthChainUnsupported"
  // ── 素材（末尾）end source ────────────────────────────────────────────────
  /** The end-source upload is still in flight. */
  | "endSourceUploading"
  /** The end-source upload failed — the slot holds no usable id. */
  | "endSourceNotReady"
  /** A range trim was requested for the end-source VIDEO but the server did not
   * report it applied, so the id points at the whole file — the exact end-slot
   * twin of `sourceTrimFailed`. Only the right-click route can arm it. */
  | "endSourceTrimFailed"
  /** An end source and a source AUDIO are attached at the same time
   * (`api/models.py` rejects the pair). `buildRequest` additionally drops the
   * end source defensively, exactly like the audio/reference slots. */
  | "endSourceConflictsWithAudio"
  /** An end source and a REFERENCE video are attached at the same time — the
   * server's other end-source exclusion (per-segment IC-LoRA injection and the
   * tail freeze both claim the same latents). */
  | "endSourceConflictsWithReference"
  /** 逆順Chained (2026-08-18, second stage): a V2V source video AND an end
   * source on a chain of 2+ clips — mirrors `chain_math.py`'s reverse-mode
   * exclusion (`:1097-1105`, `end_source_mode == "reverse" and
   * source_context_px is not None`). On a SINGLE clip the two combine fine
   * (the start+end interpolation case, unaffected — see
   * `endSourceConflictsWithAudio`'s sibling test); with 2+ clips the chain is
   * generated last-to-first, so clip 0 would be frozen at BOTH ends (its head
   * by the start source, its tail by the reverse carry) — a combination the
   * server has never run and refuses outright. `buildRequest` additionally
   * drops the end source defensively, exactly like the audio/reference slots
   * above. Next increment's scope (Start+End on a middle clip) may lift this;
   * until then it stays a flat block. */
  | "endSourceWithSourceVideoMultiClip"
  /** The end-source VIDEO is shorter than `timeline/tailAlign.ts`'s
   * `END_SOURCE_MIN_FRAMES` (9 frames at the generation rate: the 8-frame anchor
   * plus the causal VAE's keyframe primer), so it cannot supply the anchor at
   * all. Replaces v1's `endSourceTooShortForContext`, which measured against a
   * slider value that no longer exists. */
  | "endSourceTooShort"
  /** The end-source VIDEO's length could not be established at all — the
   * server reported no `frame_count`/`fps` AND the attach carried no measured
   * duration to fall back on. Blocking (rather than assuming the anchor fits) is
   * deliberate: the alternative is sending a `context_frames` the material
   * cannot supply and letting the user discover it as a 422. */
  | "endSourceLengthUnknown"
  /** An end source on a SINGLE-clip chain with `overlap_frames === 1`
   * (窓内モード). The anchor rides on the chain as one more internal segment,
   * and at のりしろ1 that exhausts the audio overlap budget — the server
   * rejects the pair outright (`chain_math`'s `kv >= 2` condition, confirmed by
   * an exhaustive 275,400-case sweep to be exactly and only kv=1). Pre-empted
   * here with the same remedy the server's message gives: raise the seam blend
   * width to 2 or more.
   *
   * 逆順Chained (2026-08-18, second stage, 2+ clips): the SAME server rule
   * exempts `reverse` mode from this `kv >= 2` floor — it appends no internal
   * segment, so the audio budget it spends is the plain chain's own (see
   * {@link endSourceAudioOverlapBudget} below for the check that replaces this
   * one there) — so this gate only ever fires with exactly one clip. */
  | "endSourceNeedsOverlap"
  /** 逆順Chained (2026-08-18, second stage): an end source on 2+ clips whose
   * audio-latent geometry exhausts the per-join crossfade budget before the
   * server's own `overlap_frames (K_v) >= 2` floor would catch it — mirrors
   * `chain_math.py`'s degenerate-audio-overlap rejection
   * (`chainUtils.endSourceAudioOverlapOk`), freed of {@link chainLayoutInvalid}'s
   * `source_audio`-only precondition (an end-source chain never carries one —
   * the two are mutually exclusive server-side). The remedy is the same as
   * `endSourceNeedsOverlap`'s: raise the seam blend width. */
  | "endSourceAudioOverlapBudget"
  | "cropInvalid"
  | "nagNegativeEmpty";

/**
 * §1-6: what the caller knows about the video it is attaching — passed only by
 * the right-click #1 `extend-video` path (`ChainedScreen`), which is the one entry
 * point that has a timeline selection to measure. Every other caller (the
 * unified picker, contract v7's drag-and-drop) omits it and keeps the exact
 * pre-§1-6 behavior: no trim query on the upload, and no length gate.
 */
export interface AttachedSourceInfo {
  /** `timeline/sourceTrim.ts`'s decision for this object. `{trim: true}` adds
   * the `trim_start_sec`/`trim_duration_sec` query to the upload; `{trim:
   * false}` produces no query at all (byte-identical request). */
  trim: SourceTrimDecision;
  /** The backing file's full duration in seconds when native could probe it
   * (`SelectionItem.mediaDurationSec > 0`), else `null`. Used as the gate-B
   * length ONLY when no trim applies — with a trim, `trim.durationSec` is what
   * is actually uploaded. */
  knownDurationSec: number | null;
}

/**
 * §3-102 (LTX 2.5 Chained, first stage): how much of the unified SOURCE slot a
 * caller is allowed to fill.
 *
 * The slot is deliberately ONE control for two different things — clip 0's
 * opening image (from-scratch / I2V) and the V2V source video — so an engine
 * that can chain but cannot continue from a video needs half of it, not none
 * of it. `imagesOnly` closes exactly that half: the native dialog is opened as
 * `kind:"image"` so a video cannot be chosen in the first place, and a video
 * that arrives another way (a typed path, a drop that slipped past the panel's
 * `accept` list) is refused with `sourceError = "VIDEO_SOURCE_UNSUPPORTED"`
 * rather than silently attached.
 *
 * Lives on the FORM rather than in the panel because both attach routes
 * (`pickSource` and `attachSourceByPath`) end in the same private
 * `attachRoutedSource`, which is where the routing decision is actually made —
 * a panel-side check would be a second, hand-maintained copy of it.
 */
export interface SourceAttachOptions {
  /** Refuse videos, accept images. Default `false` (both halves open). */
  imagesOnly?: boolean;
}

export interface UseChainFormResult {
  /** `true` once the user has *attached* a source video — attach intent, not
   * upload completion: `status ∈ {uploading, ready, error}`. Deliberately
   * includes `uploading`/`error` so a mid-upload (or failed) source doesn't
   * flip the screen back to "scratch" and spawn ghost clips; Generate is gated
   * separately while the upload is still in flight or errored (`isValid`). */
  hasSourceVideo: boolean;
  /** `"v2v"` when `hasSourceVideo`, else `"scratch"` — the read-only mode
   * badge `ChainedScreen` renders at the top of the form. */
  mode: ChainMode;

  common: ChainCommonValues;
  /** The width/height bounds currently in force. §1-15: these are the ACTIVE
   * bounds, not the raw config ones — while a reference video (or a control
   * LoRA) is active the maxima are floored onto the 128 grid and the minima
   * raised onto it, exactly like Create's `useGenerationForm.limits`. */
  limits: ChainCommonLimits;
  /** §1-15: the width/height rounding multiple currently in force — 64
   * normally, 128 while a reference video / control LoRA is active (the
   * reference conditioning's own grid). Exposed so the screen's size fields can
   * step by the same amount the setters snap to. */
  sizeMultiple: number;
  /** Sets width. `snap` (default true) rounds `value` onto the ACTIVE grid (64,
   * or 128 while a reference video is active — {@link sizeMultiple}) and clamps
   * to `limits` — used by the stepper arrows, ↑↓ keys, slider, and
   * presets/getSize flows. `snap === false` (free keyboard typing) passes the
   * value straight through with only a NaN guard (Gradio-faithful: no
   * rounding, no clamping); the grid gate lives in `isValid` so an off-grid
   * manual entry blocks submit. Mirrors Create's `useGenerationForm.setWidth`. */
  setWidth: (value: number, snap?: boolean) => void;
  /** Sets height. See `setWidth` for the `snap` semantics. */
  setHeight: (value: number, snap?: boolean) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
  /** N1: opt-in output crop (`GenerateChainRequest.crop_output`). `null` =
   * "not set" (omitted from the request). Mirrors Create's
   * `useGenerationForm.cropOutput`; see
   * `chainUtils.clampCropOutput`/`isCropOutputValid` for the 32-pixel-grid
   * constraint (LTX's VAE space-compression factor) folded into `isValid`
   * below. */
  cropOutput: CropOutput | null;
  /** Sets `cropOutput`, snapping onto the 32-pixel grid and clamping to
   * `[32, common.width/common.height]` — never the config max — via
   * `chainUtils.clampCropOutput`. `null` clears it. */
  setCropOutput: (value: CropOutput | null) => void;
  gettingSize: boolean;
  getSizeError: string | null;
  getSizeFromAviUtl2: () => Promise<void>;
  /** Resolves `name` against `config.generation_presets` (Sprint 2 item 9,
   * mirrors Create's `useGenerationForm.applyPreset`): sets the common
   * width/height, and — since Chain has no single `num_frames` field —
   * uniformly sets EVERY clip's `numFrames` to
   * `chainUtils.recommendedClipFrames`'s recommendation (the resolution's
   * `limits.spill_free_frames` comfortable cap when published, else the
   * preset's own `num_frames`), mirroring `gradio_ui/presets.py`'s
   * `apply_chain_preset`. An unknown `name` is a no-op. Doesn't itself warn
   * about `MAX_CHAIN_TOTAL_FRAMES` overshoot for a large clip count — the
   * existing `isTotalFramesValid`/`totalFramesExceeded` banner reacts to the
   * resulting `clips` the same as any other edit. */
  applyPreset: (name: string) => void;

  clips: ChainClipInput[];
  minClips: number;
  maxClips: number;
  canAddClip: boolean;
  addClip: () => void;
  removeClip: (id: string) => void;
  setClipPrompt: (id: string, value: string) => void;
  /** Sets one clip's `numFrames`. `snap` (default true) rounds `value` onto
   * the 8n+1 grid (clamped to `[MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES]`) —
   * used by the stepper arrows, ↑↓ keys, and slider. `snap === false` (free
   * keyboard typing) passes the value straight through with only a NaN
   * guard; the grid gate lives in `isValid` so an off-grid manual entry
   * blocks submit. Mirrors Create's `useGenerationForm.setNumFrames`. */
  setClipNumFrames: (id: string, value: number, snap?: boolean) => void;
  /** The single "start frame" for clip 0 (Docs/API_REFERENCE.md §5.2: "clip 0
   * のみ画像を持てる"). Reuses `useKeyframes` capped to a single image
   * (`maxItemsOverride: 1`) — only meaningful, and only rendered, in
   * `"scratch"` mode; a V2V source video supplies clip 0's starting frames
   * instead, so it's never injected into a V2V request (see `buildRequest`). */
  startFrame: UseKeyframesResult;

  overlapFrames: number;
  setOverlapFrames: (value: number) => void;
  overlapStrength: number;
  setOverlapStrength: (value: number) => void;
  /** 素材（末尾）の錨の固定強度 (`GenerateChainRequest.end_source.strength`,
   * バッチ1・2026-08-18). Defaults to 1.0 (hard freeze, byte-identical to
   * before this field existed). Unlike {@link overlapStrength} (the seam
   * blend BETWEEN generated clips), this is the freeze strength against the
   * MATERIAL itself — Stage-2 always re-hard-freezes regardless of the value,
   * so the final frame is always exactly the material; only Stage-1's
   * approach to it softens. */
  endSourceStrength: number;
  setEndSourceStrength: (value: number) => void;
  /** X3: the minimum `num_frames` every clip must have for the current
   * `overlapFrames` (= `8 * overlapFrames + 1`), the exact threshold behind the
   * `clipTooShortForOverlap` validity reason. Exposed so `ChainedScreen` can
   * interpolate it into the Generate-disabled note (`n`). */
  minFramesForOverlap: number;

  /** Opt into temporal-chunk upsampling (`GenerateChainRequest.chunked_upsample`,
   * task brief §1). Defaults to `true` — the chunked path trades time for a
   * flat VRAM ceiling, which is the right default for chains that can run
   * long/high-res; the server itself defaults to `false` when the field is
   * omitted, so this is always sent explicitly by `buildRequest`. */
  chunkedUpsample: boolean;
  setChunkedUpsample: (value: boolean) => void;

  /** Stage-2 window preset (`GenerateChainRequest.stage2_window`, owner decision
   * 2026-08-09). Defaults to `"standard"` — the server default and the frozen
   * tile geometry — and `buildRequest` omits the field entirely at that value,
   * so a chain that never touches this control sends a request byte-identical to
   * before the field existed. `"high_resolution"` trades more seams for a
   * cheaper stage-2 window, which is what keeps high resolutions inside
   * {@link chainWindowOverBudget}'s budget. */
  stage2Window: Stage2Window;
  setStage2Window: (value: Stage2Window) => void;
  /** `true` when the CURRENT width/height on the CURRENT stage-2 window exceeds
   * the comfortable per-window attention-token budget (§1-14) — drives the
   * Chain screen's resolution warning. Advisory only: it never blocks Generate,
   * because the budget is a "this will be slow and memory-hungry" line, not a
   * hard limit. */
  chainWindowOverBudget: boolean;
  /** Where the comfortable budget falls on the width/height sliders, for the
   * CURRENT stage-2 window and the CURRENT values (2026-08-12). Fed straight to
   * `CommonGenerationFields.SizeFields`' `budgetMarkers`. Purely derived — never
   * stored — so it follows a window change, a served-budget change and a
   * reference video's 128 grid on its own. */
  chainWindowMarkers: ChainWindowBudgetMarkers;
  /** The largest `contextFrames` the CURRENT stage-2 window can hold
   * (`shell/tokenBudget.ts`'s `stage2MaxContextFrames`) — 161 on `"standard"`
   * (never binding, the server caps context at 145) and 137 on
   * `"high_resolution"`. Exposed so the screen can name the number in its
   * message rather than hard-coding it. */
  stage2MaxContextFrames: number;

  sourceVideo: UseSourceUploadResult;
  contextFrames: number;
  setContextFrames: (value: number) => void;
  contextFramesLimits: { min: number; max: number };
  isContextFramesValid: boolean;

  /** `"image"` when the from-scratch start frame is set, `"video"` when a
   * V2V source video is attached (`hasSourceVideo`), else `null` when
   * neither is — the two are mutually exclusive by construction (`pickSource`/
   * `clearSource` below always clear the other slot). Drives
   * `SourceInputPanel`'s single unified thumbnail/controls/clear surface. */
  activeSourceKind: "image" | "video" | null;
  /** `true` for the whole span of a `pickSource()` call — the native Open
   * dialog plus (if a file was chosen) the subsequent upload — so the
   * unified picker button can show "Uploading…" and disable itself the same
   * way `useKeyframes.isPicking`/`useSourceUpload`'s own `uploading` status
   * do individually. */
  isPickingSource: boolean;
  /** Non-null after `pickSource` either fails to open the dialog for a reason
   * other than `CANCELLED`, or resolves to a file whose extension
   * `sourceRouting.routeSourceByExtension` doesn't recognise — surfaced by
   * `SourceInputPanel` as a warning. §3-102 adds one more value,
   * `VIDEO_SOURCE_UNSUPPORTED` (see {@link SourceAttachOptions}). Reset to
   * `null` on the next successful `pickSource` and by `clearSource`. */
  sourceError: string | null;
  /** Opens the unified "choose image or video" dialog
   * (`ui.pickFile({kind:"imageOrVideo"})`) once, then routes the result by
   * extension: an image goes to `startFrame.addFromPath` (replacing any
   * existing start frame) and detaches any source video (staying in/entering
   * scratch mode); a video goes to `sourceVideo.uploadPath` (replacing any
   * existing source video) and drops any start frame (entering V2V mode). A
   * `CANCELLED` dialog is silently ignored, mirroring every other picker in
   * this codebase; any other dialog failure, or an unrecognised extension,
   * sets `sourceError` instead of touching either slot.
   *
   * §3-102: with `{ imagesOnly: true }` the dialog is opened as
   * `kind:"image"` instead, so the VIDEO half of this one slot closes while
   * clip 0's opening image keeps working. */
  pickSource: (options?: SourceAttachOptions) => Promise<void>;
  /** Contract v7 (drag-and-drop): attaches an already-resolved local file
   * (image or video), sharing `pickSource`'s exact routing-by-extension,
   * single-slot exclusivity and `sourceError` behavior — the only difference
   * is skipping `ui.pickFile`'s native Open dialog, since the file is already
   * known. An unrecognised extension sets `sourceError` exactly like
   * `pickSource` does. `options` behaves exactly as it does on
   * {@link pickSource}. */
  attachSourceByPath: (
    filePath: string,
    fileName: string,
    info?: AttachedSourceInfo,
    options?: SourceAttachOptions,
  ) => Promise<void>;
  /** Detaches whichever of `startFrame`/`sourceVideo` is currently active (the
   * common "Clear" button for the unified source-input slot) and clears
   * `sourceError`. A no-op on either slot that's already empty. */
  clearSource: () => void;

  /** §1-6: the real-time length (seconds) of the material that will actually be
   * sent as `source_video` — `trim.durationSec` when the attach carried a live
   * trim decision, the full file's probed duration when it carried one but no
   * trim applies, and `null` whenever there is nothing to measure (a manually
   * picked / dropped file, an unprobeable object, an image slot, or no source at
   * all). `null` disables the {@link ChainValidityReason} `sourceVideoTooShort
   * ForContext` gate entirely — an unmeasured source is never blocked, the
   * server stays the final arbiter. Exposed mainly so the gate is directly
   * testable. */
  sourceDurationSec: number | null;

  /** Owner request 2026-08-10: the attached material's pixel size — probed
   * (`fs.probeMediaInfo`) once per ready attachment, for BOTH source modes (the
   * V2V source video and the from-scratch start image). `null` whenever it
   * isn't known: nothing attached, the attachment still uploading, the probe
   * rejected, or the file reports a `0` width/height. `SourceInputPanel`
   * appends it to the file-name line and omits it entirely when `null`. */
  sourceMediaSize: { width: number; height: number } | null;

  // ── §1-16 長尺A2V: the one audio track that drives the whole chain ─────────

  /** The audio slot itself (`POST /upload/audio` -> `audio_id`). Rendered by
   * `ChainAudioPanel`; `buildRequest` sends `state.id` as `source_audio`. */
  sourceAudio: UseSourceUploadResult;
  /** `true` once the user has ATTACHED an audio track — attach intent, not
   * upload completion (`status !== "idle"`), exactly like {@link hasSourceVideo}.
   * Uploading/failed count as attached so the panel keeps showing the file. */
  hasSourceAudio: boolean;
  /** Opens the native audio picker and attaches whatever was chosen. A
   * `CANCELLED` dialog is silently ignored; any other failure sets
   * {@link audioError}. */
  pickAudio: () => Promise<void>;
  /** Attaches an already-resolved local audio file (drag-and-drop, and the
   * right-click "long a2v" intents). `opts.knownDurationSec` is the EXACT length
   * the caller already measured — `timeline.extractAudio` returns one for the
   * wav it just cut — and takes priority over every probe: it is adopted as-is
   * and no `fs.probe*` call is made at all. */
  attachAudioByPath: (
    filePath: string,
    fileName: string,
    opts?: { knownDurationSec?: number },
  ) => Promise<void>;
  /** Detaches the audio track and resets everything derived from it (measured
   * duration, probe-failure flag, attach error, the once-per-track auto-fit
   * claim). Deliberately does NOT restore the clip list the auto-fit changed —
   * those lengths are now the user's chain. */
  clearAudio: () => void;
  /** Measured length of the attached track in seconds, or `null` when nothing is
   * attached / nothing could be measured (see {@link audioProbeFailed}). Three
   * sources, in priority order: `attachAudioByPath`'s `knownDurationSec`, then
   * `fs.probeAudioDuration` (exact, wav only), then `fs.probeMediaInfo`
   * (native `get_media_info`, covers mp3/m4a/…). */
  audioDurationSec: number | null;
  /** `true` when an attached track's length could NOT be measured by any of the
   * three routes. Everything length-derived is then disabled — the auto-fit
   * button, the `audioTooShort*`/`audioTooLong` gates — and the server's own
   * preflight becomes the sole authority. */
  audioProbeFailed: boolean;
  /** Non-null when the LAST attach was rejected outright and the slot was
   * detached again: `"tooLong"` = the track is longer than any chain (24 x 481)
   * could ever consume (owner spec 8). Reset by the next attach and by
   * {@link clearAudio}. */
  audioAttachError: "tooLong" | null;
  /** Non-null after {@link pickAudio} fails to open the dialog for a reason
   * other than `CANCELLED` — the `sourceError` analogue for the audio slot. */
  audioError: string | null;
  /** The "追加クリップの長さ" slider (8n+1). Seeded at
   * `chainUtils.ADDED_CLIP_FRAMES_DEFAULT` (257) and re-seeded to a preset's own
   * recommendation by {@link applyPreset}. It is the length
   * {@link adjustClipsForAudio} gives every card it may resize. */
  addedClipFrames: number;
  /** Sets {@link addedClipFrames}. `snap` (default true) rounds onto the 8n+1
   * grid within `[MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES]`, mirroring
   * `setClipNumFrames`. */
  setAddedClipFrames: (value: number, snap?: boolean) => void;
  /** "↔️ 再生時間の自動調整": re-lays the clip list over the measured audio via
   * `audioFit.planAudioFit`. A no-op without a measured duration. Cards the user
   * has touched (`intact: false`) keep their length, position and existence;
   * everything else is resized/added/dropped to cover the track. Always publishes
   * an {@link audioFitEvent}, even when nothing changed. */
  adjustClipsForAudio: () => void;
  /** The last {@link adjustClipsForAudio} result, for the screen's toast.
   * `id` is a monotonic counter so a consumer can de-duplicate (StrictMode fires
   * effects twice); `outcome` is `audioFit.ts`'s `AudioFitOutcome`. */
  audioFitEvent: { id: number; outcome: AudioFitOutcome; durationSec: number } | null;
  /** Seconds of the track no clip covers with the CURRENT clip list, or `null`
   * when that cannot be computed (nothing attached, unmeasured, or an impossible
   * geometry). Never a gate — just the panel's "末尾未使用" note. */
  audioSurplusSec: number | null;
  /** `true` when the chain is already at the 24-clip ceiling AND audio is still
   * left over — the "these are all the clips there can be" note. Not a gate: the
   * chain is perfectly submittable, it just won't use the whole track. */
  audioAtMaxClips: boolean;
  /** Per-clip window on the uploaded track (`chainUtils.audioSegmentWindows`),
   * for each clip card's "🎵 担当時間帯" badge — same order as {@link clips}.
   * `null` when no audio is attached/measured, or the geometry is impossible. */
  audioSegmentWindowsForClips: AudioSegmentWindow[] | null;

  // ── §1-15 参照動画: ONE long reference video for the whole chain ────────────
  // Same slot shape as the audio block above (an independent `useSourceUpload`,
  // never routed through the unified image/video picker): the server slices this
  // one video by frame number and injects each clip's own window into that
  // clip's stage-1 pass. Mutual exclusivity with the V2V source video is a
  // Generate gate (`referenceConflictsWithSourceVideo`), not a silent detach.

  /** The reference-video slot itself (`POST /upload/video` -> `video_id`).
   * Rendered by `ChainReferencePanel`; `buildRequest` sends `state.id` as
   * `reference_video_id`. */
  referenceVideo: UseSourceUploadResult;
  /** `true` once the user has ATTACHED a reference video — attach intent, not
   * upload completion (`status !== "idle"`), exactly like {@link hasSourceAudio}.
   * Uploading/failed count as attached so the panel keeps showing the file. */
  hasReferenceVideo: boolean;
  /** Opens the native video picker and attaches whatever was chosen. A
   * `CANCELLED` dialog is silently ignored; any other failure sets
   * {@link referenceError}. */
  pickReference: () => Promise<void>;
  /** Attaches an already-resolved local video file (drag-and-drop, and any
   * right-click intent that routes here). Uploads with the `max_frames` cap —
   * see the hook body — so a reference longer than a maximal chain could ever
   * consume is trimmed server-side at upload time instead of being stored whole.
   *
   * §1-15 W4 (2026-08-11): `trim` is the optional ribbon-range decision, passed
   * ONLY by the right-click `reference-video-chain` route (the one entry point
   * with a timeline selection to measure) — the same role `AttachedSourceInfo`
   * plays for {@link attachSourceByPath}, minus its length field, since the
   * reference slot has no `context_frames` gate to feed. Omitting it keeps the
   * request byte-identical to every pre-W4 attach: just the `max_frames` cap. */
  attachReferenceByPath: (filePath: string, fileName: string, trim?: SourceTrimDecision) => Promise<void>;
  /** Detaches the reference video and clears {@link referenceError}. The two
   * strengths and the control-LoRA selection are deliberately NOT reset — they
   * are settings, not material, and re-attaching a new file keeps them. */
  clearReference: () => void;
  /** Non-null after {@link pickReference} fails to open the dialog for a reason
   * other than `CANCELLED` — the `audioError` analogue for this slot. */
  referenceError: string | null;
  /** The ready reference video's pixel size (`fs.probeMediaInfo`), or `null`
   * whenever it isn't known: nothing attached, still uploading, the probe
   * rejected, or a `0` dimension. Same recipe (and same "once per attachment"
   * guard) as {@link sourceMediaSize}. Duration is deliberately NOT exposed:
   * the slicing is by FRAME NUMBER, so seconds would only mislead. */
  referenceMediaSize: { width: number; height: number } | null;
  /** 0.0-1.0 or `null` ("not set" — the field is omitted from the request).
   * Only ever sent alongside a ready reference video AND a non-empty merged
   * `loras[]`. Same default (`null`) and same clamp as Create's. */
  conditioningAttentionStrength: number | null;
  setConditioningAttentionStrength: (value: number | null) => void;
  /** 0.0-1.0 or `null` ("not set"). See {@link conditioningAttentionStrength}. */
  referenceVideoStrength: number | null;
  setReferenceVideoStrength: (value: number | null) => void;
  /** Chain's OWN control-LoRA selection (`null` = "none"). Deliberately NOT
   * `AppShell`'s single `controlLora` state, which stays Create-only: the two
   * screens hold independent material (Create's reference video vs. this
   * chain-length one), and sharing the selection would make choosing an adapter
   * on one screen silently arm the other screen's gates. */
  controlLora: ControlLoraSelection | null;
  setControlLora: (value: ControlLoraSelection | null) => void;
  /** Sets {@link controlLora}'s strength only, clamped to
   * `[LORA_STRENGTH_MIN, LORA_STRENGTH_MAX]` (`lora/loraTags.ts`). A no-op
   * while nothing is selected. */
  setControlLoraStrength: (value: number) => void;
  /** Selectable control-LoRA names — pass-through of
   * `UseChainFormDeps.controlLoraNames`, so the panel builds its dropdown off
   * the form rather than needing its own wiring (mirrors Create). */
  controlLoraNames: ReadonlySet<string>;
  /** `true` once a reference video is attached but the merged `loras[]` is
   * still empty — the `referenceNeedsLoras` gate, surfaced standalone so the
   * panel can render the specific warning. */
  referenceNeedsLoras: boolean;
  /** `true` once the merged `loras[]` carries a control adapter but no
   * reference video is attached — the `controlLoraNeedsReference` gate. */
  controlLoraNeedsReference: boolean;
  /** `true` while a reference video is active, i.e. width/height are on the 128
   * grid and the reference fields will ride on the request. Attaching the video
   * OR selecting a control LoRA (which requires one) is enough, matching
   * Create's `isICLora`. */
  isReferenceActive: boolean;

  // ── 素材（末尾）: ONE image or video the whole chain must END with ─────────
  // Structured exactly like the unified START slot above (`activeSourceKind`/
  // `pickSource`/`attachSourceByPath`/`clearSource`), because it is the same
  // "one file, routed by extension into one of two sub-slots" problem: an
  // IMAGE goes to `endSourceImage` (a `useKeyframes` capped at one, for its
  // `image_id` + thumbnail) and a VIDEO to `endSourceVideo` (a `useSourceUpload`,
  // for its `video_id`). Re-attaching REPLACES — the two sub-slots are never
  // both filled.

  /** The end slot's VIDEO half (`POST /upload/video` -> `video_id`). Read by
   * `ChainEndSourcePanel` for the file name/status; `buildRequest` sends
   * `state.id` as `end_source.video_id`. */
  endSourceVideo: UseSourceUploadResult;
  /** The end slot's IMAGE half (`POST /upload/image` -> `image_id`), a
   * `useKeyframes` capped at a single item purely for its upload+thumbnail
   * machinery — its `frameIdx`/`strength` are never read: the end material is a
   * hard freeze, not a conditioning image. */
  endSourceImage: UseKeyframesResult;
  /** `"image"`/`"video"` for whichever sub-slot holds material, else `null`. */
  endSourceKind: "image" | "video" | null;
  /** The two sub-slots' statuses unified into one, so consumers never have to
   * know which half is in play: `"idle"` when nothing is attached, otherwise the
   * active half's own `uploading`/`ready`/`error`. */
  endSourceStatus: SourceUploadStatus;
  /** `true` once the user has ATTACHED an end source — attach intent, not
   * upload completion (`endSourceStatus !== "idle"`), exactly like
   * {@link hasSourceAudio}. */
  hasEndSource: boolean;
  /** The measured length (seconds) of the end-source VIDEO that will actually be
   * uploaded — same three cases as {@link sourceDurationSec}, and `null`
   * (gate off) for an image, a manually picked/dropped file, or nothing
   * attached. Only the FALLBACK input for {@link endSourceLengthIssue} (used
   * when the server reported no frame count); the server's own measurement
   * leads. */
  endSourceDurationSec: number | null;
  /** Non-null after {@link pickEndSource} fails to open the dialog for a reason
   * other than `CANCELLED`, or after an unrecognised extension is attached
   * (`"UNSUPPORTED_FILE_TYPE"`) — the `sourceError` analogue for this slot. */
  endSourceError: string | null;
  /** 素材（末尾）窓内モード: how many frames at the END of the generated video
   * are frozen onto the material's head (`end_source.context_frames`) — the
   * ANCHOR. A CONSTANT, not a setting and not a derivation: always
   * `END_SOURCE_CONTEXT_FRAMES` (8) whenever the slot holds usable material,
   * whether that material is a still or a film.
   *
   * `null` means "no anchor can be sent": nothing attached, the VIDEO is below
   * `tailAlign.ts`'s `END_SOURCE_MIN_FRAMES` (9), or nothing about its length
   * could be measured at all. The last two are Generate blocks
   * (`endSourceTooShort` / `endSourceLengthUnknown`); see
   * {@link endSourceLengthIssue} for which. */
  endContextFrames: number | null;
  /** WHY {@link endContextFrames} is `null` while material IS attached —
   * `"tooShort"` (below the 9-frame floor) or `"unknown"` (nothing measurable).
   * `null` when the material is usable, or when nothing is attached at all.
   * Exposed so `ChainEndSourcePanel` can put the right sentence in its single
   * priority note without re-deriving the decision. */
  endSourceLengthIssue: "tooShort" | "unknown" | null;
  /** 窓内モード品質警告: the clip-length ceiling above which the frozen tail
   * visibly degrades its surroundings (smearing/flicker), or `null` when there
   * is nothing to warn about — no end source attached, or every clip already
   * inside the ceiling.
   *
   * The ceiling is ONE stage-2 window expressed in pixel frames
   * (`pxFromVLatent(STAGE2_WINDOW_PRESETS[stage2Window].vTile)` — 169 on
   * `standard`, 145 on `high_resolution`), because a clip that outgrows a single
   * window puts the anchor and the frames that have to blend into it in
   * DIFFERENT stage-2 tiles. Advisory only: never a `validityReason`. */
  endSourceQualityLimitFrames: number | null;
  /** §1-22 素材（末尾）複数クリップ品質警告: `true` while material is attached
   * AND the chain has 2+ clips (逆順Chained). Mutually exclusive with
   * {@link endSourceQualityLimitFrames} being non-null by construction — that
   * ceiling only ever fires at `clips.length === 1`, this one only at
   * `clips.length >= 2` — so the two never compete for the same banner slot.
   * Advisory only: never a `validityReason`. */
  endSourceMultiClipQualityWarning: boolean;
  /** Opens the unified "choose image or video" dialog and routes the result by
   * extension into this slot. Mirror of {@link pickSource}, including the
   * silent `CANCELLED` and the `endSourceError` failure surface. */
  pickEndSource: () => Promise<void>;
  /** Attaches an already-resolved local file (drag-and-drop, and the right-click
   * "end with this" intent), sharing {@link pickEndSource}'s exact routing and
   * replace policy. `info` — passed only by the right-click route, the one entry
   * point with a timeline selection to measure — adds the upload's trim query
   * and feeds the length gate, exactly as it does for
   * {@link attachSourceByPath}. */
  attachEndSourceByPath: (filePath: string, fileName: string, info?: AttachedSourceInfo) => Promise<void>;
  /** Detaches whichever half holds material and clears {@link endSourceError}.
   * A no-op on an already-empty slot. */
  clearEndSource: () => void;

  totalFrames: number;
  isTotalFramesValid: boolean;
  isClipCountValid: boolean;
  estimateSeconds: number;
  estimateLabel: string;

  /** Predicted length of the DELIVERED file, in pixel frames/seconds:
   * `computeOutputFrames(...)` — seam-blend fusion, and in V2V the source's
   * leading-frame replacement. `outputSeconds` divides by `common.frameRate`.
   *
   * 素材（末尾）窓内モード adds NOTHING here: the anchor is frozen INSIDE the
   * last clip's own window, so attaching end material leaves the delivered
   * length exactly as the clip settings describe it. */
  outputFrames: number;
  outputSeconds: number;
  /** 素材（末尾）窓内モード: how many frames at the END of {@link outputFrames}
   * are the frozen anchor rather than freely generated content — 8 with usable
   * end material attached, `0` otherwise. NOT an addition to the output length
   * (see {@link outputFrames}); it is a slice OF it, which is why the two
   * consumers both SUBTRACT it: 系統E's tail-aligned placement
   * (`ChainedScreen.handleGenerate`) and the 内訳 line. */
  outputTailFrames: number;
  /** {@link outputTailFrames} in seconds, at `common.frameRate`. */
  outputTailSeconds: number;

  /** Aggregate validity gating the Generate button — combines prompt
   * validity (owned by the shared `PromptBar`), clip-count/total-frames
   * bounds, V2V source readiness, and "no upload still in flight" for every
   * upload surface (clip-0 start frame, source video). Also unconditionally
   * `false` while the source video is `uploading`/`error`, so a half-attached
   * or failed source can never look submittable. §1-15 adds the reference-video
   * cross-field gates (see {@link ChainValidityReason}). */
  isValid: boolean;
  /** W7: the structured breakdown behind `isValid` — one code per failing gate,
   * empty exactly when `isValid` is true. Consumed by `ChainedScreen`'s
   * Generate-reasons note; see {@link ChainValidityReason}. */
  validityReasons: ChainValidityReason[];

  /** I8 §3-4 (※): whether the form has diverged from its pristine, freshly-
   * mounted default state — read by `AppShell` (via `publishChainDirty`) before
   * a #1/#6 Chain remount to decide whether to warn the user that loading would
   * discard in-progress chain work. Practical precision over exhaustive coverage
   * (per the work order): it compares the fields a user would actually mind
   * losing — an attached source, an attached audio track (§1-16),
   * clip-list edits (count / prompt / duration),
   * the common width/height/frame_rate/seed, the seam-blend overlap params, the
   * §1-15 reference video (attached material, and a control-LoRA selection that
   * costs the user a choice to redo), and
   * the output crop — against the config-derived defaults a pristine form seeds
   * with. Deliberately NOT considered: the shared prompt (owned by `PromptBar`,
   * survives a remount), `contextFrames` (only meaningful once a source is
   * attached, which already flags dirty), and `chunkedUpsample` (a single toggle
   * not worth a false "unsaved work" prompt). */
  isDirty: boolean;

  buildRequest: () => GenerateChainRequest;
}

/** Value equality for a `CropOutput | null` (used by `isDirty` to compare the
 * current crop against its config-default seed). */
function cropOutputEquals(a: CropOutput | null, b: CropOutput | null): boolean {
  if (a === null || b === null) return a === b;
  return a.width === b.width && a.height === b.height;
}

let clipIdCounter = 0;
function nextClipId(): string {
  clipIdCounter += 1;
  return `clip-${clipIdCounter}`;
}

/** §1-16: every card is BORN intact — the flag only ever goes to `false`, and
 * only when the user themselves edits this card's prompt or duration (see
 * `ChainClipInput.intact`). That covers manual `addClip`, `fitClipsToBounds`'s
 * padding, and the cards `adjustClipsForAudio` creates. */
function makeClip(numFrames: number): ChainClipInput {
  return { id: nextClipId(), prompt: "", numFrames, intact: true };
}

/** Truncates/pads `clips` so its length falls within `[min, max]`, always
 * keeping clip 0 (and its start-frame state, owned separately by
 * `useKeyframes`) first. Used on the two discrete source-attach transitions:
 * padding back up to the from-scratch 2-clip minimum when the source is
 * cleared, and allowing a single clip the moment one is attached. */
function fitClipsToBounds(clips: ChainClipInput[], min: number, max: number, defaultNumFrames: number): ChainClipInput[] {
  let next = clips;
  if (next.length > max) next = next.slice(0, max);
  while (next.length < min) next = [...next, makeClip(defaultNumFrames)];
  return next;
}

/**
 * Owns Chain mode's entire form state: the clip list (per-clip prompt override
 * + duration, clip-0-only start frame), the seam-blend params, the optional
 * V2V source video + context_frames, and the common
 * width/height/frame_rate/seed fields shared with Create
 * (Docs/API_REFERENCE.md §5.2 "common" fields — same frozen constraints as
 * `GenerateRequest`).
 *
 * There is no user-selectable submode: the mode is auto-detected from whether
 * a source video is attached (`hasSourceVideo` → `mode`). Attaching a source
 * (V2V) relaxes the clip floor to 1 and bumps clip 0 long enough to satisfy
 * `context_frames < clips[0].num_frames`; clearing it pads back up to the
 * from-scratch 2-clip minimum. Both happen on discrete `sourceVideo` status
 * transitions (see the re-fit effect below), never continuously, so a
 * mid-upload source can't make the clip list wobble.
 *
 * Mirrors `modes/single/useGenerationForm.ts`'s shape/conventions
 * deliberately (snapped setters, `config`-derived limits, a `prompt` argument
 * owned by the shared `PromptBar`) so the two are easy to read side by side;
 * `getSizeFromAviUtl2` in particular is a near-duplicate of the Create
 * version rather than a shared extraction — the two hooks' state shapes
 * differ enough (single `numFrames` vs. a clip list) that factoring just
 * this one bridge call out wasn't judged worth the indirection. Width/height/
 * frame-rate/seed *rendering* is shared via
 * `modes/single/CommonGenerationFields.tsx` (`SizeFields`/
 * `FrameRateSeedFields`), used by both this hook's caller and Create's
 * `GenerationForm`.
 */
export function useChainForm(
  config: AppConfig,
  prompt: string,
  deps: UseChainFormDeps = {},
  initial: ChainFormInitial = {},
): UseChainFormResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const controlLoraNames = deps.controlLoraNames ?? EMPTY_CONTROL_LORA_NAMES;
  // §1-15: same optional-dep shape — omitted means "no adapter is known to be
  // depth", i.e. the `depthChainUnsupported` gate simply never fires.
  const depthLoraNames = deps.depthLoraNames ?? EMPTY_CONTROL_LORA_NAMES;
  // NAG (2026-07-28): mirrors `controlLoraNames`' optional-dep pattern — the
  // caller (`AppShell`) owns the actual `NagSettings` state.
  const nag = deps.nag ?? NAG_OFF;
  // Acceleration (2026-07-31): same optional-dep shape as `nag` above.
  const acceleration = deps.acceleration ?? ACCELERATION_DEFAULTS;
  // Smart comfort marker (2026-08-31): the two inputs that, with `acceleration`
  // above, pick the served comfort row whose CHAIN budget draws the stage-2
  // window guide. Both default to "unknown", which yields the compatibility
  // shim — i.e. the pre-table budget.
  const sageAvailable = deps.sageAvailable ?? null;
  const engineFamily = deps.engineFamily;

  // ── §1-15 参照動画: declared FIRST because the width/height grid depends on
  // it. A reference video (or a control LoRA, which mandates one) puts the
  // whole form on the 128 grid — the same dominance Create's `icLoraActive`
  // expresses — so `active` below, and therefore every size setter, has to be
  // able to read this slot's state. Everything else about the slot (pick/
  // attach/clear/probe/gates) lives further down with the audio slot.
  const referenceVideo = useSourceUpload("video", { nativeBridge });
  const referenceVideoId = referenceVideo.state.id;
  /** Chain's OWN control-LoRA selection — NOT `AppShell`'s (Create-only) one;
   * see the field's doc comment on `UseChainFormResult` for why they are
   * deliberately separate states. */
  const [controlLora, setControlLora] = useState<ControlLoraSelection | null>(null);
  const setControlLoraStrength = useCallback(
    (raw: number) => {
      setControlLora((prev) => (prev === null ? prev : { name: prev.name, strength: clampLoraStrength(raw) }));
    },
    [],
  );
  /** Mirrors Create's `referenceActive`/`icLoraActive` (one flag there, since
   * Chain has no separate A2V dominance question): an attached reference video,
   * or a selected control LoRA that requires one. Deliberately NOT influenced by
   * the two strength overrides — a 🔁 replace that clears the id must drop the
   * 128 grid even while a strength is still opted in. */
  const referenceActive = referenceVideoId !== null || controlLora !== null;

  const baseLimits: ChainCommonLimits = useMemo(
    () => ({
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
    }),
    [config.limits.max_width, config.limits.max_height],
  );

  // The active rounding multiple + width/height bounds, mirroring Create's
  // `active` memo verbatim (`modes/single/useGenerationForm.ts`): with a
  // reference active the config maxima are FLOORED onto the 128 grid first
  // (1088 -> 1024) so a rounded value can never slip past the real ceiling, and
  // the minima are raised onto it. Switching the multiple itself — rather than
  // snapping once on attach — is what keeps the sliders on the 128 grid
  // afterwards; a one-shot snap would be undone by the very next drag.
  const active = useMemo(() => {
    if (!referenceActive) {
      return {
        multiple: GENERAL_MULTIPLE,
        minWidth: baseLimits.minWidth,
        maxWidth: baseLimits.maxWidth,
        minHeight: baseLimits.minHeight,
        maxHeight: baseLimits.maxHeight,
      };
    }
    const maxWidth = floorToMultiple(baseLimits.maxWidth, REFERENCE_MULTIPLE, 0, baseLimits.maxWidth);
    const maxHeight = floorToMultiple(baseLimits.maxHeight, REFERENCE_MULTIPLE, 0, baseLimits.maxHeight);
    return {
      multiple: REFERENCE_MULTIPLE,
      minWidth: ceilToMultiple(baseLimits.minWidth, REFERENCE_MULTIPLE, baseLimits.minWidth, maxWidth),
      maxWidth,
      minHeight: ceilToMultiple(baseLimits.minHeight, REFERENCE_MULTIPLE, baseLimits.minHeight, maxHeight),
      maxHeight,
    };
  }, [referenceActive, baseLimits]);

  /** The bounds the UI renders (`SizeFields`' input bounds) — the ACTIVE ones. */
  const limits: ChainCommonLimits = useMemo(
    () => ({
      minWidth: active.minWidth,
      maxWidth: active.maxWidth,
      minHeight: active.minHeight,
      maxHeight: active.maxHeight,
    }),
    [active],
  );

  const [common, setCommon] = useState<ChainCommonValues>(() => ({
    width: roundToMultiple(initial.width ?? config.generation_defaults.width, 64, MIN_WIDTH, config.limits.max_width),
    height: roundToMultiple(initial.height ?? config.generation_defaults.height, 64, MIN_HEIGHT, config.limits.max_height),
    // W7: a right-click prefill can seed fps (material policy: selection
    // rate/scale); otherwise the config default, unchanged.
    frameRate: initial.frameRate ?? config.generation_defaults.frame_rate,
    seed: config.generation_defaults.seed,
  }));

  const [gettingSize, setGettingSize] = useState(false);
  const [getSizeError, setGetSizeError] = useState<string | null>(null);

  // §1-15: `active.multiple` (64, or 128 with a reference video active) — not a
  // hard-coded 64 — is what keeps a snapped edit ON the reference grid instead
  // of knocking it back off with every stepper press.
  const setWidth = useCallback(
    (raw: number, snap = true) =>
      setCommon((prev) => {
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
      setCommon((prev) => {
        if (snap) return { ...prev, height: roundToMultiple(raw, active.multiple, active.minHeight, active.maxHeight) };
        if (!Number.isFinite(raw)) return prev;
        return { ...prev, height: raw };
      }),
    [active],
  );
  const setFrameRate = useCallback(
    (raw: number) => setCommon((prev) => ({ ...prev, frameRate: Math.min(60, Math.max(1, raw)) })),
    [],
  );
  const setSeed = useCallback((raw: number) => setCommon((prev) => ({ ...prev, seed: Math.trunc(raw) })), []);

  // N1: opt-in output crop — `null` ("not set") until the user opts in via
  // the UI's enable checkbox. Re-clamps to the CURRENT common width/height on
  // every call; a crop left stale by a later width/height shrink is instead
  // caught by `isValid` (see `chainUtils.isCropOutputValid`'s doc comment).
  const [cropOutput, setCropOutputState] = useState<CropOutput | null>(config.generation_defaults.crop_output ?? null);
  const setCropOutput = useCallback(
    (raw: CropOutput | null) => setCropOutputState(raw === null ? null : clampCropOutput(raw, common.width, common.height)),
    [common.width, common.height],
  );

  const getSizeFromAviUtl2 = useCallback(async () => {
    setGettingSize(true);
    setGetSizeError(null);
    try {
      const editInfo = await nativeBridge.request("getEditInfo", {});
      setCommon((prev) => ({
        ...prev,
        // §1-15: the ACTIVE grid — pulling the project size in while a reference
        // video is attached must land on 128, not 64.
        width: ceilToMultiple(editInfo.width, active.multiple, active.minWidth, active.maxWidth),
        height: ceilToMultiple(editInfo.height, active.multiple, active.minHeight, active.maxHeight),
      }));
    } catch (err) {
      setGetSizeError(err instanceof Error ? err.message : String(err));
    } finally {
      setGettingSize(false);
    }
  }, [nativeBridge, active]);

  const defaultClipNumFrames = useMemo(
    () => snapClipNumFrames(config.generation_defaults.num_frames),
    [config.generation_defaults.num_frames],
  );

  // Auto-detected mode: attaching a source video (attach intent — see
  // `hasSourceVideo` below) is what makes this a V2V chain.
  const sourceVideo = useSourceUpload("video", { nativeBridge });
  const sourceStatus = sourceVideo.state.status;
  // Attach intent, NOT upload completion — `uploading`/`error` count as
  // "attached" so a mid-upload (e.g. swapping the source file) or a failed
  // pick doesn't momentarily flip the screen back to scratch and re-pad the
  // clip list. Only `idle` (never picked, or explicitly cleared) is "scratch".
  const hasSourceVideo = sourceStatus === "uploading" || sourceStatus === "ready" || sourceStatus === "error";
  const mode: ChainMode = hasSourceVideo ? "v2v" : "scratch";

  const [clips, setClips] = useState<ChainClipInput[]>(() =>
    // Opens in scratch mode (no source yet) → the from-scratch 2-clip minimum.
    // W8: clip 0 seeds from the right-click DURATION policy's comfort-ceiling
    // value when a prefill supplied one (`initial.numFrames`); the padding clip
    // fitClipsToBounds adds keeps the config default. Both re-snapped onto the
    // clip 8n+1 grid.
    // X4: #1 extend-video (`forceSingleClip`) starts at the V2V 1-clip floor so
    // the source-attaching prefill doesn't leave a stray 2nd clip card; every
    // other flow opens in scratch mode at the from-scratch 2-clip minimum.
    fitClipsToBounds(
      [makeClip(snapClipNumFrames(initial.numFrames ?? config.generation_defaults.num_frames))],
      initial.forceSingleClip ? MIN_CLIPS_WITH_SOURCE : MIN_CLIPS_NO_SOURCE,
      MAX_CLIPS,
      defaultClipNumFrames,
    ),
  );

  const contextFramesLimits = useMemo(
    () => ({ min: config.limits.v2v_context_frames_min, max: config.limits.v2v_context_frames_max }),
    [config.limits.v2v_context_frames_min, config.limits.v2v_context_frames_max],
  );
  const [contextFrames, setContextFramesState] = useState(() =>
    snapContextFrames(config.limits.v2v_context_frames_default, config.limits.v2v_context_frames_min, config.limits.v2v_context_frames_max),
  );
  const setContextFrames = useCallback(
    (raw: number) => setContextFramesState(snapContextFrames(raw, contextFramesLimits.min, contextFramesLimits.max)),
    [contextFramesLimits.min, contextFramesLimits.max],
  );

  // NOTE the source-attach clip re-fit effect is NOT here any more: its "cleared"
  // arm has to read the END slot's floor too (removing the source video from a
  // chain that ends with material must not pad a stray second clip back in), and
  // that slot is declared much further down. It lives next to the end slot's own
  // re-fit effect instead, which reads the same pair.
  //
  // NOTE `minClips`/`maxClips`/`canAddClip` are not computed here either, for the
  // same reason: 素材（末尾）relaxes the clip floor a source video relaxes AND
  // closes the "add clip" button entirely (窓内モード is single-clip only). All
  // three live next to `isClipCountValid` below.

  const addClip = useCallback(() => {
    setClips((prev) => (prev.length >= MAX_CLIPS ? prev : [...prev, makeClip(defaultClipNumFrames)]));
  }, [defaultClipNumFrames]);

  const removeClip = useCallback((id: string) => {
    setClips((prev) => (prev.length <= 1 ? prev : prev.filter((clip) => clip.id !== id)));
  }, []);

  // §1-16: a REAL change to a card's prompt marks it as touched, irreversibly
  // (`intact: false`) — the auto-fit resolver must then leave it alone entirely.
  // Re-setting the same text (a controlled input echoing its own value back, a
  // re-render, a paste of identical content) is not a change and must not
  // consume the flag.
  const setClipPrompt = useCallback((id: string, value: string) => {
    setClips((prev) =>
      prev.map((clip) => {
        if (clip.id !== id) return clip;
        if (clip.prompt === value) return clip;
        return { ...clip, prompt: value, intact: false };
      }),
    );
  }, []);

  const setClipNumFrames = useCallback((id: string, value: number, snap = true) => {
    setClips((prev) =>
      prev.map((clip) => {
        if (clip.id !== id) return clip;
        // §1-16: same "only a REAL change touches the card" rule as
        // `setClipPrompt` — and it is the SNAPPED (i.e. actually applied) value
        // that is compared, so nudging a slider back onto the value it already
        // holds leaves the card intact.
        const next = snap ? snapClipNumFrames(value) : value;
        // Free manual entry: pass through untouched (no snapping, no clamp —
        // Gradio-faithful), mirroring `setWidth`/`setHeight`. Reject only
        // non-finite input, keeping the prior value.
        if (!snap && !Number.isFinite(value)) return clip;
        if (next === clip.numFrames) return clip;
        return { ...clip, numFrames: next, intact: false };
      }),
    );
  }, []);

  // §1-16: the "追加クリップの長さ" slider. Declared above `applyPreset` because
  // that callback re-seeds it (judgement point 7); see the field's doc comment
  // on `UseChainFormResult` for what it actually drives.
  const [addedClipFrames, setAddedClipFramesState] = useState(ADDED_CLIP_FRAMES_DEFAULT);
  const setAddedClipFrames = useCallback((raw: number, snap = true) => {
    setAddedClipFramesState((prev) => {
      if (snap) return snapClipNumFrames(raw);
      if (!Number.isFinite(raw)) return prev;
      return raw;
    });
  }, []);

  const applyPreset = useCallback(
    (name: string) => {
      const preset = config.generation_presets[name];
      if (!preset) return;
      // Compute the actually-applied (rounded) width/height locally so the
      // crop clamp below uses the SAME values this call is about to set —
      // not the stale `common.width/height` still in the closure (mirrors
      // Create's `useGenerationForm.applyPreset` for the same reason: never
      // clamp against values this call is about to replace). §1-15: rounded on
      // the ACTIVE grid, so a preset applied while a reference video is
      // attached cannot leave the form off the 128 grid and permanently
      // blocked.
      const newWidth = roundToMultiple(preset.width, active.multiple, active.minWidth, active.maxWidth);
      const newHeight = roundToMultiple(preset.height, active.multiple, active.minHeight, active.maxHeight);
      setCommon((prev) => ({
        ...prev,
        width: newWidth,
        height: newHeight,
      }));
      // N1: mirror the preset's own crop_output (Gradio-faithful) — reflect
      // it when present, or turn crop OFF (back to `null`) when the preset
      // doesn't define one, rather than leaving a stale prior crop in place.
      setCropOutputState(preset.crop_output ? clampCropOutput(preset.crop_output, newWidth, newHeight) : null);
      // `recommendedClipFrames` looks up `spill_free_frames` by the preset's
      // OWN (unclamped) width/height (matching `gradio_ui/presets.py`'s
      // `_chain_preset_clip_recommendation`, which keys off `width_v`/
      // `height_v` verbatim) — not the just-clamped `common` values above,
      // since the server's `spill_free_frames` map is keyed by the preset's
      // documented resolution.
      const recommended = recommendedClipFrames(preset.width, preset.height, preset.num_frames, config.limits.spill_free_frames);
      // §1-16: `intact` is deliberately NOT spread away here. Applying a preset
      // is not "the user edited this card" — it is a bulk re-seed the user asked
      // for from the preset dropdown — so an untouched card stays untouched and
      // the ↔️ auto-fit resolver may still override this length later. (A card
      // the user HAS touched stays `intact: false` and keeps the preset length
      // the resolver is now forbidden from changing, which is the intended
      // asymmetry.)
      setClips((prev) => prev.map((clip) => ({ ...clip, numFrames: recommended })));
      // Judgement point 7: re-seed the "追加クリップの長さ" slider to THIS
      // preset's recommendation. Leaving it at the fixed 257 default would make
      // the next ↔️ press silently overwrite a 768p preset's comfort ceiling
      // with a length that preset deliberately avoided.
      setAddedClipFramesState(recommended);
    },
    [config.generation_presets, config.limits.spill_free_frames, active],
  );

  // Clip 0's single start frame — `useKeyframes` capped to one image. Only
  // rendered/injected in `"scratch"` mode (a V2V source supplies clip 0's
  // starting frames instead). frame_idx stays at its default 0.
  const startFrame = useKeyframes(config, { nativeBridge }, { maxItemsOverride: 1 });

  // Chain's unified source-input picker (task brief "Chainのソース入力欄一本
  // 化"): `activeSourceKind`/`pickSource`/`clearSource` below are the single
  // entry/clear point `SourceInputPanel` renders instead of separate
  // start-frame and source-video controls. `startFrame`/`sourceVideo` stay
  // mutually exclusive by construction — every branch here that sets one
  // slot also clears the other.
  const activeSourceKind: "image" | "video" | null = hasSourceVideo
    ? "video"
    : startFrame.items.length > 0
      ? "image"
      : null;

  const [isPickingSource, setIsPickingSource] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  // §1-6: the length of the material actually being uploaded — see the
  // `sourceDurationSec` field's doc comment above for the three cases. Only the
  // right-click #1 path supplies an `AttachedSourceInfo`, so every other attach
  // route leaves this `null` and gate B never fires for it.
  const [sourceDurationSec, setSourceDurationSec] = useState<number | null>(null);

  // Shared by `pickSource` and (contract v7) `attachSourceByPath`: routes a
  // resolved file by extension and applies the single-slot-exclusive
  // attach/clear-the-other-slot logic. Extracted so drag-and-drop can reuse
  // the EXACT same routing/exclusivity/sourceError behavior as the native
  // dialog picker, rather than a second hand-maintained copy of it.
  const attachRoutedSource = useCallback(
    async (filePath: string, fileName: string, info?: AttachedSourceInfo, options?: SourceAttachOptions) => {
      const routed = routeSourceByExtension(fileName);
      // §3-102: the video half of this slot is closed on the loaded engine.
      // Checked BEFORE either branch runs, so neither slot is touched — the
      // start frame the user may already have attached must survive a refused
      // video, exactly as an unrecognised extension leaves it alone.
      if (options?.imagesOnly && routed === "video") {
        setSourceError("VIDEO_SOURCE_UNSUPPORTED");
        return;
      }
      if (routed === "image") {
        // Single-slot replace: drop any existing start frame before adding
        // the new one (mirrors `sourceVideo.uploadPath`'s own in-place
        // overwrite) — `startFrame.addFromPath` has no capacity guard of its
        // own, so this ordering is what keeps the slot at exactly one item.
        const existing = startFrame.items[0];
        if (existing) startFrame.remove(existing.id);
        await startFrame.addFromPath(filePath, fileName);
        sourceVideo.clear();
        setSourceError(null);
        // The video slot is gone — nothing left to measure.
        setSourceDurationSec(null);
      } else if (routed === "video") {
        const existing = startFrame.items[0];
        if (existing) startFrame.remove(existing.id);
        // §1-6: the length is decided BEFORE the await so it is in place the
        // moment the upload settles. Trim applied -> the cut window's length;
        // no trim -> the probed full-file length; nothing known -> null (gate
        // off). `trimQuery` returns `undefined` for every no-trim decision, so
        // an omitted `info` sends the exact pre-§1-6 request.
        setSourceDurationSec(
          info === undefined
            ? null
            : info.trim.trim
              ? info.trim.durationSec
              : info.knownDurationSec !== null && info.knownDurationSec > 0
                ? info.knownDurationSec
                : null,
        );
        await sourceVideo.uploadPath(filePath, fileName, info ? trimQuery(info.trim) : undefined);
        setSourceError(null);
      } else {
        setSourceError("UNSUPPORTED_FILE_TYPE");
      }
    },
    [startFrame, sourceVideo],
  );

  const pickSource = useCallback(
    async (options?: SourceAttachOptions) => {
      setIsPickingSource(true);
      try {
        let picked: { filePath: string; fileName: string };
        try {
          // §3-102: narrowing the KIND is what actually keeps a video out —
          // the native dialog then offers image extensions only, so the user
          // never gets as far as a file the form would have to refuse.
          picked = await nativeBridge.request("ui.pickFile", {
            kind: options?.imagesOnly ? "image" : "imageOrVideo",
          });
        } catch (err) {
          const code = errorCodeOf(err);
          // CANCELLED (user dismissed the dialog) leaves both slots untouched,
          // mirroring every other picker in this codebase.
          if (code !== "CANCELLED") setSourceError(code);
          return;
        }
        await attachRoutedSource(picked.filePath, picked.fileName, undefined, options);
      } finally {
        setIsPickingSource(false);
      }
    },
    [nativeBridge, attachRoutedSource],
  );

  // Contract v7 (drag-and-drop): attaches an already-resolved file (no native
  // Open dialog involved), sharing `pickSource`'s exact routing/exclusivity/
  // sourceError behavior via `attachRoutedSource`.
  const attachSourceByPath = useCallback(
    (filePath: string, fileName: string, info?: AttachedSourceInfo, options?: SourceAttachOptions) =>
      attachRoutedSource(filePath, fileName, info, options),
    [attachRoutedSource],
  );

  const clearSource = useCallback(() => {
    sourceVideo.clear();
    const existing = startFrame.items[0];
    if (existing) startFrame.remove(existing.id);
    setSourceError(null);
    setSourceDurationSec(null);
  }, [startFrame, sourceVideo]);

  // Owner request 2026-08-10: the attached material's pixel size, shown on the
  // SOURCE card's file-name line (`SourceInputPanel`) so the width/height
  // sliders can be matched against it. Same recipe as Create's IC-LoRA
  // reference-video duration probe (`useGenerationForm.ts`): one
  // `fs.probeMediaInfo` per ready attachment, guarded by a ref so a re-render
  // can't re-probe the same one, and re-claimed on an interrupted run so the
  // next run for the same target still probes. `.catch()` is mandatory —
  // `fs.probeMediaInfo` DOES reject (e.g. BAD_REQUEST on an empty path), and an
  // unhandled rejection fails the test run.
  //
  // Both source modes go through this ONE effect, keyed by a string that
  // encodes which slot and which attachment it is: switching modes, swapping
  // the file, or clearing all pass through a state where neither slot is ready
  // (`uploading`, or empty), which turns the key null and resets the readout —
  // so a stale size can never be attributed to a new material.
  const startFrameItem = startFrame.items[0];
  const videoProbeReady = sourceVideo.state.status === "ready" ? sourceVideo.state.id : null;
  const imageProbeReady = startFrameItem?.status === "ready" ? startFrameItem.imageId : null;
  const probeKey = videoProbeReady
    ? `video:${videoProbeReady}`
    : imageProbeReady
      ? `image:${imageProbeReady}`
      : null;
  const probeFilePath = videoProbeReady
    ? sourceVideo.state.filePath
    : imageProbeReady
      ? (startFrameItem?.filePath ?? null)
      : null;

  const [sourceMediaSize, setSourceMediaSize] = useState<{ width: number; height: number } | null>(null);
  const probedSourceKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!probeKey || !probeFilePath) {
      probedSourceKeyRef.current = null;
      setSourceMediaSize(null);
      return;
    }
    if (probedSourceKeyRef.current === probeKey) return;
    probedSourceKeyRef.current = probeKey;

    let cancelled = false;
    let settled = false;
    void nativeBridge
      .request("fs.probeMediaInfo", { filePath: probeFilePath })
      .then((info) => {
        settled = true;
        if (cancelled) return;
        // Either dimension missing (`0`, the bridge's "unknown") makes the whole
        // readout meaningless, so it is dropped rather than shown half-filled.
        setSourceMediaSize(info.width > 0 && info.height > 0 ? { width: info.width, height: info.height } : null);
      })
      .catch(() => {
        settled = true;
        if (cancelled) return;
        setSourceMediaSize(null);
      });
    return () => {
      cancelled = true;
      if (!settled) probedSourceKeyRef.current = null;
    };
  }, [probeKey, probeFilePath, nativeBridge]);

  const [overlapFrames, setOverlapFramesState] = useState(DEFAULT_OVERLAP_FRAMES);
  const setOverlapFrames = useCallback((raw: number) => setOverlapFramesState(clampOverlapFrames(raw)), []);
  const [overlapStrength, setOverlapStrengthState] = useState(DEFAULT_OVERLAP_STRENGTH);
  const setOverlapStrength = useCallback((raw: number) => setOverlapStrengthState(clampOverlapStrength(raw)), []);
  const [endSourceStrength, setEndSourceStrengthState] = useState(DEFAULT_END_SOURCE_STRENGTH);
  const setEndSourceStrength = useCallback(
    (raw: number) => setEndSourceStrengthState(clampEndSourceStrength(raw)),
    [],
  );

  const [chunkedUpsample, setChunkedUpsample] = useState(true);

  // Stage-2 window: opens on the server default, so an untouched form sends no
  // `stage2_window` key at all (see `chainUtils.buildChainRequest`).
  const [stage2Window, setStage2Window] = useState<Stage2Window>(STAGE2_WINDOW_DEFAULT);
  /** The CURRENT stage-2 window's `(vTile, vAdv)` — the geometry both
   * `chainLayoutError` and `planAudioFit` need. Read from the frozen preset table
   * so there is exactly one place these numbers live. */
  const stage2WindowGeometry = STAGE2_WINDOW_PRESETS[stage2Window];

  // ── §1-16 長尺A2V: the audio slot ─────────────────────────────────────────
  // A second, INDEPENDENT upload slot (not routed through the unified
  // image/video picker): audio is mutually exclusive with the source VIDEO
  // server-side, but the two are separate cards on screen, and holding both
  // briefly is a state the user must be able to see and undo — so this slot is
  // never auto-cleared by attaching a video. The clash is a Generate gate
  // (`audioConflictsWithSourceVideo`), not a silent detach.
  const sourceAudio = useSourceUpload("audio", { nativeBridge });
  // Pulled out as stable locals: `sourceAudio` itself is a fresh object every
  // render, so depending on it would re-create every audio callback each time.
  const audioUploadPath = sourceAudio.uploadPath;
  const audioClear = sourceAudio.clear;
  const audioStatus = sourceAudio.state.status;
  const audioId = sourceAudio.state.id;
  const audioFilePath = sourceAudio.state.filePath;
  /** Attach intent, not upload completion — same rule as `hasSourceVideo`. */
  const hasSourceAudio = audioStatus !== "idle";

  const [audioDurationSec, setAudioDurationSec] = useState<number | null>(null);
  const [audioProbeFailed, setAudioProbeFailed] = useState(false);
  const [audioAttachError, setAudioAttachError] = useState<"tooLong" | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [audioFitEvent, setAudioFitEvent] = useState<{
    id: number;
    outcome: AudioFitOutcome;
    durationSec: number;
  } | null>(null);
  const audioFitCounterRef = useRef(0);
  /** A length the CALLER already measured exactly, keyed by the file path it
   * belongs to (`timeline.extractAudio` returns one for the wav it just cut).
   * Consumed by the probe effect below INSTEAD of probing. Keyed by path rather
   * than just held as a bare number so a later manual attach of a different file
   * can never inherit it. */
  const knownAudioDurationRef = useRef<{ filePath: string; durationSec: number } | null>(null);
  /** Guards against re-probing the SAME successful upload more than once — this
   * effect's deps include `common.frameRate`/`overlapFrames`, which can change
   * after a probe already ran. Same pattern as Create's `probedAudioIdRef`. */
  const probedAudioIdRef = useRef<string | null>(null);
  /** The audio id the once-per-track auto-fit has already fired for. Attaching a
   * DIFFERENT track produces a new id and therefore fires again, exactly once. */
  const autoFitDoneForAudioIdRef = useRef<string | null>(null);

  const attachAudioByPath = useCallback(
    async (filePath: string, fileName: string, opts?: { knownDurationSec?: number }) => {
      setAudioAttachError(null);
      setAudioError(null);
      setAudioDurationSec(null);
      setAudioProbeFailed(false);
      probedAudioIdRef.current = null;
      autoFitDoneForAudioIdRef.current = null;
      // Priority 1 of the three-tier length resolution: a caller-supplied exact
      // duration. Only a POSITIVE one counts — `0`/absent falls through to the
      // probes rather than being adopted as "zero seconds".
      knownAudioDurationRef.current =
        opts?.knownDurationSec !== undefined && opts.knownDurationSec > 0
          ? { filePath, durationSec: opts.knownDurationSec }
          : null;
      await audioUploadPath(filePath, fileName);
    },
    [audioUploadPath],
  );

  const pickAudio = useCallback(async () => {
    let picked: { filePath: string; fileName: string };
    try {
      picked = await nativeBridge.request("ui.pickFile", { kind: "audio" });
    } catch (err) {
      const code = errorCodeOf(err);
      // CANCELLED (user dismissed the dialog) leaves the slot untouched,
      // mirroring every other picker in this codebase.
      if (code !== "CANCELLED") setAudioError(code);
      return;
    }
    await attachAudioByPath(picked.filePath, picked.fileName);
  }, [nativeBridge, attachAudioByPath]);

  const clearAudio = useCallback(() => {
    audioClear();
    setAudioDurationSec(null);
    setAudioProbeFailed(false);
    setAudioAttachError(null);
    setAudioError(null);
    knownAudioDurationRef.current = null;
    probedAudioIdRef.current = null;
    autoFitDoneForAudioIdRef.current = null;
    // The clip list is deliberately NOT rolled back: whatever the auto-fit
    // produced is now simply the user's chain.
  }, [audioClear]);

  // Three-tier length resolution, run exactly once per ready upload:
  //   1. a caller-supplied exact `knownDurationSec` (right-click extract),
  //   2. `fs.probeAudioDuration` — the wav header, exact, wav only,
  //   3. `fs.probeMediaInfo` — native `get_media_info`, covers mp3/m4a/…
  // All three failing sets `audioProbeFailed`, which disables every
  // length-derived gate and hands the decision to the server's own preflight.
  useEffect(() => {
    if (audioStatus !== "ready") {
      probedAudioIdRef.current = null;
      setAudioDurationSec(null);
      setAudioProbeFailed(false);
      return;
    }
    if (!audioId || probedAudioIdRef.current === audioId) return;
    probedAudioIdRef.current = audioId;

    /** Owner spec 8: a track longer than ANY chain could consume is rejected at
     * attach time — the slot is detached again and the panel shows the red
     * line. Doing this here (rather than only as a validity reason) is what
     * keeps the user from building a whole chain around an unusable file. The
     * `audioTooLong` reason still exists as the defensive re-check for a frame
     * rate / seam overlap changed AFTER a borderline attach. */
    const applyDuration = (durationSec: number) => {
      if (!(durationSec > 0)) {
        setAudioDurationSec(null);
        setAudioProbeFailed(true);
        return;
      }
      if (audioLatentsAvailable(durationSec) > maxChainAudioLatents(common.frameRate, overlapFrames)) {
        setAudioAttachError("tooLong");
        setAudioDurationSec(null);
        setAudioProbeFailed(false);
        knownAudioDurationRef.current = null;
        autoFitDoneForAudioIdRef.current = null;
        // Detaching drives `audioStatus` back to `idle`, which re-enters this
        // effect's first branch and resets the probe claim.
        audioClear();
        return;
      }
      setAudioDurationSec(durationSec);
      setAudioProbeFailed(false);
    };

    const known = knownAudioDurationRef.current;
    if (known !== null && (audioFilePath === null || known.filePath === audioFilePath)) {
      applyDuration(known.durationSec);
      return;
    }
    if (!audioFilePath) {
      // Nothing to probe (a carried-over slot with no local path): defer to the
      // server rather than pretending a length is known.
      setAudioDurationSec(null);
      setAudioProbeFailed(true);
      return;
    }

    let cancelled = false;
    let settled = false;
    void (async () => {
      let measured = 0;
      try {
        const wav = await nativeBridge.request("fs.probeAudioDuration", { filePath: audioFilePath });
        if (wav.isWav && wav.durationSec > 0) measured = wav.durationSec;
      } catch {
        // `fs.probeAudioDuration` is documented never to reject, but a bridge
        // stand-in may; fall through to the media-info probe either way.
      }
      if (measured <= 0) {
        try {
          // `.catch`-equivalent is mandatory: unlike the wav probe,
          // `fs.probeMediaInfo` DOES reject on BAD_REQUEST (empty path), and an
          // unhandled rejection fails the whole test run.
          const info = await nativeBridge.request("fs.probeMediaInfo", { filePath: audioFilePath });
          if (info.durationSec > 0) measured = info.durationSec;
        } catch {
          // Both probes exhausted — handled by `applyDuration(0)` below.
        }
      }
      settled = true;
      if (cancelled) return;
      applyDuration(measured);
    })();

    return () => {
      cancelled = true;
      // Interrupted before it settled (unmount, or a dep like `frameRate`
      // changing mid-flight): drop the claim so the next run for this same id
      // re-probes instead of silently never probing at all.
      if (!settled) probedAudioIdRef.current = null;
    };
  }, [audioStatus, audioId, audioFilePath, nativeBridge, audioClear, common.frameRate, overlapFrames]);

  /** "↔️ 再生時間の自動調整". The plan is computed from the CURRENT `clips`
   * closure (not inside a `setClips` updater) so the toast event can be published
   * as an ordinary side effect rather than from inside a state reducer. */
  const adjustClipsForAudio = useCallback(() => {
    if (audioDurationSec === null) return;
    const plan = planAudioFit({
      clips: clips.map((clip) => ({ id: clip.id, numFrames: clip.numFrames, intact: isClipIntact(clip) })),
      audioDurationSec,
      fps: common.frameRate,
      overlapFrames,
      targetClipFrames: addedClipFrames,
      stage2Window: stage2WindowGeometry,
      // A chain carrying audio never carries a source video (they are mutually
      // exclusive), so the floor is always the from-scratch minimum — which is
      // also `planAudioFit`'s own default.
    });
    if (plan.changed) {
      const byId = new Map(clips.map((clip) => [clip.id, clip]));
      setClips(
        plan.clips.map((entry) => {
          const existing = entry.sourceId === null ? undefined : byId.get(entry.sourceId);
          // An existing card keeps its prompt, its start frame AND its `intact`
          // flag — being resized by the resolver is not "the user touched it".
          return existing ? { ...existing, numFrames: entry.numFrames } : makeClip(entry.numFrames);
        }),
      );
    }
    audioFitCounterRef.current += 1;
    // Published even for an unchanged plan: "already fits" / "nothing I may
    // move" are results the user asked for and deserve a toast.
    setAudioFitEvent({ id: audioFitCounterRef.current, outcome: plan.outcome, durationSec: audioDurationSec });
  }, [audioDurationSec, clips, common.frameRate, overlapFrames, addedClipFrames, stage2WindowGeometry]);

  // Auto-fit ONCE per attached track, the moment its length is known. Swapping
  // the audio yields a new `audio_id` and fires again — exactly once for that
  // one too. Nothing else re-triggers it: changing fps or the seam overlap
  // afterwards is caught by the Generate gates and the ↔️ hint, never by a
  // surprise re-layout of the clip list.
  useEffect(() => {
    if (audioStatus !== "ready" || !audioId) return;
    if (audioDurationSec === null) return;
    if (autoFitDoneForAudioIdRef.current === audioId) return;
    autoFitDoneForAudioIdRef.current = audioId;
    adjustClipsForAudio();
  }, [audioStatus, audioId, audioDurationSec, adjustClipsForAudio]);

  // ── §1-15 参照動画: the rest of the slot (the upload itself and the control
  // selection were declared at the top of this hook — see there for why) ──────
  const referenceUploadPath = referenceVideo.uploadPath;
  const referenceClear = referenceVideo.clear;
  const referenceStatus = referenceVideo.state.status;
  const referenceFilePath = referenceVideo.state.filePath;
  /** Attach intent, not upload completion — same rule as `hasSourceAudio`. */
  const hasReferenceVideo = referenceStatus !== "idle";

  const [referenceError, setReferenceError] = useState<string | null>(null);
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

  const referenceFrameRate = common.frameRate;
  const attachReferenceByPath = useCallback(
    async (filePath: string, fileName: string, trim?: SourceTrimDecision) => {
      setReferenceError(null);
      // The `max_frames` cap rides on every reference upload — see
      // `REFERENCE_MAX_FRAMES_QUERY`. It is NOT a trim in the §1-6 sense: the
      // server answers `trimmed: false` whenever the file was already short
      // enough, which is the normal case, so `useSourceUpload` deliberately does
      // not arm `trimFailed` for a query that carries no trim key.
      //
      // §1-15 W4 (2026-08-11): the right-click route additionally asks for the
      // object's ribbon range. The two are MERGED into one query rather than
      // sent as alternatives, because `/upload/video` reads both from the same
      // params object — but note that the server SKIPS `max_frames` entirely
      // once a trim window is present (`video_upload_store.py` 143-153), so the
      // cap can no longer be relied on to bound the stored file. That is why the
      // window's own duration is clamped here to the same ceiling the cap
      // expresses: `MAX_CHAIN_TOTAL_FRAMES` generation frames, in seconds at the
      // CURRENT generation fps. (A reference recorded at a higher fps than the
      // generation still stores more source frames than that, which is harmless:
      // the surplus only costs space in `uploads/`, the temporary staging area,
      // and the run reads no further than the window計算 anyway.)
      const clamped: SourceTrimDecision | undefined =
        trim?.trim && referenceFrameRate > 0
          ? { ...trim, durationSec: Math.min(trim.durationSec, MAX_CHAIN_TOTAL_FRAMES / referenceFrameRate) }
          : trim;
      const query = { ...REFERENCE_MAX_FRAMES_QUERY, ...(clamped ? trimQuery(clamped) : undefined) };
      await referenceUploadPath(filePath, fileName, query);
    },
    [referenceUploadPath, referenceFrameRate],
  );

  const pickReference = useCallback(async () => {
    let picked: { filePath: string; fileName: string };
    try {
      picked = await nativeBridge.request("ui.pickFile", { kind: "video" });
    } catch (err) {
      const code = errorCodeOf(err);
      // CANCELLED (user dismissed the dialog) leaves the slot untouched,
      // mirroring every other picker in this codebase.
      if (code !== "CANCELLED") setReferenceError(code);
      return;
    }
    await attachReferenceByPath(picked.filePath, picked.fileName);
  }, [nativeBridge, attachReferenceByPath]);

  const clearReference = useCallback(() => {
    referenceClear();
    setReferenceError(null);
    // The strengths and the control-LoRA selection survive on purpose: they are
    // settings the user chose, not the material — re-attaching keeps them.
  }, [referenceClear]);

  // Snap width/height onto the 128 grid the moment a reference video becomes
  // READY (the id transitions null -> non-null), mirroring Create's own re-snap
  // effect. `active` already holds the 128 bounds by the time this runs, since
  // `referenceActive` flipped in the same render. The dynamic `active.multiple`
  // is what KEEPS them there afterwards; this one-shot is only about the value
  // that was already in the form before the attach. Keyed off the id (not the
  // pick call) so a CANCELLED/failed pick never moves the user's size.
  const prevReferenceVideoIdRef = useRef<string | null>(referenceVideoId);
  useEffect(() => {
    const becameReady = prevReferenceVideoIdRef.current === null && referenceVideoId !== null;
    prevReferenceVideoIdRef.current = referenceVideoId;
    if (!becameReady) return;
    setCommon((prev) => ({
      ...prev,
      width: roundToMultiple(prev.width, REFERENCE_MULTIPLE, active.minWidth, active.maxWidth),
      height: roundToMultiple(prev.height, REFERENCE_MULTIPLE, active.minHeight, active.maxHeight),
    }));
  }, [referenceVideoId, active]);

  // The reference's pixel size, for the panel's resolution readout — same
  // once-per-attachment probe (and the same mandatory `.catch()`, since
  // `fs.probeMediaInfo` DOES reject on BAD_REQUEST) as `sourceMediaSize` above.
  // Its DURATION is deliberately not probed: the server slices the reference by
  // FRAME NUMBER, so a seconds readout would invite exactly the wrong mental
  // model (the panel says so in words instead).
  const [referenceMediaSize, setReferenceMediaSize] = useState<{ width: number; height: number } | null>(null);
  const probedReferenceIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (referenceStatus !== "ready" || !referenceVideoId || !referenceFilePath) {
      probedReferenceIdRef.current = null;
      setReferenceMediaSize(null);
      return;
    }
    if (probedReferenceIdRef.current === referenceVideoId) return;
    probedReferenceIdRef.current = referenceVideoId;

    let cancelled = false;
    let settled = false;
    void nativeBridge
      .request("fs.probeMediaInfo", { filePath: referenceFilePath })
      .then((info) => {
        settled = true;
        if (cancelled) return;
        setReferenceMediaSize(info.width > 0 && info.height > 0 ? { width: info.width, height: info.height } : null);
      })
      .catch(() => {
        settled = true;
        if (cancelled) return;
        setReferenceMediaSize(null);
      });
    return () => {
      cancelled = true;
      // Interrupted before it settled: drop the claim so the next run for this
      // same id re-probes instead of silently never probing at all.
      if (!settled) probedReferenceIdRef.current = null;
    };
  }, [referenceStatus, referenceVideoId, referenceFilePath, nativeBridge]);

  // ── 素材（末尾）: the end slot ──────────────────────────────────────────────
  // The mirror image of the unified START slot (`activeSourceKind`/`pickSource`/
  // `attachRoutedSource`/`clearSource`), and built the same way: ONE user-facing
  // slot backed by two sub-slots, chosen by the file's extension through the
  // same `routeSourceByExtension`. An IMAGE needs an `image_id` and a thumbnail
  // (`useKeyframes` capped at one), a VIDEO needs a `video_id`
  // (`useSourceUpload`) — nothing else about the two differs here, because the
  // BACKEND turns the image into a still video and runs both down one code path.
  //
  // Unlike the START slot this one is NOT mutually exclusive with the source
  // video: "start from this, end with that" is the interpolation case the
  // feature exists for. It IS exclusive with the audio track and the reference
  // video, which is a Generate gate (never a silent detach), exactly like every
  // other cross-slot clash on this screen.
  const endSourceVideo = useSourceUpload("video", { nativeBridge });
  const endSourceImage = useKeyframes(config, { nativeBridge }, { maxItemsOverride: 1 });

  const endSourceVideoStatus = endSourceVideo.state.status;
  const endSourceImageItem = endSourceImage.items[0];
  /** Which sub-slot holds material. The video half wins while it is non-idle,
   * mirroring `activeSourceKind`'s own precedence — the two are never both
   * filled anyway (every attach branch clears the other). */
  const endSourceKind: "image" | "video" | null =
    endSourceVideoStatus !== "idle" ? "video" : endSourceImageItem ? "image" : null;
  /** The two sub-slots' statuses unified. `"empty"` (a `useKeyframes` card with
   * no image, which only `addEmpty` creates and this slot never calls) is folded
   * into `"uploading"` — an unfinished card is never treated as ready. */
  const endSourceStatus: SourceUploadStatus =
    endSourceKind === "video"
      ? endSourceVideoStatus
      : endSourceKind === "image"
        ? endSourceImageItem?.status === "ready"
          ? "ready"
          : endSourceImageItem?.status === "error"
            ? "error"
            : "uploading"
        : "idle";
  /** Attach intent, not upload completion — same rule as `hasSourceAudio`. */
  const hasEndSource = endSourceStatus !== "idle";

  const [endSourceError, setEndSourceError] = useState<string | null>(null);
  const [endSourceDurationSec, setEndSourceDurationSec] = useState<number | null>(null);

  const endSourceVideoFrames = endSourceVideo.state.frames;
  const endSourceVideoFps = endSourceVideo.state.fps;
  const generationFrameRate = common.frameRate;
  const endSourceContext = resolveEndSourceContext({
    kind: endSourceKind,
    serverFrames: endSourceVideoFrames,
    serverFps: endSourceVideoFps,
    attachDurationSec: endSourceDurationSec,
    genFps: generationFrameRate,
  });
  const endContextFrames = endSourceContext.frames;
  const endSourceLengthIssue = endSourceContext.issue;

  // Shared by `pickEndSource` and `attachEndSourceByPath` — `attachRoutedSource`'s
  // mirror, including its "decide the length BEFORE the await" ordering and its
  // `trimQuery(info.trim)` handling.
  //
  // Nothing about the anchor is seeded here: it is the constant 8, gated only on
  // whether this attach ends up producing usable material (see
  // `resolveEndSourceContext`). What the attach DOES set up is the upload's
  // `max_frames` cap — an end source is only ever read from its head, so storing
  // a whole feature film in `uploads/` buys nothing.
  const attachRoutedEndSource = useCallback(
    async (filePath: string, fileName: string, info?: AttachedSourceInfo) => {
      const routed = routeSourceByExtension(fileName);
      if (routed === "image") {
        // Single-slot replace, same ordering as `attachRoutedSource`: drop the
        // previous card first (`addFromPath` has no capacity guard of its own),
        // and detach the other half so the two are never both filled.
        const existing = endSourceImage.items[0];
        if (existing) endSourceImage.remove(existing.id);
        endSourceVideo.clear();
        // An image has no length to measure, and none is needed: a still can be
        // frozen for the anchor's 8 frames whatever it is.
        setEndSourceDurationSec(null);
        await endSourceImage.addFromPath(filePath, fileName);
        setEndSourceError(null);
      } else if (routed === "video") {
        const existing = endSourceImage.items[0];
        if (existing) endSourceImage.remove(existing.id);
        setEndSourceDurationSec(
          info === undefined
            ? null
            : info.trim.trim
              ? info.trim.durationSec
              : info.knownDurationSec !== null && info.knownDurationSec > 0
                ? info.knownDurationSec
                : null,
        );
        // The upload cap. Two things ride on the query, merged (never
        // alternatives) because `/upload/video` reads both from one params
        // object:
        //  - `max_frames`: END_SOURCE_MAX_UPLOAD_FRAMES, generous enough that the
        //    material's first 9 generation-rate frames are reachable even from a
        //    high-fps file, small enough that a two-hour one does not land whole
        //    in `uploads/`;
        //  - the ribbon trim, when the right-click route supplied one. The
        //    server SKIPS `max_frames` entirely once a trim window is present
        //    (`video_upload_store.py`), so the window's own duration is clamped
        //    to the same ceiling expressed in seconds — the identical treatment
        //    the §1-15 reference slot already gives its cap.
        // Asking for `max_frames` is ALSO what makes the server measure and
        // return `frame_count`/`fps`, which is what the 9-frame length gate is
        // judged on — so this query is load-bearing twice over.
        const trim = info?.trim;
        const clampedTrim: SourceTrimDecision | undefined =
          trim?.trim && generationFrameRate > 0
            ? { ...trim, durationSec: Math.min(trim.durationSec, END_SOURCE_MAX_UPLOAD_FRAMES / generationFrameRate) }
            : trim;
        const query = { ...END_SOURCE_MAX_FRAMES_QUERY, ...(clampedTrim ? trimQuery(clampedTrim) : undefined) };
        await endSourceVideo.uploadPath(filePath, fileName, query);
        setEndSourceError(null);
      } else {
        setEndSourceError("UNSUPPORTED_FILE_TYPE");
      }
    },
    [endSourceImage, endSourceVideo, generationFrameRate],
  );

  const pickEndSource = useCallback(async () => {
    let picked: { filePath: string; fileName: string };
    try {
      picked = await nativeBridge.request("ui.pickFile", { kind: "imageOrVideo" });
    } catch (err) {
      const code = errorCodeOf(err);
      // CANCELLED (user dismissed the dialog) leaves the slot untouched,
      // mirroring every other picker in this codebase.
      if (code !== "CANCELLED") setEndSourceError(code);
      return;
    }
    await attachRoutedEndSource(picked.filePath, picked.fileName);
  }, [nativeBridge, attachRoutedEndSource]);

  const attachEndSourceByPath = useCallback(
    (filePath: string, fileName: string, info?: AttachedSourceInfo) =>
      attachRoutedEndSource(filePath, fileName, info),
    [attachRoutedEndSource],
  );

  const clearEndSource = useCallback(() => {
    endSourceVideo.clear();
    const existing = endSourceImage.items[0];
    if (existing) endSourceImage.remove(existing.id);
    setEndSourceError(null);
    setEndSourceDurationSec(null);
    // Nothing else to reset: the anchor is a constant gated on the material's
    // presence, so detaching makes `endContextFrames` `null` on its own.
  }, [endSourceImage, endSourceVideo]);

  // ── Clip re-fit on the two slots' attach/clear transitions ────────────────
  //
  // Both effects live HERE, below the end slot's declarations, because both
  // floors read BOTH slots: either one relaxes the chain to a single clip, so
  // clearing one must re-fit against whatever the other still holds. (The source
  // effect used to sit up with `sourceStatus`, where `hasEndSource` is still in
  // its temporal dead zone — moving it is the same treatment `minClips` got.)

  // Clip re-fit on the two discrete source-attach transitions only (previous
  // status tracked in a ref). Firing on the *ready*/*idle* edges — never on
  // `uploading`/`error` — is what keeps the clip list from wobbling while a
  // source upload is in flight (`hasSourceVideo` already holds `mode` steady
  // through that window):
  //  - 未添付→ready: relax the floor to 1 clip and bump clip 0 up just enough
  //    that the configured default `context_frames` (e.g. 73) stays below it
  //    (default num_frames is only 49), so V2V isn't invalid out of the box.
  //  - 添付→idle (cleared): pad back up to the floor the OTHER slot leaves in
  //    force. With no end source that is exactly `MIN_CLIPS_NO_SOURCE`; with one
  //    attached it stays 1, which stops a V2V detach from silently padding a
  //    stray clip back in while the end slot's own floor still wants 1.
  const prevSourceStatus = useRef<SourceUploadStatus>(sourceStatus);
  useEffect(() => {
    const prev = prevSourceStatus.current;
    prevSourceStatus.current = sourceStatus;
    if (prev === sourceStatus) return;

    if (sourceStatus === "ready" && prev !== "ready") {
      setClips((prevClips) => {
        const fitted = fitClipsToBounds(prevClips, MIN_CLIPS_WITH_SOURCE, MAX_CLIPS, defaultClipNumFrames);
        const first = fitted[0];
        if (first && first.numFrames <= contextFrames) {
          const bumped = snapClipNumFrames(contextFrames + 8);
          return fitted.map((clip, index) => (index === 0 ? { ...clip, numFrames: bumped } : clip));
        }
        return fitted;
      });
    } else if (sourceStatus === "idle" && prev !== "idle") {
      setClips((prevClips) =>
        fitClipsToBounds(prevClips, minClipsForChain(hasEndSource), MAX_CLIPS, defaultClipNumFrames),
      );
    }
    // `uploading`/`error` transitions intentionally do nothing here.
  }, [sourceStatus, contextFrames, defaultClipNumFrames, hasEndSource]);

  // Clip re-fit on the end slot's CLEAR transition only.
  //
  // There is deliberately NO ready-edge branch. v1 carved the freeze out of the
  // user's own final clip and had to push it up on attach; the window-internal
  // mode freezes the last 8 frames of the clip the user already asked for, so
  // attaching takes nothing from the clip list and silently lengthening a clip
  // would be a change nobody asked for.
  //
  //  - 添付→idle (cleared): re-fit to the current floor, which pads a
  //    single-clip chain back up to the from-scratch 2-clip minimum. The mirror
  //    of the source effect's own cleared arm above.
  const prevEndSourceStatus = useRef<SourceUploadStatus>(endSourceStatus);
  useEffect(() => {
    const prev = prevEndSourceStatus.current;
    prevEndSourceStatus.current = endSourceStatus;
    if (prev === endSourceStatus) return;

    if (endSourceStatus === "idle" && prev !== "idle") {
      setClips((prevClips) =>
        fitClipsToBounds(prevClips, minClipsForChain(hasSourceVideo), MAX_CLIPS, defaultClipNumFrames),
      );
    }
    // `ready`/`uploading`/`error` transitions intentionally do nothing here.
  }, [endSourceStatus, hasSourceVideo, defaultClipNumFrames]);

  // のりしろ既定値1 (2.6(e), 逆順Chained second stage): a SYMMETRIC one-shot
  // nudge on the "素材（末尾）× 2+ clips" transition. Reverse mode's own
  // narrowest legal seam-blend width (§2.2's worked example uses kv=1) is a
  // very different comfortable default from a plain chain's kv=3, so ENTERING
  // that state drops `overlapFrames` to 1, and LEAVING it (back down to 1
  // clip, or the material removed) restores the plain-chain default —
  // otherwise a user who goes 2 clips -> 1 clip would find themselves stuck at
  // the very value `endSourceNeedsOverlap` (窓内モード's own `kv >= 2` floor)
  // then refuses. `useRef` tracks only the PREVIOUS edge, so this fires
  // exactly once per transition rather than fighting the user every render —
  // a value dialed in mid-state (`overlapFrames` is deliberately NOT a dep
  // here) survives untouched.
  const wasReverseEndSource = useRef(hasEndSource && clips.length > 1);
  useEffect(() => {
    const isReverse = hasEndSource && clips.length > 1;
    if (isReverse !== wasReverseEndSource.current) {
      wasReverseEndSource.current = isReverse;
      setOverlapFrames(isReverse ? 1 : DEFAULT_OVERLAP_FRAMES);
    }
  }, [hasEndSource, clips.length, setOverlapFrames]);

  const firstClipNumFrames = clips[0]?.numFrames ?? 0;
  const contextFramesValid = isContextFramesValid(
    contextFrames,
    contextFramesLimits.min,
    contextFramesLimits.max,
    firstClipNumFrames,
  );

  const totalFrames = computeTotalFrames(clips);
  const totalFramesValid = isTotalFramesValid(totalFrames);
  // 素材（末尾）relaxes the clip floor exactly like a source video does (owner
  // decision E: "これで終わる5秒の動画" is a natural single-clip request), so both
  // the floor and the count check read the same OR.
  const chainFloorRelaxed = hasSourceVideo || hasEndSource;
  const minClips = minClipsForChain(chainFloorRelaxed);
  const clipCountValid = isClipCountValid(chainFloorRelaxed, clips.length);

  // 逆順Chained (2026-08-18, second stage): an end source no longer closes the
  // ceiling — 2+ clips is the `reverse` mode the server accepts, so the only
  // ceiling left is the ordinary one every chain has. `maxClips` is exposed as
  // its own field (rather than inlining `MAX_CLIPS`) because `addClip` keeps
  // its own `MAX_CLIPS` guard: this is the button's enabled state, not the
  // state machine's rule.
  const maxClips = MAX_CLIPS;
  const canAddClip = clips.length < MAX_CLIPS;

  /** 窓内モード品質警告: the ceiling a clip may not outgrow while end material is
   * attached, or `null` when no clip does. One stage-2 window in pixel frames —
   * 169 on `standard`, 145 on `high_resolution`. Advisory: NOT a
   * `validityReason`.
   *
   * 逆順Chained (2026-08-18): SINGLE-CLIP ONLY (`clips.length === 1`) — the
   * study behind this ceiling (§62's reconnaissance experiment) ran the
   * window-internal case exclusively, so it says nothing about a multi-clip
   * reverse chain's own quality (that is one of the second-stage's own open
   * unknowns, backend VERIFICATION_LOG.md §61 / Docs/PENDING_TASKS_CLOSED.md
   * §3-84). */
  const stage2WindowPixelFrames = pxFromVLatent(stage2WindowGeometry.vTile);
  const endSourceQualityLimitFrames =
    hasEndSource && clips.length === 1 && clips.some((clip) => clip.numFrames > stage2WindowPixelFrames)
      ? stage2WindowPixelFrames
      : null;

  /** §1-22 素材（末尾）複数クリップ品質警告: connecting 2+ clips while material
   * is attached (逆順Chained) degrades quality, independent of any single
   * clip's length. Advisory: NOT a `validityReason`.
   *
   * Deliberately the plain complement of the ceiling above rather than a
   * shared derivation — `endSourceQualityLimitFrames` is single-clip-only
   * (see its own doc comment) and this is multi-clip-only, so the two
   * conditions are exhaustive and mutually exclusive over `clips.length`
   * and never render into the same banner slot at once. */
  const endSourceMultiClipQualityWarning = hasEndSource && clips.length >= 2;

  // `<lora:name:strength>` tags found in the shared prompt — parsed once here
  // and reused by `buildRequest` below instead of re-running `parseLoraPrompt`.
  const parsedPrompt = useMemo(() => parseLoraPrompt(prompt), [prompt]);

  /** 窓内モード: how much of the output's tail is the frozen anchor, `0` when
   * there is no usable material. A SLICE of the output, never an addition to it
   * — see the field's doc on `UseChainFormResult`. */
  const outputTailFrames = hasEndSource && endContextFrames !== null ? endContextFrames : 0;

  // The estimate counts the CLIPS and nothing else. 窓内モード does not add a
  // segment: the anchor's 8 frames are part of the last clip's own window, so
  // they are already inside `totalFrames` and adding them again would over-report
  // the wait by a third of a second's worth of work.
  const estimateSeconds = useMemo(
    () => estimateGenerationSeconds(common.width, common.height, totalFrames),
    [common.width, common.height, totalFrames],
  );
  const estimateLabel = `Est. ${formatEstimate(estimateSeconds)}`;

  // Predicted output length: seam-blend fusion always eats `overlapFrames`
  // per join, and V2V additionally replaces its leading `contextFrames` with
  // the source video rather than generating them anew — see
  // `computeOutputFrames`'s own doc comment in `chainUtils.ts` for the exact
  // arithmetic. Scratch mode passes `null` for the V2V-only term.
  //
  // 素材（末尾）adds NOTHING to this (2026-08-17): the anchor is frozen inside the
  // last clip's window instead of being appended after it, so the delivered file
  // is exactly as long with end material as without it.
  const outputFrames = computeOutputFrames(
    clips.map((clip) => clip.numFrames),
    overlapFrames,
    hasSourceVideo ? contextFrames : null,
  );
  const outputSeconds = common.frameRate > 0 ? outputFrames / common.frameRate : 0;
  const outputTailSeconds = common.frameRate > 0 ? outputTailFrames / common.frameRate : 0;

  // V2V needs a fully-uploaded source with valid context_frames; scratch has
  // no source gate at all.
  const sourceReady = !hasSourceVideo || (sourceStatus === "ready" && contextFramesValid);

  const noUploadInFlight = !startFrame.isUploading && sourceStatus !== "uploading";

  // Final send-time grid gate for width/height (item 12): manual keyboard
  // entry (`setWidth(raw, false)`) passes values through unsnapped, so an
  // off-grid value must block submit here rather than being silently
  // corrected — mirrors Create's `useGenerationForm.isValid`. §1-15: measured
  // against the ACTIVE grid, so with a reference video attached this demands
  // the 128 multiple (the reason code pushed for it is the reference-specific
  // one — see `validityReasons` below).
  const dimensionsOnGrid =
    isDimensionOnGrid(common.width, active.multiple, active.minWidth, active.maxWidth) &&
    isDimensionOnGrid(common.height, active.multiple, active.minHeight, active.maxHeight);

  // Same free-typed-manual-entry gate as width/height, now that
  // `setClipNumFrames` also accepts `snap=false` (`DurationField`'s number
  // input, one instance per `ClipCard`): EVERY clip's `numFrames` must be
  // checked, not just the first — a hand-typed off-grid value on any clip
  // must block Generate.
  const clipsNumFramesOnGrid = clips.every((clip) => isNumFramesOnGrid(clip.numFrames, MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES));

  // X3: client-side gate for the backend's per-clip seam-blend minimum. The
  // server rejects (HTTP 422) any clip with `overlap_frames >= (num_frames-1)//8+1`,
  // i.e. it requires `num_frames >= 8*overlap_frames + 1`. Pre-block Generate
  // here with an imperative note so the user fixes it before submitting (the
  // server 422 stays as a fallback). Boundary `num_frames == 8*overlap+1` passes.
  const minFramesForOverlap = 8 * overlapFrames + 1;
  const clipsMeetOverlapMinimum = clips.every((clip) => clip.numFrames >= minFramesForOverlap);

  // Unconditional guard: while the source video is still uploading (or failed),
  // never present the form as submittable — even though `mode` stays "v2v"
  // through that window to keep the clip list steady.
  const sourceNotBusy = sourceStatus !== "uploading" && sourceStatus !== "error";

  // §1-6 ゲートB: the material actually being sent must be able to supply
  // `contextFrames` after resample to the generation fps, or the server rejects
  // it with 422 SOURCE_VIDEO_TOO_SHORT before any GPU work.
  //
  // The `- 1` is a deliberate ONE-FRAME safety margin, not an off-by-one: the
  // server counts frames through a double rounding
  // (`round(round(span·src_fps)·gen_fps/src_fps)`) whose source fps the WebUI
  // does not know, so an exact match is not reproducible here. The boundary is
  // therefore biased to the blocking side — keeping §3-32's "no 422s reach the
  // user" property is worth occasionally asking for one more frame than
  // strictly necessary. `sourceDurationSec === null` (unmeasured) disables the
  // gate entirely, exactly like the right-click length guard's "pass on
  // unknown" rule.
  const sourceVideoTooShortForContext =
    sourceDurationSec !== null && Math.floor(sourceDurationSec * common.frameRate) - 1 < contextFrames;

  // §1-6: a trim was requested but the response did not confirm it — the stored
  // video is the whole file, so continuing would generate from the wrong part.
  const sourceTrimFailed = sourceVideo.state.trimFailed;

  // §1-14/§3-57: the narrower "high_resolution" window can hold a shorter frozen
  // V2V head (137 frames) than the server's published context cap (145), so the
  // combination "high_resolution + a long context" 422s. Blocked here so the
  // user fixes it before submitting, keeping §3-32's "no 422 reaches the user"
  // property; the server stays the final arbiter. Only checked with a source
  // video attached — `contextFrames` is meaningless without one.
  const maxContextForWindow = stage2MaxContextFrames(stage2Window);
  const contextFramesTooLongForWindow = hasSourceVideo && contextFrames > maxContextForWindow;

  // §1-14: advisory only — never folded into `validityReasons`. Exceeding the
  // comfortable per-window token budget makes a run slow and memory-hungry, it
  // does not make it invalid.
  //
  // The budget itself is SERVED (2026-08-12) so it can be tuned per machine
  // without a rebuild; `resolveChainComfortBudget` covers an older backend that
  // does not send it at all, and the offline fallback config.
  //
  // 2026-08-31: the served TABLE takes precedence when it has a row for this
  // engine + acceleration configuration — that is what widens LTX 2.5's guide
  // line (40,000 -> 44,880). With no matching row (LTX 2.3's default
  // configuration, an older backend, or before `GET /models` lands) this falls
  // back to the scalar key exactly as before.
  // D tidy (2026-08-31): memoized with the SAME deps style
  // `useGenerationForm.ts`'s own `comfortRow` uses, for consistency across the
  // two hooks that call `resolveComfortRow`.
  const comfortRow = useMemo(
    () => resolveComfortRow(config.limits, engineFamily, acceleration, sageAvailable),
    [config.limits, engineFamily, acceleration, sageAvailable],
  );
  const comfortBudget = comfortRow?.chainBudget ?? resolveChainComfortBudget(config.limits.chain_comfort_token_budget);
  const chainWindowOverBudget = isChainWindowOverBudget(common.width, common.height, stage2Window, comfortBudget);
  // The same budget, expressed as positions on the two sliders. `active.multiple`
  // is the grid the sliders can actually land on (64, or 128 with a reference
  // video active — §1-15), so the guides are floored onto reachable values.
  // Primitive deps only, mirroring `baseLimits`' flavour of memo.
  const chainWindowMarkers = useMemo(
    () => chainWindowBudgetMarkers(common.width, common.height, stage2Window, comfortBudget, active.multiple),
    [common.width, common.height, stage2Window, comfortBudget, active.multiple],
  );

  // ── §1-16 長尺A2V: the audio gates and the panel's advisory numbers ────────
  const clipNumFramesList = clips.map((clip) => clip.numFrames);
  /** The audio slot holds a usable upload. */
  const audioReady = audioStatus === "ready" && audioId !== null;
  /** The measured length, but ONLY while it may actually be gated on: `null`
   * whenever nothing is attached, the upload isn't ready, or no probe could
   * measure it (`audioProbeFailed`). An unmeasurable track is never blocked
   * here — the server's preflight decides. */
  const audioLengthKnown = audioReady && !audioProbeFailed ? audioDurationSec : null;
  const audioAvailableLatents = audioLengthKnown === null ? null : audioLatentsAvailable(audioLengthKnown);
  /** Audio-latent frames the CURRENT clip list consumes (`null` = impossible
   * geometry, which `chainLayoutInvalid`/`clipTooShortForOverlap` already cover). */
  const audioRequiredLatents =
    audioLengthKnown === null ? null : audioLatentsRequired(clipNumFramesList, common.frameRate, overlapFrames);
  const audioTooLong =
    audioAvailableLatents !== null &&
    audioAvailableLatents > maxChainAudioLatents(common.frameRate, overlapFrames);
  // Owner spec 9 (point at the Single screen) takes precedence over spec 7
  // (point at the ↔️ button): if not even the shortest possible chain fits,
  // pressing ↔️ cannot help and saying so would be a dead end.
  const audioTooShortForChain =
    !audioTooLong &&
    audioLengthKnown !== null &&
    isAudioTooShortForChain(firstClipNumFrames, audioLengthKnown, common.frameRate, overlapFrames);
  const audioTooShort =
    !audioTooLong &&
    !audioTooShortForChain &&
    audioAvailableLatents !== null &&
    audioRequiredLatents !== null &&
    audioAvailableLatents < audioRequiredLatents;
  /** Judgement point 2: only ever raised WITH audio attached. The same geometry
   * rejects plain chains too, but widening this gate is a separate ledger item
   * — here it exists so the ↔️ button can never hand over a 422. */
  const chainLayoutInvalid =
    audioReady && chainLayoutError(clipNumFramesList, common.frameRate, overlapFrames, stage2WindowGeometry) !== null;

  /** Leftover tail, in seconds — a note, never a gate. */
  const audioSurplusSec =
    audioAvailableLatents !== null && audioRequiredLatents !== null
      ? Math.max(0, (audioAvailableLatents - audioRequiredLatents) / AUDIO_LATENTS_PER_SEC)
      : null;
  /** At the 24-card ceiling with audio still left over. The leftover is measured
   * against `audioFit`'s one-frame safety margin, not against zero: a plan that
   * fits exactly still leaves that one frame behind by design, and reporting it
   * as "unused audio" would fire the note on every perfect fit. */
  const audioAtMaxClips =
    clips.length >= MAX_CLIPS &&
    audioAvailableLatents !== null &&
    audioRequiredLatents !== null &&
    audioAvailableLatents - audioRequiredLatents > AUDIO_FIT_SAFETY_MARGIN_LATENTS;

  /** Per-clip slice of the track, for the cards' "🎵 担当時間帯" badges. */
  const audioSegmentWindowsForClips = useMemo(() => {
    if (!hasSourceAudio || audioDurationSec === null) return null;
    return audioSegmentWindows(clips.map((clip) => clip.numFrames), common.frameRate, overlapFrames);
  }, [hasSourceAudio, audioDurationSec, clips, common.frameRate, overlapFrames]);

  // ── §1-15 参照動画: the cross-field gates ──────────────────────────────────
  // The final `loras[]` this form would send: Chain's own control selection
  // first, the prompt's STYLE tags after, deduped by name (`combineLoras`,
  // `lora/controlLoras.ts` — the control adapter's leading position is a hard
  // engine requirement, not cosmetic). Computed here, not only in
  // `buildRequest`, so the gates below react live. Create's `mergedLoras`
  // verbatim.
  const mergedLoras = useMemo(() => combineLoras(controlLora, parsedPrompt.loras), [controlLora, parsedPrompt]);

  // N3's rule, unchanged from Create: `reference_video_id` requires a non-empty
  // `loras[]` server-side...
  const referenceNeedsLoras = referenceVideoId !== null && mergedLoras.length === 0;
  // ...and its converse. Judged on the MERGED array's NAMES rather than on
  // `controlLora !== null` deliberately (レビューS4): Chain does not
  // auto-migrate a hand-typed control tag into the panel the way Create does,
  // so a `<lora:canny-control:1.0>` typed straight into the prompt would
  // otherwise sail past this gate and 422 at the server.
  const controlLoraNeedsReference =
    referenceVideoId === null && mergedLoras.some((lora) => controlLoraNames.has(lora.name));
  // Owner decision 2026-08-11: depth-preprocess adapters are blocked on Chain
  // UNCONDITIONALLY, not only past one clip. This is deliberately STRICTER
  // than the server (which only rejects depth on a chain of two clips or
  // more, accepting it fine on a single-clip chain) — the FE has chosen to
  // disallow depth on Chain entirely for v1 rather than carve out a
  // single-clip exception, since that state is already unreachable in
  // practice (minClips / referenceConflictsWithSourceVideo /
  // controlLoraNeedsReference already block a bare single-clip-plus-depth
  // chain from validating). Same name-based judgement as above, so a
  // hand-typed depth tag is caught too. Silent when the server never
  // published the adapter's `preprocess` (`depthLoraNames` empty) — an
  // unknown preprocess is never guessed at.
  const depthChainUnsupported = mergedLoras.some((lora) => depthLoraNames.has(lora.name));

  // ── 素材（末尾）: the gates ────────────────────────────────────────────────
  // Two LENGTH gates about the material, and four SHAPE gates about the chain
  // it is attached to. All of them stay silent while nothing is attached.
  const endSourceTooShort = hasEndSource && endSourceLengthIssue === "tooShort";
  const endSourceLengthUnknown = hasEndSource && endSourceLengthIssue === "unknown";
  // 逆順Chained (2026-08-18, second stage): a V2V source video combined with an
  // end source on 2+ clips — mirrors `chain_math.py:1097-1105`'s reverse-mode
  // exclusion. A SINGLE clip is unaffected (the start+end interpolation case,
  // implemented and tested); this only fires once `addClip` (2.6(a), no longer
  // capped by the end source) has taken the chain to 2 or more.
  const endSourceWithSourceVideoMultiClip = hasSourceVideo && hasEndSource && clips.length >= 2;
  // のりしろ1 × 素材（末尾）on a SINGLE-clip chain (窓内モード) is a hard server
  // rejection: the anchor arrives as one MORE internal segment, and at
  // `kv === 1` the audio-overlap budget (`sum_ka`) is exhausted, which
  // `chain_math` reports as its pre-existing "degenerate audio overlap" error.
  // An exhaustive 275,400-case sweep pinned the failure to kv=1 and nothing
  // else (kv>=2: zero cases at any frame rate), so the client mirror is this
  // single condition rather than a re-derivation of the audio budget.
  // `overlapFrames` is clamped to >= 1 by its setter, so this is exactly "the
  // user is at the bottom of the seam-blend slider".
  //
  // 逆順Chained (2026-08-18, second stage): `clips.length === 1` is new — the
  // server exempts `reverse` mode (2+ clips) from this `kv >= 2` floor
  // entirely (`chain_math.py`'s `end_source_mode != "reverse"` guard on the
  // rejection), because `reverse` appends no internal segment and therefore
  // spends the SAME audio-overlap budget a plain chain does. What replaces
  // this gate on 2+ clips is {@link endSourceAudioOverlapBudget} below.
  const endSourceNeedsOverlap = hasEndSource && clips.length === 1 && overlapFrames < 2;

  // 逆順Chained (2026-08-18, second stage): the two acceptance checks the
  // server ONLY runs in `reverse` mode (2+ clips) — mirrors of
  // `chain_math.py`'s own reverse-mode guards, both silent with 0 or 1 clips
  // (`chainUtils`'s helper functions are already no-ops there).
  const lastClip = clips[clips.length - 1];
  const nEndV = vTailLatents(END_SOURCE_CONTEXT_FRAMES);
  // The end band is the LAST clip's own tail and that clip's HEAD is the
  // previous clip's carried-forward のりしろ — if the two together already fill
  // it, there is nothing left to generate (chain_math.py:1076-1088). Only
  // meaningful once the material can actually supply the anchor
  // (`endContextFrames !== null`): otherwise no `end_source` is sent at all
  // (see `buildRequest`), so there is nothing for the server to reject.
  const endSourceLastClipTooShort =
    hasEndSource &&
    clips.length > 1 &&
    endContextFrames !== null &&
    lastClip !== undefined &&
    !endSourceLastClipHasFreeLatents(lastClip.numFrames, overlapFrames, nEndV);
  // The audio-overlap budget `reverse` mode spends is the plain chain's own
  // (chain_math.py:1107-1134) — no `kv >= 2` floor protects it the way
  // `endSourceNeedsOverlap` protects 窓内モード, so this is the mirror that
  // takes over once there are 2+ clips. Same "only once an end_source would
  // actually be sent" guard as the check above.
  const endSourceAudioOverlapBudget =
    hasEndSource &&
    clips.length > 1 &&
    endContextFrames !== null &&
    !endSourceAudioOverlapOk(clipNumFramesList, common.frameRate, overlapFrames);

  // N1: a crop, if enabled, must still be a valid 32-pixel-grid crop of the
  // CURRENT common width/height (see `chainUtils.isCropOutputValid`'s doc
  // comment — this is what catches a crop left stale by a later width/height
  // shrink).
  // W7: same gate set as the old `isValid` AND-chain, expressed as a reason-code
  // list so the UI can tell the user what to fix. Every sub-boolean is unchanged
  // (regression-safe). R3: the three overlapping source gates all go false at
  // once during an upload, so they're normalized here to a single source code
  // (`sourceUploading` while an upload is in flight, else `sourceNotReady`)
  // instead of three duplicate lines — the AND of the same three booleans still
  // decides whether ANY source code is pushed, so `isValid` is unchanged.
  const validityReasons: ChainValidityReason[] = [];
  if (!isValidPrompt(prompt)) validityReasons.push("promptEmpty");
  // §1-15: ONE code for the one failing grid gate — the reference-specific line
  // while the 128 grid is what is being violated, the general one otherwise.
  // Same "normalize overlapping gates to the most specific code" rule R3 already
  // applies to the three source gates below.
  if (!dimensionsOnGrid) validityReasons.push(referenceActive ? "referenceDimensionsOffGrid" : "dimensionsOffGrid");
  if (!clipsNumFramesOnGrid) validityReasons.push("clipFramesOffGrid");
  // 逆順Chained (2026-08-18): `endSourceLastClipTooShort` reuses this SAME
  // code (chosen deliberately, per plan §2.6(f) — the remedy is identical:
  // lengthen the clip or lower the seam-blend width) rather than adding a
  // new reason code just for the last clip's own geometry.
  if (!clipsMeetOverlapMinimum || endSourceLastClipTooShort) validityReasons.push("clipTooShortForOverlap");
  if (!clipCountValid) validityReasons.push("minClips");
  if (!totalFramesValid) validityReasons.push("totalFramesExceeded");
  if (!(sourceReady && noUploadInFlight && sourceNotBusy)) {
    const uploading = startFrame.isUploading || sourceStatus === "uploading";
    validityReasons.push(uploading ? "sourceUploading" : "sourceNotReady");
  }
  // §1-6: both source-material gates sit directly after the existing source
  // codes so the note reads as one "about the source video" block.
  if (sourceVideoTooShortForContext) validityReasons.push("sourceVideoTooShortForContext");
  if (sourceTrimFailed) validityReasons.push("sourceTrimFailed");
  if (contextFramesTooLongForWindow) validityReasons.push("contextFramesTooLongForWindow");
  // §1-16 長尺A2V: the audio block sits directly after the source-video codes so
  // the note reads as one "about the material" section. The conflict line comes
  // first — with both slots filled, nothing else about the audio matters until
  // one of them is removed.
  if (hasSourceVideo && hasSourceAudio) validityReasons.push("audioConflictsWithSourceVideo");
  if (audioStatus === "uploading") validityReasons.push("audioUploading");
  if (audioStatus === "error") validityReasons.push("audioNotReady");
  // Mutually exclusive by construction (see the booleans above): at most one of
  // these three is ever true, so the note never offers two contradictory fixes.
  if (audioTooLong) validityReasons.push("audioTooLong");
  if (audioTooShortForChain) validityReasons.push("audioTooShortForChain");
  if (audioTooShort) validityReasons.push("audioTooShort");
  if (chainLayoutInvalid) validityReasons.push("chainLayoutInvalid");
  // §1-15 参照動画: the reference block sits after the audio one, and reads in
  // the same order — the conflict line first (with both slots filled nothing
  // else about the reference matters until one is removed), then the upload's
  // own state, then the two cross-field rules, then the v1 depth limit.
  if (hasSourceVideo && hasReferenceVideo) validityReasons.push("referenceConflictsWithSourceVideo");
  if (referenceStatus === "uploading") validityReasons.push("referenceUploading");
  if (referenceStatus === "error") validityReasons.push("referenceNotReady");
  // §1-15 W4 (2026-08-11): a reference trim we asked for but did not get. The
  // upload itself is fine (`ready`, usable id) — what is wrong is WHICH footage
  // it holds, so this sits with the slot's other state codes rather than with
  // the cross-field rules below. Mirrors `sourceTrimFailed` above and Create's
  // `referenceTrimFailed`; only the right-click route can ever arm it.
  if (referenceVideo.state.trimFailed) validityReasons.push("referenceTrimFailed");
  if (referenceNeedsLoras) validityReasons.push("referenceNeedsLoras");
  if (controlLoraNeedsReference) validityReasons.push("controlLoraNeedsReference");
  if (depthChainUnsupported) validityReasons.push("depthChainUnsupported");
  // 素材（末尾）: the end block sits after the reference one and reads in the
  // same order — the conflicts first (with a second slot filled nothing else
  // about the end source matters until one is removed), then the upload's own
  // state, then the material's length, then the chain's shape.
  if (hasEndSource && hasSourceAudio) validityReasons.push("endSourceConflictsWithAudio");
  if (hasEndSource && hasReferenceVideo) validityReasons.push("endSourceConflictsWithReference");
  if (endSourceWithSourceVideoMultiClip) validityReasons.push("endSourceWithSourceVideoMultiClip");
  if (endSourceStatus === "uploading") validityReasons.push("endSourceUploading");
  if (endSourceStatus === "error") validityReasons.push("endSourceNotReady");
  if (endSourceVideo.state.trimFailed) validityReasons.push("endSourceTrimFailed");
  // The two length answers are mutually exclusive by construction (one
  // `endSourceLengthIssue`), so the note never offers two contradictory fixes.
  if (endSourceTooShort) validityReasons.push("endSourceTooShort");
  if (endSourceLengthUnknown) validityReasons.push("endSourceLengthUnknown");
  if (endSourceNeedsOverlap) validityReasons.push("endSourceNeedsOverlap");
  if (endSourceAudioOverlapBudget) validityReasons.push("endSourceAudioOverlapBudget");
  if (!isCropOutputValid(cropOutput, common.width, common.height)) validityReasons.push("cropInvalid");
  // NAG (2026-07-28): enabled + blank negative-prompt body blocks Generate
  // client-side (the server 422s the same combination).
  if (isNagNegativeEmpty(nag)) validityReasons.push("nagNegativeEmpty");
  const isValid = validityReasons.length === 0;

  // I8 §3-4 (※): the config-derived baseline a pristine (no-prefill) form seeds
  // with — the reference `isDirty` compares against. Mirrors the `common`
  // useState initializer above (minus the right-click `initial` override, which
  // is intentionally treated as "already edited" so a prefilled Chain still
  // prompts before being overwritten by a second right-click).
  const baselineCommon = useMemo(
    () => ({
      width: roundToMultiple(config.generation_defaults.width, 64, MIN_WIDTH, config.limits.max_width),
      height: roundToMultiple(config.generation_defaults.height, 64, MIN_HEIGHT, config.limits.max_height),
      frameRate: config.generation_defaults.frame_rate,
      seed: config.generation_defaults.seed,
    }),
    [config.generation_defaults.width, config.generation_defaults.height, config.generation_defaults.frame_rate, config.generation_defaults.seed, config.limits.max_width, config.limits.max_height],
  );
  const baselineCrop = config.generation_defaults.crop_output ?? null;

  const isDirty = useMemo(() => {
    // A source video or clip-0 start frame is attached — the clearest "work in
    // progress" (also covers contextFrames edits, which only matter with a source).
    if (activeSourceKind !== null) return true;
    // §1-16 長尺A2V: an attached audio track is the same kind of "work in
    // progress" as an attached source video — losing it to a right-click remount
    // costs the user an upload (and the auto-fit that ran off it), so it warns
    // just like the source slot does. Attach INTENT, not readiness, exactly like
    // `activeSourceKind` above: a mid-upload track still counts.
    if (hasSourceAudio) return true;
    // §1-15: an attached reference video is the same kind of losable upload; a
    // control-LoRA selection is a deliberate choice the user would have to make
    // again, so it counts too even though it costs no upload. (No remount carry
    // exists for either — same as the audio slot, deliberately.)
    if (hasReferenceVideo) return true;
    if (controlLora !== null) return true;
    // 素材（末尾）: another losable upload, and one that also re-laid the clip
    // list (the auto push-up) — attach INTENT, like every slot above.
    if (hasEndSource) return true;
    // Clip list vs. the pristine default clip count (2 clips from scratch, or 1
    // for a #1 extend-video prefill — X4's `forceSingleClip`), empty prompt,
    // default dur. (extend-video always attaches a source, so `activeSourceKind`
    // above already flags it dirty; this branch is kept explicit for clarity.)
    const baselineMinClips = initial.forceSingleClip ? MIN_CLIPS_WITH_SOURCE : MIN_CLIPS_NO_SOURCE;
    if (clips.length !== baselineMinClips) return true;
    if (clips.some((clip) => clip.prompt !== "" || clip.numFrames !== defaultClipNumFrames)) return true;
    // Common fields vs. their config-derived seeds.
    if (
      common.width !== baselineCommon.width ||
      common.height !== baselineCommon.height ||
      common.frameRate !== baselineCommon.frameRate ||
      common.seed !== baselineCommon.seed
    )
      return true;
    // Seam-blend params vs. defaults.
    if (overlapFrames !== DEFAULT_OVERLAP_FRAMES || overlapStrength !== DEFAULT_OVERLAP_STRENGTH) return true;
    // Output crop vs. its config-default seed.
    if (!cropOutputEquals(cropOutput, baselineCrop)) return true;
    return false;
  }, [activeSourceKind, hasSourceAudio, hasReferenceVideo, hasEndSource, controlLora, clips, defaultClipNumFrames, common, baselineCommon, overlapFrames, overlapStrength, cropOutput, baselineCrop, initial.forceSingleClip]);

  const buildRequest = useCallback((): GenerateChainRequest => {
    // The shared prompt is the single source of truth for STYLE
    // `<lora:name:strength>` tags (Docs/API_REFERENCE.md §5.3), same as Create's
    // `toGenerateRequest` (`modes/single/useGenerationForm.ts`): split it into
    // the body the backend renders and the tags' own specs, then send the SAME
    // `mergedLoras` (control selection first + those tags) the gates above
    // already judged — §1-15. Reuses the `parsedPrompt` memo above.
    const { strippedPrompt } = parsedPrompt;
    const loras = mergedLoras;
    // §1-15: the reference fields only ride along with a ready reference video
    // AND at least one loRA (the server 422s the pair otherwise), and never
    // alongside a V2V source video — `referenceConflictsWithSourceVideo`
    // already blocks Generate on that pair, this is the defensive second line,
    // exactly like `sourceAudio` below.
    const sendReference = referenceVideoId !== null && !hasSourceVideo && loras.length > 0;
    // 素材（末尾）: the defensive second line for this slot's three exclusions —
    // Generate is already blocked on each (`endSourceConflictsWithAudio`/
    // `endSourceConflictsWithReference`/`endSourceWithSourceVideoMultiClip`),
    // so a request built anyway drops the END source rather than sending a
    // body the server would 422. Which side gives way is fixed (the end
    // source), so the outcome is deterministic.
    const endSourceId =
      endSourceKind === "video"
        ? endSourceVideo.state.id
        : endSourceKind === "image"
          ? (endSourceImageItem?.status === "ready" ? endSourceImageItem.imageId : null)
          : null;
    // `endContextFrames === null` (too short / unmeasurable) drops the slot too.
    // Generate is already blocked on both (`endSourceTooShort` /
    // `endSourceLengthUnknown`); this is the same defensive second line the
    // exclusions get, and it is what keeps an anchor the material cannot supply
    // out of the request entirely rather than sending it hopefully. When the slot
    // IS sent, `contextFrames` is always the constant 8 — the UI states the
    // anchor explicitly rather than relying on the server's default.
    // 逆順Chained (2026-08-18): the third exclusion — a V2V source video is
    // fine alongside an end source ONLY while the chain is still a single
    // clip (the interpolation case); `clips.length >= 1` in the request body
    // is exactly `clipInputs.length` below, computed one line down, so this
    // reads it directly off `clips` instead (identical count, since
    // `clipInputs` only ever re-shapes clip 0, never adds/removes entries).
    const endSource: EndSourceInput | null =
      endSourceId !== null &&
      endSourceKind !== null &&
      endContextFrames !== null &&
      !hasSourceAudio &&
      !hasReferenceVideo &&
      !(hasSourceVideo && clips.length >= 2)
        ? { kind: endSourceKind, id: endSourceId, contextFrames: endContextFrames, strength: endSourceStrength }
        : null;
    // The single start frame only goes onto clip 0 in scratch mode — a V2V
    // source video supplies clip 0's starting content instead.
    const clipInputs: ChainClipInput[] = clips.map((clip, index) =>
      index === 0 && mode === "scratch" ? { ...clip, conditioningImages: startFrame.conditioningImages } : clip,
    );

    return buildChainRequest({
      prompt: strippedPrompt,
      width: common.width,
      height: common.height,
      cropOutput,
      frameRate: common.frameRate,
      seed: common.seed,
      overlapFrames,
      overlapStrength,
      clips: clipInputs,
      sourceVideo: sourceVideo.state.id ? { videoId: sourceVideo.state.id, contextFrames } : null,
      // §1-16: `source_audio` and `source_video` are mutually exclusive
      // server-side. Generate is already blocked on the pair
      // (`audioConflictsWithSourceVideo`), so this is the defensive second line:
      // a request built while both slots are filled drops the audio rather than
      // sending a body the server would 422.
      sourceAudio: audioId && !hasSourceVideo ? { audioId } : null,
      endSource,
      ...(loras.length > 0 ? { loras } : {}),
      // `!= null` (not truthiness) on the strengths: `0.0` is a valid, distinct
      // value that must still be sent. `buildChainRequest` applies the same
      // rule, so passing `null` simply omits the key.
      ...(sendReference
        ? {
            referenceVideoId,
            conditioningAttentionStrength,
            referenceVideoStrength,
          }
        : {}),
      chunkedUpsample,
      stage2Window,
      nag,
      acceleration,
    });
  }, [
    parsedPrompt,
    mergedLoras,
    referenceVideoId,
    conditioningAttentionStrength,
    referenceVideoStrength,
    clips,
    mode,
    startFrame.conditioningImages,
    common,
    cropOutput,
    overlapFrames,
    overlapStrength,
    sourceVideo.state.id,
    audioId,
    hasSourceVideo,
    hasSourceAudio,
    hasReferenceVideo,
    endSourceKind,
    endSourceVideo.state.id,
    endSourceImageItem,
    endContextFrames,
    endSourceStrength,
    contextFrames,
    chunkedUpsample,
    stage2Window,
    nag,
    acceleration,
  ]);

  return {
    hasSourceVideo,
    mode,
    common,
    limits,
    sizeMultiple: active.multiple,
    setWidth,
    setHeight,
    setFrameRate,
    setSeed,
    cropOutput,
    setCropOutput,
    gettingSize,
    getSizeError,
    getSizeFromAviUtl2,
    applyPreset,
    clips,
    minClips,
    maxClips,
    canAddClip,
    addClip,
    removeClip,
    setClipPrompt,
    setClipNumFrames,
    startFrame,
    overlapFrames,
    setOverlapFrames,
    overlapStrength,
    setOverlapStrength,
    endSourceStrength,
    setEndSourceStrength,
    minFramesForOverlap,
    chunkedUpsample,
    setChunkedUpsample,
    stage2Window,
    setStage2Window,
    chainWindowOverBudget,
    chainWindowMarkers,
    stage2MaxContextFrames: maxContextForWindow,
    sourceVideo,
    contextFrames,
    setContextFrames,
    contextFramesLimits,
    isContextFramesValid: contextFramesValid,
    activeSourceKind,
    isPickingSource,
    sourceError,
    pickSource,
    attachSourceByPath,
    clearSource,
    sourceDurationSec,
    sourceMediaSize,
    sourceAudio,
    hasSourceAudio,
    pickAudio,
    attachAudioByPath,
    clearAudio,
    audioDurationSec,
    audioProbeFailed,
    audioAttachError,
    audioError,
    addedClipFrames,
    setAddedClipFrames,
    adjustClipsForAudio,
    audioFitEvent,
    audioSurplusSec,
    audioAtMaxClips,
    audioSegmentWindowsForClips,
    referenceVideo,
    hasReferenceVideo,
    pickReference,
    attachReferenceByPath,
    clearReference,
    referenceError,
    referenceMediaSize,
    conditioningAttentionStrength,
    setConditioningAttentionStrength,
    referenceVideoStrength,
    setReferenceVideoStrength,
    controlLora,
    setControlLora,
    setControlLoraStrength,
    controlLoraNames,
    referenceNeedsLoras,
    controlLoraNeedsReference,
    isReferenceActive: referenceActive,
    endSourceVideo,
    endSourceImage,
    endSourceKind,
    endSourceStatus,
    hasEndSource,
    endSourceDurationSec,
    endSourceError,
    endContextFrames,
    endSourceLengthIssue,
    endSourceQualityLimitFrames,
    endSourceMultiClipQualityWarning,
    pickEndSource,
    attachEndSourceByPath,
    clearEndSource,
    totalFrames,
    isTotalFramesValid: totalFramesValid,
    isClipCountValid: clipCountValid,
    estimateSeconds,
    estimateLabel,
    outputFrames,
    outputSeconds,
    outputTailFrames,
    outputTailSeconds,
    isValid,
    validityReasons,
    isDirty,
    buildRequest,
  };
}

export { MAX_CHAIN_TOTAL_FRAMES, MAX_CLIP_NUM_FRAMES };
