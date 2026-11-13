/**
 * The single source of truth for a right-click generation's *prefill seed* —
 * the width/height/frame-rate/`num_frames` a routed menu action lands the form
 * (and the provisional reservation) on before any user edit.
 *
 * Historically this arithmetic was duplicated: `SingleScreen`/`ChainedScreen`
 * each ran their own `deriveGenerationParams` + `computeTargetNumFrames`
 * useMemo group for the FORM seed, while `AppShell`'s reservation path fell
 * back to the raw config defaults — so a freshly-placed provisional object's
 * ribbon length could diverge from the length the remounted form actually
 * showed. This module is the one純関数 both callers now share (owner requirement
 * 2026-07-21 #1: "配置時にDURATIONエンジン結果でリボンを揃える"), so the seat and
 * the form can never derive different values from the same right-click.
 *
 * It composes the two existing frozen seams rather than re-deriving:
 *  - `deriveGenerationParams` (`deriveGenerationParams.ts`) for width/height,
 *  - `computeTargetNumFrames` (`deriveDuration.ts`) for `num_frames`,
 * and adds only the intent→policy table and the intent-specific material-
 * duration decision that sits between them. `deriveDuration.ts` itself stays a
 * primitive (no import of this module) so there is no circular dependency.
 *
 * 台帳 §1-114 (2026-10-31): the Single screen is no longer remounted by a
 * right-click — `SingleScreen` applies this seed LIVE to the mounted form when
 * the `prefillMaterial` command arrives (#2/#4/#7), and `AppShell` sizes the
 * provisional off the same call. The Settings policy that ignores the material
 * is now `"panel"` (keep the panel's current width/height/fps, handed in as
 * {@link ResolvePrefillSeedArgs.panel}); it replaced `"defaults"`, which always
 * took `config.generation_defaults`. Chained (still remounted) passes no
 * `panel`, so there "the panel's values" are the rebuilt form's defaults.
 */

import type { AppConfig, TransformerQuantClass } from "../api/types";
import { pxFromVLatent } from "../modes/chained/chainUtils";
import { MIN_HEIGHT, MIN_WIDTH } from "../modes/single/defaultConfig";
import { snapFrameRate } from "../modes/single/paramUtils";
import { STAGE2_WINDOW_DEFAULT, STAGE2_WINDOW_PRESETS } from "../shell/tokenBudget";
import type { PrefillResolutionPolicy } from "../shell/PrefillPolicyContext";
import type { AccelerationSettings } from "../shell/accelerationSettings";
import {
  comfortFramesForBudget,
  resolveComfortRow,
  resolveSequenceRowsBudget,
  singleLegacySpillTable,
} from "../shell/comfortTable";
import { H3_NO_CONDITIONING, h3ComfortFrames } from "../shell/h3SequenceRows";
import { fixedFrameRate, frameLimitsFor, LTX_GENERATION_CONSTRAINTS } from "../shell/generationConstraints";
import type { GenerationConstraints } from "../shell/generationConstraints";
import { computeTargetNumFrames } from "./deriveDuration";
import type { DurationPolicy } from "./deriveDuration";
import { deriveGenerationParams } from "./deriveGenerationParams";
import type { DerivedGenerationParams } from "./deriveGenerationParams";
import { targetModeForIntent } from "./menuRouting";
import type { SelectionItem, TimelineSelection } from "./menuSelection";
import { selectionRangeFrames } from "./selectionRange";

/**
 * Which {@link DurationPolicy} each right-click generation intent uses to seed
 * the initial `num_frames` (menuRouting.ts's `intent` strings, cross-checked
 * against that table). Two families:
 *  - `comfortCeiling`: flows with no natural source duration (T2V/I2V-style,
 *    plus every Chain intent) — DURATION is the resolution's comfort ceiling.
 *  - `materialClampedToCeiling`: flows with a source clip whose length should
 *    bound DURATION (IC-LoRA reference #2, A2V #7) — the material's own
 *    length floored to the 8n+1 grid, never past the comfort ceiling.
 *
 * Intents absent from this table (e.g. `text-to-video`, `image-from-frame`) are
 * the keep-panel-state cursor items `AppShell` handles live via the published
 * form values, not a remount seed, so they never route through here.
 */
export const DURATION_POLICY_BY_INTENT: Readonly<Record<string, DurationPolicy>> = {
  "extend-video": "comfortCeiling", // #1 (Chain)
  "reference-video": "materialClampedToCeiling", // #2 (IC-LoRA)
  "image-to-video": "comfortCeiling", // #4
  "image-to-clip-chain": "comfortCeiling", // #6 (Chain)
  "audio-to-video": "materialClampedToCeiling", // #7 (A2V from an audio or video object)
  "current-frame-to-clip-chain": "comfortCeiling", // #12 (Chain)
  // 台帳§1-16 長尺A2V (2026-08-10). Deliberately `comfortCeiling`, NOT the
  // `materialClampedToCeiling` its single-shot sibling #7 uses: this seed
  // sizes ONE CLIP of the chain, while the extracted audio's length is spread
  // over the WHOLE clip list by `audioFit.planAudioFit` (which re-lays the
  // cards the moment the wav is measured). Clamping clip 0 to the track's full
  // length would seed a single 60-second clip and then immediately be undone by
  // the auto-fit — so the honest seed is the resolution's comfort ceiling, the
  // same one every other Chain intent uses.（H3 の割り当てモードは種の長さを使わず画面の割り当てが走る。`h3LongA2vAssign`・DEVLOG §166.11）
  "audio-to-long-a2v": "comfortCeiling", // 🎵 (Chain, 台帳§1-16)
  // 台帳§1-15 W4 (2026-08-11). Deliberately `comfortCeiling`, NOT the
  // `materialClampedToCeiling` its single-shot sibling `reference-video` uses,
  // for the same reason long-a2v above is: this seed sizes ONE CLIP of
  // a chain, while the reference is a conditioning input for the WHOLE chain —
  // the server slices it by frame number across every clip, so the reference's
  // own length says nothing about how long clip 0 should be. Every other Chain
  // intent is `comfortCeiling` too, so this keeps the family uniform.
  "reference-video-chain": "comfortCeiling", // 🎬 (Chain, 台帳§1-15 W4)
  // 素材（末尾）. `comfortCeiling` like every other Chain intent, and deliberately
  // NOT `materialClampedToCeiling` even though this intent DOES carry a material:
  // the end material is EMBEDDED INSIDE the generation's own length (the last
  // `END_SOURCE_CONTEXT_FRAMES` frames are frozen onto it) rather than appended
  // to it, so the output length does not change with the material and the
  // material's duration says nothing about how long the clip should be. A
  // 30-second end source and a 3-second one both land in the same clip.
  // The comfort ceiling is additionally capped — see
  // {@link END_SOURCE_SEED_MAX_FRAMES}.
  "end-with-this": "comfortCeiling", // 🎬🖼 (Chain, 素材（末尾）)
  // §1-17 Retake: a third family — the DURATION comes from neither the
  // resolution nor the material, but from the user's SELECTED FRAME RANGE.
  retake: "selectedRangeLength",
};

/**
 * 素材（末尾）窓内モード (2026-08-17): the extra ceiling the `end-with-this` seed
 * is rounded DOWN to — ONE stage-2 window of the DEFAULT window preset,
 * expressed in pixel frames (169).
 *
 * A clip that outgrows one stage-2 window puts the frozen tail and the frames
 * that have to blend into it in different tiles, which is what makes the result
 * smear and flicker around the join. `useChainForm`'s `endSourceQualityLimitFrames`
 * warns about it for a clip the user set by hand; this makes sure the RIGHT-CLICK
 * route — the one that opens a fully-formed request the user may just press
 * Generate on — never seeds a chain that trips the warning (owner ruling
 * 2026-08-17: warn on manual, round on right-click).
 *
 * The DEFAULT preset is the right one to measure against because the seed lands
 * in a freshly mounted form, whose `stage2Window` is `STAGE2_WINDOW_DEFAULT`.
 * Switching to `high_resolution` afterwards lowers the ceiling to 145 and the
 * form's own warning takes over from there.
 *
 * 逆順Chained (2026-08-18, second stage): the VALUE is unchanged and still
 * exactly right — a right-click reservation always seeds a SINGLE clip
 * (`ChainedScreen`'s `forceSingleClip`), so this remains a clip-1-窓内モード
 * ceiling specifically. `useChainForm`'s own `endSourceQualityLimitFrames`
 * gate is now single-clip only for the identical reason: クリップ1本の
 * 窓内モードの実験結果に基づく値であって、その後ユーザーがクリップを追加した
 * 多クリップ逆順チェーンの品質については何も語らない。
 */
export const END_SOURCE_SEED_MAX_FRAMES = pxFromVLatent(STAGE2_WINDOW_PRESETS[STAGE2_WINDOW_DEFAULT].vTile);

/**
 * The inclusive timeline SPAN of a selected object, in seconds — the trimmed
 * (ribbon) length the object actually occupies on the timeline, NOT the whole
 * backing file's duration. `(frameEnd - frameStart + 1)` frames (frameEnd
 * inclusive, mirroring native's `FindObjectByJob` `next = frameEnd + 1`)
 * converted to seconds via the selection's project `rate`/`scale`. Returns
 * `undefined` when there is no item, the rate/scale is unresolvable (≤ 0), or
 * the span is non-positive — the single "unknown, degrade gracefully" signal
 * both the DURATION seed and the length guard key on. Pure.
 */
export function spanDurationSec(
  item: SelectionItem | undefined,
  selection: Pick<TimelineSelection, "rate" | "scale">,
): number | undefined {
  const { rate, scale } = selection;
  if (!item || rate <= 0 || scale <= 0) return undefined;
  const frames = item.frameEnd - item.frameStart + 1;
  if (frames <= 0) return undefined;
  return frames / (rate / scale);
}

/**
 * The source material's real-time duration (seconds) that feeds the
 * `materialClampedToCeiling` DURATION policy, decided per intent — or
 * `undefined` when this intent has no material length to clamp to, or the
 * length can't be resolved (in which case the policy falls back to the config
 * default DURATION rather than a bogus value). Exported so `SingleScreen`'s
 * async `project`-policy overwrite path recomputes DURATION off the SAME
 * intent-specific span math, keeping the span semantics consistent (M-1).
 *
 * W4 (trim追従): the IC-LoRA reference (#2) and audio-to-video (#7) intents
 * both derive their length from the
 * object's inclusive timeline SPAN ({@link spanDurationSec}) rather than the
 * full backing file's `mediaDurationSec` — the DURATION follows the ribbon's
 * trimmed length, matching the owner requirement that a trimmed #2/#7 clip seed
 * the trimmed length. An unresolvable span (no item / rate·scale ≤ 0 /
 * non-positive span) → `undefined`. Every other (comfortCeiling) intent →
 * `undefined` (no material length is needed).
 */
export function materialDurationForIntent(
  intent: string,
  selection: TimelineSelection,
): number | undefined {
  const item = selection.selected[0];
  switch (intent) {
    case "reference-video":
    case "audio-to-video":
      return spanDurationSec(item, selection);
    // §1-17 `"retake"` is deliberately NOT here. Its `selectedRangeLength`
    // policy takes the SELECTED RANGE, not the object's own span — see
    // {@link selectedRangeFramesForIntent}, the sibling below. Adding a case
    // that returned a span would be dead weight the policy never reads.
    default:
      return undefined;
  }
}

/**
 * The `selectedRangeLength` policy's input: the length of the user's selected
 * frame range in **PROJECT** frames, or `undefined` when this intent has no
 * range to read (or the snapshot reports no range at all).
 *
 * The sibling of {@link materialDurationForIntent}, and deliberately a separate
 * function rather than a case in it: the two answer different questions in
 * different units (a real-time duration of the MATERIAL vs. a frame count of
 * the SELECTION), and `computeTargetNumFrames` takes them as separate
 * arguments because only one policy ever reads each.
 *
 * The counting goes through `timeline/selectionRange.ts`, the single place the
 * (still unconfirmed) closed-interval convention lives — so if F0 settles on
 * half-open, this follows automatically.
 */
export function selectedRangeFramesForIntent(
  intent: string,
  selection: TimelineSelection,
): number | undefined {
  if (intent !== "retake" || !selection.hasRange) return undefined;
  const frames = selectionRangeFrames(selection);
  return frames > 0 ? frames : undefined;
}

/** Ledger §1-119: the keyframe cards a right-click seed's DURATION counts on
 * the sequence-rows line — 1 for the intents that land an image on frame 0
 * (#4 `image-to-video`, #10 `image-from-frame`; `SingleScreen`'s
 * `prefillMaterial` / `captureFrameToKeyframe` — and, ledger §1-276, the Chain
 * intents that open clip 0 on an image: #6 `image-to-clip-chain`, #12
 * `current-frame-to-clip-chain`; `ChainedScreen`'s mount loads), 0 for every
 * other (#1 `extend-video` is V2V: its clip 0 has no opening image). A seed
 * counts no Reference card (on H3 #2 and #7 go to the Reference cards and never
 * reach a seed — `AppShell`). */
export function seedKeyframeCount(intent: string): number {
  return intent === "image-to-video" ||
    intent === "image-from-frame" ||
    intent === "image-to-clip-chain" ||
    intent === "current-frame-to-clip-chain"
    ? 1
    : 0;
}

export interface ResolvePrefillSeedArgs {
  /** The routed sub-flow identifier (`GenerationPrefill.intent`). */
  intent: string;
  /** The file-path-completed timeline selection the seed derives from. */
  selection: TimelineSelection;
  config: AppConfig;
  /** W1: the Settings SIZE policy (`usePrefillPolicy().sizePolicy`), driving
   * width/height only. `panel` ignores the material's real size and keeps
   * {@link panel}'s width/height (the config defaults when absent), still run
   * through the intent's grid — #2 IC-LoRA rounds it to 128;
   * `material`/`project` seed width/height from the material (`project` is
   * later overwritten off `getEditInfo` by the caller). */
  sizePolicy: PrefillResolutionPolicy;
  /** 台帳 §1-114: the panel's CURRENT width/height/fps, read by the `panel`
   * policy on either axis. Only the Single-bound callers pass it (`SingleScreen`
   * from its live form, `AppShell` from the published form values when the
   * target is Single); Chained/Edit omit it, so `panel` there falls back to the
   * config defaults their freshly remounted form starts from. */
  panel?: { width: number; height: number; frameRate: number } | undefined;
  /** W1: the Settings FPS policy (`usePrefillPolicy().fpsPolicy`), driving the
   * frame rate only. `panel` keeps {@link panel}'s fps (the config default when
   * absent); `project` seeds it
   * from the selection's project rate/scale (and is later overwritten off
   * `getEditInfo` by the caller); §3-13: `material` seeds it from the selected
   * object's OWN probed framerate, falling back to that same project rate/scale
   * when the material's fps can't be read. Every tier is snapped to a whole
   * frame rate by `modes/single/paramUtils.ts`'s `snapFrameRate` (台帳
   * §3-71/§3-72). */
  fpsPolicy: PrefillResolutionPolicy;
  /** Smart comfort marker (2026-08-31): the LOADED base model's engine family
   * (`useBaseModels().activeEngineFamily`), the acceleration settings
   * (EFFECTIVE ones — `AppShell`'s `accelerationControls.acceleration`) and
   * sage's availability. Together they pick the served comfort row whose
   * budget gives the SMART DURATION ceiling for a Single-screen prefill, so a
   * right-click's seeded length equals the marker the user then sees on the
   * form.
   *
   * All three are optional. The LTX token line needs all three effectively
   * present (the gate is on `acceleration`, which is the only one with no
   * meaningful default); the sequence-rows line (MiniMax H3, ledger §1-119 /
   * §1-276) needs `engineFamily` only, has nothing to do with the
   * acceleration settings, and also serves Chain-targeted seeds. Omit them —
   * as every pre-existing caller and test does — and the seed is derived
   * exactly as before, off `config.limits.spill_free_frames`. */
  engineFamily?: string | undefined;
  acceleration?: AccelerationSettings | undefined;
  sageAvailable?: boolean | null | undefined;
  /** The loaded transformer's weight class (`useBaseModels().activeWeightClass`,
   * §1-31), passed straight to `resolveComfortRow` beside
   * {@link engineFamily}. Optional; omitted or `""` = unknown (no
   * `weight_class` in the match). Every caller (`AppShell`, `SingleScreen`,
   * `ChainedScreen`) must hand in the same value (R-4). */
  weightClass?: string | undefined;
  /** §1-286: the loaded H3 transformer's quantisation kind
   * (`useBaseModels().activeQuantClass`), passed to
   * `resolveSequenceRowsBudget` so the seeded length matches the per-kind
   * rows line the marker shows. Optional; omitted or `null` = the default
   * line. Every caller must hand in the same value, like {@link weightClass}. */
  quantClass?: TransformerQuantClass | null | undefined;
  /** The LOADED base model's numeric rules
   * (`useBaseModels().activeConstraints`) — the frame grid and floor, the size
   * multiple, and a fixed frame rate (which wins over `fpsPolicy`). Applied to
   * Single-targeted intents, and (ledger §1-276) to Chain-targeted ones only
   * when the rules fix the chain's overlap (`chain` non-null — MiniMax H3,
   * whose Chained runs on its own 17n+5 grid at 24 fps); a Chain-targeted seed
   * keeps the LTX rules otherwise (the served LTX rules are not
   * `LTX_GENERATION_CONSTRAINTS` — their `maxNumFrames` is unbounded — so
   * reading them would move LTX's Chain seeds). Omitted = the LTX rules. Every
   * caller (`AppShell`, `SingleScreen`, `ChainedScreen`) must hand in the same
   * value. */
  constraints?: GenerationConstraints | undefined;
}

export interface PrefillSeed {
  /** The full `deriveGenerationParams` result (width/height + `notes`/`source`/
   * `clamped`), returned whole so the caller's existing references to
   * `derived.notes`/`derived.width`/… keep working unchanged. */
  derived: DerivedGenerationParams;
  /** The generation fps to seed the form with, or `undefined` to keep the
   * form's current value (the `panel` policy without a `panel`, or an
   * unresolvable rate/scale).
   * Always a whole frame rate in [1, 60] — see `frameRateSnappedFrom`. */
  frameRate: number | undefined;
  /** 台帳§3-71/§3-72: the RAW rate `frameRate` was snapped FROM, but only when
   * the snap actually changed the value (29.97 → 30 leaves `29.97…` here; a
   * project already at 30 leaves `undefined`). **Display only** — nothing
   * derives a number from it; `SingleScreen`/`ChainedScreen` turn it into the
   * one-shot "rounded to N fps" toast and nothing else reads it. Deliberately
   * the raw double, not a pre-formatted string, so the toast owns its own
   * formatting (`jobs/fpsConvert.ts`'s `formatFps`). */
  frameRateSnappedFrom: number | undefined;
  /** The DURATION (`num_frames`) to seed the form/reservation with, or
   * `undefined` to keep the form's current value (no DURATION policy for the intent,
   * or the policy couldn't resolve a value — e.g. an unknown material length or
   * an unresolvable comfort ceiling). */
  numFrames: number | undefined;
}

/**
 * Resolve a right-click generation's prefill seed (width/height/fps/DURATION)
 * from the routed selection + the Settings policy — the shared純関数 both the
 * form seed (`SingleScreen`/`ChainedScreen`) and the reservation length
 * (`AppShell`) call with the same inputs, so a placed provisional's ribbon
 * matches the remounted form.
 *
 * Composition: `deriveGenerationParams` (isICLora = either IC-LoRA intent,
 * useMaterialSize = `sizePolicy` ≠ `panel`) for width/height → the fps seed
 * (`panel`, or material → project → config default, see below) → the intent's
 * {@link DurationPolicy} + {@link materialDurationForIntent} fed
 * into `computeTargetNumFrames` (which owns the grid snap and the comfort-
 * ceiling clamp) → the one intent-specific DURATION cap
 * ({@link END_SOURCE_SEED_MAX_FRAMES}). `computeTargetNumFrames`'s `null` (policy
 * resolved nothing) is surfaced as `undefined` so the caller falls back to the
 * config default.
 */
export function resolvePrefillSeed(args: ResolvePrefillSeedArgs): PrefillSeed {
  const {
    intent,
    selection,
    config,
    sizePolicy,
    fpsPolicy,
    panel,
    engineFamily,
    acceleration,
    sageAvailable,
    weightClass,
    quantClass,
  } = args;
  // The served rules apply to the Single screen's seeds, and to a
  // Chain-targeted seed only on an engine that fixes the chain's overlap; any
  // other seed keeps LTX's (see the arg's doc comment).
  const singleTarget = targetModeForIntent(intent) === "single";
  const chainTarget = targetModeForIntent(intent) === "chained";
  const rules = singleTarget
    ? (args.constraints ?? LTX_GENERATION_CONSTRAINTS)
    : chainTarget && args.constraints?.chain != null
      ? args.constraints
      : LTX_GENERATION_CONSTRAINTS;
  const frameLimits = frameLimitsFor(rules, config.limits.max_num_frames);
  // Both IC-LoRA intents seed on the 128 grid. 台帳§1-15 W4 (2026-08-11): the
  // Chain-targeted one MUST be here — its material lands in Chain's reference
  // slot, which flips `useChainForm`'s size grid from 64 to 128 the moment the
  // upload turns ready. Seeding it on 64 would show one width/height at mount
  // and a re-snapped one a beat later, i.e. a size that visibly jumps on its own.
  const isICLora = intent === "reference-video" || intent === "reference-video-chain";
  // W1: the two axes are independent — width/height follow `sizePolicy`, the
  // fps follows `fpsPolicy`. `panel` on an axis ignores the material for that
  // axis and keeps the panel's value; `material`/`project` seed it from the
  // selection (`project` is later overwritten off `getEditInfo` by the caller).
  //
  // §3-13: the fps axis needs all THREE policies told apart, so it has no
  // `useMaterialFps` twin of the flag below — that boolean was true for both
  // `material` and `project` and so could never distinguish them. See the fps
  // seed further down.
  const useMaterialSize = sizePolicy !== "panel";
  const primarySel = selection.selected[0];
  // The `panel` policy's size goes in as `deriveGenerationParams`'s "project"
  // input — the same slot the config defaults take — so it is rounded onto the
  // intent's grid (128 for IC-LoRA) and clamped exactly like that fallback.
  const sizeFromPanel = sizePolicy === "panel" ? panel : undefined;

  const derivedRaw = deriveGenerationParams({
    mediaWidth: useMaterialSize ? (primarySel?.mediaWidth ?? null) : null,
    mediaHeight: useMaterialSize ? (primarySel?.mediaHeight ?? null) : null,
    projectWidth: sizeFromPanel?.width ?? config.generation_defaults.width,
    projectHeight: sizeFromPanel?.height ?? config.generation_defaults.height,
    isICLora,
    generalMultiple: rules.dimensionMultiple,
    limits: {
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
    },
  });
  // Under `panel` the size is the panel's own, so the source note ("no material
  // resolution, so the project's W×H") would be wrong — it is always
  // `notes[0]`, and only it is dropped; the rounding/clamp notes stay.
  const derived: DerivedGenerationParams =
    sizePolicy === "panel" ? { ...derivedRaw, notes: derivedRaw.notes.slice(1) } : derivedRaw;

  // fps seed (§3-13, contract v11) — three tiers, tried in order:
  //  1. `material`: the selected object's OWN framerate. Native probes it off the
  //     backing file and reports it RAW (`29.97`).
  //  2. the selection's project rate/scale — where `project` always lands, and
  //     where `material` falls back when the object has no readable fps (an older
  //     native build, a non-video object, or an unsupported container such as
  //     mkv/webm — all normal outcomes, not errors). Also RAW: an NTSC project is
  //     `30000/1001`, not 30.
  //  3. `undefined`: keep the form's value — a rate/scale that isn't
  //     resolvable either.
  // The `panel` policy skips all three and keeps the panel's own fps (already
  // a whole number — the form snaps every value it holds).
  //
  // 台帳§3-71/§3-72: BOTH value tiers are snapped to a whole frame rate by
  // `snapFrameRate` (the one source of truth, in `modes/single/paramUtils.ts`).
  // The material tier used to be the only snapped one — it isn't any more,
  // because an NTSC PROJECT reaches the request through tier 2 just as easily.
  // `snapFrameRate` also collapses every "no usable rate" shape to `undefined`,
  // which is exactly the fall-through signal each tier needs, so the old
  // `rate > 0 && scale > 0` guard is now inside it.
  const rawProjectFps =
    selection.rate > 0 && selection.scale > 0 ? selection.rate / selection.scale : undefined;
  const rawMaterialFps = fpsPolicy === "material" ? primarySel?.mediaFps : undefined;
  // Tier 1 only wins when it is USABLE, and `snapFrameRate` is exactly that test
  // (`undefined` for absent / 0 / negative / non-finite — every shape that means
  // "native told us nothing"), so it doubles as the guard here.
  const usableMaterialFps = snapFrameRate(rawMaterialFps) !== undefined ? rawMaterialFps : undefined;
  // The RAW rate the seed came from, kept beside the snapped one purely so the
  // toast can name it (see `PrefillSeed.frameRateSnappedFrom`).
  // An engine fixed at one frame rate overrides the fps
  // policy outright — there is nothing to seed and nothing to report.
  const fixedFps = fixedFrameRate(rules);
  const rawSeedFps =
    fixedFps !== null ? undefined : fpsPolicy === "panel" ? panel?.frameRate : (usableMaterialFps ?? rawProjectFps);
  const frameRate = fixedFps ?? snapFrameRate(rawSeedFps);
  // Only a snap that actually MOVED the value is worth reporting; a project or
  // material already on a whole frame rate must stay silent (no false positives).
  const frameRateSnappedFrom =
    frameRate !== undefined && rawSeedFps !== undefined && rawSeedFps !== frameRate ? rawSeedFps : undefined;

  const policy: DurationPolicy = DURATION_POLICY_BY_INTENT[intent] ?? "untouched";
  // The fps used to convert a material duration to frames — the same fps that
  // seeds the form. With no seed fps the form keeps the fps it holds: the
  // panel's (台帳 §1-114: Single is never remounted), else the config default
  // a freshly remounted Chained form starts from.
  const genFps = frameRate ?? panel?.frameRate ?? config.generation_defaults.frame_rate;
  const materialDurationSec = materialDurationForIntent(intent, selection);
  // §1-17 Retake: the selected range's length + the PROJECT's own fps, the two
  // inputs the `selectedRangeLength` policy converts into generation frames.
  // Both are spread away for every other intent, so no other flow's request
  // changes by a single field.
  const selectedRangeFrames = selectedRangeFramesForIntent(intent, selection);
  // ⚠ RAW, and it MUST stay raw — do NOT run this through `snapFrameRate`
  // (台帳§3-71/§3-72). This is not a value anybody generates at: it is the
  // PROJECT's own timebase, used by `deriveDuration.ts`'s `selectedRangeLength`
  // policy to convert the selected frame RANGE into real seconds. Snapping it
  // would mis-measure a 29.97fps project's selection by 0.1% — a Retake window
  // that lands on the wrong frames. The generation fps (`frameRate` above) is
  // the one that gets snapped; the two are deliberately different quantities
  // even though both come out of `rate/scale`. Pinned by a regression test in
  // `prefillSeed.test.ts`.
  const projectFps = rawProjectFps;

  // Smart comfort marker (2026-08-31): only the SINGLE screen's flows share
  // Create's LTX token line — a Chain-targeted intent's DURATION is a
  // per-CLIP length inside a chain, a different quantity, so on LTX it keeps
  // the legacy table. `acceleration` gates the LTX token line because a caller
  // that does not thread the Settings state cannot say what would effectively
  // run, and guessing would move seeds for callers that never opted in.
  //
  // Ledger §1-276: the sequence-rows line (MiniMax H3) is the exception — a
  // Chained clip's S is Single's formula at the clip's length, so a
  // Chain-targeted seed takes it too. With no rows line a Chain-targeted seed
  // keeps exactly the legacy table, as before.
  //
  // Computed against `derived.width/height` — the size this very seed is about
  // to apply — not the form's current size, which is what makes the seeded
  // DURATION and the marker that appears next to it agree.
  // An engine the served comfort table has no profile for
  // has NO comfort line — neither the smart one nor the legacy LTX lookup — so
  // the seed only snaps onto the grid; a sequence-rows engine (MiniMax H3,
  // ledger §1-119) takes its rows line, independent of the acceleration
  // settings, and never falls to the legacy lookup either
  // (`singleLegacySpillTable`). `acceleration` gates the LTX branch only.
  const rowsBudget =
    singleTarget || chainTarget ? resolveSequenceRowsBudget(config.limits, engineFamily, quantClass) : null;
  const spillFreeFrames = singleTarget
    ? singleLegacySpillTable(config.limits, engineFamily)
    : rowsBudget !== null
      ? null
      : config.limits.spill_free_frames;
  const smartCeiling =
    rowsBudget !== null
      ? h3ComfortFrames(
          {
            width: derived.width,
            height: derived.height,
            keyframeCount: seedKeyframeCount(intent),
            references: H3_NO_CONDITIONING.references,
          },
          rowsBudget,
          frameLimits.minNumFrames,
          frameLimits.maxNumFrames,
        )
      : singleTarget && acceleration && spillFreeFrames !== null
        ? (() => {
            const row = resolveComfortRow(
              config.limits,
              engineFamily,
              acceleration,
              sageAvailable ?? null,
              weightClass,
            );
            return row
              ? comfortFramesForBudget(
                  derived.width,
                  derived.height,
                  row.singleBudget,
                  frameLimits.minNumFrames,
                  frameLimits.maxNumFrames,
                  row.spatialFactor,
                  row.temporalFactor,
                )
              : null;
          })()
        : null;

  const targetNumFrames =
    computeTargetNumFrames({
      policy,
      width: derived.width,
      height: derived.height,
      spillFreeFrames,
      minNumFrames: frameLimits.minNumFrames,
      maxNumFrames: frameLimits.maxNumFrames,
      frameGrid: rules.frameGrid,
      ...(smartCeiling !== null ? { comfortCeilingFrames: smartCeiling } : {}),
      ...(materialDurationSec !== undefined ? { materialDurationSec } : {}),
      ...(selectedRangeFrames !== undefined ? { selectedRangeFrames } : {}),
      ...(projectFps !== undefined ? { projectFps } : {}),
      genFps,
    }) ?? undefined;

  // 素材（末尾）: the one intent whose seed gets a SECOND ceiling on top of the
  // policy's. `Math.min` on a value the policy already snapped onto the 8n+1 grid
  // is safe without a re-snap, because {@link END_SOURCE_SEED_MAX_FRAMES} is
  // itself on that grid (169 = 8x21 + 1) — `pxFromVLatent` produces nothing else.
  const numFrames =
    intent === "end-with-this" && targetNumFrames !== undefined
      ? Math.min(targetNumFrames, END_SOURCE_SEED_MAX_FRAMES)
      : targetNumFrames;

  return { derived, frameRate, frameRateSnappedFrom, numFrames };
}
