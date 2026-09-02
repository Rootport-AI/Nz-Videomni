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
 */

import type { AppConfig } from "../api/types";
import { pxFromVLatent } from "../modes/chained/chainUtils";
import { MIN_HEIGHT, MIN_NUM_FRAMES, MIN_WIDTH } from "../modes/single/defaultConfig";
import { snapFrameRate } from "../modes/single/paramUtils";
import { STAGE2_WINDOW_DEFAULT, STAGE2_WINDOW_PRESETS } from "../shell/tokenBudget";
import type { PrefillResolutionPolicy } from "../shell/PrefillPolicyContext";
import type { AccelerationSettings } from "../shell/accelerationSettings";
import { comfortFramesForBudget, resolveComfortRow } from "../shell/comfortTable";
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
 *    bound DURATION (IC-LoRA reference #2, A2V #3/#7) — the material's own
 *    length floored to the 8n+1 grid, never past the comfort ceiling.
 *
 * Intents absent from this table (e.g. `text-to-video`, `image-from-frame`) are
 * the keep-panel-state cursor items `AppShell` handles live via the published
 * form values, not a remount seed, so they never route through here.
 */
export const DURATION_POLICY_BY_INTENT: Readonly<Record<string, DurationPolicy>> = {
  "extend-video": "comfortCeiling", // #1 (Chain)
  "reference-video": "materialClampedToCeiling", // #2 (IC-LoRA)
  "video-audio-to-video": "materialClampedToCeiling", // #3 (A2V from a video's sound)
  "image-to-video": "comfortCeiling", // #4
  "image-to-clip-chain": "comfortCeiling", // #6 (Chain)
  "audio-to-video": "materialClampedToCeiling", // #7 (A2V from an audio file)
  "current-frame-to-clip-chain": "comfortCeiling", // #12 (Chain)
  // 台帳§1-16 長尺A2V (2026-08-10). Deliberately `comfortCeiling`, NOT the
  // `materialClampedToCeiling` their single-shot siblings #3/#7 use: this seed
  // sizes ONE CLIP of the chain, while the extracted audio's length is spread
  // over the WHOLE clip list by `audioFit.planAudioFit` (which re-lays the
  // cards the moment the wav is measured). Clamping clip 0 to the track's full
  // length would seed a single 60-second clip and then immediately be undone by
  // the auto-fit — so the honest seed is the resolution's comfort ceiling, the
  // same one every other Chain intent uses.
  "video-audio-to-long-a2v": "comfortCeiling", // 🎬 (Chain, 台帳§1-16)
  "audio-to-long-a2v": "comfortCeiling", // 🎵 (Chain, 台帳§1-16)
  // 台帳§1-15 W4 (2026-08-11). Deliberately `comfortCeiling`, NOT the
  // `materialClampedToCeiling` its single-shot sibling `reference-video` uses,
  // for the same reason the long-a2v pair above is: this seed sizes ONE CLIP of
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
 * W4 (trim追従): the IC-LoRA reference (#2), audio-to-video (#7) and
 * video-audio-to-video (#3) intents now ALL derive their length from the
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
    case "video-audio-to-video":
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

export interface ResolvePrefillSeedArgs {
  /** The routed sub-flow identifier (`GenerationPrefill.intent`). */
  intent: string;
  /** The file-path-completed timeline selection the seed derives from. */
  selection: TimelineSelection;
  config: AppConfig;
  /** W1: the Settings SIZE policy (`usePrefillPolicy().sizePolicy`), driving
   * width/height only. `defaults` ignores the material's real size;
   * `material`/`project` seed width/height from it (`project` is later
   * overwritten off `getEditInfo` by the caller). */
  sizePolicy: PrefillResolutionPolicy;
  /** W1: the Settings FPS policy (`usePrefillPolicy().fpsPolicy`), driving the
   * frame rate only. `defaults` keeps the config default fps; `project` seeds it
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
   * All three are optional and all three must effectively be present for the
   * smart path to run (the gate is on `acceleration`, which is the only one
   * with no meaningful default). Omit them — as every pre-existing caller and
   * test does — and the seed is derived exactly as before, off
   * `config.limits.spill_free_frames`. */
  engineFamily?: string | undefined;
  acceleration?: AccelerationSettings | undefined;
  sageAvailable?: boolean | null | undefined;
}

export interface PrefillSeed {
  /** The full `deriveGenerationParams` result (width/height + `notes`/`source`/
   * `clamped`), returned whole so the caller's existing references to
   * `derived.notes`/`derived.width`/… keep working unchanged. */
  derived: DerivedGenerationParams;
  /** The generation fps to seed the form with, or `undefined` to keep the
   * config default (the `defaults` policy, or an unresolvable rate/scale).
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
   * `undefined` to keep the config default (no DURATION policy for the intent,
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
 * useMaterialSize = `sizePolicy` ≠ `defaults`) for width/height → the three-tier
 * fps seed (material → project → config default, see below) → the intent's
 * {@link DurationPolicy} + {@link materialDurationForIntent} fed
 * into `computeTargetNumFrames` (which owns the 8n+1 snap and the comfort-
 * ceiling clamp) → the one intent-specific DURATION cap
 * ({@link END_SOURCE_SEED_MAX_FRAMES}). `computeTargetNumFrames`'s `null` (policy
 * resolved nothing) is surfaced as `undefined` so the caller falls back to the
 * config default.
 */
export function resolvePrefillSeed(args: ResolvePrefillSeedArgs): PrefillSeed {
  const { intent, selection, config, sizePolicy, fpsPolicy, engineFamily, acceleration, sageAvailable } = args;
  // Both IC-LoRA intents seed on the 128 grid. 台帳§1-15 W4 (2026-08-11): the
  // Chain-targeted one MUST be here — its material lands in Chain's reference
  // slot, which flips `useChainForm`'s size grid from 64 to 128 the moment the
  // upload turns ready. Seeding it on 64 would show one width/height at mount
  // and a re-snapped one a beat later, i.e. a size that visibly jumps on its own.
  const isICLora = intent === "reference-video" || intent === "reference-video-chain";
  // W1: the two axes are independent — width/height follow `sizePolicy`, the
  // fps follows `fpsPolicy`. `defaults` on an axis ignores the material for that
  // axis; `material`/`project` seed it from the selection (`project` is later
  // overwritten off `getEditInfo` by the caller).
  //
  // §3-13: the fps axis needs all THREE policies told apart, so it has no
  // `useMaterialFps` twin of the flag below — that boolean was true for both
  // `material` and `project` and so could never distinguish them. See the fps
  // seed further down.
  const useMaterialSize = sizePolicy !== "defaults";
  const primarySel = selection.selected[0];

  const derived = deriveGenerationParams({
    mediaWidth: useMaterialSize ? (primarySel?.mediaWidth ?? null) : null,
    mediaHeight: useMaterialSize ? (primarySel?.mediaHeight ?? null) : null,
    projectWidth: config.generation_defaults.width,
    projectHeight: config.generation_defaults.height,
    isICLora,
    limits: {
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
    },
  });

  // fps seed (§3-13, contract v11) — three tiers, tried in order:
  //  1. `material`: the selected object's OWN framerate. Native probes it off the
  //     backing file and reports it RAW (`29.97`).
  //  2. the selection's project rate/scale — where `project` always lands, and
  //     where `material` falls back when the object has no readable fps (an older
  //     native build, a non-video object, or an unsupported container such as
  //     mkv/webm — all normal outcomes, not errors). Also RAW: an NTSC project is
  //     `30000/1001`, not 30.
  //  3. `undefined`: keep the config default — the `defaults` policy, or a
  //     rate/scale that isn't resolvable either.
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
  const rawSeedFps = fpsPolicy === "defaults" ? undefined : (usableMaterialFps ?? rawProjectFps);
  const frameRate = snapFrameRate(rawSeedFps);
  // Only a snap that actually MOVED the value is worth reporting; a project or
  // material already on a whole frame rate must stay silent (no false positives).
  const frameRateSnappedFrom =
    frameRate !== undefined && rawSeedFps !== undefined && rawSeedFps !== frameRate ? rawSeedFps : undefined;

  const policy: DurationPolicy = DURATION_POLICY_BY_INTENT[intent] ?? "untouched";
  // The fps used to convert a material duration to frames — the same fps that
  // seeds the form, falling back to the config default when there's no seed fps.
  const genFps = frameRate ?? config.generation_defaults.frame_rate;
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
  // Create's per-clip comfort line — a Chain-targeted intent's DURATION is a
  // per-CLIP length inside a chain, a different quantity, so it keeps the
  // legacy table. `acceleration` gates the whole thing because a caller that
  // does not thread the Settings state cannot say what would effectively run,
  // and guessing would move seeds for callers that never opted in.
  //
  // Computed against `derived.width/height` — the size this very seed is about
  // to apply — not the form's current size, which is what makes the seeded
  // DURATION and the marker that appears next to it agree.
  const smartCeiling =
    targetModeForIntent(intent) === "single" && acceleration
      ? (() => {
          const row = resolveComfortRow(config.limits, engineFamily, acceleration, sageAvailable ?? null);
          return row
            ? comfortFramesForBudget(
                derived.width,
                derived.height,
                row.singleBudget,
                MIN_NUM_FRAMES,
                config.limits.max_num_frames,
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
      spillFreeFrames: config.limits.spill_free_frames,
      minNumFrames: MIN_NUM_FRAMES,
      maxNumFrames: config.limits.max_num_frames,
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
