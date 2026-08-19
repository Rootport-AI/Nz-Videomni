/**
 * The DURATION (`num_frames`) decision seam for right-click prefill —
 * the follow-up to `deriveGenerationParams.ts`'s width/height seam, which
 * deliberately left `num_frames` "a separate, not-yet-decided design
 * question" (see that file's docstring, §7-3-B options 1/2). This module is
 * that decision, kept as its own file (rather than folded into
 * `deriveGenerationParams.ts`) since it leans on a different input shape
 * (`spillFreeFrames`/material duration vs. resolution) and has its own
 * three-way policy split.
 *
 * Three policies, one per right-click flow family:
 *  - `"comfortCeiling"`: flows with no natural source duration (T2V/I2V-style)
 *    default DURATION to the resolution's "comfortable" `num_frames` ceiling
 *    (`AppConfig.limits.spill_free_frames`, see `spillUtils.ts`'s
 *    `resolveSpillFreeFrames`) — always, regardless of whatever DURATION
 *    already held, since there is no source material to respect instead.
 *  - `"materialClampedToCeiling"`: flows with a source clip (V2V continuation,
 *    IC-LoRA reference) must never suggest a DURATION longer than the source
 *    itself, so the comfort ceiling only ever clamps it shorter.
 *  - `"untouched"`: flows where DURATION isn't part of the prefill contract at
 *    all — the caller leaves the form's existing value alone.
 *
 * Pure and dependency-free apart from `resolveSpillFreeFrames`
 * (`spillUtils.ts`, imported not modified) and `snapNumFrames`
 * (`paramUtils.ts`, ditto) — both already the single source of truth for
 * their respective pieces of this arithmetic, so this module composes them
 * rather than re-deriving.
 */

import { snapNumFrames } from "../modes/single/paramUtils";
import { resolveSpillFreeFrames } from "../modes/single/spillUtils";

/**
 * Which rule decides right-click prefill's target `num_frames`:
 *  - `"comfortCeiling"`: always the resolution's comfort ceiling (ignores any
 *    existing/current DURATION and any source material length).
 *  - `"materialClampedToCeiling"`: the source material's own length, floored
 *    onto the 8n+1 grid and never allowed past the comfort ceiling.
 *  - `"selectedRangeLength"` (§1-17 Retake): the user's SELECTED FRAME RANGE,
 *    converted from project frames to generation frames. No 8n+1 snap, no
 *    comfort ceiling — the retake's real window is decided elsewhere.
 *  - `"untouched"`: DURATION is not part of this prefill; always yields
 *    `null`.
 */
export type DurationPolicy =
  | "comfortCeiling"
  | "materialClampedToCeiling"
  | "selectedRangeLength"
  | "untouched";

export interface ComputeTargetNumFramesArgs {
  policy: DurationPolicy;
  /** Target resolution DURATION is being derived for (the comfort ceiling is
   * resolution-dependent, `resolveSpillFreeFrames`'s nearest-area lookup). */
  width: number;
  height: number;
  /** `AppConfig.limits.spill_free_frames`, or `null` when unavailable (e.g. a
   * config that hasn't loaded yet) — treated the same as an empty map. */
  spillFreeFrames: Record<string, number> | null;
  /** `limits.minNumFrames`/`limits.maxNumFrames` (already valid 8n+1 values —
   * see `paramUtils.snapNumFrames`'s own assumption). Both policies that can
   * return non-null clamp into this range. */
  minNumFrames: number;
  maxNumFrames: number;
  /** The source material's real-time duration in seconds — required (and
   * only meaningful) for `"materialClampedToCeiling"`. */
  materialDurationSec?: number;
  /** The backend's fixed generation frame rate (`GEN_FPS` in
   * `menuSelection.ts`) used to convert `materialDurationSec` to a frame
   * count — required for `"materialClampedToCeiling"` AND for
   * `"selectedRangeLength"` (both convert a real-time quantity into
   * GENERATION frames). */
  genFps?: number;
  /** §1-17 Retake: the user's selected frame range length, in **PROJECT**
   * frames (`timeline/selectionRange.ts`'s `selectionRangeFrames`). Required
   * (and only meaningful) for `"selectedRangeLength"`. */
  selectedRangeFrames?: number;
  /** §1-17 Retake: the PROJECT's own frame rate (`selection.rate /
   * selection.scale`). Required (and only meaningful) for
   * `"selectedRangeLength"` — without it the project-frame count above cannot
   * be turned into a duration, and a raw frame count would be wrong by the
   * ratio of the two rates. */
  projectFps?: number;
}

/**
 * Decides the right-click prefill's target `num_frames` per {@link
 * DurationPolicy}, or `null` when the caller should leave DURATION alone
 * (either `policy === "untouched"`, or the policy's required inputs aren't
 * resolvable yet — see each branch below).
 *
 *  - `"untouched"`: always `null`.
 *  - `"comfortCeiling"`: `null` if the comfort ceiling can't be resolved
 *    (`spillFreeFrames` is `null`/empty/has no parseable keys — see
 *    `resolveSpillFreeFrames`); otherwise the ceiling snapped onto the 8n+1
 *    grid and clamped to `[minNumFrames, maxNumFrames]`. Never looks at a
 *    "current" DURATION — the same width/height always yields the same
 *    result, whether that raises or lowers whatever DURATION held before.
 *  - `"materialClampedToCeiling"`: `null` if `materialDurationSec`/`genFps`
 *    are missing (or `genFps` isn't positive) — there is nothing to clamp
 *    without a source length. Otherwise: floor `materialDurationSec *
 *    genFps` onto the 8n+1 grid via {@link floorToFrameGrid} (never rounds
 *    UP past the material's real length), then take the smaller of that and
 *    the comfort ceiling (when the ceiling resolves; when it doesn't, the
 *    material-derived value stands on its own), and finally raise the result
 *    to `minNumFrames` if it landed below that floor — this last step can
 *    push the result back above the comfort ceiling, since a DURATION below
 *    the form's own allowed minimum is never a valid prefill regardless of
 *    material length.
 */
export function computeTargetNumFrames(args: ComputeTargetNumFramesArgs): number | null {
  const {
    policy,
    width,
    height,
    spillFreeFrames,
    minNumFrames,
    maxNumFrames,
    materialDurationSec,
    genFps,
    selectedRangeFrames,
    projectFps,
  } = args;

  if (policy === "untouched") return null;

  // §1-17 Retake: the seed IS the user's selected range, so neither the comfort
  // ceiling nor the material length has any say — this branch returns before
  // either is consulted.
  //
  // THE UNIT CONVERSION IS MANDATORY. `selectedRangeFrames` is counted on the
  // AviUtl2 PROJECT's timeline (30fps, say) while `num_frames` is always
  // counted at the GENERATION frame rate (24fps, say) — native itself converts
  // the other way when it draws the ribbon (`timeline_math.cpp`'s
  // `× project_fps / gen_fps`). Handing the raw project-frame count through
  // would make the placed ribbon 25% too long at that pair of rates.
  //
  // No 8n+1 snap here, deliberately: this value only seeds the provisional
  // reservation's ribbon so it matches what the user just selected. The real
  // window (8n+1, clamped to [73,169]) is decided by
  // `timeline/retakeWindow.ts`'s `resolveRetakeWindow`, and the reservation is
  // re-placed with THAT length when Generate is pressed.
  if (policy === "selectedRangeLength") {
    if (selectedRangeFrames === undefined || !(selectedRangeFrames > 0)) return null;
    if (projectFps === undefined || !(projectFps > 0)) return null;
    if (genFps === undefined || !(genFps > 0)) return null;
    const frames = Math.round((selectedRangeFrames / projectFps) * genFps);
    return Math.min(Math.max(frames, minNumFrames), maxNumFrames);
  }

  const ceiling = spillFreeFrames ? resolveSpillFreeFrames(spillFreeFrames, width, height) : null;

  if (policy === "comfortCeiling") {
    if (ceiling === null) return null;
    return snapNumFrames(ceiling, minNumFrames, maxNumFrames);
  }

  // policy === "materialClampedToCeiling"
  if (materialDurationSec === undefined || genFps === undefined || genFps <= 0) return null;

  const rawFrames = Math.floor(materialDurationSec * genFps);
  let target = floorToFrameGrid(rawFrames);
  if (ceiling !== null) target = Math.min(target, ceiling);
  if (target < minNumFrames) target = minNumFrames;
  return target;
}

/**
 * Floors `frames` onto the 8n+1 `num_frames` grid (n >= 1): the largest valid
 * grid value that does not exceed `frames`, with `9` (n=1) as the absolute
 * floor for any `frames < 9`. Deliberately does NOT clamp against a caller
 * `minNumFrames`/`maxNumFrames` — that's `computeTargetNumFrames`'s job — so
 * this stays a small, total, single-purpose helper like `paramUtils.ts`'s own
 * `snapNumFrames` (round-to-nearest) counterpart.
 *
 * Same formula as `chainUtils.ts`'s `rawFramesForAudio` grid step (`((floor
 * (dur*fps) - 1) // 8) * 8 + 1`, itself a copy of the backend's
 * `handlers.suggest_frames_for_audio`) — kept as its own tiny function here
 * rather than imported, since that one is bundled with A2V-specific duration
 * math this module has no use for.
 */
export function floorToFrameGrid(frames: number): number {
  const n = Math.max(1, Math.floor((frames - 1) / 8));
  return n * 8 + 1;
}
