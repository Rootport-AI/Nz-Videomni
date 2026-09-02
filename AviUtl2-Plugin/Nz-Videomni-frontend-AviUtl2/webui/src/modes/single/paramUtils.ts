/**
 * Pure helpers enforcing the "frozen constraints" from
 * Docs/API_REFERENCE.md §5.1 that the Create form must never violate:
 * width/height are multiples of 64, num_frames is 8n+1. Kept dependency-free
 * so they're trivially unit-testable and reusable by both the form and the
 * "Get size from AviUtl2" button.
 *
 * ⚠ Not everything here is a §5.1 frozen constraint: {@link snapFrameRate}'s
 * "whole frame rates only" rule is a UI POLICY this client chose (台帳
 * §3-71/§3-72), not something the API demands. Its doc comment spells the
 * distinction out.
 */

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Rounds to the nearest multiple of `multiple`, then clamps into [min, max].
 * Assumes `min`/`max` are themselves multiples of `multiple` (true for all
 * callers here: 256/1920 and 128/1088 are both multiples of 64). */
export function roundToMultiple(value: number, multiple: number, min: number, max: number): number {
  const rounded = Math.round(value / multiple) * multiple;
  return clamp(rounded, min, max);
}

/** Rounds *up* to the nearest multiple of `multiple`, then clamps into
 * [min, max]. Used by "Get size from AviUtl2", which must never request a
 * size smaller than the edit's actual resolution. */
export function ceilToMultiple(value: number, multiple: number, min: number, max: number): number {
  const rounded = Math.ceil(value / multiple) * multiple;
  return clamp(rounded, min, max);
}

/** Rounds *down* to the nearest multiple of `multiple`, then clamps into
 * [min, max]. Used where a value must never exceed a limit that is not
 * itself a clean multiple (e.g. deriving a resolution seam's upper bound). */
export function floorToMultiple(value: number, multiple: number, min: number, max: number): number {
  const rounded = Math.floor(value / multiple) * multiple;
  return clamp(rounded, min, max);
}

/** Snaps `raw` to the nearest valid `num_frames` value (8n+1, n >= 1),
 * clamped to [min, max]. `min`/`max` are expected to already be valid 8n+1
 * values (9 and `limits.max_num_frames`, both satisfy this in practice). */
export function snapNumFrames(raw: number, min: number, max: number): number {
  const minN = Math.round((min - 1) / 8);
  const maxN = Math.round((max - 1) / 8);
  const n = clamp(Math.round((raw - 1) / 8), minN, maxN);
  return n * 8 + 1;
}

/** True iff `value` already satisfies the num_frames=8n+1 constraint. */
export function isValidNumFrames(value: number): boolean {
  return Number.isInteger(value) && value >= 1 && (value - 1) % 8 === 0;
}

/**
 * The generation frame rate's inclusive lower bound — and, with
 * {@link FRAME_RATE_MAX}, the single source of truth for every fps field's
 * `min`/`max` (Create's + Chain's shared `FrameRateSeedFields`, Retake's and
 * Outpainting's own inline inputs) as well as {@link snapFrameRate}'s clamp.
 *
 * ⚠ This range IS a frozen server constraint (`api/`'s `Field(ge=1.0, le=60.0)`
 * — a value outside it is a 422), unlike the INTEGER part, which is a UI policy
 * only; see {@link snapFrameRate}.
 */
export const FRAME_RATE_MIN = 1;
/** The generation frame rate's inclusive upper bound — see {@link FRAME_RATE_MIN}. */
export const FRAME_RATE_MAX = 60;

/**
 * The fps every form's lazy initializer falls back to when its seed resolves to
 * nothing at all (no right-click seed, and a config value that isn't usable).
 *
 * ⚠ **Cross-reference — keep these two in step**: this is deliberately the same
 * number as `modes/single/defaultConfig.ts`'s
 * `FALLBACK_APP_CONFIG.generation_defaults.frame_rate` (`24.0`, at line 44 of
 * that file). It is duplicated here rather than imported so this module stays
 * dependency-free (see the header), which means changing the config default
 * without changing this constant would leave the two disagreeing — a form whose
 * seed failed would then land on a different fps than a form that had no seed.
 * If you change one, change the other.
 */
export const FRAME_RATE_FALLBACK = 24;

/**
 * Snaps a frame rate to the integer the generation will actually run at:
 * `Math.round` then clamped into [{@link FRAME_RATE_MIN}, {@link FRAME_RATE_MAX}].
 * Returns `undefined` for every "no usable rate" shape at once — `null`,
 * `undefined`, a non-finite number, or a non-positive one — which is the signal
 * each caller turns into its own fallback (a seed falls through to the next
 * tier; a form setter falls back to {@link FRAME_RATE_MIN}).
 *
 * **Why integers (台帳 §3-71 / §3-72).** A non-integer fps that reaches a
 * generation request breaks two things downstream, both of which the backend
 * fix would have had to reach through frozen code:
 *  - §3-71: Chained's stage-2 audio tile re-assembly check accumulates rounding
 *    error and rejects perfectly ordinary clip lists with a 422 (`[257, 257]` at
 *    29.97 is the headline case — fine at 24 or 30).
 *  - §3-72: the mp4 writer truncates the fps to an integer, so a 29.97 request
 *    produces a file whose audio drifts further out of sync the longer it runs.
 * Snapping at every CLIENT entry point closes both reproduction paths without
 * touching the server.
 *
 * ⚠ **This is a UI policy, not an API constraint** — deliberately unlike the
 * width/height/`num_frames` rules this module's header describes, which are the
 * frozen `Docs/API_REFERENCE.md` §5.1 contract. The server accepts any float in
 * [1, 60]; we choose to send only whole numbers. The RANGE, on the other hand,
 * really is the server's (see {@link FRAME_RATE_MIN}), which is why out-of-range
 * input is clamped here rather than rejected.
 *
 * `Math.round` alone is deliberate: every frame rate that exists in practice
 * (29.97 → 30, 23.976 → 24, 59.94 → 60) lands on its intended integer in one
 * step, so a tolerance band would only add a branch for the values that miss.
 * (Note the sibling implementations in `mcp_server/` and `gradio_ui/` round the
 * same way but do NOT clamp — they pass out-of-range values through to the
 * server's own 422. Don't write a parity test that assumes otherwise.) Pure.
 */
export function snapFrameRate(fps: number | null | undefined): number | undefined {
  if (fps === null || fps === undefined || !Number.isFinite(fps) || fps <= 0) return undefined;
  return clamp(Math.round(fps), FRAME_RATE_MIN, FRAME_RATE_MAX);
}

/** True iff `value` is a positive multiple of 64. */
export function isValidDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0 && value % 64 === 0;
}

export function framesToSeconds(numFrames: number, frameRate: number): number {
  if (frameRate <= 0) return 0;
  return numFrames / frameRate;
}

/** Formats a bare seconds value as `12.3s` (one decimal), for the IC-LoRA /
 * A2V source cards' duration readout. NOTE: distinct from
 * `KeyframeTimeline.tsx`'s own local `formatSeconds` (which serves the pin
 * labels) — don't conflate the two; this one is the whole-file-length label. */
export function formatSecondsLabel(seconds: number): string {
  return `${seconds.toFixed(1)}s`;
}

/** Formats "N frames ≈ X.X s @ fps" for the duration hint under the frames slider. */
export function formatDurationHint(numFrames: number, frameRate: number): string {
  const seconds = framesToSeconds(numFrames, frameRate);
  return `${numFrames} frames ≈ ${seconds.toFixed(1)}s @ ${frameRate}fps`;
}

export function isValidPrompt(prompt: string): boolean {
  return prompt.length >= 1 && prompt.length <= 2000;
}

/** True iff `nativeEvent` represents a stepper/slider-driven change on a
 * width/height `<input type="number">` (spinner arrow click, ↑/↓ key, or a
 * range slider's `input` event) rather than manual keyboard entry — the
 * former should snap to the nearest valid multiple, the latter should pass
 * through untouched while the user is mid-typing. In Chromium (WebView2's
 * engine), stepper/slider `input` events either aren't `InputEvent`s or have
 * a falsy `inputType`, while manual text edits carry a truthy `inputType`
 * such as `"insertText"` or `"deleteContentBackward"`. */
export function isStepEvent(nativeEvent: Event): boolean {
  const inputType = (nativeEvent as InputEvent).inputType;
  return inputType === undefined || inputType === null || inputType === "";
}

/** True iff `value` is an integer multiple of `multiple` within [min, max].
 * Used as a final send-time validation gate for width/height. */
export function isDimensionOnGrid(value: number, multiple: number, min: number, max: number): boolean {
  return Number.isInteger(value) && value % multiple === 0 && value >= min && value <= max;
}

/** True iff `value` is a valid `num_frames` (8n+1, via {@link isValidNumFrames})
 * within `[min, max]`. Used as a final send-time validation gate for
 * `num_frames` (Create's single field, and each of Chain's per-clip
 * durations) — the same "free-typed manual entry passes through unsnapped,
 * `isValid` gates the grid at submit time" role {@link isDimensionOnGrid}
 * plays for width/height. `min`/`max` are always caller-supplied (never
 * hardcoded here) since Create and Chain draw them from different config
 * sources (`limits.minNumFrames/maxNumFrames` vs. `MIN_CLIP_NUM_FRAMES`/
 * `MAX_CLIP_NUM_FRAMES`). */
export function isNumFramesOnGrid(value: number, min: number, max: number): boolean {
  return isValidNumFrames(value) && value >= min && value <= max;
}
