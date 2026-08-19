/**
 * Pure helpers enforcing the "frozen constraints" from
 * Docs/API_REFERENCE.md §5.1 that the Create form must never violate:
 * width/height are multiples of 64, num_frames is 8n+1. Kept dependency-free
 * so they're trivially unit-testable and reusable by both the form and the
 * "Get size from AviUtl2" button.
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
