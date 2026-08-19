/**
 * Pure, DOM-free grid math for the Create screen's keyframe timeline bar
 * (Docs/KEYFRAME_TIMELINE_DESIGN_NOTES.md). Keyframes may only land on the
 * server's `conditioning_frame_idx_multiple`/`grid_offset` grid (default
 * 8/1, see `keyframeUtils.ts`'s `snapFrameIdx`): `{0, offset, offset+multiple,
 * offset+2*multiple, …}` up to whatever the current `num_frames` allows.
 *
 * `num_frames` is normally `8n+1`, but while the user is free-typing into
 * the field it can transiently be an off-grid value, so every function here
 * first floors `numFrames` down to the grid via `maxPlaceablePosition`
 * rather than trusting it's already `8n+1`.
 *
 * Position `0` is an ordinary grid position — always the smallest one,
 * for any `numFrames` — not a dedicated slot. The UI can still make it
 * easy to land on by giving the bar's left edge a generous snap
 * threshold, but this module doesn't special-case it: bar-drag snapping
 * (`nearestGridPosition`), keyboard stepping, and collision resolution
 * all treat `0` like any other grid tick.
 */
import type { KeyframeItem } from "./keyframeUtils";

/** The largest frame index a keyframe can be placed at for the current
 * (possibly off-grid) `numFrames`. Floors `numFrames` to the grid first:
 * `effective = floor((numFrames-offset)/multiple)*multiple + offset`, then
 * subtracts one grid step. Can go negative for a `numFrames` too small to
 * fit even one non-zero grid point (e.g. `numFrames<=offset`); callers that
 * care (`validPositions`) treat `maxPlaceable < offset` as "no non-zero
 * slots available". */
export function maxPlaceablePosition(numFrames: number, multiple: number, offset: number): number {
  const effective = Math.floor((numFrames - offset) / multiple) * multiple + offset;
  return effective - multiple;
}

/** All frame indices a keyframe may occupy for the current `numFrames`,
 * ascending, always starting with `0`. Returns `[0]` alone when the grid is
 * too small to fit any non-zero position. */
export function validPositions(numFrames: number, multiple: number, offset: number): number[] {
  const maxPlaceable = maxPlaceablePosition(numFrames, multiple, offset);
  if (maxPlaceable < offset) return [0];

  const positions: number[] = [0];
  for (let pos = offset; pos <= maxPlaceable; pos += multiple) {
    positions.push(pos);
  }
  return positions;
}

/** Snaps an arbitrary frame (typically a pixel-to-frame conversion during a
 * bar drag) to the nearest grid position, including `0` — the UI can widen
 * the effective snap zone around the left edge if it wants dragging near
 * the start to reliably land on `0`. Ties break toward the smaller
 * position, so an exact tie between `0` and `offset` resolves to `0`. */
export function nearestGridPosition(frame: number, numFrames: number, multiple: number, offset: number): number {
  const candidates = validPositions(numFrames, multiple, offset);
  const first = candidates[0];
  if (first === undefined) return offset;

  let best = first;
  let bestDist = Math.abs(frame - best);
  for (let i = 1; i < candidates.length; i++) {
    const pos = candidates[i];
    if (pos === undefined) continue; // unreachable: i < candidates.length
    const dist = Math.abs(frame - pos);
    if (dist < bestDist) {
      best = pos;
      bestDist = dist;
    }
  }
  return best;
}

export interface ResolveLandingArgs {
  /** The grid position the drag/drop is aiming for; must already be a
   * member of `validPositions(numFrames, multiple, offset)`. */
  desired: number;
  /** Positions currently held by *other* pins — the pin being placed must
   * already be excluded by the caller. */
  occupied: ReadonlySet<number>;
  numFrames: number;
  multiple: number;
  offset: number;
  /** Fallback position (usually the pin's pre-drag position) returned when
   * every grid position is occupied. */
  origin: number;
}

/** Resolves where a dropped/placed pin actually lands when its desired
 * position is already taken: (a) if `desired` is free, use it; (b) else
 * walk forward (toward larger positions) through the grid, jumping over any
 * run of occupied positions, and land on the first free one; (c) if nothing
 * is free forward, walk backward from `desired` instead; (d) if the entire
 * grid is occupied, fall back to `origin`. */
export function resolveLanding(args: ResolveLandingArgs): number {
  const { desired, occupied, numFrames, multiple, offset, origin } = args;
  if (!occupied.has(desired)) return desired;

  const positions = validPositions(numFrames, multiple, offset);
  const idx = positions.indexOf(desired);
  if (idx === -1) return origin;

  for (let i = idx + 1; i < positions.length; i++) {
    const pos = positions[i];
    if (pos === undefined) continue; // unreachable: i < positions.length
    if (!occupied.has(pos)) return pos;
  }
  for (let i = idx - 1; i >= 0; i--) {
    const pos = positions[i];
    if (pos === undefined) continue; // unreachable: i >= 0 within bounds
    if (!occupied.has(pos)) return pos;
  }
  return origin;
}

/** Moves a pin one grid step in `direction` (arrow-key nudge), including
 * position `0`. If the adjacent position is occupied, keeps walking in the
 * same direction to jump over the occupied run and land on the first free
 * position. Does not wrap: if there's no free position in `direction`,
 * returns `current` unchanged. */
export function stepPin(
  current: number,
  direction: -1 | 1,
  occupied: ReadonlySet<number>,
  numFrames: number,
  multiple: number,
  offset: number,
): number {
  const positions = validPositions(numFrames, multiple, offset);
  const idx = positions.indexOf(current);
  if (idx === -1) return current;

  for (let i = idx + direction; i >= 0 && i < positions.length; i += direction) {
    const pos = positions[i];
    if (pos === undefined) continue; // unreachable: loop guard bounds i
    if (!occupied.has(pos)) return pos;
  }
  return current;
}

/** Picks the landing position for the "add keyframe" button: always after
 * the rightmost existing pin, never inserted before it (adding only ever
 * grows the tail). Returns `null` when there's no room left.
 *
 * `0` when there are no pins yet. Otherwise takes `lastPos = max(occupied)`,
 * rounds the midpoint between `lastPos` and the last frame to the nearest
 * non-zero grid position, and uses that as the desired spot. If that
 * desired spot isn't strictly after `lastPos` (e.g. the midpoint rounds
 * down onto `lastPos` itself), it's replaced with the first grid position
 * after `lastPos` instead. From there, only positions after `lastPos` are
 * searched (never before) for the first free one. */
export function nextAddPosition(
  occupied: ReadonlySet<number>,
  numFrames: number,
  multiple: number,
  offset: number,
): number | null {
  if (occupied.size === 0) return 0;

  const positions = validPositions(numFrames, multiple, offset);
  const lastPos = Math.max(...occupied);
  const mid = (lastPos + (numFrames - 1)) / 2;
  const desired = nearestGridPosition(mid, numFrames, multiple, offset);

  let startIdx: number;
  if (desired <= lastPos) {
    const lastIdx = positions.indexOf(lastPos);
    if (lastIdx === -1 || lastIdx + 1 >= positions.length) return null;
    startIdx = lastIdx + 1;
  } else {
    startIdx = positions.indexOf(desired);
    if (startIdx === -1) return null;
  }

  for (let i = startIdx; i < positions.length; i++) {
    const pos = positions[i];
    if (pos === undefined) continue; // unreachable: i < positions.length
    if (!occupied.has(pos)) return pos;
  }
  return null;
}

/** Whether the "add keyframe" button should be enabled — a thin wrapper
 * around `nextAddPosition` for call sites that only need the boolean. */
export function canAddKeyframe(occupied: ReadonlySet<number>, numFrames: number, multiple: number, offset: number): boolean {
  return nextAddPosition(occupied, numFrames, multiple, offset) !== null;
}

/** Keyframes that fall off the placeable grid after `numFrames` shrinks
 * (e.g. the user lowers the duration below where a pin sits). Compares
 * against `maxPlaceablePosition`, not `numFrames-1` directly, since the
 * placeable ceiling is grid-floored and usually stricter. `frameIdx===0`
 * always survives even when `maxPlaceable` goes negative (a `numFrames`
 * too small for any non-zero slot) — `0` is always the smallest grid
 * position, so it's explicitly exempted from the `> maxPlaceable`
 * comparison rather than relying on the arithmetic to work out for a
 * negative ceiling. */
export function findOverflowFrameIdxs(
  items: readonly KeyframeItem[],
  numFrames: number,
  multiple: number,
  offset: number,
): KeyframeItem[] {
  const maxPlaceable = maxPlaceablePosition(numFrames, multiple, offset);
  return items.filter((item) => item.frameIdx !== 0 && item.frameIdx > maxPlaceable);
}

/** Converts a frame index to a `[0, 1]` bar-position ratio for CSS
 * placement (`frame / (numFrames-1)`, clamped). Returns `0` when
 * `numFrames<=1` to avoid dividing by zero. */
export function positionToRatio(frame: number, numFrames: number): number {
  if (numFrames <= 1) return 0;
  return Math.min(1, Math.max(0, frame / (numFrames - 1)));
}

/** Inverse of `positionToRatio`: converts a `[0, 1]` bar-position ratio
 * (clamped) back into a real-valued frame. Callers that need an actual
 * placeable frame index should pipe the result through
 * `nearestGridPosition` themselves — this stays a raw, unsnapped
 * conversion so it composes with both drag-preview and grid-snap use. */
export function ratioToFrame(ratio: number, numFrames: number): number {
  const clamped = Math.min(1, Math.max(0, ratio));
  return clamped * (numFrames - 1);
}

/** Total number of grid positions (including `0`) available for the
 * current `numFrames` — the upper bound the "N/max" keyframe counter should
 * clamp against alongside the panel's own `maxItems`. */
export function placeableSlotCount(numFrames: number, multiple: number, offset: number): number {
  return validPositions(numFrames, multiple, offset).length;
}
