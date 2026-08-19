/**
 * Small pure fps/frame conversion helpers shared by the Join UI (`JoinControls`)
 * — the one place that converts a trim length expressed in frames into the
 * `source_tail_seconds` the join body wants, and compares the project fps to the
 * generated-video fps for the "frame rates did not match" note.
 *
 * Kept dependency-light and framework-free (mirroring the `prefillSeed.ts`
 * rate/scale idiom) so each helper is trivially unit-testable at its boundaries.
 */

import type { JobResponse } from "../api/types";

/** Real-time seconds spanned by `frames` at `fps`. Caller supplies a positive
 * fps (see {@link jobFrameRate}); `frames` of 0 yields 0. */
export function framesToSeconds(frames: number, fps: number): number {
  return frames / fps;
}

/** Whole frames spanned by `seconds` at `fps`, rounded to the nearest frame. */
export function secondsToFrames(seconds: number, fps: number): number {
  return Math.round(seconds * fps);
}

/** The project's generation fps from an `getEditInfo` `rate`/`scale` pair
 * (AviUtl2 stores fps as the rational `rate/scale`). Returns `null` when either
 * value is non-positive, i.e. the fps can't be resolved. */
export function editInfoFps(rate: number, scale: number): number | null {
  if (rate > 0 && scale > 0) return rate / scale;
  return null;
}

/** The generation fps a completed job ran at, read from its echoed per-clip
 * request (`JobResponse.request.frame_rate`, an untyped `Record` value). Falls
 * back to 24 when the field is missing or not a positive number, so callers
 * always get a usable divisor. */
export function jobFrameRate(job: JobResponse): number {
  const frameRate = job.request.frame_rate;
  return typeof frameRate === "number" && frameRate > 0 ? frameRate : 24;
}

/** Where the *joined* V2V clip should be inserted on the timeline: at the
 * source object's tail minus the portion actually kept after the tail-trim
 * (JOIN_FEATURE_RESEARCH.md §4.5). The joined clip starts with the kept tail of
 * the original source, so aligning its head to `sourceTail − keptLength` overlays
 * that tail exactly and lets the continuation extend past it.
 *
 * `frameStart`/`frameEnd` are the source object's INCLUSIVE frame range (native
 * `timeline.getSelection` semantics, so the real source length is
 * `frameEnd − frameStart + 1` frames and its tail-next frame is `frameEnd + 1`).
 * `trimmedSourceSeconds` is the join's `trimmed_source_seconds` (source seconds
 * dropped from the head by the tail-keep). Returns a frame clamped to be no
 * earlier than `frameStart` (a trim longer than the source can't push the head
 * before the source's own start). Pure. */
export function joinedInsertFrame(params: {
  frameStart: number;
  frameEnd: number;
  projectFps: number;
  trimmedSourceSeconds: number;
}): number {
  const { frameStart, frameEnd, projectFps, trimmedSourceSeconds } = params;
  const sourceRealSeconds = (frameEnd - frameStart + 1) / projectFps;
  const keptSeconds = Math.max(0, sourceRealSeconds - trimmedSourceSeconds);
  const frame = frameEnd + 1 - Math.round(keptSeconds * projectFps);
  return Math.max(frameStart, frame);
}

/** Formats an fps for display in the join note: at most two decimals, with any
 * trailing zeros dropped (30 -> "30", 29.97 -> "29.97", 23.976 -> "23.98").
 * `Number(...)` re-parses the fixed string so an integral fps loses its ".00". */
export function formatFps(fps: number): string {
  return String(Number(fps.toFixed(2)));
}
