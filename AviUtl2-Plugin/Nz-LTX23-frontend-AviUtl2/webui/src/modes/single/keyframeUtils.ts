/**
 * Pure helpers for the M4 I2V keyframe (conditioning image) panel.
 * Docs/API_REFERENCE.md §5.1: `ConditioningImage` is `{image_id, frame_idx,
 * strength}` (the WebUI never sends `crf`). `frame_idx==0` is the leading
 * frame (latent-replace) and is sent as-is; anything greater is snapped by
 * the *server* to the `conditioning_frame_idx_multiple`/`grid_offset` grid
 * (default 8/1: `(frame_idx-1)//8*8+1`) and range-clamped — the WebUI submits
 * the user's natural value and only needs to *preview* that snap here so the
 * "will snap to N" hint under the frame-index input matches what the server
 * will actually do.
 */
import type { ConditioningImage } from "../../api/types";

/** One keyframe card's state, owned by `useKeyframes`. `id` is a client-only
 * key (never sent to the backend); `imageId` is null until the upload
 * completes. */
export interface KeyframeItem {
  id: string;
  /** Local path returned by `timeline.captureFrame`/`ui.pickFile`. */
  filePath: string;
  fileName: string;
  /** `backend.uploadFile`'s `image_id`, null while uploading or on error. */
  imageId: string | null;
  thumbnailDataUrl: string | null;
  frameIdx: number;
  strength: number;
  /** Keyframe timeline rework (2026-07-18): `"empty"` is a card with no image
   * attached yet (added via `useKeyframes.addEmpty`, filled in later via
   * `replaceFile`) — `buildConditioningImages` below already excludes it
   * automatically since only `"ready"` items are ever included. */
  status: "uploading" | "ready" | "error" | "empty";
  errorCode: string | null;
}

export const DEFAULT_STRENGTH = 0.8;
export const STRENGTH_STEP = 0.05;
export const STRENGTH_MIN = 0.0;
export const STRENGTH_MAX = 1.0;

/** Last path component after the final `/` or `\`. Mirrors native's
 * `FileNameFromPath` (native/src/bridge_core.h) so the WebUI never needs to
 * re-derive it differently. */
export function fileNameFromPath(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(idx + 1) : path;
}

/** Previews the server-side 8n+1 grid snap for a `frame_idx` value
 * (Docs/API_REFERENCE.md §5.1). `0` always stays `0` (leading frame); a
 * negative input is treated as `0` since the field's own min is 0. */
export function snapFrameIdx(frameIdx: number, multiple: number, gridOffset: number): number {
  if (frameIdx <= 0) return 0;
  return Math.floor((frameIdx - 1) / multiple) * multiple + gridOffset;
}

/** True when `frameIdx` falls outside the valid `[0, numFrames-1]` range for
 * the form's current `num_frames` — the card should show a warning border. */
export function isFrameIdxOutOfRange(frameIdx: number, numFrames: number): boolean {
  return frameIdx < 0 || frameIdx > numFrames - 1;
}

/** Clamps a strength value into `[STRENGTH_MIN, STRENGTH_MAX]`. */
export function clampStrength(value: number): number {
  return Math.min(STRENGTH_MAX, Math.max(STRENGTH_MIN, value));
}

/** Items sorted by ascending `frame_idx`, for display order only — the
 * payload sent to `/generate` doesn't care about array order. */
export function sortByFrameIdx(items: KeyframeItem[]): KeyframeItem[] {
  return [...items].sort((a, b) => a.frameIdx - b.frameIdx);
}

/** Builds the `conditioning_images` array for a `GenerateRequest` from the
 * panel's current items. Only `"ready"` items with a resolved `imageId` are
 * included — items still `"uploading"` are meant to block generation
 * entirely (see `useKeyframes.isUploading`), and `"error"` items have no
 * `image_id` to send. Returns `[]` (never omits) so callers decide whether to
 * include the key at all (T2V requests omit it entirely). */
export function buildConditioningImages(items: KeyframeItem[]): ConditioningImage[] {
  return items
    .filter((item): item is KeyframeItem & { imageId: string } => item.status === "ready" && item.imageId !== null)
    .map((item) => ({
      image_id: item.imageId,
      frame_idx: item.frameIdx,
      strength: item.strength,
    }));
}
