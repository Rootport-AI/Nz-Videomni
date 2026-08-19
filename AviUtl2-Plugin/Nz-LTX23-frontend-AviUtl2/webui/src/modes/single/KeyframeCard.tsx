import { useEffect, useState } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import { useFileDrop } from "../../shell/useFileDrop";
import { maxPlaceablePosition, resolveLanding } from "./keyframeGrid";
import { snapFrameIdx, STRENGTH_MAX, STRENGTH_MIN, STRENGTH_STEP, type KeyframeItem } from "./keyframeUtils";
import "./KeyframeCard.css";

/** Extensions the card accepts, both for a drop and (conceptually) the file
 * picker. Shared with `useFileDrop` below so a dropped file's extension is
 * validated against the same set. */
const ACCEPTED_IMAGE_EXTS = ["png", "jpg", "jpeg", "webp"] as const;

export interface KeyframeCardProps {
  item: KeyframeItem;
  /** Current form `num_frames` — bounds both the FRAME input's `max` and the
   * grid-snap preview (`snapToPlaceableGrid` below). */
  numFrames: number;
  /** Current form `frame_rate`, purely for the optional "Xs" seconds hint
   * next to the FRAME input — a `<=0` value just suppresses that hint. */
  frameRate: number;
  /** `config.limits.conditioning_frame_idx_multiple` (default 8). */
  multiple: number;
  /** `config.limits.conditioning_keyframe_grid_offset` (default 1). */
  offset: number;
  /** `frameIdx`s held by every *other* card — this card's own position must
   * already be excluded by the caller (`keyframeGrid.resolveLanding`'s own
   * contract). */
  occupiedOthers: ReadonlySet<number>;
  /** Disabled while a generation is in flight/reserved, mirroring the rest
   * of the form's fields. */
  disabled: boolean;
  /** Fired once a FRAME edit is committed (blur/Enter) with the fully
   * resolved landing position — never the raw/unsnapped typed value. */
  onCommitFrameIdx: (id: string, frameIdx: number) => void;
  onStrengthChange: (id: string, value: number) => void;
  onRemove: (id: string) => void;
  /** Fired when an image is dropped onto the card (anywhere on it) — for both
   * a plain image swap and filling in an `"empty"` card's first image. */
  onReplaceFile: (id: string, filePath: string, fileName: string) => void;
  /** The card's own 📁 "choose file..." button — wiring (the actual
   * `ui.pickFile` round trip, then `onReplaceFile`) is the caller's job; this
   * component only raises the intent. */
  onPickFileFor: (id: string) => void;
  /** The card's own 📸 "capture current frame" button — captures the timeline's
   * current frame straight into *this* card (never adds a new one). Wiring is
   * the caller's job (`useKeyframes.captureIntoCard`). */
  onCaptureInto: (id: string) => void;
  /** True while a 📸 capture round trip for any card is in flight — disables
   * the capture button so a second capture can't be started mid-flight. */
  isCapturing: boolean;
  /** True while a 📁 file-pick round trip for any card is in flight — disables
   * the choose-file button so a second pick can't be started mid-flight. */
  isPicking: boolean;
}

/** Snaps a raw (possibly off-grid, out-of-range, or negative) typed FRAME
 * value to a placeable grid position: `<=0` stays `0`; otherwise previews
 * the server's 8n+1 snap (`snapFrameIdx`) and then clamps down to the
 * largest position the current `numFrames` can still hold
 * (`keyframeGrid.maxPlaceablePosition`) so a value typed while `numFrames`
 * is transiently small/off-grid never produces a `desired` that
 * `resolveLanding` can't find in its own `validPositions`. */
function snapToPlaceableGrid(raw: number, numFrames: number, multiple: number, offset: number): number {
  const snapped = snapFrameIdx(raw, multiple, offset);
  if (snapped <= 0) return 0;
  const maxPlaceable = maxPlaceablePosition(numFrames, multiple, offset);
  if (maxPlaceable < offset) return 0; // no non-zero slot fits at all
  return Math.min(snapped, maxPlaceable);
}

/**
 * The keyframe timeline rework's (2026-07-18) "bottom card", polished
 * (2026-07-18): three stacked emoji action buttons pinned to the card's
 * top-right corner — ❌ remove, 📁 choose an image file, 📸 capture the
 * timeline's current frame into this card — over a thumbnail preview, with the
 * FRAME number input (previewing where a typed value will actually land: grid
 * snap via `snapFrameIdx` then collision resolution against every other card's
 * position via `keyframeGrid.resolveLanding`) and STRENGTH slider stacked full
 * width beneath it. The whole card is a drop target (`useFileDrop`): dropping
 * an image anywhere on it adds/replaces the image, with an inline error line
 * for an unsupported extension or a failed native resolve. `status:"empty"`
 * (no image yet) and `"error"` each get their own inline hint; `"uploading"`
 * swaps the thumbnail for a spinner.
 *
 * The bottom card rendered by `KeyframesPanel.tsx`'s timeline-bar layout.
 */
export function KeyframeCard({
  item,
  numFrames,
  frameRate,
  multiple,
  offset,
  occupiedOthers,
  disabled,
  onCommitFrameIdx,
  onStrengthChange,
  onRemove,
  onReplaceFile,
  onPickFileFor,
  onCaptureInto,
  isCapturing,
  isPicking,
}: KeyframeCardProps) {
  const strings = useStrings();
  const t = strings.single.keyframes;

  const [frameDraft, setFrameDraft] = useState(item.frameIdx);
  const [isEditingFrame, setIsEditingFrame] = useState(false);

  // Card-wide drag-and-drop (contract v7): the entire card is the drop target
  // now (not just the thumbnail), so an image dropped anywhere on it swaps the
  // keyframe's image via the same `onReplaceFile` path the 📁 picker uses.
  const { isDragOver, error: dropError, handlers: dropHandlers } = useFileDrop({
    accept: ACCEPTED_IMAGE_EXTS,
    disabled,
    onFile: (filePath, fileName) => onReplaceFile(item.id, filePath, fileName),
  });

  // Stay in sync with external frameIdx changes (e.g. a resolved drag on the
  // timeline bar elsewhere) — but never clobber the user's own in-progress
  // edit of this same card.
  useEffect(() => {
    if (!isEditingFrame) setFrameDraft(item.frameIdx);
  }, [item.frameIdx, isEditingFrame]);

  const landingDesired = snapToPlaceableGrid(frameDraft, numFrames, multiple, offset);
  const resolvedLanding = resolveLanding({
    desired: landingDesired,
    occupied: occupiedOthers,
    numFrames,
    multiple,
    offset,
    origin: item.frameIdx,
  });

  function commitFrame() {
    setIsEditingFrame(false);
    onCommitFrameIdx(item.id, resolvedLanding);
    setFrameDraft(resolvedLanding);
  }

  const seconds = frameRate > 0 ? item.frameIdx / frameRate : null;
  const isFixedStart = item.frameIdx === 0;

  const cardClassName = [
    "kfc-card",
    item.status === "error" ? "kfc-card-error" : "",
    item.status === "empty" ? "kfc-card-empty" : "",
    isDragOver ? "kfc-card-dragover" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <li
      className={cardClassName}
      onDragEnter={dropHandlers.onDragEnter}
      onDragOver={dropHandlers.onDragOver}
      onDragLeave={dropHandlers.onDragLeave}
      onDrop={dropHandlers.onDrop}
    >
      <div className="kfc-actions">
        <button
          type="button"
          className="icon-action-button kfc-remove"
          title={t.removeButton}
          aria-label={t.removeButton}
          disabled={disabled}
          onClick={() => onRemove(item.id)}
        >
          <span aria-hidden="true">❌</span>
        </button>
        <button
          type="button"
          className="icon-action-button kfc-pick"
          title={t.chooseFileButton}
          aria-label={t.chooseFileButton}
          disabled={disabled || isPicking || item.status === "uploading"}
          onClick={() => onPickFileFor(item.id)}
        >
          <span aria-hidden="true">📁</span>
        </button>
        <button
          type="button"
          className="icon-action-button kfc-capture"
          title={t.captureButton}
          aria-label={t.captureButton}
          disabled={disabled || isCapturing || item.status === "uploading"}
          onClick={() => onCaptureInto(item.id)}
        >
          <span aria-hidden="true">📸</span>
        </button>
      </div>

      <div className="kfc-thumb">
        {item.status === "uploading" ? (
          <span className="kfc-spinner" role="status" aria-label={t.uploading} />
        ) : item.thumbnailDataUrl ? (
          <img src={item.thumbnailDataUrl} alt={item.fileName || t.label} />
        ) : item.status === "empty" ? (
          <span className="kfc-thumb-empty" aria-hidden="true" />
        ) : (
          <span className="kfc-thumb-filename" title={item.fileName}>
            {item.fileName || "—"}
          </span>
        )}
      </div>

      {isFixedStart && <span className="kfc-badge-fixed">{t.startFrameFixedBadge}</span>}

      {item.status === "empty" && <p className="kfc-hint kfc-hint-warning">{t.emptyCardHint}</p>}
      {item.status === "error" && <p className="kfc-hint kfc-hint-error">{item.errorCode}</p>}
      {dropError && (
        <p className="kfc-hint kfc-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedImage : strings.dnd.resolveFailed}
        </p>
      )}

      <div className="kfc-fields">
        <label className="kfc-field">
          <span className="kfc-field-label">{t.frameIdxLabel}</span>
          <input
            type="number"
            min={0}
            max={Math.max(0, numFrames - 1)}
            value={frameDraft}
            disabled={disabled}
            onFocus={() => setIsEditingFrame(true)}
            onChange={(e) => {
              setIsEditingFrame(true);
              setFrameDraft(Number(e.target.value));
            }}
            onBlur={commitFrame}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitFrame();
              }
            }}
          />
          {seconds !== null && <span className="kfc-hint">{seconds.toFixed(1)}s</span>}
        </label>
        {isEditingFrame && <span className="kfc-hint kfc-hint-landing">{t.willLandAt(resolvedLanding)}</span>}

        <label className="kfc-field">
          <span className="kfc-field-label">{t.strengthLabel}</span>
          <input
            type="range"
            min={STRENGTH_MIN}
            max={STRENGTH_MAX}
            step={STRENGTH_STEP}
            value={item.strength}
            disabled={disabled}
            onChange={(e) => onStrengthChange(item.id, Number(e.target.value))}
          />
          <span className="kfc-hint">{item.strength.toFixed(2)}</span>
        </label>
      </div>
    </li>
  );
}
