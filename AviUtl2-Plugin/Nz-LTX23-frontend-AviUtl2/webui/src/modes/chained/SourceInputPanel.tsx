import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { useFileDrop } from "../../shell/useFileDrop";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { DEFAULT_STRENGTH, STRENGTH_MAX, STRENGTH_MIN, STRENGTH_STEP } from "../single/keyframeUtils";
import type { UseChainFormResult } from "./useChainForm";

/** Contract v7 (drag-and-drop) accepted extensions for Chain's unified source
 * input — the union of `sourceRouting.ts`'s IMAGE_EXTENSIONS/VIDEO_EXTENSIONS,
 * which itself must stay in sync with native's combined "imageOrVideo"
 * `ui.pickFile` filter (see that file's own header comment). */
const SOURCE_INPUT_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm", "mkv"];

export interface SourceInputPanelProps {
  form: UseChainFormResult;
  /** Disabled while a generation is in flight/reserved, mirroring the rest of
   * the form's fields. */
  disabled: boolean;
  /** Contract v7 (drag-and-drop): threaded down to this card's own
   * `useFileDrop` call (owner redesign 2026-07-18: the card is now the whole
   * drop target itself, no wrapping `DropZone`) so a test-injected bridge (or
   * the real one, `ChainedScreen` passes its own prop through unchanged)
   * resolves a dropped file's path — not a second, untracked default
   * singleton. Omitted -> `useFileDrop`'s own default. */
  nativeBridge?: NativeBridge | undefined;
  /** The attached material's pixel size (`useChainForm.sourceMediaSize`),
   * appended to the file-name line below — owner request 2026-08-10, so the
   * width/height sliders can be matched against the source. `null` (nothing
   * attached, still uploading, probe failed, or the file reports no size)
   * leaves the line as the bare file name, exactly as before. Taken as a prop
   * rather than read off `form` so the readout can be driven directly in
   * tests. */
  mediaSize: { width: number; height: number } | null;
}

/** Chain's unified source-input slot (task brief "Chainのソース入力欄一本
 * 化"): a single choose/change button routes to either the from-scratch
 * "start frame" (`form.startFrame`, an image) or the V2V "source video"
 * (`form.sourceVideo`, a video) depending on the picked file's extension —
 * see `useChainForm.pickSource`/`sourceRouting.ts`. Replaces the old separate
 * `StartFramePanel` and `ChainedScreen`'s inline `SourceVideoSection`.
 *
 * The thumbnail/controls area keeps the SAME footprint in every state — image
 * attached, video attached, or nothing yet — so switching between them never
 * reflows the rest of the form. To keep that true regardless of how long
 * either mode's text runs, the inline slider row is kept to "label + slider +
 * a short numeric readout" ONLY (image: strength's "0.90"; video: the bare
 * frame count) in both modes; the one piece of prose that differs in length
 * (video's `contextFramesHint` sentence vs. image having none) lives in its
 * own always-rendered `.source-input-note` line below the slider, sized by
 * CSS (`min-height` on `.source-input-controls`/`.source-input-note` in
 * `ChainedScreen.css`) rather than by whichever mode's text happens to be
 * present — see `noteText` below, which also absorbs the "no source
 * selected" hint so that line doesn't disappear (and shrink the panel) the
 * moment something IS selected. */
export function SourceInputPanel({ form, disabled, nativeBridge, mediaSize }: SourceInputPanelProps) {
  const strings = useStrings();
  const t = strings.chained.sourceInput;
  const svt = strings.chained.sourceVideo;
  const kf = strings.single.keyframes;

  const imageItem = form.startFrame.items[0] ?? null;
  const videoState = form.sourceVideo.state;
  const kind = form.activeSourceKind;

  const busy = form.isPickingSource || imageItem?.status === "uploading" || videoState.status === "uploading";
  const hasSelection = kind !== null;
  const chooseLabel = busy ? t.uploadingButton : hasSelection ? t.changeButton : t.chooseButton;

  // The one always-rendered prose line under the slider (see the class doc
  // comment above): video's descriptive `contextFramesHint` sentence, the
  // "no source selected" hint when nothing is attached yet, or a non-breaking
  // space to hold the line's height in image mode (strength needs no prose).
  // Keeping this a single slot — rather than conditionally rendering (or
  // omitting) a `<p>` per state — is what keeps the panel from reflowing on
  // every unselected/image/video transition.
  const noteText = kind === "video" ? svt.contextFramesHint(form.contextFrames) : kind === "image" ? " " : t.none;

  // Owner redesign (2026-07-18): the whole SOURCE card is the drop target now
  // (not just a `DropZone`-wrapped button row) — mirrors KeyframeCard.tsx's
  // own card-wide D&D. `accept`/`onFile` carried over unchanged from the old
  // `DropZone` usage this replaces.
  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: SOURCE_INPUT_EXTENSIONS,
    onFile: (filePath, fileName) => void form.attachSourceByPath(filePath, fileName),
    disabled: disabled || busy,
    nativeBridge,
  });

  const sectionClassName = ["field", "source-section", isDragOver ? "source-section-dragover" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={sectionClassName}
      onDragEnter={dropHandlers.onDragEnter}
      onDragOver={dropHandlers.onDragOver}
      onDragLeave={dropHandlers.onDragLeave}
      onDrop={dropHandlers.onDrop}
    >
      {/* Owner redesign (2026-07-18): choose/change and clear collapsed to two
       * emoji icon buttons, stacked top-right (🔁 above 📁) — same recipe as
       * KeyframeCard.tsx's `.kfc-actions`. Text/aria-label/disabled carried
       * over unchanged from the old text buttons this replaces. */}
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button source-input-clear"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || busy || !hasSelection}
          onClick={form.clearSource}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button source-input-pick"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || busy}
          onClick={() => void form.pickSource()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      <span className="field-label">{t.heading}</span>
      <p className="field-hint">{t.note}</p>

      <div className="source-input-body">
        <div className="keyframe-thumb">
          {kind === "image" ? (
            imageItem?.status === "uploading" ? (
              <span className="keyframe-spinner" role="status" aria-label={kf.uploading} />
            ) : imageItem?.thumbnailDataUrl ? (
              <img src={imageItem.thumbnailDataUrl} alt={imageItem.fileName || t.heading} />
            ) : (
              <span className="keyframe-thumb-filename" title={imageItem?.fileName}>
                {imageItem?.fileName || "—"}
              </span>
            )
          ) : kind === "video" ? (
            videoState.status === "uploading" ? (
              <span className="keyframe-spinner" role="status" aria-label={t.uploadingButton} />
            ) : (
              <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.videoPlaceholderAlt} />
            )
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          {/* Before anything is picked (`kind === null`), this falls into the
           * STRENGTH branch below (disabled, at the default value) rather than
           * showing nothing — the scratch/image mode is the default, and
           * keeping the same control mounted from the start means the very
           * first pick never reflows the panel either. Both branches render
           * the SAME shape — label + slider + a short numeric readout, never
           * a sentence — so neither can grow taller than the other. */}
          {kind === "video" ? (
            <label className="field field-inline">
              <span className="field-label">{svt.contextFramesLabel}</span>
              <input
                type="range"
                min={form.contextFramesLimits.min}
                max={form.contextFramesLimits.max}
                step={8}
                value={form.contextFrames}
                disabled={disabled || videoState.status !== "ready"}
                onChange={(e) => form.setContextFrames(Number(e.target.value))}
              />
              <span className="field-hint">{form.contextFrames}</span>
            </label>
          ) : (
            <label className="field field-inline">
              <span className="field-label">{t.strengthLabel}</span>
              <input
                type="range"
                min={STRENGTH_MIN}
                max={STRENGTH_MAX}
                step={STRENGTH_STEP}
                value={imageItem?.strength ?? DEFAULT_STRENGTH}
                disabled={disabled || !imageItem || imageItem.status === "error"}
                onChange={(e) => imageItem && form.startFrame.setStrength(imageItem.id, Number(e.target.value))}
              />
              <span className="field-hint">{(imageItem?.strength ?? DEFAULT_STRENGTH).toFixed(2)}</span>
            </label>
          )}
          {/* The one piece of prose that varies by mode/state (see `noteText`
           * above) — always rendered, height reserved by CSS, so it can never
           * be the thing that makes one mode/state taller than another. */}
          <p className="field-hint source-input-note">{noteText}</p>
        </div>
      </div>

      {/* Owner redesign (2026-07-18): the choose/change + clear button row is
       * gone (folded into the top-right icon buttons above) — only the
       * attached file's name survives here, same condition as before. Owner
       * request 2026-08-10: the material's pixel size is appended here (Retake's
       * `sourceReadout` precedent — the size half is dropped entirely when it
       * isn't known, never shown as "0x0" or "unknown"). */}
      {kind === "image" && imageItem?.fileName && (
        <p className="field-hint">{t.sourceReadout(imageItem.fileName, mediaSize?.width ?? 0, mediaSize?.height ?? 0)}</p>
      )}
      {kind === "video" && videoState.fileName && (
        <p className="field-hint">{t.sourceReadout(videoState.fileName, mediaSize?.width ?? 0, mediaSize?.height ?? 0)}</p>
      )}

      {/* Card-wide D&D's own error line (owner redesign 2026-07-18) — same
       * two error cases/copy `DropZone` used to show, just rendered directly
       * now that the card itself is the drop target (mirrors
       * KeyframeCard.tsx's own `.kfc-hint-error` drop-error line). Never
       * omitted: an unsupported/unresolvable drop must still be visible. */}
      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedSource : strings.dnd.resolveFailed}
        </p>
      )}

      {kind === "image" && imageItem?.status === "error" && (
        <p className="field-hint field-hint-error">{imageItem.errorCode}</p>
      )}
      {kind === "video" && videoState.status === "error" && (
        <p className="field-hint field-hint-error">{videoState.errorCode}</p>
      )}
      {form.sourceError && (
        <p className="field-hint field-hint-error">
          {form.sourceError === "UNSUPPORTED_FILE_TYPE" ? t.unsupportedType : form.sourceError}
        </p>
      )}
    </div>
  );
}
