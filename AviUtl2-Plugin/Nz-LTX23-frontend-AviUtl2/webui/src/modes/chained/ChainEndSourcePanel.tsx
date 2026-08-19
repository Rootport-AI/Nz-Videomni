import { useState } from "react";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { useFileDrop } from "../../shell/useFileDrop";
import { MAX_END_SOURCE_STRENGTH, MIN_END_SOURCE_STRENGTH } from "./chainUtils";
import type { UseChainFormResult } from "./useChainForm";

/** This card's own drop-target extensions — the union of `sourceRouting.ts`'s
 * IMAGE/VIDEO lists, i.e. exactly what `SourceInputPanel`'s
 * `SOURCE_INPUT_EXTENSIONS` accepts (the two slots route the same file types
 * through the same `routeSourceByExtension`). Kept as a local copy rather than a
 * shared export for the same reason every other card here keeps its own — see
 * `shell/useFileDrop.ts`'s doc comment on why only the AUDIO list is shared. */
const END_SOURCE_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm", "mkv"];

export interface ChainEndSourcePanelProps {
  form: UseChainFormResult;
  /** Disabled while a generation is in flight, mirroring the rest of the form. */
  disabled: boolean;
  /** Test/integration seam, threaded down to this card's own `useFileDrop` so a
   * dropped file's path is resolved through the SAME bridge instance the form
   * hook uses — exactly like `ChainReferencePanel`'s identical prop. */
  nativeBridge?: NativeBridge | undefined;
  /** `ChainedScreen` wraps this panel in a `<details className="form-accordion">`
   * with the heading moved to the `<summary>` — pass `false` there so the card
   * doesn't repeat its own `t.heading` line inside the accordion body. Defaults
   * to `true` so direct-render tests see the heading. Same convention as
   * `ChainReferencePanel`/`ChainAudioPanel`. */
  showHeading?: boolean | undefined;
}

/**
 * 素材（末尾）: the card for the ONE image or video the generated chain must END
 * with (`GenerateChainRequest.end_source`) — the mirror image of
 * `SourceInputPanel`'s START slot, built to `ChainReferencePanel.tsx`'s exact
 * recipe (the card is itself the drop target, 🔁 clear / 📁 choose stacked in
 * the top-right corner, a 64x64 thumbnail, a controls column beside it, and ONE
 * always-mounted priority note so the card never reflows).
 *
 * One thing is specific to this card:
 *
 * - the ANCHOR LENGTH has no control. v1 had an "end frames" slider here;
 *   窓内モード freezes the last `END_SOURCE_CONTEXT_FRAMES` (8) frames of the
 *   clip onto the head of the material, always — so the length side of this
 *   card is pure readout: the priority note reports that 8 for a video, the
 *   still-image sentence for an image, and the two length failures ("shorter
 *   than 9 frames" / "length unknown") when the material cannot supply the
 *   anchor at all. Below the note sits the one ADVISORY the card carries, the
 *   clip-length quality warning — separate from the priority rotation because
 *   it is about the CLIP, not the material, and can therefore be true at the
 *   same time as any of the readouts. The anchor's STRENGTH, unlike its
 *   length, IS a control (バッチ1・2026-08-18) — see the strength slider below.
 */
export function ChainEndSourcePanel({ form, disabled, nativeBridge, showHeading = true }: ChainEndSourcePanelProps) {
  const strings = useStrings();
  const t = strings.chained.endSource;

  const kind = form.endSourceKind;
  const videoState = form.endSourceVideo.state;
  const imageItem = form.endSourceImage.items[0] ?? null;
  const uploading = form.endSourceStatus === "uploading";
  const chooseLabel = uploading ? t.uploadingButton : form.hasEndSource ? t.changeButton : t.chooseButton;

  // Same fix as `ChainReferencePanel`'s identical token (owner real-device
  // report, 2026-07-18): `clearEndSource()` cannot reach `useFileDrop`'s own
  // internal error state, so a rejected drop's message would survive a 🔁 click.
  const [dropResetToken, setDropResetToken] = useState(0);

  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: END_SOURCE_EXTENSIONS,
    onFile: (filePath, fileName) => void form.attachEndSourceByPath(filePath, fileName),
    disabled: disabled || uploading,
    nativeBridge,
    resetErrorSignal: dropResetToken,
  });

  const sectionClassName = ["field", "source-section", isDragOver ? "source-section-dragover" : ""]
    .filter(Boolean)
    .join(" ");

  // The ONE priority note, in priority order. Everything below reads live form
  // state — nothing is recomputed here.
  //
  // The two LENGTH failures sit between the conflicts and the ordinary readouts:
  // they are red (the material genuinely cannot be used as it is) and they have
  // to beat the plain frame-count line, which has no number to show in those
  // states anyway (`endContextFrames` is `null`).
  let noteText: string;
  /** `"error"` paints the line red (`.field-hint-error`). */
  let noteKind: "plain" | "error" = "plain";
  if (form.hasEndSource && form.hasSourceAudio) {
    noteText = t.conflictsWithAudio;
    noteKind = "error";
  } else if (form.hasEndSource && form.hasReferenceVideo) {
    noteText = t.conflictsWithReference;
    noteKind = "error";
  } else if (form.endSourceLengthIssue === "tooShort") {
    noteText = t.tooShortNote;
    noteKind = "error";
  } else if (form.endSourceLengthIssue === "unknown") {
    noteText = t.lengthUnknownNote;
    noteKind = "error";
  } else if (kind === "image") {
    noteText = t.stillImageNote;
  } else if (kind === "video" && form.endContextFrames !== null) {
    noteText = t.contextFramesHint(form.endContextFrames);
  } else {
    noteText = t.none;
  }
  const noteClassName = ["field-hint", "chain-end-source-note", noteKind === "error" ? "field-hint-error" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={`${sectionClassName} chain-end-source-section`}
      onDragEnter={dropHandlers.onDragEnter}
      onDragOver={dropHandlers.onDragOver}
      onDragLeave={dropHandlers.onDragLeave}
      onDrop={dropHandlers.onDrop}
    >
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button chain-end-source-clear"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading || !form.hasEndSource}
          onClick={() => {
            form.clearEndSource();
            setDropResetToken((n) => n + 1);
          }}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button chain-end-source-pick"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void form.pickEndSource()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      {showHeading && <span className="field-label">{t.heading}</span>}
      <p className="field-hint">{t.note}</p>

      <div className="source-input-body">
        <div className="keyframe-thumb">
          {uploading ? (
            <span className="keyframe-spinner" role="status" aria-label={t.uploadingButton} />
          ) : kind === "image" ? (
            imageItem?.thumbnailDataUrl ? (
              <img src={imageItem.thumbnailDataUrl} alt={imageItem.fileName || t.heading} />
            ) : (
              <span className="keyframe-thumb-filename" title={imageItem?.fileName}>
                {imageItem?.fileName || "—"}
              </span>
            )
          ) : kind === "video" ? (
            <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.videoPlaceholderAlt} />
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          {(kind === "video" ? videoState.fileName : imageItem?.fileName) && (
            <p
              className="field-hint chain-end-source-filename"
              title={kind === "video" ? (videoState.fileName ?? undefined) : imageItem?.fileName}
            >
              {kind === "video" ? videoState.fileName : imageItem?.fileName}
            </p>
          )}
          {/* The anchor's LENGTH has no control (see the panel's doc comment) —
              this is the anchor's STRENGTH instead (バッチ1・2026-08-18),
              markup lifted from `ChainedScreen.tsx`'s `overlapStrength`
              slider. Unconditional, like that slider: it seeds the request
              value whether or not material is attached yet, exactly like
              `overlapFrames`/`overlapStrength` do for the seam between clips. */}
          <label className="field chain-end-source-strength">
            <span className="field-label">{t.strengthLabel}</span>
            <input
              type="range"
              min={MIN_END_SOURCE_STRENGTH}
              max={MAX_END_SOURCE_STRENGTH}
              step={0.05}
              value={form.endSourceStrength}
              disabled={disabled}
              onChange={(e) => form.setEndSourceStrength(Number(e.target.value))}
            />
            <span className="field-hint">{form.endSourceStrength.toFixed(2)}</span>
          </label>
          <p className="field-hint chain-end-source-strength-hint">{t.strengthHint}</p>
        </div>
      </div>

      <p className={noteClassName}>{noteText}</p>

      {/* 窓内モード品質警告 (2026-08-17): a clip longer than one stage-2 window
          puts the frozen tail and the frames blending into it in different
          tiles, which smears and flickers around the join. Amber/non-blocking
          (`warning-banner-mild`, the same treatment the stage-1 reference
          advisory gets) — the result is degraded, not invalid, and the remedy is
          in the sentence. */}
      {form.endSourceQualityLimitFrames !== null && (
        <p className="warning-banner warning-banner-mild">{t.qualityWarning(form.endSourceQualityLimitFrames)}</p>
      )}

      {/* §1-22 (2026-08-18): the multi-clip counterpart of the warning above —
          same card, same priority band, same mild/non-blocking treatment. The
          two conditions are mutually exclusive by construction
          (`endSourceQualityLimitFrames` is single-clip-only,
          `endSourceMultiClipQualityWarning` is multi-clip-only — see
          `useChainForm`'s own doc comments), so no configuration renders both
          banners at once today, but each is guarded independently rather than
          as an else-branch of the other in case that ever changes. */}
      {form.endSourceMultiClipQualityWarning && (
        <p className="warning-banner warning-banner-mild chain-end-source-multiclip-warning">
          {t.multiClipQualityWarning}
        </p>
      )}

      {/* バッチ3 (2026-08-18): when the material has an audio track it is now
          frozen into the tail along with the video — no longer generated
          independently. Materials without audio, and still images, still
          fall back to independent generation, so the note stays worth
          stating explicitly rather than leaving implicit. */}
      <p className="field-hint chain-end-source-audio-note">{t.audioModeNote}</p>

      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedSource : strings.dnd.resolveFailed}
        </p>
      )}

      {kind === "video" && videoState.status === "error" && (
        <p className="field-hint field-hint-error">{videoState.errorCode}</p>
      )}
      {kind === "image" && imageItem?.status === "error" && (
        <p className="field-hint field-hint-error">{imageItem.errorCode}</p>
      )}
      {form.endSourceError && (
        <p className="field-hint field-hint-error">
          {/* The extension list is identical to the START slot's, so its sentence
              is reused verbatim rather than written twice. */}
          {form.endSourceError === "UNSUPPORTED_FILE_TYPE"
            ? strings.chained.sourceInput.unsupportedType
            : form.endSourceError}
        </p>
      )}
    </div>
  );
}
