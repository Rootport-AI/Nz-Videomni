import { useEffect, useRef, useState } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import type { NativeBridge } from "../../bridge";
import {
  REFERENCE_IMAGE_EXTENSIONS,
  REFERENCE_VIDEO_EXTENSIONS,
  SOURCE_AUDIO_INPUT_EXTENSIONS,
  useFileDrop,
} from "../../shell/useFileDrop";
import { AUDIO_PLACEHOLDER_DATA_URL, VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { formatSecondsLabel } from "./paramUtils";
import { REFERENCE_KINDS, effectiveA2vSlot, referenceTag } from "./referenceRules";
import type { ReferenceKind, ReferenceSlot, ReferenceSlotUse, ReferenceSlotsApi } from "./referenceRules";
import "../../styles/formAccordion.css";
import "./ReferenceAccordion.css";

/** The A2V (Audio to video) side of the Audios cards. */
export interface ReferenceAccordionA2v {
  /** `constraints.singleSourceAudio != null`: the Strength slider is shown
   * only then. */
  available: boolean;
  strength: number;
  onStrengthChange: (value: number) => void;
  strengthMin: number;
  strengthMax: number;
  strengthStep: number;
  /** The existing `audioPrecheck` warning text, or `null`. */
  tooShortWarning: string | null;
}

export interface ReferenceAccordionProps {
  api: ReferenceSlotsApi;
  /** Bump to force the accordion open (right-click delivery). */
  openSignal: number;
  /** The pen button: appends the card's tag (`<Picture 1>`) to the prompt. */
  onAppendTag: (tag: string) => void;
  /** The Audios cards' A2V side. Not needed when {@link embedded}. */
  a2v?: ReferenceAccordionA2v | undefined;
  /** Test/DI seam for the drop zones; defaults to the app-wide bridge. */
  nativeBridge?: NativeBridge | undefined;
  /** A submit in flight: every control is locked (the form's `disabled`). */
  disabled?: boolean | undefined;
  /** §1-109: one Chained clip's accordion. The Enable checkbox, the ON badge
   * and the Audios cards' "Audio to video" / Strength / too-short warning are
   * not drawn — the screen-wide References checkbox stands in for Enable, and
   * a chain has no A2V card. Default `false` (Create, unchanged). */
  embedded?: boolean | undefined;
}

const ACCEPT: Record<ReferenceKind, readonly string[]> = {
  picture: REFERENCE_IMAGE_EXTENSIONS,
  video: REFERENCE_VIDEO_EXTENSIONS,
  audio: SOURCE_AUDIO_INPUT_EXTENSIONS,
};

function ReferenceSlotCard({
  slot,
  use,
  api,
  a2v,
  onAppendTag,
  nativeBridge,
  disabled,
  embedded,
}: {
  slot: ReferenceSlot;
  use: ReferenceSlotUse;
  api: ReferenceSlotsApi;
  a2v: ReferenceAccordionA2v | undefined;
  onAppendTag: (tag: string) => void;
  nativeBridge?: NativeBridge | undefined;
  disabled: boolean;
  embedded: boolean;
}) {
  const strings = useStrings();
  const t = strings.single.reference;
  const { kind } = slot;
  const tag = referenceTag(kind, slot.slot);
  const uploading = slot.status === "uploading";
  const ready = slot.status === "ready";
  const [dropResetToken, setDropResetToken] = useState(0);

  const {
    isDragOver,
    error: dropError,
    handlers,
  } = useFileDrop({
    accept: ACCEPT[kind],
    onFile: (filePath, fileName) => void api.attach(kind, slot.slot, filePath, fileName),
    disabled: disabled || uploading,
    nativeBridge,
    resetErrorSignal: dropResetToken,
  });

  // Enable off, left out by a rule, or an A2V card in effect elsewhere (then
  // every other card is grey, empty ones included).
  const a2vInEffect = effectiveA2vSlot(api.state) !== null;
  const grey = !api.state.enabled || use.ignored !== null || (a2vInEffect && !use.isA2v);
  const className = [
    "field",
    "source-section",
    "ref-slot",
    isDragOver ? "source-section-dragover" : "",
    grey ? "ref-slot-ignored" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const chooseLabel = ready ? t.changeButton : t.chooseButton;

  let thumb;
  if (uploading) {
    thumb = <span className="keyframe-spinner" role="status" aria-label={t.uploading} />;
  } else if (ready) {
    if (kind === "audio") {
      thumb = <img src={AUDIO_PLACEHOLDER_DATA_URL} alt={t.audioAlt} />;
    } else if (kind === "video") {
      thumb = <img src={slot.thumbnailDataUrl ?? VIDEO_PLACEHOLDER_DATA_URL} alt={t.videoAlt} />;
    } else if (slot.thumbnailDataUrl) {
      thumb = <img src={slot.thumbnailDataUrl} alt={t.pictureAlt} />;
    } else {
      thumb = <span className="keyframe-thumb-filename">{slot.fileName}</span>;
    }
  } else {
    thumb = <span className="keyframe-thumb-filename">{t.emptySlot}</span>;
  }

  const isA2vChecked = api.state.a2vSlot === slot.slot;

  return (
    <div
      className={className}
      data-testid={`ref-slot-${kind}-${slot.slot}`}
      onDragEnter={handlers.onDragEnter}
      onDragOver={handlers.onDragOver}
      onDragLeave={handlers.onDragLeave}
      onDrop={handlers.onDrop}
    >
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading}
          onClick={() => {
            api.clear(kind, slot.slot);
            setDropResetToken((n) => n + 1);
          }}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void api.pick(kind, slot.slot)}
        >
          <span aria-hidden="true">📁</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={t.tagButton}
          aria-label={t.tagButton}
          disabled={disabled || !ready}
          onClick={() => onAppendTag(tag)}
        >
          <span aria-hidden="true">✒</span>
        </button>
      </div>

      <p className="ref-slot-title">{tag}</p>

      <div className="source-input-body">
        <div className="keyframe-thumb">{thumb}</div>
        <div className="source-input-controls">
          {slot.fileName && <p className="field-hint ref-slot-name">{slot.fileName}</p>}
          {kind !== "picture" && ready && slot.durationSec !== null && (
            <p className="field-hint">{formatSecondsLabel(slot.durationSec)}</p>
          )}
        </div>
      </div>

      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "resolveFailed"
            ? strings.dnd.resolveFailed
            : kind === "picture"
              ? strings.dnd.unsupportedImage
              : kind === "video"
                ? strings.dnd.unsupportedVideo
                : strings.dnd.unsupportedAudio}
        </p>
      )}
      {slot.status === "error" && <p className="field-hint field-hint-error">{slot.errorCode}</p>}
      {use.truncated && use.usedSec !== null && (
        <p className="warning-banner warning-banner-mild">{t.truncatedWarning(use.usedSec)}</p>
      )}
      {use.tooShort && <p className="warning-banner warning-banner-mild">{t.tooShortWarning(api.rules.clipMinSec)}</p>}
      {use.durationUnknown && <p className="warning-banner warning-banner-mild">{t.durationUnknownWarning}</p>}
      {slot.trimFailed && <p className="warning-banner warning-banner-mild">{t.trimFailedWarning}</p>}

      {kind === "audio" && !embedded && a2v && (
        <>
          <label className="field field-inline">
            <input
              type="checkbox"
              checked={isA2vChecked}
              disabled={disabled}
              onChange={(e) => api.setA2vSlot(e.target.checked ? slot.slot : null)}
            />
            <span>{t.audioToVideoLabel}</span>
          </label>
          {isA2vChecked && a2v.available && (
            <>
              <label className="kfc-field">
                <span className="kfc-field-label">{strings.single.sourceAudio.strengthLabel}</span>
                <input
                  type="range"
                  min={a2v.strengthMin}
                  max={a2v.strengthMax}
                  step={a2v.strengthStep}
                  value={a2v.strength}
                  disabled={disabled}
                  aria-label={strings.single.sourceAudio.strengthLabel}
                  data-testid="source-audio-strength"
                  onChange={(e) => a2v.onStrengthChange(Number(e.target.value))}
                />
                <span className="kfc-hint" data-testid="source-audio-strength-value">
                  {a2v.strength.toFixed(2)}
                </span>
              </label>
              <p className="field-hint">{strings.single.sourceAudio.strengthHint}</p>
            </>
          )}
          {isA2vChecked && a2v.tooShortWarning && (
            <p className="warning-banner warning-banner-mild">{a2v.tooShortWarning}</p>
          )}
        </>
      )}
    </div>
  );
}

function ReferenceRow({
  kind,
  api,
  a2v,
  onAppendTag,
  nativeBridge,
  disabled,
  embedded,
}: {
  kind: ReferenceKind;
  api: ReferenceSlotsApi;
  a2v: ReferenceAccordionA2v | undefined;
  onAppendTag: (tag: string) => void;
  nativeBridge?: NativeBridge | undefined;
  disabled: boolean;
  embedded: boolean;
}) {
  const t = useStrings().single.reference;
  const { summary, rules, state } = api;
  const max = kind === "picture" ? rules.maxImages : kind === "video" ? rules.maxVideos : rules.maxAudios;
  const countOver = summary.over[kind];
  const secKind = kind === "picture" ? null : kind;
  const secTotal = kind === "video" ? rules.videoTotalSec : rules.audioTotalSec;
  const secOver = kind === "video" ? summary.over.videoSec : kind === "audio" ? summary.over.audioSec : false;

  return (
    <div className="ref-row" data-testid={`ref-row-${kind}`}>
      <div className="ref-row-title">
        <span>{t.rowTitle[kind]}</span>
        <span className={`ref-row-count${countOver ? " field-hint-error ref-over" : ""}`}>
          {summary.filled[kind]}/{max}
        </span>
        {secKind && (
          <span className={`ref-row-count${secOver ? " field-hint-error ref-over" : ""}`}>
            ({formatSecondsLabel(summary.seconds[secKind])}/{secTotal}s)
          </span>
        )}
      </div>
      <div className="ref-row-scroll">
        {state.slots[kind].map((slot, i) => {
          const use = summary.slotUse[kind][i];
          if (!use) return null;
          return (
            <ReferenceSlotCard
              key={slot.slot}
              slot={slot}
              use={use}
              api={api}
              a2v={a2v}
              onAppendTag={onAppendTag}
              nativeBridge={nativeBridge}
              disabled={disabled}
              embedded={embedded}
            />
          );
        })}
      </div>
    </div>
  );
}

/**
 * §1-107: MiniMax H3's "Reference" accordion, with three rows
 * (Pictures / Videos / Audios) of horizontally scrolling cards. Presentation
 * only: every value and mutation goes through `api` (`useReferenceSlots`).
 * `<details>` is uncontrolled (like `NagAccordion`); `openSignal` opens it.
 * §1-109: `embedded` is one Chained clip's copy (see the prop).
 */
export function ReferenceAccordion({
  api,
  openSignal,
  onAppendTag,
  a2v,
  nativeBridge,
  disabled = false,
  embedded = false,
}: ReferenceAccordionProps) {
  const t = useStrings().single.reference;
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const lastSignal = useRef(openSignal);
  useEffect(() => {
    if (lastSignal.current === openSignal) return;
    lastSignal.current = openSignal;
    if (detailsRef.current) detailsRef.current.open = true;
  }, [openSignal]);

  const { summary, rules, state } = api;

  return (
    <details className="form-accordion ref-accordion" ref={detailsRef}>
      <summary className="form-accordion-summary">
        {t.heading}
        {!embedded && state.enabled && <span className="nag-on-badge">{t.onBadge}</span>}
        <span
          className={`ref-summary-count${summary.over.total ? " field-hint-error ref-over" : ""}`}
          data-testid="ref-total-count"
        >
          {summary.filled.total}/{rules.maxTotal}
        </span>
      </summary>
      <div className="form-accordion-body">
        {!embedded && (
          <label className="field field-inline">
            <input
              type="checkbox"
              checked={state.enabled}
              disabled={disabled}
              onChange={(e) => api.setEnabled(e.target.checked)}
            />
            <span className="field-label">{t.enableLabel}</span>
          </label>
        )}
        {REFERENCE_KINDS.map((kind) => (
          <ReferenceRow
            key={kind}
            kind={kind}
            api={api}
            a2v={a2v}
            onAppendTag={onAppendTag}
            nativeBridge={nativeBridge}
            disabled={disabled}
            embedded={embedded}
          />
        ))}
      </div>
    </details>
  );
}
