import { useState } from "react";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { AUDIO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { SOURCE_AUDIO_INPUT_EXTENSIONS, useFileDrop } from "../../shell/useFileDrop";
import { formatSecondsLabel } from "../single/paramUtils";
import { AUDIO_FIT_SAFETY_MARGIN_LATENTS } from "./audioFit";
import {
  AUDIO_LATENTS_PER_SEC,
  MAX_CLIP_NUM_FRAMES,
  MIN_CLIP_NUM_FRAMES,
  maxChainAudioLatents,
} from "./chainUtils";
import type { UseChainFormResult } from "./useChainForm";

/** Leftover tail (seconds) below which the "the end of the audio won't be used"
 * note stays quiet: `audioFit` deliberately leaves one audio-latent frame
 * (0.04s) unconsumed on EVERY plan as its measurement safety margin, so
 * reporting anything at or under that would fire the note on a perfect fit.
 * Same threshold `useChainForm.audioAtMaxClips` applies. */
const SURPLUS_NOTE_THRESHOLD_SEC = AUDIO_FIT_SAFETY_MARGIN_LATENTS / AUDIO_LATENTS_PER_SEC;

export interface ChainAudioPanelProps {
  form: UseChainFormResult;
  /** Disabled while a generation is in flight, mirroring the rest of the form. */
  disabled: boolean;
  /** Test/integration seam, threaded down to this card's own `useFileDrop` so a
   * dropped file's path is resolved through the SAME bridge instance the form
   * hook uses — exactly like `SourceInputPanel`'s identical prop. */
  nativeBridge?: NativeBridge | undefined;
  /** W3 レイアウト再構成 (2026-08-11): `ChainedScreen` now wraps this panel in
   * a `<details className="form-accordion">`, with the heading moved to the
   * `<summary>` — pass `false` there so the card doesn't repeat its own
   * `t.heading` line inside the accordion body. Defaults to `true` so the
   * direct-render tests (`ChainAudioPanel.test.tsx`) that predate the
   * accordion keep seeing the heading exactly as before. */
  showHeading?: boolean | undefined;
  /** §1-106: the loaded base model has closed this panel. Everything that
   * ADDS or EDITS material goes dead (as with `disabled`), but the clear
   * button stays live so material attached on another base model can be
   * removed here (it blocks Generate with `materialInClosedPanel` meanwhile).
   * Omitted -> open. */
  unavailable?: boolean | undefined;
  /** §1-120: the distribute mode (MiniMax H3 with
   * `constraints.chain.source_audio`). When given, the ↔️ button and the
   * "added clip length" slider are not drawn; a Strength slider and the
   * "distribute to each clip" button are drawn instead. Omitted = the LTX card,
   * byte for byte. */
  h3?: ChainAudioPanelH3 | undefined;
}

/** §1-120: see {@link ChainAudioPanelProps.h3}. */
export interface ChainAudioPanelH3 {
  strength: number;
  strengthRange: { min: number; max: number; step: number };
  onStrengthChange: (value: number) => void;
  onDistribute: () => void;
  /** Extra stop for the button beyond the card's own (in flight, uploading,
   * not ready, a source video attached). */
  distributeDisabled?: boolean | undefined;
  /** Seconds of the audio past the end of the chain while 17 frames or more
   * are left, else `null` (no note). */
  unusedTailSec: number | null;
}

/**
 * §1-16 長尺A2V: Chain's audio slot — ONE track attached to the whole chain,
 * with the server assigning each clip its own slice of it (the per-clip slices
 * are shown as 🎵 badges on the clip cards themselves, see `ClipCard`).
 *
 * Built to `SourceInputPanel.tsx`'s recipe, deliberately: an independent card
 * that is itself the drop target, with 🔁 clear / 📁 choose stacked in the
 * top-right corner, a 64x64 thumbnail (here the ♫ placeholder Create's A2V card
 * already uses) and a controls column beside it. Two things are Chain-only:
 *
 * - the ↔️ auto-fit button (`form.adjustClipsForAudio`), which re-lays the clip
 *   list over the measured track, and the "added clip length" slider that feeds
 *   it (the length every card the resolver may resize gets);
 * - a single, ALWAYS-MOUNTED note line. Five different situations can want to
 *   say something here — see `noteText` below — and rendering them as five
 *   conditional paragraphs would make the card jump every time the audio, the
 *   clip list or the frame rate changed. One slot, one priority order, height
 *   reserved by CSS (`.chain-audio-note`), same trick as
 *   `SourceInputPanel`'s `.source-input-note`.
 */
export function ChainAudioPanel({
  form,
  disabled: inFlight,
  nativeBridge,
  showHeading = true,
  unavailable = false,
  h3,
}: ChainAudioPanelProps) {
  // §1-106: clear reads `inFlight` only.
  const disabled = inFlight || unavailable;
  const strings = useStrings();
  const t = strings.chained.sourceAudio;

  const state = form.sourceAudio.state;
  const uploading = state.status === "uploading";
  const ready = state.status === "ready";
  const chooseLabel = uploading ? t.uploadingButton : ready ? t.changeButton : t.chooseButton;

  // Same bug fix as Create's audio card (owner real-device report, 2026-07-18):
  // `clearAudio()` cannot reach `useFileDrop`'s own internal error state, so a
  // rejected drop's message would survive a 🔁 click. Bumping this token clears
  // it too.
  const [dropResetToken, setDropResetToken] = useState(0);

  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: SOURCE_AUDIO_INPUT_EXTENSIONS,
    onFile: (filePath, fileName) => void form.attachAudioByPath(filePath, fileName),
    disabled: disabled || uploading,
    nativeBridge,
    resetErrorSignal: dropResetToken,
  });

  const sectionClassName = ["field", "source-section", isDragOver ? "source-section-dragover" : ""]
    .filter(Boolean)
    .join(" ");

  // The ONE note (see the class doc comment), in priority order. Everything
  // below reads live form state — nothing is recomputed here.
  const surplusSec = form.audioSurplusSec;
  let noteText: string;
  /** `"error"` paints the line red (`.field-hint-error`), `"warning"` amber. */
  let noteKind: "plain" | "warning" | "error" = "plain";
  if (h3) {
    // §1-120: the distribute mode's own order — the V2V clash, an unmeasurable
    // length, the unused tail — then the LTX card's "no audio attached" line,
    // else a blank line (the slot stays mounted). No attach error here: the
    // distribute mode never detaches a long audio.
    if (form.hasSourceVideo && form.hasSourceAudio) {
      noteText = t.conflictsWithSourceVideo;
      noteKind = "warning";
    } else if (form.audioProbeFailed) {
      noteText = t.h3.probeFailed;
    } else if (h3.unusedTailSec !== null) {
      noteText = t.h3.unusedTailNote(h3.unusedTailSec.toFixed(1));
    } else if (!form.hasSourceAudio) {
      noteText = t.none;
    } else {
      noteText = " ";
    }
  } else if (form.audioAttachError === "tooLong") {
    // The ceiling depends on the CURRENT frame rate / seam width, so it is
    // derived rather than hard-coded — `Math.floor` keeps the spoken number
    // safely inside the real limit.
    noteText = t.tooLongError(
      Math.floor(maxChainAudioLatents(form.common.frameRate, form.overlapFrames) / AUDIO_LATENTS_PER_SEC),
    );
    noteKind = "error";
  } else if (form.hasSourceVideo && form.hasSourceAudio) {
    noteText = t.conflictsWithSourceVideo;
    noteKind = "warning";
  } else if (form.audioProbeFailed) {
    noteText = t.probeFailed;
  } else if (form.audioAtMaxClips) {
    noteText = t.atMaxClipsNote((surplusSec ?? 0).toFixed(1));
    noteKind = "warning";
  } else if (surplusSec !== null && surplusSec > SURPLUS_NOTE_THRESHOLD_SEC) {
    noteText = t.surplusNote(surplusSec.toFixed(1));
  } else if (!form.hasSourceAudio) {
    noteText = t.none;
  } else {
    // Attached, measured and fitting — the slot stays mounted but silent.
    noteText = " ";
  }
  const noteClassName = [
    "field-hint",
    "chain-audio-note",
    noteKind === "error" ? "field-hint-error" : "",
    noteKind === "warning" ? "chain-audio-note-warning" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={`${sectionClassName} chain-audio-section`}
      onDragEnter={dropHandlers.onDragEnter}
      onDragOver={dropHandlers.onDragOver}
      onDragLeave={dropHandlers.onDragLeave}
      onDrop={dropHandlers.onDrop}
    >
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button chain-audio-clear"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={inFlight || uploading || !form.hasSourceAudio}
          onClick={() => {
            form.clearAudio();
            setDropResetToken((n) => n + 1);
          }}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button chain-audio-pick"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void form.pickAudio()}
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
          ) : ready ? (
            <img src={AUDIO_PLACEHOLDER_DATA_URL} alt={t.audioPlaceholderAlt} />
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          {state.fileName && (
            <p className="field-hint chain-audio-filename" title={state.fileName}>
              {state.fileName}
            </p>
          )}
          {/* The measured length. Absent whenever no probe could read it — the
              `probeFailed` note below then explains why the ↔️ button is off. */}
          {ready && form.audioDurationSec != null && (
            <p className="field-hint">{formatSecondsLabel(form.audioDurationSec)}</p>
          )}

          {/* §1-120: the distribute mode's Strength slider and button, in
              place of the LTX ↔️ button and slider below. */}
          {h3 && (
            <>
              <label className="kfc-field">
                <span className="kfc-field-label">{strings.single.sourceAudio.strengthLabel}</span>
                <input
                  type="range"
                  min={h3.strengthRange.min}
                  max={h3.strengthRange.max}
                  step={h3.strengthRange.step}
                  value={h3.strength}
                  disabled={disabled}
                  aria-label={strings.single.sourceAudio.strengthLabel}
                  data-testid="chain-audio-strength"
                  onChange={(e) => h3.onStrengthChange(Number(e.target.value))}
                />
                <span className="kfc-hint" data-testid="chain-audio-strength-value">
                  {h3.strength.toFixed(2)}
                </span>
              </label>
              <button
                type="button"
                className="secondary-button chain-audio-distribute"
                disabled={disabled || uploading || !ready || form.hasSourceVideo || h3.distributeDisabled === true}
                onClick={h3.onDistribute}
              >
                {t.h3.distributeButton}
              </button>
            </>
          )}
          {/* ↔️ auto-fit. Needs a READY upload and a measured length: without one
              `adjustClipsForAudio` is a documented no-op, so the button would
              look broken rather than unavailable. */}
          {!h3 && (
          <button
            type="button"
            className="secondary-button chain-audio-adjust"
            disabled={disabled || !ready || form.audioDurationSec === null}
            onClick={form.adjustClipsForAudio}
          >
            {t.adjustButton}
          </button>
          )}

          {!h3 && (
          <label className="field field-inline chain-audio-added-frames">
            <span className="field-label">{t.addedClipFramesLabel}</span>
            <input
              type="range"
              min={MIN_CLIP_NUM_FRAMES}
              max={MAX_CLIP_NUM_FRAMES}
              step={8}
              value={form.addedClipFrames}
              disabled={disabled}
              onChange={(e) => form.setAddedClipFrames(Number(e.target.value))}
            />
            <span className="field-hint">{form.addedClipFrames}</span>
          </label>
          )}
        </div>
      </div>

      <p className={noteClassName}>{noteText}</p>

      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedAudio : strings.dnd.resolveFailed}
        </p>
      )}
      {state.status === "error" && <p className="field-hint field-hint-error">{state.errorCode}</p>}
      {form.audioError && <p className="field-hint field-hint-error">{form.audioError}</p>}
    </div>
  );
}
