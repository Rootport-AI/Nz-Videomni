import { useState } from "react";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { selectableControlLoraNames } from "../../lora/controlLoras";
import { LORA_STRENGTH_MAX, LORA_STRENGTH_MIN } from "../../lora/loraTags";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { useFileDrop } from "../../shell/useFileDrop";
import { ReferenceStrengthField } from "../single/ReferenceStrengthField";
import type { UseChainFormResult } from "./useChainForm";

/** The reference-video card's own drop-target extensions — Single's
 * `ReferenceVideoSection` (`modes/single/GenerationForm.tsx`) accepts the
 * identical list; kept as a local copy rather than a shared export for the
 * same reason `SourceInputPanel`/`ChainAudioPanel` each keep their own (see
 * `shell/useFileDrop.ts`'s doc comment on why only the AUDIO list is shared). */
const REFERENCE_VIDEO_EXTENSIONS = ["mp4", "mov", "webm", "mkv"];

/** Seeded strength for a freshly-picked control LoRA — mirrors Single's own
 * `DEFAULT_CONTROL_LORA_STRENGTH` (`modes/single/GenerationForm.tsx`). */
const DEFAULT_CONTROL_LORA_STRENGTH = 1.0;

export interface ChainReferencePanelProps {
  form: UseChainFormResult;
  /** Disabled while a generation is in flight, mirroring the rest of the form. */
  disabled: boolean;
  /** Test/integration seam, threaded down to this card's own `useFileDrop` so
   * a dropped file's path is resolved through the SAME bridge instance the
   * form hook uses — exactly like `ChainAudioPanel`'s identical prop. */
  nativeBridge?: NativeBridge | undefined;
  /** W3 レイアウト再構成 (2026-08-11): `ChainedScreen` now wraps this panel in
   * a `<details className="form-accordion">`, with the heading moved to the
   * `<summary>` — pass `false` there so the card doesn't repeat its own
   * `t.heading` line inside the accordion body. Defaults to `true` so the
   * direct-render tests (`ChainReferencePanel.test.tsx`) that predate the
   * accordion keep seeing the heading exactly as before. */
  showHeading?: boolean | undefined;
}

/**
 * §1-15 (clip-wise IC-LoRA reference): Chain's own reference-video card — ONE
 * video attached to the WHOLE chain, with the server slicing it by FRAME
 * NUMBER and injecting each clip's own window into that clip's stage-1 pass
 * (the per-clip windows are not shown here; unlike §1-16's audio track there
 * is no "担当秒数バッジ" — owner spec).
 *
 * Built to `ChainAudioPanel.tsx`'s exact recipe: an independent card that is
 * itself the drop target, 🔁 clear / 📁 choose stacked in the top-right
 * corner, a 64x64 thumbnail, and a controls column beside it. Three things
 * are specific to this card:
 *
 * - the fps reminder (`t.fpsNote`) is a SEPARATE always-mounted line from the
 *   priority note below — it never competes with the other messages for the
 *   slot, because it says the same thing regardless of attach state;
 * - the control-LoRA `<select>` + its strength slider, and the two opt-in
 *   `ReferenceStrengthField`s (conditioning-attention / reference-video
 *   strength) — mirrors Single's `ReferenceVideoSection` for the identical
 *   `conditioning_attention_strength`/`reference_video_strength` request
 *   fields off a reference-video CONTROL IC-LoRA;
 * - the always-mounted priority note has only two tiers (`"error"`/`"plain"`,
 *   no amber "warning" — that tier lives in `ChainedScreen`'s own stage-1
 *   comfort-budget banner instead, since it needs the whole form's
 *   resolution/clip-length, not just this card's own state).
 */
export function ChainReferencePanel({ form, disabled, nativeBridge, showHeading = true }: ChainReferencePanelProps) {
  const strings = useStrings();
  const t = strings.chained.referenceVideo;

  const state = form.referenceVideo.state;
  const uploading = state.status === "uploading";
  const ready = state.status === "ready";
  const chooseLabel = uploading ? t.uploadingButton : ready ? t.changeButton : t.chooseButton;

  // Same fix as `ChainAudioPanel`'s identical token (owner real-device report,
  // 2026-07-18): `clearReference()` cannot reach `useFileDrop`'s own internal
  // error state, so a rejected drop's message would survive a 🔁 click.
  const [dropResetToken, setDropResetToken] = useState(0);

  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: REFERENCE_VIDEO_EXTENSIONS,
    onFile: (filePath, fileName) => void form.attachReferenceByPath(filePath, fileName),
    disabled: disabled || uploading,
    nativeBridge,
    resetErrorSignal: dropResetToken,
  });

  const sectionClassName = ["field", "source-section", isDragOver ? "source-section-dragover" : ""]
    .filter(Boolean)
    .join(" ");

  // Dropdown-only narrowing: `in-outpainting` has its own dedicated place (the
  // Edit tab's Outpainting panel), so it is excluded HERE — never from
  // `form.controlLoraNames` itself, which the depth-block gate and other
  // consumers still need unfiltered (see `lora/controlLoras.ts`'s
  // `UI_HIDDEN_CONTROL_LORA_NAMES` doc comment).
  const controlLoraOptions = Array.from(selectableControlLoraNames(form.controlLoraNames));

  // The ONE priority note (see the class doc comment above), in priority
  // order. Everything below reads live form state — nothing recomputed here.
  let noteText: string;
  /** `"error"` paints the line red (`.field-hint-error`); no amber tier here
   * (see the class doc comment for why). */
  let noteKind: "plain" | "error" = "plain";
  if (form.hasSourceVideo && form.hasReferenceVideo) {
    noteText = t.conflictsWithSourceVideo;
    noteKind = "error";
  } else if (ready && form.referenceMediaSize === null) {
    // A ready upload whose resolution could not be probed — the same
    // "attached, ready() true, but the measured value is still null" signal
    // `referenceMediaSize`'s own doc comment describes. Never true while
    // still uploading (ready is false then too).
    noteText = t.probeFailed;
  } else if (form.hasReferenceVideo) {
    // Attached — either measured and fine, or still uploading. Either way
    // the static "short reference" reminder applies.
    noteText = t.shortReferenceNote;
  } else {
    noteText = t.none;
  }
  const noteClassName = [
    "field-hint",
    "chain-reference-note",
    noteKind === "error" ? "field-hint-error" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={`${sectionClassName} chain-reference-section`}
      onDragEnter={dropHandlers.onDragEnter}
      onDragOver={dropHandlers.onDragOver}
      onDragLeave={dropHandlers.onDragLeave}
      onDrop={dropHandlers.onDrop}
    >
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button chain-reference-clear"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading || !form.hasReferenceVideo}
          onClick={() => {
            form.clearReference();
            setDropResetToken((n) => n + 1);
          }}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button chain-reference-pick"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void form.pickReference()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      {showHeading && <span className="field-label">{t.heading}</span>}
      {/* Always-mounted, independent of attach state — see the class doc
          comment for why this is its own line rather than folded into the
          priority note below. */}
      <p className="field-hint">{t.fpsNote}</p>

      <div className="source-input-body">
        <div className="keyframe-thumb">
          {uploading ? (
            <span className="keyframe-spinner" role="status" aria-label={t.uploadingButton} />
          ) : ready ? (
            <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.videoPlaceholderAlt} />
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          {state.fileName && (
            <p className="field-hint chain-reference-filename" title={state.fileName}>
              {state.fileName}
            </p>
          )}
          {/* The measured pixel size. Absent whenever it isn't known — the
              `probeFailed` priority note below then explains why. No duration
              counterpart: the slicing is by FRAME NUMBER (see `t.fpsNote`). */}
          {ready && form.referenceMediaSize != null && (
            <p className="field-hint">{t.resolutionLabel(form.referenceMediaSize.width, form.referenceMediaSize.height)}</p>
          )}
        </div>
      </div>

      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedVideo : strings.dnd.resolveFailed}
        </p>
      )}

      {controlLoraOptions.length > 0 && (
        <label className="field">
          <span className="field-label">{t.controlLoraLabel}</span>
          <select
            className="field-select"
            disabled={disabled}
            value={form.controlLora?.name ?? ""}
            onChange={(e) => {
              const name = e.target.value;
              form.setControlLora(
                name ? { name, strength: form.controlLora?.strength ?? DEFAULT_CONTROL_LORA_STRENGTH } : null,
              );
            }}
          >
            <option value="">{t.controlLoraNoneOption}</option>
            {controlLoraOptions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      )}
      {form.controlLora !== null && (
        <label className="field">
          <span className="field-label">{t.controlLoraStrengthLabel}</span>
          <input
            type="range"
            min={LORA_STRENGTH_MIN}
            max={LORA_STRENGTH_MAX}
            step={0.05}
            value={form.controlLora.strength}
            disabled={disabled}
            onChange={(e) => form.setControlLoraStrength(Number(e.target.value))}
          />
          <span className="field-hint">{form.controlLora.strength.toFixed(2)}</span>
        </label>
      )}

      <p className={noteClassName}>{noteText}</p>

      {state.status === "error" && <p className="field-hint field-hint-error">{state.errorCode}</p>}
      {form.referenceError && <p className="field-hint field-hint-error">{form.referenceError}</p>}

      {ready && (
        <>
          <ReferenceStrengthField
            label={t.conditioningAttentionStrengthLabel}
            value={form.conditioningAttentionStrength}
            onChange={form.setConditioningAttentionStrength}
            disabled={disabled}
          />
          <ReferenceStrengthField
            label={t.referenceVideoStrengthLabel}
            value={form.referenceVideoStrength}
            onChange={form.setReferenceVideoStrength}
            disabled={disabled}
          />
        </>
      )}
    </div>
  );
}
