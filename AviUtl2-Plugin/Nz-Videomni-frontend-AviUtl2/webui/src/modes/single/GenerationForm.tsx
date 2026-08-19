import { useLayoutEffect, useRef, useState } from "react";
import type { AppConfig } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { selectableControlLoraNames } from "../../lora/controlLoras";
import { LORA_STRENGTH_MAX, LORA_STRENGTH_MIN } from "../../lora/loraTags";
import { AUDIO_PLACEHOLDER_DATA_URL, VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { SOURCE_AUDIO_EXTENSIONS, useFileDrop } from "../../shell/useFileDrop";
import { CropOutputField, DurationField, FrameRateSeedFields, ReservedFields, SizeFields } from "./CommonGenerationFields";
import { KeyframesPanel } from "./KeyframesPanel";
import { formatSecondsLabel } from "./paramUtils";
import { PresetDropdown } from "./PresetDropdown";
import { ReferenceStrengthField } from "./ReferenceStrengthField";
import type { UseGenerationFormResult } from "./useGenerationForm";
import type { UseKeyframesResult } from "./useKeyframes";

/** Contract v7 (drag-and-drop) accepted extensions for the reference-video
 * and source-audio slots — mirrors native's `ui.pickFile` filters exactly
 * (`bridge_core.cpp`'s `PickFileFilter`). */
const REFERENCE_VIDEO_EXTENSIONS = ["mp4", "mov", "webm", "mkv"];
// The audio list moved to `shell/useFileDrop.ts` (§1-16 長尺A2V): Chain's own
// audio card needs exactly the same set, and a shared module beats a second
// copy that could drift from native's filter.

/** Seed strength for a FRESH control-LoRA dropdown selection (the panel's
 * weight slider then takes over) — same "1.0, not the range's minimum"
 * convention `lora/loraTags.ts`'s `LORA_STRENGTH_DEFAULT` uses for a style
 * tag's own default. */
const DEFAULT_CONTROL_LORA_STRENGTH = 1.0;

export interface GenerationFormProps {
  form: UseGenerationFormResult;
  keyframes: UseKeyframesResult;
  /** `config.limits.conditioning_frame_idx_multiple` (default 8). */
  frameIdxMultiple: number;
  /** `config.limits.conditioning_keyframe_grid_offset` (default 1). */
  gridOffset: number;
  /** Keyframe timeline rework (2026-07-18): DURATION edits now route through
   * `SingleScreen`'s `useDurationShrinkGuard` instead of `form.setNumFrames`
   * directly, so a shrink that would strand a keyframe can raise the
   * confirmation modal. `onDurationChange` is the live (per-drag/keystroke)
   * stream; `onDurationCommit` fires once the edit settles. */
  onDurationChange: (value: number, snap: boolean) => void;
  onDurationCommit: (value: number, snap: boolean) => void;
  /** Keyframe timeline rework (2026-07-18): preset selection likewise routes
   * through the guard (a preset carries its own `num_frames` and can shrink
   * it), not `form.applyPreset` directly. */
  onApplyPreset: (name: string) => void;
  /** U-R1 MJ-1: the whole form is only disabled while a submit is in flight
   * (`submitState.phase === "submitting"`) — a running job no longer freezes
   * the form, only the Generate button (which `SingleScreen` disables
   * separately via `hasActiveJob`). */
  disabled: boolean;
  configFallbackWarning: boolean;
  /** `deriveGenerationParams`'s human-readable reasons for the auto-chosen
   * resolution (only present on a right-click prefill). Surfaced in the
   * IC-LoRA reference block to explain the 128-multiple rounding. */
  resolutionNotes?: string[] | undefined;
  /** Read by `PresetDropdown`. The IC-LoRA control-LoRA dropdown (below) no
   * longer reads this directly — it uses `form.controlLoraNames` instead
   * (第5波: state-owned selection, not a `config.model.ic_loras` lookup at
   * render time). */
  config: AppConfig;
  /** Contract v7 (drag-and-drop): threaded down to the reference-video/
   * source-audio cards' own `useFileDrop` calls (each card is its own
   * whole-card drop target now) so a test-injected bridge (or the real one,
   * `SingleScreen` passes its own prop through unchanged) resolves a dropped
   * file's path — not a second, untracked default singleton. Omitted ->
   * `useFileDrop`'s own default (the app-wide bridge singleton). */
  nativeBridge?: NativeBridge | undefined;
  /** Right-click redesign W2 (accordion auto-expand): true when this mount's
   * `initialIntent` was `"reference-video"` (`SingleScreen` derives it from
   * `menuRouting.ts`'s intent string). Opens the IC-LoRA `<details>`
   * imperatively once on mount via `referenceDetailsRef` — see that effect's
   * doc comment for why this stays a ref write rather than a controlled
   * `open` prop. Omitted/false leaves the accordion exactly as before
   * (collapsed, user-togglable). */
  autoOpenReference?: boolean | undefined;
  /** Right-click redesign W2: true when `initialIntent.intent` was
   * `"video-audio-to-video"` or `"audio-to-video"`. Opens the A2V
   * `<details>` on mount the same way `autoOpenReference` does for IC-LoRA. */
  autoOpenA2v?: boolean | undefined;
}

/** The Create form: size/duration/fps/seed/presets, the M4 I2V keyframe
 * panel, plus the IC-LoRA (reference video) and A2V (source audio)
 * accordions. The Generate button + estimate readout no longer live here —
 * `SingleScreen` renders them via `GenerateButtonBar` in the right-hand
 * column (U2). The prompt field itself lives in the shared `PromptBar` above
 * the mode tabs. */
export function GenerationForm({
  form,
  keyframes,
  frameIdxMultiple,
  gridOffset,
  onDurationChange,
  onDurationCommit,
  onApplyPreset,
  disabled,
  configFallbackWarning,
  resolutionNotes,
  config,
  nativeBridge,
  autoOpenReference,
  autoOpenA2v,
}: GenerationFormProps) {
  const strings = useStrings();
  // Right-click redesign W2: the IC-LoRA/A2V accordions default to collapsed,
  // but a right-click prefill that lands material INSIDE one of them
  // (referenceVideo / videoAudioToVideo / audioToVideo) should open it so the
  // user immediately sees where the material went — see intents routed by
  // `menuRouting.ts` and SingleScreen's `autoOpenReference`/`autoOpenA2v`
  // derivation. Kept as an imperative `.open = true` on mount (NOT a
  // controlled `open={autoOpenReference}` prop) so the `<details>` stays
  // fully uncontrolled afterward — the user's own manual expand/collapse is
  // never fought by a prop that doesn't change again. `GenerationForm` only
  // ever mounts fresh for these three intents (SingleScreen remounts the
  // whole form via `key` on a new intent), so a mount-only effect is exactly
  // "open once, then hands off" — no dependency array churn to guard against.
  // `useLayoutEffect` (not `useEffect`): this write must land in the SAME
  // synchronous commit phase as the rest of the mount, before any passive
  // effect elsewhere in the tree runs. `SingleScreen`'s own auto-load mount
  // effect (the sibling `useEffect` that shows the "loaded from a right-click"
  // note / kicks off the #3 audio extraction) is a passive effect too; a
  // passive-effect write here landed AFTER it in practice and could get
  // coalesced with that effect's own note-area DOM updates under React's
  // batching, which starved a real regression test
  // (`App.prefill.test.tsx`'s videoAudioToVideo case) of ever observing the
  // transient "Extracting audio…" note. Layout effects run before ANY
  // passive effect in the commit, so this write no longer shares a batching
  // window with that unrelated async note sequence.
  const referenceDetailsRef = useRef<HTMLDetailsElement>(null);
  const a2vDetailsRef = useRef<HTMLDetailsElement>(null);
  useLayoutEffect(() => {
    if (autoOpenReference && referenceDetailsRef.current) {
      referenceDetailsRef.current.open = true;
    }
    if (autoOpenA2v && a2vDetailsRef.current) {
      a2vDetailsRef.current.open = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only: this form remounts fresh per intent (key change), never re-renders with a new autoOpen* value on the same instance
  }, []);
  const {
    values,
    limits,
    durationHint,
    gettingSize,
    getSizeError,
    spillThresholdFrames,
    isOverSpillThreshold,
    isComfortMarkerSmart,
  } = form;
  const keyframeCount = keyframes.conditioningImages.length;
  // A2V+IC-LoRA combined mode: the two are no longer mutually exclusive, so
  // the badge is a list (both can show at once) rather than a single
  // exclusive choice. IC-LoRA and A2V are judged independently; only when
  // neither holds does the T2V/I2V default apply.
  const modeBadges: string[] = [];
  if (form.isICLora) modeBadges.push(strings.single.keyframes.modeICLora);
  if (form.isA2v) modeBadges.push(strings.single.keyframes.modeA2V);
  if (modeBadges.length === 0) {
    modeBadges.push(
      keyframeCount === 0 ? strings.single.keyframes.modeT2V : strings.single.keyframes.modeI2V(keyframeCount),
    );
  }
  const modeBadgeAccent = form.isICLora || keyframeCount > 0;

  return (
    <div className="generation-form">
      {configFallbackWarning && <p className="warning-banner">{strings.single.configFallbackWarning}</p>}

      <div className="mode-badge-row">
        {modeBadges.map((label) => (
          <span key={label} className={`mode-badge${modeBadgeAccent ? " mode-badge-i2v" : ""}`}>
            {label}
          </span>
        ))}
      </div>

      <PresetDropdown
        config={config}
        onSelect={onApplyPreset}
        disabled={disabled}
        label={strings.single.presets.label}
        placeholder={strings.single.presets.placeholder}
        orientationLabels={strings.single.presets.orientation}
      />

      <SizeFields
        width={values.width}
        height={values.height}
        minWidth={limits.minWidth}
        maxWidth={limits.maxWidth}
        minHeight={limits.minHeight}
        maxHeight={limits.maxHeight}
        disabled={disabled}
        onWidthChange={form.setWidth}
        onHeightChange={form.setHeight}
        gettingSize={gettingSize}
        getSizeError={getSizeError}
        onGetSize={() => void form.getSizeFromAviUtl2()}
      />

      <CropOutputField
        value={form.cropOutput}
        onChange={form.setCropOutput}
        maxWidth={values.width}
        maxHeight={values.height}
        disabled={disabled}
      />

      <DurationField
        label={strings.single.duration.label}
        value={values.numFrames}
        min={limits.minNumFrames}
        max={limits.maxNumFrames}
        disabled={disabled}
        onChange={onDurationChange}
        onCommit={onDurationCommit}
        hint={durationHint}
        spillThresholdFrames={spillThresholdFrames}
        isOverSpillThreshold={isOverSpillThreshold}
        // Smart comfort marker (2026-08-18): the smart per-resolution marker
        // and the coarse spill_free_frames fallback claim different things
        // about how much generation slows down past the threshold, so the
        // copy is picked by which one is currently active
        // (`form.isComfortMarkerSmart`) — see `useGenerationForm`'s own doc
        // comment on `isComfortMarkerSmart`.
        spillWarningText={isComfortMarkerSmart ? strings.single.comfortWarningSmart : strings.single.spillWarning}
      />

      <FrameRateSeedFields
        frameRate={values.frameRate}
        seed={values.seed}
        disabled={disabled}
        onFrameRateChange={form.setFrameRate}
        onSeedChange={form.setSeed}
      />

      <ReservedFields />

      <KeyframesPanel
        keyframes={keyframes}
        numFrames={values.numFrames}
        frameRate={values.frameRate}
        frameIdxMultiple={frameIdxMultiple}
        gridOffset={gridOffset}
        disabled={disabled}
      />

      {/* IC-LoRA (reference video) — always rendered now (Gradio-faithful,
          U2). A2V+IC-LoRA now combine (no exclusivity): both accordions stay
          fully usable at once (see `ReferenceVideoSection`). Collapsed by
          default. */}
      <details className="form-accordion" ref={referenceDetailsRef}>
        <summary className="form-accordion-summary">{strings.single.referenceVideo.heading}</summary>
        <div className="form-accordion-body">
          <ReferenceVideoSection
            form={form}
            disabled={disabled}
            resolutionNotes={resolutionNotes}
            nativeBridge={nativeBridge}
          />
        </div>
      </details>

      {/* A2V (source audio) — always rendered, collapsed by default. */}
      <details className="form-accordion" ref={a2vDetailsRef}>
        <summary className="form-accordion-summary">{strings.single.sourceAudio.heading}</summary>
        <div className="form-accordion-body">
          <SourceAudioSection form={form} disabled={disabled} nativeBridge={nativeBridge} />
        </div>
      </details>
    </div>
  );
}

/** The IC-LoRA (`referenceVideo`) reference-clip block. Renders inside the
 * IC-LoRA accordion (heading lives in the `<summary>`). Mirrors Chain's
 * `SourceVideoSection` (choose/change/uploading + a filename + a clear
 * button), plus the "/generate-only" note and the auto-resolution reasons
 * from `deriveGenerationParams`.
 *
 * A2V+IC-LoRA combined mode: the picker stays enabled even with an A2V audio
 * file attached — the two combine (a2v guided by the IC-LoRA reference video)
 * rather than excluding each other. When both are attached a combined-mode
 * note shows, plus a non-blocking spill warning when the generated duration
 * outruns the reference video (isValid is never affected).
 *
 * IC-LoRA UI redesign (第5波): the control-LoRA dropdown is a
 * selection-preserving `<select value={form.controlLora?.name ?? ""}>` —
 * NOT the old one-shot `value=""` `<select>` that injected a prompt tag and
 * immediately snapped back to its placeholder (the owner's original
 * complaint: "選択後にplaceholderへ戻り、有効になったことに気づきにくい").
 * Choosing a name sets `form.controlLora` directly; choosing "none" (the
 * first option) clears it. A weight slider appears right under it once a
 * selection exists — this pairing (dropdown immediately followed by its own
 * slider, both reflecting live state) is what makes the selection visibly
 * "stick" instead of looking like a fire-and-forget action. N3's
 * loras-required warning plus the two opt-in strength sliders remain,
 * gated by `useGenerationForm`'s `referenceVideoNeedsLoras`/
 * `controlLoraNeedsReferenceVideo`. */
function ReferenceVideoSection({
  form,
  disabled,
  resolutionNotes,
  nativeBridge,
}: {
  form: UseGenerationFormResult;
  disabled: boolean;
  resolutionNotes?: string[] | undefined;
  nativeBridge?: NativeBridge | undefined;
}) {
  const strings = useStrings();
  const t = strings.single.referenceVideo;
  const { state, pick, clear, uploadPath } = form.referenceVideo;
  // A2V+IC-LoRA combined mode: the reference-video picker is no longer blocked
  // by an attached audio file — the two combine. `bothAttached` now drives the
  // combined-mode note (and the spill warning below), not any disabling.
  const bothAttached = form.sourceAudio.state.status === "ready" && state.status === "ready";
  // Soft (non-blocking) IC-LoRA spill: the generated duration outruns the
  // reference video, so the tail generates without IC-LoRA guidance. Uses the
  // live `values` (already re-adjusted to the wav length by useGenerationForm),
  // never touches `isValid`. No warning when the reference duration is unknown.
  const genSec = form.values.numFrames / form.values.frameRate;
  const refSec = form.referenceVideoDurationSec;
  const showIcLoraSpill = bothAttached && refSec != null && genSec > refSec;
  // Dropdown-only narrowing: `in-outpainting` has its own dedicated place (the
  // Edit tab's Outpainting panel), so it is excluded HERE — never from
  // `form.controlLoraNames` itself, which the reference-required gate and the
  // hand-typed-tag auto-migration still need unfiltered (see
  // `lora/controlLoras.ts`'s `UI_HIDDEN_CONTROL_LORA_NAMES` doc comment).
  const controlLoraOptions = Array.from(selectableControlLoraNames(form.controlLoraNames));
  const uploading = state.status === "uploading";
  const ready = state.status === "ready";
  const chooseLabel = uploading ? t.uploadingButton : ready ? t.changeButton : t.chooseButton;

  // W6: is there ANY IC-LoRA state to tear down? — a ready reference video, a
  // selected control LoRA, or either opt-in strength. The ❌ button is enabled
  // whenever at least one holds (so it can also mop up an "orphan" left by a
  // bare 🔁 clear: strengths/selection surviving with no reference video), and
  // disabled only when there is nothing at all to clear.
  const hasIcLoraState =
    ready ||
    form.controlLora !== null ||
    form.conditioningAttentionStrength !== null ||
    form.referenceVideoStrength !== null;
  const removeIcLora = () => {
    clear();
    form.setControlLora(null);
    form.setConditioningAttentionStrength(null);
    form.setReferenceVideoStrength(null);
  };

  // IC-LoRA/A2V card redesign: the whole card is the drop target now (mirrors
  // Chain's `SourceInputPanel`), replacing the old `DropZone` wrapper. A
  // dropped video is uploaded straight through `referenceVideo.uploadPath` —
  // the 128-grid re-snap effect (useGenerationForm.ts, keyed off
  // referenceVideoId turning non-null) already reacts to that regardless of
  // how the upload was started, so no extra wiring is needed here.
  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: REFERENCE_VIDEO_EXTENSIONS,
    onFile: (filePath, fileName) => void uploadPath(filePath, fileName),
    disabled: disabled || uploading,
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
      {/* Redesign: choose/change and clear collapsed to two stacked emoji icon
          buttons (🔁 above 📁), same recipe as `SourceInputPanel`. Text is
          carried over unchanged as `aria-label`/`title`. */}
      <div className="source-input-actions">
        {/* W6: ❌ fully tears IC-LoRA down (reference + control LoRA + both
            strengths) in one click — enabled whenever any of that state exists,
            so it also clears an orphan left by a bare 🔁 (which only clears the
            reference video). 🔁's own condition is unchanged. */}
        <button
          type="button"
          className="icon-action-button"
          title={t.removeIcLoraButton}
          aria-label={t.removeIcLoraButton}
          disabled={disabled || uploading || !hasIcLoraState}
          onClick={removeIcLora}
        >
          <span aria-hidden="true">❌</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading || !ready}
          onClick={clear}
        >
          <span aria-hidden="true">🔁</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void pick()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      <p className="field-hint">{t.icLoraNote}</p>
      <p className="field-hint">{t.controlAdapterAspectHint}</p>
      <p className="field-hint">{t.controlAdapterDepthHint}</p>
      <p className="field-hint">{t.controlAdapterDeblurHint}</p>

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
          {state.fileName && <p className="field-hint">{state.fileName}</p>}
          {ready && form.referenceVideoDurationSec != null && (
            <p className="field-hint">{formatSecondsLabel(form.referenceVideoDurationSec)}</p>
          )}
        </div>
      </div>

      {/* Card-wide D&D's own error line (redesign) — the same two error cases/
          copy `DropZone` used to show, rendered directly now that the card
          itself is the drop target. */}
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
              form.setControlLora(name ? { name, strength: form.controlLora?.strength ?? DEFAULT_CONTROL_LORA_STRENGTH } : null);
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

      {bothAttached && <p className="field-hint">{t.blockedByAudioNote}</p>}
      {showIcLoraSpill && refSec != null && (
        <p className="warning-banner warning-banner-mild">{t.icLoraSpillWarning(genSec.toFixed(1), refSec.toFixed(1))}</p>
      )}
      {state.status === "idle" && <p className="field-hint">{t.none}</p>}
      {state.status === "error" && <p className="field-hint field-hint-error">{state.errorCode}</p>}
      {resolutionNotes && resolutionNotes.length > 0 && (
        <p className="field-hint">{resolutionNotes.join(" / ")}</p>
      )}
      {form.controlLoraNeedsReferenceVideo && (
        <p className="warning-banner">{t.referenceVideoRequiredWarning}</p>
      )}
      {state.status === "ready" && form.referenceVideoNeedsLoras && (
        <p className="warning-banner">{t.lorasRequiredWarning}</p>
      )}

      {state.status === "ready" && (
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

/** Group3 item11: the always-present, optional A2V source-audio block
 * (Gradio-faithful — an attachment slot the user can leave empty for a plain
 * T2V/I2V generation, or fill to switch the whole form into A2V). Renders
 * inside the A2V accordion (heading lives in the `<summary>`). Mirrors
 * `ReferenceVideoSection`'s choose/change/uploading/filename/clear row.
 * A2V+IC-LoRA combined mode: the picker stays enabled even with a reference
 * video active (the two combine); when both are attached a combined-mode note
 * shows. Also shows the "audio too short" warning from `audioPrecheck` once a
 * measured wav can't cover the current duration. */
function SourceAudioSection({
  form,
  disabled,
  nativeBridge,
}: {
  form: UseGenerationFormResult;
  disabled: boolean;
  nativeBridge?: NativeBridge | undefined;
}) {
  const strings = useStrings();
  const t = strings.single.sourceAudio;
  const { state, pick, clear } = form.sourceAudio;
  // A2V+IC-LoRA combined mode: the audio picker is no longer blocked by an
  // active reference video — the two combine. `bothAttached` now drives the
  // combined-mode note, not any disabling.
  const bothAttached = form.referenceVideo.state.status === "ready" && state.status === "ready";
  const uploading = state.status === "uploading";
  const ready = state.status === "ready";
  const chooseLabel = uploading ? t.uploadingButton : ready ? t.changeButton : t.chooseButton;
  // Bug fix (owner real-device report, 2026-07-18): a rejected drop's
  // "Unsupported file type" message otherwise survives a "Remove audio"
  // click — `clear()` only resets `sourceAudio.state`, which is entirely
  // independent of `useFileDrop`'s own internal `error` state (that only
  // clears at the start of the NEXT drop). Bumping this token on Clear and
  // threading it into `useFileDrop` as `resetErrorSignal` force-clears it too.
  const [dropResetToken, setDropResetToken] = useState(0);

  // IC-LoRA/A2V card redesign: whole-card drop target (mirrors Chain's
  // `SourceInputPanel`), replacing the old `DropZone` wrapper. Routed through
  // `form.attachSourceAudioByPath` (NOT `sourceAudio.uploadPath` directly) — it
  // also primes `lastAudioFilePathRef` so the wav-duration auto-adjust probe
  // still fires for a dropped file (see that function's doc comment in
  // useGenerationForm.ts for why this is the easiest spot to regress).
  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({
    accept: SOURCE_AUDIO_EXTENSIONS,
    onFile: (filePath, fileName) => void form.attachSourceAudioByPath(filePath, fileName),
    disabled: disabled || uploading,
    nativeBridge,
    resetErrorSignal: dropResetToken,
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
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading || !ready}
          onClick={() => {
            clear();
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
          onClick={() => void pick()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

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
          {state.fileName && <p className="field-hint">{state.fileName}</p>}
          {/* A2V duration is wav-only (owner-approved): a non-wav attach leaves
              `audioDurationSec` null, so the seconds line simply doesn't show. */}
          {ready && form.audioDurationSec != null && (
            <p className="field-hint">{formatSecondsLabel(form.audioDurationSec)}</p>
          )}
        </div>
      </div>

      {dropError && (
        <p className="field-hint field-hint-error">
          {dropError === "unsupported" ? strings.dnd.unsupportedAudio : strings.dnd.resolveFailed}
        </p>
      )}
      {state.status === "idle" && <p className="field-hint">{t.none}</p>}
      {state.status === "error" && <p className="field-hint field-hint-error">{state.errorCode}</p>}
      {bothAttached && <p className="field-hint">{t.exclusiveWithReferenceNote}</p>}
      {form.isA2v && <p className="field-hint">{t.singleClipNote}</p>}
      {form.audioPrecheck && !form.audioPrecheck.ok && (
        <p className="warning-banner warning-banner-mild">{t.tooShortWarning(form.audioPrecheck.requiredSeconds)}</p>
      )}
    </div>
  );
}
