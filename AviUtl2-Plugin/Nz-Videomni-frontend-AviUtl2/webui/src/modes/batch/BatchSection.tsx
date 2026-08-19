import { useEffect, useState } from "react";
import type { AppConfig } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import type { NagSettings } from "../../shell/nagSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import type { UseKeyframesResult } from "../single/useKeyframes";
import { BatchTable } from "./BatchTable";
import { useBatchForm } from "./useBatchForm";
import type { BatchGenerationValues } from "./useBatchForm";
import "./BatchSection.css";

export interface BatchSectionProps {
  /** Reused from `useConfig()` at the `SingleScreen` level — this component
   * never fetches its own config (task brief: "組み込みのみ"). */
  config: AppConfig;
  /** The app's single shared `PromptBar` value — the batch's common prompt
   * (Add/Replace-combined with each row's own `prompt` column), mirroring
   * `gradio_ui/ui.py`'s batch closure reusing the Generate tab's own
   * `prompt` component rather than a separate batch-local textbox. */
  prompt: string;
  /** U4: width/height/frameRate/seed come from the Create form now (the batch
   * no longer owns its own resolution/fps/seed fields). */
  generationValues: BatchGenerationValues;
  /** U4 guard 3: true while a reference video (IC-LoRA, 128 grid) is active on
   * the Create form — shown as a non-blocking warning next to Start. */
  icLoraActive: boolean;
  /** Batch A2V Shared spec (2026-07-18): the Create screen's own `useKeyframes`
   * instance (its KEYFRAMES panel). Batch has no keyframe panel of its own — a
   * `Shared`-image row reuses this panel's first image as the leading frame.
   * Passed straight into `useBatchForm`. */
  keyframes: UseKeyframesResult;
  nativeBridge?: NativeBridge | undefined;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings.
   * Batch has no NAG UI of its own — it silently inherits whatever Create's
   * accordion currently holds (owner decision §UX 1), the same way it already
   * does for width/height/frameRate/seed via `generationValues`. Passed
   * straight into `useBatchForm`, which defaults an omitted `nag` to the
   * frozen `NAG_OFF` sentinel. */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31): the Settings panel's shared attention-backend
   * choice, threaded straight into `useBatchForm` (which defaults an omitted
   * value to the frozen `ACCELERATION_DEFAULTS` sentinel) — Batch has no
   * acceleration UI of its own, same as `nag` above. */
  acceleration?: AccelerationSettings | undefined;
  /** §1-7 相互ロック 第2段 (2026-07-31): `JobsContext.hasActiveJob` — a
   * generation (single or from the other batch) is in flight, so this batch may
   * not start. The exact counterpart of `BatchI2vLongSection`'s prop of the
   * same name; see `UseBatchFormDeps.hasActiveJob` for why the shared run lock
   * alone was not enough. */
  hasActiveJob?: boolean;
}

/**
 * The "Batch A2V" panel (webui-B): a collapsed-by-default `<details>` section
 * on the Create screen (task brief: "既定閉") that scans a folder of audio
 * files into a CSV-manifest-backed queue and drives them through
 * `POST /generate/chain` one at a time via {@link useBatchForm}. All state
 * ownership lives in `useBatchForm`/`useBatchRunner`; this component is pure
 * rendering + event wiring, matching the `*Screen.tsx` convention elsewhere in
 * this codebase (`ChainedScreen.tsx`/`SingleScreen.tsx`).
 *
 * U4: the width/height/FPS/SEED fields were removed — those values come from
 * the Create form (`generationValues`). The three folder rows gained hand-typed
 * text inputs (onBlur-confirmed) alongside their 📁 pickers, and three guards
 * (off-grid resolution / stale-FPS / IC-LoRA-active) are surfaced next to Start.
 */
export function BatchSection({
  config,
  prompt,
  generationValues,
  icLoraActive,
  keyframes,
  nativeBridge,
  nag,
  acceleration,
  hasActiveJob = false,
}: BatchSectionProps) {
  const strings = useStrings();
  const t = strings.batch;
  const form = useBatchForm(config, prompt, generationValues, keyframes, {
    ...(nativeBridge !== undefined ? { nativeBridge } : {}),
    ...(nag !== undefined ? { nag } : {}),
    ...(acceleration !== undefined ? { acceleration } : {}),
    hasActiveJob,
  });

  // The manifest's `stat` column is only safe to hand-edit (folder pickers,
  // scan, row resets, settings) while the runner isn't actively writing it —
  // mirrors every other mode's `disabled` gate on "isGenerating". Also gated
  // on `isScanning` (L1 remediation): `scan()`'s async merge (`setRows`) can
  // land mid-edit and clobber it otherwise.
  const disabled = form.runnerState !== "idle" || form.isScanning;

  return (
    <details className="batch-section">
      <summary className="batch-section-summary">{t.heading}</summary>
      <div className="batch-section-body">
        <p className="field-hint batch-notice">{t.notice}</p>

        <FolderRow
          label={t.wavDir.label}
          value={form.wavDir}
          disabled={disabled}
          pickTitle={t.wavDir.button}
          onPick={() => void form.pickWavDir(t.wavDir.button)}
          onCommit={(value) => void form.setWavDir(value)}
          hint={!form.wavDir ? t.wavDir.none : null}
        />

        <FolderRow
          label={t.imgDir.label}
          value={form.imgDir}
          disabled={disabled}
          pickTitle={t.imgDir.button}
          onPick={() => void form.pickImgDir(t.imgDir.button)}
          onCommit={(value) => void form.setImgDir(value)}
          hint={!form.imgDir ? t.imgDir.none : null}
        />

        <FolderRow
          label={t.outDir.label}
          value={form.outDir}
          disabled={disabled}
          pickTitle={t.outDir.button}
          onPick={() => void form.pickOutDir(t.outDir.button)}
          onCommit={(value) => void form.setOutDir(value)}
          hint={form.outDir ? (form.outDirIsAuto ? t.outDir.auto(form.outDir) : null) : t.outDir.none}
        />

        <div className="field">
          <span className="field-label">{t.promptMode.label}</span>
          <div className="batch-radio-row">
            <label className="field field-inline">
              <input
                type="radio"
                name="batch-prompt-mode"
                checked={form.values.promptMode === "add"}
                disabled={disabled}
                onChange={() => form.setPromptMode("add")}
              />
              <span>{t.promptMode.add}</span>
            </label>
            <label className="field field-inline">
              <input
                type="radio"
                name="batch-prompt-mode"
                checked={form.values.promptMode === "replace"}
                disabled={disabled}
                onChange={() => form.setPromptMode("replace")}
              />
              <span>{t.promptMode.replace}</span>
            </label>
          </div>
        </div>

        <label className="field field-inline">
          <input
            type="checkbox"
            checked={form.values.chunkedUpsample}
            disabled={disabled}
            onChange={(e) => form.setChunkedUpsample(e.target.checked)}
          />
          <span className="field-label">{strings.chained.chunkedUpsampleLabel}</span>
        </label>
        <p className="field-hint">{strings.chained.chunkedUpsampleHint}</p>

        {/* Batch A2V Shared spec (2026-07-18): no keyframe panel of its own —
            a `Shared`-image row reuses the Create screen's KEYFRAMES panel
            (first image → leading frame). Just a one-line note explaining that,
            plus the "no ready image yet" warning when a Shared row is queued. */}
        <p className="field-hint">{t.sharedKeyframes.note}</p>
        {form.sharedKeyframeMissing && <p className="warning-banner">{t.sharedKeyframes.missingWarning}</p>}

        <div className="field">
          <button type="button" className="secondary-button" disabled={!form.canScan} onClick={() => void form.scan()}>
            {form.isScanning ? t.scanningButton : t.scanButton}
          </button>
          {form.scanError && <p className="warning-banner">{t.scanError(form.scanError)}</p>}
        </div>

        <BatchTable
          rows={form.rows}
          disabled={disabled}
          imageOptions={form.imageOptions}
          onResetRow={(queue) => form.resetRowToWaiting(queue)}
          onUpdateRowPrompt={(index, value) => form.setRowPromptLocal(index, value)}
          onCopyCommonPrompt={(index) => form.copyCommonPromptToRow(index, prompt)}
          onUpdateRowImage={(index, imageName) => form.updateRowImage(index, imageName)}
        />

        <p className="field-hint">{t.summary(form.summary.done, form.summary.total)}</p>
        {form.currentRow && <p className="field-hint">{t.currentRow(form.currentRow.wav)}</p>}

        {/* U4 guards. Only resolutionOffGrid still blocks `canStart`; the
            fpsMismatch and IC-LoRA-active notes are informational only
            (fpsMismatch is auto-corrected by the start-time re-judgment). */}
        {!form.resolutionValid && <p className="warning-banner">{t.resolutionOffGrid}</p>}
        {form.fpsMismatch && <p className="field-hint">{t.fpsMismatch}</p>}
        {icLoraActive && <p className="warning-banner warning-banner-mild">{t.icLoraActiveWarning}</p>}
        {/* NAG (2026-07-28, D7): Batch has no `GenerateReasonsNote` (that's
            Create/Chain's own convention) — just a `canStart` gate plus this
            one-line warning, mirroring the resolution/IC-LoRA guards above. */}
        {form.nagInvalid && <p className="warning-banner">{strings.single.generateReasons.nagNegativeEmpty}</p>}
        {/* §1-7 相互ロック: the Chain screen's Batch i2v-long is running, so the
            shared run lock (`shell/runLock.ts`) is held by it. Blocks Start —
            shown as its own line so the disabled button is explained. */}
        {form.lockedByOther && <p className="warning-banner">{t.lockedByOther}</p>}
        {/* §1-7 相互ロック 第2段: the backend's single job slot is busy (the
            other batch, a single generation, or a job that outlived a reload).
            Only shown while THIS panel is idle — during its own run the Start
            button already reads "Running…", and repeating it here would just be
            noise. Kept a separate line from `lockedByOther` because the two say
            different things: that one names the other batch panel, this one
            covers every other way the slot can be occupied. */}
        {form.jobActive && form.runnerState === "idle" && !form.lockedByOther && (
          <p className="warning-banner">{t.jobActive}</p>
        )}

        <div className="generate-row">
          <button type="button" className="primary-button generate-button" disabled={!form.canStart} onClick={form.start}>
            {form.runnerState === "running" ? t.runningButton : form.runnerState === "stopping" ? t.stoppingButton : t.startButton}
          </button>
          {form.runnerState !== "idle" && (
            <button
              type="button"
              className="secondary-button"
              disabled={form.runnerState === "stopping"}
              onClick={form.stop}
            >
              {t.stopButton}
            </button>
          )}
        </div>
      </div>
    </details>
  );
}

interface FolderRowProps {
  label: string;
  /** The committed folder path (or null) owned by `useBatchForm`. */
  value: string | null;
  disabled: boolean;
  /** Tooltip/aria-label for the 📁 pick button (reuses the existing per-folder
   * "Choose … folder…" string). */
  pickTitle: string;
  onPick: () => void;
  /** onBlur commit of the hand-typed path (U4). */
  onCommit: (value: string) => void;
  /** Secondary hint line under the row (none/auto guidance). */
  hint?: string | null;
}

/** One "label + hand-typable text input + 📁 picker (+ optional clear)" folder
 * row (U4 §7.1). The text input keeps a local `draft` so keystrokes don't churn
 * `useBatchForm`'s folder state or re-derive `outDir`; the draft is committed on
 * blur and re-synced whenever the committed value changes externally (e.g. via
 * the 📁 picker or auto-derivation). */
function FolderRow({ label, value, disabled, pickTitle, onPick, onCommit, hint }: FolderRowProps) {
  const [draft, setDraft] = useState(value ?? "");
  useEffect(() => {
    setDraft(value ?? "");
  }, [value]);

  return (
    <div className="field source-section">
      <span className="field-label">{label}</span>
      <div className="folder-row">
        <input
          type="text"
          className="folder-input"
          value={draft}
          disabled={disabled}
          aria-label={label}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => onCommit(draft)}
        />
        <button
          type="button"
          className="secondary-button folder-pick"
          disabled={disabled}
          title={pickTitle}
          aria-label={pickTitle}
          onClick={onPick}
        >
          📁
        </button>
      </div>
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  );
}
