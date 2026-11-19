import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import type { Strings } from "../../i18n/strings";
import { buildChainReasonMessages } from "../chained/generateReasonMessages";
import { GenerateReasonsNote } from "../single/GenerateReasonsNote";
import { referenceTag } from "../single/referenceRules";
import { FolderInputs } from "./FolderInputs";
import { V2vRefLongTable } from "./V2vRefLongTable";
import type { BaseField, UseV2vRefLongFormResult } from "./types";
import "../batch/BatchSection.css";
import "../../styles/formAccordion.css";
import "./BatchV2vRefLongSection.css";

export interface V2vRefLongBodyProps {
  form: UseV2vRefLongFormResult;
  nativeBridge?: NativeBridge | undefined;
}

/** A field's name in the "no files" notice. */
function emptyFieldLabel(t: Strings["batchV2vRefLong"], field: BaseField): string {
  if (field.type === "source") return t.videoDir.label;
  return t.table.clipFieldLabel(field.frame + 1, referenceTag(field.kind, field.slot));
}

/**
 * Batch v2v-ref-long's section body: folders, scan, the row table, progress,
 * the Start blockers and Start/Stop. Rendering only — the section shell owns
 * the hook and hands its result in as `form`.
 */
export function V2vRefLongBody({ form, nativeBridge }: V2vRefLongBodyProps) {
  const strings = useStrings();
  const t = strings.batchV2vRefLong;
  const { state } = form;
  const runState = state.run.state;
  const disabled = form.inputsLocked;

  const reasonMessages: Record<string, string> = {
    // `minClips` is never kept (`CHAIN_REASON_POLICY`), so its number is unused.
    ...buildChainReasonMessages(strings, {
      minFramesForOverlap: form.minFramesForOverlap,
      minClips: 1,
      fixedOverlapFrames: form.fixedOverlapFrames,
    }),
    ...t.blockReasons,
    ...(form.promptEmptyRows.length > 0
      ? { promptEmpty: t.promptRowIssues.empty(form.promptEmptyRows.map((row) => row + 1)) }
      : {}),
  };
  const allReasons = [...form.blockReasons, ...form.chainBlockReasons];

  const progress = state.scanProgress;
  const currentRow = form.currentRow;
  const currentName = currentRow
    ? (form.rowViews.find((view) => view.row.index === currentRow.index)?.baseFileName ?? `#${currentRow.index + 1}`)
    : null;

  return (
    <div className="v2v-ref-long-body">
      <FolderInputs form={form} />

      <div className="field">
        <div>
          <button type="button" className="secondary-button" disabled={!form.canScan} onClick={() => void form.scan()}>
            {state.scanning ? t.scanningButton(progress?.done ?? 0, progress?.total ?? 0) : t.scanButton}
          </button>
        </div>
        {state.scanError && <p className="warning-banner">{t.scanError(state.scanError)}</p>}
        {form.emptyFields.map((field) => {
          const label = emptyFieldLabel(t, field);
          return (
            <p className="warning-banner warning-banner-mild" key={label}>
              {t.emptyField(label)}
            </p>
          );
        })}
        {form.allFieldsEmpty && <p className="warning-banner warning-banner-mild">{t.allFieldsEmpty}</p>}
      </div>

      <div className="field">
        <span className="field-label">{t.table.clipCount(form.clipCount)}</span>
        <div className="batch-radio-row">
          <span className="field-label">{t.table.promptMode.label}</span>
          {(["add", "replace"] as const).map((mode) => (
            <label className="field field-inline" key={mode}>
              <input
                type="radio"
                name="batch-v2v-ref-long-main-prompt-mode"
                checked={state.mainPromptMode === mode}
                disabled={disabled}
                onChange={() => form.setMainPromptMode(mode)}
              />
              <span>{t.table.promptMode[mode]}</span>
            </label>
          ))}
        </div>
      </div>

      <V2vRefLongTable form={form} nativeBridge={nativeBridge} />

      <p className="field-hint">{t.summary(form.summary.done, form.summary.total)}</p>
      {currentName !== null && <p className="field-hint">{t.currentRow(currentName)}</p>}

      <p className="field-hint">{t.notes.concurrency}</p>
      {form.seedFixed && <p className="field-hint">{t.notes.seedFixed}</p>}

      <GenerateReasonsNote reasons={allReasons} messages={reasonMessages} />

      <div className="generate-row">
        <button type="button" className="primary-button generate-button" disabled={!form.canStart} onClick={() => form.start()}>
          {runState === "running" ? t.runningButton : runState === "stopping" ? t.stoppingButton : t.startButton}
        </button>
        {runState !== "idle" && (
          <button type="button" className="secondary-button" disabled={runState === "stopping"} onClick={() => form.stop()}>
            {t.stopButton}
          </button>
        )}
      </div>
    </div>
  );
}
