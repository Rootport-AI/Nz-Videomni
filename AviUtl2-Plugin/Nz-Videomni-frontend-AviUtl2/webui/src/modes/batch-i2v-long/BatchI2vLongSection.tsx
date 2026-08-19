import { useEffect, useState } from "react";
import type { AppConfig } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { GenerateReasonsNote } from "../single/GenerateReasonsNote";
import { buildChainReasonMessages } from "../chained/generateReasonMessages";
import { BatchI2vLongTable } from "./BatchI2vLongTable";
import type { ChainSnapshotSource } from "./chainSnapshot";
import { useBatchI2vLongForm } from "./useBatchI2vLongForm";
// The whole visual language (the `<details>` shell, the folder rows, the queue
// table) is Batch A2V's, reused verbatim — this stylesheet is the ONLY thing
// this feature takes from `modes/batch/`.
import "../batch/BatchSection.css";

export interface BatchI2vLongSectionProps {
  /** Reused from the Chain screen's own `useConfig()` — this panel never
   * fetches its own config. Only `upload.allowed_image_extensions` /
   * `upload.max_image_size_mb` are read (the folder scan). */
  config: AppConfig;
  /** The Chain screen's live `useChainForm` result, seen through the narrow
   * structural {@link ChainSnapshotSource}. Every generation parameter comes
   * from here; this panel owns no size/length/seed field of its own. */
  chain: ChainSnapshotSource;
  /** `JobsContext.hasActiveJob` — a single generation (or another batch's row)
   * is in flight, so this batch may not start. */
  hasActiveJob: boolean;
  nativeBridge?: NativeBridge | undefined;
}

/**
 * The "Batch i2v-long" panel (§1-7): a collapsed-by-default `<details>` section
 * on the CHAIN screen that walks a folder of images and generates one long clip
 * chain per image, serially.
 *
 * Deliberately NOT co-located with Batch A2V (which lives on Create): the two
 * are separate features with separate row models, and the settings each borrows
 * come from the screen it sits on. All state lives in `useBatchI2vLongForm` /
 * `runtime.ts`; this component is pure rendering + event wiring, matching the
 * `BatchSection.tsx` convention it mirrors.
 *
 * Must be rendered OUTSIDE `.single-layout` (the Chain screen's 2-column grid),
 * as a full-width sibling below it — see `ChainedScreen.tsx`'s fragment.
 */
export function BatchI2vLongSection({ config, chain, hasActiveJob, nativeBridge }: BatchI2vLongSectionProps) {
  const strings = useStrings();
  const t = strings.batchI2vLong;
  const form = useBatchI2vLongForm(config, chain, hasActiveJob, {
    ...(nativeBridge !== undefined ? { nativeBridge } : {}),
  });

  // Everything is hand-editable only while the runner isn't writing rows (and
  // while a scan's async `setRows` can't land mid-edit) — the same global gate
  // Batch A2V applies.
  const disabled = form.runnerState !== "idle" || form.isScanning;

  // The block-reason list merges this panel's own codes with the Chain screen's
  // `validityReasons`, rendered through the SAME `GenerateReasonsNote` +
  // Chain wording the Generate button uses — so "why can't I start?" reads
  // identically in both places.
  const reasonMessages: Record<string, string> = {
    ...buildChainReasonMessages(strings, {
      minFramesForOverlap: chain.minFramesForOverlap,
      minClips: chain.minClips,
    }),
    ...t.blockReasons,
    // Once the folder is scanned the panel knows exactly WHICH rows carry an
    // empty/over-long prompt, so the generic line is replaced by one naming
    // them (2026-07-30 row-prompt redesign).
    ...(form.promptEmptyQueues.length > 0
      ? { promptEmpty: t.promptRowIssues.empty(formatQueues(form.promptEmptyQueues)) }
      : {}),
    ...(form.promptTooLongQueues.length > 0
      ? { promptTooLong: t.promptRowIssues.tooLong(formatQueues(form.promptTooLongQueues)) }
      : {}),
  };
  const allReasons = [...form.blockReasons, ...form.chainBlockReasons];

  return (
    <details className="batch-section">
      <summary className="batch-section-summary">{t.heading}</summary>
      <div className="batch-section-body">
        {/* 2026-07-30 owner feedback: the two explanatory paragraphs that used
            to open this panel are gone (they are in
            `Docs/BATCH_I2V_WORKORDER.md` instead). The one-line chain summary
            stays — it shows the SETTINGS actually in effect, which no document
            can. */}
        <p className="field-hint">{t.chainSummary(chain.clips.length, chain.outputFrames, chain.outputSeconds)}</p>

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

        {/* One add/replace mode for the whole batch; the TEXT is per row, in
            the table's Prompt column below (Batch A2V's arrangement). */}
        <div className="field">
          <span className="field-label">{t.promptMode.label}</span>
          <div className="batch-radio-row">
            <label className="field field-inline">
              <input
                type="radio"
                name="batch-i2v-long-prompt-mode"
                checked={form.promptMode === "add"}
                disabled={disabled}
                onChange={() => form.setPromptMode("add")}
              />
              <span>{t.promptMode.add}</span>
            </label>
            <label className="field field-inline">
              <input
                type="radio"
                name="batch-i2v-long-prompt-mode"
                checked={form.promptMode === "replace"}
                disabled={disabled}
                onChange={() => form.setPromptMode("replace")}
              />
              <span>{t.promptMode.replace}</span>
            </label>
          </div>
        </div>

        <div className="field">
          <button type="button" className="secondary-button" disabled={!form.canScan} onClick={() => void form.scan()}>
            {form.isScanning ? t.scanningButton : t.scanButton}
          </button>
          {form.scanError && <p className="warning-banner">{t.scanError(form.scanError)}</p>}
        </div>

        <BatchI2vLongTable
          rows={form.rows}
          disabled={disabled}
          onResetRow={(queue) => form.resetRowToWaiting(queue)}
          onUpdateRowPrompt={(index, value) => form.setRowPromptLocal(index, value)}
          onCopyChainPrompt={(index) => form.copyChainPromptToRow(index)}
        />

        <p className="field-hint">{t.summary(form.summary.done, form.summary.total)}</p>
        {form.currentRow && <p className="field-hint">{t.currentRow(form.currentRow.image)}</p>}

        {/* Advisory notes — none of these blocks Start. `clip0PromptOverride`
            is the strong one (a warning banner rather than a hint): the row's
            image always lands on clip 0, so an override there means the batch
            prompt never touches the part of the video the image controls. */}
        {form.notes.clip0PromptOverride && <p className="warning-banner">{t.notes.clip0PromptOverride}</p>}
        {form.notes.seedFixed && <p className="field-hint">{t.notes.seedFixed}</p>}
        {form.notes.otherClipPromptOverride && <p className="field-hint">{t.notes.clipPromptOverride}</p>}
        {form.notes.startFrameIgnored && <p className="field-hint">{t.notes.startFrameIgnored}</p>}
        <p className="field-hint">{t.notes.concurrency}</p>

        <GenerateReasonsNote reasons={allReasons} messages={reasonMessages} />

        <div className="generate-row">
          <button type="button" className="primary-button generate-button" disabled={!form.canStart} onClick={form.start}>
            {form.runnerState === "running" ? t.runningButton : form.runnerState === "stopping" ? t.stoppingButton : t.startButton}
          </button>
          {form.runnerState !== "idle" && (
            <button type="button" className="secondary-button" disabled={form.runnerState === "stopping"} onClick={form.stop}>
              {t.stopButton}
            </button>
          )}
        </div>
      </div>
    </details>
  );
}

/** Renders a row-number list for the two prompt block reasons — `#` prefixed
 * (matching the table's own `#` column header) and capped at 10 entries so a
 * folder of 200 images cannot turn one reason line into a wall of numbers. */
function formatQueues(queues: number[], limit = 10): string {
  const shown = queues.slice(0, limit).map((q) => `#${q}`).join(", ");
  return queues.length > limit ? `${shown}, …(+${queues.length - limit})` : shown;
}

interface FolderRowProps {
  label: string;
  value: string | null;
  disabled: boolean;
  pickTitle: string;
  onPick: () => void;
  onCommit: (value: string) => void;
  hint?: string | null;
}

/** One "label + hand-typable text input + 📁 picker" folder row. Local copy of
 * `modes/batch/BatchSection.tsx`'s own `FolderRow` (:222-255) — that one is not
 * exported, and Batch A2V is outside this sprint's edit scope, so duplicating
 * the ~30 lines is preferable to reaching into another feature's module. The
 * local `draft` keeps keystrokes from churning the form's folder state (which
 * would re-derive `outDir` on every character); it is committed on blur and
 * re-synced whenever the committed value changes externally (📁 picker,
 * auto-derivation, or a remount re-attaching to a running batch). */
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
