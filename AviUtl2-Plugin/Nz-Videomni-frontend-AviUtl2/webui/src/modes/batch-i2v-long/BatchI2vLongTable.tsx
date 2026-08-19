import { useStrings } from "../../i18n/LanguageContext";
import type { I2vLongRow, I2vLongStat } from "./imageRows";

/** Stat icon, the same at-a-glance vocabulary Batch A2V's `BatchTable.tsx`
 * (and `gradio_ui/ui.py`'s `_STAT_ICON`) use. No `⛔Skip` — this feature's row
 * model has no `Skip` state (`imageRows.ts`). */
const STAT_ICON: Record<I2vLongStat, string> = {
  Waiting: "⚪",
  Generating: "⏳",
  Done: "✅",
  Failed: "❌",
};

/** Rows 🔁 can put back into the queue — same set as
 * `useBatchI2vLongForm.resetRowToWaiting` honours. */
const RESETTABLE_STATS: ReadonlySet<I2vLongStat> = new Set<I2vLongStat>(["Done", "Failed"]);

/** The row prompt `<input>`'s cap — the server's own `prompt` limit
 * (`api/models.py`, mirrored by `useBatchI2vLongForm.MAX_PROMPT_LENGTH`).
 * Duplicated as a literal rather than imported so this presentational component
 * keeps depending on nothing but the row model; the composed-length guard in
 * the hook is what actually protects the request. */
const MAX_ROW_PROMPT_LENGTH = 2000;

export interface BatchI2vLongTableProps {
  rows: I2vLongRow[];
  /** True while a run is in flight (or a scan is landing) — disables every
   * row's 🔁 AND its prompt editors, matching Batch A2V's global (not per-row)
   * edit gate. */
  disabled: boolean;
  onResetRow: (queue: number) => void;
  /** Row `index`'s prompt cell edited to `value` (a controlled `<input>`'s
   * `onChange`) — same signature as A2V's `onUpdateRowPrompt`. */
  onUpdateRowPrompt: (index: number, value: string) => void;
  /** 📝 on row `index`: fill it with the Clip Chain's shared prompt. */
  onCopyChainPrompt: (index: number) => void;
}

/**
 * Batch i2v-long's queue table: six columns (`#` / image / prompt / status /
 * output / 🔁). Narrower than Batch A2V's seven-column manifest table — there
 * is no per-row image `<select>` and no duration, because every generation
 * parameter comes from the Chain form and each row is simply "one image -> one
 * clip chain" — but the prompt column is A2V's, editor and 📝 button included
 * (2026-07-30 owner feedback: the prompt has to be per image).
 *
 * Markup/classes are A2V's `BatchTable.tsx` verbatim (`.batch-table*`,
 * `.batch-stat-cell`, `.icon-action-button`), which is why this file imports no
 * CSS of its own — `BatchI2vLongSection` pulls in `../batch/BatchSection.css`
 * for both of them.
 */
export function BatchI2vLongTable({ rows, disabled, onResetRow, onUpdateRowPrompt, onCopyChainPrompt }: BatchI2vLongTableProps) {
  const strings = useStrings();
  const t = strings.batchI2vLong.table;

  if (rows.length === 0) {
    return <p className="field-hint">{t.empty}</p>;
  }

  return (
    <div className="batch-table-scroll">
      <table className="batch-table">
        <thead>
          <tr>
            <th>{t.queue}</th>
            <th>{t.image}</th>
            <th>{t.prompt}</th>
            <th>{t.stat}</th>
            <th>{t.output}</th>
            <th aria-hidden="true" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.queue}>
              <td>{row.queue}</td>
              <td className="batch-table-wav">{row.image}</td>
              <td className="batch-table-prompt-cell">
                <input
                  type="text"
                  className="batch-table-prompt-input"
                  aria-label={t.promptInputLabel}
                  value={row.prompt}
                  maxLength={MAX_ROW_PROMPT_LENGTH}
                  disabled={disabled}
                  onChange={(e) => onUpdateRowPrompt(index, e.target.value)}
                />
                <button
                  type="button"
                  className="icon-action-button batch-copy-prompt-button"
                  title={t.copyChainPromptButton}
                  aria-label={t.copyChainPromptButton}
                  disabled={disabled}
                  onClick={() => onCopyChainPrompt(index)}
                >
                  <span aria-hidden="true">📝</span>
                </button>
              </td>
              <td>
                {/* A `Failed` row's full error text is the cell's tooltip —
                    the table itself stays one line per row even when the
                    failure reason is a long server message. */}
                <span className="batch-stat-cell" title={row.stat === "Failed" && row.error ? row.error : undefined}>
                  {STAT_ICON[row.stat]} {strings.batchI2vLong.stat[statKey(row.stat)]}
                </span>
              </td>
              <td className="batch-table-truncate">{row.output}</td>
              <td>
                <button
                  type="button"
                  className="icon-action-button batch-reset-button"
                  title={t.resetButton}
                  aria-label={t.resetButton}
                  disabled={disabled || !RESETTABLE_STATS.has(row.stat)}
                  onClick={() => onResetRow(row.queue)}
                >
                  <span aria-hidden="true">🔁</span>
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function statKey(stat: I2vLongStat): "waiting" | "generating" | "done" | "failed" {
  switch (stat) {
    case "Waiting":
      return "waiting";
    case "Generating":
      return "generating";
    case "Done":
      return "done";
    case "Failed":
      return "failed";
  }
}
