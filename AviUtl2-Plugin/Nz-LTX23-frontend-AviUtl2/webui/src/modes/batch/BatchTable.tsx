import { useStrings } from "../../i18n/LanguageContext";
import type { BatchRow, BatchStat } from "./manifestMerge";

/** Stat icon, mirroring `gradio_ui/ui.py`'s `_STAT_ICON` mapping
 * (⚪Waiting/⏳Generating/✅Done/❌Failed/⛔Skip) so a user who has also seen
 * the Gradio GUI recognizes the same at-a-glance vocabulary. */
const STAT_ICON: Record<BatchStat, string> = {
  Waiting: "⚪",
  Generating: "⏳",
  Done: "✅",
  Failed: "❌",
  Skip: "⛔",
};

const RESETTABLE_STATS: ReadonlySet<BatchStat> = new Set(["Done", "Failed", "Skip"]);

export interface BatchTableProps {
  rows: BatchRow[];
  /** True while a run is in flight — disables every row's reset button, the
   * prompt/image editors, and the "copy common prompt" button alike (the
   * canonical CSV is only safe to hand-edit while idle; N5 deliberately
   * gates this globally rather than per-row). */
  disabled: boolean;
  /** N5 "A3: 行別画像割当" — the shared `<select>` option list (always leads
   * with the `Shared` sentinel), from `useBatchForm`'s `imageOptions`. */
  imageOptions: string[];
  onResetRow: (queue: number) => void;
  /** N5 "A1: プロンプト直接編集" — row `index`'s `prompt` cell edited to
   * `value` in memory (a controlled `<input>`'s `onChange`). */
  onUpdateRowPrompt: (index: number, value: string) => void;
  /** N5 "A2: 共通プロンプト流し込み" — row `index`'s "use shared prompt" button pressed. */
  onCopyCommonPrompt: (index: number) => void;
  /** N5 "A3: 行別画像割当" — row `index`'s `<select>` committed `imageName`. */
  onUpdateRowImage: (index: number, imageName: string) => void;
}

/** The `<select>` options to render for one row: `imageOptions` as-is, unless
 * the row's current `image` isn't among them (e.g. a manifest referencing a
 * file the image folder no longer has, or no image folder scanned yet) — in
 * which case it's appended so the `<select>` always has a matching, visible
 * option instead of silently rendering blank. */
function optionsForRow(imageOptions: string[], current: string): string[] {
  return imageOptions.includes(current) ? imageOptions : [...imageOptions, current];
}

/**
 * Renders the batch manifest's first 7 columns (spec §3: "先頭7列のみ表示" —
 * `frames`/`skip_reason`/`error` are CSV-only management columns), plus a
 * per-row "Reset to Waiting" action for `Done`/`Failed`/`Skip` rows (spec
 * §4's manual regenerate rule). A plain `<table>` rather than a virtualized
 * grid — the spec's own working scale is "100〜200個" rows, comfortably
 * within what a native table handles without extra dependencies.
 */
export function BatchTable({
  rows,
  disabled,
  imageOptions,
  onResetRow,
  onUpdateRowPrompt,
  onCopyCommonPrompt,
  onUpdateRowImage,
}: BatchTableProps) {
  const strings = useStrings();
  const t = strings.batch.table;

  if (rows.length === 0) {
    return <p className="field-hint">{t.empty}</p>;
  }

  return (
    <div className="batch-table-scroll">
      <table className="batch-table">
        <thead>
          <tr>
            <th>{t.queue}</th>
            <th>{t.wav}</th>
            <th>{t.duration}</th>
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
              <td className="batch-table-wav">{row.wav}</td>
              <td>{row.duration.toFixed(1)}s</td>
              <td className="batch-table-image-cell">
                <select
                  className="batch-table-image-select"
                  aria-label={t.imageSelectLabel}
                  value={row.image}
                  disabled={disabled}
                  onChange={(e) => onUpdateRowImage(index, e.target.value)}
                >
                  {optionsForRow(imageOptions, row.image).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </td>
              <td className="batch-table-prompt-cell">
                <input
                  type="text"
                  className="batch-table-prompt-input"
                  aria-label={t.promptInputLabel}
                  value={row.prompt}
                  disabled={disabled}
                  onChange={(e) => onUpdateRowPrompt(index, e.target.value)}
                />
                <button
                  type="button"
                  className="icon-action-button batch-copy-prompt-button"
                  title={t.copyCommonPromptButton}
                  aria-label={t.copyCommonPromptButton}
                  disabled={disabled}
                  onClick={() => onCopyCommonPrompt(index)}
                >
                  <span aria-hidden="true">📝</span>
                </button>
              </td>
              <td>
                <span className="batch-stat-cell" title={skipReasonTitle(row.stat, row.skipReason, strings.batch.skipReasons)}>
                  {STAT_ICON[row.stat]} {strings.batch.stat[statKey(row.stat)]}
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

/** Title for a `Skip` row's stat badge (spec: "❌ Skipバッジのツールチップ") —
 * only when `row.stat === "Skip"` AND `skipReason` is one of today's known
 * keys (`strings.batch.skipReasons`, keyed by `manifestMerge.ts`'s
 * `BatchRow.skipReason`). A future/unknown skipReason (e.g. a value added by
 * a newer manifest scan than this build knows about) falls through to
 * `undefined` rather than indexing the map with an unchecked string (TS7053)
 * or showing a blank/garbled tooltip. */
function skipReasonTitle(stat: BatchStat, skipReason: string, skipReasons: Record<string, string>): string | undefined {
  if (stat !== "Skip") return undefined;
  return skipReason in skipReasons ? skipReasons[skipReason] : undefined;
}

function statKey(stat: BatchStat): "waiting" | "generating" | "done" | "failed" | "skip" {
  switch (stat) {
    case "Waiting":
      return "waiting";
    case "Generating":
      return "generating";
    case "Done":
      return "done";
    case "Failed":
      return "failed";
    case "Skip":
      return "skip";
  }
}
