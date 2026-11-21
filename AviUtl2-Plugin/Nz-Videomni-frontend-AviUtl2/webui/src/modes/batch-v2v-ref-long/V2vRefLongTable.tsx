import { Fragment, useState } from "react";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import type { Strings } from "../../i18n/strings";
import { referenceTag } from "../single/referenceRules";
import { RowClipCards } from "./RowClipCards";
import type { BaseField, UseV2vRefLongFormResult, V2vRowBlockReason, V2vRowStat, V2vRowView } from "./types";

/** The same icons as `batch-i2v-long/BatchI2vLongTable.tsx`. */
const STAT_ICON: Record<V2vRowStat, string> = {
  Waiting: "⚪",
  Generating: "⏳",
  Done: "✅",
  Failed: "❌",
};

const STAT_KEY: Record<V2vRowStat, "waiting" | "generating" | "done" | "failed"> = {
  Waiting: "waiting",
  Generating: "generating",
  Done: "done",
  Failed: "failed",
};

const RESETTABLE_STATS: ReadonlySet<V2vRowStat> = new Set<V2vRowStat>(["Done", "Failed"]);

/** The columns, ▶ and 🔁 included (the expanded row spans them all). */
const COLUMN_COUNT = 8;

type V2vStrings = Strings["batchV2vRefLong"];

/** The name column's heading: the opening video, `<Picture 1>` (clip 1) or
 * `Clip 2 <Picture 1>`. */
function baseColumnLabel(t: V2vStrings, base: BaseField | null): string {
  if (base === null || base.type === "source") return t.table.baseFile(t.table.sourceLabel);
  const tag = referenceTag(base.kind, base.slot);
  return t.table.baseFile(base.frame === 0 ? tag : t.table.clipFieldLabel(base.frame + 1, tag));
}

function rowBlockReasonText(t: V2vStrings, reason: V2vRowBlockReason): string {
  switch (reason.code) {
    case "noSourceSingleClip":
      return t.rowBlockReasons.noSourceSingleClip;
    case "sourceTooShort":
      return t.rowBlockReasons.sourceTooShort(reason.frames);
    case "clip0TooShort":
      return t.rowBlockReasons.clip0TooShort(reason.minFrames);
    case "fileTooLarge":
      return t.rowBlockReasons.fileTooLarge(reason.fileName, reason.limitMb);
    case "noMaterial":
      return t.rowBlockReasons.noMaterial;
  }
}

/** The status cell: 🚫 (blocked Waiting/Failed), ⚠ (Waiting with a greyed
 * card), else the usual icon; the tooltip carries the why. */
function statCell(t: V2vStrings, view: V2vRowView): { text: string; title: string | undefined } {
  const { stat, error } = view.row;
  if (view.blockReasons.length > 0 && (stat === "Waiting" || stat === "Failed")) {
    const lines = view.blockReasons.map((reason) => rowBlockReasonText(t, reason));
    // A failed row keeps its failure reason visible after the 🚫 reasons.
    if (stat === "Failed" && error) lines.push(error);
    return { text: `🚫 ${t.stat.skip}`, title: lines.join("\n") };
  }
  if (stat === "Waiting" && view.ignoredClips.length > 0) {
    return { text: `⚠ ${t.stat.waiting}`, title: t.ignoredClips(view.ignoredClips.map((clip) => clip + 1)) };
  }
  return {
    text: `${STAT_ICON[stat]} ${t.stat[STAT_KEY[stat]]}`,
    title: stat === "Failed" && error ? error : undefined,
  };
}

export interface V2vRefLongTableProps {
  form: UseV2vRefLongFormResult;
  nativeBridge?: NativeBridge | undefined;
}

/** Batch v2v-ref-long's row table; a row opens into its clip cards. */
export function V2vRefLongTable({ form, nativeBridge }: V2vRefLongTableProps) {
  const t = useStrings().batchV2vRefLong;
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(() => new Set());
  const disabled = form.inputsLocked;

  if (form.rowViews.length === 0) {
    return <p className="field-hint">{t.table.empty}</p>;
  }

  const toggle = (index: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });

  return (
    <div className="batch-table-scroll v2v-table-scroll">
      <table className="batch-table">
        <thead>
          <tr>
            <th aria-hidden="true" />
            <th>{t.table.queue}</th>
            <th>{baseColumnLabel(t, form.baseField)}</th>
            <th>{t.table.v2v}</th>
            <th>{t.table.mainPrompt}</th>
            <th>{t.table.stat}</th>
            <th>{t.table.output}</th>
            <th aria-hidden="true" />
          </tr>
        </thead>
        <tbody>
          {form.rowViews.map((view) => {
            const { row } = view;
            const open = expanded.has(row.index);
            const stat = statCell(t, view);
            const expandLabel = open ? t.table.collapseRow : t.table.expandRow;
            return (
              <Fragment key={row.index}>
                <tr data-testid={`v2v-row-${row.index}`}>
                  <td>
                    <button
                      type="button"
                      className="icon-action-button v2v-expand-button"
                      title={expandLabel}
                      aria-label={expandLabel}
                      aria-expanded={open}
                      onClick={() => toggle(row.index)}
                    >
                      <span aria-hidden="true">{open ? "▼" : "▶"}</span>
                    </button>
                  </td>
                  <td>{row.index + 1}</td>
                  <td className="batch-table-wav">{view.baseFileName ?? ""}</td>
                  <td>{view.hasSource ? "🎬" : ""}</td>
                  <td className="batch-table-prompt-cell">
                    <input
                      type="text"
                      className="batch-table-prompt-input"
                      aria-label={t.table.mainPromptInputLabel}
                      value={row.mainPrompt}
                      disabled={disabled}
                      onChange={(e) => form.setRowMainPrompt(row.index, e.target.value)}
                    />
                    <button
                      type="button"
                      className="icon-action-button batch-copy-prompt-button"
                      title={t.table.copyChainPromptButton}
                      aria-label={t.table.copyChainPromptButton}
                      disabled={disabled}
                      onClick={() => form.copyChainPromptToRow(row.index)}
                    >
                      <span aria-hidden="true">📝</span>
                    </button>
                  </td>
                  <td>
                    <span className="batch-stat-cell" title={stat.title}>
                      {stat.text}
                    </span>
                  </td>
                  <td className="batch-table-truncate">{row.output}</td>
                  <td>
                    <button
                      type="button"
                      className="icon-action-button batch-reset-button"
                      title={t.table.resetButton}
                      aria-label={t.table.resetButton}
                      disabled={disabled || !RESETTABLE_STATS.has(row.stat)}
                      onClick={() => form.resetRow(row.index)}
                    >
                      <span aria-hidden="true">🔁</span>
                    </button>
                  </td>
                </tr>
                {open && (
                  <tr className="v2v-row-detail" data-testid={`v2v-row-detail-${row.index}`}>
                    <td colSpan={COLUMN_COUNT}>
                      <RowClipCards form={form} rowIndex={row.index} nativeBridge={nativeBridge} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
