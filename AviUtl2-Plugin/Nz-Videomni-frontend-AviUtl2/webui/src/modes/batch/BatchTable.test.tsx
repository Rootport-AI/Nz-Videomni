import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { BatchTable } from "./BatchTable";
import type { BatchTableProps } from "./BatchTable";
import type { BatchMode, BatchRow } from "./manifestMerge";

function makeRow(overrides: Partial<BatchRow> = {}): BatchRow {
  return {
    queue: 1,
    wav: "a.wav",
    duration: 3.2,
    image: "Shared",
    prompt: "",
    stat: "Waiting",
    output: "",
    frames: 49,
    skipReason: "",
    error: "",
    ...overrides,
  };
}

function renderTable(rows: BatchRow[], mode: BatchMode | null = "a2v") {
  const props: BatchTableProps = {
    rows,
    mode,
    disabled: false,
    imageOptions: ["Shared"],
    onResetRow: vi.fn(),
    onUpdateRowPrompt: vi.fn(),
    onCopyCommonPrompt: vi.fn(),
    onUpdateRowImage: vi.fn(),
  };
  return render(
    <LanguageProvider>
      <BatchTable {...props} />
    </LanguageProvider>,
  );
}

/** The `.batch-stat-cell` span for row `queue` — there is exactly one per
 * row, so a `querySelector` scoped to nothing works fine since each test
 * renders a single row (or picks the row by its queue-number cell text). */
function statCellFor(container: HTMLElement, queue: number): HTMLElement {
  const cells = Array.from(container.querySelectorAll(".batch-stat-cell"));
  const rows = Array.from(container.querySelectorAll("tbody tr"));
  const index = rows.findIndex((tr) => tr.firstElementChild?.textContent === String(queue));
  return cells[index] as HTMLElement;
}

describe("BatchTable skip badge tooltip", () => {
  it("shows the over-cap tooltip text on an over-cap Skip row's badge", () => {
    const row = makeRow({ queue: 1, stat: "Skip", skipReason: "over-cap" });
    const { container } = renderTable([row]);

    expect(statCellFor(container, 1)).toHaveAttribute("title", en.batch.skipReasons["over-cap"]);
  });

  it("shows the wav-only-alpha tooltip text on a wav-only-alpha Skip row's badge", () => {
    const row = makeRow({ queue: 1, stat: "Skip", skipReason: "wav-only-alpha" });
    const { container } = renderTable([row]);

    expect(statCellFor(container, 1)).toHaveAttribute("title", en.batch.skipReasons["wav-only-alpha"]);
  });

  it("does not set a title on non-Skip rows (Waiting, Done, etc.)", () => {
    const rows = [
      makeRow({ queue: 1, stat: "Waiting" }),
      makeRow({ queue: 2, stat: "Generating" }),
      makeRow({ queue: 3, stat: "Done" }),
      makeRow({ queue: 4, stat: "Failed" }),
    ];
    const { container } = renderTable(rows);

    for (const row of rows) {
      expect(statCellFor(container, row.queue)).not.toHaveAttribute("title");
    }
  });

  it("renders a Skip row with an unrecognized skipReason without a title and without crashing", () => {
    const row = makeRow({ queue: 1, stat: "Skip", skipReason: "future-reason" });
    const { container } = renderTable([row]);

    expect(screen.getByText(row.wav)).toBeInTheDocument();
    expect(statCellFor(container, 1)).not.toHaveAttribute("title");
  });
});

/** 各行の音声セル（表の2列目）のテキスト。画像`<select>`の`<option>`が同じ
 * ファイル名を持ちうるので、セルを直接見る。 */
function wavCellTexts(container: HTMLElement): (string | null)[] {
  return Array.from(container.querySelectorAll(".batch-table-wav")).map((td) => td.textContent);
}

/** 各行の長さセル（表の3列目 = `t.duration`列）のテキスト。行内のどこかに
 * "15.0s"があることではなく、長さ列そのものを見るための添字。 */
const DURATION_COLUMN_INDEX = 2;
function durationCellTexts(container: HTMLElement): (string | null)[] {
  return Array.from(container.querySelectorAll("tbody tr")).map(
    (tr) => tr.children[DURATION_COLUMN_INDEX]?.textContent ?? null,
  );
}

// i2vモード（D2, 2026-09-15）: 音声ファイルが存在しないので、音声セルはモードを
// 示す固定ラベルになる。列見出しは「音声ファイル」のまま据え置き。
describe("BatchTable i2vモードの音声セル", () => {
  it("mode=i2v なら全行の音声セルがi2vラベルになり、スキャン元の画像名はそこに出ない", () => {
    const rows = [
      makeRow({ queue: 1, wav: "cat01.png", image: "cat01.png" }),
      makeRow({ queue: 2, wav: "cat02.png", image: "cat02.png" }),
    ];
    const { container } = renderTable(rows, "i2v");

    expect(wavCellTexts(container)).toEqual([en.batch.table.i2vRowLabel, en.batch.table.i2vRowLabel]);
  });

  it("mode=a2v なら従来どおり row.wav を出す", () => {
    const { container } = renderTable([makeRow({ queue: 1, wav: "line01.wav" })], "a2v");

    expect(wavCellTexts(container)).toEqual(["line01.wav"]);
  });

  it("長さ列はどちらのモードも 15.0s 形式のまま", () => {
    const row = makeRow({ queue: 1, wav: "cat01.png", duration: 15 });
    expect(durationCellTexts(renderTable([row], "i2v").container)).toEqual(["15.0s"]);
    expect(durationCellTexts(renderTable([row], "a2v").container)).toEqual(["15.0s"]);
  });
});
