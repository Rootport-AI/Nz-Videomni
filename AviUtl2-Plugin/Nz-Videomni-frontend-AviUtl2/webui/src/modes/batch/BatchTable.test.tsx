import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { BatchTable } from "./BatchTable";
import type { BatchTableProps } from "./BatchTable";
import type { BatchRow } from "./manifestMerge";

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

function renderTable(rows: BatchRow[]) {
  const props: BatchTableProps = {
    rows,
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
