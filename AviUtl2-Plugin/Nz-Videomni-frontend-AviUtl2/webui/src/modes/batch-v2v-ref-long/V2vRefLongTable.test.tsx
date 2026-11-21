import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { V2vRefLongTable } from "./V2vRefLongTable";
import { makeForm, makeRow, makeView } from "./fakeForm.testSupport";
import type { UseV2vRefLongFormResult } from "./types";

const t = en.batchV2vRefLong;
const SKIP = `🚫 ${t.stat.skip}`;

function renderTable(form: UseV2vRefLongFormResult) {
  return render(
    <LanguageProvider>
      <V2vRefLongTable form={form} />
    </LanguageProvider>,
  );
}

function headers(): string[] {
  return screen.getAllByRole("columnheader").map((th) => th.textContent ?? "");
}

describe("V2vRefLongTable", () => {
  it("shows the empty line without rows", () => {
    renderTable(makeForm());
    expect(screen.getByText("No rows yet — choose the folders and scan them.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("names the file column after the base field", () => {
    const rowViews = [makeView(makeRow(0))];
    const a = renderTable(makeForm({ rowViews, baseField: { type: "source" } }));
    expect(headers()).toContain("Opening video file name");
    a.unmount();
    const b = renderTable(makeForm({ rowViews, baseField: { type: "ref", frame: 0, kind: "picture", slot: 1 } }));
    expect(headers()).toContain("<Picture 1> file name");
    b.unmount();
    renderTable(makeForm({ rowViews, baseField: { type: "ref", frame: 1, kind: "video", slot: 2 } }));
    expect(headers()).toContain("Clip 2 <Video 2> file name");
  });

  it("columns in order; # is the display number; 🎬 only on rows with an opening video", () => {
    const rowViews = [
      makeView(makeRow(0), { baseFileName: "a.mp4", hasSource: true }),
      makeView(makeRow(1), { baseFileName: "b.png", hasSource: false }),
    ];
    renderTable(makeForm({ rowViews, baseField: { type: "source" } }));
    // The ▶ and 🔁 headers are blank and aria-hidden.
    expect(headers()).toEqual(["#", "Opening video file name", "v2v", "Main prompt edit", "Status", "Output"]);
    expect(screen.getByRole("table").querySelectorAll("thead th")).toHaveLength(8);
    const first = screen.getByTestId("v2v-row-0");
    const second = screen.getByTestId("v2v-row-1");
    const cells = (row: HTMLElement) => within(row).getAllByRole("cell").map((c) => c.textContent);
    expect(cells(first)[1]).toBe("1");
    expect(cells(first)[2]).toBe("a.mp4");
    expect(cells(first)[3]).toBe("🎬");
    expect(cells(second)[1]).toBe("2");
    expect(cells(second)[3]).toBe("");
  });

  it("🚫 Skip on a blocked Waiting/Failed row, with every reason in the tooltip", () => {
    const blockReasons = [
      { code: "noSourceSingleClip" as const },
      { code: "fileTooLarge" as const, fileName: "big.mp4", filePath: "C:\\big.mp4", limitMb: 200 },
    ];
    const rowViews = [
      makeView(makeRow(0), { blockReasons, ignoredClips: [0] }),
      makeView(makeRow(1, { stat: "Failed", error: "boom" }), { blockReasons }),
      makeView(makeRow(2, { stat: "Done" }), { blockReasons }),
    ];
    renderTable(makeForm({ rowViews }));
    const skip = within(screen.getByTestId("v2v-row-0")).getByText(SKIP);
    expect(skip).toHaveAttribute(
      "title",
      `${t.rowBlockReasons.noSourceSingleClip}\nbig.mp4 is over the 200 MB limit.`,
    );
    // A failed row keeps its failure reason after the 🚫 reasons.
    expect(within(screen.getByTestId("v2v-row-1")).getByText(SKIP)).toHaveAttribute(
      "title",
      `${t.rowBlockReasons.noSourceSingleClip}\nbig.mp4 is over the 200 MB limit.\nboom`,
    );
    // A finished row keeps its own status.
    expect(within(screen.getByTestId("v2v-row-2")).getByText("✅ Done")).toBeInTheDocument();
  });

  it("a 🚫 Failed row shows its failure reason after the 🚫 reasons; a 🚫 Waiting row does not", () => {
    const blockReasons = [{ code: "noMaterial" as const }];
    const rowViews = [
      makeView(makeRow(0, { stat: "Failed", error: "Join failed: x" }), { blockReasons }),
      makeView(makeRow(1, { error: "stale" }), { blockReasons }),
    ];
    renderTable(makeForm({ rowViews }));
    const noMaterial = "This row has no material (neither an opening video nor any reference is assigned).";
    expect(within(screen.getByTestId("v2v-row-0")).getByText(SKIP)).toHaveAttribute(
      "title",
      `${noMaterial}\nJoin failed: x`,
    );
    expect(within(screen.getByTestId("v2v-row-1")).getByText(SKIP)).toHaveAttribute("title", noMaterial);
  });

  it("⚠ Waiting with the clip display numbers in the tooltip", () => {
    const rowViews = [
      makeView(makeRow(0), { ignoredClips: [0, 2] }),
      makeView(makeRow(1, { stat: "Generating" }), { ignoredClips: [1] }),
    ];
    renderTable(makeForm({ rowViews }));
    const warn = within(screen.getByTestId("v2v-row-0")).getByText("⚠ Waiting");
    expect(warn).toHaveAttribute("title", t.ignoredClips([1, 3]));
    expect(within(screen.getByTestId("v2v-row-1")).getByText("⏳ Generating")).toBeInTheDocument();
  });

  it("the usual statuses; a failure's reason is the tooltip", () => {
    const rowViews = [
      makeView(makeRow(0)),
      makeView(makeRow(1, { stat: "Generating" })),
      makeView(makeRow(2, { stat: "Done", output: "a.mp4" })),
      makeView(makeRow(3, { stat: "Failed", error: "Server said no" })),
    ];
    renderTable(makeForm({ rowViews }));
    expect(within(screen.getByTestId("v2v-row-0")).getByText("⚪ Waiting")).not.toHaveAttribute("title");
    expect(within(screen.getByTestId("v2v-row-1")).getByText("⏳ Generating")).toBeInTheDocument();
    expect(within(screen.getByTestId("v2v-row-2")).getByText("✅ Done")).toBeInTheDocument();
    expect(within(screen.getByTestId("v2v-row-2")).getByText("a.mp4")).toBeInTheDocument();
    expect(within(screen.getByTestId("v2v-row-3")).getByText("❌ Failed")).toHaveAttribute("title", "Server said no");
  });

  it("🔁 only on Done/Failed rows, with the 0-based row index", () => {
    const rowViews = [
      makeView(makeRow(0)),
      makeView(makeRow(1, { stat: "Generating" })),
      makeView(makeRow(2, { stat: "Done" })),
      makeView(makeRow(3, { stat: "Failed" })),
    ];
    const form = makeForm({ rowViews });
    renderTable(form);
    const reset = (i: number) => within(screen.getByTestId(`v2v-row-${i}`)).getByRole("button", { name: "Reset to Waiting" });
    expect(reset(0)).toBeDisabled();
    expect(reset(1)).toBeDisabled();
    expect(reset(2)).toBeEnabled();
    expect(reset(3)).toBeEnabled();
    fireEvent.click(reset(3));
    expect(form.resetRow).toHaveBeenCalledWith(3);
  });

  it("edits the main prompt and copies the chain prompt through the form", () => {
    const rowViews = [makeView(makeRow(0)), makeView(makeRow(1, { mainPrompt: "hello" }))];
    const form = makeForm({ rowViews });
    renderTable(form);
    const row = within(screen.getByTestId("v2v-row-1"));
    const input = row.getByRole("textbox", { name: "Main prompt edit" });
    expect(input).toHaveValue("hello");
    fireEvent.change(input, { target: { value: "hello world" } });
    expect(form.setRowMainPrompt).toHaveBeenCalledWith(1, "hello world");
    fireEvent.click(row.getByRole("button", { name: "Use the prompt" }));
    expect(form.copyChainPromptToRow).toHaveBeenCalledWith(1);
  });

  it("inputsLocked stops the row inputs but not the expand button", () => {
    const rowViews = [makeView(makeRow(0, { stat: "Done" }))];
    renderTable(makeForm({ rowViews, inputsLocked: true }));
    const row = within(screen.getByTestId("v2v-row-0"));
    expect(row.getByRole("textbox", { name: "Main prompt edit" })).toBeDisabled();
    expect(row.getByRole("button", { name: "Use the prompt" })).toBeDisabled();
    expect(row.getByRole("button", { name: "Reset to Waiting" })).toBeDisabled();
    expect(row.getByRole("button", { name: "Show this row's clips" })).toBeEnabled();
  });

  it("opens and closes rows independently; an open row shows its clip cards below it", () => {
    const rowViews = [makeView(makeRow(0)), makeView(makeRow(1))];
    renderTable(makeForm({ rowViews, clipCount: 3 }));
    expect(screen.queryByTestId("v2v-row-detail-0")).toBeNull();

    fireEvent.click(within(screen.getByTestId("v2v-row-0")).getByRole("button", { name: "Show this row's clips" }));
    fireEvent.click(within(screen.getByTestId("v2v-row-1")).getByRole("button", { name: "Show this row's clips" }));
    const detail0 = screen.getByTestId("v2v-row-detail-0");
    expect(within(detail0).getAllByText(/^Clip \d$/)).toHaveLength(3);
    expect(screen.getByTestId("v2v-row-detail-1")).toBeInTheDocument();
    expect(screen.getByTestId("v2v-row-0").nextElementSibling).toBe(detail0);
    expect(detail0.querySelector("td")).toHaveAttribute("colspan", "8");

    const collapse = within(screen.getByTestId("v2v-row-0")).getByRole("button", { name: "Hide this row's clips" });
    expect(collapse).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(collapse);
    expect(screen.queryByTestId("v2v-row-detail-0")).toBeNull();
    expect(screen.getByTestId("v2v-row-detail-1")).toBeInTheDocument();
  });
});
