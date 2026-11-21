import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { V2vRefLongBody } from "./V2vRefLongBody";
import { makeForm, makeRow, makeView } from "./fakeForm.testSupport";
import { createInitialV2vRefLongState } from "./store";
import type { UseV2vRefLongFormResult, V2vRefLongState } from "./types";

function renderBody(form: UseV2vRefLongFormResult) {
  return render(
    <LanguageProvider>
      <V2vRefLongBody form={form} />
    </LanguageProvider>,
  );
}

function stateWith(patch: Partial<V2vRefLongState>): V2vRefLongState {
  return { ...createInitialV2vRefLongState(), ...patch };
}

function reasonLines(): string[] {
  const note = screen.queryByRole("note");
  return note ? within(note).getAllByRole("listitem").map((li) => li.textContent ?? "") : [];
}

describe("V2vRefLongBody", () => {
  it("draws the parts in order: folders, scan, clip count and mode, table, progress, notes, Start", () => {
    const { container } = renderBody(makeForm({ clipCount: 3 }));
    const text = container.textContent ?? "";
    const order = [
      "Video folder (opening)",
      "Reference folders (0)",
      "Output folder",
      "Scan folders",
      "Chained clips: 3",
      "How the main prompt is used:",
      "No rows yet",
      "0 / 0 done",
      "Only one batch runs at a time",
      "Start batch",
    ].map((s) => text.indexOf(s));
    expect(order.every((i) => i >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });

  it("Scan calls the form; while scanning it shows the progress and is disabled", () => {
    const form = makeForm();
    const idle = renderBody(form);
    const scanButton = screen.getByRole("button", { name: "Scan folders" });
    // Straight under `.field` (full width, as Batch i2v-long's).
    expect(scanButton.parentElement).toHaveClass("field");
    fireEvent.click(scanButton);
    expect(form.scan).toHaveBeenCalled();
    idle.unmount();

    renderBody(
      makeForm({
        canScan: false,
        inputsLocked: true,
        state: stateWith({ scanning: true, scanProgress: { done: 3, total: 12 } }),
      }),
    );
    expect(screen.getByRole("button", { name: "Scanning… 3/12" })).toBeDisabled();
  });

  it("shows a scan failure, the fields with no files and the all-empty note", () => {
    renderBody(
      makeForm({
        state: stateWith({ scanError: "disk gone" }),
        emptyFields: [{ type: "source" }, { type: "ref", frame: 1, kind: "audio", slot: 2 }],
        allFieldsEmpty: true,
      }),
    );
    expect(screen.getByText("Scan failed: disk gone")).toBeInTheDocument();
    expect(
      screen.getByText("Video folder (opening): no files of a supported format. This field is not used."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Clip 2 <Audio 2>: no files of a supported format. This field is not used."),
    ).toBeInTheDocument();
    expect(screen.getByText("None of the folders has files of a supported format.")).toBeInTheDocument();
  });

  it("the main prompt mode radios go through the form", () => {
    const form = makeForm({ state: stateWith({ mainPromptMode: "add" }) });
    renderBody(form);
    expect(screen.getByRole("radio", { name: "Add to the prompt" })).toBeChecked();
    fireEvent.click(screen.getByRole("radio", { name: "Replace the prompt" }));
    expect(form.setMainPromptMode).toHaveBeenCalledWith("replace");
  });

  it("progress: n / N done and the row being processed by its file name", () => {
    const row = makeRow(1, { stat: "Generating" });
    renderBody(
      makeForm({
        rowViews: [makeView(makeRow(0)), makeView(row, { baseFileName: "b.mp4" })],
        summary: { done: 1, total: 2 },
        currentRow: row,
      }),
    );
    expect(screen.getByText("1 / 2 done")).toBeInTheDocument();
    expect(screen.getByText("Processing: b.mp4")).toBeInTheDocument();
  });

  it("the seed note shows only while the seed is fixed", () => {
    const off = renderBody(makeForm());
    expect(screen.queryByText(/The seed is fixed/)).toBeNull();
    off.unmount();
    renderBody(makeForm({ seedFixed: true }));
    expect(screen.getByText(/The seed is fixed/)).toBeInTheDocument();
  });

  it("lists the panel's reasons and the kept Chain reasons; promptEmpty names the display rows", () => {
    renderBody(
      makeForm({
        blockReasons: ["fl2vaLoaded", "promptEmpty", "clipTooShortForOverlap", "lockedByOther"],
        chainBlockReasons: ["dimensionsOffGrid"],
        promptEmptyRows: [0, 4],
      }),
    );
    const lines = reasonLines();
    expect(lines).toContain(en.batchV2vRefLong.blockReasons.fl2vaLoaded);
    expect(lines).toContain(
      "These rows would be sent with an empty prompt: 1, 5. Write a prompt on the Clip Chain form, or in each row's Main prompt edit column.",
    );
    expect(lines.some((l) => l.includes("39"))).toBe(true);
    expect(lines).toHaveLength(5);
  });

  it("promptEmpty without rows falls back to the panel's own line", () => {
    renderBody(makeForm({ blockReasons: ["notScanned", "promptEmpty"] }));
    expect(reasonLines()).toEqual([
      "Scan the folders first.",
      "Write a prompt — either on the Clip Chain form or in a row's Main prompt edit column.",
    ]);
  });

  it("no reasons, no note", () => {
    renderBody(makeForm({ canStart: true }));
    expect(screen.queryByRole("note")).toBeNull();
  });

  it("Start and Stop follow the run state", () => {
    const idle = makeForm({ canStart: true });
    const a = renderBody(idle);
    fireEvent.click(screen.getByRole("button", { name: "Start batch" }));
    expect(idle.start).toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    a.unmount();

    const blocked = renderBody(makeForm({ canStart: false }));
    expect(screen.getByRole("button", { name: "Start batch" })).toBeDisabled();
    blocked.unmount();

    const running = makeForm({
      canStart: false,
      inputsLocked: true,
      state: stateWith({ run: { state: "running", lastStartResult: null, clipCount: 2 } }),
    });
    const b = renderBody(running);
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(running.stop).toHaveBeenCalled();
    b.unmount();

    renderBody(
      makeForm({
        canStart: false,
        inputsLocked: true,
        state: stateWith({ run: { state: "stopping", lastStartResult: null, clipCount: 2 } }),
      }),
    );
    expect(screen.getByRole("button", { name: "Stopping…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  });
});
