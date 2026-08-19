import { useState } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LanguageProvider } from "../i18n/LanguageContext";
import { NoteArea } from "./NoteArea";
import type { AppNote, NoteKind } from "./NoteArea";

// A minimal host that owns the single note seat exactly as `AppShell` does
// (`showNote` REPLACES; `dismiss` clears), so the "single seat / replace /
// dismiss" contract (§6-1) is exercised through the same shape the app uses.
function Harness() {
  const [note, setNote] = useState<AppNote | null>(null);
  const showNote = (kind: NoteKind, message: string) => setNote({ kind, message });
  return (
    <LanguageProvider>
      <button type="button" onClick={() => showNote("warning", "first note")}>
        show first
      </button>
      <button type="button" onClick={() => showNote("error", "second note")}>
        show second
      </button>
      <NoteArea note={note} onDismiss={() => setNote(null)} />
    </LanguageProvider>
  );
}

function renderNote(note: AppNote | null) {
  return render(
    <LanguageProvider>
      <NoteArea note={note} onDismiss={() => {}} />
    </LanguageProvider>,
  );
}

describe("NoteArea", () => {
  it("renders nothing when there is no note", () => {
    renderNote(null);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows the note message with polite live-region a11y attributes", () => {
    renderNote({ kind: "info", message: "hello note" });
    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent("hello note");
  });

  it("carries the kind as a modifier class for tonal styling", () => {
    renderNote({ kind: "warning", message: "watch out" });
    expect(screen.getByRole("status")).toHaveClass("note-area-warning");
  });

  it("dismisses via the × button (labelled from common.dismissNotification)", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("button", { name: "show first" }));
    expect(screen.getByRole("status")).toHaveTextContent("first note");

    await user.click(screen.getByRole("button", { name: /dismiss notification/i }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("replaces the current note when a new one is shown (single seat, later wins)", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("button", { name: "show first" }));
    expect(screen.getByRole("status")).toHaveTextContent("first note");

    await user.click(screen.getByRole("button", { name: "show second" }));
    // Only one note at a time — the first is gone, the second replaced it.
    const regions = screen.getAllByRole("status");
    expect(regions).toHaveLength(1);
    expect(regions[0]).toHaveTextContent("second note");
    expect(regions[0]).not.toHaveTextContent("first note");
    expect(regions[0]).toHaveClass("note-area-error");
  });
});
