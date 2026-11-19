import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { RowClipCards } from "./RowClipCards";
import { emptyRefs, makeForm, makeRow, makeView, withCard } from "./fakeForm.testSupport";
import type { UseV2vRefLongFormResult } from "./types";

function fakeBridge(): NativeBridge {
  return { request: vi.fn().mockResolvedValue({ dataUrl: "data:image/png;base64,AA" }) } as unknown as NativeBridge;
}

function renderCards(form: UseV2vRefLongFormResult, rowIndex = 0) {
  return render(
    <LanguageProvider>
      <RowClipCards form={form} rowIndex={rowIndex} nativeBridge={fakeBridge()} />
    </LanguageProvider>,
  );
}

function card(clip: number): HTMLElement {
  return screen.getByTestId(`v2v-clip-card-${clip}`);
}

function topRow(container: HTMLElement): HTMLElement | null {
  return container.querySelector(".v2v-row-cards-top");
}

function strip(container: HTMLElement): HTMLElement {
  return container.querySelector(".v2v-row-cards-strip") as HTMLElement;
}

describe("RowClipCards", () => {
  it("draws clipCount cards, without a frame-count field", () => {
    const { container } = renderCards(makeForm({ clipCount: 3 }));
    expect(within(strip(container)).getAllByText(/^Clip \d$/).map((el) => el.textContent)).toEqual([
      "Clip 1",
      "Clip 2",
      "Clip 3",
    ]);
    expect(container.querySelector('input[type="range"]')).toBeNull();
    expect(container.querySelector('input[type="number"]')).toBeNull();
    expect(screen.queryByText(/frames/i)).toBeNull();
  });

  it("the Reference button counts the clip's filled cards against maxTotal; ⚠ on an ignored clip", () => {
    let refs = withCard(emptyRefs(), "picture", 1, "C:\\a.png");
    refs = withCard(refs, "video", 1, "C:\\b.mp4", 4);
    const form = makeForm({
      clipCount: 2,
      rowViews: [makeView(makeRow(0), { ignoredClips: [1] })],
      clips: { "0:0": { promptMode: "add", prompt: "", refs } },
    });
    renderCards(form);
    expect(within(card(0)).getByRole("button", { name: /Reference 2\/12/ })).toBeInTheDocument();
    expect(within(card(0)).queryByText("⚠", { exact: false })).toBeNull();
    const second = within(card(1)).getByRole("button", { name: /Reference 0\/12/ });
    expect(second).toHaveTextContent("⚠");
    expect(within(second).getByTitle("Clip 2: the Reference has cards that are not used (greyed out).")).toBeInTheDocument();
  });

  it("opens one Reference at a time: the open card goes to the top, the other one back to the strip", () => {
    const { container } = renderCards(makeForm({ clipCount: 3 }));
    expect(topRow(container)).toBeNull();

    fireEvent.click(within(card(1)).getByRole("button", { name: /Reference/ }));
    let top = topRow(container) as HTMLElement;
    expect(within(top).getByTestId("v2v-clip-card-1")).toBeInTheDocument();
    expect(within(top).queryByRole("button", { name: /^▶?Reference \d+\/\d+/ })).toBeNull();
    const details = top.querySelector("details") as HTMLDetailsElement;
    expect(details.open).toBe(true);
    expect(within(strip(container)).queryByTestId("v2v-clip-card-1")).toBeNull();
    expect(container.querySelectorAll("details")).toHaveLength(1);

    fireEvent.click(within(card(2)).getByRole("button", { name: /Reference/ }));
    top = topRow(container) as HTMLElement;
    expect(within(top).getByTestId("v2v-clip-card-2")).toBeInTheDocument();
    expect(within(strip(container)).getByTestId("v2v-clip-card-1")).toBeInTheDocument();
    expect(container.querySelectorAll("details")).toHaveLength(1);
  });

  it("closing the accordion puts the card back in the strip", () => {
    const { container } = renderCards(makeForm({ clipCount: 2 }));
    fireEvent.click(within(card(0)).getByRole("button", { name: /Reference/ }));
    const details = (topRow(container) as HTMLElement).querySelector("details") as HTMLDetailsElement;
    details.open = false;
    fireEvent(details, new Event("toggle"));
    expect(topRow(container)).toBeNull();
    expect(within(strip(container)).getByTestId("v2v-clip-card-0")).toBeInTheDocument();
  });

  it("the open accordion has no Audio to video box, and ✒ appends to this row's clip", () => {
    const refs = withCard(withCard(emptyRefs(), "audio", 1, "C:\\a.wav", 5), "picture", 1, "C:\\p.png");
    const form = makeForm({ clipCount: 2, clips: { "3:1": { promptMode: "add", prompt: "", refs } } });
    renderCards(form, 3);
    fireEvent.click(within(card(1)).getByRole("button", { name: /Reference/ }));
    expect(screen.queryByRole("checkbox", { name: "Audio to video" })).toBeNull();
    expect(screen.queryByTestId("source-audio-strength")).toBeNull();
    fireEvent.click(within(screen.getByTestId("ref-slot-picture-1")).getByRole("button", { name: /add this tag/i }));
    expect(form.appendClipTag).toHaveBeenCalledWith(3, 1, "<Picture 1>");
  });

  it("the clip prompt and its add/replace mode go through the form", () => {
    const form = makeForm({
      clipCount: 2,
      clips: { "0:1": { promptMode: "replace", prompt: "rain", refs: emptyRefs() } },
    });
    renderCards(form);
    const second = within(card(1));
    expect(second.getByRole("textbox", { name: "Clip prompt" })).toHaveValue("rain");
    expect(second.getByRole("radio", { name: "Replace" })).toBeChecked();
    fireEvent.change(second.getByRole("textbox", { name: "Clip prompt" }), { target: { value: "snow" } });
    expect(form.setClipPrompt).toHaveBeenCalledWith(0, 1, "snow");
    fireEvent.click(second.getByRole("radio", { name: "Add" }));
    expect(form.setClipPromptMode).toHaveBeenCalledWith(0, 1, "add");
    expect(within(card(0)).getByRole("radio", { name: "Add" })).toBeChecked();
  });

  it("inputsLocked disables the prompt inputs and the open accordion, not the Reference button", () => {
    const { container } = renderCards(makeForm({ clipCount: 2, inputsLocked: true }));
    expect(within(card(0)).getByRole("textbox", { name: "Clip prompt" })).toBeDisabled();
    expect(within(card(0)).getByRole("radio", { name: "Add" })).toBeDisabled();
    fireEvent.click(within(card(0)).getByRole("button", { name: /Reference/ }));
    const top = topRow(container) as HTMLElement;
    expect(within(within(top).getByTestId("ref-slot-picture-1")).getByRole("button", { name: /choose/i })).toBeDisabled();
  });

  it("a clip count below the open card closes it", () => {
    const form = makeForm({ clipCount: 3 });
    const { container, rerender } = renderCards(form);
    fireEvent.click(within(card(2)).getByRole("button", { name: /Reference/ }));
    expect(topRow(container)).not.toBeNull();
    rerender(
      <LanguageProvider>
        <RowClipCards form={{ ...form, clipCount: 2 }} rowIndex={0} nativeBridge={fakeBridge()} />
      </LanguageProvider>,
    );
    expect(topRow(container)).toBeNull();
    expect(within(strip(container)).getAllByText(/^Clip \d$/)).toHaveLength(2);
  });
});
