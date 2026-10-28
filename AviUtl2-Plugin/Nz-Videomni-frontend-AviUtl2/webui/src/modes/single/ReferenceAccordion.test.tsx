import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { ReferenceAccordion } from "./ReferenceAccordion";
import type { ReferenceAccordionProps } from "./ReferenceAccordion";
import { createInitialState, referenceReducer, summarizeReference } from "./referenceRules";
import type { ReferenceRules, ReferenceSlotsApi, ReferenceState } from "./referenceRules";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

/** Fills a slot through the real reducer (attachStart then attachDone). */
function fill(
  state: ReferenceState,
  kind: "picture" | "video" | "audio",
  slot: number,
  durationSec: number | null = null,
): ReferenceState {
  const started = referenceReducer(state, {
    type: "attachStart",
    kind,
    slot,
    filePath: `C:\\x\\${kind}${slot}`,
    fileName: `${kind}${slot}.bin`,
  });
  return referenceReducer(started, {
    type: "attachDone",
    kind,
    slot,
    id: `id-${kind}-${slot}`,
    durationSec,
    thumbnailDataUrl: null,
    trimFailed: false,
  });
}

function makeApi(state: ReferenceState, overrides: Partial<ReferenceSlotsApi> = {}): ReferenceSlotsApi {
  return {
    state,
    summary: summarizeReference(state, RULES),
    rules: RULES,
    attach: vi.fn().mockResolvedValue(undefined),
    pick: vi.fn().mockResolvedValue(undefined),
    clear: vi.fn(),
    attachToFirstEmpty: vi.fn().mockReturnValue(1),
    setEnabled: vi.fn(),
    setA2vSlot: vi.fn(),
    ...overrides,
  };
}

function renderAccordion(api: ReferenceSlotsApi, props: Partial<ReferenceAccordionProps> = {}) {
  const base: ReferenceAccordionProps = {
    api,
    openSignal: 0,
    onAppendTag: vi.fn(),
    a2v: {
      available: true,
      strength: 0.5,
      onStrengthChange: vi.fn(),
      strengthMin: 0,
      strengthMax: 1,
      strengthStep: 0.05,
      tooShortWarning: null,
    },
    ...props,
  };
  const view = render(
    <LanguageProvider>
      <ReferenceAccordion {...base} />
    </LanguageProvider>,
  );
  return { ...view, props: base };
}

function enabled(state: ReferenceState): ReferenceState {
  return referenceReducer(state, { type: "setEnabled", enabled: true });
}

describe("ReferenceAccordion", () => {
  it("shows the ON badge only while enabled", () => {
    const off = renderAccordion(makeApi(createInitialState(RULES)));
    expect(screen.queryByText("【🔴ON】")).toBeNull();
    off.unmount();
    renderAccordion(makeApi(enabled(createInitialState(RULES))));
    expect(screen.getByText("【🔴ON】")).toBeInTheDocument();
  });

  it("toggles Enable through the api", async () => {
    const api = makeApi(createInitialState(RULES));
    renderAccordion(api);
    await userEvent.click(screen.getByRole("checkbox", { name: /enable/i }));
    expect(api.setEnabled).toHaveBeenCalledWith(true);
  });

  it("builds three rows with slot counts from the rules", () => {
    renderAccordion(makeApi(createInitialState(RULES)));
    expect(within(screen.getByTestId("ref-row-picture")).getAllByText(/^<Picture \d>$/)).toHaveLength(9);
    expect(within(screen.getByTestId("ref-row-video")).getAllByText(/^<Video \d>$/)).toHaveLength(3);
    expect(within(screen.getByTestId("ref-row-audio")).getAllByText(/^<Audio \d>$/)).toHaveLength(3);
    expect(screen.getByText("Pictures")).toBeInTheDocument();
    expect(screen.getByText("Videos")).toBeInTheDocument();
    expect(screen.getByText("Audios")).toBeInTheDocument();
    expect(screen.getByTestId("ref-total-count")).toHaveTextContent("0/12");
  });

  it("follows other rules for the slot counts and the denominators", () => {
    const rules: ReferenceRules = { ...RULES, maxImages: 2, maxVideos: 1, maxAudios: 1, maxTotal: 4 };
    const state = createInitialState(rules);
    renderAccordion(makeApi(state, { rules, summary: summarizeReference(state, rules) }));
    expect(within(screen.getByTestId("ref-row-picture")).getAllByText(/^<Picture \d>$/)).toHaveLength(2);
    expect(screen.getByTestId("ref-total-count")).toHaveTextContent("0/4");
  });

  it("counts filled slots and seconds in the row titles", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "picture", 1);
    state = fill(state, "video", 1, 10.2);
    renderAccordion(makeApi(state));
    expect(screen.getByTestId("ref-total-count")).toHaveTextContent("2/12");
    expect(within(screen.getByTestId("ref-row-video")).getByText("(10.2s/15s)")).toBeInTheDocument();
    expect(within(screen.getByTestId("ref-row-video")).getByText("1/3")).toBeInTheDocument();
  });

  it("shows n/12 in red bold only when over the total", () => {
    let state = enabled(createInitialState(RULES));
    for (let i = 1; i <= 9; i += 1) state = fill(state, "picture", i);
    for (let i = 1; i <= 3; i += 1) state = fill(state, "video", i, 3);
    const { unmount } = renderAccordion(makeApi(state));
    expect(screen.getByTestId("ref-total-count")).not.toHaveClass("ref-over");
    unmount();
    state = fill(state, "audio", 1, 3);
    renderAccordion(makeApi(state));
    const count = screen.getByTestId("ref-total-count");
    expect(count).toHaveTextContent("13/12");
    // The two-class rule (`.ref-summary-count.ref-over`) carries the red bold.
    expect(count).toHaveClass("ref-summary-count", "field-hint-error", "ref-over");
  });

  it("marks the seconds counter over the limit", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "video", 1, 10);
    state = fill(state, "video", 2, 11);
    renderAccordion(makeApi(state));
    const counter = within(screen.getByTestId("ref-row-video")).getByText("(21.0s/15s)");
    expect(counter).toHaveClass("ref-row-count", "field-hint-error", "ref-over");
  });

  it("enables the pen button only on a ready card, with the other two buttons present", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "picture", 1);
    state = referenceReducer(state, {
      type: "attachStart",
      kind: "picture",
      slot: 3,
      filePath: "C:\\x\\p3",
      fileName: "p3.png",
    });
    state = referenceReducer(state, { type: "attachFail", kind: "picture", slot: 3, errorCode: "UPLOAD_FAILED" });
    renderAccordion(makeApi(state));
    expect(
      within(screen.getByTestId("ref-slot-picture-3")).getByRole("button", { name: /add this tag/i }),
    ).toBeDisabled();
    const filled = screen.getByTestId("ref-slot-picture-1");
    expect(within(filled).getByRole("button", { name: /clear this card/i })).toBeEnabled();
    expect(within(filled).getByRole("button", { name: /change file/i })).toBeEnabled();
    expect(within(filled).getByRole("button", { name: /add this tag/i })).toBeEnabled();
    const empty = screen.getByTestId("ref-slot-picture-2");
    expect(within(empty).getByRole("button", { name: /clear this card/i })).toBeInTheDocument();
    expect(within(empty).getByRole("button", { name: /choose file/i })).toBeInTheDocument();
    expect(within(empty).getByRole("button", { name: /add this tag/i })).toBeDisabled();
  });

  it("appends the card's tag from the pen button and calls pick / clear", async () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "video", 2, 5);
    const api = makeApi(state);
    const { props } = renderAccordion(api);
    const card = screen.getByTestId("ref-slot-video-2");
    await userEvent.click(within(card).getByRole("button", { name: /add this tag/i }));
    expect(props.onAppendTag).toHaveBeenCalledWith("<Video 2>");
    await userEvent.click(within(card).getByRole("button", { name: /change file/i }));
    expect(api.pick).toHaveBeenCalledWith("video", 2);
    await userEvent.click(within(card).getByRole("button", { name: /clear this card/i }));
    expect(api.clear).toHaveBeenCalledWith("video", 2);
  });

  it("attaches a dropped file with an accepted extension", async () => {
    const api = makeApi(enabled(createInitialState(RULES)));
    const nativeBridge = {
      requestWithFiles: vi.fn().mockResolvedValue({ files: [{ filePath: "C:\\a\\p.png", fileName: "p.png" }] }),
    } as unknown as NativeBridge;
    renderAccordion(api, { nativeBridge });
    fireEvent.drop(screen.getByTestId("ref-slot-picture-3"), {
      dataTransfer: { files: [new File([], "p.png")] },
    });
    await waitFor(() => expect(api.attach).toHaveBeenCalledWith("picture", 3, "C:\\a\\p.png", "p.png"));
  });

  it("rejects a dropped file of the wrong kind", async () => {
    const api = makeApi(enabled(createInitialState(RULES)));
    const nativeBridge = {
      requestWithFiles: vi.fn().mockResolvedValue({ files: [{ filePath: "C:\\a\\s.wav", fileName: "s.wav" }] }),
    } as unknown as NativeBridge;
    renderAccordion(api, { nativeBridge });
    const card = screen.getByTestId("ref-slot-picture-1");
    fireEvent.drop(card, { dataTransfer: { files: [new File([], "s.wav")] } });
    expect(await within(card).findByText(/Unsupported file type/)).toBeInTheDocument();
    expect(api.attach).not.toHaveBeenCalled();
  });

  it("greys every card while Enable is off and any ignored card, but keeps them operable", () => {
    let state = createInitialState(RULES);
    state = fill(state, "picture", 1);
    const { unmount } = renderAccordion(makeApi(state));
    expect(screen.getByTestId("ref-slot-picture-1")).toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-picture-2")).toHaveClass("ref-slot-ignored");
    expect(within(screen.getByTestId("ref-slot-picture-1")).getByRole("button", { name: /clear this card/i })).toBeEnabled();
    unmount();

    // Enabled: the 13th (by input order) card is greyed, the others are not.
    state = enabled(createInitialState(RULES));
    for (let i = 1; i <= 9; i += 1) state = fill(state, "picture", i);
    for (let i = 1; i <= 3; i += 1) state = fill(state, "video", i, 3);
    state = fill(state, "audio", 1, 3);
    renderAccordion(makeApi(state));
    expect(screen.getByTestId("ref-slot-audio-1")).toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-picture-1")).not.toHaveClass("ref-slot-ignored");
  });

  it("greys every card but the A2V one while A2V is in effect, empty cards included", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "picture", 1);
    state = fill(state, "audio", 2, 5);
    state = referenceReducer(state, { type: "setA2vSlot", slot: 2 });
    renderAccordion(makeApi(state));
    expect(screen.getByTestId("ref-slot-audio-2")).not.toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-picture-1")).toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-picture-2")).toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-video-1")).toHaveClass("ref-slot-ignored");
    expect(screen.getByTestId("ref-slot-audio-1")).toHaveClass("ref-slot-ignored");
  });

  it("locks every control while a submit is in flight", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "audio", 1, 5);
    state = referenceReducer(state, { type: "setA2vSlot", slot: 1 });
    renderAccordion(makeApi(state), { disabled: true });
    const card = screen.getByTestId("ref-slot-audio-1");
    expect(within(card).getByRole("button", { name: /clear this card/i })).toBeDisabled();
    expect(within(card).getByRole("button", { name: /change file/i })).toBeDisabled();
    expect(within(card).getByRole("button", { name: /add this tag/i })).toBeDisabled();
    expect(within(card).getByRole("checkbox", { name: /audio to video/i })).toBeDisabled();
    expect(screen.getByTestId("source-audio-strength")).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: /^enable$/i })).toBeDisabled();
  });

  it("shows the A2V checkbox only on Audios cards and moves it between cards", async () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "audio", 1, 5);
    state = fill(state, "audio", 2, 5);
    state = referenceReducer(state, { type: "setA2vSlot", slot: 1 });
    const api = makeApi(state);
    renderAccordion(api);
    expect(screen.getAllByRole("checkbox", { name: /audio to video/i })).toHaveLength(3);
    expect(within(screen.getByTestId("ref-slot-picture-1")).queryByRole("checkbox")).toBeNull();
    expect(within(screen.getByTestId("ref-slot-audio-1")).getByRole("checkbox")).toBeChecked();
    await userEvent.click(within(screen.getByTestId("ref-slot-audio-2")).getByRole("checkbox"));
    expect(api.setA2vSlot).toHaveBeenCalledWith(2);
    await userEvent.click(within(screen.getByTestId("ref-slot-audio-1")).getByRole("checkbox"));
    expect(api.setA2vSlot).toHaveBeenCalledWith(null);
  });

  it("shows Strength and the too-short warning only under the checked card", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "audio", 2, 1);
    state = referenceReducer(state, { type: "setA2vSlot", slot: 2 });
    renderAccordion(makeApi(state), {
      a2v: {
        available: true,
        strength: 0.75,
        onStrengthChange: vi.fn(),
        strengthMin: 0,
        strengthMax: 1,
        strengthStep: 0.05,
        tooShortWarning: "too short for A2V",
      },
    });
    const sliders = screen.getAllByTestId("source-audio-strength");
    expect(sliders).toHaveLength(1);
    expect(screen.getByTestId("ref-slot-audio-2")).toContainElement(sliders[0]!);
    expect(screen.getByTestId("source-audio-strength-value")).toHaveTextContent("0.75");
    expect(screen.getByText("too short for A2V")).toBeInTheDocument();
  });

  it("hides Strength when A2V is not available, and when no card is checked", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "audio", 1, 5);
    state = referenceReducer(state, { type: "setA2vSlot", slot: 1 });
    const { unmount } = renderAccordion(makeApi(state), {
      a2v: {
        available: false,
        strength: 0.5,
        onStrengthChange: vi.fn(),
        strengthMin: 0,
        strengthMax: 1,
        strengthStep: 0.05,
        tooShortWarning: null,
      },
    });
    expect(screen.queryByTestId("source-audio-strength")).toBeNull();
    unmount();
    renderAccordion(makeApi(enabled(createInitialState(RULES))));
    expect(screen.queryByTestId("source-audio-strength")).toBeNull();
  });

  it("shows the truncated, too-short and unknown-length warnings", () => {
    let state = enabled(createInitialState(RULES));
    state = fill(state, "video", 1, 10);
    state = fill(state, "video", 2, 6); // truncated to 5s
    state = fill(state, "audio", 1, 1); // too short
    state = fill(state, "audio", 2, null); // unknown
    renderAccordion(makeApi(state));
    expect(within(screen.getByTestId("ref-slot-video-2")).getByText(/Only the first 5\.0s/)).toBeInTheDocument();
    expect(within(screen.getByTestId("ref-slot-video-1")).queryByText(/Only the first/)).toBeNull();
    expect(within(screen.getByTestId("ref-slot-audio-1")).getByText(/Shorter than 2s/)).toBeInTheDocument();
    expect(within(screen.getByTestId("ref-slot-audio-2")).getByText(/could not be measured/)).toBeInTheDocument();
  });

  it("shows the uploading state and the error code", () => {
    let state = enabled(createInitialState(RULES));
    state = referenceReducer(state, {
      type: "attachStart",
      kind: "picture",
      slot: 1,
      filePath: "C:\\x\\a.png",
      fileName: "a.png",
    });
    state = referenceReducer(state, {
      type: "attachStart",
      kind: "picture",
      slot: 2,
      filePath: "C:\\x\\b.png",
      fileName: "b.png",
    });
    state = referenceReducer(state, { type: "attachFail", kind: "picture", slot: 2, errorCode: "UPLOAD_FAILED" });
    renderAccordion(makeApi(state));
    expect(within(screen.getByTestId("ref-slot-picture-1")).getByRole("status", { name: /uploading/i })).toBeInTheDocument();
    expect(within(screen.getByTestId("ref-slot-picture-2")).getByText("UPLOAD_FAILED")).toBeInTheDocument();
  });

  it("opens when openSignal changes", () => {
    const api = makeApi(createInitialState(RULES));
    const { container, rerender, props } = renderAccordion(api);
    const details = container.querySelector("details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    rerender(
      <LanguageProvider>
        <ReferenceAccordion {...props} openSignal={1} />
      </LanguageProvider>,
    );
    expect(details.open).toBe(true);
  });
});
