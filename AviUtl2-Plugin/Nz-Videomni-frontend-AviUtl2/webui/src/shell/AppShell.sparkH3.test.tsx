import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { NativeBridge, ResultOf } from "../bridge";
import { createMockBridge } from "../bridge/mockBridge";
import { en } from "../i18n/strings";
import { resetCreateLiveCommands } from "../timeline/createLiveCommands";
import { resetExtractObjectAudioChain } from "../timeline/extractObjectAudio";
import { resetInpaintSlots } from "../timeline/inpaintSlots";
import { resetProvisionalReservation } from "../timeline/provisionalReservation";
import { readStoredAcceleration } from "./accelerationSettings";
import { AppShell } from "./AppShell";

// 裁定 113 (MINIMAX_H3_DESIGN §6.11), end to end through the fixture server:
// Spark-H3's two notes show under the Generate button of Single, Chained and
// Edit's Retake / Outpainting / Inpainting on MiniMax H3, Generate stays
// pressable while they show, and an LTX base model hides the row and writes a
// leftover "on" back to off (so no note shows there).

const VISIBLE_PANEL = '[role="tabpanel"]:not([hidden])';
const STEPS_NOTE = en.sparkH3Notes.stepsTooFew;
const SHORT_NOTE = en.sparkH3Notes.videoTooShort;

function visiblePanel(): HTMLElement {
  const el = document.querySelector(VISIBLE_PANEL);
  if (!el) throw new Error("no visible tab panel");
  return el as HTMLElement;
}

/** The visible screen's Generate button (each sub-tab renders only its own). */
function generateButton(): HTMLButtonElement {
  const button = visiblePanel().querySelector<HTMLButtonElement>(".generation-column .generate-button");
  if (!button) throw new Error("no Generate button");
  return button;
}

async function renderApp(bridge: NativeBridge = createMockBridge({ delayMs: 0 })): Promise<NativeBridge> {
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  const select = (await screen.findByRole("combobox", { name: /base model/i })) as HTMLSelectElement;
  await waitFor(() => expect(select.value).toBe("LTX23"));
  // A prompt, so the only thing left between the screens and a pressable
  // Generate is whatever this test sets up.
  fireEvent.change(screen.getByPlaceholderText(/describe the video/i), { target: { value: "a cat riding a skateboard" } });
  return bridge;
}

async function switchTo(id: string) {
  const user = userEvent.setup();
  const select = screen.getByRole("combobox", { name: /base model/i }) as HTMLSelectElement;
  await user.selectOptions(select, id);
  await waitFor(() => expect(select.value).toBe(id), { timeout: 5_000 });
}

/** Opens Settings, presses Spark-H3 and the given Fast VAE choice, closes. */
async function setInSettings(fastVae: "Off" | "pi-H3_4s" | "pi-H3_2s") {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Settings" }));
  const dialog = await screen.findByRole("dialog", { name: "Settings" });
  const spark = within(within(dialog).getByRole("group", { name: "Spark-H3" })).getByRole("button", { name: "Spark-H3" });
  if (spark.getAttribute("aria-pressed") !== "true") await user.click(spark);
  await user.click(within(within(dialog).getByRole("group", { name: "Fast VAE" })).getByRole("button", { name: fastVae }));
  await user.click(within(dialog).getByRole("button", { name: /^Close$/ }));
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Settings" })).not.toBeInTheDocument());
}

function videoSelection(): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: 48,
    rangeEnd: 96,
    selected: [
      {
        layer: 3,
        frameStart: 10,
        frameEnd: 130,
        effectName: "動画ファイル",
        filePath: "C:\\v\\a.mp4",
        objectName: "a",
        textContent: null,
        mediaWidth: 1920,
        mediaHeight: 1080,
        mediaDurationSec: 5,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

describe("AppShell — Spark-H3 notes (裁定 113)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetExtractObjectAudioChain();
    resetProvisionalReservation();
    resetCreateLiveCommands();
    resetInpaintSlots();
  });
  afterEach(() => {
    window.localStorage.clear();
    resetInpaintSlots();
    vi.restoreAllMocks();
  });

  it(
    "LTX 2.3: a stored Spark-H3 on is written back to off, and no note shows",
    async () => {
      window.localStorage.setItem(
        "nzvideomni.acceleration",
        JSON.stringify({ sparkH3: true, sparkH3TopkPercent: 13, sparkH3FullSteps: 10 }),
      );
      await renderApp();
      await waitFor(() => expect(readStoredAcceleration().sparkH3).toBe(false));
      // The two slider values stay stored.
      expect(readStoredAcceleration()).toMatchObject({ sparkH3TopkPercent: 13, sparkH3FullSteps: 10 });
      expect(screen.queryByText(STEPS_NOTE)).not.toBeInTheDocument();
      expect(screen.queryByText(SHORT_NOTE)).not.toBeInTheDocument();
      await userEvent.setup().click(screen.getByRole("button", { name: "Settings" }));
      const dialog = await screen.findByRole("dialog", { name: "Settings" });
      expect(within(dialog).queryByRole("group", { name: "Spark-H3" })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "Single: the steps note follows pi-H3's steps, the short note follows the frame count, Generate stays enabled",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(visiblePanel()).getByText("Steps")).toBeInTheDocument());

      // Spark-H3 on, full-attention steps 2 (the default), pi-H3 2 steps: 2 <= 2.
      await setInSettings("pi-H3_2s");
      const panel = visiblePanel();
      await waitFor(() => expect(within(panel).getByText(STEPS_NOTE)).toBeInTheDocument());
      expect(within(panel).getByText(STEPS_NOTE)).toHaveClass("warning-banner", "warning-banner-mild");
      expect(generateButton()).toBeEnabled();
      // pi-H3 4 steps: 4 > 2, the note goes.
      await setInSettings("pi-H3_4s");
      await waitFor(() => expect(within(panel).queryByText(STEPS_NOTE)).not.toBeInTheDocument());

      // The shortest clip (22 frames at 1280x768) is below 12,288 rows.
      expect(within(panel).queryByText(SHORT_NOTE)).not.toBeInTheDocument();
      const duration = within(panel).getByText("Duration").closest("label")!.querySelector('input[type="number"]')!;
      fireEvent.change(duration, { target: { value: "22" } });
      fireEvent.blur(duration);
      await waitFor(() => expect(within(panel).getByText(SHORT_NOTE)).toBeInTheDocument());
      expect(generateButton()).toBeEnabled();
    },
    25_000,
  );

  it(
    "Chained: the steps note shows under Generate and Generate stays enabled",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled());
      await setInSettings("pi-H3_2s");
      await userEvent.setup().click(screen.getByRole("tab", { name: "Chained" }));
      await waitFor(() => expect(within(visiblePanel()).getByText("Clips")).toBeInTheDocument());
      const panel = visiblePanel();
      await waitFor(() => expect(within(panel).getByText(STEPS_NOTE)).toBeInTheDocument());
      // The prompt `renderApp` wrote makes the chain otherwise ready to send.
      await waitFor(() => expect(generateButton()).toBeEnabled());
      expect(within(panel).getByText(STEPS_NOTE)).toBeInTheDocument();

      // One short clip (the chain's 39-frame floor) is enough for the
      // short-video note.
      expect(within(panel).queryByText(SHORT_NOTE)).not.toBeInTheDocument();
      const durations = Array.from(panel.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']"));
      expect(durations.length).toBeGreaterThanOrEqual(2);
      fireEvent.change(durations[1]!, { target: { value: "39" } });
      fireEvent.blur(durations[1]!);
      await waitFor(() => expect(within(panel).getByText(SHORT_NOTE)).toBeInTheDocument());
      expect(generateButton()).toBeEnabled();
    },
    25_000,
  );

  // The Edit sub-tabs' "Generate stays pressable" half is in their own screen
  // tests (`EditScreen.retake.test.tsx`, `OutpaintingPanel.h3.test.tsx`,
  // `EditScreen.inpaint.test.tsx`), where a form can be made complete; this
  // fixture selection leaves each form with a reason of its own.
  for (const { action, subTab, panelClass } of [
    { action: "retakeRange", subTab: /^Retake/, panelClass: ".retake-panel" },
    { action: "outpaintVideo", subTab: /^Outpainting/, panelClass: ".outpaint-panel" },
    { action: "inpaintVideo", subTab: /^Inpainting/, panelClass: ".inpaint-panel" },
  ] as const) {
    it(
      `Edit (${action}): the steps note shows under that sub-tab's Generate`,
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(within(visiblePanel()).getByText("Steps")).toBeInTheDocument());
        await setInSettings("pi-H3_2s");
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action, selection: videoSelection() });
        });
        await waitFor(() => expect(screen.getByRole("tab", { name: "Edit" })).toHaveAttribute("aria-selected", "true"), {
          timeout: 5_000,
        });
        const strip = document.querySelector(".edit-subtabs") as HTMLElement;
        await waitFor(() => expect(within(strip).getByRole("tab", { name: subTab })).toHaveAttribute("aria-selected", "true"));
        expect(visiblePanel().querySelector(panelClass)).not.toBeNull();
        const column = visiblePanel().querySelector(".generation-column") as HTMLElement;
        await waitFor(() => expect(within(column).getByText(STEPS_NOTE)).toBeInTheDocument());
        expect(within(column).getByText(STEPS_NOTE)).toHaveClass("warning-banner", "warning-banner-mild");
        // One copy in the Edit column: only the visible sub-tab draws its notes.
        expect(within(column).getAllByText(STEPS_NOTE)).toHaveLength(1);
      },
      25_000,
    );
  }
});
