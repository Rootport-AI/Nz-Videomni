import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider, useToasts } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { ChainedScreen } from "./ChainedScreen";

/**
 * §1-16 長尺A2V, Step F3: the two pieces of the audio UI that live on the SCREEN
 * rather than in the card — the per-clip 🎵 badge (`ClipCard`, fed by
 * `useChainForm.audioSegmentWindowsForClips`) and the auto-fit toasts.
 *
 * Both are driven end to end: a real audio file is dropped on the real
 * `ChainAudioPanel`, which runs the real once-per-track auto-fit, whose real
 * outcome reaches the real toast effect. The four outcomes are produced by
 * choosing the TRACK LENGTH and the clip list, exactly as a user would.
 */

/** `ToastProvider` only holds state — `shell/Toasts.tsx` renders it in the app.
 * This stands in for that renderer so the messages are assertable here. */
function ToastSpy() {
  const { toasts } = useToasts();
  return (
    <ul data-testid="toasts">
      {toasts.map((toast) => (
        <li key={toast.id} data-kind={toast.kind}>
          {toast.message}
        </li>
      ))}
    </ul>
  );
}

function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge}>
              {children}
              <ToastSpy />
            </JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

async function renderChain(options: MockBridgeOptions = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
  await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
  return { ...view, nativeBridge };
}

function dropAudio(container: HTMLElement, fileName = "track.mp3") {
  fireEvent.drop(container.querySelector(".chain-audio-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function toastMessages(): string[] {
  return Array.from(screen.getByTestId("toasts").children).map((li) => li.textContent ?? "");
}

const fit = en.chained.sourceAudio.fitToast;

describe("ChainedScreen — §1-16 長尺A2V", () => {
  describe("the per-clip 🎵 badge", () => {
    it("is absent while no audio is attached", async () => {
      const { container } = await renderChain();
      expect(container.querySelectorAll(".clip-card-audio-window")).toHaveLength(0);
    });

    it("appears on every clip card once a measured track is attached, starting at 0.00s", async () => {
      const { container } = await renderChain({ probeMediaInfoDurationSec: 60 });

      dropAudio(container);

      await waitFor(() => expect(container.querySelectorAll(".clip-card-audio-window").length).toBeGreaterThan(0));
      const badges = Array.from(container.querySelectorAll(".clip-card-audio-window"));
      // One per clip card (the auto-fit may have changed how many there are).
      expect(badges).toHaveLength(container.querySelectorAll(".clip-card").length);
      expect(badges[0]?.textContent).toMatch(/^🎵 Audio 0\.00s – \d+\.\d\ds$/);
      expect(badges[0]).toHaveAttribute("title", en.chained.clipAudioWindow.tooltip);
      // Consecutive windows OVERLAP (the seam crossfade the tooltip describes):
      // the second one starts before the first one ends.
      if (badges.length > 1) {
        const first = badges[0]!.textContent!.match(/(\d+\.\d\d)s$/)![1]!;
        const second = badges[1]!.textContent!.match(/ (\d+\.\d\d)s –/)![1]!;
        expect(Number(second)).toBeLessThan(Number(first));
      }
    });

    it("disappears again when the track is removed", async () => {
      const user = userEvent.setup();
      const { container } = await renderChain({ probeMediaInfoDurationSec: 60 });

      dropAudio(container);
      await waitFor(() => expect(container.querySelectorAll(".clip-card-audio-window").length).toBeGreaterThan(0));

      await user.click(screen.getByRole("button", { name: en.chained.sourceAudio.clearButton }));

      expect(container.querySelectorAll(".clip-card-audio-window")).toHaveLength(0);
    });
  });

  describe("the ↔️ auto-fit toasts", () => {
    it("adjusted: a long track re-lays the clips out and says so, exactly once", async () => {
      const { container } = await renderChain({ probeMediaInfoDurationSec: 60 });

      dropAudio(container);

      await waitFor(() => expect(toastMessages()).toEqual([fit.adjusted(60)]));
      expect(within(screen.getByTestId("toasts")).getByText(fit.adjusted(60))).toHaveAttribute("data-kind", "success");
      // The once-per-track claim plus the id-keyed ref: the effect re-runs on
      // every later render (the clip list changed, the badges mounted) without
      // ever re-toasting.
      fireEvent.change(screen.getAllByRole("slider")[0]!, { target: { value: "1024" } });
      await waitFor(() => expect(toastMessages()).toHaveLength(1));
    });

    it("alreadyFits: pressing ↔️ again on a chain that already matches", async () => {
      const user = userEvent.setup();
      const { container } = await renderChain({ probeMediaInfoDurationSec: 60 });

      dropAudio(container);
      await waitFor(() => expect(toastMessages()).toHaveLength(1));

      await user.click(screen.getByRole("button", { name: en.chained.sourceAudio.adjustButton }));

      await waitFor(() => expect(toastMessages()).toEqual([fit.adjusted(60), fit.alreadyFits]));
    });

    it("cannotFit: a track too short for even the smallest chain, with the 'lower the slider' way out", async () => {
      const { container } = await renderChain({ probeMediaInfoDurationSec: 0.5 });

      dropAudio(container);

      await waitFor(() => expect(toastMessages()).toEqual([fit.cannotFit]));
      expect(within(screen.getByTestId("toasts")).getByText(fit.cannotFit)).toHaveAttribute("data-kind", "warning");
      // Step F1 hand-off: the resolver's smallest chain is the slider value, so
      // the copy MUST offer lowering it — otherwise this is a dead end.
      expect(fit.cannotFit).toMatch(/added clip length/i);
    });

    it("noFlexibleClips: every card edited by hand leaves the resolver nothing to move", async () => {
      const { container } = await renderChain({ probeMediaInfoDurationSec: 60 });

      // Touching BOTH clips' prompts consumes both `intact` flags.
      for (const textarea of screen.getAllByPlaceholderText(en.chained.clipPromptPlaceholder)) {
        fireEvent.change(textarea, { target: { value: "a sunset" } });
      }

      dropAudio(container);

      await waitFor(() => expect(toastMessages()).toEqual([fit.noFlexibleClips]));
      // Nothing was touched: the clip list is still the pair the user edited.
      expect(container.querySelectorAll(".clip-card")).toHaveLength(2);
    });
  });
});
