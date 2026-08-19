import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { ChainedScreen } from "./ChainedScreen";

/**
 * §1-15 (clip-wise IC-LoRA reference), plan F5: the two pieces that live on
 * the SCREEN rather than in `ChainReferencePanel` itself — the panel's actual
 * placement (W3 レイアウト再構成, 2026-08-11: inside its own accordion, below
 * the clip list, directly ABOVE the A2V accordion — reference before audio,
 * owner order) and the stage-1 comfort-budget warning banner, which needs the
 * WHOLE form's resolution and longest clip length, not just the panel's own
 * state.
 */
function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge}>{children}</JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

const CONTROL_NAMES = new Set(["canny-control"]);
const DOWNSCALE_FACTORS = new Map([["canny-control", 2]]);

async function renderChain(options: { probeMediaInfoWidth?: number; probeMediaInfoHeight?: number } = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={CONTROL_NAMES}
        referenceDownscaleFactors={DOWNSCALE_FACTORS}
      />
    </Providers>,
  );
  await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
  return { ...view, nativeBridge };
}

function attachReference(container: HTMLElement, fileName = "ref.mp4") {
  fireEvent.drop(container.querySelector(".chain-reference-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

/** Cranks the chain up to a size/length combination that lands comfortably
 * over the stage-1 budget with a reference attached: the mock config's own
 * maximum resolution (1920x1088 — `mockBridge.ts`'s `limits.max_width/
 * max_height`) and the longest a clip can be (481 frames,
 * `MAX_CLIP_NUM_FRAMES`). At `refScale=2` that's `30*17*61 + 15*8*61 =
 * 38,430` tokens against the 25,000 budget (`chainStage1Tokens`'s own
 * formula) — even the smaller deblur factor (`refScale=1`) would still clear
 * it, so this is a deliberately generous margin rather than a
 * knee-of-the-curve pin (that calibration lives in the backend's own
 * `tests/test_chain_math_reference.py`, plan G4).
 *
 * Driven by direct DOM queries rather than `getByRole`/`getByLabelText`:
 * `SizeFields`/`DurationField` wrap a RANGE and a NUMBER input in the SAME
 * `<label>`, so the accessible-name algorithm folds both inputs' live values
 * into every one of their names (e.g. "Width 1280") — there is no name string
 * stable enough to query by. */
function maximizeStage1Load(container: HTMLElement) {
  const sizeFields = container.querySelectorAll<HTMLElement>(".size-field-stack .field");
  const widthInput = sizeFields[0]!.querySelector<HTMLInputElement>('input[type="number"]')!;
  fireEvent.change(widthInput, { target: { value: "1920" } });
  const heightInput = sizeFields[1]!.querySelector<HTMLInputElement>('input[type="number"]')!;
  fireEvent.change(heightInput, { target: { value: "1088" } });

  const firstClipNumFramesInput = container.querySelector<HTMLInputElement>('.clip-card input[type="number"]')!;
  fireEvent.change(firstClipNumFramesInput, { target: { value: "481" } });
}

describe("ChainedScreen — §1-15 参照動画", () => {
  it("renders the clip list before the reference accordion, which comes before the audio accordion (W3, 2026-08-11)", async () => {
    const { container } = await renderChain();

    const audio = container.querySelector(".chain-audio-section");
    const reference = container.querySelector(".chain-reference-section");
    const clipList = container.querySelector(".clip-list");
    expect(audio).toBeInTheDocument();
    expect(reference).toBeInTheDocument();
    expect(clipList).toBeInTheDocument();

    // DOCUMENT_POSITION_FOLLOWING (4): the clip list comes first, then the
    // reference accordion, then the audio accordion (owner order: reference
    // before audio — swapped from the pre-W3 audio-first layout, and both now
    // sit below the clip list instead of above it).
    expect(clipList!.compareDocumentPosition(reference!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(reference!.compareDocumentPosition(audio!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  describe("the stage-1 comfort-budget banner", () => {
    it("stays hidden with no reference attached, even at a huge resolution/length", async () => {
      const { container } = await renderChain();

      maximizeStage1Load(container);

      expect(screen.queryByText(en.chained.referenceVideo.stage1OverBudgetWarning)).not.toBeInTheDocument();
    });

    it("stays hidden with a reference attached at a modest resolution/length", async () => {
      const { container } = await renderChain({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      attachReference(container);
      await screen.findByAltText(en.chained.referenceVideo.videoPlaceholderAlt);

      expect(screen.queryByText(en.chained.referenceVideo.stage1OverBudgetWarning)).not.toBeInTheDocument();
    });

    it("appears once a reference is attached AND the chain is over the comfort budget — and never blocks Generate", async () => {
      const { container } = await renderChain({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      attachReference(container);
      await screen.findByAltText(en.chained.referenceVideo.videoPlaceholderAlt);

      const user = userEvent.setup();
      await user.selectOptions(screen.getByRole("combobox", { name: en.chained.referenceVideo.controlLoraLabel }), "canny-control");

      maximizeStage1Load(container);

      expect(await screen.findByText(en.chained.referenceVideo.stage1OverBudgetWarning)).toBeInTheDocument();
      // Amber advisory, not the blocking red banner class.
      const banner = screen.getByText(en.chained.referenceVideo.stage1OverBudgetWarning);
      expect(banner).toHaveClass("warning-banner", "warning-banner-mild");
      // Non-blocking: Generate stays enabled (the reference is attached, a
      // control LoRA is selected, everything else about the default chain is
      // still valid).
      expect(screen.getByRole("button", { name: en.chained.generateButton })).toBeEnabled();
    });

    it("disappears again once the reference AND the control LoRA are both cleared (isReferenceActive turns false)", async () => {
      const { container } = await renderChain({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      attachReference(container);
      await screen.findByAltText(en.chained.referenceVideo.videoPlaceholderAlt);
      const user = userEvent.setup();
      const controlLoraSelect = screen.getByRole("combobox", { name: en.chained.referenceVideo.controlLoraLabel });
      await user.selectOptions(controlLoraSelect, "canny-control");
      maximizeStage1Load(container);
      await screen.findByText(en.chained.referenceVideo.stage1OverBudgetWarning);

      // `clearReference()` deliberately leaves the control-LoRA selection in
      // place (it's a setting, not material — see `useChainForm`'s own doc
      // comment), so `isReferenceActive` only turns false once BOTH are gone.
      await user.click(screen.getByRole("button", { name: en.chained.referenceVideo.clearButton }));
      expect(screen.getByText(en.chained.referenceVideo.stage1OverBudgetWarning)).toBeInTheDocument();

      await user.selectOptions(controlLoraSelect, "");

      expect(screen.queryByText(en.chained.referenceVideo.stage1OverBudgetWarning)).not.toBeInTheDocument();
    });
  });
});
