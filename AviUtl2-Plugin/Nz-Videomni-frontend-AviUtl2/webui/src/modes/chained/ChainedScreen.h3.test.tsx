import { render, screen, waitFor, within } from "@testing-library/react";
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
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { ChainedScreen } from "./ChainedScreen";
import type { ChainedScreenProps } from "./ChainedScreen";

// §1-106: the Chained screen given MiniMax H3's served constraints and the
// booleans `AppShell` derives from its `unsupported_features` — rendered
// directly, with no engine id anywhere. `AppShell.minimaxH3.test.tsx` covers the
// same screen end to end from the fixture's `GET /models`.

const H3: GenerationConstraints = resolveGenerationConstraints({
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  chain: { overlap_frames: 22 },
});

const H3_PROPS: Partial<ChainedScreenProps> = {
  constraints: H3,
  v2vUnavailable: true,
  a2vUnavailable: true,
  endSourceUnavailable: true,
  referenceUnavailable: true,
  startImageUnavailable: true,
  batchI2vLongUnavailable: true,
  loraAudioHidden: true,
};

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

function screenWith(nativeBridge: ReturnType<typeof createMockBridge>, extra: Partial<ChainedScreenProps>) {
  return (
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
        {...extra}
      />
    </Providers>
  );
}

function renderChain(extra: Partial<ChainedScreenProps> = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return render(screenWith(nativeBridge, extra));
}

describe("ChainedScreen — fixed-overlap mode (§1-106)", () => {
  it("hides the seam blend, chunked upsample, stage-2 window, CFG row and the Total/Est. line", async () => {
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(screen.queryByText(en.chained.overlapFramesLabel)).not.toBeInTheDocument();
    expect(screen.queryByText(en.chained.overlapStrengthLabel)).not.toBeInTheDocument();
    expect(screen.queryByText(en.chained.chunkedUpsampleLabel)).not.toBeInTheDocument();
    expect(screen.queryByText(en.chained.stage2Window.label)).not.toBeInTheDocument();
    expect(container.querySelector(".chained-section-divider")).toBeNull();
    expect(screen.queryByText(en.single.cfgScale.label)).not.toBeInTheDocument();
    expect(screen.getByText(en.single.sampling.steps)).toBeInTheDocument();
    expect(container.querySelector(".generate-bar-hint")).toBeNull();
    // No stage-2 comfort guides on the size sliders.
    expect(container.querySelector(".size-marker")).toBeNull();
  });

  it("closes the SOURCE card whole, shows the opening-image line (not the video one) and drops Batch i2v-long", async () => {
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(screen.getByText(en.chained.unavailableOnBaseModel.startImage)).toBeInTheDocument();
    expect(screen.queryByText(en.chained.unavailableOnBaseModel.sourceVideo)).not.toBeInTheDocument();
    const source = container.querySelector(".source-section") as HTMLElement;
    expect(within(source).getByRole("button", { name: en.chained.sourceInput.chooseImageOnlyButton })).toBeDisabled();
    expect(container.querySelector("details.batch-section")).toBeNull();
    expect(screen.queryByText(en.batchI2vLong.heading)).not.toBeInTheDocument();
  });

  it("steps the clip durations by 17 and pins fps", async () => {
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    const inputs = container.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']");
    expect(inputs).toHaveLength(2);
    for (const input of inputs) {
      expect(input.step).toBe("17");
      expect(input.min).toBe("22");
    }
    expect(screen.getByRole("spinbutton", { name: /fps/i })).toBeDisabled();
  });

  it("LTX (no constraints, no flags) is unchanged: blend, chunked, window, CFG row, Total line, Batch i2v-long", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(screen.getByText(en.chained.overlapFramesLabel)).toBeInTheDocument();
    expect(screen.getByText(en.chained.chunkedUpsampleLabel)).toBeInTheDocument();
    expect(screen.getByText(en.chained.stage2Window.label)).toBeInTheDocument();
    expect(screen.getByText(en.single.cfgScale.label)).toBeInTheDocument();
    expect(container.querySelector(".generate-bar-hint")?.textContent).toMatch(/Total input:/);
    // The stage-2 comfort guides ARE drawn here, so the H3 case's "none" is meaningful.
    expect(container.querySelector(".size-marker")).not.toBeNull();
    expect(container.querySelector("details.batch-section")).not.toBeNull();
    expect(screen.queryByText(en.chained.unavailableOnBaseModel.startImage)).not.toBeInTheDocument();
    for (const input of container.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']")) {
      expect(input.step).toBe("8");
    }
  });

  it("a closed panel keeps its clear button: leftover material can be removed and the reason goes away", async () => {
    // An opening image attached while the panel was open (LTX), then the base
    // model switches to one that closes it (same screen instance, as in the app).
    const nativeBridge = createMockBridge({ delayMs: 0 });
    const { container, rerender } = render(screenWith(nativeBridge, {}));
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    const user = userEvent.setup();
    const source = () => container.querySelector(".source-section") as HTMLElement;
    await user.click(within(source()).getByRole("button", { name: en.chained.sourceInput.chooseButton }));
    await waitFor(() => expect(container.querySelector(".source-section img")).not.toBeNull(), { timeout: 5_000 });

    rerender(screenWith(nativeBridge, H3_PROPS));
    const reason = en.chained.generateReasons.materialInClosedPanel;
    await screen.findByText(reason);
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
    // Adding is closed, clearing is not.
    expect(within(source()).getByRole("button", { name: en.chained.sourceInput.changeButton })).toBeDisabled();
    const clear = within(source()).getByRole("button", { name: en.chained.sourceInput.clearButton });
    expect(clear).toBeEnabled();
    await user.click(clear);
    await waitFor(() => expect(screen.queryByText(reason)).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeEnabled();
  });
});
