import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { ChainAudioPanel } from "./ChainAudioPanel";
import type { ChainAudioPanelH3 } from "./ChainAudioPanel";
import { useChainForm } from "./useChainForm";
import type { UseChainFormResult } from "./useChainForm";

// §1-120: the long-audio card's distribute mode (`h3` given). Kept apart from
// `ChainAudioPanel.test.tsx`, which pins the LTX card unchanged.

function Harness({
  nativeBridge,
  h3,
  overrides,
}: {
  nativeBridge: NativeBridge;
  h3: ChainAudioPanelH3;
  overrides?: Partial<UseChainFormResult>;
}) {
  const form = useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge });
  return (
    <LanguageProvider>
      <ChainAudioPanel form={{ ...form, ...overrides }} disabled={false} nativeBridge={nativeBridge} h3={h3} />
    </LanguageProvider>
  );
}

const t = en.chained.sourceAudio;

function makeH3(partial: Partial<ChainAudioPanelH3> = {}): ChainAudioPanelH3 {
  return {
    strength: 0.5,
    strengthRange: { min: 0, max: 1, step: 0.05 },
    onStrengthChange: vi.fn(),
    onDistribute: vi.fn(),
    unusedTailSec: null,
    ...partial,
  };
}

function setup(h3: ChainAudioPanelH3, options: MockBridgeOptions = {}, overrides?: Partial<UseChainFormResult>) {
  const nativeBridge = createMockBridge({ delayMs: 0, probeMediaInfoDurationSec: 30, ...options });
  const view = render(<Harness nativeBridge={nativeBridge} h3={h3} {...(overrides ? { overrides } : {})} />);
  return { ...view, nativeBridge };
}

function dropAudio(container: HTMLElement, fileName = "track.mp3") {
  fireEvent.drop(container.querySelector(".chain-audio-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

describe("ChainAudioPanel — §1-120 distribute mode", () => {
  it("draws the Strength slider and the distribute button instead of ↔️ and the added-clip slider", () => {
    setup(makeH3());
    expect(screen.queryByRole("button", { name: t.adjustButton })).toBeNull();
    expect(screen.queryByText(t.addedClipFramesLabel)).toBeNull();
    expect(screen.getByTestId("chain-audio-strength")).toHaveValue("0.5");
    expect(screen.getByTestId("chain-audio-strength-value")).toHaveTextContent("0.50");
    // Nothing attached yet: the button is off.
    expect(screen.getByRole("button", { name: t.h3.distributeButton })).toBeDisabled();
  });

  it("the slider and the button report to the screen once the audio is ready", async () => {
    const h3 = makeH3();
    const { container } = setup(h3);
    dropAudio(container);
    const button = screen.getByRole("button", { name: t.h3.distributeButton });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    expect(h3.onDistribute).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByTestId("chain-audio-strength"), { target: { value: "0.8" } });
    expect(h3.onStrengthChange).toHaveBeenCalledWith(0.8);
  });

  it("distributeDisabled stops the button", async () => {
    const { container } = setup(makeH3({ distributeDisabled: true }));
    dropAudio(container);
    await waitFor(() => expect(screen.getByText(/track\.mp3/)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: t.h3.distributeButton })).toBeDisabled();
  });

  it("note order: the V2V clash, then the unmeasurable length, then the unused tail", () => {
    const clash = setup(makeH3({ unusedTailSec: 12.3 }), {}, { hasSourceVideo: true, hasSourceAudio: true, audioProbeFailed: true });
    expect(screen.getByText(t.conflictsWithSourceVideo)).toBeInTheDocument();
    clash.unmount();
    const probe = setup(makeH3({ unusedTailSec: 12.3 }), {}, { hasSourceAudio: true, audioProbeFailed: true });
    expect(screen.getByText(t.h3.probeFailed)).toBeInTheDocument();
    probe.unmount();
    setup(makeH3({ unusedTailSec: 12.3 }), {}, { hasSourceAudio: true });
    expect(screen.getByText(t.h3.unusedTailNote("12.3"))).toBeInTheDocument();
  });

  it("no unused tail and nothing attached: the LTX card's empty line", () => {
    setup(makeH3());
    expect(screen.getByText(t.none)).toBeInTheDocument();
  });
});
