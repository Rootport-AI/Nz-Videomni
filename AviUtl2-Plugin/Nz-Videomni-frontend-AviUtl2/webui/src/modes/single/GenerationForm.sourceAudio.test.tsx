import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFs } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { LTX_GENERATION_CONSTRAINTS } from "../../shell/generationConstraints";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { GenerationForm } from "./GenerationForm";
import { useGenerationForm } from "./useGenerationForm";
import { useKeyframes } from "./useKeyframes";

vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({ jobs: [] }),
}));

// IC-LoRA/A2V card redesign: coverage for the whole-card, emoji-button A2V
// source-audio section (`SourceAudioSection` inside `GenerationForm`). Same
// real-hook harness as `GenerationForm.referenceVideo.test.tsx`.

const H3_SOURCE_AUDIO = { strength: { min: 0, max: 1, step: 0.05, default: 1 } };

function Harness({
  bridge,
  singleSourceAudio = null,
  onStrengthChange = () => {},
}: {
  bridge: NativeBridge;
  singleSourceAudio?: typeof H3_SOURCE_AUDIO | null;
  onStrengthChange?: (value: number) => void;
}) {
  const form = useGenerationForm(FALLBACK_APP_CONFIG, "a music video", { nativeBridge: bridge });
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
  return (
    <GenerationForm
      form={form}
      keyframes={keyframes}
      frameIdxMultiple={LTX_GENERATION_CONSTRAINTS.keyframeGrid.step}
      gridOffset={LTX_GENERATION_CONSTRAINTS.keyframeGrid.offset}
      onDurationChange={(value) => form.setNumFrames(value)}
      onDurationCommit={(value) => form.setNumFrames(value)}
      onApplyPreset={form.applyPreset}
      disabled={false}
      configFallbackWarning={false}
      config={FALLBACK_APP_CONFIG}
      nativeBridge={bridge}
      singleSourceAudio={singleSourceAudio}
      sourceAudioStrength={1}
      onSourceAudioStrengthChange={onStrengthChange}
    />
  );
}

function renderForm(bridge: NativeBridge, props: Omit<Parameters<typeof Harness>[0], "bridge"> = {}) {
  return render(
    <LanguageProvider>
      <Harness bridge={bridge} {...props} />
    </LanguageProvider>,
  );
}

/** The A2V card is the second `.source-section` (the reference-video card is
 * the first). */
function a2vCard(container: HTMLElement): HTMLElement {
  return container.querySelectorAll<HTMLElement>(".source-section")[1]!;
}
function referenceCard(container: HTMLElement): HTMLElement {
  return container.querySelectorAll<HTMLElement>(".source-section")[0]!;
}

/** A mock filesystem seeding the wav durations both attach paths land on:
 * `ui.pickFile`'s default `C:\Users\mock\Pictures\picked-audio-1.wav` and a
 * dropped file's `C:\Users\mock\Downloads\<name>`. */
function fsWithWavDurations(): MockFs {
  return createMockFs({
    folders: {
      "C:\\Users\\mock\\Pictures": [{ name: "picked-audio-1.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 3.0 }],
      "C:\\Users\\mock\\Downloads": [{ name: "dropped.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 3.0 }],
    },
  });
}

describe("GenerationForm — SourceAudioSection (A2V card redesign)", () => {
  it("unselected: choose enabled, remove disabled, 'no audio' hint", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    renderForm(bridge);

    expect(screen.getByRole("button", { name: /choose audio file/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /remove audio/i })).toBeDisabled();
    expect(screen.getByText(/no audio file attached/i)).toBeInTheDocument();
  });

  it("picking a wav shows the ♫ placeholder, the file name, and the 3.0s duration", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose audio file/i }));

    expect(await screen.findByAltText(/audio file/i)).toBeInTheDocument();
    expect(screen.getByText("picked-audio-1.wav")).toBeInTheDocument();
    expect(await screen.findByText("3.0s")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /remove audio/i })).toBeEnabled();
  });

  it("dropping a wav on the card behaves like picking one", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    const { container } = renderForm(bridge);

    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "dropped.wav")] } });

    expect(await screen.findByAltText(/audio file/i)).toBeInTheDocument();
    expect(await screen.findByText("3.0s")).toBeInTheDocument();
  });

  it("dragging over the card highlights it with .source-section-dragover", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge);
    const card = a2vCard(container);

    fireEvent.dragEnter(card, { dataTransfer: { files: [] } });
    expect(card).toHaveClass("source-section-dragover");

    fireEvent.dragLeave(card, { dataTransfer: { files: [] } });
    expect(card).not.toHaveClass("source-section-dragover");
  });

  it("removing returns to the unselected state", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await screen.findByText("3.0s");

    await user.click(screen.getByRole("button", { name: /remove audio/i }));

    expect(screen.getByText(/no audio file attached/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /remove audio/i })).toBeDisabled();
  });

  it("an unsupported drop shows the message inline, and Remove clears it via resetErrorSignal", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    const { container } = renderForm(bridge);
    const card = a2vCard(container);

    // Attach a wav first (so Remove is enabled), then drop an unsupported file.
    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await screen.findByText("3.0s");

    fireEvent.drop(card, { dataTransfer: { files: [new File([], "notes.txt")] } });
    await waitFor(() => {
      const errorLine = card.querySelector(".field-hint-error");
      expect(errorLine?.textContent).toMatch(/unsupported file type/i);
    });

    await user.click(screen.getByRole("button", { name: /remove audio/i }));

    // Both the audio AND the stale drop error are gone.
    expect(card.querySelector(".field-hint-error")).toBeNull();
    expect(screen.getByText(/no audio file attached/i)).toBeInTheDocument();
  });

  it("a non-wav audio file attaches but shows no seconds line (wav-only duration)", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "song.mp3" });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await screen.findByText("song.mp3");

    const controls = a2vCard(container).querySelector(".source-input-controls");
    expect(controls?.textContent).toContain("song.mp3");
    expect(controls?.textContent ?? "").not.toMatch(/\d\.\ds/);
  });

  it("the audio picker stays enabled while a reference video (IC-LoRA) is active (combined mode)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge);

    fireEvent.drop(referenceCard(container), { dataTransfer: { files: [new File([], "clip.mp4")] } });

    // A2V+IC-LoRA combined mode: the two are no longer mutually exclusive, so an
    // active reference video must NOT disable the audio picker.
    await screen.findByAltText(/reference video/i);
    expect(screen.getByRole("button", { name: /choose audio file/i })).toBeEnabled();
  });

  it("shows the combined-mode note when both an audio file and a reference video are attached", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await screen.findByText("3.0s");
    fireEvent.drop(referenceCard(container), { dataTransfer: { files: [new File([], "clip.mp4")] } });

    // The same combined-mode copy is rendered in BOTH cards (reference + audio).
    await waitFor(() =>
      expect(
        screen.getAllByText(/both a reference video and an audio file are attached/i).length,
      ).toBeGreaterThanOrEqual(2),
    );
  });
});

describe("GenerationForm — single-clip A2V Strength (constraints.source_audio)", () => {
  it("the slider and its 1.00 readout render only when the engine publishes source_audio; the chain hint goes with it", async () => {
    const bridge = createMockBridge({ delayMs: 0, fs: fsWithWavDurations() });
    const user = userEvent.setup();
    const onStrengthChange = vi.fn();
    const { unmount } = renderForm(bridge, { singleSourceAudio: H3_SOURCE_AUDIO, onStrengthChange });

    const slider = screen.getByTestId("source-audio-strength") as HTMLInputElement;
    expect(slider.max).toBe("1");
    expect(slider.step).toBe("0.05");
    expect(screen.getByTestId("source-audio-strength-value")).toHaveTextContent("1.00");
    fireEvent.change(slider, { target: { value: "0.5" } });
    expect(onStrengthChange).toHaveBeenCalledWith(0.5);

    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /remove audio/i })).toBeEnabled());
    expect(screen.queryByText(/long audio can use chained/i)).not.toBeInTheDocument();
    unmount();

    // LTX-like: no slider, and the "long audio -> Chained" note shows once audio is attached.
    renderForm(bridge, { singleSourceAudio: null });
    expect(screen.queryByTestId("source-audio-strength")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /choose audio file/i }));
    await waitFor(() => expect(screen.getByText(/long audio can use chained/i)).toBeInTheDocument());
  });
});
