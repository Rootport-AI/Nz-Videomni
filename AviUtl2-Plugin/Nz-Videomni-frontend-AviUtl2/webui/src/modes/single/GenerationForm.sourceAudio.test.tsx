import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFs } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
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

function Harness({ bridge }: { bridge: NativeBridge }) {
  const form = useGenerationForm(FALLBACK_APP_CONFIG, "a music video", { nativeBridge: bridge });
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
  return (
    <GenerationForm
      form={form}
      keyframes={keyframes}
      frameIdxMultiple={FALLBACK_APP_CONFIG.limits.conditioning_frame_idx_multiple}
      gridOffset={FALLBACK_APP_CONFIG.limits.conditioning_keyframe_grid_offset}
      onDurationChange={(value) => form.setNumFrames(value)}
      onDurationCommit={(value) => form.setNumFrames(value)}
      onApplyPreset={form.applyPreset}
      disabled={false}
      configFallbackWarning={false}
      config={FALLBACK_APP_CONFIG}
      nativeBridge={bridge}
    />
  );
}

function renderForm(bridge: NativeBridge) {
  return render(
    <LanguageProvider>
      <Harness bridge={bridge} />
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
