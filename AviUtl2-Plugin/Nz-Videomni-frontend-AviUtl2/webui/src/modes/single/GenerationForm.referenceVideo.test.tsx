import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { GenerationForm } from "./GenerationForm";
import { useGenerationForm } from "./useGenerationForm";
import type { UseGenerationFormDeps } from "./useGenerationForm";
import { useKeyframes } from "./useKeyframes";

// `FrameRateSeedFields` reads the job ledger via `useJobsContext`; stub it the
// same way the accordion test does so this unit test needs no `JobsProvider`.
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({ jobs: [] }),
}));

// IC-LoRA/A2V card redesign: coverage for the whole-card, emoji-button
// reference-video section (`ReferenceVideoSection` inside `GenerationForm`).
// Harness drives the REAL `useGenerationForm`/`useKeyframes` hooks (mirroring
// `GenerationForm.accordion.test.tsx`), so pick/drop/probe all flow through the
// same mock bridge. jsdom renders `<details>` internals regardless of `open`,
// so no accordion expansion is needed.

function Harness({ bridge, formDeps }: { bridge: NativeBridge; formDeps?: Partial<UseGenerationFormDeps> | undefined }) {
  const form = useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: bridge, ...formDeps });
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

function renderForm(bridge: NativeBridge, formDeps?: Partial<UseGenerationFormDeps>) {
  return render(
    <LanguageProvider>
      <Harness bridge={bridge} formDeps={formDeps} />
    </LanguageProvider>,
  );
}

/** The reference-video card is the first `.source-section` in the form (the
 * A2V card is the second). */
function referenceCard(container: HTMLElement): HTMLElement {
  return container.querySelectorAll<HTMLElement>(".source-section")[0]!;
}
function a2vCard(container: HTMLElement): HTMLElement {
  return container.querySelectorAll<HTMLElement>(".source-section")[1]!;
}

describe("GenerationForm — ReferenceVideoSection (IC-LoRA card redesign)", () => {
  it("unselected: choose enabled, clear disabled, 'no reference' hint, em-dash thumb", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    renderForm(bridge);

    expect(screen.getByRole("button", { name: /choose reference video/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
    expect(screen.getByText(/no reference video selected/i)).toBeInTheDocument();
  });

  it("picking a video shows the placeholder thumbnail, the file name, and the 12.3s duration", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
    renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));

    expect(await screen.findByAltText(/reference video/i)).toBeInTheDocument();
    expect(screen.getByText("ref.mp4")).toBeInTheDocument();
    expect(await screen.findByText("12.3s")).toBeInTheDocument();
    // The button flips to "Change…" and Clear becomes enabled.
    expect(screen.getByRole("button", { name: /change reference video/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeEnabled();
  });

  it("clearing returns to the unselected state", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
    renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("12.3s");

    await user.click(screen.getByRole("button", { name: /^clear$/i }));

    expect(screen.getByText(/no reference video selected/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
  });

  it("dropping a video on the card behaves like picking one", async () => {
    const bridge = createMockBridge({ delayMs: 0, probeMediaInfoDurationSec: 12.3 });
    const { container } = renderForm(bridge);

    fireEvent.drop(referenceCard(container), { dataTransfer: { files: [new File([], "ref.mp4")] } });

    expect(await screen.findByAltText(/reference video/i)).toBeInTheDocument();
    expect(await screen.findByText("12.3s")).toBeInTheDocument();
  });

  it("dragging over the card highlights it with .source-section-dragover", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge);
    const card = referenceCard(container);

    fireEvent.dragEnter(card, { dataTransfer: { files: [] } });
    expect(card).toHaveClass("source-section-dragover");

    fireEvent.dragLeave(card, { dataTransfer: { files: [] } });
    expect(card).not.toHaveClass("source-section-dragover");
  });

  it("dropping an unsupported extension shows the unsupported-video message inline", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge);
    const card = referenceCard(container);

    fireEvent.drop(card, { dataTransfer: { files: [new File([], "notes.txt")] } });

    await waitFor(() => {
      const errorLine = card.querySelector(".field-hint-error");
      expect(errorLine).not.toBeNull();
      expect(errorLine?.textContent).toMatch(/unsupported file type/i);
    });
    expect(screen.getByText(/no reference video selected/i)).toBeInTheDocument();
  });

  it("a zero duration (unknown/still image) hides the seconds line", async () => {
    const user = userEvent.setup();
    // probeMediaInfoDurationSec defaults to 0 -> referenceVideoDurationSec null.
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4" });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("ref.mp4");

    const controls = referenceCard(container).querySelector(".source-input-controls");
    expect(controls?.textContent).toContain("ref.mp4");
    expect(controls?.textContent ?? "").not.toMatch(/\d\.\ds/);
  });

  it("the reference picker stays enabled while an A2V audio file is attached (combined mode)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge);

    // A2V+IC-LoRA combined mode: an attached audio file must NOT disable the
    // reference-video picker anymore — the two combine.
    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "track.wav")] } });

    await screen.findByAltText(/audio file/i);
    expect(screen.getByRole("button", { name: /choose reference video/i })).toBeEnabled();
  });

  it("shows the combined-mode note in both cards once a reference video and audio are attached", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("12.3s");
    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "track.wav")] } });

    await waitFor(() =>
      expect(
        screen.getAllByText(/both a reference video and an audio file are attached/i).length,
      ).toBeGreaterThanOrEqual(2),
    );
  });

  it("shows the soft IC-LoRA spill warning when the generated duration exceeds the reference (15.0s > 2.0s)", async () => {
    const user = userEvent.setup();
    // Default duration is 361 frames @ 24fps = 15.0s (raised 2026-08-19 to the
    // SMART comfort ceiling — see `comfortTable.comfortFramesForBudget`); a 2.0s
    // reference is shorter, so the spill warning must show. The dropped
    // (unseeded) wav sets A2V without a measured duration, so numFrames stays
    // at its default.
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 2.0 });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("2.0s");
    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "track.wav")] } });

    await waitFor(() =>
      expect(screen.getByText(/generated duration \(15\.0s\) is longer than the reference video \(2\.0s\)/i)).toBeInTheDocument(),
    );
  });

  it("hides the spill warning when the reference is longer than the generated duration (16.0s > 15.0s)", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 16.0 });
    const { container } = renderForm(bridge);

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("16.0s");
    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "track.wav")] } });

    // Both attached (combined note shows), but generated 15.0s < reference 16.0s,
    // so no spill warning.
    await waitFor(() =>
      expect(
        screen.getAllByText(/both a reference video and an audio file are attached/i).length,
      ).toBeGreaterThanOrEqual(2),
    );
    expect(screen.queryByText(/is longer than the reference video/i)).not.toBeInTheDocument();
  });

  it("renders two mode badges (IC-LoRA + A2V) when both are active, one when only a reference video is", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
    const { container } = renderForm(bridge);

    // Reference only: a single IC-LoRA badge.
    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("12.3s");
    let badges = container.querySelectorAll(".mode-badge");
    expect(badges).toHaveLength(1);
    expect(badges[0]?.textContent).toMatch(/IC-LoRA/);

    // Add audio: now IC-LoRA + A2V, side by side.
    fireEvent.drop(a2vCard(container), { dataTransfer: { files: [new File([], "track.wav")] } });
    await waitFor(() => expect(container.querySelectorAll(".mode-badge")).toHaveLength(2));
    badges = container.querySelectorAll(".mode-badge");
    const labels = Array.from(badges).map((b) => b.textContent);
    expect(labels).toContain("IC-LoRA");
    expect(labels).toContain("A2V");
  });

  // W6: the ❌ "Remove IC-LoRA" button — fully tears IC-LoRA down (reference
  // video + control LoRA + both opt-in strengths) in one click, and is enabled
  // whenever any of that state exists (so it also mops up an orphan left by 🔁).
  describe("❌ Remove IC-LoRA button (W6)", () => {
    it("is disabled when there is nothing to clear (no reference, no control LoRA, no strengths)", () => {
      const bridge = createMockBridge({ delayMs: 0 });
      renderForm(bridge);
      expect(screen.getByRole("button", { name: /remove ic-lora/i })).toBeDisabled();
    });

    it("becomes enabled once a reference video is ready and clears it back to the unselected state", async () => {
      const user = userEvent.setup();
      const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
      renderForm(bridge);

      await user.click(screen.getByRole("button", { name: /choose reference video/i }));
      await screen.findByText("12.3s");
      expect(screen.getByRole("button", { name: /remove ic-lora/i })).toBeEnabled();

      await user.click(screen.getByRole("button", { name: /remove ic-lora/i }));

      expect(screen.getByText(/no reference video selected/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /remove ic-lora/i })).toBeDisabled();
    });

    it("clears an orphan strength left behind by a bare 🔁 clear", async () => {
      const user = userEvent.setup();
      const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
      const { container } = renderForm(bridge);

      // Attach a reference video, then opt into a strength override.
      await user.click(screen.getByRole("button", { name: /choose reference video/i }));
      await screen.findByText("12.3s");
      const card = referenceCard(container);
      await user.click(within(card).getByRole("checkbox", { name: /conditioning attention strength/i }));

      // 🔁 clears ONLY the reference video — the strength survives as an orphan,
      // so ❌ must stay enabled to mop it up.
      await user.click(screen.getByRole("button", { name: /^clear$/i }));
      expect(screen.getByText(/no reference video selected/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /remove ic-lora/i })).toBeEnabled();

      // ❌ clears the orphan too → nothing left, button disables again.
      await user.click(screen.getByRole("button", { name: /remove ic-lora/i }));
      expect(screen.getByRole("button", { name: /remove ic-lora/i })).toBeDisabled();
    });
  });

  it("exposes the Control LoRA dropdown, and the reference-strength sliders only once ready", async () => {
    const user = userEvent.setup();
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "ref.mp4", probeMediaInfoDurationSec: 12.3 });
    const { container } = renderForm(bridge, { controlLoraNames: new Set(["canny-control"]) });

    const card = referenceCard(container);
    // The dropdown is present from the start (config-driven, not upload-driven).
    expect(within(card).getByRole("combobox", { name: /^control lora$/i })).toBeInTheDocument();
    // The two opt-in strength checkboxes only appear once a reference video is ready.
    expect(within(card).queryByRole("checkbox", { name: /conditioning attention strength/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /choose reference video/i }));
    await screen.findByText("12.3s");

    expect(within(card).getByRole("checkbox", { name: /conditioning attention strength/i })).toBeInTheDocument();
    expect(within(card).getByRole("checkbox", { name: /reference video strength/i })).toBeInTheDocument();
  });

  it("excludes in-outpainting from the dropdown even when the form reports it as a control LoRA", () => {
    // Docs/PENDING_TASKS_CLOSED.md §3-70 (filed as §1-13 at the time), T2
    // (2026-08-11): `in-outpainting` has its own dedicated place —
    // the Edit tab's Outpainting panel — so Create's own dropdown must never
    // offer it, even though `form.controlLoraNames` (unfiltered, per
    // `lora/controlLoras.ts`'s doc comment) legitimately contains it.
    const bridge = createMockBridge({ delayMs: 0 });
    const { container } = renderForm(bridge, {
      controlLoraNames: new Set(["canny-control", OUTPAINT_LORA_NAME, "pose-control"]),
    });

    const card = referenceCard(container);
    const select = within(card).getByRole("combobox", { name: /^control lora$/i }) as HTMLSelectElement;
    const values = Array.from(select.options)
      .map((o) => o.value)
      .filter((value) => value !== "");
    expect(values.sort()).toEqual(["canny-control", "pose-control"]);
    expect(values).not.toContain(OUTPAINT_LORA_NAME);
  });
});
