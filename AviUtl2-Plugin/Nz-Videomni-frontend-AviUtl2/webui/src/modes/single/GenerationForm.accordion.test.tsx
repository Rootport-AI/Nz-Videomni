import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { GenerationForm } from "./GenerationForm";
import type { GenerationFormProps } from "./GenerationForm";
import { useGenerationForm } from "./useGenerationForm";
import { useKeyframes } from "./useKeyframes";

// `FrameRateSeedFields` (rendered inside `GenerationForm`) reads the job
// ledger via `useJobsContext` for its ♻ seed-reuse button; stub it the same
// way `CommonGenerationFields.test.tsx` does so this unit test doesn't need a
// full `JobsProvider` (irrelevant to the accordion behavior under test).
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({ jobs: [] }),
}));

// Right-click redesign W2 (accordion auto-expand): coverage for
// `GenerationForm`'s `autoOpenReference`/`autoOpenA2v` props — the IC-LoRA and
// A2V `<details className="form-accordion">` blocks should open on mount when
// `SingleScreen` derives one of them true from a right-click prefill's intent
// (`referenceVideo` -> "reference-video", `videoAudioToVideo`/`audioToVideo`
// -> "video-audio-to-video"/"audio-to-video" — see `timeline/menuRouting.ts`),
// and stay collapsed exactly like before otherwise. Unit-level: this harness
// drives the REAL `useGenerationForm`/`useKeyframes` hooks (mirroring
// `KeyframesPanel.test.tsx`'s pattern) straight into the real
// `GenerationForm`, but bypasses `SingleScreen` entirely — the intent ->
// prop derivation itself is a two-line static mapping in `SingleScreen.tsx`
// verified by reading the code, not re-exercised through the full
// AppShell/menu-routing integration stack (keeps this test independent of the
// parallel work in `App.prefill.test.tsx`).

function Harness(props: Partial<Pick<GenerationFormProps, "autoOpenReference" | "autoOpenA2v">>) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const form = useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge });
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge });
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
      nativeBridge={nativeBridge}
      autoOpenReference={props.autoOpenReference}
      autoOpenA2v={props.autoOpenA2v}
    />
  );
}

function renderForm(props: Partial<Pick<GenerationFormProps, "autoOpenReference" | "autoOpenA2v">> = {}) {
  return render(
    <LanguageProvider>
      <Harness {...props} />
    </LanguageProvider>,
  );
}

/** The IC-LoRA (reference video) `<details>` — matched by its summary text,
 * same query style `AppShell.controlLora.test.tsx` uses for the same block. */
function referenceDetails(): HTMLDetailsElement {
  return screen.getByText(/reference video \(ic-lora\)/i).closest("details") as HTMLDetailsElement;
}

/** The A2V (source audio) `<details>` — heading text is
 * `strings.single.sourceAudio.heading` ("Audio to video (A2V)"). */
function a2vDetails(): HTMLDetailsElement {
  return screen.getByText(/audio to video \(a2v\)/i).closest("details") as HTMLDetailsElement;
}

describe("GenerationForm — accordion auto-expand (right-click redesign W2)", () => {
  it("autoOpenReference opens the IC-LoRA accordion on mount and leaves A2V collapsed", () => {
    renderForm({ autoOpenReference: true });

    expect(referenceDetails().open).toBe(true);
    expect(a2vDetails().open).toBe(false);
  });

  it("autoOpenA2v opens the A2V accordion on mount and leaves IC-LoRA collapsed", () => {
    renderForm({ autoOpenA2v: true });

    expect(a2vDetails().open).toBe(true);
    expect(referenceDetails().open).toBe(false);
  });

  it("neither prop set (normal, non-prefilled mount): both accordions stay collapsed", () => {
    renderForm();

    expect(referenceDetails().open).toBe(false);
    expect(a2vDetails().open).toBe(false);
  });

  it("autoOpenReference=false (explicit) behaves identically to omitted — still collapsed", () => {
    renderForm({ autoOpenReference: false, autoOpenA2v: false });

    expect(referenceDetails().open).toBe(false);
    expect(a2vDetails().open).toBe(false);
  });
});
