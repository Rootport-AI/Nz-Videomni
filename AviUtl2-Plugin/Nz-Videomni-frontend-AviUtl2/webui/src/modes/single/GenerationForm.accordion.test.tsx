import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { LTX_GENERATION_CONSTRAINTS } from "../../shell/generationConstraints";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { GenerationForm } from "./GenerationForm";
import type { GenerationFormProps } from "./GenerationForm";
import { useGenerationForm } from "./useGenerationForm";
import { useKeyframes } from "./useKeyframes";
import { useReferenceSlots } from "./useReferenceSlots";
import type { ReferenceRules } from "./referenceRules";

// `FrameRateSeedFields` (rendered inside `GenerationForm`) reads the job
// ledger via `useJobsContext` for its ♻ seed-reuse button; stub it the same
// way `CommonGenerationFields.test.tsx` does so this unit test doesn't need a
// full `JobsProvider` (irrelevant to the accordion behavior under test).
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({ jobs: [] }),
}));

// Right-click redesign W2 (accordion auto-expand), live since 台帳 §1-114:
// coverage for `GenerationForm`'s `referenceOpenSignal`/`a2vOpenSignal` props —
// the IC-LoRA and A2V `<details className="form-accordion">` blocks open each
// time `SingleScreen` bumps the matching counter for a `prefillMaterial`
// (#2 reference video / #7 audio-to-video), never on mount, and the user's own
// collapse is not fought until the next bump. Unit-level: this harness
// drives the REAL `useGenerationForm`/`useKeyframes` hooks (mirroring
// `KeyframesPanel.test.tsx`'s pattern) straight into the real
// `GenerationForm`, but bypasses `SingleScreen` entirely — the bump itself is
// covered end to end by `App.prefill.test.tsx`.

/** MiniMax H3's served `constraints.reference` (camelCase). */
const H3_REFERENCE_RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

type HarnessProps = Partial<Pick<GenerationFormProps, "referenceOpenSignal" | "a2vOpenSignal">> & {
  /** §1-107: pass the `reference` prop (what `SingleScreen` does while the
   * constraints carry `reference`). */
  withReference?: boolean;
};

function Harness(props: HarnessProps) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const form = useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge });
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge });
  const reference = useReferenceSlots(H3_REFERENCE_RULES, { nativeBridge });
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
      nativeBridge={nativeBridge}
      singleSourceAudio={null}
      sourceAudioStrength={1}
      onSourceAudioStrengthChange={() => {}}
      referenceOpenSignal={props.referenceOpenSignal}
      a2vOpenSignal={props.a2vOpenSignal}
      reference={
        props.withReference
          ? {
              api: reference,
              openSignal: 0,
              onAppendTag: () => {},
              a2v: {
                available: false,
                strength: 1,
                onStrengthChange: () => {},
                strengthMin: 0,
                strengthMax: 1,
                strengthStep: 0.05,
                tooShortWarning: null,
              },
            }
          : undefined
      }
    />
  );
}

function renderForm(props: HarnessProps = {}) {
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
  it("a mount never opens either accordion, whatever the starting signals", () => {
    renderForm({ referenceOpenSignal: 3, a2vOpenSignal: 5 });

    expect(referenceDetails().open).toBe(false);
    expect(a2vDetails().open).toBe(false);
  });

  it("bumping referenceOpenSignal opens the IC-LoRA accordion and leaves A2V collapsed", () => {
    const { rerender } = renderForm({ referenceOpenSignal: 0, a2vOpenSignal: 0 });
    rerender(
      <LanguageProvider>
        <Harness referenceOpenSignal={1} a2vOpenSignal={0} />
      </LanguageProvider>,
    );

    expect(referenceDetails().open).toBe(true);
    expect(a2vDetails().open).toBe(false);
  });

  it("bumping a2vOpenSignal opens the A2V accordion and leaves IC-LoRA collapsed", () => {
    const { rerender } = renderForm({ referenceOpenSignal: 0, a2vOpenSignal: 0 });
    rerender(
      <LanguageProvider>
        <Harness referenceOpenSignal={0} a2vOpenSignal={1} />
      </LanguageProvider>,
    );

    expect(a2vDetails().open).toBe(true);
    expect(referenceDetails().open).toBe(false);
  });

  it("a user collapse stays collapsed on an unrelated re-render and reopens on the next bump", () => {
    const { rerender } = renderForm({ referenceOpenSignal: 0, a2vOpenSignal: 0 });
    const rerenderWith = (referenceOpenSignal: number, a2vOpenSignal: number) =>
      rerender(
        <LanguageProvider>
          <Harness referenceOpenSignal={referenceOpenSignal} a2vOpenSignal={a2vOpenSignal} />
        </LanguageProvider>,
      );
    rerenderWith(0, 1);
    expect(a2vDetails().open).toBe(true);

    a2vDetails().open = false;
    rerenderWith(0, 1);
    expect(a2vDetails().open).toBe(false);

    rerenderWith(0, 2);
    expect(a2vDetails().open).toBe(true);
  });
});

describe("GenerationForm — §1-107 Reference accordion (MiniMax H3)", () => {
  it("with `reference`: the Reference accordion is drawn after the keyframes and the A2V accordion is not", () => {
    renderForm({ withReference: true });

    const referenceAccordion = screen.getByText("Reference", { selector: "summary" }).closest("details");
    expect(referenceAccordion).not.toBeNull();
    expect(screen.queryByText(/audio to video \(a2v\)/i)).not.toBeInTheDocument();
    // Right under the keyframes: the keyframe list precedes it in the document.
    const addKeyframe = screen.getByRole("button", { name: /add keyframe/i });
    expect(addKeyframe.compareDocumentPosition(referenceAccordion!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The IC-LoRA accordion is untouched by this prop.
    expect(referenceDetails()).not.toBeNull();
  });

  it("without `reference` (LTX): no Reference accordion, the A2V accordion as before", () => {
    renderForm();

    expect(screen.queryByText("Reference", { selector: "summary" })).not.toBeInTheDocument();
    expect(a2vDetails()).not.toBeNull();
  });
});
