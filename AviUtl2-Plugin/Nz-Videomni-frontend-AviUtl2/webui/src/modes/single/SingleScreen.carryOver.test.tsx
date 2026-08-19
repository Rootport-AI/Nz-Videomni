import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { SingleScreen } from "./SingleScreen";

// X5 (併用プリフィルでアコーディオンを閉じない): `SingleScreen` derives
// `autoOpenReference`/`autoOpenA2v` (the two accordion auto-expand props it
// hands `GenerationForm`) not only from THIS prefill's intent but also from the
// OPPOSITE slot carried across the remount (`initialIntent.carryOver`). Before
// this fix, running #3/#7 (audio) with an existing reference video, or #2 with
// existing audio, kept the carried material but collapsed its accordion. These
// tests exercise the derivation through a real `SingleScreen` render (the
// intent->prop mapping lives here, not in `GenerationForm`); `GenerationForm`'s
// own open-on-prop behavior is covered by `GenerationForm.accordion.test.tsx`.

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

function selectionWithFile(filePath: string) {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 120,
    selected: [
      {
        layer: 1,
        frameStart: 0,
        frameEnd: 120,
        effectName: "音声ファイル",
        filePath,
        objectName: "a",
        textContent: null,
        mediaWidth: 1280,
        mediaHeight: 720,
        mediaDurationSec: 4,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

const CARRIED_REFERENCE = {
  id: "ref-123",
  fileName: "reference.mp4",
  filePath: "C:\\v\\reference.mp4",
  conditioningAttentionStrength: 1,
  referenceVideoStrength: 1,
};

const CARRIED_AUDIO = {
  id: "aud-123",
  fileName: "bg.wav",
  filePath: "C:\\a\\bg.wav",
};

function renderCreate(initialIntent?: GenerationPrefill) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  render(
    <Providers nativeBridge={nativeBridge}>
      <SingleScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        initialIntent={initialIntent}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLora={null}
        setControlLora={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
}

/** The IC-LoRA (reference video) `<details>` — scoped to the accordion
 * `<summary>` specifically. NOTE: when a reference video is active the embedded
 * `BatchSection` also renders a `<p>` warning containing "Reference video
 * (IC-LoRA)", so a plain text match is ambiguous — filter to the SUMMARY. */
function referenceDetails(): HTMLDetailsElement {
  const summary = screen
    .getAllByText(/reference video \(ic-lora\)/i)
    .find((el) => el.tagName === "SUMMARY");
  if (!summary) throw new Error("IC-LoRA accordion summary not found");
  return summary.closest("details") as HTMLDetailsElement;
}
function a2vDetails(): HTMLDetailsElement {
  const summary = screen.getAllByText(/audio to video \(a2v\)/i).find((el) => el.tagName === "SUMMARY");
  if (!summary) throw new Error("A2V accordion summary not found");
  return summary.closest("details") as HTMLDetailsElement;
}

/** Waits for the async `/config` load to resolve and the form to render (the
 * IC-LoRA accordion `<summary>` is always present once it has). */
async function waitForMount(): Promise<void> {
  await waitFor(
    () => {
      const summary = screen
        .getAllByText(/reference video \(ic-lora\)/i)
        .find((el) => el.tagName === "SUMMARY");
      if (!summary) throw new Error("not mounted yet");
    },
    { timeout: 5_000 },
  );
}

describe("SingleScreen — accordion auto-open from carryOver (X5)", () => {
  it("opens the IC-LoRA accordion when a #7 audio prefill carries an existing reference video (combo: both open)", async () => {
    renderCreate({
      intent: "audio-to-video",
      targetMode: "single",
      selection: selectionWithFile("C:\\a\\voice.wav"),
      carryOver: { referenceVideo: CARRIED_REFERENCE },
    });
    await waitForMount();

    // A2V opens from the intent; IC-LoRA opens purely from the carried reference
    // video — that's the X5 fix (before it, IC-LoRA collapsed and hid the carry).
    expect(a2vDetails().open).toBe(true);
    expect(referenceDetails().open).toBe(true);
  });

  it("opens the A2V accordion when a #2 reference prefill carries existing audio (combo: both open)", async () => {
    renderCreate({
      intent: "reference-video",
      targetMode: "single",
      selection: selectionWithFile("C:\\v\\ref.mp4"),
      carryOver: { sourceAudio: CARRIED_AUDIO },
    });
    await waitForMount();

    expect(referenceDetails().open).toBe(true);
    expect(a2vDetails().open).toBe(true);
  });

  it("leaves IC-LoRA collapsed when a #7 audio prefill carries nothing (null carryOver does not open it)", async () => {
    renderCreate({
      intent: "audio-to-video",
      targetMode: "single",
      selection: selectionWithFile("C:\\a\\voice.wav"),
    });
    await waitForMount();

    expect(a2vDetails().open).toBe(true);
    expect(referenceDetails().open).toBe(false);
  });

  it("leaves both accordions collapsed on a normal (non-prefilled) mount", async () => {
    renderCreate(undefined);
    await waitForMount();

    expect(referenceDetails().open).toBe(false);
    expect(a2vDetails().open).toBe(false);
  });
});
