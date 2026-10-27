import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useState, type ReactNode } from "react";
import { bridge as appBridge } from "../../bridge";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { dispatchCreateCommand, resetCreateLiveCommands } from "../../timeline/createLiveCommands";
import { SingleScreen } from "./SingleScreen";

// §1-107 (UI first): the Reference accordion wired into Create. With MiniMax
// H3's constraints (which carry `reference`):
//  - ✒ appends the card's tag to the shared prompt with one space;
//  - an Audios card ticked "Audio to video" (Enable on) reaches `/generate` as
//    the usual `source_audio {audio_id, strength}`, through `form.sourceAudio`;
//  - Enable off: the tick does nothing, no `source_audio`;
//  - the references themselves never reach the body (UI first);
// and with LTX's constraints Create is exactly as before.

const H3_CONSTRAINTS = resolveGenerationConstraints({
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: null,
  source_audio: { strength: { min: 0, max: 1, step: 0.05, default: 0.5 } },
  reference: {
    max_images: 9,
    max_videos: 3,
    max_audios: 3,
    max_total: 12,
    video_total_sec: 15,
    audio_total_sec: 15,
    clip_min_sec: 2,
  },
});

const notes: { kind: string; message: string }[] = [];

function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={(kind, message) => notes.push({ kind, message })}>
            <JobsProvider nativeBridge={nativeBridge}>{children}</JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

/** Owns the prompt the way `AppShell` does, and shows it for the assertions. */
function Harness({
  constraints,
  nativeBridge,
  initialPrompt,
}: {
  constraints: GenerationConstraints | undefined;
  nativeBridge: ReturnType<typeof createMockBridge>;
  initialPrompt: string;
}) {
  const [prompt, setPrompt] = useState(initialPrompt);
  return (
    <>
      <output data-testid="prompt">{prompt}</output>
      <SingleScreen
        prompt={prompt}
        onPromptChange={setPrompt}
        baseUrl={null}
        constraints={constraints}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLora={null}
        setControlLora={() => {}}
        controlLoraNames={new Set()}
      />
    </>
  );
}

function renderCreate(constraints: GenerationConstraints | undefined, initialPrompt = "a cat riding a skateboard") {
  // A 20 s wav, so the audio covers the form's default duration.
  const fs = createMockFs({
    folders: {
      "C:\\a": [{ name: "bg.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 20 }],
      "C:\\v": [{ name: "clip.mp4", sizeBytes: 4_096, mtimeMs: 0, durationSec: 5 }],
    },
  });
  const nativeBridge = createMockBridge({ delayMs: 0, fs });
  // Submissions go through the app-wide singleton client, so spy on that.
  const requestSpy = vi.spyOn(appBridge, "request");
  render(
    <Providers nativeBridge={nativeBridge}>
      <Harness constraints={constraints} nativeBridge={nativeBridge} initialPrompt={initialPrompt} />
    </Providers>,
  );
  const postsTo = (path: string) =>
    requestSpy.mock.calls
      .filter((c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === path)
      .map((c) => (c[1] as { body: Record<string, unknown> }).body);
  return { postsTo };
}

const slot = (id: string) => screen.getByTestId(id);

/** Waits for Create to mount its form (config loaded) and the card to exist. */
async function waitForCard(id: string) {
  await screen.findByTestId(id, {}, { timeout: 5_000 });
}

/** The card's ✒ button turns enabled once the card holds a file. */
async function waitForReady(id: string) {
  await waitFor(
    () => {
      const card = slot(id);
      expect(within(card).getByRole("button", { name: /add this tag/i })).toBeEnabled();
      expect(card.querySelector(".keyframe-spinner")).toBeNull();
    },
    { timeout: 5_000 },
  );
}

async function clickGenerateOnH3Grid() {
  // Without an AviUtl2 selection the form starts off H3's 17n+5 grid; type a
  // valid length, re-applied until Generate opens (the wav probe may re-seed it).
  await waitFor(
    () => {
      const duration = document.querySelector<HTMLInputElement>("input[type=number][step='17']")!;
      fireEvent.change(duration, { target: { value: "277" } });
      fireEvent.blur(duration);
      expect(duration.value).toBe("277");
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeEnabled();
    },
    { timeout: 5_000 },
  );
  fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));
}

beforeEach(() => {
  notes.length = 0;
});

afterEach(() => {
  resetCreateLiveCommands();
  vi.restoreAllMocks();
});

describe("SingleScreen — §1-107 Reference accordion", () => {
  it(
    "✒ appends the card's tag to the prompt with one space, and a right-clicked video opens the accordion",
    async () => {
      renderCreate(H3_CONSTRAINTS, "a cat  ");
      await waitForCard("ref-slot-video-1");
      const details = screen.getByText("Reference", { selector: "summary" }).closest("details") as HTMLDetailsElement;
      expect(details.open).toBe(false);

      act(() => {
        dispatchCreateCommand({ type: "addReferenceVideo", filePath: "C:\\v\\clip.mp4", fileName: "clip.mp4" });
      });
      await waitForReady("ref-slot-video-1");
      expect(details.open).toBe(true);
      expect(notes).toContainEqual({ kind: "info", message: "Loaded from a right-click: clip.mp4" });

      fireEvent.click(within(slot("ref-slot-video-1")).getByRole("button", { name: /add this tag/i }));
      expect(screen.getByTestId("prompt").textContent).toBe("a cat <Video 1>");
      // A second press appends again (duplicates are allowed, Q12).
      fireEvent.click(within(slot("ref-slot-video-1")).getByRole("button", { name: /add this tag/i }));
      expect(screen.getByTestId("prompt").textContent).toBe("a cat <Video 1> <Video 1>");
    },
    15_000,
  );

  it(
    "a right-click that finds no free card is refused with the full note; a missing file gets the missing-material note",
    async () => {
      renderCreate(H3_CONSTRAINTS);
      await waitForCard("ref-slot-audio-3");
      for (let n = 1; n <= 3; n++) {
        act(() => {
          dispatchCreateCommand({ type: "addReferenceAudio", filePath: `C:\\a\\bg.wav`, fileName: `bg${n}.wav` });
        });
      }
      await waitForReady("ref-slot-audio-3");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg4.wav" });
      });
      expect(notes.at(-1)).toEqual({
        kind: "warning",
        message: "The Reference Audios cards are full. Clear a card with 🔁 and try again.",
      });

      act(() => {
        dispatchCreateCommand({ type: "addReferenceVideo", filePath: null, fileName: "" });
      });
      expect(notes.at(-1)).toEqual({
        kind: "warning",
        message: "The source file was not found. It may have been moved or deleted.",
      });
    },
    15_000,
  );

  it(
    "Enable on + Audio to video: /generate carries source_audio {audio_id, strength} and no reference key",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS);
      await waitForCard("ref-slot-audio-1");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg.wav" });
        dispatchCreateCommand({ type: "addReferenceVideo", filePath: "C:\\v\\clip.mp4", fileName: "clip.mp4" });
      });
      await waitForReady("ref-slot-audio-1");
      await waitForReady("ref-slot-video-1");

      fireEvent.click(screen.getByRole("checkbox", { name: "Enable" }));
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      // The Strength slider sits under the ticked card (moved from the A2V accordion).
      const strength = await within(slot("ref-slot-audio-1")).findByTestId("source-audio-strength");
      fireEvent.change(strength, { target: { value: "0.5" } });

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const body = postsTo("/api/v1/generate")[0]!;
      const sourceAudio = body.source_audio as { audio_id: unknown; strength: unknown };
      expect(sourceAudio.strength).toBe(0.5);
      expect(sourceAudio.audio_id).toMatch(/^mock-audio-/);
      // UI first: the references themselves are not sent.
      for (const key of Object.keys(body)) expect(key).not.toMatch(/reference/i);
      expect(postsTo("/api/v1/generate/chain")).toHaveLength(0);
    },
    20_000,
  );

  it(
    "Enable off: the ticked Audio to video does nothing, no source_audio",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS);
      await waitForCard("ref-slot-audio-1");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg.wav" });
      });
      await waitForReady("ref-slot-audio-1");
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      expect(screen.getByRole("checkbox", { name: "Enable" })).not.toBeChecked();

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const body = postsTo("/api/v1/generate")[0]!;
      expect(body).not.toHaveProperty("source_audio");
      for (const key of Object.keys(body)) expect(key).not.toMatch(/reference/i);
    },
    20_000,
  );

  it(
    "LTX constraints: no Reference accordion, the A2V accordion as before",
    async () => {
      renderCreate(undefined);
      await screen.findByText(/audio to video \(a2v\)/i, {}, { timeout: 5_000 });
      expect(screen.queryByText("Reference", { selector: "summary" })).not.toBeInTheDocument();
      expect(screen.queryByTestId("ref-slot-audio-1")).not.toBeInTheDocument();
    },
    15_000,
  );
});
