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
import type { TransformerRole } from "../../api/types";
import { SingleScreen } from "./SingleScreen";

// §1-107: the Reference accordion wired into Create. With MiniMax H3's
// constraints (which carry `reference`):
//  - ✒ appends the card's tag to the shared prompt with one space;
//  - an Audios card ticked "Audio to video" (Enable on) reaches `/generate` as
//    the usual `source_audio {audio_id, strength}`, through `form.sourceAudio`,
//    and no `references` (A2V is exclusive);
//  - Enable off: the tick does nothing (the checkbox is disabled, its check
//    kept), no `source_audio`, no `references`;
//  - Enable on + a tick on an empty card: Generate stops with its reason; on
//    a card still uploading, with the upload-wait reason (§1-117);
//  - Enable on: the used cards go out as `references` (not on FL2VA, never
//    off H3), a still-uploading card that would be sent blocks Generate, and
//    the role / empty notices show under Generate;
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
  transformerRole = null,
}: {
  constraints: GenerationConstraints | undefined;
  nativeBridge: ReturnType<typeof createMockBridge>;
  initialPrompt: string;
  transformerRole?: TransformerRole | null;
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
        transformerRole={transformerRole}
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

function renderCreate(
  constraints: GenerationConstraints | undefined,
  initialPrompt = "a cat riding a skateboard",
  options: { transformerRole?: TransformerRole | null; holdUploads?: boolean } = {},
) {
  // A 20 s wav, so the audio covers the form's default duration.
  const fs = createMockFs({
    folders: {
      "C:\\a": [{ name: "bg.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 20 }],
      "C:\\v": [{ name: "clip.mp4", sizeBytes: 4_096, mtimeMs: 0, durationSec: 5 }],
    },
  });
  const nativeBridge = createMockBridge({ delayMs: 0, fs, holdUploads: options.holdUploads ?? false });
  // Submissions go through the app-wide singleton client, so spy on that.
  const requestSpy = vi.spyOn(appBridge, "request");
  const tree = (next: GenerationConstraints | undefined, role: TransformerRole | null) => (
    <Providers nativeBridge={nativeBridge}>
      <Harness constraints={next} nativeBridge={nativeBridge} initialPrompt={initialPrompt} transformerRole={role} />
    </Providers>
  );
  const view = render(tree(constraints, options.transformerRole ?? null));
  const postsTo = (path: string) =>
    requestSpy.mock.calls
      .filter((c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === path)
      .map((c) => (c[1] as { body: Record<string, unknown> }).body);
  /** Re-renders with another base model's constraints / role (a switch). */
  const switchTo = (next: GenerationConstraints | undefined, role: TransformerRole | null = null) =>
    view.rerender(tree(next, role));
  return { postsTo, switchTo, nativeBridge };
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
      // A2V is exclusive: the other cards are left out, so no `references`.
      for (const key of Object.keys(body)) expect(key).not.toMatch(/reference/i);
      expect(postsTo("/api/v1/generate/chain")).toHaveLength(0);
    },
    20_000,
  );

  it(
    "Enable off: the ticked Audio to video does nothing (disabled, still checked), no source_audio",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS);
      await waitForCard("ref-slot-audio-1");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg.wav" });
      });
      await waitForReady("ref-slot-audio-1");
      fireEvent.click(screen.getByRole("checkbox", { name: "Enable" }));
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      fireEvent.click(screen.getByRole("checkbox", { name: "Enable" }));
      expect(screen.getByRole("checkbox", { name: "Enable" })).not.toBeChecked();
      const a2vBox = within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" });
      expect(a2vBox).toBeDisabled();
      expect(a2vBox).toBeChecked();

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

// ── §1-107: the `references` body, the upload gate and the notices ─────────────

const NOTICE = {
  fl2va: "An FL2VA model is loaded. The references will be ignored. Load a Ref2VA model or a Hybrid model.",
  ref2va: "A Ref2VA model is loaded. When no references are used, an FL2VA model or a Hybrid model is more accurate.",
  none: "There are no reference files.",
};
const CARD_UPLOADING = "Wait for the reference file upload to finish.";
const A2V_AUDIO_MISSING = "Add the audio to use for Audio to video.";

const generateButton = () => document.querySelector<HTMLButtonElement>(".generate-button")!;

/** Puts the form on H3's 17n+5 grid (as `clickGenerateOnH3Grid` does) and
 * waits for Generate to open, without clicking it. */
async function putOnH3Grid() {
  await waitFor(
    () => {
      const duration = document.querySelector<HTMLInputElement>("input[type=number][step='17']")!;
      fireEvent.change(duration, { target: { value: "277" } });
      fireEvent.blur(duration);
      expect(duration.value).toBe("277");
      expect(generateButton()).toBeEnabled();
    },
    { timeout: 5_000 },
  );
}

function openReference() {
  const summary = screen.getByText("Reference", { selector: "summary" });
  const details = summary.closest("details") as HTMLDetailsElement;
  if (!details.open) fireEvent.click(summary);
}

function setEnabled(on: boolean) {
  const box = screen.getByRole("checkbox", { name: "Enable" }) as HTMLInputElement;
  if (box.checked !== on) fireEvent.click(box);
}

function pickPicture(slotNumber: number) {
  fireEvent.click(within(slot(`ref-slot-picture-${slotNumber}`)).getByRole("button", { name: /choose file/i }));
}

async function clickGenerateOnLtxGrid() {
  await waitFor(
    () => {
      const duration = document.querySelector<HTMLInputElement>("input[type=number][step='8']")!;
      fireEvent.change(duration, { target: { value: "121" } });
      fireEvent.blur(duration);
      expect(duration.value).toBe("121");
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeEnabled();
    },
    { timeout: 5_000 },
  );
  fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));
}

const noticeShown = (text: string) => screen.queryByText(text, { selector: "p.warning-banner" }) !== null;

describe("SingleScreen — §1-107 Reference body and notices", () => {
  it(
    "Enable on with one picture: /generate carries references [{kind, id, slot}]",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS, "a cat <Picture 2>", { transformerRole: "ref2va" });
      await waitForCard("ref-slot-picture-2");
      openReference();
      pickPicture(2);
      await waitForReady("ref-slot-picture-2");
      setEnabled(true);

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const body = postsTo("/api/v1/generate")[0]!;
      expect(body.references).toEqual([{ kind: "picture", id: expect.stringMatching(/^mock-image-/), slot: 2 }]);
      expect(body).not.toHaveProperty("source_audio");
    },
    20_000,
  );

  it(
    "a video truncated to the 15 s total carries max_sec; the shorter one before it does not",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS);
      await waitForCard("ref-slot-video-2");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceVideo", filePath: "C:\\v\\clip.mp4", fileName: "clip.mp4" });
        dispatchCreateCommand({
          type: "addReferenceVideo",
          filePath: "C:\\v\\clip.mp4",
          fileName: "long.mp4",
          durationSec: 20,
        });
      });
      await waitForReady("ref-slot-video-1");
      await waitForReady("ref-slot-video-2");
      setEnabled(true);

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const refs = postsTo("/api/v1/generate")[0]!.references as Array<Record<string, unknown>>;
      expect(refs).toHaveLength(2);
      expect(refs[0]).toEqual({ kind: "video", id: expect.any(String), slot: 1 });
      expect(refs[1]).toEqual({ kind: "video", id: expect.any(String), slot: 2, max_sec: 10 });
    },
    20_000,
  );

  it(
    "FL2VA loaded: Enable on sends no references, and ① + ③ / ① show under Generate",
    async () => {
      const { postsTo } = renderCreate(H3_CONSTRAINTS, "a cat", { transformerRole: "fl2va" });
      await waitForCard("ref-slot-picture-1");
      openReference();
      expect(noticeShown(NOTICE.fl2va)).toBe(false);
      setEnabled(true);
      // Enable on and nothing used: ① and ③ both (no rule suppresses the other).
      expect(noticeShown(NOTICE.fl2va)).toBe(true);
      expect(noticeShown(NOTICE.none)).toBe(true);

      pickPicture(1);
      await waitForReady("ref-slot-picture-1");
      await waitFor(() => expect(noticeShown(NOTICE.none)).toBe(false));
      expect(noticeShown(NOTICE.fl2va)).toBe(true);
      expect(noticeShown(NOTICE.ref2va)).toBe(false);

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      expect(postsTo("/api/v1/generate")[0]).not.toHaveProperty("references");
    },
    20_000,
  );

  it(
    "Ref2VA loaded: ② while Enable is off; Enable on with no card shows ③ only",
    async () => {
      renderCreate(H3_CONSTRAINTS, "a cat", { transformerRole: "ref2va" });
      await waitForCard("ref-slot-picture-1");
      openReference();
      expect(noticeShown(NOTICE.ref2va)).toBe(true);
      expect(noticeShown(NOTICE.none)).toBe(false);
      setEnabled(true);
      expect(noticeShown(NOTICE.ref2va)).toBe(false);
      expect(noticeShown(NOTICE.none)).toBe(true);
      expect(noticeShown(NOTICE.fl2va)).toBe(false);
    },
    15_000,
  );

  it(
    "FL2VA with Audio to video in effect: neither ① nor ③",
    async () => {
      renderCreate(H3_CONSTRAINTS, "a cat", { transformerRole: "fl2va" });
      await waitForCard("ref-slot-audio-1");
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg.wav" });
      });
      await waitForReady("ref-slot-audio-1");
      setEnabled(true);
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      await waitFor(() => expect(noticeShown(NOTICE.fl2va)).toBe(false));
      expect(noticeShown(NOTICE.none)).toBe(false);
    },
    15_000,
  );

  it(
    "LTX constraints: no notice, even with a role served",
    async () => {
      renderCreate(undefined, "a cat", { transformerRole: "ref2va" });
      await screen.findByText(/audio to video \(a2v\)/i, {}, { timeout: 5_000 });
      for (const text of Object.values(NOTICE)) expect(noticeShown(text)).toBe(false);
    },
    15_000,
  );

  it(
    "H3 with Enable on, then a switch to LTX: the kept cards are not sent",
    async () => {
      const { postsTo, switchTo } = renderCreate(H3_CONSTRAINTS, "a cat");
      await waitForCard("ref-slot-picture-1");
      openReference();
      pickPicture(1);
      await waitForReady("ref-slot-picture-1");
      setEnabled(true);

      switchTo(undefined);
      await waitFor(() => expect(screen.queryByTestId("ref-slot-picture-1")).not.toBeInTheDocument());
      await clickGenerateOnLtxGrid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      expect(postsTo("/api/v1/generate")[0]).not.toHaveProperty("references");
    },
    20_000,
  );

  it(
    "a card that would be sent and is still uploading blocks Generate with its reason; Enable off does not",
    async () => {
      const { nativeBridge } = renderCreate(H3_CONSTRAINTS, "a cat", { holdUploads: true });
      await waitForCard("ref-slot-picture-1");
      openReference();
      pickPicture(1);
      await waitFor(() => expect(slot("ref-slot-picture-1").querySelector(".keyframe-spinner")).not.toBeNull());
      // Enable off: the card is not sent, so it does not block.
      expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument();
      setEnabled(true);
      expect(await screen.findByText(CARD_UPLOADING)).toBeInTheDocument();

      act(() => nativeBridge.releaseUploads());
      await waitForReady("ref-slot-picture-1");
      expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "Audio to video ticked on an empty card: Generate is disabled with its reason; Enable off lifts it (§1-117)",
    async () => {
      renderCreate(H3_CONSTRAINTS, "a cat");
      await waitForCard("ref-slot-audio-1");
      openReference();
      setEnabled(true);
      await putOnH3Grid();
      expect(screen.queryByText(A2V_AUDIO_MISSING)).not.toBeInTheDocument();
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      expect(await screen.findByText(A2V_AUDIO_MISSING)).toBeInTheDocument();
      expect(generateButton()).toBeDisabled();
      // Every other card is grey, the empty ticked one is not.
      expect(slot("ref-slot-picture-1")).toHaveClass("ref-slot-ignored");
      expect(slot("ref-slot-audio-1")).not.toHaveClass("ref-slot-ignored");

      setEnabled(false);
      await waitFor(() => expect(screen.queryByText(A2V_AUDIO_MISSING)).not.toBeInTheDocument());
      expect(generateButton()).toBeEnabled();
      expect(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" })).toBeDisabled();
    },
    15_000,
  );

  it(
    "Audio to video ticked, then the audio attached: waits while uploading, sends source_audio once ready, 🔁 brings the reason back (§1-117)",
    async () => {
      const { nativeBridge, postsTo } = renderCreate(H3_CONSTRAINTS, "a cat", { holdUploads: true });
      await waitForCard("ref-slot-audio-1");
      openReference();
      setEnabled(true);
      fireEvent.click(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" }));
      act(() => {
        dispatchCreateCommand({ type: "addReferenceAudio", filePath: "C:\\a\\bg.wav", fileName: "bg.wav" });
      });
      await waitFor(() => expect(slot("ref-slot-audio-1").querySelector(".keyframe-spinner")).not.toBeNull());
      expect(await screen.findByText(CARD_UPLOADING)).toBeInTheDocument();
      expect(screen.queryByText(A2V_AUDIO_MISSING)).not.toBeInTheDocument();
      expect(generateButton()).toBeDisabled();

      act(() => nativeBridge.releaseUploads());
      await waitForReady("ref-slot-audio-1");
      await waitFor(() => expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument());
      expect(screen.queryByText(A2V_AUDIO_MISSING)).not.toBeInTheDocument();

      await clickGenerateOnH3Grid();
      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const body = postsTo("/api/v1/generate")[0]!;
      expect((body.source_audio as { audio_id: unknown }).audio_id).toMatch(/^mock-audio-/);
      expect(body).not.toHaveProperty("references");

      // The form is locked while the submit is in flight; 🔁 opens once it lands.
      const clearButton = within(slot("ref-slot-audio-1")).getByRole("button", { name: /clear this card/i });
      await waitFor(() => expect(clearButton).toBeEnabled());
      fireEvent.click(clearButton);
      expect(await screen.findByText(A2V_AUDIO_MISSING)).toBeInTheDocument();
      expect(within(slot("ref-slot-audio-1")).getByRole("checkbox", { name: "Audio to video" })).toBeChecked();
    },
    15_000,
  );

  it(
    "FL2VA: an uploading card does not block (it would not be sent)",
    async () => {
      renderCreate(H3_CONSTRAINTS, "a cat", { transformerRole: "fl2va", holdUploads: true });
      await waitForCard("ref-slot-picture-1");
      openReference();
      setEnabled(true);
      pickPicture(1);
      await waitFor(() => expect(slot("ref-slot-picture-1").querySelector(".keyframe-spinner")).not.toBeNull());
      expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument();
    },
    15_000,
  );
});
