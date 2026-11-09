import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Profiler, type ReactNode } from "react";
import { apiClient } from "../../api/client";
import { bridge as appBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider, useToasts } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import { resetProvisionalReservation } from "../../timeline/provisionalReservation";
import { ChainedScreen } from "./ChainedScreen";
import type { ChainedScreenProps } from "./ChainedScreen";

// §1-120: MiniMax H3's long A2V on Chained — the long audio card's
// "distribute to each clip" mode, opened by `constraints.chain.source_audio`.
// Rendered directly with H3's served constraints, end to end through the real
// mock bridge (upload, probe) and the real per-clip Reference cards.

const CHAIN_PATH = "/api/v1/generate/chain";

const RAW_H3 = {
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  reference: {
    max_images: 9,
    max_videos: 3,
    max_audios: 3,
    max_total: 12,
    video_total_sec: 15,
    audio_total_sec: 15,
    clip_min_sec: 2,
  },
};

const H3_A2V = resolveGenerationConstraints({
  ...RAW_H3,
  chain: { overlap_frames: 22, source_audio: { strength: { min: 0, max: 1, step: 0.05, default: 0.5 } } },
});

const H3_PROPS: Partial<ChainedScreenProps> = {
  constraints: H3_A2V,
  endSourceUnavailable: true,
  referenceUnavailable: true,
  endSourceHidden: true,
  referenceHidden: true,
  startImageStrengthHidden: true,
  batchI2vLongUnavailable: true,
  loraAudioHidden: true,
};

const t = en.chained.sourceAudio;
const reasons = en.chained.generateReasons;

/** Stands in for `shell/Toasts.tsx` so the messages are assertable. */
function ToastSpy() {
  const { toasts } = useToasts();
  return (
    <ul data-testid="toasts">
      {toasts.map((toast) => (
        <li key={toast.id} data-kind={toast.kind}>
          {toast.message}
        </li>
      ))}
    </ul>
  );
}

function toastMessages(): string[] {
  return Array.from(screen.getByTestId("toasts").children).map((li) => li.textContent ?? "");
}

function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge}>
              {children}
              <ToastSpy />
            </JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

let commits = 0;

function renderChain(extra: Partial<ChainedScreenProps> = H3_PROPS, options: MockBridgeOptions = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0, probeMediaInfoDurationSec: 30, ...options });
  const nativeSpy = vi.spyOn(nativeBridge, "request");
  const requestSpy = vi.spyOn(appBridge, "request");
  const tree = (props: Partial<ChainedScreenProps>) => (
    <Providers nativeBridge={nativeBridge}>
      <Profiler id="chained" onRender={() => (commits += 1)}>
        <ChainedScreen
          prompt="a cat riding a skateboard"
          baseUrl={null}
          nativeBridge={nativeBridge}
          highlightedJobId={null}
          onJobSubmitted={() => {}}
          controlLoraNames={new Set()}
          {...props}
        />
      </Profiler>
    </Providers>
  );
  const view = render(tree(extra));
  const chainBodies = () =>
    requestSpy.mock.calls
      .filter((c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === CHAIN_PATH)
      .map((c) => (c[1] as { body: Record<string, unknown> }).body);
  const uploadCount = () => nativeSpy.mock.calls.filter((c) => c[0] === "backend.uploadFile").length;
  const switchTo = (props: Partial<ChainedScreenProps>) => view.rerender(tree(props));
  return { ...view, nativeBridge, chainBodies, uploadCount, switchTo };
}

const clipCards = () => Array.from(document.querySelectorAll<HTMLElement>(".clip-card"));
const clipCard = (index: number) => {
  const card = clipCards()[index];
  if (!card) throw new Error(`no clip card ${index}`);
  return card;
};
const frameInputs = () => Array.from(document.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']"));
const clipFrames = () => frameInputs().map((input) => Number(input.value));
const referencesBox = () => screen.getByRole("checkbox", { name: en.chained.referencesToggleLabel }) as HTMLInputElement;
const generateButton = () => screen.getByRole("button", { name: /^generate$/i });
const distributeButton = () => screen.getByRole("button", { name: t.h3.distributeButton });
/** The "why is Generate disabled?" note under the button. */
const reasonsNote = () => document.querySelector<HTMLElement>(".generate-reasons-note");
const reasonShown = (text: string) => {
  const note = reasonsNote();
  return note !== null && within(note).queryByText(text) !== null;
};

async function waitForClips() {
  await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
}

function setFrames(index: number, value: number) {
  const input = frameInputs()[index]!;
  fireEvent.change(input, { target: { value: String(value) } });
  fireEvent.blur(input);
}

function dropAudio(container: HTMLElement, fileName = "track.mp3") {
  fireEvent.drop(container.querySelector(".chain-audio-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function openClipReference(card: HTMLElement) {
  const summary = within(card).getByText(en.single.reference.heading, { selector: "summary" });
  const details = summary.closest("details") as HTMLDetailsElement;
  if (!details.open) fireEvent.click(summary);
}

const a2vBox = (card: HTMLElement, slot: number) =>
  within(within(card).getByTestId(`ref-slot-audio-${slot}`)).getByRole("checkbox", {
    name: en.single.reference.audioToVideoLabel,
  }) as HTMLInputElement;

/** S_i of each clip, from the lengths actually sent. */
function startsOf(clips: Record<string, unknown>[]): number[] {
  const starts: number[] = [];
  let s = 0;
  clips.forEach((clip, i) => {
    starts.push(s);
    if (i < clips.length - 1) s += (clip.num_frames as number) - 22;
  });
  return starts;
}

async function waitDistributed(count: number) {
  await waitFor(
    () => {
      expect(clipCards()).toHaveLength(count);
      for (const card of clipCards()) expect(a2vBox(card, 1).checked).toBe(true);
    },
    { timeout: 5_000 },
  );
}

async function drainJobs(): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    const jobs = await apiClient.listJobs();
    if (jobs.every((j) => j.status !== "queued" && j.status !== "running")) return;
  }
}

beforeEach(() => {
  commits = 0;
  resetProvisionalReservation();
});

afterEach(async () => {
  vi.restoreAllMocks();
  await drainJobs();
  resetProvisionalReservation();
});

describe("ChainedScreen — §1-120 H3 long A2V (distribute to each clip)", () => {
  it(
    "the A2V card opens with a Strength slider and the distribute button (no ↔️, no added-clip slider)",
    async () => {
      renderChain();
      await waitForClips();
      expect(screen.queryByText(en.chained.unavailableOnBaseModel.sourceAudio)).not.toBeInTheDocument();
      expect(screen.getByTestId("chain-audio-strength")).toHaveValue("0.5");
      expect(distributeButton()).toBeDisabled();
      expect(screen.queryByRole("button", { name: t.adjustButton })).toBeNull();
      expect(screen.queryByText(t.addedClipFramesLabel)).toBeNull();
    },
    15_000,
  );

  it(
    "a 30 s audio is distributed once on attach: clips added, References on, the same id in every clip, no extra upload",
    async () => {
      const { container, chainBodies, uploadCount } = renderChain();
      await waitForClips();
      setFrames(0, 226);
      setFrames(1, 226);
      await waitFor(() => expect(clipFrames()).toEqual([226, 226]));

      dropAudio(container);
      await waitDistributed(4);
      expect(clipFrames()).toEqual([226, 226, 226, 107]);
      expect(referencesBox().checked).toBe(true);
      // Only the long audio itself was uploaded.
      expect(uploadCount()).toBe(1);

      // Each clip's card: checked, Strength 0.50, its window badge.
      clipCards().forEach((card, i) => {
        openClipReference(card);
        const slot = within(card).getByTestId("ref-slot-audio-1");
        expect(within(slot).getByTestId("source-audio-strength-value")).toHaveTextContent("0.50");
        const start = [0, 204, 408, 612][i]!;
        const end = start + [226, 226, 226, 107][i]!;
        expect(within(slot).getByTestId("ref-a2v-window")).toHaveTextContent(
          en.chained.clipReferenceA2vWindow.label((start / 24).toFixed(2), (end / 24).toFixed(2)),
        );
        expect(within(card).getByText(en.single.reference.onBadge)).toBeInTheDocument();
      });
      // No 🎵 on the clip cards themselves.
      expect(container.querySelectorAll(".clip-card-audio-window")).toHaveLength(0);

      // Editing a length afterwards does not re-run the distribution.
      setFrames(3, 90);
      await waitFor(() => expect(clipFrames()).toEqual([226, 226, 226, 90]));

      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      fireEvent.click(generateButton());
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      const body = chainBodies()[0]!;
      expect(body).not.toHaveProperty("source_audio");
      const clips = body.clips as Record<string, unknown>[];
      const starts = startsOf(clips);
      const ids = new Set<string>();
      clips.forEach((clip, i) => {
        const sa = clip.source_audio as { audio_id: string; strength: number; offset_frames: number };
        expect(sa).toEqual({ audio_id: expect.stringMatching(/^mock-audio-/), strength: 0.5, offset_frames: starts[i] });
        ids.add(sa.audio_id);
        expect(clip).not.toHaveProperty("references");
        // `source_audio` comes after `references` / `num_frames`.
        expect(Object.keys(clip).at(-1)).toBe("source_audio");
      });
      expect(ids.size).toBe(1);
      expect(uploadCount()).toBe(1);
    },
    30_000,
  );

  it(
    "a clip's other cards are not sent while its Audio to video is declared (decision 1)",
    async () => {
      const { container, chainBodies } = renderChain();
      await waitForClips();
      fireEvent.click(referencesBox());
      openClipReference(clipCard(0));
      const picture = within(clipCard(0)).getByTestId("ref-slot-picture-1");
      fireEvent.click(within(picture).getByRole("button", { name: en.single.reference.chooseButton }));
      await waitFor(
        () => expect(within(picture).getByRole("button", { name: en.single.reference.tagButton })).toBeEnabled(),
        { timeout: 5_000 },
      );

      dropAudio(container);
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      fireEvent.click(generateButton());
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      const clip0 = (chainBodies()[0]!.clips as Record<string, unknown>[])[0]!;
      expect(clip0).not.toHaveProperty("references");
      expect(clip0).toHaveProperty("source_audio");
    },
    30_000,
  );

  it(
    "Generate stops: a clip past the audio's end, References off, an audio not distributed",
    async () => {
      const { container } = renderChain();
      await waitForClips();
      dropAudio(container);
      await waitFor(() => expect(referencesBox().checked).toBe(true), { timeout: 5_000 });
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });

      // Lengthen the last clip past the 30 s audio.
      const last = clipFrames().length - 1;
      setFrames(last, 481);
      await waitFor(() => expect(reasonShown(reasons.chainA2vAudioTooShort)).toBe(true));
      expect(generateButton()).toBeDisabled();
      // The badge's warning on that clip's card says the same.
      openClipReference(clipCard(last));
      expect(within(clipCard(last)).getByText(t.h3.longTooShortWarning)).toBeInTheDocument();

      // References off: only the "turn References on" line.
      fireEvent.click(referencesBox());
      await waitFor(() => expect(reasonShown(reasons.chainA2vReferencesOff)).toBe(true));
      expect(reasonShown(reasons.chainA2vAudioTooShort)).toBe(false);
      expect(generateButton()).toBeDisabled();
      fireEvent.click(referencesBox());

      // Unchecking every card: the long audio is still held (not "not
      // distributed"), and nothing is declared.
      for (const card of clipCards()) {
        openClipReference(card);
        fireEvent.click(a2vBox(card, 1));
      }
      await waitFor(() => expect(reasonShown(reasons.chainA2vAudioTooShort)).toBe(false));
    },
    30_000,
  );

  it(
    "🔁 on the long audio empties the distributed cards and unchecks them",
    async () => {
      const { container, chainBodies } = renderChain();
      await waitForClips();
      dropAudio(container);
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });

      fireEvent.click(screen.getByRole("button", { name: t.clearButton }));
      await waitFor(() => {
        for (const card of clipCards()) {
          expect(a2vBox(card, 1).checked).toBe(false);
          expect(within(card).getByTestId("ref-slot-audio-1")).toHaveTextContent(en.single.reference.emptySlot);
        }
      });
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      fireEvent.click(generateButton());
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      for (const clip of chainBodies()[0]!.clips as Record<string, unknown>[]) {
        expect(clip).not.toHaveProperty("source_audio");
      }
    },
    30_000,
  );

  it(
    "🔁 on the LTX screen empties them too (back on H3 nothing is left as a hand-put audio)",
    async () => {
      const { container, switchTo } = renderChain();
      await waitForClips();
      dropAudio(container);
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });

      switchTo({});
      await waitFor(() => expect(screen.queryByRole("checkbox", { name: en.chained.referencesToggleLabel })).toBeNull());
      fireEvent.click(screen.getByRole("button", { name: t.clearButton }));

      switchTo(H3_PROPS);
      await waitFor(() => expect(referencesBox().checked).toBe(true));
      for (const card of clipCards()) {
        openClipReference(card);
        expect(a2vBox(card, 1).checked).toBe(false);
        expect(within(card).getByTestId("ref-slot-audio-1")).toHaveTextContent(en.single.reference.emptySlot);
      }
    },
    30_000,
  );

  it(
    "an audio attached on LTX is not distributed by a switch to H3 (the button still works)",
    async () => {
      const { container, switchTo } = renderChain({});
      await waitForClips();
      dropAudio(container);
      // The LTX auto-fit has run once the length is known (↔️ enabled).
      await waitFor(() => expect(screen.getByRole("button", { name: t.adjustButton })).toBeEnabled(), {
        timeout: 5_000,
      });
      await waitFor(() => expect(clipFrames().length).toBeGreaterThan(2), { timeout: 5_000 });

      switchTo(H3_PROPS);
      await waitFor(() => expect(screen.getByRole("button", { name: t.h3.distributeButton })).toBeEnabled());
      // (The lengths re-snap onto H3's grid on the switch — existing behaviour.)
      const h3Frames = clipFrames();
      expect(await screen.findByText(reasons.chainA2vNotDistributed)).toBeInTheDocument();
      expect(referencesBox().checked).toBe(false);
      expect(clipFrames()).toEqual(h3Frames);
      for (const card of clipCards()) {
        fireEvent.click(referencesBox());
        openClipReference(card);
        expect(a2vBox(card, 1).checked).toBe(false);
        fireEvent.click(referencesBox());
      }
      expect(generateButton()).toBeDisabled();

      fireEvent.click(distributeButton());
      await waitFor(() => expect(referencesBox().checked).toBe(true), { timeout: 5_000 });
      await waitFor(() => expect(screen.queryByText(reasons.chainA2vNotDistributed)).toBeNull(), { timeout: 5_000 });
    },
    30_000,
  );

  it(
    "shortening a clip the user edited asks first; No leaves everything as it was",
    async () => {
      const { container } = renderChain(H3_PROPS, { probeMediaInfoDurationSec: 10 });
      await waitForClips();
      setFrames(0, 226);
      setFrames(1, 226);
      await waitFor(() => expect(clipFrames()).toEqual([226, 226]));

      dropAudio(container);
      const body = t.confirm.body(
        [
          { clip: 1, from: 226, to: 209 },
          { clip: 2, from: 226, to: 39 },
        ],
        [],
      );
      expect(await screen.findByText(body, {}, { timeout: 5_000 })).toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: t.confirm.cancelLabel }));
      await waitFor(() => expect(screen.queryByText(body)).toBeNull());
      expect(clipFrames()).toEqual([226, 226]);
      expect(referencesBox().checked).toBe(false);
      expect(await screen.findByText(reasons.chainA2vNotDistributed)).toBeInTheDocument();

      fireEvent.click(distributeButton());
      fireEvent.click(await screen.findByRole("button", { name: t.confirm.confirmLabel }));
      await waitDistributed(2);
      expect(clipFrames()).toEqual([209, 39]);
    },
    30_000,
  );

  it(
    "a hand-put audio with Audio to video goes out with offset_frames 0",
    async () => {
      const { chainBodies } = renderChain();
      await waitForClips();
      fireEvent.click(referencesBox());
      const card = clipCard(1);
      openClipReference(card);
      const slot = within(card).getByTestId("ref-slot-audio-2");
      fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.chooseButton }));
      await waitFor(
        () => expect(within(slot).getByRole("button", { name: en.single.reference.tagButton })).toBeEnabled(),
        { timeout: 5_000 },
      );
      fireEvent.click(a2vBox(card, 2));
      // No window badge on a hand-put audio.
      expect(within(slot).queryByTestId("ref-a2v-window")).toBeNull();

      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      fireEvent.click(generateButton());
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      const clips = chainBodies()[0]!.clips as Record<string, unknown>[];
      expect(clips[0]).not.toHaveProperty("source_audio");
      expect(clips[1]!.source_audio).toEqual({
        audio_id: expect.stringMatching(/^mock-audio-/),
        strength: 0.5,
        offset_frames: 0,
      });
    },
    30_000,
  );

  it(
    "a source video with a declared A2V stops Generate with the V2V line",
    async () => {
      const { container } = renderChain(H3_PROPS, { pickFileName: "clip.mp4" });
      await waitForClips();
      dropAudio(container);
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });

      const source = container.querySelector(".source-section") as HTMLElement;
      fireEvent.click(within(source).getByRole("button", { name: en.chained.sourceInput.chooseButton }));
      expect(await screen.findByText(reasons.audioConflictsWithSourceVideo, {}, { timeout: 5_000 })).toBeInTheDocument();
      expect(screen.getAllByText(reasons.audioConflictsWithSourceVideo)).toHaveLength(1);
      expect(generateButton()).toBeDisabled();
      expect(distributeButton()).toBeDisabled();
    },
    30_000,
  );

  it(
    "the distribution settles in a bounded number of commits (no registration loop)",
    async () => {
      const { container } = renderChain();
      await waitForClips();
      setFrames(0, 226);
      setFrames(1, 226);
      await waitFor(() => expect(clipFrames()).toEqual([226, 226]));
      commits = 0;
      dropAudio(container);
      await waitDistributed(4);
      // The whole distribution (upload, probe, re-layout, injection) is a
      // bounded burst, and nothing keeps committing afterwards.
      expect(commits).toBeLessThanOrEqual(60);
      commits = 0;
      await new Promise((resolve) => setTimeout(resolve, 200));
      expect(commits).toBeLessThanOrEqual(5);
    },
    30_000,
  );

  it(
    "a backend without `constraints.chain.source_audio`: the panel stays closed and no clip shows Audio to video",
    async () => {
      const oldH3 = resolveGenerationConstraints({ ...RAW_H3, chain: { overlap_frames: 22 } });
      renderChain({ ...H3_PROPS, constraints: oldH3, a2vUnavailable: true });
      await waitForClips();
      expect(screen.getByText(en.chained.unavailableOnBaseModel.sourceAudio)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: t.h3.distributeButton })).toBeNull();
      fireEvent.click(referencesBox());
      openClipReference(clipCard(0));
      expect(within(clipCard(0)).queryByText(en.single.reference.audioToVideoLabel)).toBeNull();
    },
    15_000,
  );

  it(
    "📁 with another audio: the old id's cards are emptied and the new audio is distributed once more",
    async () => {
      const { container } = renderChain();
      await waitForClips();
      dropAudio(container, "first.mp3");
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });
      openClipReference(clipCard(0));
      expect(within(clipCard(0)).getByTestId("ref-slot-audio-1")).toHaveTextContent("first.mp3");

      fireEvent.click(screen.getByRole("button", { name: t.changeButton }));
      await waitFor(
        () => {
          for (const card of clipCards()) {
            openClipReference(card);
            const slot = within(card).getByTestId("ref-slot-audio-1");
            expect(slot).toHaveTextContent(/picked-audio-\d+\.wav/);
            expect(a2vBox(card, 1).checked).toBe(true);
          }
        },
        { timeout: 5_000 },
      );
      expect(screen.queryByText("first.mp3", { selector: ".ref-slot-name" })).toBeNull();
      expect(toastMessages().filter((m) => m === t.h3.toast.distributed(30))).toHaveLength(2);
    },
    30_000,
  );

  it(
    "a Strength set on the long-audio card before distributing is copied into every clip and sent",
    async () => {
      const { container, chainBodies } = renderChain();
      await waitForClips();
      fireEvent.change(screen.getByTestId("chain-audio-strength"), { target: { value: "0.8" } });
      dropAudio(container);
      await waitFor(() => expect(a2vBox(clipCard(0), 1).checked).toBe(true), { timeout: 5_000 });
      for (const card of clipCards()) {
        openClipReference(card);
        expect(within(within(card).getByTestId("ref-slot-audio-1")).getByTestId("source-audio-strength-value")).toHaveTextContent(
          "0.80",
        );
      }
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      fireEvent.click(generateButton());
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      for (const clip of chainBodies()[0]!.clips as Record<string, unknown>[]) {
        expect((clip.source_audio as { strength: number }).strength).toBe(0.8);
      }
    },
    30_000,
  );

  it(
    "untouched cards are shortened without a confirmation (the #18 path)",
    async () => {
      const { container } = renderChain(H3_PROPS, { probeMediaInfoDurationSec: 10 });
      await waitForClips();
      // The cards keep their defaults (intact); 10 s = 240 frames.
      dropAudio(container);
      await waitDistributed(2);
      expect(clipFrames()).toEqual([209, 39]);
      expect(screen.queryByRole("button", { name: t.confirm.confirmLabel })).toBeNull();
      expect(screen.queryByRole("dialog")).toBeNull();
    },
    30_000,
  );

  it(
    "toasts: an audio too short for a chain, and a clip with no free audio card",
    async () => {
      const short = renderChain(H3_PROPS, { probeMediaInfoDurationSec: 1.5 });
      await waitForClips();
      dropAudio(short.container);
      await waitFor(() => expect(toastMessages()).toContain(t.h3.toast.tooShortForChain), { timeout: 5_000 });
      expect(referencesBox().checked).toBe(false);
      short.unmount();

      const { container } = renderChain();
      await waitForClips();
      fireEvent.click(referencesBox());
      const card = clipCard(0);
      openClipReference(card);
      for (const n of [1, 2, 3]) {
        const slot = within(card).getByTestId(`ref-slot-audio-${n}`);
        fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.chooseButton }));
        await waitFor(
          () => expect(within(slot).getByRole("button", { name: en.single.reference.tagButton })).toBeEnabled(),
          { timeout: 5_000 },
        );
      }
      dropAudio(container);
      await waitFor(() => expect(toastMessages()).toContain(t.h3.toast.noFreeSlot([1])), { timeout: 5_000 });
      // Clip 1 is left as it was; clip 2 got the long audio.
      expect(a2vBox(clipCard(0), 1).checked).toBe(false);
      await waitFor(() => expect(a2vBox(clipCard(1), 1).checked).toBe(true));
    },
    30_000,
  );
});
