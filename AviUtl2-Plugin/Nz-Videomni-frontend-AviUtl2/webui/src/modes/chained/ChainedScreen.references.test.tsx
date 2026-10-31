import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Profiler, type ReactNode } from "react";
import { apiClient } from "../../api/client";
import type { TransformerRole } from "../../api/types";
import { bridge as appBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import { getChainDirty, resetProvisionalReservation } from "../../timeline/provisionalReservation";
import { ChainedScreen } from "./ChainedScreen";
import type { ChainedScreenProps } from "./ChainedScreen";

// §1-109: MiniMax H3's per-clip Reference cards on the Chained screen —
// rendered directly with H3's served constraints (which carry `reference`).
//  - the References checkbox (H3 only, default off) shows one embedded
//    accordion per clip; off hides them and keeps the cards;
//  - only the clips with a used card get `clips[i].references`, never off H3
//    and never on FL2VA;
//  - a card that will be sent and is still uploading blocks Generate;
//  - the notices are Create's, the "no references" one only when every clip
//    has none; a filled card makes the screen dirty;
//  - removing a clip drops its cards; 24 clips settle in a bounded number of
//    commits.

const CHAIN_PATH = "/api/v1/generate/chain";

const H3_REFERENCE = resolveGenerationConstraints({
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: {
    num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
    shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
    shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
  },
  chain: { overlap_frames: 22 },
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

const H3_PROPS: Partial<ChainedScreenProps> = {
  constraints: H3_REFERENCE,
  v2vUnavailable: true,
  a2vUnavailable: true,
  endSourceUnavailable: true,
  referenceUnavailable: true,
  startImageUnavailable: true,
  batchI2vLongUnavailable: true,
  loraAudioHidden: true,
};

const NOTICE = {
  fl2va: en.single.reference.notices.fl2vaLoaded,
  ref2va: en.single.reference.notices.ref2vaLoaded,
  none: en.single.reference.notices.noReferences,
};
const CARD_UPLOADING = en.single.generateReasons.referenceCardUploading;

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

let commits = 0;

function renderChain(
  extra: Partial<ChainedScreenProps> = H3_PROPS,
  options: { holdUploads?: boolean; transformerRole?: TransformerRole | null } = {},
) {
  const nativeBridge = createMockBridge({ delayMs: 0, holdUploads: options.holdUploads ?? false });
  // Submissions go through the app-wide singleton client, so spy on that.
  const requestSpy = vi.spyOn(appBridge, "request");
  const tree = (props: Partial<ChainedScreenProps>, role: TransformerRole | null) => (
    <Providers nativeBridge={nativeBridge}>
      <Profiler id="chained" onRender={() => (commits += 1)}>
        <ChainedScreen
          prompt="a cat riding a skateboard"
          baseUrl={null}
          nativeBridge={nativeBridge}
          highlightedJobId={null}
          onJobSubmitted={() => {}}
          controlLoraNames={new Set()}
          transformerRole={role}
          {...props}
        />
      </Profiler>
    </Providers>
  );
  const view = render(tree(extra, options.transformerRole ?? null));
  const chainBodies = () =>
    requestSpy.mock.calls
      .filter((c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === CHAIN_PATH)
      .map((c) => (c[1] as { body: Record<string, unknown> }).body);
  /** Another base model's props / role on the same screen instance (a switch). */
  const switchTo = (props: Partial<ChainedScreenProps>, role: TransformerRole | null = null) =>
    view.rerender(tree(props, role));
  return { ...view, nativeBridge, chainBodies, switchTo };
}

const clipCards = () => Array.from(document.querySelectorAll<HTMLElement>(".clip-card"));
const clipCard = (index: number) => {
  const card = clipCards()[index];
  if (!card) throw new Error(`no clip card ${index}`);
  return card;
};
const referencesBox = () => screen.queryByRole("checkbox", { name: en.chained.referencesToggleLabel });

async function waitForClips() {
  await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
}

function setReferences(on: boolean) {
  const box = referencesBox() as HTMLInputElement;
  if (box.checked !== on) fireEvent.click(box);
}

function openClipReference(card: HTMLElement) {
  const summary = within(card).getByText(en.single.reference.heading, { selector: "summary" });
  const details = summary.closest("details") as HTMLDetailsElement;
  if (!details.open) fireEvent.click(summary);
}

function pickPicture(card: HTMLElement, slotNumber: number) {
  openClipReference(card);
  const slot = within(card).getByTestId(`ref-slot-picture-${slotNumber}`);
  fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.chooseButton }));
}

async function waitForReady(cardIndex: number, slotNumber: number) {
  await waitFor(
    () => {
      const slot = within(clipCard(cardIndex)).getByTestId(`ref-slot-picture-${slotNumber}`);
      expect(within(slot).getByRole("button", { name: en.single.reference.tagButton })).toBeEnabled();
      expect(slot.querySelector(".keyframe-spinner")).toBeNull();
    },
    { timeout: 5_000 },
  );
}

const generateButton = () => screen.getByRole("button", { name: /^generate$/i });

async function clickGenerate() {
  await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
  fireEvent.click(generateButton());
}

/** On LTX the clips sit on the 8n+1 grid; re-applied until Generate opens. */
async function clickGenerateOnLtxGrid() {
  await waitFor(
    () => {
      for (const input of document.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']")) {
        fireEvent.change(input, { target: { value: "121" } });
        fireEvent.blur(input);
      }
      expect(generateButton()).toBeEnabled();
    },
    { timeout: 5_000 },
  );
  fireEvent.click(generateButton());
}

const noticeShown = (text: string) => screen.queryByText(text, { selector: "p.warning-banner" }) !== null;

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

describe("ChainedScreen — §1-109 per-clip References", () => {
  it(
    "LTX: no References checkbox and no clip accordion",
    async () => {
      renderChain({});
      await waitForClips();
      expect(referencesBox()).toBeNull();
      expect(document.querySelector(".ref-accordion")).toBeNull();
    },
    15_000,
  );

  it(
    "H3: off by default; on shows one embedded accordion per clip (no Enable, no ON badge, no Audio to video)",
    async () => {
      renderChain();
      await waitForClips();
      expect(referencesBox()).not.toBeChecked();
      expect(document.querySelector(".ref-accordion")).toBeNull();

      setReferences(true);
      await waitFor(() => expect(document.querySelectorAll(".ref-accordion")).toHaveLength(2), { timeout: 5_000 });
      for (const card of clipCards()) {
        expect(within(card).getByTestId("ref-row-picture")).toBeInTheDocument();
        expect(within(card).queryByRole("checkbox", { name: en.single.reference.enableLabel })).toBeNull();
        expect(within(card).queryByText(en.single.reference.onBadge)).toBeNull();
        expect(within(card).queryByText(en.single.reference.audioToVideoLabel)).toBeNull();
      }
    },
    15_000,
  );

  it(
    "only the second clip holds a picture: clips[1].references alone, and ✒ appends to that clip's own prompt",
    async () => {
      const { chainBodies } = renderChain();
      await waitForClips();
      setReferences(true);
      pickPicture(clipCard(1), 2);
      await waitForReady(1, 2);

      const slot = within(clipCard(1)).getByTestId("ref-slot-picture-2");
      fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.tagButton }));
      expect((within(clipCard(1)).getByRole("textbox") as HTMLTextAreaElement).value).toBe("<Picture 2>");
      expect((within(clipCard(0)).getByRole("textbox") as HTMLTextAreaElement).value).toBe("");

      await clickGenerate();
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      const clips = chainBodies()[0]!.clips as Record<string, unknown>[];
      expect(clips).toHaveLength(2);
      expect(clips[0]).not.toHaveProperty("references");
      expect(clips[1]!.references).toEqual([{ kind: "picture", id: expect.stringMatching(/^mock-image-/), slot: 2 }]);
    },
    20_000,
  );

  it(
    "off: nothing is sent and the accordions hide, but the cards are kept for the next on",
    async () => {
      const { chainBodies } = renderChain();
      await waitForClips();
      setReferences(true);
      pickPicture(clipCard(0), 1);
      await waitForReady(0, 1);

      setReferences(false);
      expect(document.querySelector(".ref-accordion")).toBeNull();
      await clickGenerate();
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      for (const clip of chainBodies()[0]!.clips as Record<string, unknown>[]) {
        expect(clip).not.toHaveProperty("references");
      }

      setReferences(true);
      await waitForReady(0, 1);
    },
    20_000,
  );

  it(
    "H3 with a card, then LTX: no references in the body and no checkbox; back on H3 the card is still there",
    async () => {
      const { chainBodies, switchTo } = renderChain();
      await waitForClips();
      setReferences(true);
      pickPicture(clipCard(0), 1);
      await waitForReady(0, 1);

      switchTo({});
      await waitFor(() => expect(referencesBox()).toBeNull(), { timeout: 5_000 });
      expect(document.querySelector(".ref-accordion")).toBeNull();
      await clickGenerateOnLtxGrid();
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      for (const clip of chainBodies()[0]!.clips as Record<string, unknown>[]) {
        expect(clip).not.toHaveProperty("references");
      }

      switchTo(H3_PROPS);
      await waitFor(() => expect(referencesBox()).toBeChecked(), { timeout: 5_000 });
      await waitForReady(0, 1);
    },
    20_000,
  );

  it(
    "FL2VA: on shows ① and sends no references",
    async () => {
      const { chainBodies } = renderChain(H3_PROPS, { transformerRole: "fl2va" });
      await waitForClips();
      expect(noticeShown(NOTICE.fl2va)).toBe(false);
      setReferences(true);
      expect(noticeShown(NOTICE.fl2va)).toBe(true);
      pickPicture(clipCard(0), 1);
      await waitForReady(0, 1);

      await clickGenerate();
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      for (const clip of chainBodies()[0]!.clips as Record<string, unknown>[]) {
        expect(clip).not.toHaveProperty("references");
      }
    },
    20_000,
  );

  it(
    "Ref2VA: ② while off; on, ③ shows only while EVERY clip has no card (one clip with a card is enough)",
    async () => {
      renderChain(H3_PROPS, { transformerRole: "ref2va" });
      await waitForClips();
      expect(noticeShown(NOTICE.ref2va)).toBe(true);
      expect(noticeShown(NOTICE.none)).toBe(false);

      setReferences(true);
      expect(noticeShown(NOTICE.ref2va)).toBe(false);
      expect(noticeShown(NOTICE.none)).toBe(true);

      // Clip 2 only — clip 1 stays empty, and ③ goes away.
      pickPicture(clipCard(1), 1);
      await waitForReady(1, 1);
      await waitFor(() => expect(noticeShown(NOTICE.none)).toBe(false), { timeout: 5_000 });
      expect(noticeShown(NOTICE.fl2va)).toBe(false);

      // Cleared again: every clip is empty, ③ is back.
      const slot = within(clipCard(1)).getByTestId("ref-slot-picture-1");
      fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.clearButton }));
      await waitFor(() => expect(noticeShown(NOTICE.none)).toBe(true), { timeout: 5_000 });
    },
    20_000,
  );

  it(
    "a card that will be sent and is still uploading blocks Generate with its reason; off does not",
    async () => {
      const { nativeBridge } = renderChain(H3_PROPS, { holdUploads: true });
      await waitForClips();
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
      setReferences(true);
      pickPicture(clipCard(1), 1);
      await waitFor(
        () =>
          expect(within(clipCard(1)).getByTestId("ref-slot-picture-1").querySelector(".keyframe-spinner")).not.toBeNull(),
        { timeout: 5_000 },
      );
      expect(await screen.findByText(CARD_UPLOADING, {}, { timeout: 5_000 })).toBeInTheDocument();
      expect(generateButton()).toBeDisabled();

      // Off: the card would not be sent, so it does not block.
      setReferences(false);
      await waitFor(() => expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument(), { timeout: 5_000 });
      expect(generateButton()).toBeEnabled();

      setReferences(true);
      expect(await screen.findByText(CARD_UPLOADING, {}, { timeout: 5_000 })).toBeInTheDocument();
      act(() => nativeBridge.releaseUploads());
      await waitForReady(1, 1);
      expect(screen.queryByText(CARD_UPLOADING)).not.toBeInTheDocument();
      expect(generateButton()).toBeEnabled();
    },
    20_000,
  );

  it(
    "a filled card makes the screen dirty (the right-click remount asks first)",
    async () => {
      renderChain();
      await waitForClips();
      await waitFor(() => expect(getChainDirty()).toBe(false), { timeout: 5_000 });
      setReferences(true);
      expect(getChainDirty()).toBe(false);
      pickPicture(clipCard(0), 1);
      await waitFor(() => expect(getChainDirty()).toBe(true), { timeout: 5_000 });
    },
    15_000,
  );

  it(
    "removing a clip drops its cards: ③ is back and nothing is sent",
    async () => {
      const { chainBodies } = renderChain();
      await waitForClips();
      fireEvent.click(screen.getByRole("button", { name: en.chained.addClipButton }));
      await waitFor(() => expect(clipCards()).toHaveLength(3), { timeout: 5_000 });
      setReferences(true);
      pickPicture(clipCard(2), 1);
      await waitForReady(2, 1);
      await waitFor(() => expect(noticeShown(NOTICE.none)).toBe(false), { timeout: 5_000 });

      fireEvent.click(within(clipCard(2)).getByRole("button", { name: en.chained.removeClipButton }));
      await waitFor(() => expect(clipCards()).toHaveLength(2), { timeout: 5_000 });
      await waitFor(() => expect(noticeShown(NOTICE.none)).toBe(true), { timeout: 5_000 });

      await clickGenerate();
      await waitFor(() => expect(chainBodies()).toHaveLength(1), { timeout: 5_000 });
      const clips = chainBodies()[0]!.clips as Record<string, unknown>[];
      expect(clips).toHaveLength(2);
      for (const clip of clips) expect(clip).not.toHaveProperty("references");
    },
    20_000,
  );

  it(
    "24 clips: switching to H3 and turning References on settle in a bounded number of commits",
    async () => {
      const { switchTo } = renderChain({});
      await waitForClips();
      const add = screen.getByRole("button", { name: en.chained.addClipButton });
      while (clipCards().length < 24) fireEvent.click(add);
      expect(clipCards()).toHaveLength(24);

      // H3 for the first time: 24 `ClipReferences` mount, enable and register.
      commits = 0;
      switchTo(H3_PROPS);
      await waitFor(() => expect(referencesBox()).not.toBeNull(), { timeout: 5_000 });
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
      });
      expect(commits).toBeLessThanOrEqual(10);

      commits = 0;
      setReferences(true);
      await waitFor(() => expect(document.querySelectorAll(".ref-accordion")).toHaveLength(24), { timeout: 5_000 });
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
      });
      expect(commits).toBeLessThanOrEqual(5);
      expect(noticeShown(NOTICE.none)).toBe(true);
    },
    30_000,
  );
});
