import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { apiClient } from "../../api/client";
import type { TransformerRole } from "../../api/types";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { ChainedScreen } from "./ChainedScreen";
import type { ChainedScreenProps } from "./ChainedScreen";

// Ledger §1-276: MiniMax H3's per-clip comfort marker on the Chained screen —
// each clip's duration slider carries a tick at the sequence-rows marker
// computed for THAT clip (Single's formula at the clip's length): the shared
// size, clip 0's opening image as its one keyframe (none on a V2V chain), and
// the clip's own Reference cards while References is on. The mock config
// serves `comfort_budgets.h3.single_rows_budget = 219516`. LTX draws none.

const H3 = resolveGenerationConstraints({
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

/** What `AppShell` derives from MiniMax H3's list (台帳 §1-121), plus the
 * engine family that picks the served sequence-rows line. */
const H3_PROPS: Partial<ChainedScreenProps> = {
  constraints: H3,
  engineFamily: "h3",
  a2vUnavailable: true,
  endSourceUnavailable: true,
  referenceUnavailable: true,
  endSourceHidden: true,
  referenceHidden: true,
  startImageStrengthHidden: true,
  loraAudioHidden: true,
};

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

function chainTree(
  nativeBridge: ReturnType<typeof createMockBridge>,
  extra: Partial<ChainedScreenProps>,
  transformerRole: TransformerRole | null,
) {
  return (
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
        transformerRole={transformerRole}
        {...extra}
      />
    </Providers>
  );
}

function renderChain(
  extra: Partial<ChainedScreenProps> = H3_PROPS,
  options: Omit<MockBridgeOptions, "delayMs"> = {},
  transformerRole: TransformerRole | null = null,
) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(chainTree(nativeBridge, extra, transformerRole));
  /** Re-renders the SAME tree (state kept) with new props. */
  const rerenderWith = (next: Partial<ChainedScreenProps>) =>
    view.rerender(chainTree(nativeBridge, next, transformerRole));
  return { ...view, rerenderWith };
}

const clipCards = () => Array.from(document.querySelectorAll<HTMLElement>(".clip-card"));
const clipCard = (index: number) => {
  const card = clipCards()[index];
  if (!card) throw new Error(`no clip card ${index}`);
  return card;
};
/** Clip `index`'s comfort tick value, or `null` when none is drawn. */
const marker = (index: number) => clipCard(index).querySelector("datalist option")?.getAttribute("value") ?? null;
const durationInput = (index: number) => clipCard(index).querySelector<HTMLInputElement>("input[type='number']")!;
const sourceSection = () => document.querySelector(".source-section") as HTMLElement;

async function waitForClips() {
  await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
}

/** Types the shared size into the Width/Height number boxes. */
function setSize(width: number, height: number) {
  for (const [label, value] of [
    [en.single.size.width, width],
    [en.single.size.height, height],
  ] as const) {
    const field = screen.getByText(label, { selector: ".field-label" }).closest("label") as HTMLElement;
    const input = field.querySelector<HTMLInputElement>("input[type='number']")!;
    fireEvent.change(input, { target: { value: String(value) } });
    fireEvent.blur(input);
  }
}

async function attachOpeningImage() {
  const user = userEvent.setup();
  await user.click(within(sourceSection()).getByRole("button", { name: en.chained.sourceInput.chooseButton }));
  await waitFor(() => expect(sourceSection().querySelector("img")).not.toBeNull(), { timeout: 5_000 });
}

afterEach(async () => {
  vi.restoreAllMocks();
  for (let i = 0; i < 50; i += 1) {
    const jobs = await apiClient.listJobs();
    if (jobs.every((j) => j.status !== "queued" && j.status !== "running")) return;
  }
});

describe("ChainedScreen — MiniMax H3 per-clip comfort marker (§1-276)", () => {
  it("1280x768: both clips are marked at the 481 ceiling", async () => {
    renderChain();
    await waitForClips();
    await waitFor(() => expect(marker(0)).toBe("481"));
    expect(marker(1)).toBe("481");
  });

  it("1920x1088: clip 0 with the opening image at 345, clip 1 (no keyframe) at 362", async () => {
    renderChain(H3_PROPS, { pickFileName: "start.png" });
    await waitForClips();
    setSize(1920, 1088);
    await waitFor(() => expect(marker(0)).toBe("362"));
    expect(marker(1)).toBe("362");
    await attachOpeningImage();
    await waitFor(() => expect(marker(0)).toBe("345"));
    expect(marker(1)).toBe("362");
  });

  // Ledger §1-286: the w4a8 weights take the per-kind line
  // (`single_rows_budget_by_quant_class.w4a8 = 203092`); int8/fp8 keep 219516.
  it("1920x1088 with the opening image: w4a8 marks clip 0 at 328, int8 at 345", async () => {
    for (const [kind, expected] of [
      ["w4a8", "328"],
      ["int8", "345"],
    ] as const) {
      const view = renderChain({ ...H3_PROPS, transformerQuantClass: kind }, { pickFileName: "start.png" });
      await waitForClips();
      setSize(1920, 1088);
      await attachOpeningImage();
      await waitFor(() => expect(marker(0)).toBe(expected));
      view.unmount();
    }
  });

  it("changing only the quantisation kind moves the marker 345 -> 328 on the same screen", async () => {
    const view = renderChain({ ...H3_PROPS, transformerQuantClass: "int8" }, { pickFileName: "start.png" });
    await waitForClips();
    setSize(1920, 1088);
    await attachOpeningImage();
    await waitFor(() => expect(marker(0)).toBe("345"));
    view.rerenderWith({ ...H3_PROPS, transformerQuantClass: "w4a8" });
    await waitFor(() => expect(marker(0)).toBe("328"));
  });

  it("2560x1472 with the opening image: 192 on both clips", async () => {
    // The fixture's size limits stop at 1920x1088; a config that allows 1440p.
    const served = apiClient.getConfig.bind(apiClient);
    vi.spyOn(apiClient, "getConfig").mockImplementation(async () => {
      const config = await served();
      return { ...config, limits: { ...config.limits, max_width: 2560, max_height: 1472 } };
    });
    renderChain(H3_PROPS, { pickFileName: "start.png" });
    await waitForClips();
    await attachOpeningImage();
    setSize(2560, 1472);
    await waitFor(() => expect(marker(0)).toBe("192"));
    expect(marker(1)).toBe("192");
  });

  it(
    "V2V: clip 0 carries no keyframe — 362 at 1920x1088",
    async () => {
      renderChain(H3_PROPS, { pickFileName: "clip.mp4" });
      await waitForClips();
      setSize(1920, 1088);
      const user = userEvent.setup();
      await user.click(within(sourceSection()).getByRole("button", { name: en.chained.sourceInput.chooseButton }));
      await screen.findByText(en.chained.sourceVideo.contextFramesHint(22), {}, { timeout: 5_000 });
      await waitFor(() => expect(marker(0)).toBe("362"));
    },
    15_000,
  );

  it("past the marker the clip shows the out-of-VRAM warning (1920x1088, clip 1 at 379)", async () => {
    renderChain();
    await waitForClips();
    setSize(1920, 1088);
    await waitFor(() => expect(marker(1)).toBe("362"));
    fireEvent.change(durationInput(1), { target: { value: "362" } });
    fireEvent.blur(durationInput(1));
    await waitFor(() => expect(durationInput(1).value).toBe("362"));
    expect(within(clipCard(1)).queryByText(en.single.comfortWarningOutOfMemory)).not.toBeInTheDocument();
    fireEvent.change(durationInput(1), { target: { value: "379" } });
    fireEvent.blur(durationInput(1));
    await waitFor(() => expect(within(clipCard(1)).getByText(en.single.comfortWarningOutOfMemory)).toBeInTheDocument());
    // Only the clip past its own marker warns.
    expect(within(clipCard(0)).queryByText(en.single.comfortWarningOutOfMemory)).not.toBeInTheDocument();
  });

  it("each clip's slider points at a tick of its own (two clips, two datalist ids)", async () => {
    renderChain();
    await waitForClips();
    await waitFor(() => expect(marker(1)).not.toBeNull());
    const ids = clipCards().map((card) => card.querySelector("datalist")?.id);
    expect(ids).toHaveLength(2);
    expect(new Set(ids).size).toBe(2);
    for (const [index, card] of clipCards().entries()) {
      const id = ids[index]!;
      expect(id).toMatch(/^chain-clip-comfort-tick-/);
      expect(card.querySelector("input[type='range']")?.getAttribute("list")).toBe(id);
    }
    expect(ids).not.toContain("spill-tick");
  });

  it(
    "a 15 s reference video on clip 1 moves its marker to 345 while References is on; off it is 481 again",
    async () => {
      renderChain(H3_PROPS, { probeMediaInfoDurationSec: 15 }, "ref2va");
      await waitForClips();
      await waitFor(() => expect(marker(1)).toBe("481"));
      const box = screen.getByRole("checkbox", { name: en.chained.referencesToggleLabel });
      fireEvent.click(box);
      const card = clipCard(1);
      const summary = within(card).getByText(en.single.reference.heading, { selector: "summary" });
      const details = summary.closest("details") as HTMLDetailsElement;
      if (!details.open) fireEvent.click(summary);
      const slot = within(card).getByTestId("ref-slot-video-1");
      fireEvent.click(within(slot).getByRole("button", { name: en.single.reference.chooseButton }));
      await waitFor(
        () => {
          const ready = within(clipCard(1)).getByTestId("ref-slot-video-1");
          expect(within(ready).getByRole("button", { name: en.single.reference.tagButton })).toBeEnabled();
        },
        { timeout: 5_000 },
      );
      await waitFor(() => expect(marker(1)).toBe("345"));
      expect(marker(0)).toBe("481");
      fireEvent.click(box);
      await waitFor(() => expect(marker(1)).toBe("481"));
    },
    15_000,
  );

  it("a right-click #6 at 1920x1088 seeds clip 0 at its marker, 345", async () => {
    const initialIntent: GenerationPrefill = {
      intent: "image-to-clip-chain",
      targetMode: "chained",
      selection: {
        hasRange: true,
        rangeStart: 0,
        rangeEnd: 120,
        selected: [
          {
            layer: 1,
            frameStart: 0,
            frameEnd: 120,
            effectName: "画像ファイル",
            filePath: "C:\\v\\still.png",
            objectName: "still",
            textContent: null,
            mediaWidth: 1920,
            mediaHeight: 1088,
            mediaDurationSec: 0,
          },
        ],
        cursorFrame: 0,
        cursorLayer: 1,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
      },
    };
    renderChain({ ...H3_PROPS, initialIntent });
    await waitForClips();
    await waitFor(() => expect(durationInput(0).value).toBe("345"));
    await waitFor(() => expect(marker(0)).toBe("345"), { timeout: 5_000 });
  });

  it("LTX (no constraints, no rows line) draws no per-clip marker", async () => {
    renderChain({ engineFamily: "ltx" });
    await waitForClips();
    for (const card of clipCards()) {
      expect(card.querySelector("datalist")).toBeNull();
      expect(card.querySelector("input[type='range']")?.getAttribute("list")).toBeNull();
    }
  });
});
