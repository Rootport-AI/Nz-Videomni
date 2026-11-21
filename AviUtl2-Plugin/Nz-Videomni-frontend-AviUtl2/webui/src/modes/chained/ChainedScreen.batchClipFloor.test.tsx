import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import type { GenerationConstraints } from "../../shell/generationConstraints";
import { __resetV2vRefLongStoreForTests, v2vRefLongStore } from "../batch-v2v-ref-long/store";
import type { ScanResult } from "../batch-v2v-ref-long/types";
import { ChainedScreen } from "./ChainedScreen";
import type { ChainedScreenProps } from "./ChainedScreen";

// Batch v2v-ref-long: on MiniMax H3, a scan result with an opening video lets
// the Chained clip cards go down to one clip. The manual Generate keeps its own
// 2-clip check (reason shown, button disabled), LTX is unaffected, and losing
// the trigger never adds a clip back.

const H3_REFERENCE: GenerationConstraints = resolveGenerationConstraints({
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
  a2vUnavailable: true,
  endSourceUnavailable: true,
  referenceUnavailable: true,
  endSourceHidden: true,
  referenceHidden: true,
  startImageStrengthHidden: true,
  loraAudioHidden: true,
};

const RULES = H3_REFERENCE.reference!;

function scanWithOpeningVideos(): ScanResult {
  return {
    inputs: { videoDir: "C:\\src", frames: [] },
    source: [
      { name: "s0.mp4", path: "C:\\src\\s0.mp4", sizeBytes: 1, durationSec: 5 },
      { name: "s1.mp4", path: "C:\\src\\s1.mp4", sizeBytes: 1, durationSec: 5 },
    ],
    frames: [],
    rules: RULES,
  };
}

function scanWithoutOpeningVideo(): ScanResult {
  return {
    inputs: { videoDir: "", frames: [] },
    source: [],
    frames: [
      [
        {
          kind: "picture",
          slot: 1,
          path: "C:\\pics",
          files: [{ name: "p0.png", path: "C:\\pics\\p0.png", sizeBytes: 1, durationSec: null }],
        },
      ],
    ],
    rules: RULES,
  };
}

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

function renderChain(extra: Partial<ChainedScreenProps> = {}) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
        {...extra}
      />
    </Providers>,
  );
}

/** The clip cards' remove buttons (the form column only, never the batch section). */
function removeButtons(container: HTMLElement): HTMLElement[] {
  const layout = container.querySelector(".single-layout") as HTMLElement;
  return within(layout).queryAllByRole("button", { name: en.chained.removeClipButton });
}

function clipCount(container: HTMLElement): number {
  const layout = container.querySelector(".single-layout") as HTMLElement;
  return within(layout).queryAllByText(/^clip \d+$/i).length;
}

afterEach(() => {
  __resetV2vRefLongStoreForTests();
});

describe("ChainedScreen — Batch v2v-ref-long relaxes the clip removal floor", () => {
  it("H3 with a scanned opening video: the remove button trims to one clip; the manual Generate stays blocked", async () => {
    v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scanWithOpeningVideos() });
    const user = userEvent.setup();
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(clipCount(container)).toBe(2);
    // Two clips: Generate is ready, so the later disable is the clip count's.
    const generate = screen.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(generate).toBeEnabled(), { timeout: 5_000 });

    const buttons = removeButtons(container);
    expect(buttons).toHaveLength(2);
    await user.click(buttons[1]!);

    expect(clipCount(container)).toBe(1);
    expect(removeButtons(container)).toHaveLength(0);
    // The sentence appears once inside the layout: in the reason list under
    // Generate (`warning-banner-mild`), not as a red settings-column banner
    // (Batch i2v-long's own list, outside the layout, keeps blocking too).
    const layout = container.querySelector(".single-layout") as HTMLElement;
    const sentences = within(layout).getAllByText(en.chained.minClipsError(2));
    expect(sentences).toHaveLength(1);
    expect(sentences[0]!.closest(".warning-banner-mild")).not.toBeNull();
    const redBanners = Array.from(layout.querySelectorAll(".warning-banner:not(.warning-banner-mild)"));
    expect(redBanners.some((banner) => banner.textContent === en.chained.minClipsError(2))).toBe(false);
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
  });

  it.each([
    ["no scan", null],
    ["a scan without opening videos", scanWithoutOpeningVideo()],
  ] as const)("H3 with %s: two clips show no remove button", async (_label, scan) => {
    if (scan) v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scan });
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(clipCount(container)).toBe(2);
    expect(removeButtons(container)).toHaveLength(0);
  });

  it("LTX (no constraints) ignores a scanned opening video: two clips show no remove button", async () => {
    v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scanWithOpeningVideos() });
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(clipCount(container)).toBe(2);
    expect(removeButtons(container)).toHaveLength(0);
  });

  it("a re-scan without opening videos keeps the single clip", async () => {
    v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scanWithOpeningVideos() });
    const user = userEvent.setup();
    const { container } = renderChain(H3_PROPS);
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    await user.click(removeButtons(container)[1]!);
    expect(clipCount(container)).toBe(1);

    act(() => {
      v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scanWithoutOpeningVideo() });
    });
    expect(clipCount(container)).toBe(1);
    expect(removeButtons(container)).toHaveLength(0);
  });
});
