import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { bridge as appBridge } from "../../bridge";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { resolveGenerationConstraints } from "../../shell/generationConstraints";
import { SingleScreen } from "./SingleScreen";

// Single-clip A2V routing: an engine that publishes `constraints.source_audio`
// (MiniMax H3) submits the card through `/generate` with
// `source_audio: {audio_id, strength}`; one that does not (LTX) keeps the
// chain submission. The audio arrives as a carried-over prefill slot.

const H3_CONSTRAINTS = resolveGenerationConstraints({
  num_frames: { step: 17, offset: 5, min: 22, max: 481 },
  dimension_multiple: 32,
  frame_rate: { min: 24, max: 24 },
  sampling: null,
  source_audio: { strength: { min: 0, max: 1, step: 0.05, default: 0.5 } },
});

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

function renderWithAudio(constraints: typeof H3_CONSTRAINTS | undefined) {
  // A 20 s wav, so the audio covers the form's default duration.
  const fs = createMockFs({ folders: { "C:\\a": [{ name: "bg.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 20 }] } });
  const nativeBridge = createMockBridge({ delayMs: 0, fs });
  // Submissions go through the app-wide singleton client, so spy on that.
  const requestSpy = vi.spyOn(appBridge, "request");
  render(
    <Providers nativeBridge={nativeBridge}>
      <SingleScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        constraints={constraints}
        initialIntent={{
          intent: "audio-to-video",
          targetMode: "single",
          selection: {
            hasRange: true,
            rangeStart: 0,
            rangeEnd: 120,
            selected: [],
            cursorFrame: 0,
            cursorLayer: 1,
            rate: 30,
            scale: 1,
            sampleRate: 44100,
          },
          carryOver: { sourceAudio: { id: "aud-123", fileName: "bg.wav", filePath: "C:\\a\\bg.wav" } },
        }}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLora={null}
        setControlLora={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
  const postsTo = (path: string) =>
    requestSpy.mock.calls
      .filter((c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === path)
      .map((c) => (c[1] as { body: Record<string, unknown> }).body);
  return { postsTo };
}

async function clickGenerate() {
  // Re-query on every poll: the button is re-rendered while the form settles.
  await waitFor(() => expect(screen.getByRole("button", { name: /^generate$/i })).toBeEnabled(), { timeout: 5_000 });
  fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SingleScreen — single-clip A2V routing", () => {
  it(
    "with constraints.source_audio: /generate carries source_audio {audio_id, strength} (the slider's value) and no chain-only fields",
    async () => {
      const { postsTo } = renderWithAudio(H3_CONSTRAINTS);
      fireEvent.change(await screen.findByTestId("source-audio-strength"), { target: { value: "0.5" } });
      // Without an AviUtl2 selection the prefill leaves DURATION at 273 (off H3's 17n+5 grid)
      // until the wav probe has settled (it re-seeds DURATION once); type a valid one, re-applied until Generate opens.
      await waitFor(() => {
        const duration = document.querySelector<HTMLInputElement>("input[type=number][step='17']")!;
        fireEvent.change(duration, { target: { value: "277" } });
        fireEvent.blur(duration);
        expect(duration.value).toBe("277");
        expect(screen.getByRole("button", { name: /^generate$/i })).toBeEnabled();
      }, { timeout: 5_000 });
      await clickGenerate();

      await waitFor(() => expect(postsTo("/api/v1/generate")).toHaveLength(1));
      const body = postsTo("/api/v1/generate")[0]!;
      expect(body.source_audio).toEqual({ audio_id: "aud-123", strength: 0.5 });
      for (const key of ["clips", "stage2_window", "overlap_frames", "overlap_strength", "chunked_upsample", "pipeline"]) {
        expect(body, key).not.toHaveProperty(key);
      }
      expect(postsTo("/api/v1/generate/chain")).toHaveLength(0);
    },
    15_000,
  );

  it(
    "without it (LTX): the A2V card still goes through /generate/chain, as before",
    async () => {
      const { postsTo } = renderWithAudio(undefined);
      expect(screen.queryByTestId("source-audio-strength")).not.toBeInTheDocument();
      await clickGenerate();

      await waitFor(() => expect(postsTo("/api/v1/generate/chain")).toHaveLength(1));
      expect(postsTo("/api/v1/generate")).toHaveLength(0);
    },
    15_000,
  );
});
