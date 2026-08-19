import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { ChainedScreen } from "./ChainedScreen";

// X4 (右クリックv2vのクリップ1枚化): a #1 `extend-video` right-click prefill
// attaches a source video (V2V, 1-clip floor), but the source is still idle at
// mount, so without `forceSingleClip` the scratch 2-clip minimum would pad in a
// stray 2nd clip card. `ChainedScreen` sets `forceSingleClip` only for
// extend-video; every other flow keeps the 2-clip scratch default. These render
// the real `ChainedScreen` end-to-end; the hook-level branch is covered by
// `useChainForm.test.ts`.

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
        effectName: "動画ファイル",
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

/** §1-6: `selectionWithFile`'s object plus the contract-v10 playback fields, so
 * `decideSourceTrim` actually reaches a `trim: true`. 150 frames @30fps = a 5.0s
 * ribbon that plays from 12.5s to the end of a 60s file — the played window
 * (47.5s) is far longer than the ribbon, so `min(span, window)` picks the span
 * and the emitted duration is 5.000. */
function selectionWithTrimmableFile(filePath: string) {
  const base = selectionWithFile(filePath);
  const item = base.selected[0]!;
  return {
    ...base,
    selected: [
      {
        ...item,
        frameStart: 0,
        frameEnd: 149,
        mediaDurationSec: 60,
        hasPlaybackRange: true,
        playbackStartSec: 12.5,
        playbackEndSec: 60,
      },
    ],
  };
}

function renderChain(initialIntent?: GenerationPrefill) {
  const nativeBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
  const requestSpy = vi.spyOn(nativeBridge, "request");
  const rendered = render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        initialIntent={initialIntent}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
  return { ...rendered, nativeBridge, requestSpy };
}

/** The `backend.uploadFile` calls seen so far, as `[method, params]` pairs.
 * Structurally typed on just `mock.calls` so it needs no `MockInstance` generics. */
function uploadCalls(spy: { mock: { calls: unknown[][] } }): unknown[][] {
  return spy.mock.calls.filter((call) => call[0] === "backend.uploadFile");
}

describe("ChainedScreen — #1 extend-video single-clip prefill (X4)", () => {
  it("opens with exactly one clip card for an extend-video prefill", async () => {
    const { container } = renderChain({
      intent: "extend-video",
      targetMode: "chained",
      selection: selectionWithFile("C:\\v\\clip.mp4"),
    });

    // Wait for the config to load and the clip list to render.
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    // Exactly one clip card — no stray padded 2nd clip.
    expect(container.querySelectorAll(".clip-card")).toHaveLength(1);
    expect(screen.queryByText(/^clip 2$/i)).toBeNull();

    // ...and it stays one after the auto-loaded source video settles (the
    // ready-transition fit relaxes the floor but never re-pads back up).
    await waitFor(() => expect(container.querySelectorAll(".clip-card")).toHaveLength(1), { timeout: 5_000 });
    expect(screen.queryByText(/^clip 2$/i)).toBeNull();
  });

  it("keeps the scratch 2-clip default for a #6 image-to-clip-chain prefill (only extend-video is single)", async () => {
    const { container } = renderChain({
      intent: "image-to-clip-chain",
      targetMode: "chained",
      selection: selectionWithFile("C:\\i\\photo.png"),
    });

    await screen.findByText(/^clip 2$/i, {}, { timeout: 5_000 });
    expect(container.querySelectorAll(".clip-card")).toHaveLength(2);
  });
});

// §1-6 (V2Vリボン範囲トリム): 無トリム時のアップロード要求が改修前と**完全に同一**
// であることを機械的に固定する（T24）のと、トリム発火時にクエリ2キーが `toFixed(3)`
// 文字列で渡ることの確認（T23）。
//
// T24 is the WebUI half of the four-layer "an untrimmed upload is byte-identical
// to before §1-6" guarantee: a strict `toHaveBeenCalledWith` on the WHOLE params
// object, so a stray `query` key (even an empty or `undefined` one) fails here
// rather than in the field. Today EVERY real right-click lands in this case,
// because the bridge reports no playback position yet.
describe("ChainedScreen — #1 extend-video source trim (§1-6)", () => {
  it("T24: sends the exact pre-§1-6 params (no `query` key at all) when no trim applies", async () => {
    const { requestSpy } = renderChain({
      intent: "extend-video",
      targetMode: "chained",
      selection: selectionWithFile("C:\\v\\clip.mp4"),
    });

    await waitFor(() => expect(uploadCalls(requestSpy).length).toBeGreaterThan(0), { timeout: 5_000 });
    expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
    });
    // …and no upload carried a query, however shaped.
    for (const [, params] of uploadCalls(requestSpy)) {
      expect(Object.keys(params as object).sort()).toEqual(["filePath", "kind"]);
    }
  });

  it("T23: passes both trim parameters as toFixed(3) strings once a trim fires", async () => {
    const { requestSpy } = renderChain({
      intent: "extend-video",
      targetMode: "chained",
      selection: selectionWithTrimmableFile("C:\\v\\clip.mp4"),
    });

    await waitFor(() => expect(uploadCalls(requestSpy).length).toBeGreaterThan(0), { timeout: 5_000 });
    expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
      query: { trim_start_sec: "12.500", trim_duration_sec: "5.000" },
    });
  });
});
