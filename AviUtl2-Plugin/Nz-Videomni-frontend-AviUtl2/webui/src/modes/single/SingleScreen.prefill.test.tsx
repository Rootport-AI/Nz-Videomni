import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { SingleScreen } from "./SingleScreen";

// §1-6 拡張 (2026-08-01): 右クリック#2「IC-LoRA参照動画」も、V2V(#1)と同じように
// リボンの範囲だけを切り出してアップロードする。実機症状は「321フレーム素材の
// 31〜120フレームのリボンでIC-LoRAを実行すると、参照が元動画の1〜90フレームに
// なる」で、原因はフル尺のままアップロードしていたこと（パイプラインはファイル
// 先頭から frame_cap 分を読む）。
//
// ここは配線テスト: `SingleScreen` の autoLoad が `decideSourceTrim` の判定を
// `referenceVideo.uploadPath` のクエリとして渡していることを、実際の
// `SingleScreen` レンダリング越しに固定する。判定そのものの網羅は
// `timeline/sourceTrim.test.ts`（純関数）側。
//
// R2 is the Create-side twin of `ChainedScreen.prefill.test.tsx`'s T24: a strict
// `toHaveBeenCalledWith` on the WHOLE params object, so a stray `query` key
// (even an empty or `undefined` one) fails here rather than in the field.

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

/** A #2 selection whose ribbon covers the WHOLE backing file: 121 inclusive
 * frames @30fps ≈ 4.033s of a 4s file, and no playback range reported. Every
 * `decideSourceTrim` guard resolves to "no trim", so the upload must stay
 * byte-identical to the pre-§1-6 one. */
function selectionWholeFile(filePath: string) {
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

/**
 * The owner's real-hardware repro, as a fixture: a 321-frame source at 30fps
 * (10.7s) whose ribbon is frames 31..120 — 90 inclusive frames = 3.0s starting
 * at 1.0s on the source's own time axis. The played window (1.0s..4.0s) matches
 * the ribbon exactly, so `min(span, window)` yields 3.000 either way.
 */
function selectionRibbon31to120(filePath: string) {
  const base = selectionWholeFile(filePath);
  const item = base.selected[0]!;
  return {
    ...base,
    selected: [
      {
        ...item,
        frameStart: 30,
        frameEnd: 119,
        mediaDurationSec: 10.7,
        hasPlaybackRange: true,
        playbackStartSec: 1.0,
        playbackEndSec: 4.0,
      },
    ],
  };
}

function renderCreate(initialIntent?: GenerationPrefill) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const requestSpy = vi.spyOn(nativeBridge, "request");
  const rendered = render(
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
  return { ...rendered, nativeBridge, requestSpy };
}

/** The `backend.uploadFile` calls seen so far, as `[method, params]` pairs.
 * Structurally typed on just `mock.calls` so it needs no `MockInstance` generics. */
function uploadCalls(spy: { mock: { calls: unknown[][] } }): unknown[][] {
  return spy.mock.calls.filter((call) => call[0] === "backend.uploadFile");
}

describe("SingleScreen — #2 reference-video ribbon trim (§1-6 拡張)", () => {
  it("R1: passes both trim parameters as toFixed(3) strings for a ribbon that is a window into a longer file", async () => {
    const { requestSpy } = renderCreate({
      intent: "reference-video",
      targetMode: "single",
      selection: selectionRibbon31to120("C:\\v\\ref.mp4"),
    });

    await waitFor(() => expect(uploadCalls(requestSpy).length).toBeGreaterThan(0), { timeout: 5_000 });
    expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\ref.mp4",
      query: { trim_start_sec: "1.000", trim_duration_sec: "3.000" },
    });
  });

  it("R2: sends the exact pre-§1-6 params (no `query` key at all) when no trim applies", async () => {
    const { requestSpy } = renderCreate({
      intent: "reference-video",
      targetMode: "single",
      selection: selectionWholeFile("C:\\v\\ref.mp4"),
    });

    await waitFor(() => expect(uploadCalls(requestSpy).length).toBeGreaterThan(0), { timeout: 5_000 });
    expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\ref.mp4",
    });
    // …and no upload carried a query, however shaped.
    for (const [, params] of uploadCalls(requestSpy)) {
      expect(Object.keys(params as object).sort()).toEqual(["filePath", "kind"]);
    }
  });
});
