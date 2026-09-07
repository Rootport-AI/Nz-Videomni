import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { bridge as defaultBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { END_SOURCE_SEED_MAX_FRAMES } from "../../timeline/prefillSeed";
import {
  getReservationState,
  reservePlacement,
  resetProvisionalReservation,
} from "../../timeline/provisionalReservation";
import { END_SOURCE_CONTEXT_FRAMES } from "../../timeline/tailAlign";
import { ChainedScreen } from "./ChainedScreen";

/**
 * 素材（末尾）: the pieces that live on the SCREEN rather than in
 * `ChainEndSourcePanel` — the accordion's placement (directly after the clip
 * list, before the reference/audio pair), the right-click `"end-with-this"`
 * intent's three-part wiring (the early-return gate, the attach branch, the
 * accordion auto-open), the 窓内モード single-clip門 as the clip list renders it,
 * and the Generate-time 打ち直し. Forgetting any part of the right-click wiring
 * is a SILENT no-op, so all of it is asserted end-to-end here through the real
 * screen; the hook-level routing itself is covered by `useChainForm.test.ts`.
 */
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

function selectionWithFile(filePath: string, frameStart = 0) {
  return {
    hasRange: true,
    rangeStart: frameStart,
    rangeEnd: frameStart + 120,
    selected: [
      {
        layer: 1,
        frameStart,
        frameEnd: frameStart + 120,
        effectName: "動画ファイル",
        filePath,
        objectName: "a",
        textContent: null,
        mediaWidth: 1280,
        mediaHeight: 720,
        mediaDurationSec: 30,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

/** The app-wide bridge the SUBMIT goes through (`useGenerationSubmit` uses the
 * default `apiClient`, which is bound to this singleton — the screen's
 * `nativeBridge` prop only carries the timeline RPCs). Captured before any spy
 * so a spy can delegate to it without recursing into itself. */
const DEFAULT_BRIDGE_REQUEST = defaultBridge.request.bind(defaultBridge);

/** The bridge's own `request` signature, before any spy wraps it. */
type RawRequest = ReturnType<typeof createMockBridge>["request"];

/**
 * Empties the app-wide mock's job fixture.
 *
 * That bridge is a MODULE SINGLETON, so a job one test submits is still "on
 * the server" while the next test renders — and `JobsContext.serverBusy`
 * renames every Generate button to 「Busy…」, which makes the very next
 * `getByRole("button", { name: "Generate" })` fail. This file used to get away
 * without a cleanup only by accident: the mock 422'd a 1-clip `end_source`
 * chain, so no job was ever created. §3-116 fixed that — `api/models.py`'s
 * clip-count floor exempts `end_source` (a single clip is "a video that ends
 * with this"), so the real server answers 202 here and the mock now does too.
 * The submits in this describe therefore really do leave a job behind, and it
 * has to be cleaned up explicitly.
 *
 * The fixture's queued->running->completed clock is driven by poll count, so
 * the drain alternates `GET /jobs` (advances every job) with `DELETE
 * /jobs/{id}` (flags a still-running job cancelled; removes an already
 * terminal one) until the list comes back empty.
 */
async function drainDefaultBridgeJobs(): Promise<void> {
  for (let guard = 0; guard < 10; guard += 1) {
    const listed = await DEFAULT_BRIDGE_REQUEST("backend.request", { method: "GET", path: "/api/v1/jobs" });
    const jobs = (listed.body ?? []) as { job_id: string }[];
    if (jobs.length === 0) return;
    for (const job of jobs) {
      await DEFAULT_BRIDGE_REQUEST("backend.request", { method: "DELETE", path: `/api/v1/jobs/${job.job_id}` });
    }
  }
  throw new Error("the app-wide mock bridge's job fixture never drained");
}

function renderChain(
  initialIntent?: GenerationPrefill,
  /** Extra mock options. `pickFileName` is the only one used so far (§3-90's
   * start-source test needs the 📁 dialog to answer with a VIDEO — the
   * `imageOrVideo` kind otherwise defaults to a .png). */
  mockOptions: Parameters<typeof createMockBridge>[0] = {},
) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...mockOptions });
  /** Same capture, for the per-test bridge. */
  const nativeRequest = nativeBridge.request.bind(nativeBridge);
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
  return { ...rendered, nativeBridge, requestSpy, nativeRequest };
}

/** The `<details>` the end-source card lives in. */
function endAccordion(container: HTMLElement): HTMLDetailsElement {
  return container.querySelector<HTMLDetailsElement>(".chain-end-source-section")!.closest("details")!;
}

/** The predicted-output line, parsed back into its two numbers. The screen is
 * the only place that knows the seeded clip length for this prefill, so reading
 * the rendered total is what keeps these tests free of a hand-copied seed. */
function predictedOutput(container: HTMLElement): { total: number; tail: number } {
  const line = [...container.querySelectorAll(".field-hint")]
    .map((el) => el.textContent ?? "")
    .find((text) => text.startsWith("Predicted output:"));
  if (!line) throw new Error("predicted-output line not rendered");
  const numbers = [...line.matchAll(/\((\d+) frames\)/g)].map((m) => Number(m[1]));
  const [total, tail] = numbers;
  if (total === undefined) throw new Error(`could not parse frames out of: ${line}`);
  return { total, tail: tail ?? 0 };
}

/** The end-source card's own quality warning. Scoped to the card on purpose:
 * `.warning-banner-mild` is also the class of the Generate-reasons note the
 * batch section below renders, so an unscoped query would find that instead. */
function qualityWarning(container: HTMLElement): HTMLElement | null {
  return container.querySelector<HTMLElement>(".chain-end-source-section .warning-banner-mild");
}

/** The 「クリップを追加」 button. */
function addClipButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: en.chained.addClipButton }) as HTMLButtonElement;
}

describe("ChainedScreen — 素材（末尾）", () => {
  // Both halves of the shared state this file leaves behind, cleaned up BEFORE
  // the next render rather than after the previous test: Testing Library's own
  // auto-cleanup (an `afterEach`) has unmounted the previous screen by now, so
  // nothing is still polling while the fixture is drained. The spies on the
  // singleton bridge are dropped first, so the drain talks to the untouched
  // fixture and no stale `order` array is still being written to.
  beforeEach(async () => {
    vi.restoreAllMocks();
    resetProvisionalReservation();
    await drainDefaultBridgeJobs();
  });

  it("places the end-source accordion after the clip list and before the reference/audio pair", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    const clipList = container.querySelector(".clip-list")!;
    const end = container.querySelector(".chain-end-source-section")!;
    const reference = container.querySelector(".chain-reference-section")!;
    expect(clipList.compareDocumentPosition(end) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(end.compareDocumentPosition(reference) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("starts collapsed, and the START slot is now labelled as the start material", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    expect(endAccordion(container).open).toBe(false);
    expect(screen.getByText(en.chained.endSource.heading)).toBeInTheDocument();
    expect(screen.getByText(en.chained.sourceInput.heading)).toBeInTheDocument();
    expect(en.chained.sourceInput.heading).toBe("Start source");
  });

  describe("the right-click 「これで終わる動画を作る」 intent", () => {
    it("opens the accordion and loads a VIDEO into the end slot", async () => {
      const { container, requestSpy } = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\v\\ending.mp4"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

      expect(endAccordion(container).open).toBe(true);
      await waitFor(() => expect(screen.getByAltText(en.chained.endSource.videoPlaceholderAlt)).toBeInTheDocument(), {
        timeout: 5_000,
      });
      expect(screen.getByText("ending.mp4")).toBeInTheDocument();
      // It went to the END slot, not the START one: the start card still says
      // "no source selected".
      expect(screen.getByText(en.chained.sourceInput.none)).toBeInTheDocument();
      // The end slot always caps its upload — which is also what makes the
      // server measure the file and answer with `frame_count`/`fps`.
      expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
        kind: "video",
        filePath: "C:\\v\\ending.mp4",
        query: { max_frames: "685" },
      });
    });

    it("opens the accordion and loads an IMAGE into the end slot (never the clip-0 start frame)", async () => {
      const { container, requestSpy } = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\i\\ending.png"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

      expect(endAccordion(container).open).toBe(true);
      // The image-only still warning is the end card's own note, and it can only
      // appear once the image really landed in THIS slot.
      await waitFor(
        () =>
          expect(container.querySelector(".chain-end-source-note")?.textContent).toBe(
            en.chained.endSource.stillImageNote,
          ),
        { timeout: 5_000 },
      );
      // The clip-0 start frame stayed empty — the branch sits BEFORE the
      // image→start-frame fallback.
      expect(screen.getByText(en.chained.sourceInput.none)).toBeInTheDocument();
      expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
        kind: "image",
        filePath: "C:\\i\\ending.png",
      });
    });

    // 窓内モード (2026-08-17): the seed is rounded DOWN to one stage-2 window, so
    // the request the右クリック hands the user is already inside the quality
    // ceiling — pressing Generate straight away cannot produce the smearing the
    // card warns about.
    it("seeds clip 0 at the 169-frame quality ceiling, and the output does NOT include the frozen tail", async () => {
      const { container } = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\v\\ending.mp4"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await waitFor(() => expect(predictedOutput(container).tail).toBe(END_SOURCE_CONTEXT_FRAMES), {
        timeout: 5_000,
      });

      // ONE clip (`forceSingleClip`), seeded at the rounded comfort ceiling.
      expect(container.querySelectorAll(".clip-list > li")).toHaveLength(1);
      expect(END_SOURCE_SEED_MAX_FRAMES).toBe(169);
      const { total, tail } = predictedOutput(container);
      expect(total).toBe(END_SOURCE_SEED_MAX_FRAMES);
      // 窓内モード's headline spec change: 出力長 = クリップ尺. The tail is a slice
      // OF the total, not an addition to it.
      expect(tail).toBe(8);
      expect(container.textContent).toContain("are the material");
      // …and the seeded clip is inside the ceiling, so the warning is silent.
      expect(qualityWarning(container)).toBeNull();
    });

    // 逆順Chained (2026-08-18, second stage): the single-clip note and the
    // button-disabling behind it are GONE — 2+ clips is the `reverse` mode the
    // server now accepts, so 「クリップを追加」 stays exactly as enabled as it is
    // on any other chain (only the ordinary 24-clip ceiling can disable it).
    it("keeps 「クリップを追加」 enabled while material is attached, and adding a clip enters reverse mode", async () => {
      const { container } = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\v\\ending.mp4"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await waitFor(() => expect(container.querySelectorAll(".clip-list > li")).toHaveLength(1), {
        timeout: 5_000,
      });

      expect(addClipButton()).toBeEnabled();
      expect(screen.queryByText(en.chained.maxClipsReached(24))).not.toBeInTheDocument();

      fireEvent.click(addClipButton());
      await waitFor(() => expect(container.querySelectorAll(".clip-list > li")).toHaveLength(2), {
        timeout: 5_000,
      });
      expect(addClipButton()).toBeEnabled();
      // 2.6(e): the seam-blend width was nudged to 1 for this now-2-clip chain,
      // and the hint says so.
      expect(screen.getByText(en.chained.endSource.reverseOverlapHint)).toBeInTheDocument();

      // Detaching leaves the button enabled (it always was) and the note goes.
      fireEvent.click(container.querySelector(".chain-end-source-clear")!);
      await waitFor(() => expect(screen.queryByText(en.chained.endSource.reverseOverlapHint)).not.toBeInTheDocument(), {
        timeout: 5_000,
      });
    });

    // §3-90 (2026-09-07): the hint belongs to the REVERSE chain only. Adding a
    // 素材（冒頭）makes the same 2-clip chain the FORWARD `bridge` chain — every
    // clip generated in order, only the last one conditioned on both sides —
    // where nothing was nudged and there is nothing to explain. Asserted on the
    // real screen because the hint's condition is the screen's own.
    it("hides the 遡り生成 hint once a START source joins the same 2-clip chain", async () => {
      const { container } = renderChain(
        {
          intent: "end-with-this",
          targetMode: "chained",
          selection: selectionWithFile("C:\\v\\ending.mp4"),
        },
        { pickFileName: "start.mp4" },
      );
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await waitFor(() => expect(container.querySelectorAll(".clip-list > li")).toHaveLength(1), {
        timeout: 5_000,
      });

      fireEvent.click(addClipButton());
      await waitFor(() => expect(container.querySelectorAll(".clip-list > li")).toHaveLength(2), {
        timeout: 5_000,
      });
      expect(screen.getByText(en.chained.endSource.reverseOverlapHint)).toBeInTheDocument();

      // The START card's 📁 button — the mock dialog answers "start.mp4", so
      // this really is a source VIDEO (an image would land on clip 0 as a
      // keyframe and leave the chain in reverse mode).
      fireEvent.click(container.querySelector(".source-input-pick")!);
      await waitFor(() => expect(screen.getByAltText(en.chained.sourceInput.videoPlaceholderAlt)).toBeInTheDocument(), {
        timeout: 5_000,
      });

      expect(screen.queryByText(en.chained.endSource.reverseOverlapHint)).not.toBeInTheDocument();
      // The clip list is untouched: still the 2-clip chain, now a bridge one.
      expect(container.querySelectorAll(".clip-list > li")).toHaveLength(2);
    });

    it("shows the quality warning once the clip outgrows one stage-2 window, and hides it again", async () => {
      const { container } = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\v\\ending.mp4"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await waitFor(() => expect(predictedOutput(container).tail).toBe(8), { timeout: 5_000 });
      expect(qualityWarning(container)).toBeNull();

      // The clip's own DURATION number input: push it one 8n+1 step past 169.
      const duration = container.querySelector<HTMLInputElement>(".clip-list input[type='number']")!;
      fireEvent.change(duration, { target: { value: "177" } });
      await waitFor(
        () => expect(qualityWarning(container)?.textContent).toBe(en.chained.endSource.qualityWarning(169)),
        { timeout: 5_000 },
      );

      fireEvent.change(duration, { target: { value: "169" } });
      await waitFor(() => expect(qualityWarning(container)).toBeNull(), { timeout: 5_000 });
    });

    it("leaves the accordion closed and the slot empty for any other intent", async () => {
      const { container } = renderChain({
        intent: "image-to-clip-chain",
        targetMode: "chained",
        selection: selectionWithFile("C:\\i\\photo.png"),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

      expect(endAccordion(container).open).toBe(false);
      expect(container.querySelector(".chain-end-source-note")?.textContent).toBe(en.chained.endSource.none);
    });
  });

  /**
   * 系統E の打ち直し。右クリック時の予約はシード値で置かれるので、Generate を
   * 押した瞬間に確定値で置き直してから送信する。順序（打ち直し → 送信）そのものが
   * 仕様なので、両方のブリッジ呼び出しを1本の記録配列に流し込んで並びを検証する。
   */
  describe("the Generate-time 打ち直し", () => {
    const MATERIAL_FRAME_START = 600;

    /** Renders the right-click flow, waits for the end material to land, and
     * takes the seat the way `AppShell` Step 8 would. */
    async function renderReserved() {
      const view = renderChain({
        intent: "end-with-this",
        targetMode: "chained",
        selection: selectionWithFile("C:\\v\\ending.mp4", MATERIAL_FRAME_START),
      });
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await waitFor(() => expect(predictedOutput(view.container).tail).toBe(8), { timeout: 5_000 });

      // AppShell's own reservation — deliberately at a frame the confirmed
      // re-place will move away from.
      await reservePlacement(view.nativeBridge, {
        placement: "E",
        material: { layer: 1, frameStart: 0, frameEnd: MATERIAL_FRAME_START + 120 },
        numFrames: 999,
        genFps: 24,
        displayText: "(reserved)",
      });
      expect(getReservationState().phase).toBe("reserved");
      return view;
    }

    /** One ordered log of BOTH channels: the reservation RPC rides the screen's
     * `nativeBridge` prop, the submit rides the app-wide default bridge's
     * `backend.request`. Both spies DELEGATE to the pre-spy implementation, so
     * the mock's real behaviour is untouched — only the ordering is observed. */
    function recordOrder(
      view: { nativeBridge: ReturnType<typeof createMockBridge>; nativeRequest: RawRequest },
      onReserve?: () => Promise<never> | null,
    ): string[] {
      const order: string[] = [];
      vi.spyOn(view.nativeBridge, "request").mockImplementation(((method: string, params: unknown) => {
        if (method === "timeline.updateProvisionalReservation" || method === "timeline.insertProvisional") {
          order.push(method);
          const override = onReserve?.();
          if (override) return override;
        }
        return view.nativeRequest(method as never, params as never);
      }) as never);
      vi.spyOn(defaultBridge, "request").mockImplementation(((method: string, params: unknown) => {
        const path = (params as { path?: string } | undefined)?.path;
        if (method === "backend.request" && path?.includes("/generate/chain")) order.push("submit");
        return DEFAULT_BRIDGE_REQUEST(method as never, params as never);
      }) as never);
      return order;
    }

    it("re-places the reservation with the CONFIRMED length, then submits", async () => {
      const view = await renderReserved();
      const order = recordOrder(view);
      const nativeSpy = vi.mocked(view.nativeBridge.request);

      fireEvent.click(screen.getByRole("button", { name: en.chained.generateButton }));

      await waitFor(() => expect(order).toContain("submit"), { timeout: 5_000 });
      // 順序が仕様: 打ち直し → 送信。
      expect(order.indexOf("timeline.updateProvisionalReservation")).toBeGreaterThanOrEqual(0);
      expect(order.indexOf("timeline.updateProvisionalReservation")).toBeLessThan(order.indexOf("submit"));

      const call = nativeSpy.mock.calls.find(([method]) => method === "timeline.updateProvisionalReservation");
      const params = call?.[1] as {
        placement: string;
        materialFrameStart: number;
        numFrames: number;
        genFps: number;
      };
      // "E" never reaches the wire — `placementParams` maps it onto "B".
      expect(params.placement).toBe("B");
      const { total, tail } = predictedOutput(view.container);
      // The CONFIRMED length: the whole delivered file, which 窓内モード makes
      // exactly the clip length.
      expect(params.numFrames).toBe(total);
      expect(tail).toBe(8);
      // 末尾合わせ（厳密整列）: 素材開始 − round((出力長 − 凍結分 − 1) ×
      // projectFps / genFps), ties UP. The `− 1` is the causal VAE's keyframe
      // primer: the material's OWN 1st frame lands one frame further back than
      // the frozen-tail overlap alone would place it (2026-08-17実測確認).
      // `projectFps` is the selection's own rate/scale (30); `genFps` is read
      // back off the same call rather than hard-coded, since the prefill seeds
      // the form's frame rate from the material.
      const expectedStart =
        MATERIAL_FRAME_START - Math.floor(((total - tail - 1) * 30) / params.genFps + 0.5);
      expect(params.materialFrameStart).toBe(expectedStart);
      // The estimate's frame 0 really did move, and the frozen tail still lands
      // ON the material rather than past it.
      expect(params.materialFrameStart).not.toBe(0);
      expect(params.materialFrameStart).toBeLessThan(MATERIAL_FRAME_START);
    });

    it("submits anyway when the re-place rejects", async () => {
      const view = await renderReserved();
      // The re-place RPC rejects; everything else still behaves normally.
      const order = recordOrder(view, () => Promise.reject(new Error("PROVISIONAL_FAILED")) as never);

      fireEvent.click(screen.getByRole("button", { name: en.chained.generateButton }));

      // 打ち直しに失敗しても送信は続く（リボンの位置がずれるだけ）。
      await waitFor(() => expect(order).toContain("submit"), { timeout: 5_000 });
      expect(order.indexOf("timeline.updateProvisionalReservation")).toBeLessThan(order.indexOf("submit"));
    });

    it("skips the re-place entirely for a MANUAL Generate (no right-click anchor)", async () => {
      // No `initialIntent` at all: the tab was opened by hand. Even with a seat
      // held by some other flow, this screen must not insert/move anything —
      // otherwise a plain chain would drop a placeholder nobody asked for.
      const view = renderChain();
      await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
      await reservePlacement(view.nativeBridge, {
        placement: "C",
        cursor: { layer: 3, frame: 100 },
        numFrames: 100,
        genFps: 24,
        displayText: "(someone else)",
      });
      const order = recordOrder(view);

      fireEvent.click(screen.getByRole("button", { name: en.chained.generateButton }));

      await waitFor(() => expect(order).toContain("submit"), { timeout: 5_000 });
      expect(order).not.toContain("timeline.updateProvisionalReservation");
      expect(order).not.toContain("timeline.insertProvisional");
    });
  });
});
