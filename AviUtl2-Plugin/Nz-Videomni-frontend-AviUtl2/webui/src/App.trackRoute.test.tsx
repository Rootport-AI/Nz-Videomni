import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "./bridge/mockBridge";
import type { MockBridgeOptions } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT, TIMELINE_TRACK_PROGRESS_EVENT } from "./bridge";
import type { ResultOf } from "./bridge";
import { apiClient as defaultApiClient } from "./api/client";
import type { StatusResponse } from "./api/types";
import { AppShell } from "./shell/AppShell";
import { writeStoredObjectTracking } from "./shell/objectTrackingSettings";
import { getReservationState, resetProvisionalReservation } from "./timeline/provisionalReservation";

// §3-54 物体追尾 (2026-09-11): the 追尾 right-click, end to end. This is the
// piece no unit test can prove on its own — native event -> route -> §4 guard
// -> tab switch -> remount token -> the RPC going out with the STORED settings
// and `frame = frameStart` -> progress rendering -> the lost-range readout.
//
// Skeleton borrowed from `App.editRoute.test.tsx` (the W0 Edit-系 spine), which
// is the same shape one promotion earlier.
//
// The panel's own behaviour (each control, the greying, the error wording) is
// `modes/toolbox/ObjectTrackingSection.test.tsx`'s job and is not repeated here.

/** A selected 部分フィルタ spanning frames 100..340 on layer 4.
 *
 * `filePath: null` is not laziness — a 部分フィルタ has no backing file, so
 * native really does report null, which means `resolveMenuSelection` re-queries
 * `timeline.getSelection` before the guard runs. That is exactly what the real
 * flow does (the `menuInvoked` snapshot carries no `effectName` either), so the
 * fixture below is ALSO handed to the mock bridge as its `selection`, making the
 * re-query answer the same object. */
function partialFilterSelection(
  overrides: Partial<ResultOf<"timeline.getSelection">> = {},
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [
      {
        layer: 4,
        frameStart: 100,
        frameEnd: 340,
        effectName: "部分フィルタ",
        filePath: null,
        objectName: "mosaic",
        textContent: null,
        mediaWidth: 0,
        mediaHeight: 0,
        mediaDurationSec: 0,
      },
    ],
    // A playback cursor sitting somewhere else entirely, so a route that used
    // it instead of `frameStart` would be caught rather than coincidentally
    // right.
    cursorFrame: 250,
    cursorLayer: 9,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
    ...overrides,
  };
}

/** A selected VIDEO — for the §4 guard case. */
function videoSelection(): ResultOf<"timeline.getSelection"> {
  return {
    ...partialFilterSelection(),
    selected: [
      {
        layer: 3,
        frameStart: 10,
        frameEnd: 130,
        effectName: "動画ファイル",
        filePath: "C:\\v\\a.mp4",
        objectName: "a",
        textContent: null,
        mediaWidth: 1920,
        mediaHeight: 1080,
        mediaDurationSec: 5,
      },
    ],
  };
}

const panel = () => screen.getByRole("tabpanel");

/** A minimal `/status` body carrying the given `tracking` block. Only the
 * fields `useServerStatus` and the header badge read have to be real. */
function statusWithTracking(tracking: NonNullable<StatusResponse["tracking"]>): StatusResponse {
  return {
    server: "running",
    version: "0.4.0-stub",
    host: "127.0.0.1",
    port: 18620,
    pipeline_loaded: true,
    pipeline_type: "distilled",
    state: "ready",
    gpu: { available: true, name: "stub", vram_total_mb: 1, vram_used_mb: 0, vram_free_mb: 1 },
    queue: { mode: "single_job_in_memory", pending: 0, running: 0, completed: 0, failed: 0 },
    tracking,
  };
}

async function renderShell(bridgeOptions: MockBridgeOptions = {}) {
  const bridge = createMockBridge({
    delayMs: 0,
    selection: partialFilterSelection(),
    ...bridgeOptions,
  });
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^(generate|busy…)$/i }, { timeout: 5_000 });
  // `trackingAvailable` is derived from the `/status` poll, so the route must
  // not be fired before that poll has landed — otherwise the shell would refuse
  // with the "not available" note for a reason that has nothing to do with what
  // is being tested. The header badge IS that poll's arrival.
  await screen.findByText(/server online/i, undefined, { timeout: 5_000 });
  return bridge;
}

function emit(
  bridge: ReturnType<typeof createMockBridge>,
  action: string,
  selection: ResultOf<"timeline.getSelection">,
) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action, selection });
  });
}

/** Resolves once the Toolbox screen is the visible mode panel. The tracking
 * heading only exists there, and a `hidden` panel is out of the accessibility
 * tree — so finding it by ROLE is the "we're on Toolbox now" signal. */
async function waitForToolboxPanel(): Promise<HTMLElement> {
  await screen.findByRole("heading", { name: /object tracking/i }, { timeout: 5_000 });
  return panel();
}

describe("App / §3-54 物体追尾 right-click routing", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it(
    "opens the Toolbox tab and fires trackObject with the STORED settings and frame = frameStart",
    async () => {
      // A set of values that is different from the defaults in every field, so
      // "the RPC carried the stored settings" cannot pass by coincidence.
      writeStoredObjectTracking({
        searchFactor: 5.5,
        lostScoreThreshold: 0.6,
        lostBehavior: "continue",
        smoothing: 0.75,
        followSize: false,
        keyframeStride: 4,
      });
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "trackObject", partialFilterSelection());

      const visible = await waitForToolboxPanel();
      expect(within(visible).getByRole("heading", { name: /object tracking/i })).toBeInTheDocument();
      // Create is hidden along with its panel (a hidden subtree is out of the
      // accessibility tree, so reaching nothing is the assertion).
      expect(screen.queryByRole("button", { name: /get size from aviutl2/i })).not.toBeInTheDocument();

      await waitFor(() => {
        expect(spy.mock.calls.some(([m]) => m === "timeline.trackObject")).toBe(true);
      });
      const call = spy.mock.calls.find(([m]) => m === "timeline.trackObject");
      expect(call?.[1]).toEqual({
        layer: 4,
        // THE INVARIANT: the object's own head (100), never the playback
        // cursor (250).
        frame: 100,
        searchFactor: 5.5,
        smoothing: 0.75,
        followSize: false,
        lostScoreThreshold: 0.6,
        lostBehavior: "continue",
        keyframeStride: 4,
      });
    },
    20_000,
  );

  it(
    "falls back to the defaults when nothing has been stored",
    async () => {
      // Nothing was seeded, so the shell starts from
      // `OBJECT_TRACKING_DEFAULTS` — which is what must reach the wire.
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "trackObject", partialFilterSelection());
      await waitForToolboxPanel();

      await waitFor(() => {
        expect(spy.mock.calls.some(([m]) => m === "timeline.trackObject")).toBe(true);
      });
      const call = spy.mock.calls.find(([m]) => m === "timeline.trackObject");
      expect(call?.[1]).toMatchObject({
        searchFactor: 4,
        lostScoreThreshold: 0.35,
        lostBehavior: "hold",
        smoothing: 0.3,
        followSize: true,
        keyframeStride: 1,
      });
    },
    20_000,
  );

  it(
    "renders the progress pushes that arrive while the RPC is still pending",
    async () => {
      const bridge = await renderShell({
        trackProgressFrames: [
          { frame: 100, index: 1, total: 241, score: 0.95, lost: false },
          { frame: 220, index: 121, total: 241, score: 0.42, lost: true },
        ],
      });
      emit(bridge, "trackObject", partialFilterSelection());
      await waitForToolboxPanel();

      await waitFor(() => {
        expect(screen.getByText(/121 \/ 241 frames/)).toBeInTheDocument();
      });
    },
    20_000,
  );

  it(
    "lists the lost ranges as the AviUtl2 frame numbers native sent (contract v12: absolute)",
    async () => {
      const bridge = await renderShell({
        trackObjectResult: {
          frames: 241,
          keyframes: 241,
          elapsedMs: 12_300,
          // Already absolute on the wire — the object's head is 100, so a
          // WebUI that "helpfully" added it again would show 320–334 here.
          lostRanges: [{ start: 220, end: 234 }],
        },
      });
      emit(bridge, "trackObject", partialFilterSelection());
      await waitForToolboxPanel();

      await waitFor(() => {
        expect(screen.getByText("220–234")).toBeInTheDocument();
      });
      expect(screen.getByText(/tracked 241 frames and wrote 241 keyframes in 12\.3s/i)).toBeInTheDocument();
    },
    20_000,
  );

  it(
    "reserves NOTHING: 追尾 edits an object that already exists",
    async () => {
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "trackObject", partialFilterSelection());

      // Settle on a POSITIVE signal (the tab switch) before asserting the
      // absence, so this cannot pass merely by out-running the route.
      await waitForToolboxPanel();
      expect(spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")).toBeUndefined();
      expect(getReservationState().phase).toBe("idle");
    },
    20_000,
  );

  it(
    "is refused with the §4 note when the selection is a video, not a 部分フィルタ",
    async () => {
      const bridge = await renderShell({ selection: videoSelection() });
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "trackObject", videoSelection());

      // Guidance only — the §4 contract's "破壊的な動作をしない".
      await screen.findByText(/this action requires a partial filter/i, undefined, {
        timeout: 5_000,
      });
      // Still on Create: no tab switch, and nothing was tracked.
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: /object tracking/i })).not.toBeInTheDocument();
      expect(spy.mock.calls.find(([m]) => m === "timeline.trackObject")).toBeUndefined();
    },
    20_000,
  );

  it(
    "is refused with a note — and no tab switch — on a server that cannot track",
    async () => {
      // `AppShell` polls `/status` through the app-wide singleton client, not
      // the injected bridge (`useServerStatus` takes neither), so the fixture
      // knob cannot reach it here — the singleton is stubbed instead. That is
      // also the honest shape of what is being tested: the shell's refusal
      // depends on the status it actually polled.
      vi.spyOn(defaultApiClient, "getStatus").mockResolvedValue(
        statusWithTracking({ available: false, reason: "not installed" }),
      );
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "trackObject", partialFilterSelection());

      await screen.findByText(/object tracking is not available on this server/i, undefined, {
        timeout: 5_000,
      });
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: /object tracking/i })).not.toBeInTheDocument();
      expect(spy.mock.calls.find(([m]) => m === "timeline.trackObject")).toBeUndefined();
    },
    20_000,
  );

  it(
    "does NOT remount on a second 追尾 right-click while a run is going — it shows guidance and leaves the run alone",
    async () => {
      // Owner gate 2026-09-11. Before this rule the second right-click bumped
      // the remount token: the running run's progress subscription and its
      // pending promise went with the old screen, so the readout froze at
      // whatever number it had reached and the fresh panel showed native's
      // `TRACK_BUSY` under a "Tracking failed" heading — while the run itself
      // ran to completion, unaffected. Now the shell refuses the second click
      // before any of that, and the FIRST panel keeps watching its own run.
      const bridge = await renderShell();
      const original = bridge.request.bind(bridge);
      let trackCalls = 0;
      // Run #1 is modelled as a call that never settles: it is still going when
      // the second right-click lands.
      const spy = vi.spyOn(bridge, "request").mockImplementation((method, params) => {
        if (method === "timeline.trackObject") {
          trackCalls += 1;
          return new Promise(() => {}) as ReturnType<typeof original>;
        }
        return original(method, params);
      });

      emit(bridge, "trackObject", partialFilterSelection());
      await waitForToolboxPanel();
      // Run #1 is genuinely mid-flight: a progress push lands and is shown.
      act(() => {
        bridge.emit(TIMELINE_TRACK_PROGRESS_EVENT, {
          frame: 160,
          index: 61,
          total: 241,
          score: 0.9,
          lost: false,
        });
      });
      await screen.findByText(/61 \/ 241 frames/, undefined, { timeout: 5_000 });

      // The second right-click, while that run is still going.
      emit(bridge, "trackObject", partialFilterSelection());
      await screen.findByText(/object tracking is already running/i, undefined, {
        timeout: 5_000,
      });

      // Nothing was re-fired: one RPC for one run.
      expect(trackCalls).toBe(1);
      // The first run's subscription is still attached — a LATER push (one that
      // a remount would have orphaned) still reaches the readout.
      act(() => {
        bridge.emit(TIMELINE_TRACK_PROGRESS_EVENT, {
          frame: 200,
          index: 101,
          total: 241,
          score: 0.9,
          lost: false,
        });
      });
      await screen.findByText(/101 \/ 241 frames/, undefined, { timeout: 5_000 });
      // And the panel reports no failure: the refusal is guidance, not an error
      // (`NoteArea` is `role="status"`, so an alert here could only be the
      // panel's own error box).
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();

      // Stop is what the note tells the user to press, so it must still reach
      // the run that is going.
      const stop = screen.getByRole("button", { name: /stop/i });
      expect(stop).toBeEnabled();
      await act(async () => {
        stop.click();
      });
      await waitFor(() => {
        expect(spy.mock.calls.some(([m]) => m === "timeline.cancelTracking")).toBe(true);
      });
    },
    20_000,
  );

  it(
    "rounds a fractional keyframe interval — native rejects a fractional stride outright",
    async () => {
      const bridge = await renderShell();
      await act(async () => {
        screen.getByRole("tab", { name: "Toolbox" }).click();
      });
      await waitForToolboxPanel();
      // The pair is [slider, number box]; only the box can be typed to a value
      // its `step` does not allow.
      const box = screen.getAllByLabelText(/keyframe interval/i)[1] as HTMLInputElement;
      fireEvent.change(box, { target: { value: "1.5" } });
      // The shell owns the settings, so the rounded value comes back down as
      // the box's own displayed value.
      await waitFor(() => {
        expect(box).toHaveValue(2);
      });
      expect(bridge).toBeDefined();
    },
    20_000,
  );

  it(
    "starts nothing when the Toolbox tab is opened by hand",
    async () => {
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      await act(async () => {
        screen.getByRole("tab", { name: "Toolbox" }).click();
      });
      await waitForToolboxPanel();
      expect(spy.mock.calls.find(([m]) => m === "timeline.trackObject")).toBeUndefined();
      expect(screen.getByText(/nothing is being tracked/i)).toBeInTheDocument();
    },
    20_000,
  );
});
