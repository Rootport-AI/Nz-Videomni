import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import type { ResultOf } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { OBJECT_TRACKING_DEFAULTS } from "../../shell/objectTrackingSettings";
import type { ObjectTrackingSettings } from "../../shell/objectTrackingSettings";
import { ObjectTrackingSection } from "./ObjectTrackingSection";
import type { ObjectTrackRequest } from "./useObjectTracking";

// §3-54 物体追尾 (2026-09-11). The panel's own behaviour, in isolation from the
// right-click spine (`App.trackRoute.test.tsx` covers that end to end): the
// seven controls, the mock model radio, what a run in flight does to them, the
// stop button, the unavailable state, and how a failure is worded.

/** A single selected 部分フィルタ spanning frames 100..340 on layer 4. */
function partialFilterSelection(): ResultOf<"timeline.getSelection"> {
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
    cursorFrame: 7,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

function trackRequest(): ObjectTrackRequest {
  return { selection: partialFilterSelection() };
}

interface RenderOptions {
  trackingAvailable?: boolean;
  trackingReason?: string;
  request?: ObjectTrackRequest | undefined;
  settings?: Partial<ObjectTrackingSettings>;
  bridgeOptions?: MockBridgeOptions;
  /** Controlled clock for the rate readout. */
  now?: () => number;
}

function renderSection(options: RenderOptions = {}) {
  const bridge = createMockBridge({ delayMs: 0, ...options.bridgeOptions });
  const handlers = {
    onSearchFactorChange: vi.fn(),
    onLostScoreThresholdChange: vi.fn(),
    onLostBehaviorChange: vi.fn(),
    onSmoothingChange: vi.fn(),
    onFollowSizeChange: vi.fn(),
    onKeyframeStrideChange: vi.fn(),
  };
  render(
    <LanguageProvider>
      <ObjectTrackingSection
        trackingAvailable={options.trackingAvailable ?? true}
        trackingReason={options.trackingReason}
        trackRequest={options.request}
        nativeBridge={bridge}
        settings={{ ...OBJECT_TRACKING_DEFAULTS, ...options.settings }}
        now={options.now}
        {...handlers}
      />
    </LanguageProvider>,
  );
  return { bridge, handlers };
}

const control = (name: RegExp | string) => screen.getAllByLabelText(name);

describe("ObjectTrackingSection — the seven settings", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("renders each numeric knob as a slider AND a number box on one value", () => {
    renderSection({ settings: { searchFactor: 4, smoothing: 0.3 } });
    for (const label of [/search area/i, /lost threshold/i, /smoothness/i, /keyframe interval/i]) {
      const pair = control(label);
      expect(pair).toHaveLength(2);
      expect(pair[0]).toHaveAttribute("type", "range");
      expect(pair[1]).toHaveAttribute("type", "number");
    }
  });

  it("gives each slider the range the contract accepts", () => {
    renderSection();
    const [search] = control(/search area/i);
    expect(search).toHaveAttribute("min", "2");
    expect(search).toHaveAttribute("max", "6");
    expect(search).toHaveAttribute("step", "0.1");
    const [threshold] = control(/lost threshold/i);
    expect(threshold).toHaveAttribute("min", "0");
    expect(threshold).toHaveAttribute("max", "1");
    expect(threshold).toHaveAttribute("step", "0.01");
    const [smoothing] = control(/smoothness/i);
    expect(smoothing).toHaveAttribute("step", "0.05");
    const [stride] = control(/keyframe interval/i);
    expect(stride).toHaveAttribute("min", "1");
    expect(stride).toHaveAttribute("max", "10");
    expect(stride).toHaveAttribute("step", "1");
  });

  it("shows the stored values, not the defaults", () => {
    renderSection({
      settings: {
        searchFactor: 5.5,
        lostScoreThreshold: 0.6,
        smoothing: 0.75,
        keyframeStride: 4,
        followSize: false,
        lostBehavior: "continue",
      },
    });
    // The NUMBER box is asserted on (a range input reports its value as a
    // string), and the slider beside it is proven to carry the same one.
    expect(control(/search area/i)[1]).toHaveValue(5.5);
    expect(control(/search area/i)[0]).toHaveValue("5.5");
    expect(control(/lost threshold/i)[1]).toHaveValue(0.6);
    expect(control(/smoothness/i)[1]).toHaveValue(0.75);
    expect(control(/keyframe interval/i)[1]).toHaveValue(4);
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByRole("radio", { name: /keep following anyway/i })).toBeChecked();
  });

  // `fireEvent.change` rather than `userEvent`: these are CONTROLLED inputs
  // whose value comes from props the test holds still (the setters are spies),
  // so `userEvent.type` would append to a box that never updates. What is being
  // asserted is the change handler's arithmetic, not the browser's typing.
  it("reports a slider move to its setter", () => {
    const { handlers } = renderSection();
    fireEvent.change(control(/keyframe interval/i)[0] as HTMLInputElement, {
      target: { value: "3" },
    });
    expect(handlers.onKeyframeStrideChange).toHaveBeenCalledWith(3);
  });

  it("reports a number-box edit to the SAME setter as its slider", () => {
    const { handlers } = renderSection();
    fireEvent.change(control(/search area/i)[1] as HTMLInputElement, {
      target: { value: "5.2" },
    });
    expect(handlers.onSearchFactorChange).toHaveBeenCalledWith(5.2);
  });

  it("clamps a number box typed past the maximum", () => {
    const { handlers } = renderSection({ settings: { keyframeStride: 1 } });
    fireEvent.change(control(/keyframe interval/i)[1] as HTMLInputElement, {
      target: { value: "99" },
    });
    // Never above the slider's own ceiling, whatever was typed — native would
    // answer `BAD_REQUEST` minutes into a run otherwise.
    expect(handlers.onKeyframeStrideChange).toHaveBeenCalledWith(10);
  });

  it("clamps a number box typed below the minimum", () => {
    const { handlers } = renderSection();
    fireEvent.change(control(/search area/i)[1] as HTMLInputElement, {
      target: { value: "0" },
    });
    expect(handlers.onSearchFactorChange).toHaveBeenCalledWith(2);
  });

  it("ignores an emptied number box rather than sending NaN", () => {
    const { handlers } = renderSection();
    fireEvent.change(control(/smoothness/i)[1] as HTMLInputElement, { target: { value: "" } });
    expect(handlers.onSmoothingChange).not.toHaveBeenCalled();
  });

  it("offers the two lost-behaviour choices with hold selected by default", () => {
    renderSection();
    expect(screen.getByRole("radio", { name: /stay at the last position/i })).toBeChecked();
    expect(screen.getByRole("radio", { name: /keep following anyway/i })).not.toBeChecked();
  });

  it("reports a lost-behaviour choice to its setter", async () => {
    const { handlers } = renderSection();
    await userEvent.click(screen.getByRole("radio", { name: /keep following anyway/i }));
    expect(handlers.onLostBehaviorChange).toHaveBeenCalledWith("continue");
  });

  it("reports the follow-size checkbox to its setter", async () => {
    const { handlers } = renderSection({ settings: { followSize: true } });
    await userEvent.click(screen.getByRole("checkbox"));
    expect(handlers.onFollowSizeChange).toHaveBeenCalledWith(false);
  });

  // The seventh control is a mock: a seat kept for a second tracker.
  it("shows BOTH model radios disabled, with UETrack picked", () => {
    renderSection();
    const uetrack = screen.getByRole("radio", { name: "UETrack" });
    const other = screen.getByRole("radio", { name: "Other" });
    expect(uetrack).toBeDisabled();
    expect(other).toBeDisabled();
    expect(uetrack).toBeChecked();
    expect(other).not.toBeChecked();
  });
});

describe("ObjectTrackingSection — a run", () => {
  // What the RPC is CALLED WITH is not asserted here: the spy can only be
  // attached after the render that already fired it. `App.trackRoute.test.tsx`
  // owns that assertion, where the route is driven by an event and the spy goes
  // on first.
  it("greys every setting while a run is in flight and frees them afterwards", async () => {
    const { bridge } = renderSection({
      request: trackRequest(),
      bridgeOptions: { holdUploads: false, delayMs: 30 },
    });
    // In flight.
    await waitFor(() => {
      expect(control(/search area/i)[0]).toBeDisabled();
    });
    expect(screen.getByRole("radio", { name: /stay at the last position/i })).toBeDisabled();
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: /stop/i })).toBeEnabled();
    // Settled.
    await waitFor(() => {
      expect(control(/search area/i)[0]).toBeEnabled();
    });
    expect(screen.getByRole("button", { name: /stop/i })).toBeDisabled();
    expect(bridge).toBeDefined();
  });

  it("leaves Stop disabled when nothing is running", () => {
    renderSection();
    expect(screen.getByRole("button", { name: /stop/i })).toBeDisabled();
  });

  it("calls timeline.cancelTracking when Stop is pressed", async () => {
    const { bridge } = renderSection({
      request: trackRequest(),
      bridgeOptions: { delayMs: 40 },
    });
    const spy = vi.spyOn(bridge, "request");
    const stop = screen.getByRole("button", { name: /stop/i });
    await waitFor(() => {
      expect(stop).toBeEnabled();
    });
    await userEvent.click(stop);
    await waitFor(() => {
      expect(spy.mock.calls.some(([m]) => m === "timeline.cancelTracking")).toBe(true);
    });
  });

  it("shows the progress count and, once a rate can be measured, the fps", async () => {
    // A controlled clock: the first push at t=0, the second 500ms later having
    // advanced 10 frames -> 20.0 fps.
    const times = [0, 500];
    let i = 0;
    renderSection({
      request: trackRequest(),
      now: () => times[Math.min(i++, times.length - 1)] ?? 0,
      bridgeOptions: {
        trackProgressFrames: [
          { frame: 100, index: 1, total: 241, score: 0.9, lost: false },
          { frame: 110, index: 11, total: 241, score: 0.8, lost: false },
        ],
      },
    });
    await waitFor(() => {
      expect(screen.getByText(/11 \/ 241 frames/)).toBeInTheDocument();
    });
    expect(screen.getByText(/20\.0 fps/)).toBeInTheDocument();
  });

  it("shows an em dash for the rate on the very first push", async () => {
    renderSection({
      request: trackRequest(),
      bridgeOptions: {
        trackProgressFrames: [{ frame: 100, index: 1, total: 241, score: 0.9, lost: false }],
      },
    });
    await waitFor(() => {
      expect(screen.getByText(/1 \/ 241 frames/)).toBeInTheDocument();
    });
    expect(screen.getByText(/—/)).toBeInTheDocument();
  });

  it("lists the lost ranges as the AviUtl2 frame numbers native sent", async () => {
    renderSection({
      request: trackRequest(),
      bridgeOptions: {
        // Contract v12: already ABSOLUTE — the panel adds nothing.
        trackObjectResult: { lostRanges: [{ start: 220, end: 234 }] },
      },
    });
    await waitFor(() => {
      expect(screen.getByText("220–234")).toBeInTheDocument();
    });
  });

  it("says so plainly when nothing was lost", async () => {
    renderSection({ request: trackRequest(), bridgeOptions: { trackObjectResult: { lostRanges: [] } } });
    await waitFor(() => {
      expect(screen.getByText(/no lost frames/i)).toBeInTheDocument();
    });
  });

  it("reports a stopped run's own numbers (a cancel still writes back)", async () => {
    renderSection({
      request: trackRequest(),
      bridgeOptions: {
        trackObjectResult: { cancelled: true, frames: 42, keyframes: 42, elapsedMs: 2_100 },
      },
    });
    await waitFor(() => {
      expect(screen.getByText(/stopped\./i)).toBeInTheDocument();
    });
    expect(screen.getByText(/42 frames were tracked/i)).toBeInTheDocument();
    expect(screen.getByText(/42 keyframes/i)).toBeInTheDocument();
  });

  it("starts nothing without a routed request", async () => {
    const { bridge } = renderSection();
    const spy = vi.spyOn(bridge, "request");
    expect(screen.getByText(/nothing is being tracked/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(spy.mock.calls.some(([m]) => m === "timeline.trackObject")).toBe(false);
    });
  });
});

describe("ObjectTrackingSection — failures", () => {
  it.each([
    ["TRACK_BUSY", /another tracking run is already going/i],
    ["TRACK_SEED_INVALID", /not a usable partial filter/i],
    ["TRACK_WRITEBACK_FAILED", /could not be written to the object/i],
    ["TRACK_FAILED", /stopped partway/i],
    ["TRACK_UNAVAILABLE", /not usable on this server/i],
    ["TRACK_SESSION_NOT_FOUND", /session was not found/i],
    ["TRACK_FRAME_INVALID", /invalid format/i],
    ["NO_EDIT_HANDLE", /no aviutl2 project is open/i],
  ])("words %s in its own sentence", async (code, expected) => {
    renderSection({ request: trackRequest(), bridgeOptions: { trackObjectError: code } });
    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it("shows an unknown code verbatim rather than guessing at it", async () => {
    renderSection({
      request: trackRequest(),
      bridgeOptions: { trackObjectError: "TRACK_SOMETHING_NEW" },
    });
    await waitFor(() => {
      expect(screen.getByText("TRACK_SOMETHING_NEW")).toBeInTheDocument();
    });
  });

  it("shows failures in the panel, NOT in the shared note area", async () => {
    // The section renders standalone here — there is no `NoteArea` in the tree
    // at all — so a failure that reached the screen proves it did not need one.
    renderSection({ request: trackRequest(), bridgeOptions: { trackObjectError: "TRACK_FAILED" } });
    await waitFor(() => {
      expect(screen.getByText(/tracking failed/i)).toBeInTheDocument();
    });
  });
});

describe("ObjectTrackingSection — the server cannot track", () => {
  it("greys every setting and names the batch file", () => {
    renderSection({ trackingAvailable: false, trackingReason: "not installed" });
    expect(control(/search area/i)[0]).toBeDisabled();
    expect(control(/lost threshold/i)[1]).toBeDisabled();
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByRole("radio", { name: /stay at the last position/i })).toBeDisabled();
    expect(screen.getByText(/install-UETrack\.bat/)).toBeInTheDocument();
  });

  it("points at the worker log when the worker is what failed", () => {
    renderSection({ trackingAvailable: false, trackingReason: "worker failed" });
    expect(screen.getByText(/utils_worker\.log/)).toBeInTheDocument();
    expect(screen.queryByText(/install-UETrack\.bat/)).not.toBeInTheDocument();
  });

  it("falls back to the install guidance for an unknown reason", () => {
    renderSection({ trackingAvailable: false, trackingReason: "something new" });
    expect(screen.getByText(/install-UETrack\.bat/)).toBeInTheDocument();
  });

  it("says nothing at all when tracking IS available", () => {
    renderSection({ trackingAvailable: true });
    expect(screen.queryByText(/install-UETrack\.bat/)).not.toBeInTheDocument();
    expect(screen.queryByText(/utils_worker\.log/)).not.toBeInTheDocument();
    expect(control(/search area/i)[0]).toBeEnabled();
  });
});
