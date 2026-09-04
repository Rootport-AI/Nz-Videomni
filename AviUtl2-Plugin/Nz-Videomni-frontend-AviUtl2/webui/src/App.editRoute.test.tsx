import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "./bridge/mockBridge";
import type { MockBridgeOptions } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { ResultOf } from "./bridge";
import { apiClient as defaultApiClient } from "./api/client";
import type { ApiClient } from "./api/client";
import type { JobResponse } from "./api/types";
import { AppShell } from "./shell/AppShell";
import { ACCELERATION_STORAGE_KEY } from "./shell/accelerationSettings";
import { getReservationState, resetProvisionalReservation } from "./timeline/provisionalReservation";
import { withExtraUnsupportedFeatures } from "./test/unsupportedFeatures";

// W0 (共通スパイン, 2026-08-09): the two Edit-系 right-click commands
// (`outpaintVideo` / `retakeRange`) are the first ones to route to the Edit
// tab. These app-level tests cover the SPINE end to end — the piece no unit
// test can prove on its own: native event -> route -> guard -> tab switch ->
// remount token -> sub-tab auto-selection -> placement系統 D on the wire.
//
// Nothing here asserts on panel CONTENT beyond which sub-tab is showing — the
// Outpainting panel's own behaviour is covered by
// `modes/edit/OutpaintingPanel.test.tsx`, and Retake is still a placeholder.

/** A video object selection, with the frame range controllable per test. */
function videoSelection(
  overrides: Partial<ResultOf<"timeline.getSelection">> = {},
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: 48,
    rangeEnd: 96,
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
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
    ...overrides,
  };
}

/** The single visible mode panel. */
const panel = () => screen.getByRole("tabpanel");

/** A ledger with exactly one job in the given non-terminal state, served
 * through a stub `ApiClient` — the shape `App.reservationRelease.test.tsx`
 * already uses. Everything else falls through to the real client (which never
 * gets called here, since the shell only polls `GET /jobs`). */
function runningJobClient(): ApiClient {
  const job: JobResponse = {
    job_id: "job-run",
    status: "running",
    progress: null,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    created_at: "2026-08-10T00:00:00Z",
    started_at: "2026-08-10T00:00:01Z",
    completed_at: null,
    error: null,
    is_v2v: false,
    joined: false,
    request: {},
    result: null,
  };
  return { ...defaultApiClient, listJobs: vi.fn(async () => [job]) } as unknown as ApiClient;
}

async function renderShell(apiClient?: ApiClient, bridgeOptions: MockBridgeOptions = {}) {
  const bridge = createMockBridge({ delayMs: 0, ...bridgeOptions });
  render(<AppShell nativeBridge={bridge} {...(apiClient ? { apiClient } : {})} />);
  // Create finished seeding from GET /config => the shell is mounted and the
  // menu router is subscribed. The label is "Busy…" instead of "Generate" when
  // the stub ledger holds a running job (the busy tests below), so match either
  // — what is being waited for is the button's EXISTENCE, not its wording.
  await screen.findByRole("button", { name: /^(generate|busy…)$/i }, { timeout: 5_000 });
  return bridge;
}

/** Fire a native menu event.
 *
 * `AppShell.handleRoute` is ASYNC — it awaits `reconcileFromTimeline` (§6-b)
 * before any tab switch, and its `placeProvisional` is fire-and-forget on top
 * of that. The mock bridge resolves each RPC through a `setTimeout` (a MACROtask
 * even at `delayMs: 0`), so flushing microtasks cannot settle a route: every
 * assertion below is therefore reached through `waitFor`/`findBy*`, which polls
 * across real timer ticks.
 */
function emit(
  bridge: ReturnType<typeof createMockBridge>,
  action: string,
  selection: ResultOf<"timeline.getSelection">,
) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action, selection });
  });
}

/** Resolves once the Edit screen is the visible mode panel — i.e. the route ran
 * to completion and switched tabs. Edit's sub-tabs only exist inside its own
 * screen, so finding one IS the "we're on Edit now" signal. */
async function waitForEditPanel(): Promise<HTMLElement> {
  await screen.findByRole("tab", { name: "Outpainting" }, { timeout: 5_000 });
  return panel();
}

describe("App / W0 Edit-系 right-click routing", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "outpaintVideo opens the Edit tab on the Outpainting sub-tab",
    async () => {
      const bridge = await renderShell();
      emit(bridge, "outpaintVideo", videoSelection());

      const visible = await waitForEditPanel();
      expect(within(visible).getByRole("tab", { name: "Outpainting" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
      expect(within(visible).getByRole("tab", { name: "Retake" })).toHaveAttribute(
        "aria-selected",
        "false",
      );
      // Create is hidden along with its panel. Probed through Create's OWN
      // "Get size from AviUtl2" button rather than a generic Generate button:
      // the Outpainting panel grew a Generate button of its own
      // (Docs/PENDING_TASKS_CLOSED.md §3-70, filed as §1-13 at the time,
      // 2026-08-09), so "no Generate anywhere" stopped being a Create-specific
      // signal. A `hidden` panel is out of the accessibility tree, so a plain
      // `queryByRole` reaching nothing is exactly the assertion wanted.
      expect(screen.queryByRole("button", { name: /get size from aviutl2/i })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "retakeRange opens the Edit tab on the Retake sub-tab",
    async () => {
      const bridge = await renderShell();
      emit(bridge, "retakeRange", videoSelection());

      const visible = await waitForEditPanel();
      expect(within(visible).getByRole("tab", { name: "Retake" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
      expect(within(visible).getByRole("tab", { name: "Outpainting" })).toHaveAttribute(
        "aria-selected",
        "false",
      );
    },
    20_000,
  );

  it(
    "retakeRange reserves the provisional at the SELECTED RANGE start (系統D -> B)",
    async () => {
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", videoSelection({ rangeStart: 48, rangeEnd: 96 }));

      await waitFor(
        () => {
          expect(spy.mock.calls.some(([m]) => m === "timeline.insertProvisional")).toBe(true);
        },
        { timeout: 5_000 },
      );
      const insert = spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")!;
      // "D" must never reach the wire, and the frame must be the RANGE start
      // (48) — NOT the object's own frameStart (10).
      expect(insert[1]).toMatchObject({ placement: "B", materialFrameStart: 48 });
      await waitFor(() => {
        expect(getReservationState().phase).toBe("reserved");
      });
    },
    20_000,
  );

  it(
    "outpaintVideo reserves NOTHING (placement null — W2 owns the geometry)",
    async () => {
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "outpaintVideo", videoSelection());

      // Settle on a POSITIVE signal (the tab switch) before asserting the
      // absence, so this can't pass merely by out-running the route.
      await waitForEditPanel();
      expect(spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")).toBeUndefined();
      expect(getReservationState().phase).toBe("idle");
    },
    20_000,
  );

  it(
    "retakeRange with NO range selected is refused: note shown, no tab switch, no reservation",
    async () => {
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", videoSelection({ hasRange: false }));

      // The §4 guard's "破壊的な動作をしない" contract: guidance only. The note is
      // the positive signal the refusal actually ran.
      await screen.findByText(/select the range you want to redo/i, undefined, { timeout: 5_000 });
      // Still on Create — no tab switch happened.
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Outpainting" })).not.toBeInTheDocument();
      expect(spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")).toBeUndefined();
      expect(getReservationState().phase).toBe("idle");
    },
    20_000,
  );

  // ① busy ガード（オーナー目視 2026-08-10）。`retakeRange` **だけ**を塞ぐ:
  // Step 5 の共通ガード（予約席）に足すと素材系/カーソル系まで巻き添えになる。
  it(
    "retakeRange is refused while a job is running: note shown, no tab switch, no reservation",
    async () => {
      const bridge = await renderShell(runningJobClient());
      // 台帳にその走行中ジョブが届いてから（= `jobsRef` が更新されてから）叩く。
      await within(panel()).findByText(/jobs \(1\)/i, undefined, { timeout: 5_000 });
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", videoSelection());

      // 既存の「予約席が埋まっている」ノートを流用している（新しい文言は作らない）。
      await screen.findByText(/A new reservation cannot be made/i, undefined, { timeout: 5_000 });
      // Create のまま（タブ切替が起きていない）。走行中なので Create のボタンは
      // busy ラベルになっている。
      expect(screen.getByRole("button", { name: /^busy…$/i })).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Outpainting" })).not.toBeInTheDocument();
      expect(spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")).toBeUndefined();
      expect(getReservationState().phase).toBe("idle");
    },
    20_000,
  );

  it(
    "a non-Retake right-click is NOT blocked by a running job (the guard is retakeRange-only)",
    async () => {
      const bridge = await renderShell(runningJobClient());
      await within(panel()).findByText(/jobs \(1\)/i, undefined, { timeout: 5_000 });
      emit(bridge, "outpaintVideo", videoSelection());

      const visible = await waitForEditPanel();
      expect(within(visible).getByRole("tab", { name: "Outpainting" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
    },
    20_000,
  );

  it(
    "an Edit-系 command on a non-video is refused by the existing type guard",
    async () => {
      const bridge = await renderShell();
      const audio = videoSelection();
      audio.selected[0]!.effectName = "音声ファイル";
      emit(bridge, "outpaintVideo", audio);

      await screen.findByText(/requires a video/i, undefined, { timeout: 5_000 });
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
    },
    20_000,
  );
});

// §1-27 (2026-09-05): the SPINE gap this ledger item exists to close —
// `AppShell` never threaded its shared `acceleration` prop down into
// `EditScreen`, so every job Edit submitted ran the server-default config no
// matter what Settings held. Every other unit test in this file/`modes/edit/`
// stubs its own bridge or renders `EditScreen` directly, so none of them can
// see whether the SHELL actually wires the prop through — this is that one
// test, rendering `AppShell` for real like the routing tests above and
// checking the wire the way `App.prefill.test.tsx`'s `pinAllOnAcceleration`
// and `EditScreen.retake.test.tsx`'s `chainBody` do for their own layers.
describe("App / Edit screens receive Settings' acceleration (§1-27)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  /** `videoSelection()` above (span 4.03s inside a 5s file, no playback range
   * reported) is exactly what the OTHER tests in this file want — it never
   * clears `decideSourceTrim`'s `unknownPlaybackPosition` gate, so
   * `useRetakeForm`'s `mapping` never resolves and Generate stays disabled
   * forever. Those tests only assert on the ROUTE (tab switch / reservation),
   * so that never mattered before. This test clicks Generate, so it needs a
   * selection the form can actually turn into a window — same numbers
   * `useRetakeForm.test.tsx`'s own `makeSelection()` uses (a reported playback
   * range 2.0-12.0s inside a 20s file), proven there to resolve to a valid
   * 113-frame window. */
  function retakeReadySelection(): ResultOf<"timeline.getSelection"> {
    return {
      hasRange: true,
      rangeStart: 60,
      rangeEnd: 209, // 150 frames = 5.0s @30fps
      selected: [
        {
          layer: 3,
          frameStart: 0,
          frameEnd: 299,
          effectName: "動画ファイル",
          filePath: "C:\\v\\retake-take1.mp4",
          objectName: "take1",
          textContent: null,
          mediaWidth: 1280,
          mediaHeight: 768,
          mediaDurationSec: 20,
          hasPlaybackRange: true,
          playbackStartSec: 2,
          playbackEndSec: 12,
        },
      ],
      cursorFrame: 60,
      cursorLayer: 3,
      rate: 30,
      scale: 1,
      sampleRate: 44100,
    };
  }

  /** Pins ONLY `attentionBackend` off its server default. `readStoredAcceleration`
   * falls back per-field to `STORED_DEFAULTS` for whatever is absent from the
   * JSON blob (its own doc comment), so this is "sage, everything else
   * untouched" — unlike `App.prefill.test.tsx`'s `pinAllOnAcceleration`, which
   * pins all five for a different purpose (matching the comfort marker's
   * `requires` row) this test has no need of. Written BEFORE render:
   * `useAccelerationSettings`'s lazy initializer reads `localStorage` once, on
   * mount. */
  function pinSageAcceleration(): void {
    window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify({ attentionBackend: "sage" }));
  }

  it(
    "retakeRange's POST /generate/chain carries the pinned acceleration (attention_backend: sage)",
    async () => {
      pinSageAcceleration();
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", retakeReadySelection());
      await waitForEditPanel();

      const generate = await screen.findByRole(
        "button",
        { name: /redo the selected range/i },
        { timeout: 5_000 },
      );
      await waitFor(() => expect(generate).toBeEnabled(), { timeout: 5_000 });
      await userEvent.setup().click(generate);

      const isChain = ([method, params]: [string, unknown]) =>
        method === "backend.request" && (params as { path?: string }).path?.endsWith("/generate/chain") === true;
      await waitFor(() => expect(spy.mock.calls.some((c) => isChain(c as [string, unknown]))).toBe(true), {
        timeout: 5_000,
      });
      const call = spy.mock.calls.find((c) => isChain(c as [string, unknown])) as
        | [string, { body: Record<string, unknown> }]
        | undefined;
      if (!call) throw new Error("no POST /generate/chain call was made");
      expect(call[1].body.attention_backend).toBe("sage");
    },
    20_000,
  );
});

// §3-98 P5 — the OTHER way into a mode whose tab is greyed.
//
// `ModeTabs` greys a mode the loaded base model's engine cannot run, and the
// bounce effect gets the user out of one they were already on. Neither reaches
// the timeline's own context menu: AviUtl2 draws that, and every Edit-系 item is
// still on it whatever this app has greyed. Before the Step 0 gate, such a route
// ran the whole of `handleRoute` — `reservePlacement` wrote a ⏳ provisional onto
// the timeline and `setMode("edit")` opened the tab, and only THEN did the
// bounce effect fire and put the user back on Single. The tab switch was undone;
// the object on the timeline was not. It stayed there, bound to nothing, with
// nothing to ever clean it up.
//
// The fixture reaches this state through `withExtraUnsupportedFeatures`, which
// declares BOTH `retake` and `outpaint` for LTX 2.5 so `disabledModesFor`'s
// `needsAnyOf` takes the whole Edit tab. Both names are synthetic now — the
// engine runs both modes.
/** Wraps a fixture bridge so a base model declares extra `unsupported_features`
 * — see the helper's own note for why these tests need it. */
const AS_LTX25: MockBridgeOptions = {
  ltx25Install: "full",
  supportedBaseModels: ["LTX23", "LTX25"],
};

describe("App / Edit-系 right-click on a base model that cannot run Edit", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  /** Render, switch the header dropdown to LTX 2.5, and wait for the Edit tab to
   * actually go grey — that greying is the positive signal the new
   * `unsupported_features` list has landed, so nothing below can pass merely by
   * out-running the switch.
   *
   * THE FIXTURE'S OWN LTX 2.5 NO LONGER QUALIFIES — and since the Outpainting
   * increment it does not supply EITHER half. Edit greys only when BOTH of its
   * sub-modes are refused; the Retake increment gave the engine 撮り直し and the
   * Outpainting increment gave it 画角拡張, so LTX 2.5 now greys neither the tab
   * nor a sub-tab. What these tests are about is the ROUTE GATE, which keys on
   * `disabledModes`, so the base model is given BOTH names
   * (`withExtraUnsupportedFeatures`) rather than the tests being re-pointed at
   * whatever LTX 2.5 happens to refuse this month. That is the whole bargain of
   * the helper: the MECHANISM is under test here, and the real list is asserted
   * where it belongs — `bridge/mockBridge.test.ts`, against the server's own. */
  async function renderOnLtx25() {
    const bridge = withExtraUnsupportedFeatures(
      createMockBridge({ delayMs: 0, ...AS_LTX25 }),
      "LTX25",
      ["retake", "outpaint"],
    );
    render(<AppShell nativeBridge={bridge} />);
    await screen.findByRole("button", { name: /^(generate|busy…)$/i }, { timeout: 5_000 });
    const select = (await screen.findByRole("combobox", { name: /base model/i })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("LTX23"));
    await userEvent.setup().selectOptions(select, "LTX25");
    await waitFor(() => expect(select.value).toBe("LTX25"));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled());
    return bridge;
  }

  it(
    "refuses retakeRange BEFORE any provisional is written",
    async () => {
      const bridge = await renderOnLtx25();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", videoSelection());

      // The note is the positive signal the refusal actually ran (the §4 guard
      // contract's shape: guidance only, nothing destructive).
      await screen.findByText(/This command cannot be run on the selected base model/i, undefined, {
        timeout: 5_000,
      });

      // The heart of it: NO provisional on the timeline. `insertProvisional` is
      // the RPC `reservePlacement` issues, and the seat is still idle.
      expect(spy.mock.calls.find(([m]) => m === "timeline.insertProvisional")).toBeUndefined();
      expect(getReservationState().phase).toBe("idle");

      // …and no tab switch: still on Single, and Edit never opened (its sub-tabs
      // only exist inside its own screen, so their absence IS that assertion).
      expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByRole("tab", { name: "Outpainting" })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "refuses outpaintVideo too — the gate is the MODE, not the one route that reserves",
    async () => {
      // `outpaintVideo` has `placement: null`, so it never had a provisional to
      // leave behind; what it did have was the tab switch and the remount. Both
      // must be gone as well, or the user lands on a form whose every submission
      // comes back 422.
      const bridge = await renderOnLtx25();
      emit(bridge, "outpaintVideo", videoSelection());

      await screen.findByText(/This command cannot be run on the selected base model/i, undefined, {
        timeout: 5_000,
      });
      expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByRole("tab", { name: "Outpainting" })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "leaves a route to a mode the SAME base model CAN run completely alone",
    async () => {
      // The corollary, and the reason the gate keys on `disabledModes` rather
      // than on the action name: LTX 2.5 runs Single and Chained perfectly well.
      // A gate that refused every right-click on a restricted base model would
      // pass all three assertions above and still be wrong.
      //
      // ✨ `textToVideoHere` is the probe: `targetMode: "single"`, placement C at
      // the cursor — so it both switches nothing (already on Single) and DOES
      // reserve, which is precisely the write the two tests above assert is
      // absent.
      const bridge = await renderOnLtx25();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "textToVideoHere", videoSelection({ cursorLayer: 2, cursorFrame: 60 }));

      await waitFor(
        () => {
          expect(spy.mock.calls.some(([m]) => m === "timeline.insertProvisional")).toBe(true);
        },
        { timeout: 5_000 },
      );
      await waitFor(() => expect(getReservationState().phase).toBe("reserved"));
      expect(
        screen.queryByText(/This command cannot be run on the selected base model/i),
      ).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "still routes Edit-系 normally on LTX 2.3 (the ordinary case is untouched)",
    async () => {
      // The regression guard for the gate itself: an unrestricted base model
      // must behave exactly as it did before Step 0 existed.
      const bridge = await renderShell();
      const spy = vi.spyOn(bridge, "request");
      emit(bridge, "retakeRange", videoSelection());

      await screen.findByRole("tab", { name: "Outpainting" }, { timeout: 5_000 });
      await waitFor(() => {
        expect(spy.mock.calls.some(([m]) => m === "timeline.insertProvisional")).toBe(true);
      });
    },
    20_000,
  );
});
