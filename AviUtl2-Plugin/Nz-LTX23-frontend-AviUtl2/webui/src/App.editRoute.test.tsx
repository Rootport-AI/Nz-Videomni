import { act, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { ResultOf } from "./bridge";
import { apiClient as defaultApiClient } from "./api/client";
import type { ApiClient } from "./api/client";
import type { JobResponse } from "./api/types";
import { AppShell } from "./shell/AppShell";
import { getReservationState, resetProvisionalReservation } from "./timeline/provisionalReservation";

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

async function renderShell(apiClient?: ApiClient) {
  const bridge = createMockBridge({ delayMs: 0 });
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
      // the Outpainting panel grew a Generate button of its own (§1-13,
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
