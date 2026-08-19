import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMockBridge } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { BridgeMethod, ResultOf } from "./bridge";
import { apiClient as defaultApiClient, BackendApiError } from "./api/client";
import type { ApiClient } from "./api/client";
import type { JobResponse, JobStatus } from "./api/types";
import { AppShell } from "./shell/AppShell";
import { resetProvisionalReservation } from "./timeline/provisionalReservation";

// X2 (予約詰まり修正): the permanent post-422 block. A Chain 422 leaves a stale
// `NzLTX23#<jobId>` tag on the timeline whose job is already terminal in the
// ledger. Before X2(b) `reconcileFromTimeline` resurrected that tag to
// `generating` on every generation-origin right-click, so the busy-guard refused
// every new reservation forever. These integration tests pin that a settled
// tag no longer blocks (and no longer gets deleted), while a genuinely running
// job's tag still does.

function makeJob(job_id: string, status: JobStatus, overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id,
    status,
    progress: status === "completed" ? 100 : null,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    created_at: "2026-07-22T00:00:00Z",
    started_at: null,
    completed_at: status === "completed" ? "2026-07-22T00:01:00Z" : null,
    error: status === "failed" ? "boom" : null,
    is_v2v: false,
    joined: false,
    request: {},
    result: null,
    ...overrides,
  };
}

function fakeApiClient(jobs: JobResponse[]): ApiClient {
  return {
    getStatus: vi.fn(),
    getConfig: vi.fn(),
    generate: vi.fn(),
    generateChain: vi.fn(),
    getJob: vi.fn(),
    listJobs: vi.fn(async () => jobs),
    deleteJob: vi.fn(),
    joinJob: vi.fn(),
    getLoras: vi.fn(),
    reloadLoras: vi.fn(),
    getModels: vi.fn(),
    loadPipeline: vi.fn(),
    unloadPipeline: vi.fn(),
  } as unknown as ApiClient;
}

/** A layer-menu (cursor) selection for the ✨ `textToVideoHere` reservation:
 * no selected object, a valid edit cursor for placement C. */
function cursorSelection(): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [],
    cursorFrame: 100,
    cursorLayer: 3,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

const panel = () => screen.getByRole("tabpanel");

async function waitForJobCount(count: number) {
  await within(panel()).findByText(new RegExp(`jobs \\(${count}\\)`, "i"), undefined, { timeout: 5_000 });
}

async function renderShell(jobs: JobResponse[], orphans: Array<{ jobId: string; layer: number; frame: number }>) {
  const bridge = createMockBridge({ delayMs: 0, provisionalOrphans: orphans });
  render(<AppShell nativeBridge={bridge} apiClient={fakeApiClient(jobs)} />);
  await waitForJobCount(jobs.length);
  return bridge;
}

function emitTextToVideoHere(bridge: ReturnType<typeof createMockBridge>) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "textToVideoHere", selection: cursorSelection() });
  });
}

/** Bridge calls to a given method captured on a `request` spy. */
function callsTo(spy: { mock: { calls: unknown[][] } }, method: BridgeMethod): unknown[][] {
  return spy.mock.calls.filter((c) => c[0] === method);
}

const RESERVATION_BUSY = /a new reservation cannot be made until the current generation finishes/i;

describe("App / X2 reservation release (予約詰まり修正)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "a settled job's leftover tag does not block a new reservation, and is NOT deleted",
    async () => {
      // The exact post-422 timeline state: a leftover `NzLTX23#job-done` tag whose
      // job the ledger reports completed.
      const bridge = await renderShell(
        [makeJob("job-done", "completed")],
        [{ jobId: "job-done", layer: 3, frame: 100 }],
      );
      await waitForJobCount(1);
      const spy = vi.spyOn(bridge, "request");

      emitTextToVideoHere(bridge);

      // The reservation proceeds: a fresh provisional is inserted (fire-and-forget,
      // so wait for it). Before X2(b) the busy-guard returned before this ever ran.
      await vi.waitFor(() => expect(callsTo(spy, "timeline.insertProvisional").length).toBe(1), { timeout: 5_000 });

      // No busy note — the settled tag never held the seat `generating`.
      expect(screen.queryByText(RESERVATION_BUSY)).toBeNull();
      // The `NzLTX23#job-done` marker is preserved (🎞 insert still needs it) —
      // reconcile never deletes, and no rollback targeted it.
      expect(callsTo(spy, "timeline.deleteProvisionalByJob")).toHaveLength(0);
    },
    15_000,
  );

  it(
    "a genuinely running job's tag STILL blocks a new reservation (busy note, no reservation)",
    async () => {
      // A job-bound tag whose job is running is NOT terminal, so it must still
      // hold the seat generating and refuse the reservation.
      const bridge = await renderShell(
        [makeJob("job-run", "running")],
        [{ jobId: "job-run", layer: 3, frame: 100 }],
      );
      await waitForJobCount(1);
      const spy = vi.spyOn(bridge, "request");

      emitTextToVideoHere(bridge);

      await screen.findByText(RESERVATION_BUSY, undefined, { timeout: 5_000 });
      // Refused: no fresh reservation was placed.
      expect(callsTo(spy, "timeline.insertProvisional")).toHaveLength(0);
    },
    15_000,
  );

  it(
    "a second reservation right-click MOVES the single seat (no pending-ghost proliferation)",
    async () => {
      // Clean timeline: the first ✨ inserts a reservation, the second MOVES it
      // (updateProvisionalReservation) rather than inserting a second placeholder.
      const bridge = await renderShell([], []);
      const spy = vi.spyOn(bridge, "request");

      emitTextToVideoHere(bridge);
      await vi.waitFor(() => expect(callsTo(spy, "timeline.insertProvisional").length).toBe(1), { timeout: 5_000 });

      emitTextToVideoHere(bridge);
      await vi.waitFor(
        () => expect(callsTo(spy, "timeline.updateProvisionalReservation").length).toBe(1),
        { timeout: 5_000 },
      );

      // Exactly ONE insert total — the second click moved the seat, it did not
      // spawn a second pending placeholder.
      expect(callsTo(spy, "timeline.insertProvisional")).toHaveLength(1);
      expect(screen.queryByText(RESERVATION_BUSY)).toBeNull();
    },
    15_000,
  );
});

// ── X2(a) 送信失敗時のロールバック配線 ─────────────────────────────────────
// A 422 (VALIDATION_ERROR) submit never reaches `onSubmitted`, so the reserved
// seat + its `NzLTX23#pending-…` placeholder must be rolled back via the screen's
// `onFailed` → `rollbackReservedPlacement` → `timeline.deleteProvisionalByJob`
// wiring. The single seat is app-global, so a ✨ reservation established on Create
// is what either screen's failed Generate rolls back.

/** The pending id burned into the reservation's `insertProvisional` call. */
function reservedPendingId(spy: ReturnType<typeof vi.spyOn>): string {
  const call = callsTo(spy, "timeline.insertProvisional")[0];
  return (call?.[1] as { jobId: string }).jobId;
}

describe("App / X2(a) submit-failure rollback wiring", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    // The submit path uses the module-level default `apiClient` (SingleScreen/
    // ChainedScreen call `useGenerationSubmit` without injecting one), so make it
    // reject with a 422 VALIDATION_ERROR — the exact real-hardware failure.
    const reject422 = () => Promise.reject(new BackendApiError("VALIDATION_ERROR", "bad request", 422));
    vi.spyOn(defaultApiClient, "generate").mockImplementation(reject422);
    vi.spyOn(defaultApiClient, "generateChain").mockImplementation(reject422);
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it(
    "SingleScreen: a 422 Generate rolls back the reservation (deleteProvisionalByJob fires)",
    async () => {
      const user = userEvent.setup();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} apiClient={fakeApiClient([])} />);
      // Empty ledger renders immediately; wait for the prompt bar.
      const promptBox = await screen.findByPlaceholderText(/describe the video/i, {}, { timeout: 5_000 });
      const spy = vi.spyOn(bridge, "request");

      // Reserve a seat at the cursor (✨), then confirm it landed (reserved).
      emitTextToVideoHere(bridge);
      await vi.waitFor(() => expect(callsTo(spy, "timeline.insertProvisional").length).toBe(1), { timeout: 5_000 });
      const pendingId = reservedPendingId(spy);

      // A prompt makes the Create Generate valid; the submit then 422s.
      await user.type(promptBox, "a cat riding a skateboard");
      const generateButton = await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await user.click(generateButton);

      // onFailed → rollbackReservedPlacement → delete THIS reservation's placeholder.
      await vi.waitFor(
        () => {
          const deletes = callsTo(spy, "timeline.deleteProvisionalByJob");
          expect(deletes.some((c) => (c[1] as { jobId: string }).jobId === pendingId)).toBe(true);
        },
        { timeout: 5_000 },
      );
    },
    20_000,
  );

  it(
    "ChainedScreen: a 422 Generate rolls back the reservation (deleteProvisionalByJob fires)",
    async () => {
      const user = userEvent.setup();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} apiClient={fakeApiClient([])} />);
      const promptBox = await screen.findByPlaceholderText(/describe the video/i, {}, { timeout: 5_000 });
      await user.type(promptBox, "a cat riding a skateboard, then a dog joins in");
      const spy = vi.spyOn(bridge, "request");

      // Reserve the single app seat via ✨, then switch to Chain.
      emitTextToVideoHere(bridge);
      await vi.waitFor(() => expect(callsTo(spy, "timeline.insertProvisional").length).toBe(1), { timeout: 5_000 });
      const pendingId = reservedPendingId(spy);

      const chainedTab = await screen.findByRole("tab", { name: /^chained$/i }, { timeout: 5_000 });
      await user.click(chainedTab);

      // Chain scratch mode Generate is enabled with the default 2 clips + prompt;
      // the submit 422s and Chain's onFailed rolls the seat back.
      const generateButton = await within(panel()).findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await user.click(generateButton);

      await vi.waitFor(
        () => {
          const deletes = callsTo(spy, "timeline.deleteProvisionalByJob");
          expect(deletes.some((c) => (c[1] as { jobId: string }).jobId === pendingId)).toBe(true);
        },
        { timeout: 5_000 },
      );
    },
    20_000,
  );
});
