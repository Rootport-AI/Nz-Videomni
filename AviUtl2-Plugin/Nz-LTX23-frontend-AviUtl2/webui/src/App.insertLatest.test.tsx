import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import { createMockBridge } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { ResultOf } from "./bridge";
import type { ApiClient } from "./api/client";
import type { JobResponse, JobStatus } from "./api/types";
import { AppShell, latestCompletedJob } from "./shell/AppShell";
import { resetProvisionalReservation } from "./timeline/provisionalReservation";

// W3 (⬇ insertLatestResultHere): a layer-menu quick-insert that drops the most
// recently completed generation result at the right-click position as a PLAIN
// insert. `downloadAndInsertVideo` is mocked so the tests assert routing +
// `plainInsertAt` wiring without a real download/insert.
vi.mock("./jobs/downloadAndInsert", () => ({
  downloadAndInsertVideo: vi.fn(),
}));
import { downloadAndInsertVideo } from "./jobs/downloadAndInsert";
const mockDownloadAndInsert = vi.mocked(downloadAndInsertVideo);

function makeJob(job_id: string, status: JobStatus, completed_at: string | null = null): JobResponse {
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
    completed_at,
    error: null,
    is_v2v: false,
    joined: false,
    request: {},
    result: null,
  };
}

describe("latestCompletedJob (pure)", () => {
  it("returns null for an empty ledger", () => {
    expect(latestCompletedJob([])).toBeNull();
  });

  it("returns null when no job is completed", () => {
    expect(latestCompletedJob([makeJob("a", "running"), makeJob("b", "queued"), makeJob("c", "failed")])).toBeNull();
  });

  it("ignores non-completed jobs and picks the latest completed by completed_at", () => {
    const jobs = [
      makeJob("old", "completed", "2026-07-22T00:01:00Z"),
      makeJob("running", "running"),
      makeJob("new", "completed", "2026-07-22T00:05:00Z"),
      makeJob("mid", "completed", "2026-07-22T00:03:00Z"),
    ];
    expect(latestCompletedJob(jobs)?.job_id).toBe("new");
  });

  it("treats a null completed_at as the OLDEST (tail of the descending order)", () => {
    const jobs = [
      makeJob("nullish", "completed", null),
      makeJob("timed", "completed", "2026-07-22T00:01:00Z"),
    ];
    // The timestamped job wins over the null one.
    expect(latestCompletedJob(jobs)?.job_id).toBe("timed");
  });

  it("is deterministic across equal timestamps (and two nulls), breaking ties on job_id", () => {
    const a = [makeJob("j-a", "completed", null), makeJob("j-b", "completed", null)];
    const b = [makeJob("j-b", "completed", null), makeJob("j-a", "completed", null)];
    // Same set, different input order -> same winner.
    expect(latestCompletedJob(a)?.job_id).toBe(latestCompletedJob(b)?.job_id);
  });
});

// --- Integration through AppShell -----------------------------------------

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

/** A layer-menu selection (no selected object; ⬇-here generates at a cursor). */
function layerSelection(cursorLayer: number, cursorFrame: number): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [],
    cursorFrame,
    cursorLayer,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

const panel = () => screen.getByRole("tabpanel");

/** Wait until the (visible-panel) ledger heading reflects the seeded count — a
 * stable ready signal (unlike the Generate button, which flips to "busy" when a
 * running job is seeded) that also guarantees the mirrored jobsRef is populated. */
async function waitForJobCount(count: number) {
  await within(panel()).findByText(new RegExp(`jobs \\(${count}\\)`, "i"), undefined, { timeout: 5_000 });
}

async function renderShell(jobs: JobResponse[]) {
  const bridge = createMockBridge({ delayMs: 0 });
  render(<AppShell nativeBridge={bridge} apiClient={fakeApiClient(jobs)} />);
  await waitForJobCount(jobs.length);
  return bridge;
}

function emit(bridge: ReturnType<typeof createMockBridge>, selection: ResultOf<"timeline.getSelection">) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "insertLatestResultHere", selection });
  });
}

describe("App / W3 ⬇ insert the latest generation result here (insertLatestResultHere)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    mockDownloadAndInsert.mockReset();
    mockDownloadAndInsert.mockResolvedValue({ layer: 0, frame: 0, filePath: "C:/out.mp4" });
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "inserts the latest completed result at the cursor via plainInsertAt and shows the success toast",
    async () => {
      const bridge = await renderShell([
        makeJob("job-old", "completed", "2026-07-22T00:01:00Z"),
        makeJob("job-new", "completed", "2026-07-22T00:05:00Z"),
      ]);
      await waitForJobCount(2);

      emit(bridge, layerSelection(4, 360));

      await screen.findByText(/inserted the latest generation result/i, undefined, { timeout: 5_000 });
      // The LATEST completed job, dropped at the right-click cursor via a PLAIN
      // positioned insert (plainInsertAt), never the replace-RPC.
      expect(mockDownloadAndInsert).toHaveBeenCalledTimes(1);
      expect(mockDownloadAndInsert).toHaveBeenCalledWith(bridge, "job-new", undefined, {
        plainInsertAt: { layer: 4, frame: 360 },
      });
    },
    15_000,
  );

  it(
    "Y3 据え置き: does NOT flip any JobCard to ✅ (W3 is an additive insert, not a job fulfillment)",
    async () => {
      const bridge = await renderShell([makeJob("job-done", "completed", "2026-07-22T00:05:00Z")]);
      await waitForJobCount(1);

      emit(bridge, layerSelection(4, 360));

      await screen.findByText(/inserted the latest generation result/i, undefined, { timeout: 5_000 });
      // W3 never publishes to the shared inserted-store (Y3 leaves W3 据え置き),
      // so every card stays at the idle 🎞 Insert — never ✅ Inserted.
      expect(screen.getAllByRole("button", { name: /^insert$/i }).length).toBeGreaterThan(0);
      expect(screen.queryByRole("button", { name: /^inserted$/i })).toBeNull();
    },
    15_000,
  );

  it(
    "shows the 'no completed result' toast and inserts nothing when nothing is completed",
    async () => {
      const bridge = await renderShell([makeJob("job-run", "running")]);
      await waitForJobCount(1);

      emit(bridge, layerSelection(4, 360));

      await screen.findByText(/no completed generation result/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "aborts with the 'could not determine the insert position' toast on an invalid (-1) cursor",
    async () => {
      const bridge = await renderShell([makeJob("job-done", "completed", "2026-07-22T00:05:00Z")]);
      await waitForJobCount(1);

      // A completed job exists (passes the no-result check), but native reported
      // an unresolved cursor — abort rather than mis-place the clip.
      emit(bridge, layerSelection(-1, -1));

      await screen.findByText(/could not determine the insert position/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "shows the failure toast when the insert throws",
    async () => {
      mockDownloadAndInsert.mockRejectedValueOnce(new Error("INSERT_FAILED"));
      const bridge = await renderShell([makeJob("job-done", "completed", "2026-07-22T00:05:00Z")]);
      await waitForJobCount(1);

      emit(bridge, layerSelection(4, 360));

      await screen.findByText(/could not insert the latest generation result/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).toHaveBeenCalledWith(bridge, "job-done", undefined, {
        plainInsertAt: { layer: 4, frame: 360 },
      });
    },
    15_000,
  );
});
